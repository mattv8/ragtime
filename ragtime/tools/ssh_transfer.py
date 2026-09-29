"""Request-scoped orchestration for the SSH file transfer core."""

import asyncio
import base64
import hashlib
import json
import threading
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragtime.content_protection import service as content_protection_service
from ragtime.content_protection.models import ContentProtectionError
from ragtime.core.ssh import SSHConfig
from ragtime.core.tool_timeouts import resolve_effective_command_timeout
from ragtime.indexer.utils import safe_tool_name

MAX_INLINE_BYTES = 1024 * 1024
WorkspaceRead = Callable[[str, str], Awaitable[str]]
WorkspaceWrite = Callable[[str, str, str, bool], Awaitable[None]]
VisibleConfigResolver = Callable[[], Awaitable[list[dict[str, Any]]]]


class SSHTransferInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Source endpoint: inline, workspace:/path, or ssh://configured-name/absolute/path.")
    destination: str = Field(description="Destination endpoint: inline, workspace:/path, or ssh://configured-name/absolute/path.")
    content: str | None = Field(default=None, description="Inline source content; required only when source is inline.")
    encoding: str = Field(default="text", pattern="^(text|base64)$", description="Encoding for inline input or output: text or base64.")
    overwrite: bool = Field(default=False, description="Whether an existing destination file may be replaced.")
    recursive: bool = Field(default=False, description="Whether to recursively copy an SSH directory; workspace endpoints never support recursion.")
    reason: str = Field(default="SSH file transfer", description="Brief reason for the file transfer.")
    timeout: int = Field(default=300, ge=1, le=300, description="Transfer timeout in seconds, from 1 through 300.")
    workspace_id: str | None = Field(default=None, description="Authorized workspace identifier when a workspace endpoint is used.")


def normalized_ssh_name(value: str) -> str:
    """Use the same stable normalization as configured SSH tool names."""
    return safe_tool_name(value)


def parse_endpoint(value: str, configs: dict[str, dict[str, Any]]) -> tuple[str, str | None, dict[str, Any] | None]:
    """Parse an endpoint without resolving aliases outside the visible map."""
    if value == "inline":
        return "inline", None, None
    if value.startswith("workspace:/"):
        return "workspace", value[len("workspace:/") :], None

    parsed = urlparse(value)
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise ValueError("SSH endpoint syntax is invalid") from exc
    hostname = parsed.hostname or ""
    if (
        parsed.scheme != "ssh"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed_port is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or bool(parsed.params)
        or parsed.netloc != hostname
        or unquote(hostname) != hostname
        or normalized_ssh_name(hostname) != hostname
    ):
        raise ValueError("endpoint must be inline, workspace:/path, or ssh://configured-name/absolute/path")
    config = configs.get(hostname)
    if config is None:
        raise ValueError("SSH endpoint is not available")
    return "ssh", unquote(parsed.path), config


def _result(status: str, error: str | None = None, **extra: Any) -> str:
    payload: dict[str, Any] = {
        "tool": "ssh_transfer",
        "status": status,
        "bytes_transferred": 0,
        "files_transferred": 0,
        "errors": [],
        "skipped": [],
    }
    if error:
        payload["errors"] = [error]
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def ssh_transfer_validation_error(_error: Any) -> str:
    """Return a payload-safe validation failure for LangChain wrappers."""
    return _result("rejected", "invalid ssh_transfer arguments")


def _ssh_config(config: dict[str, Any], timeout: int) -> SSHConfig:
    connection = config.get("connection_config") or {}
    return SSHConfig(
        host=connection.get("host", ""),
        port=connection.get("port", 22),
        user=connection.get("user", ""),
        password=connection.get("password"),
        key_path=connection.get("key_path"),
        key_content=connection.get("key_content"),
        key_passphrase=connection.get("key_passphrase"),
        timeout=timeout,
    )


def _visible_config_map(
    configs: list[dict[str, Any]],
    *,
    visible_config_ids: frozenset[str] | None = None,
    visible_endpoint_names: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for config in configs:
        config_id = str(config.get("id") or "")
        if visible_config_ids is not None and config_id not in visible_config_ids:
            continue
        if config.get("tool_type") != "ssh_shell" or not config.get("enabled", True):
            continue
        name = normalized_ssh_name(str(config.get("name") or ""))
        if not name or (visible_endpoint_names is not None and name not in visible_endpoint_names):
            continue
        if name in by_name:
            ambiguous.add(name)
        by_name[name] = config
    for name in ambiguous:
        by_name.pop(name, None)
    return by_name


def build_workspace_file_callbacks(
    *,
    user_id: str,
    is_admin: bool = False,
    allowed_workspace_id: str | None = None,
) -> tuple[WorkspaceRead, WorkspaceWrite]:
    """Build shared ACL-checked, conditional workspace text-file callbacks."""

    def require_workspace(active_workspace_id: str) -> None:
        if not active_workspace_id or (allowed_workspace_id is not None and active_workspace_id != allowed_workspace_id):
            raise ValueError("workspace endpoint is not authorized")

    async def workspace_read(active_workspace_id: str, path: str) -> str:
        require_workspace(active_workspace_id)
        from fastapi import HTTPException

        from ragtime.userspace.service import userspace_service

        try:
            return (await userspace_service.get_workspace_file(active_workspace_id, path, user_id, is_admin=is_admin)).content
        except HTTPException as exc:
            if exc.status_code == 415:
                raise ValueError("workspace source accepts UTF-8 text only") from None
            if exc.status_code == 403:
                raise ValueError("workspace endpoint is not authorized") from None
            if exc.status_code == 409:
                raise ValueError("workspace file changed; retry the transfer") from None
            raise

    async def workspace_write(active_workspace_id: str, path: str, content: str, overwrite: bool) -> None:
        require_workspace(active_workspace_id)
        from fastapi import HTTPException

        from ragtime.userspace.models import UpsertWorkspaceFileRequest
        from ragtime.userspace.service import userspace_service

        existing = None
        try:
            existing = await userspace_service.get_workspace_file(active_workspace_id, path, user_id, is_admin=is_admin)
        except HTTPException as exc:
            if exc.status_code == 403:
                raise ValueError("workspace endpoint is not authorized") from None
            if exc.status_code == 415:
                raise ValueError("workspace destination accepts UTF-8 text only") from None
            if exc.status_code == 409:
                raise ValueError("workspace file changed; retry the transfer") from None
            if exc.status_code != 404:
                raise
        if existing is not None and not overwrite:
            raise ValueError("workspace destination already exists")
        expected_hash = hashlib.sha256(existing.content.encode("utf-8")).hexdigest() if existing is not None else None
        try:
            await userspace_service.upsert_workspace_file(
                active_workspace_id,
                path,
                UpsertWorkspaceFileRequest(content=content),
                user_id,
                expected_content_hash=expected_hash,
                require_content_hash=True,
            )
        except HTTPException as exc:
            errors = {
                403: "workspace endpoint is not authorized",
                409: "workspace file changed; retry the transfer",
                415: "workspace destination accepts UTF-8 text only",
            }
            if exc.status_code in errors:
                raise ValueError(errors[exc.status_code]) from None
            raise

    return workspace_read, workspace_write


def build_ssh_transfer_tool(
    visible_configs: list[dict[str, Any]],
    *,
    workspace_read: WorkspaceRead | None = None,
    workspace_write: WorkspaceWrite | None = None,
    workspace_id: str | None = None,
    max_output_chars: int | None = None,
    visible_config_resolver: VisibleConfigResolver | None = None,
    subagent_file_scope: list[str] | None = None,
) -> Callable[..., Awaitable[str]]:
    """Return a transfer callable limited to a construction-time visible allowlist."""
    visible_config_ids = frozenset(str(config.get("id") or "") for config in visible_configs if config.get("id"))
    visible_endpoint_names = frozenset(normalized_ssh_name(str(config.get("name") or "")) for config in visible_configs if config.get("name"))
    initial_configs = _visible_config_map(
        visible_configs,
        visible_config_ids=visible_config_ids,
        visible_endpoint_names=visible_endpoint_names,
    )
    normalized_scope: tuple[str, ...] = tuple(_normalize_workspace_path(path) for path in (subagent_file_scope or []))

    def assert_workspace_scope(path: str) -> str:
        normalized = _normalize_workspace_path(path)
        if normalized_scope and not any(normalized == scope or normalized.startswith(f"{scope}/") for scope in normalized_scope):
            raise ValueError("workspace path is outside this subagent's declared file scope")
        return normalized

    async def authorize_endpoint_content(candidate: Any, direction: str, *configs_to_check: dict[str, Any] | None) -> None:
        context = content_protection_service.current_context()
        for config in {str(config.get("id") or ""): config for config in configs_to_check if config and config.get("id")}.values():
            await content_protection_service.authorize_content(
                candidate,
                direction=direction,
                context=context,
                tool_id=str(config["id"]),
                operation="ssh_transfer",
            )

    def policy_candidate(value: dict[str, Any]) -> dict[str, Any]:
        """Make valid UTF-8 transfer bytes inspectable without changing the transfer."""
        candidate = dict(value)
        content = candidate.get("content")
        if isinstance(content, (bytes, bytearray)):
            try:
                candidate["content"] = bytes(content).decode("utf-8")
            except UnicodeDecodeError:
                # Preserve opaque binary so content protection fails closed.
                pass
        return candidate

    async def transfer(**kwargs: Any) -> str:
        try:
            try:
                args = SSHTransferInput(**kwargs)
            except ValidationError:
                return _result("rejected", "invalid ssh_transfer arguments")

            configs = initial_configs
            if visible_config_resolver is not None:
                configs = _visible_config_map(
                    await visible_config_resolver(),
                    visible_config_ids=visible_config_ids,
                    visible_endpoint_names=visible_endpoint_names,
                )

            source_kind, source_path, source_config = parse_endpoint(args.source, configs)
            destination_kind, destination_path, destination_config = parse_endpoint(args.destination, configs)
            if source_kind != "ssh" and destination_kind != "ssh":
                return _result("rejected", "at least one endpoint must be SSH")

            uses_workspace = source_kind == "workspace" or destination_kind == "workspace"
            active_workspace_id = args.workspace_id or workspace_id
            if uses_workspace and (not active_workspace_id or (workspace_id is not None and active_workspace_id != workspace_id)):
                return _result("rejected", "workspace endpoint is not authorized")
            if uses_workspace and args.recursive:
                return _result("rejected", "workspace transfers support one authorized UTF-8 text file only")
            if source_kind == "workspace" and workspace_read is None:
                return _result("rejected", "workspace transfers support one authorized UTF-8 text file only")
            if destination_kind == "workspace" and workspace_write is None:
                return _result("rejected", "workspace transfers support one authorized UTF-8 text file only")
            if destination_kind == "ssh":
                if destination_config is None or not bool(destination_config.get("allow_write", False)):
                    return _result("rejected", "destination SSH endpoint is read-only")

            if source_kind == "workspace":
                source_path = assert_workspace_scope(source_path or "")
            if destination_kind == "workspace":
                destination_path = assert_workspace_scope(destination_path or "")

            content: bytes | None = None
            if source_kind == "inline":
                if args.content is None:
                    return _result("rejected", "content is required when source is inline")
                try:
                    content = base64.b64decode(args.content, validate=True) if args.encoding == "base64" else args.content.encode("utf-8")
                except (ValueError, UnicodeError):
                    return _result("rejected", "invalid inline payload")
                if len(content) > MAX_INLINE_BYTES:
                    return _result("rejected", "inline payload exceeds 1 MiB")
            elif source_kind == "workspace":
                assert workspace_read is not None and active_workspace_id is not None
                text = await workspace_read(active_workspace_id, source_path or "")
                content = text.encode("utf-8")
                if len(content) > MAX_INLINE_BYTES:
                    return _result("rejected", "workspace file exceeds 1 MiB")

            operation_candidate = dict(kwargs)
            if content is not None:
                operation_candidate["content"] = content
            await authorize_endpoint_content(policy_candidate(operation_candidate), "proposed_operation", source_config, destination_config)

            from ragtime.core.ssh_transfer import transfer_ssh_files

            cancel_event = threading.Event()
            endpoint_caps = [int(config.get("timeout_max_seconds", 0) or 0) for config in (source_config, destination_config) if config is not None]
            effective_timeout = min([args.timeout, 300] + [resolve_effective_command_timeout(args.timeout, cap) for cap in endpoint_caps if cap > 0])
            source_ssh = _ssh_config(source_config, effective_timeout) if source_config is not None else None
            destination_ssh = _ssh_config(destination_config, effective_timeout) if destination_config is not None else None
            future = asyncio.create_task(
                asyncio.to_thread(
                    transfer_ssh_files,
                    source_ssh,
                    destination_ssh,
                    source_path if source_kind == "ssh" else None,
                    destination_path if destination_kind == "ssh" else None,
                    source_root=str((source_config or {}).get("connection_config", {}).get("working_directory", "")),
                    destination_root=str((destination_config or {}).get("connection_config", {}).get("working_directory", "")),
                    content=content,
                    overwrite=args.overwrite,
                    recursive=args.recursive,
                    timeout=effective_timeout,
                    max_file_bytes=MAX_INLINE_BYTES if destination_kind in {"inline", "workspace"} else 50 * 1024 * 1024,
                    cancel_event=cancel_event,
                )
            )
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError:
                cancel_event.set()
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                raise

            await authorize_endpoint_content(policy_candidate(result), "tool_result", source_config, destination_config)
            if destination_kind == "inline" and result.get("status") == "ok":
                output = result.pop("content", b"")
                if len(output) > MAX_INLINE_BYTES:
                    return _result("transfer_failed", "inline result exceeds 1 MiB")
                if args.encoding == "base64":
                    result["content"] = base64.b64encode(output).decode("ascii")
                else:
                    try:
                        result["content"] = output.decode("utf-8")
                    except UnicodeDecodeError:
                        return _result("rejected", "inline text output is not valid UTF-8; request base64 encoding")
                result["encoding"] = args.encoding
            elif destination_kind == "workspace" and result.get("status") == "ok":
                output = result.pop("content", b"")
                try:
                    text = output.decode("utf-8")
                except UnicodeDecodeError:
                    return _result("rejected", "workspace destination accepts UTF-8 text only")
                assert workspace_write is not None and active_workspace_id is not None
                await workspace_write(active_workspace_id, destination_path or "", text, args.overwrite)

            serialized = json.dumps(result, ensure_ascii=False)
            if max_output_chars is not None and max_output_chars > 0 and len(serialized) > max_output_chars:
                return _result("transfer_failed", "transfer result exceeds the surface output budget")
            return serialized
        except ValueError as exc:
            return _result("rejected", str(exc))
        except ContentProtectionError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            return _result("transfer_failed", "transfer could not be completed")

    return transfer


def _normalize_workspace_path(path: str) -> str:
    normalized = "/".join(part for part in str(path or "").strip().replace("\\", "/").split("/") if part and part != ".")
    if not normalized or normalized == ".." or normalized.startswith("../") or "/../" in f"/{normalized}/":
        raise ValueError("workspace path is invalid")
    return normalized
