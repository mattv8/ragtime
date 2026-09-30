"""SSH operations exposed only through the workspace-development facade."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException

from ragtime.content_protection import service as content_protection_service
from ragtime.content_protection.models import ContentProtectionError
from ragtime.core.tool_access import resolve_tool_access
from ragtime.indexer.repository import repository
from ragtime.rag.components import rag
from ragtime.tools.ssh_transfer import build_ssh_transfer_tool, build_workspace_file_callbacks, normalized_ssh_name
from ragtime.userspace.development_bootstrap import MAX_MCP_RESPONSE_BYTES, serialize_mcp_payload
from ragtime.userspace.models import UpsertWorkspaceFileRequest
from ragtime.userspace.planning_service import planning_service
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service
from ragtime.userspace.workspace_tool_options import load_workspace_tool_options, resolve_workspace_tool_write_access


def _tool_type(config: Any) -> str:
    return str(getattr(getattr(config, "tool_type", None), "value", getattr(config, "tool_type", "")))


async def authorized_ssh_configs(principal: Any, workspace: Any) -> tuple[list[Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve current selected SSH configs, caller access and owner access."""
    selected = await planning_service._selected_tool_ids(workspace)
    caller = await resolve_tool_access(
        user_id=principal.user_id,
        is_admin=principal.is_admin,
        surface="workspace",
        tool_config_ids=selected,
    )
    owner = await userspace_service._resolve_workspace_owner_tool_access(workspace, selected)
    configs = []
    for tool_id in selected:
        config = await repository.get_tool_config(tool_id)
        if (
            config
            and bool(getattr(config, "enabled", False))
            and _tool_type(config) == "ssh_shell"
            and caller.get(tool_id) in {"read", "read_write"}
            and owner.get(tool_id) in {"read", "read_write"}
        ):
            configs.append(config)
    return configs, caller, owner


def effective_write(workspace: Any, config: Any, caller: Mapping[str, Any], owner: Mapping[str, Any]) -> bool:
    tool_id = str(getattr(config, "id", ""))
    raw = (getattr(workspace, "tool_options", {}) or {}).get(tool_id)
    return bool(
        resolve_workspace_tool_write_access(bool(getattr(config, "allow_write", False)), load_workspace_tool_options(raw))
        and caller.get(tool_id) == "read_write"
        and owner.get(tool_id) == "read_write"
    )


def runtime_config(config: Any, *, allow_write: bool) -> dict[str, Any]:
    # Repository ToolConfig models already contain decrypted connection fields.
    return userspace_service._tool_config_runtime_dict(config, allow_write=allow_write)


async def resource_metadata(
    principal: Any,
    workspace: Any,
    *,
    configs: list[Any] | None = None,
    caller_access: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if configs is None or caller_access is None:
        configs, caller, owner = await authorized_ssh_configs(principal, workspace)
    else:
        caller = caller_access
        owner = await userspace_service._resolve_workspace_owner_tool_access(workspace, [str(item.id) for item in configs])

    visible = [config for config in configs if caller.get(str(config.id)) in {"read", "read_write"} and owner.get(str(config.id)) in {"read", "read_write"}]
    aliases = [normalized_ssh_name(str(getattr(config, "name", "") or "")) for config in visible]
    alias_counts = Counter(alias for alias in aliases if alias)
    result: list[dict[str, Any]] = []
    for config, alias in zip(visible, aliases, strict=True):
        writable = effective_write(workspace, config, caller, owner)
        tool_id = str(config.id)
        transferable = bool(alias and alias_counts[alias] == 1 and "exec" in principal.scopes)
        result.append(
            {
                "component_id": tool_id,
                "name": str(getattr(config, "name", "") or ""),
                "tool_type": "ssh_shell",
                "access": caller[tool_id],
                "ssh_endpoint_alias": alias if transferable else None,
                "ssh_operations": {
                    "ssh_execute": {
                        "supported": writable and "exec" in principal.scopes and "write" in principal.scopes,
                        "requires_scopes": ["exec", "write"],
                    },
                    "ssh_transfer": {
                        "supported": transferable,
                        "requires_scopes": ["exec"],
                        "can_read": transferable,
                        "can_write": transferable and writable and "write" in principal.scopes,
                    },
                },
            }
        )
    return result


def _validate_timeout(value: Any) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 300):
        raise HTTPException(status_code=422, detail="arguments.timeout must be between 1 and 300")


async def _authorize_tool_content(candidate: Any, *, direction: str, tool_id: str, operation: str, execution_completed: bool = False) -> None:
    try:
        await content_protection_service.authorize_content(
            candidate,
            direction=direction,
            context=content_protection_service.current_context(),
            tool_id=tool_id,
            operation=operation,
        )
    except ContentProtectionError as exc:
        if execution_completed and direction == "tool_result" and exc.code == "content_denied":
            raise ContentProtectionError(
                exc.code,
                exc.request_id,
                reason=exc.reason,
                reason_code=exc.reason_code,
                recovery_eligible=True,
                execution_status="completed_response_withheld",
            ) from exc
        raise


async def _audit_ssh_operation(
    principal: Any,
    workspace_id: str,
    operation: str,
    *,
    component_id: str | None = None,
    source_component_id: str | None = None,
    destination_component_id: str | None = None,
) -> None:
    payload: dict[str, Any] = {"operation": operation, "credential_id": principal.credential_id}
    if component_id is not None:
        payload["component_id"] = component_id
    if source_component_id is not None:
        payload["source_component_id"] = source_component_id
    if destination_component_id is not None:
        payload["destination_component_id"] = destination_component_id
    await userspace_runtime_service._audit(
        workspace_id,
        "development_ssh_operation",
        user_id=principal.user_id,
        session_id=None,
        payload=payload,
    )


def _serialized_size(value: Any) -> int:
    return len(serialize_mcp_payload(value, limit=None).encode("utf-8"))


def _fit_command_result(result: dict[str, Any]) -> dict[str, Any]:
    if _serialized_size(result) <= MAX_MCP_RESPONSE_BYTES:
        return result

    fitted = dict(result)
    streams = {key: value for key in ("stdout", "stderr") if isinstance((value := fitted.get(key)), str) and value}
    for key in streams:
        fitted[key] = ""
        fitted[f"{key}_truncated"] = True
    fitted["truncated"] = True
    fitted["execution_status"] = "completed"

    if _serialized_size(fitted) > MAX_MCP_RESPONSE_BYTES:
        fitted = {
            key: value
            for key, value in fitted.items()
            if key in {"tool", "status", "exit_code", "truncated", "execution_status", "stdout_truncated", "stderr_truncated"}
        }

    for key, original in streams.items():
        low, high = 0, len(original)
        while low < high:
            middle = (low + high + 1) // 2
            fitted[key] = original[:middle]
            if _serialized_size(fitted) <= MAX_MCP_RESPONSE_BYTES:
                low = middle
            else:
                high = middle - 1
        fitted[key] = original[:low]
        if low == len(original):
            fitted.pop(f"{key}_truncated", None)
    if _serialized_size(fitted) > MAX_MCP_RESPONSE_BYTES:
        status = result.get("status")
        fitted = {
            "status": "response_too_large",
            "command_status": status if isinstance(status, str) and len(status.encode("utf-8")) <= 128 else "unknown",
            "exit_code": result.get("exit_code") if isinstance(result.get("exit_code"), int) else None,
            "execution_status": "completed_response_withheld",
            "truncated": True,
        }
    return fitted


def _sanitize_command_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in {"ok", "completed"} and "error" in result:
        result = dict(result)
        result["error"] = "SSH command failed"
    return result


async def execute(principal: Any, workspace_id: str, workspace: Any, args: dict[str, Any]) -> dict[str, Any]:
    if "write" not in principal.scopes:
        raise HTTPException(status_code=403, detail="ssh_execute also requires write scope")
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise HTTPException(status_code=422, detail="arguments.command must be a nonempty string")
    _validate_timeout(args.get("timeout"))

    configs, caller, owner = await authorized_ssh_configs(principal, workspace)
    config = next((item for item in configs if str(item.id) == args.get("component_id")), None)
    if config is None or not effective_write(workspace, config, caller, owner):
        raise HTTPException(status_code=403, detail="SSH command execution is not authorized for this connection")
    component_id = str(config.id)
    await _audit_ssh_operation(principal, workspace_id, "ssh_execute", component_id=component_id)
    await _authorize_tool_content(args, direction="proposed_operation", tool_id=component_id, operation="ssh_execute")

    try:
        tool = await rag.build_primary_runtime_tool_from_config(runtime_config(config, allow_write=True))
    except ContentProtectionError:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail="SSH connection is unavailable") from exc
    if tool is None:
        raise HTTPException(status_code=409, detail="SSH connection is unavailable")

    invocation = {"command": command, "reason": args.get("reason", "SSH command")}
    if "timeout" in args:
        invocation["timeout"] = args["timeout"]
    try:
        value = await tool.ainvoke(invocation)
    except ContentProtectionError:
        raise
    except Exception:
        value = {"status": "command_failed", "exit_code": -1, "error": "SSH command failed"}
    try:
        result = json.loads(value) if isinstance(value, str) else dict(value)
        if not isinstance(result, dict):
            raise TypeError
    except (TypeError, ValueError):
        result = {"status": "command_failed", "exit_code": -1, "error": "SSH command returned an invalid result"}
    result = _fit_command_result(_sanitize_command_result(result))
    await _authorize_tool_content(result, direction="tool_result", tool_id=component_id, operation="ssh_execute", execution_completed=True)
    return result


def _validate_transfer_arguments(args: dict[str, Any]) -> tuple[str, str, str | None]:
    _validate_timeout(args.get("timeout"))
    source = str(args.get("source") or "")
    destination = str(args.get("destination") or "")
    source_inline = source == "inline"
    workspace_source = source.startswith("workspace:/")
    workspace_destination = destination.startswith("workspace:/")
    if source_inline and not isinstance(args.get("content"), str):
        raise HTTPException(status_code=422, detail="arguments.content is required when source is inline")
    if not source_inline and "content" in args:
        raise HTTPException(status_code=422, detail="arguments.content is only supported when source is inline")
    if (workspace_source or workspace_destination) and bool(args.get("recursive", False)):
        raise HTTPException(status_code=422, detail="recursive is not supported for workspace endpoints")
    if not source.startswith("ssh://") and not destination.startswith("ssh://"):
        raise HTTPException(status_code=422, detail="at least one transfer endpoint must be SSH")
    if workspace_destination and "expected_content_hash" not in args:
        raise HTTPException(status_code=422, detail="expected_content_hash is required for workspace destinations")
    if workspace_destination and "overwrite" in args:
        raise HTTPException(status_code=422, detail="overwrite is not supported for workspace destinations")
    if not workspace_destination and "expected_content_hash" in args:
        raise HTTPException(status_code=422, detail="expected_content_hash is only supported for workspace destinations")
    expected = args.get("expected_content_hash")
    if expected is not None:
        if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise HTTPException(status_code=422, detail="expected_content_hash must be a SHA-256 digest or null")
        expected = expected.lower()
    return source, destination, expected


def _endpoint_component_id(endpoint: str, configs: list[Any]) -> str | None:
    if not endpoint.startswith("ssh://"):
        return None
    alias = urlparse(endpoint).hostname or ""
    matches = [str(config.id) for config in configs if normalized_ssh_name(str(getattr(config, "name", "") or "")) == alias]
    return matches[0] if len(matches) == 1 else None


def _plain_text_dashboard_module(path: str) -> bool:
    normalized = path.strip().lower().replace("\\", "/")
    return normalized.startswith("dashboard/") and normalized.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))


async def transfer(principal: Any, workspace_id: str, workspace: Any, args: dict[str, Any]) -> dict[str, Any]:
    source, destination, expected = _validate_transfer_arguments(args)
    workspace_destination = destination.startswith("workspace:/")
    if (destination.startswith("ssh://") or workspace_destination) and "write" not in principal.scopes:
        raise HTTPException(status_code=403, detail="SSH transfer destination requires write scope")

    args = dict(args)
    args.pop("expected_content_hash", None)
    configs, caller, owner = await authorized_ssh_configs(principal, workspace)
    runtime = [runtime_config(item, allow_write=effective_write(workspace, item, caller, owner) and "write" in principal.scopes) for item in configs]
    if not runtime:
        raise HTTPException(status_code=403, detail="No SSH connection is authorized")

    source_component_id = _endpoint_component_id(source, configs)
    destination_component_id = _endpoint_component_id(destination, configs)
    await _audit_ssh_operation(
        principal,
        workspace_id,
        "ssh_transfer",
        source_component_id=source_component_id,
        destination_component_id=destination_component_id,
    )

    read, _write = build_workspace_file_callbacks(
        user_id=principal.user_id,
        is_admin=principal.is_admin,
        allowed_workspace_id=workspace_id,
    )
    callback_error: HTTPException | None = None

    async def write(active_workspace_id: str, path: str, content: str, _overwrite: bool) -> None:
        nonlocal callback_error
        # The service wires dashboard modules before its durable CAS check. A
        # targeted preflight prevents that incidental write for already-stale
        # plain-text transfer requests; authoritative CAS remains enabled below.
        if _plain_text_dashboard_module(path):
            current_hash: str | None = None
            try:
                current = await userspace_service.get_workspace_file(
                    active_workspace_id,
                    path,
                    principal.user_id,
                    is_admin=principal.is_admin,
                )
                current_hash = hashlib.sha256(current.content.encode("utf-8")).hexdigest()
            except HTTPException as exc:
                if exc.status_code != 404:
                    callback_error = exc
                    raise ValueError("workspace destination is unavailable") from None
            if current_hash != expected:
                callback_error = HTTPException(
                    status_code=409,
                    detail={"code": "content_hash_conflict", "expected_hash": expected, "actual_hash": current_hash},
                )
                raise ValueError("workspace file changed; retry the transfer")

        try:
            await userspace_service.upsert_workspace_file(
                active_workspace_id,
                path,
                UpsertWorkspaceFileRequest(content=content, artifact_type=None, live_data_requested=False),
                principal.user_id,
                expected_content_hash=expected,
                require_content_hash=True,
            )
        except HTTPException as exc:
            if exc.status_code in {403, 409, 415}:
                callback_error = exc
                raise ValueError("workspace destination write failed") from None
            raise

    value = await build_ssh_transfer_tool(runtime, workspace_read=read, workspace_write=write, workspace_id=workspace_id)(**args)
    if callback_error is not None:
        raise callback_error
    try:
        result = json.loads(value)
        if not isinstance(result, dict):
            raise TypeError
    except (TypeError, ValueError):
        result = {"status": "transfer_failed", "errors": ["SSH transfer returned an invalid result"]}

    completed = result.get("status") == "ok"
    if _serialized_size(result) > MAX_MCP_RESPONSE_BYTES:
        transfer_status = result.get("status")
        if not isinstance(transfer_status, str) or len(transfer_status.encode("utf-8")) > 128:
            transfer_status = "unknown"
        counts = [result.get("files_transferred"), result.get("bytes_transferred")]
        has_completed_files = any(isinstance(count, int) and count > 0 for count in counts)
        if completed:
            execution_status = "completed_response_withheld"
        elif has_completed_files:
            execution_status = "partially_completed_response_withheld"
        elif transfer_status == "rejected" and all(count == 0 for count in counts):
            execution_status = "not_started"
        else:
            # A failed publication may have reached the server even when no
            # acknowledged files were counted. Never imply that retry is safe.
            execution_status = "unknown_response_withheld"
        result = {
            "status": "response_too_large",
            "transfer_status": transfer_status,
            "bytes_transferred": result.get("bytes_transferred") if isinstance(result.get("bytes_transferred"), int) else None,
            "files_transferred": result.get("files_transferred") if isinstance(result.get("files_transferred"), int) else None,
            "execution_status": execution_status,
            "truncated": True,
        }
    return result
