"""SolidWorks operation execution and host lifecycle delegation."""

import copy
import hashlib
import json
import logging
import ntpath
import re
import tempfile
import time
import traceback
import unicodedata
from collections.abc import Callable
from pathlib import Path

from ragtime.config import settings
from ragtime.content_protection.service import authorize_content, current_context
from ragtime.solidworks import pdm_link
from ragtime.solidworks.artifacts import MAX_ARTIFACT_BYTES, ArtifactTooLarge, store_artifact
from ragtime.solidworks.guidance import read_guide
from ragtime.solidworks.host_activity import HostBusyError, HostUpdatingError, host_activity, host_key
from ragtime.solidworks.host_artifact import get_host_artifact
from ragtime.solidworks.host_updater import host_updater, parse_host_version
from ragtime.solidworks.protocol import (
    SolidworksProtocolError,
    build_request,
    encode_request,
    error_response,
)
from ragtime.solidworks.transport import SolidworksTransport, SolidworksTransportError, SshSolidworksTransport

DEFAULT_TIMEOUT_MAX_SECONDS = 100
MIN_OPERATION_SECONDS = 6
MAX_RESULT_BYTES = 32768
MAX_CACHED_TOOLS = 256
HOST_MISMATCH_MESSAGE = "The SolidWorks agent answered a different request."
UNKNOWN_OPERATION_MESSAGE = "The SolidWorks agent does not offer this operation; call describe for the operation list."
logger = logging.getLogger(__name__)
_ARTIFACT_PATH = re.compile(
    r'^[A-Za-z]:\\(?:[^\\/:*?"<>|]+\\)+AppData\\Local\\RagtimeSolidWorks\\work\\[A-Za-z0-9][A-Za-z0-9._-]{0,127}$',
    re.IGNORECASE,
)

_GUIDE_OPERATION = {
    "name": "guide",
    "summary": "Read SolidWorks tool guidance.",
    "access": "read",
    "paged": False,
    "arguments": {"type": "object", "properties": {"topic": {"type": "string"}}},
    "result": "Guidance topics and markdown.",
}

_LIMIT_SCHEMA = {"type": "integer", "minimum": 1, "maximum": 25, "default": 10}
_PDM_OPERATIONS = [
    {
        "name": "pdm.search",
        "summary": "Search the indexed PDM vault.",
        "access": "read",
        "paged": False,
        "arguments": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "document_type": {"type": "string", "pattern": "^[sS][lL][dD]([pP][rR][tT]|[aA][sS][mM]|[dD][rR][wW])$"},
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "result": "Indexed vault search results.",
    },
    {
        "name": "pdm.lookup",
        "summary": "Look up indexed PDM documents.",
        "access": "read",
        "paged": False,
        "arguments": {
            "type": "object",
            "properties": {
                "document_id": {"type": "integer"},
                "filename": {"type": "string", "minLength": 1, "maxLength": 260},
                "part_number": {"type": "string", "minLength": 1, "maxLength": 260},
                "configuration": {"type": "string", "minLength": 1, "maxLength": 260},
                "limit": _LIMIT_SCHEMA,
            },
            "anyOf": [{"required": [key]} for key in ("document_id", "filename", "part_number")],
            "additionalProperties": False,
        },
        "result": "Indexed vault document details.",
    },
]


class _HostMismatchError(Exception):
    """The host envelope does not answer the request that was sent."""


def _artifact_remote_path(value: object) -> str:
    if not isinstance(value, str) or not _ARTIFACT_PATH.fullmatch(value):
        raise ValueError("Invalid artifact path")
    if any(part in (".", "..") or any(unicodedata.category(c) == "Cc" for c in part) for part in value.split("\\")):
        raise ValueError("Invalid artifact path")
    return value.replace("\\", "/")


def _safe_result_text(value: object, limit: int) -> str:
    if not isinstance(value, str) or "/" in value or "\\" in value:
        return ""
    return value[:limit]


def resolve_timeout_cap(config: dict) -> int:
    """Return the per-tool SolidWorks timeout cap; anything but a positive integer means the default."""
    cap = config.get("timeout_max_seconds")
    if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0:
        return cap
    return DEFAULT_TIMEOUT_MAX_SECONDS


def _is_positive_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _invalid_script_field(arguments: object, timeout: object) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments"
    source = arguments.get("source")
    try:
        if not isinstance(source, str) or not 1 <= len(source.encode("utf-8")) <= 16384:
            return "source"
    except UnicodeError:
        return "source"
    try:
        purpose = arguments.get("purpose")
        if not isinstance(purpose, str) or not 1 <= len(purpose) <= 500 or any(unicodedata.category(c) == "Cc" for c in purpose):
            return "purpose"
        script_arguments = arguments.get("arguments", {})
        if (
            not isinstance(script_arguments, dict)
            or len(json.dumps(script_arguments, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")) > 4096
        ):
            return "arguments"
    except (TypeError, ValueError, UnicodeError):
        return "arguments"
    if "instance" in arguments and (not isinstance(arguments["instance"], str) or not re.fullmatch(r"[ia][1-9][0-9]*", arguments["instance"])):
        return "instance"
    if set(arguments) - {"source", "purpose", "arguments", "instance"}:
        return "arguments"
    if timeout is not None and (not isinstance(timeout, int | float) or not _is_positive_number(timeout) or timeout > 600):
        return "timeout"
    return None


def _tool_key(config: dict) -> str | None:
    """Cache key for a tool; tools without an id are never cached."""
    tool_id = config.get("id")
    return tool_id if isinstance(tool_id, str) and tool_id else None


class SolidworksService:
    """Execute operations and delegate host lifecycle operations."""

    def __init__(
        self,
        transport_factory: Callable[[dict], SolidworksTransport] = SshSolidworksTransport,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.transport_factory = transport_factory
        self.clock = clock
        self._contracts: dict[str, tuple[object, dict]] = {}

    def _cached_contract(self, key: str | None) -> dict | None:
        entry = self._contracts.get(key) if key else None
        return copy.deepcopy(entry[1]) if entry else None

    def _store_contract(self, key: str | None, version: object, response: dict) -> None:
        if key is None:
            return
        self._contracts.pop(key, None)
        if len(self._contracts) >= MAX_CACHED_TOOLS:
            self._contracts.pop(next(iter(self._contracts)))
        self._contracts[key] = (version, response)

    def _forget_if_version_changed(self, key: str | None, host_version: object) -> None:
        if key is None or host_version is None:
            return
        entry = self._contracts.get(key)
        if entry is not None and entry[0] != host_version:
            del self._contracts[key]

    async def _call(self, transport: SolidworksTransport, key: str | None, request: dict, timeout_seconds: float, config: dict) -> dict:
        response = await transport.call(request, timeout_seconds=timeout_seconds)
        host = response.get("host") or {}
        version = host.get("host_version") if isinstance(host, dict) else None
        if isinstance(version, str):
            host_updater.observe(config, version)
        if response.get("id") != request["id"] or response.get("operation") != request["operation"]:
            raise _HostMismatchError()
        if response.get("ok"):
            self._forget_if_version_changed(key, (response.get("host") or {}).get("host_version"))
        return response

    async def _describe(self, transport: SolidworksTransport, key: str | None, context: dict, timeout_seconds: float, config: dict) -> dict:
        request = build_request("describe", {}, context)
        try:
            response = await self._call(transport, key, request, timeout_seconds, config)
        except SolidworksTransportError:
            cached = self._cached_contract(key)
            if cached is not None:
                return cached
            raise
        if not response.get("ok"):
            cached = self._cached_contract(key)
            return cached if cached is not None else response
        response = copy.deepcopy(response)
        result = response["result"]
        result["operations"] = [
            entry for entry in result["operations"] if entry.get("name") != "guide" and entry.get("name") not in pdm_link.PDM_ADMIN_OPERATIONS
        ]
        result["operations"].append(copy.deepcopy(_GUIDE_OPERATION))
        version = result.get("host_version") or response.get("host", {}).get("host_version")
        self._store_contract(key, version, copy.deepcopy(response))
        return response

    async def execute(self, config: dict, operation: str, arguments: dict | None = None, timeout: int | None = None) -> str:
        safe_operation = operation if isinstance(operation, str) else ""
        if safe_operation in pdm_link.PDM_ADMIN_OPERATIONS:
            return self._error(safe_operation, "unknown_operation", f"There is no operation named '{safe_operation}'. Call describe for the list.")
        if operation == "script.run":
            if not (config.get("connection_config") or {}).get("allow_scripts", False):
                return self._error(operation, "scripts_not_allowed", "Scripts are not enabled for this tool.")
            field = _invalid_script_field(arguments, timeout)
            if field or arguments is None:
                return self._error(operation, "invalid_arguments", "Invalid script argument.", {"field": field or "arguments"})
            connection = config.get("connection_config") or {}
            try:
                encode_request(build_request(operation, arguments, self._context(connection, bool(config.get("allow_write", False)), 600)))
            except SolidworksProtocolError:
                return self._error(operation, "invalid_arguments", "The request is too large.", {"field": "source"})
            await authorize_content(
                arguments["source"],
                direction="inbound",
                context=current_context(),
                tool_id=config.get("id"),
                operation="solidworks.script.run",
                supporting_context={"purpose": arguments["purpose"], "arguments": arguments.get("arguments", {})},
            )
        try:
            if not isinstance(operation, str) or not operation.strip() or not (arguments is None or isinstance(arguments, dict)):
                return self._error(safe_operation, "invalid_arguments", "Provide an operation and an arguments object.")
            if operation == "guide":
                topic = (arguments or {}).get("topic")
                if topic is not None and not isinstance(topic, str):
                    return self._error(operation, "invalid_arguments", "The guide topic must be a string.")
                try:
                    result = read_guide(topic)
                except ValueError as exc:
                    return self._error(operation, "invalid_arguments", str(exc))
                response = error_response(operation, "", "")
                response.pop("error")
                response.update(ok=True, result=result)
                return self._json(response)
            if timeout is not None and not _is_positive_number(timeout):
                return self._error(operation, "invalid_arguments", "timeout must be a positive number of seconds")
            if operation.startswith("host."):
                return self._error(operation, "unknown_operation", UNKNOWN_OPERATION_MESSAGE)

            if operation.startswith("pdm."):
                try:
                    settings = await pdm_link.load_stored_pdm_settings(config)
                    if settings is not None:
                        if not settings["enabled"]:
                            raise pdm_link.PdmLinkError("no_link")
                        config = {
                            **config,
                            "allow_write": settings["allow_write"],
                            "connection_config": {
                                **(config.get("connection_config") or {}),
                                "pdm_tool_id": settings["pdm_tool_id"],
                                "pdm_approved_folders": settings["pdm_approved_folders"],
                            },
                        }
                except pdm_link.PdmLinkError as exc:
                    return self._error(operation, "pdm_not_linked", "The PDM tool is not linked.", {"reason": exc.reason})

            link = None
            try:
                link = await pdm_link.resolve_pdm_link(config)
            except pdm_link.PdmLinkError as exc:
                if operation.startswith("pdm.") and operation != "pdm.status":
                    return self._error(operation, "pdm_not_linked", "The PDM tool is not linked.", {"reason": exc.reason})
            except Exception:
                if operation.startswith("pdm.") and operation != "pdm.status":
                    raise
            if operation in pdm_link.PDM_INDEX_OPERATIONS:
                if link is None:
                    return self._error(operation, "pdm_not_linked", "The PDM tool is not linked.", {"reason": "no_link"})
                try:
                    result = await pdm_link.execute_index_operation(config, link, operation, arguments or {})
                except pdm_link.PdmArgumentError as exc:
                    return self._error(operation, "invalid_arguments", str(exc))
                except Exception as exc:
                    frames = [(frame.filename, frame.lineno, frame.name) for frame in traceback.extract_tb(exc.__traceback__)]
                    logger.error("PDM index operation failed for tool %s, operation %s: %s; frames %s", config.get("id"), operation, type(exc).__name__, frames)
                    return self._error(operation, "internal_error", "The PDM index search failed; check the embedding provider and the index.")
                response = error_response(operation, "", "")
                response.pop("error")
                response.update(ok=True, result=result)
                return self._json(response)

            cap = resolve_timeout_cap(config)
            if operation == "script.run":
                cap = min(600, cap)
            budget = float(min(120, cap) if operation == "script.run" else cap) if timeout is None else min(float(timeout), float(cap))
            deadline = self.clock() + budget
            connection = config.get("connection_config") or {}
            allow_write = bool(config.get("allow_write", False))
            if operation in pdm_link.PDM_WRITE_OPERATIONS:
                if not allow_write:
                    return self._error(operation, "write_not_allowed", "Enable write access for this operation.")
                if not connection.get("pdm_approved_folders"):
                    return self._error(operation, "pdm_folder_not_approved", "No PDM folders are approved.", {"approved_folders": []})
            key = _tool_key(config)

            async def work(ticket):
                if ticket.waited and self._remaining(deadline) < MIN_OPERATION_SECONDS:
                    return self._updating_error(operation, config)
                return await self._execute_host(config, operation, arguments, connection, allow_write, key, link, deadline)

            return await host_activity.run_call(host_key(config), max(0.0, self._remaining(deadline) - MIN_OPERATION_SECONDS), work)
        except HostUpdatingError:
            return self._updating_error(safe_operation, config)
        except _HostMismatchError:
            return self._error(safe_operation, "internal_error", HOST_MISMATCH_MESSAGE)
        except SolidworksTransportError as exc:
            if exc.output_limit:
                return self._error(safe_operation, "internal_error", str(exc))
            if exc.kind == "timeout":
                return self._error(safe_operation, "operation_timeout", "The operation timed out; check the SolidWorks agent and the timeout.")
            if exc.kind == "no_response":
                return self._error(safe_operation, "host_not_running", "Check that the SolidWorks agent is installed and running.")
            return self._error(safe_operation, "internal_error", "Check the SSH connection and credentials.")
        except Exception:
            return self._error(safe_operation, "internal_error", "The operation failed; check the SolidWorks agent and the tool's connection settings.")

    async def _execute_host(self, config, operation, arguments, connection, allow_write, key, link, deadline):
        transport = self.transport_factory(config)
        pdm_context = pdm_link.build_pdm_context(config, link, await pdm_link.resolve_actor_label()) if link and operation != "script.run" else None

        def context():
            value = self._context(connection, allow_write, self._remaining(deadline))
            if pdm_context is not None:
                value["pdm"] = pdm_context
            return value

        if operation == "describe":
            response = await self._describe(transport, key, context(), self._remaining(deadline), config)
            if link and response.get("ok"):
                response["result"]["operations"].extend(copy.deepcopy(_PDM_OPERATIONS))
            if response.get("ok"):
                response["result"]["scripts_enabled"] = bool(connection.get("allow_scripts", False))
            return self._json(response)

        fetched = False
        if operation != "status":
            contract = self._cached_contract(key)
            if contract is not None and operation.startswith("pdm.") and not any(item.get("name") == operation for item in contract["result"]["operations"]):
                if key is not None:
                    self._contracts.pop(key, None)
                contract = None
            if contract is None:
                fetched = True
                contract = await self._describe(transport, key, context(), self._remaining(deadline), config)
                if not contract.get("ok"):
                    return self._json(contract)
            contract, denied, fetched = await self._validate_contract(contract, operation, allow_write, fetched, transport, key, context, deadline, config)
            if denied is not None:
                return denied
            entry = next(item for item in contract["result"]["operations"] if item.get("name") == operation)
            if operation.startswith("pdm.") and entry.get("access") == "write" and not connection.get("pdm_approved_folders"):
                return self._error(operation, "pdm_folder_not_approved", "No PDM folders are approved.", {"approved_folders": []})
        remaining = self._remaining(deadline)
        if fetched and remaining < MIN_OPERATION_SECONDS:
            return self._error(operation, "operation_timeout", "The operation timed out; check the SolidWorks agent and the timeout.")

        request = build_request(operation, arguments, context())
        try:
            encode_request(request)
        except SolidworksProtocolError:
            details = {"field": "source"} if operation == "script.run" else None
            return self._error(operation, "invalid_arguments", "The request is invalid or too large.", details)
        response = await self._call(transport, key, request, remaining, config)
        if operation in ("model.export", "model.preview"):
            if response.get("ok"):
                return await self._fetch_artifacts(response, transport, config, deadline)
            error = response.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            code = code if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,64}", code) else "internal_error"
            return self._error(operation, code, "The SolidWorks agent could not complete the requested operation.")
        if operation == "status" and response.get("ok"):
            response["result"]["scripts_enabled"] = bool(connection.get("allow_scripts", False))
        return self._json(response)

    async def _fetch_artifacts(self, response, transport, config, deadline):
        operation = response["operation"]
        result = response.get("result")
        tool_id = config.get("id")
        tool_id = tool_id if isinstance(tool_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tool_id) else "unknown"
        if not isinstance(result, dict) or not isinstance(result.get("files"), list):
            logger.warning("SolidWorks artifact failure category=invalid_metadata phase=verify operation=%s tool_id=%s", operation, tool_id)
            return self._error(operation, "artifact_transfer_failed", "The agent returned invalid artifact metadata.")
        files = result["files"]
        paths = []
        invalid_path = False
        for item in files:
            try:
                if not isinstance(item, dict):
                    raise ValueError()
                paths.append(_artifact_remote_path(item.get("path")))
            except ValueError:
                invalid_path = True

        phase = "verify"
        try:
            if invalid_path:
                logger.warning("SolidWorks artifact failure category=invalid_path phase=verify operation=%s tool_id=%s", operation, tool_id)
                return self._error(operation, "artifact_transfer_failed", "The agent returned an invalid artifact path.")
            if not 1 <= len(files) <= 3:
                logger.warning("SolidWorks artifact failure category=invalid_metadata phase=verify operation=%s tool_id=%s", operation, tool_id)
                return self._error(operation, "artifact_transfer_failed", "The agent returned invalid artifact metadata.")
            temp_root = Path(settings.index_data_path)
            with tempfile.TemporaryDirectory(dir=temp_root if temp_root.is_dir() else None) as directory:
                downloaded = []
                for index, (item, remote) in enumerate(zip(files, paths)):
                    size, digest = item.get("size_bytes"), item.get("sha256")
                    if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("Invalid artifact metadata")
                    if size > MAX_ARTIFACT_BYTES:
                        raise ArtifactTooLarge()
                    remaining = self._remaining(deadline)
                    if remaining <= 0:
                        raise TimeoutError()
                    local = Path(directory) / str(index)
                    phase = "download"
                    transferred = await transport.download_file(remote, local, max_bytes=MAX_ARTIFACT_BYTES, timeout_seconds=remaining)
                    phase = "verify"
                    if transferred != size or local.stat().st_size != size:
                        raise ValueError("Artifact size mismatch")
                    hasher = hashlib.sha256()
                    with local.open("rb") as source:
                        while chunk := source.read(65536):
                            hasher.update(chunk)
                    if hasher.hexdigest() != digest:
                        raise ValueError("Artifact hash mismatch")
                    downloaded.append((item, local))

                context = current_context()
                stored = []
                preview_id = None
                for item, local in sorted(downloaded, key=lambda pair: pair[0].get("role") != "preview"):
                    phase = "verify"
                    role = item.get("role")
                    formats = {"preview": {"png"}, "model": {"step", "stl", "3mf", "glb"}, "native": {"zip", "sldprt", "sldasm", "slddrw"}}
                    if role not in formats or item.get("format") not in formats[role] or not isinstance(item.get("file_name"), str):
                        raise ValueError("Invalid artifact role")
                    phase = "store"
                    artifact = await store_artifact(
                        local,
                        file_name=item.get("file_name"),
                        kind=role,
                        format=item.get("format"),
                        sha256=item["sha256"],
                        owner_user_id=getattr(context, "user_id", None),
                        tool_config_id=config.get("id"),
                        surface=getattr(context, "surface", None),
                        mcp_route=getattr(context, "mcp_route", None),
                        preview_artifact_id=preview_id if role != "preview" else None,
                    )
                    stored.append(artifact)
                    if role == "preview":
                        preview_id = artifact["id"]
                primary = next((artifact for artifact in stored if artifact["kind"] != "preview"), stored[0])
                document_value = result.get("document")
                document = document_value if isinstance(document_value, dict) else {}
                if not document.get("title") and isinstance(document.get("path"), str):
                    document = {**document, "title": ntpath.basename(document["path"])}
                warnings_value = result.get("warnings")
                warnings = warnings_value if isinstance(warnings_value, list) else []
                response = {
                    "protocol": response.get("protocol"),
                    "ok": True,
                    "operation": operation,
                    "result": {
                        "artifact": primary,
                        "artifacts": [{key: artifact[key] for key in ("id", "kind", "format", "size_bytes")} for artifact in stored],
                        "document": {key: _safe_result_text(document.get(key), 120) for key in ("type", "configuration", "title")},
                        "warnings": [_safe_result_text(warning, 160) for warning in warnings[:3]],
                    },
                    "host": {
                        key: _safe_result_text((response.get("host") or {}).get(key), 100)
                        for key in ("host_version", "solidworks_revision", "solidworks_version", "instance_id")
                    },
                }
                for key in ("warnings", "document", "host"):
                    if len(self._json(response)) <= 1800:
                        break
                    if key == "host":
                        response.pop(key, None)
                    else:
                        response["result"].pop(key, None)
                output = self._json(response)
                if len(output) > 1800:
                    raise ValueError("Artifact metadata too large")
                return output
        except ArtifactTooLarge:
            logger.warning("SolidWorks artifact failure category=too_large phase=%s operation=%s tool_id=%s", phase, operation, tool_id)
            return self._error(operation, "artifact_too_large", "The exported artifact exceeds the size limit.")
        except Exception:
            logger.warning("SolidWorks artifact failure category=failed phase=%s operation=%s tool_id=%s", phase, operation, tool_id)
            return self._error(operation, "artifact_transfer_failed", "The exported artifact could not be transferred or verified.")
        finally:
            for remote in paths:
                try:
                    await transport.remove_file(remote, timeout_seconds=max(0.01, min(2.0, self._remaining(deadline))))
                except Exception:
                    logger.warning("SolidWorks artifact failure category=failed phase=delete operation=%s tool_id=%s", operation, tool_id)

    def _updating_error(self, operation: str, config: dict, message: str = "The SolidWorks agent is being updated. Try again in a few minutes.") -> str:
        state = host_updater.state_for(config.get("id") or "") or {}
        return self._error(operation, "host_updating", message, {key: state.get(key) for key in ("state", "installed_version", "target_version")})

    async def _validate_contract(self, contract, operation, allow_write, fetched, transport, key, context, deadline, config):
        denied = self._check_contract(contract, operation, allow_write)
        missing = not any(item.get("name") == operation for item in contract["result"]["operations"])
        if missing and self._outdated_contract(contract):
            if not fetched:
                if key is not None:
                    self._contracts.pop(key, None)
                fetched = True
                contract = await self._describe(transport, key, context(), self._remaining(deadline), config)
                if not contract.get("ok"):
                    return contract, self._json(contract), fetched
                denied = self._check_contract(contract, operation, allow_write)
                missing = not any(item.get("name") == operation for item in contract["result"]["operations"])
            if missing and self._outdated_contract(contract):
                if (config.get("connection_config") or {}).get("auto_update_host", True) is not False:
                    host_updater.request_now(config)
                    message = "This operation needs a newer SolidWorks agent. Ragtime updates the SolidWorks agent when it is idle; try again in a few minutes."
                    return contract, self._updating_error(operation, config, message), fetched
                message = "This operation needs a newer SolidWorks agent. Update it from the SolidWorks agent panel in the tool's settings."
                return contract, self._error(operation, "unknown_operation", message), fetched
        return contract, denied, fetched

    @staticmethod
    def _outdated_contract(contract: dict) -> bool:
        version = contract["result"].get("host_version") or (contract.get("host") or {}).get("host_version")
        installed = parse_host_version(version)
        artifact = get_host_artifact()
        target = parse_host_version(artifact.host_version) if artifact else None
        return installed is not None and target is not None and installed < target

    def _remaining(self, deadline: float) -> float:
        return deadline - self.clock()

    def _check_contract(self, contract: dict, operation: str, allow_write: bool) -> str | None:
        """Return an error envelope when the host contract does not allow the operation, else None."""
        entry = next((item for item in contract["result"]["operations"] if item.get("name") == operation), None)
        if entry is None:
            message = "Update the SolidWorks agent to use PDM operations." if operation.startswith("pdm.") else UNKNOWN_OPERATION_MESSAGE
            return self._error(operation, "unknown_operation", message)
        if entry.get("access") == "write" and not allow_write:
            return self._error(operation, "write_not_allowed", "Enable write access for this operation.")
        return None

    @staticmethod
    def _context(connection: dict, allow_write: bool, remaining_seconds: float) -> dict:
        return {
            "solidworks_version": connection.get("solidworks_version"),
            "instance_mode": connection.get("instance_mode", "background"),
            "max_instances": connection.get("max_instances", 1),
            "idle_shutdown_seconds": connection.get("idle_shutdown_seconds", 900),
            "allow_write": allow_write,
            "allow_scripts": bool(connection.get("allow_scripts", False)),
            "deadline_seconds": max(5, int(remaining_seconds) - 10),
            "max_result_bytes": MAX_RESULT_BYTES,
        }

    @staticmethod
    def _json(response: dict) -> str:
        return json.dumps(response, separators=(",", ":"), ensure_ascii=False)

    def _error(self, operation: str, code: str, message: str, details: dict | None = None) -> str:
        return self._json(error_response(operation, code, message, details))

    async def host_status(self, config: dict) -> dict:
        from ragtime.solidworks.host_status import SolidworksHostStatusService

        return await SolidworksHostStatusService(transport_factory=self.transport_factory).host_status(config)

    async def test_connection(self, config: dict) -> tuple[bool, str, dict | None]:
        from ragtime.solidworks.host_status import SolidworksHostStatusService

        return await SolidworksHostStatusService(transport_factory=self.transport_factory).test_connection(config)

    async def install_host(self, config: dict) -> dict:
        from ragtime.solidworks.host_installer import SolidworksHostInstaller
        from ragtime.solidworks.host_status import SolidworksHostStatusService

        status = SolidworksHostStatusService(transport_factory=self.transport_factory)
        installer = SolidworksHostInstaller(transport_factory=self.transport_factory, status_service=status)

        async def work():
            result = await installer.install_host(config)
            try:
                host_updater.note_manual_action(config, result)
            except Exception as exc:
                logger.warning("Host updater manual action hook failed: %s", type(exc).__name__)
            return result

        try:
            return await host_activity.run_exclusive(host_key(config), 60, work)
        except HostBusyError:
            return await installer.busy_status(config)

    async def uninstall_host(self, config: dict) -> dict:
        from ragtime.solidworks.host_installer import SolidworksHostInstaller
        from ragtime.solidworks.host_status import SolidworksHostStatusService

        status = SolidworksHostStatusService(transport_factory=self.transport_factory)
        installer = SolidworksHostInstaller(transport_factory=self.transport_factory, status_service=status)

        async def work():
            result = await installer.uninstall_host(config)
            try:
                host_updater.note_manual_action(config, result)
            except Exception as exc:
                logger.warning("Host updater manual action hook failed: %s", type(exc).__name__)
            return result

        try:
            return await host_activity.run_exclusive(host_key(config), 60, work)
        except HostBusyError:
            return await installer.busy_status(config)


solidworks_service = SolidworksService()
