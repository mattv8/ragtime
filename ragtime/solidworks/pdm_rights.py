"""Administrative vault rights and folder discovery."""

import logging
import time
from collections.abc import Callable

from ragtime.indexer.models import ToolType
from ragtime.indexer.repository import repository
from ragtime.solidworks.host_activity import HostUpdatingError, host_activity, host_key
from ragtime.solidworks.host_status import _MESSAGES, _host_context, _status_timeout_seconds
from ragtime.solidworks.pdm_link import normalize_vault_folder, pdm_context_tool_id
from ragtime.solidworks.protocol import SolidworksProtocolError, build_request, encode_request
from ragtime.solidworks.transport import SolidworksTransport, SolidworksTransportError, SshSolidworksTransport

logger = logging.getLogger(__name__)

_ERRORS = {
    "ssh_auth_failed": _MESSAGES["ssh_auth_failed"],
    "ssh_unreachable": _MESSAGES["ssh_unreachable"],
    "host_not_running": _MESSAGES["host_not_running"],
    "no_interactive_session": _MESSAGES["no_interactive_session"],
    "operation_timeout": "The SolidWorks agent did not answer in time. Try again.",
    "host_updating": "The SolidWorks agent is being updated. Try again in a few minutes.",
    "host_update_required": "Update the SolidWorks agent to 0.4.0 or later to check vault access.",
    "pdm_not_found": "That vault folder no longer exists.",
    "stale_snapshot": "The folder changed while you were browsing. Reload it.",
    "invalid_arguments": "The vault path is not valid.",
    "internal_error": "The SolidWorks agent gave an unexpected answer.",
    "pdm_vault_invalid": "The search index's vault database name cannot be used as a vault name.",
}
_PDM_REASONS = {
    "client_missing": "The PDM client is not installed on this machine.",
    "view_missing": "This machine has no vault view for the linked vault.",
    "login_failed": "Logging in to the vault view failed. Log in once in the PDM client on this machine.",
    "busy": "The PDM client is busy with an earlier request. Try again in a moment.",
}
_RIGHTS = ("read", "add_files", "check_out", "add_folders")


class PdmToolNotFound(Exception):
    pass


def _error(code: str, reason: object = None) -> dict:
    code = {"timeout": "operation_timeout", "no_response": "host_not_running", "unknown_operation": "host_update_required"}.get(code, code)
    if code == "pdm_unavailable" and isinstance(reason, str) and reason in _PDM_REASONS:
        return {"code": code, "message": _PDM_REASONS[reason]}
    if code not in _ERRORS:
        code = "internal_error"
    return {"code": code, "message": _ERRORS[code]}


def _valid_rights(value: object) -> bool:
    return isinstance(value, dict) and all(type(value.get(key)) is bool for key in _RIGHTS)


def _write_ready(rights: dict | None) -> bool:
    return bool(rights and rights["read"] and rights["add_files"] and rights["check_out"])


def _unchecked(path: str) -> dict:
    return {"path": path, "checked": False, "exists": None, "rights": None, "write_ready": False}


def _rights_items(result: object, paths: list[str]) -> tuple[str | None, list[dict]]:
    if not isinstance(result, dict) or "user" not in result or not (result.get("user") is None or isinstance(result.get("user"), str)):
        raise ValueError("Invalid result")
    items = result.get("items")
    if not isinstance(items, list) or len(items) != len(paths):
        raise ValueError("Invalid items")
    output = []
    for item, path in zip(items, paths, strict=True):
        if (
            not isinstance(item, dict)
            or not {"vault_path", "checked", "exists", "rights"} <= item.keys()
            or item.get("vault_path") != path
            or type(item.get("checked")) is not bool
        ):
            raise ValueError("Invalid item")
        checked, exists, rights = item["checked"], item.get("exists"), item.get("rights")
        if (not checked and (exists is not None or rights is not None)) or (
            checked and (type(exists) is not bool or (exists and not _valid_rights(rights)) or (not exists and rights is not None))
        ):
            raise ValueError("Invalid rights")
        output.append({"path": path, "checked": checked, "exists": exists, "rights": rights, "write_ready": _write_ready(rights)})
    return result.get("user"), output


def _folder_result(result: object) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise ValueError("Invalid result")
    if type(result.get("offset")) is not int or result["offset"] < 0 or type(result.get("total")) is not int or result["total"] < 0:
        raise ValueError("Invalid page")
    next_offset = result.get("next_offset")
    if next_offset is not None and (type(next_offset) is not int or next_offset < 0):
        raise ValueError("Invalid next offset")
    snapshot = result.get("snapshot")
    if not isinstance(snapshot, str) or len(snapshot) != 64 or any(c not in "0123456789abcdef" for c in snapshot):
        raise ValueError("Invalid snapshot")
    items = []
    for item in result["items"]:
        if not isinstance(item, dict) or "rights" not in item or not isinstance(item.get("name"), str) or not isinstance(item.get("vault_path"), str):
            raise ValueError("Invalid child")
        rights = item.get("rights")
        if rights is not None and not _valid_rights(rights):
            raise ValueError("Invalid rights")
        items.append({"name": item["name"], "vault_path": item["vault_path"], "rights": rights, "write_ready": _write_ready(rights)})
    return {"items": items, "offset": result["offset"], "next_offset": next_offset, "total": result["total"], "snapshot": snapshot, "error": None}


class PdmRightsService:
    def __init__(self, transport_factory: Callable[[dict], SolidworksTransport] = SshSolidworksTransport):
        self.transport_factory = transport_factory

    async def _context(self, config: dict, pdm_tool_id: str) -> tuple[dict | None, int]:
        tool = await repository.get_tool_config(pdm_tool_id)
        if tool is None or tool.tool_type != ToolType.SOLIDWORKS_PDM:
            raise PdmToolNotFound
        vault = (tool.connection_config or {}).get("database")
        if not isinstance(vault, str) or not 1 <= len(vault) <= 128 or any(c in "\\/" or ord(c) < 32 or ord(c) == 127 for c in vault):
            return None, 0
        timeout = _status_timeout_seconds(config)
        context = _host_context(config, timeout)
        context["allow_write"] = False
        context["pdm"] = {"tool_id": pdm_context_tool_id(config), "vault": vault, "approved_folders": [], "comment_prefix": ""}
        return context, timeout

    async def _call(self, transport: SolidworksTransport, request: dict, remaining: float) -> tuple[object, dict | None]:
        if remaining <= 0:
            return None, _error("timeout")
        try:
            response = await transport.call(request, timeout_seconds=remaining)
            if (
                not isinstance(response, dict)
                or response.get("id") != request["id"]
                or response.get("operation") != request["operation"]
                or type(response.get("ok")) is not bool
            ):
                return None, _error("internal_error")
            if not response["ok"]:
                host_error = response.get("error")
                if not isinstance(host_error, dict):
                    return None, _error("internal_error")
                details = host_error.get("details")
                reason = details.get("reason") if isinstance(details, dict) else None
                code = host_error.get("code")
                return None, _error(code if isinstance(code, str) else "internal_error", reason)
            return response.get("result"), None
        except SolidworksTransportError as exc:
            logger.warning("PDM host call failed: %s", type(exc).__name__)
            return None, _error(exc.kind)
        except Exception as exc:
            logger.warning("PDM host call failed: %s", type(exc).__name__)
            return None, _error("internal_error")

    async def rights(self, config: dict, pdm_tool_id: str, folders: list[str]) -> dict:
        paths = ["\\"]
        seen = {"\\"}
        for folder in folders:
            path = normalize_vault_folder(folder)
            if path.casefold() not in seen:
                seen.add(path.casefold())
                paths.append(path)
        context, timeout = await self._context(config, pdm_tool_id)
        items = [_unchecked(path) for path in paths]
        error = _error("pdm_vault_invalid") if context is None else None
        user = None
        if context is not None:
            deadline = time.monotonic() + timeout

            async def work(ticket):
                nonlocal error, user
                transport = self.transport_factory(config)
                start = 0
                while start < len(paths):
                    end = min(start + 25, len(paths))
                    while end > start:
                        request = build_request("pdm.folder.rights", {"paths": paths[start:end]}, context)
                        try:
                            encode_request(request)
                            break
                        except SolidworksProtocolError:
                            end -= 1
                    if end == start:
                        error = error or _error("internal_error")
                        start += 1
                        continue
                    result, failure = await self._call(transport, request, deadline - time.monotonic())
                    if failure is None:
                        try:
                            batch_user, batch = _rights_items(result, paths[start:end])
                            user = user or batch_user
                            items[start:end] = batch
                        except (ValueError, TypeError, KeyError):
                            failure = _error("internal_error")
                    error = error or failure
                    start = end

            try:
                await host_activity.run_call(host_key(config), timeout, work)
            except HostUpdatingError:
                error = _error("host_updating")
        checked = all(item["checked"] for item in items)
        if not checked and error is None:
            error = _error("timeout")
        return {"checked": checked, "user": user, "root": items[0], "folders": items[1:], "error": error}

    async def folders(self, config: dict, pdm_tool_id: str, path: str, offset: int, snapshot: str | None) -> dict:
        empty = {"items": [], "offset": offset, "next_offset": None, "total": 0, "snapshot": None}
        context, timeout = await self._context(config, pdm_tool_id)
        if context is None:
            return {**empty, "error": _error("pdm_vault_invalid")}
        arguments = {"path": path, "offset": offset, "limit": 50}
        if snapshot is not None:
            arguments["snapshot"] = snapshot
        request = build_request("pdm.folder.children", arguments, context)
        deadline = time.monotonic() + timeout

        async def work(ticket):
            return await self._call(self.transport_factory(config), request, deadline - time.monotonic())

        try:
            result, error = await host_activity.run_call(host_key(config), timeout, work)
        except HostUpdatingError:
            return {**empty, "error": _error("host_updating")}
        if error is None:
            try:
                return _folder_result(result)
            except (ValueError, TypeError, KeyError):
                error = _error("internal_error")
        return {**empty, "error": error}


pdm_rights_service = PdmRightsService()
