"""Admin-only creation and approval of a single PDM sandbox folder."""

import copy
import json
import re
import time
from typing import NoReturn

from fastapi import HTTPException

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.database import get_db
from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.indexer.repository import repository
from ragtime.mcp.server import notify_tools_changed
from ragtime.solidworks.host_activity import HostUpdatingError, host_activity, host_key
from ragtime.solidworks.host_updater import host_updater
from ragtime.solidworks.pdm_link import PdmLink, build_pdm_context, normalize_vault_folder, resolve_actor_label, validate_approved_folders
from ragtime.solidworks.pdm_rights import PdmToolNotFound, pdm_rights_service
from ragtime.solidworks.protocol import build_request, encode_request
from ragtime.solidworks.service import solidworks_service
from ragtime.solidworks.transport import SolidworksTransportError

_REQUIRED_RIGHTS = ("read", "add_folders", "add_files", "check_out")
_LINK_MESSAGE = "Save the PDM link before creating a sandbox folder."


def _fail(status: int, code: str, message: str, **details) -> NoReturn:
    raise HTTPException(status_code=status, detail={"code": code, "message": message, **details})


def _agent_failure(error: dict) -> NoReturn:
    code = error.get("code")
    if not isinstance(code, str):
        code = "pdm_error"
    code = {"timeout": "operation_timeout", "no_response": "host_not_running"}.get(code, code)
    status = {"host_not_running": 503, "host_updating": 503, "operation_timeout": 504}.get(code, 502)
    message = error.get("message")
    if not isinstance(message, str) or not 0 < len(message) <= 1024:
        message = "The SolidWorks agent could not create the folder."
    _fail(status, code if status != 502 else "pdm_error", message)


async def _load(tool_id: str, pdm_tool_id: str) -> tuple[dict, ToolConfig]:
    tool = await repository.get_tool_config(tool_id)
    if tool is None or tool.tool_type != ToolType.SOLIDWORKS or (tool.connection_config or {}).get("pdm_tool_id") != pdm_tool_id:
        _fail(409, "pdm_link_not_saved", _LINK_MESSAGE)
    pdm = await repository.get_tool_config(pdm_tool_id)
    if pdm is None or pdm.tool_type != ToolType.SOLIDWORKS_PDM:
        _fail(409, "pdm_link_not_saved", _LINK_MESSAGE)
    if not tool.enabled or not pdm.enabled:
        _fail(409, "tool_disabled", "Enable the SolidWorks tool and its PDM tool first.")
    return copy.deepcopy(tool.model_dump()), copy.deepcopy(pdm)


def _approval(config: dict, target: str) -> tuple[object, list[str], bool]:
    connection = config.get("connection_config") or {}
    old = connection.get("pdm_approved_folders")
    try:
        folders = validate_approved_folders(old if old is not None else [])
        folded = target.casefold()
        covered = any(folded == folder.casefold() or folded.startswith(folder.casefold() + "\\") for folder in folders)
        for folder in folders:
            if folder.casefold().startswith(folded + "\\"):
                _fail(409, "sandbox_overlap", f"“{target}” overlaps the approved folder “{folder}”.")
        approved = folders if covered else validate_approved_folders([*folders, target])
    except (ValueError, TypeError) as exc:
        _fail(409, "sandbox_overlap", str(exc))
    return old, approved, covered


async def _create(config: dict, pdm_tool_id: str, target: str, pdm: ToolConfig) -> bool:
    try:
        context, timeout = await pdm_rights_service.pdm_context(config, pdm_tool_id, pdm)
    except PdmToolNotFound:
        _agent_failure({"message": "The linked PDM tool was not found."})
    if context is None:
        _agent_failure({"message": "The linked vault name is invalid."})
    context["allow_write"] = True
    link = PdmLink(pdm_tool_id, pdm.name, "", context["pdm"]["vault"])
    context["pdm"] = build_pdm_context(config, link, await resolve_actor_label())
    context["pdm"]["approved_folders"] = [target]
    request = build_request("pdm.folder.create", {"path": target}, context)
    encode_request(request)
    deadline = time.monotonic() + timeout

    async def work(ticket):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _agent_failure({"code": "operation_timeout", "message": "The SolidWorks agent did not answer in time."})
        return await pdm_rights_service.transport_factory(config).call(request, timeout_seconds=remaining)

    try:
        response = await host_activity.run_call(host_key(config), timeout, work)
    except HostUpdatingError:
        _agent_failure({"code": "host_updating", "message": "The SolidWorks agent is being updated. Try again in a few minutes."})
    except SolidworksTransportError as exc:
        _agent_failure({"code": exc.kind, "message": "The SolidWorks agent could not be reached."})
    except HTTPException:
        raise
    except Exception:
        _agent_failure({"message": "The SolidWorks agent could not create the folder."})
    if (
        not isinstance(response, dict)
        or response.get("id") != request["id"]
        or response.get("operation") != request["operation"]
        or type(response.get("ok")) is not bool
    ):
        _agent_failure({"message": "The SolidWorks agent gave an unexpected answer."})
    host = response.get("host")
    version = host.get("host_version") if isinstance(host, dict) else None
    if isinstance(version, str):
        host_updater.observe(config, version)
        solidworks_service._forget_if_version_changed(config.get("id"), version)
    if not response["ok"]:
        error = response.get("error")
        _agent_failure(error if isinstance(error, dict) else {})
    result = response.get("result")
    if not isinstance(result, dict) or type(result.get("created")) is not bool:
        _agent_failure({"message": "The SolidWorks agent gave an unexpected answer."})
    return result["created"]


def _security_state(config: dict, pdm: ToolConfig) -> tuple:
    connection = dict(config.get("connection_config") or {})
    connection.pop("pdm_approved_folders", None)
    return (
        config.get("id"),
        config.get("name"),
        config.get("enabled"),
        config.get("allow_write"),
        config.get("timeout_max_seconds"),
        connection,
        pdm.id,
        pdm.enabled,
        pdm.tool_type,
        pdm.connection_config,
    )


async def create_sandbox(tool_id: str, pdm_tool_id: str, parent: str, name: str) -> dict:
    try:
        parent = normalize_vault_folder(parent)
        if parent == "\\" or not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}", name) is None or name.endswith((".", " ")):
            raise ValueError("Invalid sandbox parent or name.")
        normalize_vault_folder(parent + "\\" + name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    target = parent + "\\" + name
    config, pdm = await _load(tool_id, pdm_tool_id)
    initial_security = _security_state(config, pdm)
    old, approved, covered = _approval(config, target)
    try:
        rights = await pdm_rights_service.rights(config, pdm_tool_id, [parent], tool=pdm)
    except PdmToolNotFound:
        _agent_failure({"message": "The linked PDM tool was not found."})
    if rights["error"]:
        _agent_failure(rights["error"])
    item = rights["folders"][0]
    granted = item.get("rights") or {}
    missing = [right for right in _REQUIRED_RIGHTS if not granted.get(right)]
    if missing:
        _fail(409, "pdm_rights_missing", "The PDM user is missing required rights.", missing=missing)
    created = await _create(config, pdm_tool_id, target, pdm)
    if not created:
        _fail(409, "sandbox_exists", f"A folder named “{name}” already exists in {parent}. Choose another name.")
    db = await get_db()
    for attempt in range(2):
        count = await db.execute_raw(
            """UPDATE tool_configs SET connection_config = jsonb_set(connection_config, '{pdm_approved_folders}', $1::jsonb, true),
               allow_write = true, updated_at = CURRENT_TIMESTAMP WHERE id = $2 AND connection_config->'pdm_approved_folders' IS NOT DISTINCT FROM $3::jsonb
               AND connection_config->>'pdm_tool_id' = $4 AND tool_type = 'solidworks' AND enabled = true AND updated_at = $5
               AND EXISTS (SELECT 1 FROM tool_configs AS pdm WHERE pdm.id = $4 AND pdm.tool_type = 'solidworks_pdm'
                   AND pdm.enabled = true AND pdm.updated_at = $6 AND pdm.connection_config->>'database' = $7 FOR SHARE)""",
            json.dumps(approved),
            tool_id,
            json.dumps(old) if "pdm_approved_folders" in (config.get("connection_config") or {}) else None,
            pdm_tool_id,
            config["updated_at"],
            pdm.updated_at,
            pdm.connection_config.get("database"),
        )
        if count:
            invalidate_settings_cache()
            notify_tools_changed()
            solidworks_service._contracts.pop(tool_id, None)
            return {
                "folder": target,
                "created": created,
                "covered": covered,
                "approved_folders": approved,
                "allow_write": True,
                "scripts_enabled": bool((config.get("connection_config") or {}).get("allow_scripts", False)),
            }
        if attempt == 0:
            try:
                config, pdm = await _load(tool_id, pdm_tool_id)
            except HTTPException:
                _fail(409, "sandbox_conflict", "The tool settings changed while creating the sandbox. Try again.")
            if _security_state(config, pdm) != initial_security:
                _fail(409, "sandbox_conflict", "The tool settings changed while creating the sandbox. Try again.")
            old, approved, covered = _approval(config, target)
    _fail(409, "sandbox_conflict", "The tool settings changed while creating the sandbox. Try again.")
