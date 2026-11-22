"""PDM link configuration and indexed vault operations."""

import json
import re
from dataclasses import dataclass
from typing import Any

from ragtime.core.database import get_db
from ragtime.indexer.repository import repository
from ragtime.indexer.utils import safe_tool_name

PDM_HOST_OPERATION_PREFIX = "pdm."
PDM_INDEX_OPERATIONS = ("pdm.search", "pdm.lookup")
PDM_WRITE_OPERATIONS = frozenset(
    {
        "pdm.folder.create",
        "pdm.file.add",
        "pdm.file.checkout",
        "pdm.file.undo_checkout",
        "pdm.file.checkin",
        "pdm.variables.set",
    }
)
_NOTE = "Indexed snapshot of checked-in vault data; it can lag the vault. Use pdm.file.info for live state."


class PdmLinkError(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class PdmArgumentError(ValueError):
    pass


@dataclass(frozen=True)
class PdmLink:
    tool_id: str
    tool_name: str
    index_name: str
    vault: str


def normalize_vault_folder(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 260:
        raise ValueError("A vault path must contain 1-260 characters.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("A vault path cannot contain control characters.")
    if any(ord(char) > 126 for char in value):
        raise ValueError("Use plain ASCII characters in approved folder paths.")
    path = value.replace("/", "\\")
    if not path.startswith("\\") or path.startswith("\\\\") or any(char in path for char in ':*?"<>|'):
        raise ValueError("A vault path must be vault-relative.")
    if path == "\\":
        return path
    segments = path[1:].split("\\")
    if any(not part or part in (".", "..") or part != part.strip(" ") or part.endswith(".") for part in segments):
        raise ValueError("A vault path contains an invalid segment.")
    return "\\" + "\\".join(segments)


def validate_approved_folders(values: list[str]) -> list[str]:
    if not isinstance(values, list) or len(values) > 25:
        raise ValueError("Provide at most 25 approved folders.")
    folders = [normalize_vault_folder(value) for value in values]
    folded = [folder.casefold() for folder in folders]
    if "\\" in folded or len(set(folded)) != len(folded):
        raise ValueError("Approved folders cannot include the root or duplicates.")
    if any(other.startswith(folder + "\\") for folder in folded for other in folded if folder != other):
        raise ValueError("Approved folders cannot overlap.")
    return folders


async def load_stored_pdm_settings(config: dict) -> dict | None:
    tool_id = config.get("id")
    if not isinstance(tool_id, str):
        return None
    tool = await repository.get_tool_config(tool_id)
    if tool is None:
        raise PdmLinkError("no_link")
    connection = tool.connection_config or {}
    return {
        "allow_write": bool(tool.allow_write),
        "pdm_tool_id": connection.get("pdm_tool_id"),
        "pdm_approved_folders": connection.get("pdm_approved_folders") or [],
        "enabled": bool(tool.enabled),
    }


async def resolve_pdm_link(config: dict) -> PdmLink:
    tool_id = (config.get("connection_config") or {}).get("pdm_tool_id")
    if not tool_id:
        raise PdmLinkError("no_link")
    tool = await repository.get_tool_config(tool_id)
    if tool is None or tool.tool_type != "solidworks_pdm":
        raise PdmLinkError("linked_tool_missing")
    if not tool.enabled:
        raise PdmLinkError("linked_tool_disabled")
    name = safe_tool_name(tool.name)
    index_name = f"pdm_{name}"
    db = await get_db()
    rows = await db.query_raw("SELECT COUNT(*) as count FROM pdm_embeddings WHERE index_name = $1", index_name)
    if not rows or not int(rows[0].get("count", 0)):
        index_name = f"pdm_{tool_id}"
    return PdmLink(tool_id, tool.name, index_name, tool.connection_config["database"])


async def resolve_actor_label() -> str:
    try:
        from ragtime.content_protection.service import current_context
        from ragtime.core.database import get_db

        context = current_context()
        if context and context.user_id:
            db = await get_db()
            user = await db.user.find_unique(where={"id": context.user_id})
            if user and user.username:
                return user.username
    except Exception:
        pass
    return "unknown"


def build_pdm_context(config: dict, link: PdmLink, actor: str) -> dict:
    def clean(value: str) -> str:
        chars = []
        units = 0
        for char in value:
            if ord(char) < 32 or ord(char) == 127 or char in "[];":
                continue
            size = 2 if ord(char) > 0xFFFF else 1
            if units + size > 60:
                break
            chars.append(char)
            units += size
        return "".join(chars)

    tool_id = config.get("id")
    if not isinstance(tool_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_id) is None:
        tool_id = "unsaved"
    return {
        "tool_id": tool_id,
        "vault": link.vault,
        "approved_folders": (config.get("connection_config") or {}).get("pdm_approved_folders") or [],
        "comment_prefix": f"[Ragtime: {clean(config.get('name') or '')}; user: {clean(actor)}] ",
    }


async def execute_index_operation(config: dict, link: PdmLink, operation: str, arguments: dict) -> dict:
    from ragtime.indexer.pdm_service import get_pdm_index_summary, lookup_pdm_document_states, search_pdm_documents  # inline-import: keep

    allowed = {"limit", "query", "document_type"} if operation == "pdm.search" else {"limit", "document_id", "filename", "part_number", "configuration"}
    if operation not in PDM_INDEX_OPERATIONS or not isinstance(arguments, dict) or arguments.keys() - allowed:
        raise PdmArgumentError("Invalid arguments.")
    limit = arguments.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 25:
        raise PdmArgumentError("limit must be an integer from 1 to 25.")
    if operation == "pdm.search":
        query = arguments.get("query")
        document_type = arguments.get("document_type")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
            raise PdmArgumentError("query must contain 1-500 characters.")
        query = query.strip()
        if "document_type" in arguments:
            if not isinstance(document_type, str) or document_type.upper() not in ("SLDPRT", "SLDASM", "SLDDRW"):
                raise PdmArgumentError("Invalid document_type.")
            document_type = document_type.upper()
        items = await search_pdm_documents(query, link.index_name, document_type, limit)
    else:
        if not any(key in arguments for key in ("document_id", "filename", "part_number")):
            raise PdmArgumentError("Provide a document_id, filename or part_number.")
        if "document_id" in arguments and type(arguments["document_id"]) is not int:
            raise PdmArgumentError("document_id must be an integer.")
        for key in ("filename", "part_number", "configuration"):
            if key in arguments and (not isinstance(arguments[key], str) or not 1 <= len(arguments[key]) <= 260):
                raise PdmArgumentError(f"{key} must contain 1-260 characters.")
        items = await lookup_pdm_document_states(link.index_name, arguments.get("document_id"), arguments.get("filename"), arguments.get("part_number"), limit)
        if "configuration" in arguments:
            name = arguments["configuration"].casefold()
            items = [{**item, "configurations": [c for c in item["configurations"] if name != "@" and c["name"].casefold() == name]} for item in items]
    summary = await get_pdm_index_summary(link.index_name)
    result: dict[str, Any] = {"items": items, "index": {"tool": link.tool_name, **summary}, "note": _NOTE}
    while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 32768 and result["items"]:
        result["items"] = result["items"][:-1]
        result["truncated"] = True
    return result
