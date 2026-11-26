"""PDM link configuration and indexed vault operations."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from ragtime.core.database import get_db
from ragtime.indexer.repository import repository
from ragtime.indexer.utils import safe_tool_name

logger = logging.getLogger(__name__)

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
PDM_DOCUMENT_TYPES = ("SLDPRT", "SLDASM", "SLDDRW")
PDM_DETAILS = ("summary", "full")
MAX_LOOKUP_OFFSET = 1000
MAX_RESULT_BYTES = 32768
TOO_LARGE_NOTE = "One item is too large for detail=full; use detail=summary or document_id."
SEMANTIC_UNAVAILABLE_NOTE = "Semantic search is unavailable (the embedding provider failed or is not configured); only filename matches are shown."
_SEARCH_ARGUMENTS = frozenset({"limit", "query", "document_type"})
_LOOKUP_ARGUMENTS = frozenset({"limit", "document_id", "filename", "part_number", "configuration", "document_type", "offset", "detail"})
_LOOKUP_KEYS = ("document_id", "filename", "part_number")


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


async def vault_linked(connection: dict) -> bool:
    """True when the connection links an enabled solidworks_pdm tool. A failed lookup counts as not linked."""
    tool_id = connection.get("pdm_tool_id")
    if not tool_id:
        return False
    try:
        tool = await repository.get_tool_config(tool_id)
    except Exception as exc:
        logger.warning("Could not read the linked PDM tool (%s); treating the vault as not linked.", type(exc).__name__)
        return False
    return tool is not None and tool.tool_type == "solidworks_pdm" and bool(tool.enabled)


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


PDM_ADMIN_OPERATIONS = frozenset({"pdm.folder.rights", "pdm.folder.children"})


def pdm_context_tool_id(config: dict) -> str:
    tool_id = config.get("id")
    if not isinstance(tool_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_id) is None:
        return "unsaved"
    return tool_id


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

    return {
        "tool_id": pdm_context_tool_id(config),
        "vault": link.vault,
        "approved_folders": (config.get("connection_config") or {}).get("pdm_approved_folders") or [],
        "comment_prefix": f"[Ragtime: {clean(config.get('name') or '')}; user: {clean(actor)}] ",
    }


def _limit_argument(arguments: dict) -> int:
    limit = arguments.get("limit", 10)
    if type(limit) is not int or not 1 <= limit <= 25:
        raise PdmArgumentError("limit must be an integer from 1 to 25.")
    return limit


def _document_type_argument(arguments: dict) -> str | None:
    if "document_type" not in arguments:
        return None
    value = arguments["document_type"]
    if not isinstance(value, str) or value.upper() not in PDM_DOCUMENT_TYPES:
        raise PdmArgumentError("Invalid document_type.")
    return value.upper()


def _search_query(arguments: dict) -> str:
    query = arguments.get("query")
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise PdmArgumentError("query must contain 1-500 characters.")
    return query.strip()


def _validate_lookup_arguments(arguments: dict) -> None:
    if not any(key in arguments for key in _LOOKUP_KEYS):
        raise PdmArgumentError("Provide a document_id, filename or part_number.")
    if "document_id" in arguments and type(arguments["document_id"]) is not int:
        raise PdmArgumentError("document_id must be an integer.")
    for key in ("filename", "part_number", "configuration"):
        if key in arguments and (not isinstance(arguments[key], str) or not 1 <= len(arguments[key]) <= 260):
            raise PdmArgumentError(f"{key} must contain 1-260 characters.")


def _offset_argument(arguments: dict) -> int:
    offset = arguments.get("offset", 0)
    if type(offset) is not int or not 0 <= offset <= MAX_LOOKUP_OFFSET:
        raise PdmArgumentError("offset must be an integer from 0 to 1000.")
    return offset


def _detail_argument(arguments: dict) -> str:
    detail = arguments.get("detail", "summary")
    if detail not in PDM_DETAILS:
        raise PdmArgumentError("detail must be summary or full.")
    return detail


def _fit_to_budget(result: dict) -> dict:
    """Drop trailing items until the JSON result fits the byte budget, marking it truncated."""
    while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_RESULT_BYTES and result["items"]:
        result["items"] = result["items"][:-1]
        result["truncated"] = True
    return result


async def _index_result(link: PdmLink, items: list[dict]) -> dict:
    from ragtime.indexer.pdm_service import get_pdm_index_summary  # inline-import: keep

    summary = await get_pdm_index_summary(link.index_name)
    return {"items": items, "index": {"tool": link.tool_name, **summary}, "note": _NOTE}


async def execute_index_operation(config: dict, link: PdmLink, operation: str, arguments: dict) -> dict:
    from ragtime.indexer.pdm_service import lookup_pdm_document_page, search_pdm_documents  # inline-import: keep

    allowed = _SEARCH_ARGUMENTS if operation == "pdm.search" else _LOOKUP_ARGUMENTS
    if operation not in PDM_INDEX_OPERATIONS or not isinstance(arguments, dict) or arguments.keys() - allowed:
        raise PdmArgumentError("Invalid arguments.")
    limit = _limit_argument(arguments)
    document_type = _document_type_argument(arguments)
    if operation == "pdm.search":
        found = await search_pdm_documents(_search_query(arguments), link.index_name, document_type, limit)
        result = _fit_to_budget(await _index_result(link, found.items))
        if found.semantic_unavailable:
            result["note"] = SEMANTIC_UNAVAILABLE_NOTE
        return result

    _validate_lookup_arguments(arguments)
    offset = _offset_argument(arguments)
    page = await lookup_pdm_document_page(
        link.index_name,
        arguments.get("document_id"),
        arguments.get("filename"),
        arguments.get("part_number"),
        limit=limit,
        offset=offset,
        document_type=document_type,
        configuration=arguments.get("configuration"),
        detail=_detail_argument(arguments),
    )
    result = _fit_to_budget({**await _index_result(link, page.items), "next_offset": None})
    kept = len(result["items"])
    if result.get("truncated"):
        # Rows after the last kept item are read again next page; a lone over-budget item is skipped past.
        result["next_offset"] = offset + (page.positions[kept - 1] if kept else page.positions[0])
    elif page.has_more:
        # Every examined row was returned or is unparseable, so all of them count as consumed.
        result["next_offset"] = offset + page.examined
    if page.items and not kept:
        result["note"] = TOO_LARGE_NOTE
    return result
