import asyncio
import json
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ragtime.indexer import pdm_service
from ragtime.indexer.repository import repository
from ragtime.solidworks.pdm_link import (
    SEMANTIC_UNAVAILABLE_NOTE,
    TOO_LARGE_NOTE,
    PdmArgumentError,
    PdmLink,
    PdmLinkError,
    build_pdm_context,
    execute_index_operation,
    load_stored_pdm_settings,
    normalize_vault_folder,
    pdm_context_tool_id,
    validate_approved_folders,
    vault_linked,
)


def run_async(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@run_async
async def test_load_stored_pdm_settings():
    record = SimpleNamespace(allow_write=True, enabled=True, connection_config={"pdm_tool_id": "pdm", "pdm_approved_folders": ["\\approved"]})
    with patch.object(repository, "get_tool_config", AsyncMock(return_value=record)) as get:
        assert await load_stored_pdm_settings({}) is None
        get.assert_not_awaited()
        assert await load_stored_pdm_settings({"id": "id"}) == {
            "allow_write": True,
            "enabled": True,
            "pdm_tool_id": "pdm",
            "pdm_approved_folders": ["\\approved"],
        }
        get.return_value = None
        with pytest.raises(PdmLinkError, match="no_link"):
            await load_stored_pdm_settings({"id": "id"})


@pytest.mark.parametrize("value,expected", [("\\Folder\\Sub", "\\Folder\\Sub"), ("/Folder/Sub", "\\Folder\\Sub"), ("\\", "\\")])
def test_normalize_vault_folder(value, expected):
    assert normalize_vault_folder(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "C:\\Folder", "\\\\server\\share", "\\a\\..", "\\a\\.", "\\a\\", "\\a\\ b", "\\a ", "\\a.", "\\a*", "\\a:b", "\\" + "a" * 260],
)
def test_invalid_folder(value):
    with pytest.raises(ValueError):
        normalize_vault_folder(value)


@pytest.mark.parametrize("value", ["\\café", "\\\u0080", "\\😀"])
def test_approved_folders_require_ascii(value):
    with pytest.raises(ValueError, match="^Use plain ASCII characters in approved folder paths\\.$"):
        validate_approved_folders([value])


def test_control_character_error_takes_precedence_over_ascii():
    with pytest.raises(ValueError, match="control characters"):
        normalize_vault_folder("\\\x01é")


@pytest.mark.parametrize("folders", [["\\"], ["\\a", "\\A"], ["\\a", "\\A\\b"], [f"\\{i}" for i in range(26)]])
def test_invalid_approved_folders(folders):
    with pytest.raises(ValueError):
        validate_approved_folders(folders)


def test_context_sanitizes_prefix():
    link = PdmLink("id", "name", "index", "vault")
    context = build_pdm_context({"name": "x[;]\n" + "a" * 80, "connection_config": {"pdm_approved_folders": ["\\A"]}}, link, "u;[]\n")
    assert context["approved_folders"] == ["\\A"]
    assert context["comment_prefix"] == f"[Ragtime: {'x' + 'a' * 59}; user: u] "
    assert len(context["comment_prefix"]) <= 160


def test_context_prefix_limits_utf16_units():
    link = PdmLink("id", "name", "index", "vault")
    prefix = build_pdm_context({"name": "😀" * 60}, link, "😀" * 60)["comment_prefix"]
    assert len(prefix.encode("utf-16-le")) // 2 <= 160
    assert prefix == f"[Ragtime: {'😀' * 30}; user: {'😀' * 30}] "


@pytest.mark.parametrize(
    "tool_id,expected",
    [(None, "unsaved"), ("", "unsaved"), ("bad id", "unsaved"), ("a" * 65, "unsaved"), (7, "unsaved"), ("a_Z-9", "a_Z-9"), ("x" * 64, "x" * 64)],
)
def test_context_tool_id(tool_id, expected):
    link = PdmLink("pdm-id", "name", "index", "vault")
    assert pdm_context_tool_id({"id": tool_id}) == expected
    assert build_pdm_context({"id": tool_id}, link, "user")["tool_id"] == expected


@pytest.mark.parametrize(
    "operation,args",
    [
        ("pdm.search", {}),
        ("pdm.search", {"query": "  "}),
        ("pdm.search", {"query": "x" * 501}),
        ("pdm.search", {"query": 1}),
        ("pdm.search", {"query": "x", "document_type": None}),
        ("pdm.search", {"query": "x", "document_type": "bad"}),
        ("pdm.search", {"query": "x", "limit": True}),
        ("pdm.search", {"query": "x", "limit": 26}),
        ("pdm.search", {"query": "x", "extra": 1}),
        ("pdm.lookup", {}),
        ("pdm.lookup", {"document_id": True}),
        ("pdm.lookup", {"filename": ""}),
        ("pdm.lookup", {"part_number": "x" * 261}),
        ("pdm.lookup", {"filename": "x", "configuration": 1}),
        ("pdm.lookup", {"document_id": 1, "limit": 0}),
        ("pdm.lookup", {"document_id": 1, "extra": 1}),
        ("pdm.lookup", {"document_id": 1, "offset": 1001}),
        ("pdm.lookup", {"document_id": 1, "offset": -1}),
        ("pdm.lookup", {"document_id": 1, "offset": True}),
        ("pdm.lookup", {"document_id": 1, "detail": "all"}),
        ("pdm.lookup", {"document_id": 1, "document_type": "PDF"}),
        ("pdm.lookup", {"document_id": 1, "document_type": None}),
        ("pdm.search", {"query": "x", "offset": 1}),
    ],
)
@run_async
async def test_index_invalid_arguments(operation, args):
    with pytest.raises(PdmArgumentError):
        await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), operation, args)


@run_async
async def test_index_search_passes_document_type_and_returns_no_paging(monkeypatch):
    search = AsyncMock(return_value=pdm_service.SearchResult([{"filename": "A"}], False))
    monkeypatch.setattr(pdm_service, "search_pdm_documents", search)
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    link = PdmLink("id", "tool", "index", "vault")
    result = await execute_index_operation({}, link, "pdm.search", {"query": " bolt ", "document_type": "sldprt"})
    search.assert_awaited_once_with("bolt", "index", "SLDPRT", 10)
    assert result["items"] == [{"filename": "A"}]
    assert result["index"] == {"tool": "tool", "last_indexed_at": None, "document_count": 1}
    assert result["note"] == "Indexed snapshot of checked-in vault data; it can lag the vault. Use pdm.file.info for live state."
    assert "next_offset" not in result


@run_async
async def test_index_lookup_passes_paging_filters_and_configuration(monkeypatch):
    page = AsyncMock(return_value=pdm_service.LookupPage([{"matched_on": "filename"}], [1], True, 1))
    monkeypatch.setattr(pdm_service, "lookup_pdm_document_page", page)
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    link = PdmLink("id", "tool", "index", "vault")
    result = await execute_index_operation(
        {}, link, "pdm.lookup", {"filename": "A", "configuration": "default", "document_type": "sldasm", "offset": 25, "detail": "full", "limit": 5}
    )
    page.assert_awaited_once_with("index", None, "A", None, limit=5, offset=25, document_type="SLDASM", configuration="default", detail="full")
    assert result["items"] == [{"matched_on": "filename"}]
    assert result["next_offset"] == 26


@run_async
async def test_index_lookup_next_offset_is_null_on_last_page(monkeypatch):
    monkeypatch.setattr(pdm_service, "lookup_pdm_document_page", AsyncMock(return_value=pdm_service.LookupPage([{"document_id": 1}], [1], False, 1)))
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.lookup", {"document_id": 1})
    assert result["next_offset"] is None
    assert "truncated" not in result


@run_async
async def test_index_truncates_utf8_bytes(monkeypatch):
    monkeypatch.setattr(pdm_service, "search_pdm_documents", AsyncMock(return_value=pdm_service.SearchResult([{"text": "é" * 10000}] * 3, False)))
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 3}))
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.search", {"query": "x"})
    assert result["truncated"] is True
    assert len(result["items"]) == 1
    assert len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()) <= 32768


@run_async
async def test_index_lookup_trim_keeps_paging_past_trimmed_rows(monkeypatch):
    rows = [{"text": "é" * 10000}] * 3
    monkeypatch.setattr(pdm_service, "lookup_pdm_document_page", AsyncMock(return_value=pdm_service.LookupPage(rows, [1, 2, 3], False, 3)))
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 3}))
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.lookup", {"filename": "A", "offset": 10})
    assert result["truncated"] is True
    assert len(result["items"]) == 1
    assert result["next_offset"] == 11
    assert len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()) <= 32768


def _good_state(document_id: int, **extra: object) -> dict:
    return {
        "document_id": document_id,
        "filename": f"A{document_id}.SLDPRT",
        "document_type": "SLDPRT",
        "target_revision": 1,
        "folder_paths": [],
        "configurations": [],
        "bom_components": [],
        "warnings": [],
        **extra,
    }


def _good_row(document_id: int, **extra: object) -> dict:
    return {"document_id": document_id, "state_json": _good_state(document_id, **extra), "extracted_at": None}


BAD_ROW = {"document_id": 0, "state_json": "{not json", "extracted_at": None}


@run_async
async def test_lookup_paging_past_unparseable_rows_neither_repeats_nor_stalls(monkeypatch):
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 4}))
    link = PdmLink("id", "tool", "index", "vault")
    monkeypatch.setattr(
        pdm_service, "get_db", AsyncMock(return_value=SimpleNamespace(query_raw=AsyncMock(return_value=[_good_row(1), BAD_ROW, _good_row(3), _good_row(4)])))
    )
    result = await execute_index_operation({}, link, "pdm.lookup", {"filename": "A", "limit": 3})
    assert [item["document_id"] for item in result["items"]] == [1, 3]
    assert result["next_offset"] == 3  # row 4 is next; row 3 is not repeated

    monkeypatch.setattr(pdm_service, "get_db", AsyncMock(return_value=SimpleNamespace(query_raw=AsyncMock(return_value=[BAD_ROW] * 4))))
    result = await execute_index_operation({}, link, "pdm.lookup", {"filename": "A", "limit": 3, "offset": 6})
    assert result["items"] == []
    assert result["next_offset"] == 9


@run_async
async def test_full_item_over_budget_returns_guidance_and_advances(monkeypatch):
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    monkeypatch.setattr(
        pdm_service, "get_db", AsyncMock(return_value=SimpleNamespace(query_raw=AsyncMock(return_value=[_good_row(1, warnings=["w" * 40000])])))
    )
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.lookup", {"document_id": 1, "detail": "full", "offset": 5})
    assert result["items"] == []
    assert result["truncated"] is True
    assert result["next_offset"] == 6
    assert result["note"] == TOO_LARGE_NOTE == "One item is too large for detail=full; use detail=summary or document_id."


@run_async
async def test_unavailable_semantic_search_is_noted_and_keeps_filename_hits(monkeypatch):
    monkeypatch.setattr(pdm_service, "search_pdm_documents", AsyncMock(return_value=pdm_service.SearchResult([{"document_id": 7, "match": "filename"}], True)))
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.search", {"query": "IDB0061"})
    assert result["note"] == SEMANTIC_UNAVAILABLE_NOTE
    assert result["items"] == [{"document_id": 7, "match": "filename"}]


@run_async
async def test_vault_linked_requires_an_enabled_solidworks_pdm_tool():
    assert await vault_linked({}) is False
    with patch.object(repository, "get_tool_config", AsyncMock()) as get:
        assert await vault_linked({"pdm_tool_id": None}) is False
        get.assert_not_awaited()
    cases = [
        (None, False),
        (SimpleNamespace(tool_type="solidworks_pdm", enabled=True), True),
        (SimpleNamespace(tool_type="solidworks_pdm", enabled=False), False),
        (SimpleNamespace(tool_type="other", enabled=True), False),
    ]
    for record, expected in cases:
        with patch.object(repository, "get_tool_config", AsyncMock(return_value=record)):
            assert await vault_linked({"pdm_tool_id": "pdm"}) is expected
    with patch.object(repository, "get_tool_config", AsyncMock(side_effect=RuntimeError("down"))):
        assert await vault_linked({"pdm_tool_id": "pdm"}) is False
