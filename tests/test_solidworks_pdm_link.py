import asyncio
import json
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ragtime.indexer import pdm_service
from ragtime.indexer.repository import repository
from ragtime.solidworks.pdm_link import (
    PdmArgumentError,
    PdmLink,
    PdmLinkError,
    build_pdm_context,
    execute_index_operation,
    load_stored_pdm_settings,
    normalize_vault_folder,
    validate_approved_folders,
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
    ],
)
@run_async
async def test_index_invalid_arguments(operation, args):
    with pytest.raises(PdmArgumentError):
        await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), operation, args)


@run_async
async def test_index_results_and_narrowing(monkeypatch):
    search = AsyncMock(return_value=[{"filename": "A"}])
    lookup = AsyncMock(return_value=[{"document_values": {"x": "y"}, "configurations": [{"name": "Default"}, {"name": "@"}, {"name": "Other"}]}])
    monkeypatch.setattr(pdm_service, "search_pdm_documents", search)
    monkeypatch.setattr(pdm_service, "lookup_pdm_document_states", lookup)
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 1}))
    link = PdmLink("id", "tool", "index", "vault")
    result = await execute_index_operation({}, link, "pdm.search", {"query": " bolt ", "document_type": "sldprt"})
    search.assert_awaited_once_with("bolt", "index", "SLDPRT", 10)
    assert result["index"] == {"tool": "tool", "last_indexed_at": None, "document_count": 1}
    assert result["note"] == "Indexed snapshot of checked-in vault data; it can lag the vault. Use pdm.file.info for live state."
    result = await execute_index_operation({}, link, "pdm.lookup", {"filename": "A", "configuration": "default"})
    assert result["items"][0]["configurations"] == [{"name": "Default"}]
    result = await execute_index_operation({}, link, "pdm.lookup", {"document_id": 1, "configuration": "@"})
    assert result["items"][0]["configurations"] == []
    assert result["items"][0]["document_values"] == {"x": "y"}


@run_async
async def test_index_truncates_utf8_bytes(monkeypatch):
    monkeypatch.setattr(pdm_service, "search_pdm_documents", AsyncMock(return_value=[{"text": "é" * 10000}] * 3))
    monkeypatch.setattr(pdm_service, "get_pdm_index_summary", AsyncMock(return_value={"last_indexed_at": None, "document_count": 3}))
    result = await execute_index_operation({}, PdmLink("id", "tool", "index", "vault"), "pdm.search", {"query": "x"})
    assert result["truncated"] is True
    assert len(result["items"]) == 1
    assert len(json.dumps(result, separators=(",", ":"), ensure_ascii=False).encode()) <= 32768
