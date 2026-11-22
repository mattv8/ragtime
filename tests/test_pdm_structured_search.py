import asyncio
from datetime import datetime, timezone
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragtime.indexer import pdm_service as service


def run_async(test):
    @wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapper


@pytest.fixture
def state():
    return {
        "document_id": 7,
        "filename": "A.SLDPRT",
        "document_type": "SLDPRT",
        "target_revision": 3,
        "part_number": "P7",
        "description": "Part",
        "folder_paths": ["\\Z\\", "/a//", "\\A\\"],
        "document_values": {"X": {"variable_id": 1, "variable_name": "X", "value_text": "old", "is_blank": True, "origin_revision": 3}},
        "configurations": [{"configuration_id": 2, "name": "Default", "values": {}}],
        "bom_components": [{"document_id": 8, "filename": "B.SLDPRT", "configuration": "", "quantity": None}],
        "warnings": ["warning"],
    }


@run_async
async def test_search_shape_and_fallback(monkeypatch, state):
    rows = [
        {"document_id": 7, "filename": "A.SLDPRT", "similarity": 0.8},
        {"document_id": 9, "filename": "B.SLDASM", "folder_path": "\\Folder\\", "document_type": "SLDASM"},
    ]
    monkeypatch.setattr(service, "_search_pdm_rows", AsyncMock(return_value=rows))
    db = type("DB", (), {"query_raw": AsyncMock(return_value=[{"document_id": 7, "state_json": state}])})()
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    items = await service.search_pdm_documents("query", "pdm_test")
    assert items[0] == {
        "document_id": 7,
        "filename": "A.SLDPRT",
        "document_type": "SLDPRT",
        "part_number": "P7",
        "description": "Part",
        "revision": 3,
        "vault_paths": ["\\a\\A.SLDPRT", "\\Z\\A.SLDPRT"],
        "similarity": 0.8,
        "configurations": ["Default"],
        "bom_count": 1,
    }
    assert items[1]["vault_paths"] == ["\\Folder\\B.SLDASM"]
    assert items[1]["description"] is None and items[1]["configurations"] == []


@run_async
async def test_lookup_and_text_output(monkeypatch, state):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[{"state_json": state, "extracted_at": when}]))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    items = await service.lookup_pdm_document_states("index", filename="a_%", part_number="x\\y")
    assert db.query_raw.call_args.args[1:] == ("index", "%a\\_\\%%", "%x\\\\y%")
    assert items[0]["document_values"] == {"X": ""}
    assert items[0]["configurations"] == [{"name": "Default", "values": {}}]
    assert items[0]["bom"] == [{"document_id": 8, "filename": "B.SLDPRT", "configuration": "", "quantity": None}]
    assert items[0]["extracted_at"] == when.isoformat()
    assert await service.lookup_pdm_documents("index", document_id=7) == (
        "[A.SLDPRT] (PN: P7) [SLDPRT] target revision 3\n"
        "Source: indexed snapshot (checked-in view)\nExtracted at: 2026-01-01 00:00:00+00:00\n"
        "Folders: /a//, \\A\\, \\Z\\\nWarnings: warning\n@ document values:\n"
        "- X: (empty, cleared at v3)\nConfiguration: Default (ID: 2)"
    )


@run_async
async def test_filter_and_search_text(monkeypatch):
    with pytest.raises(ValueError):
        await service.search_pdm_documents("q", "index", "SLDPRT'; DROP TABLE x")
    monkeypatch.setattr(
        service,
        "_search_pdm_rows",
        AsyncMock(return_value=[{"filename": "a", "part_number": "p", "document_type": "SLDPRT", "similarity": 0.5, "content": "body"}]),
    )
    assert await service.search_pdm_index("q", "index") == "[a] (PN: p) [SLDPRT] (similarity: 0.500)\nbody"


@run_async
async def test_summary_fallback_and_latest(monkeypatch):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db = type(
        "DB",
        (),
        {
            "query_raw": AsyncMock(
                side_effect=[
                    [{"count": 0, "newest": None}],
                    [{"count": 5, "newest": when}],
                ]
            )
        },
    )()
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    assert await service.get_pdm_index_summary("index") == {"document_count": 5, "last_indexed_at": when.isoformat()}
