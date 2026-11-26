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


async def _page(*args, **kwargs):
    page = await service.lookup_pdm_document_page(*args, **kwargs)
    return page.items, page.has_more


async def _search(*args, **kwargs):
    return (await service.search_pdm_documents(*args, **kwargs)).items


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
    items = await _search("query", "pdm_test")
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
        "match": "semantic",
    }
    assert items[1]["vault_paths"] == ["\\Folder\\B.SLDASM"]
    assert items[1]["description"] is None and items[1]["configurations"] == []


@run_async
async def test_lookup_and_text_output(monkeypatch, state):
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[{"state_json": state, "extracted_at": when}]))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    items, more = await _page("index", filename="a_%", part_number="x\\y", detail="full")
    assert more is False
    assert db.query_raw.call_args.args[1:4] == ("index", "%a\\_\\%%", "%x\\\\y%")
    assert items[0]["matched_on"] == "properties"
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
        await _search("q", "index", "SLDPRT'; DROP TABLE x")
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


WHEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
ORDER_BY = (
    "ORDER BY CASE WHEN lower(filename) = $3 OR lower(filename) LIKE $4 ESCAPE '\\' THEN 0 "
    "WHEN lower(filename) LIKE $5 ESCAPE '\\' THEN 1 "
    "WHEN lower(filename) LIKE '%.sldasm' OR lower(filename) LIKE '%.sldprt' OR lower(filename) LIKE '%.slddrw' THEN 2 "
    "ELSE 3 END, CASE WHEN lower(filename) LIKE '%.sldasm' OR lower(filename) LIKE '%.sldprt' OR lower(filename) LIKE '%.slddrw' "
    "THEN 0 ELSE 1 END, filename, document_id"
)


def _state_json(document_id: int = 7, filename: str = "A.SLDPRT", **extra: object) -> dict:
    return {
        "document_id": document_id,
        "filename": filename,
        "document_type": filename.rpartition(".")[2].upper(),
        "target_revision": 3,
        "part_number": "P7",
        "description": "Part",
        "folder_paths": ["\\Folder\\"],
        "configurations": [{"configuration_id": 2, "name": "Default", "values": {}}],
        "bom_components": [],
        "warnings": [],
        **extra,
    }


def _row(document_id: int, filename: str, **extra: object) -> dict:
    return {"document_id": document_id, "state_json": _state_json(document_id, filename, **extra), "extracted_at": WHEN}


def _db(rows: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(query_raw=AsyncMock(return_value=rows))


@run_async
async def test_lookup_orders_exact_then_prefix_then_cad_then_rest(monkeypatch):
    db = _db([])
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    await _page("index", filename="IDB0061", limit=5)
    query, *params = db.query_raw.call_args.args
    assert ORDER_BY in query
    assert " LIMIT 6 OFFSET 0" in query
    assert params == ["index", "%IDB0061%", "idb0061", "idb0061.%", "idb0061%"]


@run_async
async def test_lookup_without_filename_orders_cad_first_and_pages_with_offset(monkeypatch):
    db = _db([])
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    await _page("index", part_number="22-100", limit=25, offset=50)
    query, *params = db.query_raw.call_args.args
    assert "ELSE 3 END, CASE WHEN" in query and "THEN 0 ELSE 1 END, filename, document_id LIMIT 26 OFFSET 50" in query
    assert "WHEN lower(filename) = " not in query
    assert params == ["index", "%22-100%"]


@run_async
async def test_page_fetches_one_extra_row_and_reports_more(monkeypatch):
    rows = [_row(i, f"P{i}.SLDPRT") for i in range(1, 27)]
    db = _db(rows)
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    items, more = await _page("index", filename="P", limit=25, offset=50)
    assert len(items) == 25 and more is True
    db.query_raw.return_value = rows[:3]
    items, more = await _page("index", filename="P", limit=25, offset=50)
    assert len(items) == 3 and more is False


@run_async
async def test_document_type_filters_by_extension_parameter(monkeypatch):
    db = _db([])
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    await _page("index", part_number="P", document_type="SLDASM")
    query, *params = db.query_raw.call_args.args
    assert "AND lower(filename) LIKE $3 " in query
    assert params == ["index", "%P%", "%.sldasm"]


@run_async
async def test_matched_on_precedence(monkeypatch):
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=_db([_row(7, "CU-IDB0061.SLDASM", part_number="22-100-1")])))
    cases = [
        ({"filename": "idb0061"}, "filename"),
        ({"part_number": "22-100"}, "part_number"),
        ({"filename": "zz", "part_number": "22-100"}, "part_number"),
        ({"part_number": "nope"}, "properties"),
        ({"document_id": 7}, "document_id"),
    ]
    for arguments, expected in cases:
        items, _ = await _page("index", **arguments)
        assert items[0]["matched_on"] == expected, arguments


@run_async
async def test_summary_item_shape_and_configuration_cap(monkeypatch):
    configurations = [{"configuration_id": i, "name": f"C{i:02d}", "values": {}} for i in range(60)]
    bom = [{"document_id": 8, "filename": "B.SLDPRT", "configuration": "", "quantity": None}]
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=_db([_row(7, "A.SLDPRT", configurations=configurations, bom_components=bom)])))
    items, _ = await _page("index", document_id=7)
    assert set(items[0]) == {
        "document_id",
        "filename",
        "document_type",
        "part_number",
        "description",
        "revision",
        "vault_paths",
        "extracted_at",
        "configuration_names",
        "configuration_count",
        "bom_count",
        "matched_on",
    }
    assert items[0]["configuration_names"] == [f"C{i:02d}" for i in range(50)]
    assert items[0]["configuration_count"] == 60
    assert items[0]["bom_count"] == 1
    assert items[0]["extracted_at"] == WHEN.isoformat()


@run_async
async def test_configuration_narrows_summary_and_full_items(monkeypatch):
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=_db([_row(7, "A.SLDPRT")])))
    items, _ = await _page("index", document_id=7, configuration="default")
    assert items[0]["configuration_names"] == ["Default"]
    items, _ = await _page("index", document_id=7, configuration="@", detail="full")
    assert items[0]["configurations"] == []
    assert items[0]["bom"] == []


@run_async
async def test_identifier_query_prepends_filename_prefix_hits(monkeypatch):
    lookup_rows = [_row(7, "IDB0061-19.SLDASM"), _row(8, "CU-IDB0061-19.SLDPRT")]
    state_rows = [{"document_id": 9, "state_json": _state_json(9, "X.SLDPRT")}]

    def query_raw(query: str, *params: object) -> list[dict]:
        return lookup_rows if "ORDER BY CASE" in query else state_rows

    db = SimpleNamespace(query_raw=AsyncMock(side_effect=query_raw))
    semantic = [{"document_id": 7, "similarity": 0.9}, {"document_id": 9, "similarity": 0.5}]
    monkeypatch.setattr(service, "_search_pdm_rows", AsyncMock(return_value=semantic))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    items = await _search("IDB0061-19", "index", None, 10)
    assert [(item["document_id"], item["match"]) for item in items] == [(7, "filename"), (9, "semantic")]


@run_async
async def test_plain_word_query_skips_filename_boost(monkeypatch):
    db = _db([])
    monkeypatch.setattr(service, "_search_pdm_rows", AsyncMock(return_value=[]))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    assert await _search("gear", "index") == []
    db.query_raw.assert_not_awaited()


@run_async
async def test_cad_files_rank_first_within_the_exact_tier(monkeypatch):
    db = _db([])
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    await _page("index", filename="IDB0061")
    query = db.query_raw.call_args.args[0]
    assert "ELSE 3 END, CASE WHEN lower(filename) LIKE '%.sldasm'" in query
    assert query.index("CASE WHEN lower(filename) LIKE '%.sldasm'") < query.index(", filename, document_id")


@run_async
async def test_configuration_count_stays_the_document_total(monkeypatch):
    configurations = [{"configuration_id": i, "name": f"C{i:02d}", "values": {}} for i in range(60)]
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=_db([_row(7, "A.SLDPRT", configurations=configurations)])))
    items, _ = await _page("index", document_id=7, configuration="C05")
    assert items[0]["configuration_names"] == ["C05"]
    assert items[0]["configuration_count"] == 60
    items, _ = await _page("index", document_id=7, configuration="@")
    assert items[0]["configuration_names"] == []
    assert items[0]["configuration_count"] == 60


@run_async
async def test_identifier_hits_survive_unavailable_embeddings(monkeypatch):
    db = _db([_row(7, "IDB0061-19.SLDASM")])
    monkeypatch.setattr(service, "_search_pdm_rows", AsyncMock(side_effect=service._PdmConfigurationError("no embeddings")))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=db))
    result = await service.search_pdm_documents("IDB0061-19", "index")
    assert result.semantic_unavailable is True
    assert [item["match"] for item in result.items] == ["filename"]


@run_async
async def test_embedding_failure_without_identifier_hits_still_raises(monkeypatch):
    monkeypatch.setattr(service, "_search_pdm_rows", AsyncMock(side_effect=service._PdmConfigurationError("no embeddings")))
    monkeypatch.setattr(service, "get_db", AsyncMock(return_value=_db([])))
    with pytest.raises(service._PdmConfigurationError):
        await service.search_pdm_documents("gear", "index")
