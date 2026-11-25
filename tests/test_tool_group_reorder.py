import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from ragtime.core.security import require_admin
from ragtime.indexer import routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_admin] = lambda: object()
    return TestClient(app)


def test_reorder_route_requires_admin():
    route = next(
        route
        for route in routes.router.routes
        if isinstance(route, APIRoute) and route.path == "/indexes/tool-groups/reorder" and "POST" in (route.methods or set())
    )
    assert any(dep.call is require_admin for dep in route.dependant.dependencies)


def test_reorder_route_passes_order_to_repository(client, monkeypatch):
    reorder = AsyncMock()
    monkeypatch.setattr(routes.repository, "reorder_tool_groups", reorder, raising=False)
    response = client.post("/indexes/tool-groups/reorder", json={"group_ids": ["b", "a"]})
    assert response.status_code == 200
    assert response.json() == {"message": "Tool groups reordered"}
    reorder.assert_awaited_once_with(["b", "a"])


def test_reorder_route_reports_invalid_order(client, monkeypatch):
    monkeypatch.setattr(routes.repository, "reorder_tool_groups", AsyncMock(side_effect=ValueError("invalid order")), raising=False)
    response = client.post("/indexes/tool-groups/reorder", json={"group_ids": ["a"]})
    assert response.status_code == 400
    assert response.json() == {"detail": "invalid order"}


@pytest.fixture
def db(monkeypatch):
    db = SimpleNamespace(
        toolgroup=SimpleNamespace(find_many=AsyncMock(return_value=[SimpleNamespace(id=id) for id in ["a", "b", "c"]])),
        execute_raw=AsyncMock(),
    )
    monkeypatch.setattr(routes.repository, "_get_db", AsyncMock(return_value=db))
    return db


@pytest.mark.parametrize(
    ("ids", "error"),
    [
        (["a", "a", "b", "c"], "unique"),
        (["a", "b", "c", "unknown"], "unknown"),
        (["a", "b"], "c"),
        ([], "a"),
    ],
)
def test_repository_rejects_invalid_ids(db, ids, error):
    with pytest.raises(ValueError, match=f"(?i){error}"):
        asyncio.run(routes.repository.reorder_tool_groups(ids))
    db.execute_raw.assert_not_awaited()


def test_repository_reorders_groups_in_one_update(db):
    asyncio.run(routes.repository.reorder_tool_groups(["c", "a", "b"]))
    db.execute_raw.assert_awaited_once()
    sql = db.execute_raw.await_args.args[0]
    assert "UPDATE tool_groups" in sql
    payload = sql.split("jsonb_to_recordset('", 1)[1].split("'::jsonb)", 1)[0].replace("''", "'")
    assert json.loads(payload) == [
        {"id": "c", "sort_order": 0},
        {"id": "a", "sort_order": 100},
        {"id": "b", "sort_order": 200},
    ]
