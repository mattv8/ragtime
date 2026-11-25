from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from ragtime.core.security import require_admin
from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.solidworks import routes
from ragtime.solidworks.host_artifact import HostArtifact
from ragtime.solidworks.host_installer import HostArtifactUnavailableError
from ragtime.solidworks.host_status import empty_host_status


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_admin] = lambda: object()
    return TestClient(app)


def test_all_routes_require_admin():
    api_routes = [route for route in routes.router.routes if isinstance(route, APIRoute)]
    assert len(routes.router.routes) == len(api_routes) == 7
    assert all(any(dep.call is require_admin for dep in route.dependant.dependencies) for route in api_routes)


def test_absent_artifact(client, monkeypatch):
    monkeypatch.setattr(routes, "get_host_artifact", lambda: None)
    assert client.get("/indexes/solidworks/host/artifact").json() == {
        "available": False,
        "host_version": None,
        "protocol": None,
        "sha256": None,
        "size_bytes": None,
    }
    assert client.get("/indexes/solidworks/host/download").status_code == 503


def test_artifact_metadata_and_download(client, monkeypatch, tmp_path):
    path = tmp_path / "host.zip"
    path.write_bytes(b"zip")
    artifact = HostArtifact("1.0", 1, "host.zip", "a" * 64, 3, path)
    monkeypatch.setattr(routes, "get_host_artifact", lambda: artifact)
    assert client.get("/indexes/solidworks/host/artifact").json() == {
        "available": True,
        "host_version": "1.0",
        "protocol": 1,
        "sha256": "a" * 64,
        "size_bytes": 3,
    }
    monkeypatch.setattr(routes, "verify_host_artifact", lambda value: True)
    response = client.get("/indexes/solidworks/host/download")
    assert response.content == b"zip"
    assert response.headers["content-type"] == "application/zip"


def test_download_refused_when_hash_does_not_match(client, monkeypatch, tmp_path):
    path = tmp_path / "host.zip"
    path.write_bytes(b"tampered")
    artifact = HostArtifact("1.0", 1, "host.zip", "a" * 64, 8, path)
    monkeypatch.setattr(routes, "get_host_artifact", lambda: artifact)
    monkeypatch.setattr(routes, "verify_host_artifact", lambda value: False)
    assert client.get("/indexes/solidworks/host/download").status_code == 503


@pytest.mark.parametrize("tool", [None, ToolConfig(name="other", tool_type=ToolType.SSH_SHELL, connection_config={})])
def test_missing_or_wrong_tool(client, monkeypatch, tool):
    monkeypatch.setattr(routes.repository, "get_tool_config", AsyncMock(return_value=tool))
    assert client.get("/indexes/solidworks/tools/id/host").status_code == 404
    assert client.post("/indexes/solidworks/tools/id/host/install").status_code == 404
    assert client.post("/indexes/solidworks/tools/id/host/uninstall").status_code == 404
    assert client.post("/indexes/solidworks/tools/id/pdm/rights", json={"pdm_tool_id": "pdm", "folders": []}).status_code == 404
    assert client.post("/indexes/solidworks/tools/id/pdm/folders", json={"pdm_tool_id": "pdm"}).status_code == 404


@pytest.mark.parametrize(
    "method,path,service_method", [("get", "", "host_status"), ("post", "/install", "install_host"), ("post", "/uninstall", "uninstall_host")]
)
def test_forwarding(client, monkeypatch, method, path, service_method):
    tool = ToolConfig(id="id", name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={"password": "secret"})
    monkeypatch.setattr(routes.repository, "get_tool_config", AsyncMock(return_value=tool))
    call = AsyncMock(return_value={**empty_host_status(), "error": {"code": "host_not_running", "message": "not running"}})
    monkeypatch.setattr(routes.solidworks_service, service_method, call)
    response = getattr(client, method)(f"/indexes/solidworks/tools/id/host{path}")
    assert response.status_code == 200
    assert "secret" not in response.text
    assert response.json()["warning"] is None
    call.assert_awaited_once_with(tool.model_dump())


def test_install_missing_artifact(client, monkeypatch):
    tool = ToolConfig(name="sw", tool_type=ToolType.SOLIDWORKS, connection_config={})
    monkeypatch.setattr(routes.repository, "get_tool_config", AsyncMock(return_value=tool))
    monkeypatch.setattr(routes.solidworks_service, "install_host", AsyncMock(side_effect=HostArtifactUnavailableError()))
    assert client.post("/indexes/solidworks/tools/id/host/install").status_code == 503
