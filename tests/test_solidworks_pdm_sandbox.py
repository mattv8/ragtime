import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ragtime.indexer.models import ToolType
from ragtime.solidworks import pdm_sandbox as sandbox
from ragtime.solidworks.routes import PdmSandboxRequest


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.created = True
        self.error = None
        self.create_error = None
        self.rights = dict.fromkeys(("read", "add_folders", "add_files", "check_out"), True)

    async def call(self, request, timeout_seconds):
        self.requests.append(request)
        operation = request["operation"]
        response = {"id": request["id"], "operation": operation, "ok": True}
        if self.error:
            return {**response, "ok": False, "error": self.error}
        if operation == "pdm.folder.create" and self.create_error:
            return {**response, "ok": False, "error": self.create_error}
        if operation == "pdm.folder.rights":
            return {
                **response,
                "result": {
                    "user": "tester",
                    "items": [{"vault_path": path, "checked": True, "exists": True, "rights": self.rights} for path in request["arguments"]["paths"]],
                },
            }
        return {**response, "result": {"created": self.created}}


@pytest.fixture
def setup(monkeypatch):
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    config = {"id": "sw", "name": "SW", "enabled": True, "updated_at": stamp, "connection_config": {"pdm_tool_id": "pdm", "pdm_approved_folders": []}}
    sw = SimpleNamespace(tool_type=ToolType.SOLIDWORKS, enabled=True, connection_config=config["connection_config"], model_dump=lambda: config)
    pdm = SimpleNamespace(id="pdm", name="Vault", enabled=True, updated_at=stamp, tool_type=ToolType.SOLIDWORKS_PDM, connection_config={"database": "vault"})
    monkeypatch.setattr(sandbox.repository, "get_tool_config", AsyncMock(side_effect=lambda key: sw if key == "sw" else pdm))
    transport = FakeTransport()
    monkeypatch.setattr(sandbox.pdm_rights_service, "transport_factory", lambda config: transport)
    db = SimpleNamespace(execute_raw=AsyncMock(return_value=1))
    monkeypatch.setattr(sandbox, "get_db", AsyncMock(return_value=db))
    settings = Mock()
    notify = Mock()
    monkeypatch.setattr(sandbox, "invalidate_settings_cache", settings)
    monkeypatch.setattr(sandbox, "notify_tools_changed", notify)
    return config, sw, transport, db, settings, notify


@pytest.mark.anyio
async def test_create_approves_and_invalidates(setup):
    config, _, transport, db, settings, notify = setup
    result = await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert result == {
        "folder": "\\Parent\\Sandbox",
        "created": True,
        "covered": False,
        "approved_folders": ["\\Parent\\Sandbox"],
        "allow_write": True,
        "scripts_enabled": False,
    }
    request = transport.requests[-1]
    assert request["context"]["allow_write"] is True
    assert request["context"]["pdm"]["approved_folders"] == ["\\Parent\\Sandbox"]
    assert request["context"]["pdm"]["tool_id"] == "sw"
    assert request["context"]["pdm"]["comment_prefix"] == "[Ragtime: SW; user: unknown] "
    query, new, tool_id, old, pdm_id, sw_updated, pdm_updated, database = db.execute_raw.await_args.args
    assert json.loads(new) == result["approved_folders"]
    assert (tool_id, json.loads(old), pdm_id, sw_updated, pdm_updated, database) == ("sw", [], "pdm", config["updated_at"], config["updated_at"], "vault")
    for predicate in (
        "jsonb_set(connection_config",
        "connection_config->'pdm_approved_folders' IS NOT DISTINCT FROM $3::jsonb",
        "connection_config->>'pdm_tool_id' = $4",
        "enabled = true AND updated_at = $5",
        "pdm.id = $4",
        "pdm.enabled = true AND pdm.updated_at = $6",
        "pdm.connection_config->>'database' = $7",
    ):
        assert predicate in query
    settings.assert_called_once()
    notify.assert_called_once()


@pytest.mark.anyio
async def test_existing_folder_is_never_approved(setup):
    _, _, transport, db, settings, notify = setup
    transport.created = False
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert exc.value.status_code == 409
    assert exc.value.detail == {
        "code": "sandbox_exists",
        "message": "A folder named “Sandbox” already exists in \\Parent. Choose another name.",
    }
    db.execute_raw.assert_not_awaited()
    settings.assert_not_called()
    notify.assert_not_called()


@pytest.mark.anyio
async def test_covered_folder_is_still_created(setup):
    config, _, transport, _, _, _ = setup
    config["connection_config"]["pdm_approved_folders"] = ["\\Parent"]
    result = await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert result["covered"] is True
    assert result["approved_folders"] == ["\\Parent"]
    assert transport.requests[-1]["context"]["pdm"]["approved_folders"] == ["\\Parent\\Sandbox"]


@pytest.mark.anyio
async def test_overlap_rejected_before_agent(setup):
    config, _, transport, _, _, _ = setup
    config["connection_config"]["pdm_approved_folders"] = ["\\Parent\\Sandbox\\Child"]
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert isinstance(exc.value.detail, dict)
    assert cast(Any, exc.value.detail)["code"] == "sandbox_overlap"
    assert cast(Any, exc.value.detail)["message"] == "“\\Parent\\Sandbox” overlaps the approved folder “\\Parent\\Sandbox\\Child”."
    assert not transport.requests


@pytest.mark.anyio
async def test_missing_rights_and_unsaved_link(setup):
    _, sw, transport, _, _, _ = setup
    transport.rights["add_folders"] = False
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert isinstance(exc.value.detail, dict)
    assert cast(Any, exc.value.detail)["missing"] == ["add_folders"]
    assert len(transport.requests) == 1
    sw.connection_config["pdm_tool_id"] = "other"
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert isinstance(exc.value.detail, dict)
    assert cast(Any, exc.value.detail)["code"] == "pdm_link_not_saved"


@pytest.mark.anyio
async def test_conflict_retries_without_creating_twice(setup):
    _, _, transport, db, settings, notify = setup
    db.execute_raw.side_effect = [0, 0]
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert isinstance(exc.value.detail, dict)
    assert cast(Any, exc.value.detail)["code"] == "sandbox_conflict"
    assert db.execute_raw.await_count == 2
    assert [r["operation"] for r in transport.requests].count("pdm.folder.create") == 1
    settings.assert_not_called()
    notify.assert_not_called()


@pytest.mark.anyio
async def test_conflict_retry_succeeds_without_creating_twice(setup):
    _, _, transport, db, _, _ = setup
    db.execute_raw.side_effect = [0, 1]
    result = await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert result["created"] is True
    assert db.execute_raw.await_count == 2
    assert [r["operation"] for r in transport.requests].count("pdm.folder.create") == 1


@pytest.mark.anyio
@pytest.mark.parametrize("overlap", [False, True])
async def test_retry_merges_or_rejects_new_folder(setup, overlap):
    config, _, _, db, _, _ = setup

    async def update(*args):
        if db.execute_raw.await_count == 1:
            config["connection_config"]["pdm_approved_folders"] = ["\\Parent\\Sandbox\\Child" if overlap else "\\Other"]
            return 0
        return 1

    db.execute_raw.side_effect = update
    if overlap:
        with pytest.raises(HTTPException) as exc:
            await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
        assert cast(Any, exc.value.detail)["code"] == "sandbox_overlap"
        assert db.execute_raw.await_count == 1
    else:
        result = await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
        assert result["approved_folders"] == ["\\Other", "\\Parent\\Sandbox"]
        assert json.loads(db.execute_raw.await_args.args[3]) == ["\\Other"]


@pytest.mark.anyio
async def test_changed_vault_conflicts_after_creation(setup):
    _, _, transport, db, settings, notify = setup
    pdm = cast(Any, sandbox.repository.get_tool_config).side_effect("pdm")

    async def update(*args):
        pdm.connection_config["database"] = "another"
        return 0

    db.execute_raw.side_effect = update
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert cast(Any, exc.value.detail)["code"] == "sandbox_conflict"
    assert db.execute_raw.await_count == 1
    assert [request["context"]["pdm"]["vault"] for request in transport.requests] == ["vault", "vault"]
    settings.assert_not_called()
    notify.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("which", ["sw", "pdm"])
async def test_disabled_tools_rejected(setup, which):
    _, sw, transport, _, _, _ = setup
    tool = sw if which == "sw" else cast(Any, sandbox.repository.get_tool_config).side_effect("pdm")
    tool.enabled = False
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert exc.value.detail == {"code": "tool_disabled", "message": "Enable the SolidWorks tool and its PDM tool first."}
    assert not transport.requests


@pytest.mark.anyio
@pytest.mark.parametrize(("parent", "name"), [("\\", "ok"), ("\\Parent", "é"), ("\\Parent", "bad."), ("\\Parent", "a" * 65)])
async def test_service_validates_input(setup, parent, name):
    _, _, transport, _, _, _ = setup
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", parent, name)
    assert exc.value.status_code == 422
    assert not transport.requests


@pytest.mark.anyio
async def test_create_agent_error_preserves_message(setup):
    _, _, transport, db, _, _ = setup
    transport.create_error = {"code": "pdm_conflict", "message": "A file already has that name."}
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert isinstance(exc.value.detail, dict)
    assert exc.value.detail == {"code": "pdm_error", "message": "A file already has that name."}
    db.execute_raw.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("code", "expected", "status"),
    [("host_not_running", "host_not_running", 503), ("operation_timeout", "operation_timeout", 504), ("pdm_unavailable", "pdm_error", 502)],
)
async def test_agent_error_mapping(setup, code, expected, status):
    _, _, transport, _, _, _ = setup
    transport.error = {"code": code, "message": "agent failure"}
    with pytest.raises(HTTPException) as exc:
        await sandbox.create_sandbox("sw", "pdm", "\\Parent", "Sandbox")
    assert exc.value.status_code == status
    assert isinstance(exc.value.detail, dict)
    assert cast(Any, exc.value.detail)["code"] == expected


@pytest.mark.parametrize(("parent", "name"), [("\\", "ok"), ("\\Parent", "é"), ("\\Parent", "bad."), ("\\Parent", "a" * 65)])
def test_invalid_input(parent, name):
    with pytest.raises(ValidationError):
        PdmSandboxRequest(pdm_tool_id="pdm", parent=parent, name=name)
