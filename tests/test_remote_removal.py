from __future__ import annotations

import asyncio
import base64
import json
import re
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def test_cleanup_request_rejects_unconfirmed_machine_wide_removal() -> None:
    from ragtime.remote_client.removal import validate_cleanup_request

    with pytest.raises(ValueError, match="confirm_machine_wide"):
        validate_cleanup_request("disable_ssh", False)


def test_cleanup_request_rejects_confirmation_without_disable_mode() -> None:
    from ragtime.remote_client.removal import validate_cleanup_request

    with pytest.raises(ValueError, match="only valid"):
        validate_cleanup_request("pairing_only", True)


def test_windows_pairing_requires_windows_type_and_uuid_pairing_id() -> None:
    from ragtime.remote_client.removal import is_windows_pairing_tool

    pairing_id = "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd"
    assert is_windows_pairing_tool("windows_shell", {"pairing_id": pairing_id})
    assert not is_windows_pairing_tool("ssh_shell", {"pairing_id": pairing_id})
    assert not is_windows_pairing_tool("windows_shell", {"pairing_id": "../unsafe"})


def test_local_recovery_command_encodes_ssh_user_and_requires_https() -> None:
    from ragtime.remote_client.removal import local_removal_command

    pairing_id = "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd"
    target_user = "DOMAIN\\O'Brien"
    command = local_removal_command("https://ragtime.example", pairing_id, "pairing_only", target_user)
    invocation = base64.b64decode(command.split("-EncodedCommand ")[-1]).decode("utf-16le")
    encoded = re.search(r"FromBase64String\('([^']+)'\)", invocation)

    assert target_user not in command
    assert encoded is not None
    assert json.loads(base64.b64decode(encoded.group(1))) == {"pairing_id": pairing_id, "target_user": target_user}
    assert "powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand" in command
    assert local_removal_command("http://ragtime.example", pairing_id, "pairing_only", target_user) == ""


@pytest.mark.parametrize(
    "bad_config,stdout", [({"port": "bad"}, ""), ({}, 'RAGTIME_REMOVAL_RESULT:{"steps":{"pairing_cleanup":[],"ssh_service":"confirmed"}}')]
)
def test_cleanup_malformed_config_or_output_returns_unconfirmed(monkeypatch, bad_config, stdout) -> None:
    from ragtime.core.ssh import SSHResult
    from ragtime.remote_client.removal import execute_windows_cleanup

    monkeypatch.setattr("ragtime.remote_client.removal.execute_ssh_command", lambda *_args: SSHResult(stdout, "", 0, True))
    result = asyncio.run(execute_windows_cleanup({"host": "host", "user": "user", **bad_config}, "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd", "pairing_only", False))
    assert result.steps["pairing_cleanup"] == "unconfirmed"


def test_http_recovery_provides_trusted_transfer_and_exact_local_invocation(monkeypatch) -> None:
    from ragtime.core.ssh import SSHResult
    from ragtime.remote_client.removal import execute_windows_cleanup

    monkeypatch.setattr("ragtime.remote_client.removal.execute_ssh_command", lambda *_args: SSHResult("", "offline", -1, False))
    result = asyncio.run(
        execute_windows_cleanup(
            {"host": "host", "user": "DOMAIN\\O'Brien"}, "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd", "disable_ssh", True, server_url="http://example"
        )
    )
    assert result.local_removal_command == ""
    instructions = result.as_dict()["local_removal_instructions"]
    assert "trusted" in instructions.lower()
    invocation_encoded = instructions.split("-EncodedCommand ")[-1]
    invocation = base64.b64decode(invocation_encoded).decode("utf-16le")
    assert "-CleanupMode DisableSsh -ConfirmMachineWide" in invocation
    assert "-TargetUser" in invocation
    assert "iwr" not in instructions


def test_remote_cleanup_parses_only_a_valid_final_removal_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.core.ssh import SSHConfig, SSHResult
    from ragtime.remote_client.removal import execute_windows_cleanup

    captured: dict[str, object] = {}

    def fake_execute(config, command, input_data):  # type: ignore[no-untyped-def]
        captured.update(config=config, command=command, input_data=input_data)
        return SSHResult(
            stdout='noise\nRAGTIME_REMOVAL_RESULT:{"steps":{"pairing_cleanup":"confirmed","ssh_service":"unconfirmed"},"warnings":["connection will close"]}\n',
            stderr="",
            exit_code=0,
            success=True,
        )

    monkeypatch.setattr("ragtime.remote_client.removal.execute_ssh_command", fake_execute)
    result = asyncio.run(
        execute_windows_cleanup(
            {"host": "windows.example", "user": "DOMAIN\\user", "key_content": "key", "shell": "powershell"},
            "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd",
            "disable_ssh",
            True,
        )
    )

    assert result.steps == {"pairing_cleanup": "confirmed", "ssh_service": "unconfirmed"}
    assert result.warnings == ["connection will close"]
    assert "RAGTIME_REMOVAL_ARGS" in str(captured["command"])
    assert isinstance(captured["input_data"], str) and "CmdletBinding" in captured["input_data"]
    ssh_config = cast(SSHConfig, captured["config"])
    assert ssh_config.timeout <= 30
    assert ssh_config.total_timeout is not None and ssh_config.total_timeout <= 30


def test_remote_cleanup_does_not_treat_disconnect_as_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.core.ssh import SSHResult
    from ragtime.remote_client.removal import execute_windows_cleanup

    monkeypatch.setattr(
        "ragtime.remote_client.removal.execute_ssh_command",
        lambda *_args: SSHResult(stdout="", stderr="connection closed", exit_code=-1, success=False),
    )
    result = asyncio.run(
        execute_windows_cleanup(
            {"host": "windows.example", "user": "user", "key_content": "key", "shell": "powershell"},
            "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd",
            "pairing_only",
            False,
        )
    )

    assert result.steps == {"pairing_cleanup": "unconfirmed", "ssh_service": "not_requested"}
    assert result.warnings == ["connection closed"]


def test_shared_tool_deletion_revokes_all_live_pairing_relays(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.indexer.repository import IndexerRepository

    class FakeDatabase:
        class ToolConfig:
            async def delete(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                return None

        toolconfig = ToolConfig()

        async def query_raw(self, *_args):  # type: ignore[no-untyped-def]
            return [{"id": "first"}, {"id": "second"}]

    revoked_relays: list[str] = []

    async def fake_relay_revoke(pairing_id: str) -> None:
        revoked_relays.append(pairing_id)

    repository = IndexerRepository()
    monkeypatch.setattr(repository, "_get_db", lambda: _return(FakeDatabase()))
    monkeypatch.setattr("ragtime.remote_client.relay.relay_manager.revoke", fake_relay_revoke)

    assert asyncio.run(repository.delete_tool_config("tool"))
    assert set(revoked_relays) == {"first", "second"}


async def _return(value):  # type: ignore[no-untyped-def]
    return value


def test_delete_route_attempts_cleanup_before_revocation_and_database_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.indexer import routes
    from ragtime.remote_client.removal import CleanupResult

    events: list[str] = []
    tool = _windows_tool()

    async def get_tool_config(_tool_id: str):
        return tool

    async def get_pairing(_pairing_id: str):
        return {"toolId": "tool-id", "status": "paired"}

    async def cleanup(*_args, **_kwargs):
        events.append("cleanup")
        return CleanupResult("disable_ssh", {"pairing_cleanup": "confirmed", "ssh_service": "unconfirmed"}, [], "local command")

    async def delete_tool(_tool_id: str):
        events.append("delete")
        return True

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, get_pairing, cleanup, delete_tool)

    response = asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup="disable_ssh", confirm_machine_wide=True, _user=cast(Any, None)))

    assert events == ["cleanup", "delete"]
    assert response["cleanup"]["steps"]["ssh_service"] == "unconfirmed"


def test_delete_route_deletes_after_unconfirmed_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.indexer import routes
    from ragtime.remote_client.removal import CleanupResult

    events: list[str] = []

    async def get_tool_config(_tool_id: str):
        return _windows_tool()

    async def get_pairing(_pairing_id: str):
        return {"toolId": "tool-id", "status": "paired"}

    async def cleanup(*_args, **_kwargs):
        events.append("cleanup")
        return CleanupResult("pairing_only", {"pairing_cleanup": "unconfirmed", "ssh_service": "not_requested"}, ["offline"], "local command")

    async def delete_tool(_tool_id: str):
        events.append("delete")
        return True

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, get_pairing, cleanup, delete_tool)

    response = asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup="pairing_only", confirm_machine_wide=None, _user=cast(Any, None)))

    assert events == ["cleanup", "delete"]
    assert response["cleanup"]["warnings"] == ["offline"]


def test_delete_route_without_cleanup_does_not_mutate_host_and_revokes_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.indexer import routes

    events: list[str] = []

    async def get_tool_config(_tool_id: str):
        return _windows_tool()

    async def no_host_cleanup(*_args, **_kwargs):
        raise AssertionError("legacy deletion must not attempt host cleanup")

    async def delete_tool(_tool_id: str):
        events.append("delete")
        return True

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, None, no_host_cleanup, delete_tool)

    response = asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup=None, confirm_machine_wide=None, _user=cast(Any, None)))

    assert events == ["delete"]
    assert response == {"message": "Tool configuration deleted"}


@pytest.mark.parametrize(
    "mode,confirm", [("disable_ssh", False), ("disable_ssh", None), (None, True), (None, False), ("pairing_only", True), ("pairing_only", False)]
)
def test_delete_route_rejects_invalid_cleanup_before_host_or_database_actions(monkeypatch: pytest.MonkeyPatch, mode, confirm) -> None:
    from ragtime.indexer import routes

    actions: list[str] = []

    async def get_tool_config(_tool_id: str):
        return _windows_tool()

    async def action(*_args, **_kwargs):
        actions.append("action")

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, action, action, action)

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup=mode, confirm_machine_wide=confirm, _user=cast(Any, None)))

    assert error.value.status_code == 400
    assert actions == []


@pytest.mark.parametrize("pairing", [{"toolId": "other-tool", "status": "paired"}, {"toolId": "tool-id", "status": "pending"}, None])
def test_delete_route_rejects_foreign_or_unpaired_pairing_before_host_actions(monkeypatch: pytest.MonkeyPatch, pairing) -> None:
    from ragtime.indexer import routes

    actions: list[str] = []

    async def get_tool_config(_tool_id: str):
        return _windows_tool()

    async def foreign_pairing(_pairing_id: str):
        return pairing

    async def action(*_args, **_kwargs):
        actions.append("action")

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, foreign_pairing, action, action)

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup="pairing_only", confirm_machine_wide=None, _user=cast(Any, None)))

    assert error.value.status_code == 400
    assert actions == []


@pytest.mark.parametrize("bad_port,stdout", [("bad", ""), (22, 'RAGTIME_REMOVAL_RESULT:{"steps":{"pairing_cleanup":[],"ssh_service":"confirmed"}}')])
def test_route_still_deletes_after_real_coordinator_handles_bad_inputs(monkeypatch, bad_port, stdout) -> None:
    from ragtime.core.ssh import SSHResult
    from ragtime.indexer import routes
    from ragtime.remote_client.removal import execute_windows_cleanup

    tool = _windows_tool()
    tool.connection_config["port"] = bad_port
    deleted: list[str] = []

    async def get_tool_config(_tool_id):
        return tool

    async def get_pairing(_pairing_id):
        return {"toolId": "tool-id", "status": "paired"}

    async def delete_tool(tool_id):
        deleted.append(tool_id)
        return True

    monkeypatch.setattr("ragtime.remote_client.removal.execute_ssh_command", lambda *_args: SSHResult(stdout, "", 0, True))
    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, get_pairing, execute_windows_cleanup, delete_tool)
    result = asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup="pairing_only", confirm_machine_wide=None, _user=cast(Any, None)))
    assert deleted == ["tool-id"]
    assert result["cleanup"]["steps"]["pairing_cleanup"] == "unconfirmed"


def test_delete_route_includes_cleanup_details_when_database_delete_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from ragtime.indexer import routes
    from ragtime.remote_client.removal import CleanupResult

    async def get_tool_config(_tool_id: str):
        return _windows_tool()

    async def get_pairing(_pairing_id: str):
        return {"toolId": "tool-id", "status": "paired"}

    async def cleanup(*_args, **_kwargs):
        return CleanupResult("pairing_only", {"pairing_cleanup": "failed", "ssh_service": "not_requested"}, ["remote failure"], "local command")

    async def delete_tool(_tool_id: str):
        return False

    _patch_delete_route_dependencies(monkeypatch, routes, get_tool_config, get_pairing, cleanup, delete_tool)

    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.delete_tool_config("tool-id", _request(), ssh_cleanup="pairing_only", confirm_machine_wide=None, _user=cast(Any, None)))

    assert error.value.status_code == 404
    assert error.value.detail == {
        "message": "Tool configuration not found",
        "cleanup": {
            "mode": "pairing_only",
            "steps": {"pairing_cleanup": "failed", "ssh_service": "not_requested"},
            "warnings": ["remote failure"],
            "local_removal_command": "local command",
        },
    }


def _windows_tool() -> SimpleNamespace:
    return SimpleNamespace(
        name="Windows remote",
        tool_type="windows_shell",
        connection_config={"pairing_id": "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd", "host": "windows.example"},
    )


def _request() -> Request:
    return Request({"type": "http", "method": "DELETE", "scheme": "https", "path": "/tools/tool-id", "headers": [(b"host", b"ragtime.example")]})


def _patch_delete_route_dependencies(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, routes, get_tool_config, get_pairing, cleanup, delete_tool
) -> None:
    monkeypatch.setattr(routes.repository, "get_tool_config", get_tool_config)
    monkeypatch.setattr(routes.repository, "delete_tool_config", delete_tool)
    monkeypatch.setattr(routes, "execute_windows_cleanup", cleanup)
    if get_pairing is not None:
        monkeypatch.setattr(routes.remote_pairing_repository, "get", get_pairing)
    monkeypatch.setattr(routes, "invalidate_settings_cache", lambda: None)
    monkeypatch.setattr(routes, "notify_tools_changed", lambda: None)
    monkeypatch.setattr(routes.rag, "initialize", _no_op)


async def _no_op() -> None:
    return None
