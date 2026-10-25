import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ragtime.indexer.models import SHELL_TOOL_TYPES, ToolType
from ragtime.remote_client.models import EnrollmentCompleteRequest, PairingCreateRequest
from ragtime.remote_client.service import RemotePairingService, _is_terminal_direct_probe_error, _most_actionable_probe_error, _safe_probe_address


def test_remote_pairing_accepts_only_remote_shell_tool_types():
    assert PairingCreateRequest.model_validate({"tool_type": "ssh_shell"}).tool_type is ToolType.SSH_SHELL
    assert PairingCreateRequest.model_validate({"tool_type": "windows_shell"}).tool_type is ToolType.WINDOWS_SHELL
    with pytest.raises(ValidationError):
        PairingCreateRequest.model_validate({"tool_type": "postgres"})


def test_remote_facts_bound_the_untrusted_enrollment_payload():
    payload = {
        "nonce": "n" * 16,
        "mode": "direct",
        "facts": {
            "os": "linux",
            "username": "remote",
            "hostname": "host",
            "addresses": ["10.0.0.1"],
            "host_keys": ["ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKzd7CnRZTlNJvDg79mNhrEWeSSOvlmRjMRmsm6U9Pux"],
        },
    }
    assert EnrollmentCompleteRequest.model_validate(payload).facts.sshd_port == 22
    payload["facts"]["addresses"] = ["10.0.0.1"] * 9
    with pytest.raises(ValidationError):
        EnrollmentCompleteRequest.model_validate(payload)


def test_probe_target_filter_rejects_non_routable_addresses():
    assert _safe_probe_address("10.0.0.1") == "10.0.0.1"
    for address in ("127.0.0.1", "169.254.1.1", "224.0.0.1", "0.0.0.0", "not-an-ip"):
        assert _safe_probe_address(address) is None


def test_probe_error_prefers_only_authentication_or_policy_failure_over_completion_order():
    assert _most_actionable_probe_error(["SSH connection timed out or was unreachable", "Host key verification failed"]) == "Host key verification failed"
    assert _most_actionable_probe_error(["authentication failed", "SSH connection timed out or was unreachable"]) == "authentication failed"
    assert not _is_terminal_direct_probe_error("Host key verification failed")
    assert _is_terminal_direct_probe_error("authentication failed")
    assert _is_terminal_direct_probe_error("policy rejected public-key authentication")


def test_auto_tunnels_after_only_host_key_mismatches_but_direct_mode_fails():
    pairing = {"id": "pairing", "toolId": "tool", "status": "consumed"}
    facts = {"os": "linux", "addresses": ["198.51.100.10"], "sshd_port": 22, "username": "remote", "host_keys": ["ssh-ed25519 key"]}
    service = RemotePairingService()
    with (
        patch("ragtime.remote_client.service.remote_pairing_repository.get_by_token", new=AsyncMock(return_value=pairing)),
        patch("ragtime.remote_client.service.remote_pairing_repository.redeem", new=AsyncMock(return_value=pairing)),
        patch(
            "ragtime.remote_client.service.remote_pairing_repository.select_tunnel", new=AsyncMock(return_value={**pairing, "transport": "tunnel"})
        ) as select_tunnel,
        patch("ragtime.remote_client.service.remote_pairing_repository.fail", new=AsyncMock(return_value={**pairing, "status": "failed"})) as fail,
        patch("ragtime.remote_client.service.tool_repository.get_tool_config", new=AsyncMock(return_value=SimpleNamespace(tool_type=ToolType.SSH_SHELL))),
        patch.object(service, "_verify_direct_ssh", new=AsyncMock(return_value=(None, "Host key verification failed"))),
    ):
        auto = asyncio.run(service.complete("token", "nonce", facts, "auto"))
        direct = asyncio.run(service.complete("token", "nonce", facts, "direct"))
    assert auto and auto["transport"] == "tunnel"
    select_tunnel.assert_awaited_once_with("pairing")
    assert direct and direct["status"] == "failed"
    fail.assert_awaited_once_with("pairing", "Host key verification failed")


def test_auto_fails_after_pinned_target_authentication_failure():
    pairing = {"id": "pairing", "toolId": "tool", "status": "consumed"}
    facts = {"os": "linux", "addresses": ["198.51.100.10"], "sshd_port": 22, "username": "remote", "host_keys": ["ssh-ed25519 key"]}
    service = RemotePairingService()
    with (
        patch("ragtime.remote_client.service.remote_pairing_repository.get_by_token", new=AsyncMock(return_value=pairing)),
        patch("ragtime.remote_client.service.remote_pairing_repository.redeem", new=AsyncMock(return_value=pairing)),
        patch("ragtime.remote_client.service.remote_pairing_repository.fail", new=AsyncMock(return_value={**pairing, "status": "failed"})) as fail,
        patch("ragtime.remote_client.service.tool_repository.get_tool_config", new=AsyncMock(return_value=SimpleNamespace(tool_type=ToolType.SSH_SHELL))),
        patch.object(service, "_verify_direct_ssh", new=AsyncMock(return_value=(None, "Authentication failed"))),
    ):
        result = asyncio.run(service.complete("token", "nonce", facts, "auto"))
    assert result and result["status"] == "failed"
    fail.assert_awaited_once_with("pairing", "Authentication failed")


def test_shell_tool_types_include_windows_execution():
    assert SHELL_TOOL_TYPES == frozenset({ToolType.ODOO_SHELL, ToolType.SSH_SHELL, ToolType.WINDOWS_SHELL})
