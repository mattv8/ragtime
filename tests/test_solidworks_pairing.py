import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from ragtime.indexer.models import ToolType
from ragtime.remote_client.models import PairingCreateRequest
from ragtime.remote_client.removal import is_windows_pairing_tool
from ragtime.remote_client.repository import RemotePairingRepository
from ragtime.remote_client.service import RemotePairingService


def test_pairing_request_types():
    for tool_type in (ToolType.SOLIDWORKS, ToolType.SSH_SHELL, ToolType.WINDOWS_SHELL):
        assert PairingCreateRequest(tool_type=tool_type).tool_type == tool_type
    with pytest.raises(ValidationError):
        PairingCreateRequest(tool_type="postgres")  # pyright: ignore[reportArgumentType] - deliberately invalid tool type


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize(
    ("tool_type", "os", "accepted"),
    [
        (ToolType.SOLIDWORKS, "windows", True),
        (ToolType.SOLIDWORKS, "linux", False),
        (ToolType.WINDOWS_SHELL, "windows", True),
        (ToolType.WINDOWS_SHELL, "linux", False),
        (ToolType.SSH_SHELL, "linux", True),
        (ToolType.SSH_SHELL, "windows", False),
    ],
)
def test_pairing_requires_matching_os(offline, tool_type, os, accepted):
    pairing = {"id": "pairing", "toolId": "tool", "status": "consumed"}
    facts = {"os": os, "sshd_port": 22, "username": "user", "host_keys": ["ssh-ed25519 key"]}
    service = RemotePairingService()
    with (
        patch("ragtime.remote_client.service.remote_pairing_repository.get_by_token", new=AsyncMock(return_value=pairing)),
        patch("ragtime.remote_client.service.remote_pairing_repository.get", new=AsyncMock(return_value=pairing)),
        patch("ragtime.remote_client.service.remote_pairing_repository.redeem", new=AsyncMock(return_value=pairing)) as redeem,
        patch("ragtime.remote_client.service.remote_pairing_repository.redeem_offline", new=AsyncMock(return_value=pairing)) as redeem_offline,
        patch("ragtime.remote_client.service.tool_repository.get_tool_config", new=AsyncMock(return_value=SimpleNamespace(tool_type=tool_type))),
        patch.object(service, "_verify_direct_ssh", new=AsyncMock(return_value=("192.0.2.1", None))),
        patch.object(service, "_mark_verified_pairing", new=AsyncMock(return_value=pairing)) as mark,
    ):
        result = asyncio.run(service.complete_offline("pairing", "nonce", facts) if offline else service.complete("token", "nonce", facts, "direct"))
    assert (result is not None) is accepted
    assert (redeem_offline if offline else redeem).await_count == int(accepted)
    if accepted:
        assert mark.await_args is not None
        assert mark.await_args.args[1]["shell"] == ("powershell" if os == "windows" else "posix")


def test_solidworks_draft_creation():
    db = SimpleNamespace(
        remotepairing=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(id="pairing")), update=AsyncMock()),
        toolaccesspolicy=SimpleNamespace(upsert=AsyncMock()),
    )
    repository = RemotePairingRepository()
    with (
        patch("ragtime.remote_client.repository.get_db", new=AsyncMock(return_value=db)),
        patch("ragtime.remote_client.repository.encrypt_secret", return_value="encrypted"),
        patch("ragtime.remote_client.repository.tool_repository.create_tool_config", new=AsyncMock(return_value=SimpleNamespace(id="tool"))) as create,
        patch.object(repository, "get", new=AsyncMock(return_value={"id": "pairing"})),
    ):
        asyncio.run(repository.create(tool_type=ToolType.SOLIDWORKS, token="token", tunnel_credential="credential", private_key="key", public_key="public"))
    assert create.await_args is not None
    draft = create.await_args.args[0]
    assert draft.name == "SolidWorks remote client (pending)"
    assert draft.tool_type == ToolType.SOLIDWORKS
    assert not draft.enabled
    assert draft.connection_config == {"shell": "powershell"}


def test_solidworks_windows_removal_requires_pairing_id():
    assert is_windows_pairing_tool("solidworks", {"pairing_id": "e57fa6ec-0f30-4323-a12f-9f4c2b3a98fd"})
    assert not is_windows_pairing_tool("solidworks", {})
