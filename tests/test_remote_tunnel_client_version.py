from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from ragtime.remote_client import routes, tunnel_versions
from ragtime.remote_client.tunnel_protocol import TUNNEL_CLIENT_VERSION, TUNNEL_CLIENT_VERSION_HEADER

SCRIPTS = Path(__file__).resolve().parents[1] / "ragtime" / "remote_client" / "scripts"
HASHES = {
    2: {
        "tunnel-client.ps1": "9d4547cd11abffa7b2713935107dd321b04d4a83247c47e8e2c44d59037c8297",
        "tunnel-client.py": "cd5f19184ddac843fc989fa5dc2c1a0bc4d515fce3013856d35d923b6426124d",
    }
}
PIN_MESSAGE = "Changing a tunnel client: bump TUNNEL_CLIENT_VERSION in both scripts and tunnel_protocol.py, then update these hashes."


@pytest.mark.parametrize(
    ("name", "pattern"),
    [("tunnel-client.ps1", r"\$TunnelClientVersion = (\d+)"), ("tunnel-client.py", r"TUNNEL_CLIENT_VERSION = (\d+)")],
)
def test_client_version_and_hash(name: str, pattern: str) -> None:
    content = (SCRIPTS / name).read_bytes().replace(b"\r\n", b"\n")
    text = content.decode()
    match = re.search(pattern, text)
    assert match and int(match.group(1)) == TUNNEL_CLIENT_VERSION
    assert TUNNEL_CLIENT_VERSION_HEADER in text
    if name.endswith(".ps1"):
        assert "SetRequestHeader('X-Ragtime-Tunnel-Client-Version', [string]$TunnelClientVersion)" in text
    else:
        assert '"X-Ragtime-Tunnel-Client-Version: {0}".format(TUNNEL_CLIENT_VERSION)' in text
    assert hashlib.sha256(content).hexdigest() == HASHES.get(TUNNEL_CLIENT_VERSION, {}).get(name), PIN_MESSAGE


@pytest.mark.parametrize(
    ("config", "state", "expected"),
    [
        ({"transport": "ssh", "pairing_id": "p"}, (True, 2), None),
        ({"transport": "tunnel"}, (True, 2), None),
        ({"transport": "tunnel", "pairing_id": "p"}, None, None),
        ({"transport": "tunnel", "pairing_id": "p"}, (True, 2), (True, 2, False)),
        ({"transport": "tunnel", "pairing_id": "p"}, (True, 1), (True, 1, True)),
        ({"transport": "tunnel", "pairing_id": "p"}, (True, None), (True, None, True)),
        ({"transport": "tunnel", "pairing_id": "p"}, (False, 2), (False, 2, False)),
        ({"transport": "tunnel", "pairing_id": "p"}, (True, 3), (True, 3, False)),
    ],
)
def test_summary(monkeypatch, config, state, expected) -> None:
    monkeypatch.setattr(tunnel_versions.relay_manager, "client_version", lambda _id: state)
    result = tunnel_versions.tunnel_client_summary(config)
    if expected is None:
        assert result is None
    else:
        connected, version, outdated = expected
        assert result == {"connected": connected, "version": version, "current_version": 2, "outdated": outdated}


def test_pairing_response_includes_tunnel_client(monkeypatch) -> None:
    monkeypatch.setattr(routes.tool_repository, "get_tool_config", AsyncMock(return_value=None))
    monkeypatch.setattr(tunnel_versions.relay_manager, "client_version", lambda _id: (True, 1))
    pairing = {"id": "p", "toolId": "t", "status": "active", "enrollmentExpiresAt": datetime.now(timezone.utc), "transport": "tunnel"}
    response = asyncio.run(routes._response(pairing))
    assert response.model_dump()["tunnel_client"] == {"connected": True, "version": 1, "current_version": 2, "outdated": True}


def test_tunnel_passes_parsed_header_to_relay(monkeypatch) -> None:
    websocket = SimpleNamespace(
        headers={"authorization": "Bearer credential", "sec-websocket-protocol": "ragtime-tunnel-v1", TUNNEL_CLIENT_VERSION_HEADER: "2"},
        accept=AsyncMock(),
        close=AsyncMock(),
    )
    monkeypatch.setattr(routes.remote_pairing_repository, "get_by_tunnel_credential", AsyncMock(return_value={"id": "p", "status": "active"}))
    attach = AsyncMock()
    monkeypatch.setattr(routes.relay_manager, "attach", attach)
    monkeypatch.setattr(routes.relay_manager, "endpoint_for", lambda _id: ("127.0.0.1", 1))
    monkeypatch.setattr(routes.remote_pairing_service, "verify_tunnel", AsyncMock())
    asyncio.run(routes.remote_tunnel(cast(Any, websocket)))
    attach.assert_awaited_once_with("p", websocket, client_version=2)
