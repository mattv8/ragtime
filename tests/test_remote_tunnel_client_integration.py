from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from ragtime.remote_client.tunnel_protocol import TunnelMessageType, decode_frame, encode_frame
from tests.remote_tunnel_test_support import (
    close_server,
    create_test_certificate,
    read_client_frame,
    read_upgrade_headers,
    server_frame,
    server_port,
    start_loopback_server,
    tls_server_context,
    upgrade_response,
    websocket_key,
    write_tunnel_config,
)

CLIENT_PATH = Path(__file__).parents[1] / "ragtime" / "remote_client" / "scripts" / "tunnel-client.py"


def _tunnel_client_module():
    spec = importlib.util.spec_from_file_location("ragtime_remote_tunnel_client", CLIENT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_tunnel_client_drains_queued_writes_before_directional_shutdown() -> None:
    module = _tunnel_client_module()

    class WebSocket:
        def send_binary(self, payload: bytes) -> None:
            return None

    class Connection:
        def __init__(self) -> None:
            self.shutdown_calls: list[int] = []

        def shutdown(self, how: int) -> None:
            self.shutdown_calls.append(how)

    tunnel = module.Tunnel(WebSocket(), 22)
    stream = module._ClientStream()
    connection = Connection()
    stream.connection = connection
    stream.pending_bytes = 4
    stream.write_in_flight = True
    tunnel.streams[1] = stream

    tunnel._close_remote(1)
    assert connection.shutdown_calls == []

    stream.pending_bytes = 0
    stream.write_in_flight = False
    tunnel._shutdown_local_write_if_drained(1, stream)
    assert connection.shutdown_calls == [module.socket.SHUT_WR]


def test_python_tunnel_client_decodes_frames_coalesced_with_upgrade_response(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _tunnel_client_module()
    key = base64.b64encode(b"\0" * 16).decode("ascii")
    response = upgrade_response(key) + server_frame(b"first") + server_frame(b"second")

    class Connection:
        def __init__(self) -> None:
            self.received = [response]

        def recv(self, size: int) -> bytes:
            return self.received.pop(0) if self.received else b""

        def sendall(self, data: bytes) -> None:
            return None

        def settimeout(self, timeout: int) -> None:
            return None

    connection = Connection()

    class Context:
        def wrap_socket(self, raw: Connection, server_hostname: str) -> Connection:
            return raw

    monkeypatch.setattr(module.os, "urandom", lambda size: b"\0" * size)
    monkeypatch.setattr(module.socket, "create_connection", lambda address, timeout: connection)
    monkeypatch.setattr(module.ssl, "create_default_context", lambda cafile: Context())

    websocket = module.connect(
        {
            "server_url": "wss://example.test/tunnel",
            "credential": "credential",
            "pairing_id": "pairing",
        }
    )

    assert websocket.receive_binary() == b"first"
    assert websocket.receive_binary() == b"second"


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is required to generate the test-only TLS certificate")
def test_python_tunnel_client_round_trips_over_verified_local_tls() -> None:
    async def exercise() -> None:
        received = asyncio.Event()

        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
            writer.close()

        async def websocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            headers = await read_upgrade_headers(reader)
            assert any(line == "Sec-WebSocket-Protocol: ragtime-tunnel-v1" for line in headers)
            writer.write(upgrade_response(websocket_key(headers)))
            await writer.drain()
            writer.write(server_frame(encode_frame(TunnelMessageType.OPEN, 1)))
            writer.write(server_frame(encode_frame(TunnelMessageType.DATA, 1, b"echo")))
            await writer.drain()
            while True:
                kind, stream_id, payload = decode_frame(await read_client_frame(reader))
                if (kind, stream_id, payload) == (TunnelMessageType.DATA, 1, b"echo"):
                    received.set()
                    break
            writer.close()
            await writer.wait_closed()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            certificate, private_key = create_test_certificate(temporary)
            echo_server = await start_loopback_server(echo)
            websocket_server = await start_loopback_server(websocket, tls_server_context(certificate, private_key))
            config = write_tunnel_config(temporary, server_port(websocket_server), server_port(echo_server), certificate)
            client = await asyncio.create_subprocess_exec(sys.executable, str(CLIENT_PATH), str(config))
            try:
                await asyncio.wait_for(received.wait(), timeout=10)
            finally:
                client.terminate()
                await client.wait()
                await close_server(websocket_server)
                await close_server(echo_server)

    asyncio.run(exercise())
