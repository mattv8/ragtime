from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import json
import shutil
import ssl
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ragtime.remote_client.tunnel_protocol import TunnelMessageType, decode_frame, encode_frame

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


def _server_frame(payload: bytes) -> bytes:
    size = len(payload)
    if size < 126:
        return bytes((0x82, size)) + payload
    return bytes((0x82, 126)) + size.to_bytes(2, "big") + payload


def test_python_tunnel_client_decodes_frames_coalesced_with_upgrade_response(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _tunnel_client_module()
    key = base64.b64encode(b"\0" * 16).decode("ascii")
    accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()).decode("ascii")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        "Sec-WebSocket-Protocol: ragtime-tunnel-v1\r\n\r\n"
    ).encode("ascii")

    class Connection:
        def __init__(self) -> None:
            self.received = [response + _server_frame(b"first") + _server_frame(b"second")]

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


async def _read_client_frame(reader: asyncio.StreamReader) -> bytes:
    first, second = await reader.readexactly(2)
    assert first == 0x82
    assert second & 0x80
    size = second & 0x7F
    if size == 126:
        size = int.from_bytes(await reader.readexactly(2), "big")
    mask = await reader.readexactly(4)
    payload = bytearray(await reader.readexactly(size))
    return bytes(value ^ mask[index % 4] for index, value in enumerate(payload))


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
            request = await reader.readuntil(b"\r\n\r\n")
            headers = request.decode("iso-8859-1").split("\r\n")
            assert any(line == "Sec-WebSocket-Protocol: ragtime-tunnel-v1" for line in headers)
            key = next(line.split(": ", 1)[1] for line in headers if line.lower().startswith("sec-websocket-key:"))
            accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
            writer.write(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n"
                    "Sec-WebSocket-Protocol: ragtime-tunnel-v1\r\n\r\n"
                ).encode()
            )
            await writer.drain()
            writer.write(_server_frame(encode_frame(TunnelMessageType.OPEN, 1)))
            writer.write(_server_frame(encode_frame(TunnelMessageType.DATA, 1, b"echo")))
            await writer.drain()
            while True:
                kind, stream_id, payload = decode_frame(await _read_client_frame(reader))
                if (kind, stream_id, payload) == (TunnelMessageType.DATA, 1, b"echo"):
                    received.set()
                    break
            writer.close()
            await writer.wait_closed()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            certificate, private_key = temporary / "certificate.pem", temporary / "private-key.pem"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(private_key),
                    "-out",
                    str(certificate),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=127.0.0.1",
                    "-addext",
                    "subjectAltName=IP:127.0.0.1",
                ],
                check=True,
                capture_output=True,
            )
            echo_server = await asyncio.start_server(echo, "127.0.0.1", 0)
            target_port = echo_server.sockets[0].getsockname()[1]
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificate, private_key)
            websocket_server = await asyncio.start_server(websocket, "127.0.0.1", 0, ssl=context)
            websocket_port = websocket_server.sockets[0].getsockname()[1]
            config = temporary / "client.json"
            config.write_text(
                json.dumps(
                    {
                        "server_url": f"wss://127.0.0.1:{websocket_port}/indexes/remote/tunnel",
                        "pairing_id": "test-pairing",
                        "credential": "test-credential",
                        "sshd_port": target_port,
                        "ca_file": str(certificate),
                    }
                )
            )
            client = await asyncio.create_subprocess_exec(sys.executable, str(CLIENT_PATH), str(config))
            try:
                await asyncio.wait_for(received.wait(), timeout=10)
            finally:
                client.terminate()
                await client.wait()
                websocket_server.close()
                echo_server.close()
                await websocket_server.wait_closed()
                await echo_server.wait_closed()

    asyncio.run(exercise())
