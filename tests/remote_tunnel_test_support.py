from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import ssl
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

TUNNEL_PROTOCOL = "ragtime-tunnel-v1"
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

ConnectionHandler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


def create_test_certificate(directory: Path) -> tuple[Path, Path]:
    """Create a throwaway self-signed certificate valid for 127.0.0.1 inside ``directory``."""
    certificate, private_key = directory / "certificate.pem", directory / "private-key.pem"
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
    return certificate, private_key


def tls_server_context(certificate: Path, private_key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, private_key)
    return context


async def start_loopback_server(handler: ConnectionHandler, ssl_context: ssl.SSLContext | None = None) -> asyncio.Server:
    return await asyncio.start_server(handler, "127.0.0.1", 0, ssl=ssl_context)


def server_port(server: asyncio.Server) -> int:
    return server.sockets[0].getsockname()[1]


async def close_server(server: asyncio.Server) -> None:
    server.close()
    await server.wait_closed()


def write_tunnel_config(directory: Path, websocket_port: int, sshd_port: int, ca_file: Path | None = None) -> Path:
    config: dict[str, object] = {
        "server_url": f"wss://127.0.0.1:{websocket_port}/indexes/remote/tunnel",
        "pairing_id": "test-pairing",
        "credential": "test-credential",
        "sshd_port": sshd_port,
    }
    if ca_file is not None:
        config["ca_file"] = str(ca_file)
    path = directory / "client.json"
    path.write_text(json.dumps(config))
    return path


async def read_upgrade_headers(reader: asyncio.StreamReader) -> list[str]:
    request = await reader.readuntil(b"\r\n\r\n")
    return request.decode("iso-8859-1").split("\r\n")


def websocket_key(headers: list[str]) -> str:
    return next(line.split(": ", 1)[1] for line in headers if line.lower().startswith("sec-websocket-key:"))


def upgrade_response(key: str) -> bytes:
    accept = base64.b64encode(hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()).decode("ascii")
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n"
        f"Sec-WebSocket-Protocol: {TUNNEL_PROTOCOL}\r\n\r\n"
    ).encode("ascii")


def server_frame(payload: bytes) -> bytes:
    """Wrap a payload in an unmasked server-to-client binary WebSocket frame."""
    size = len(payload)
    if size < 126:
        return bytes((0x82, size)) + payload
    return bytes((0x82, 126)) + size.to_bytes(2, "big") + payload


async def read_client_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one masked client-to-server binary WebSocket frame and return its unmasked payload."""
    first, second = await reader.readexactly(2)
    assert first == 0x82
    assert second & 0x80
    size = second & 0x7F
    if size == 126:
        size = int.from_bytes(await reader.readexactly(2), "big")
    mask = await reader.readexactly(4)
    payload = bytearray(await reader.readexactly(size))
    return bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
