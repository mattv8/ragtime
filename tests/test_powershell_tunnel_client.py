from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

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

CLIENT_PATH = Path(__file__).parents[1] / "ragtime" / "remote_client" / "scripts" / "tunnel-client.ps1"
# Plain wire values from the tunnel protocol. Importing the ragtime package here would pull in app routes.
OPEN, DATA, PING, PONG = 1, 2, 4, 5
RUN_TIMEOUT_SECONDS = 15
RESPONSE_TIMEOUT_SECONDS = 0.5
STOP_TIMEOUT_SECONDS = 5
WARMUP_PAYLOAD = b"warm"
TIMED_PAYLOAD = b"echo"
TunnelFrame = tuple[int, int, bytes]


def _tunnel_frame(message_type: int, stream_id: int = 0, payload: bytes = b"") -> bytes:
    return server_frame(bytes((message_type,)) + stream_id.to_bytes(4, "big") + payload)


def _decode_tunnel_frame(frame: bytes) -> TunnelFrame:
    return frame[0], int.from_bytes(frame[1:5], "big"), frame[5:]


async def _echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(65536):
        writer.write(data)
        await writer.drain()
    writer.close()


async def _expect_frame(reader: asyncio.StreamReader, expected: TunnelFrame, timeout: float | None) -> None:
    received = _decode_tunnel_frame(await asyncio.wait_for(read_client_frame(reader), timeout=timeout))
    assert received == expected


async def _stop_client(client: asyncio.subprocess.Process) -> None:
    if client.returncode is not None:
        return
    client.terminate()
    try:
        await asyncio.wait_for(client.wait(), timeout=STOP_TIMEOUT_SECONDS)
    except TimeoutError:
        client.kill()
        await client.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="the test CA is scoped through the Linux SSL_CERT_FILE trust store")
@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell is required to exercise the PowerShell tunnel pump")
@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is required to generate the test-only TLS certificate")
def test_powershell_tunnel_client_dispatches_newly_queued_data_without_poll_delay() -> None:
    async def exercise() -> None:
        websocket_completed = asyncio.Event()
        websocket_errors: list[BaseException] = []

        async def websocket(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(upgrade_response(websocket_key(await read_upgrade_headers(reader))))
                writer.write(_tunnel_frame(OPEN, 1))
                writer.write(_tunnel_frame(DATA, 1, WARMUP_PAYLOAD))
                await writer.drain()
                # Warmup echo: the stream is connected and its first write has completed.
                await _expect_frame(reader, (DATA, 1, WARMUP_PAYLOAD), timeout=None)
                # PING/PONG barrier: PONG is sent only after the per-stream phase has cleared.
                writer.write(_tunnel_frame(PING))
                await writer.drain()
                await _expect_frame(reader, (PONG, 0, b""), timeout=None)
                # Timed DATA: the client has a pending read and no in-flight write.
                writer.write(_tunnel_frame(DATA, 1, TIMED_PAYLOAD))
                await writer.drain()
                await _expect_frame(reader, (DATA, 1, TIMED_PAYLOAD), timeout=RESPONSE_TIMEOUT_SECONDS)
                writer.close()
                await writer.wait_closed()
            except BaseException as error:
                websocket_errors.append(error)
            finally:
                websocket_completed.set()

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            certificate, private_key = create_test_certificate(temporary)
            echo_server = await start_loopback_server(_echo)
            websocket_server = await start_loopback_server(websocket, tls_server_context(certificate, private_key))
            config = write_tunnel_config(temporary, server_port(websocket_server), server_port(echo_server))
            stderr_path = temporary / "pwsh-stderr.log"
            environment = {**os.environ, "SSL_CERT_FILE": str(certificate)}
            with stderr_path.open("wb") as stderr:
                client = await asyncio.create_subprocess_exec(
                    "pwsh",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-File",
                    str(CLIENT_PATH),
                    str(config),
                    env=environment,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=stderr,
                )
                try:
                    await asyncio.wait_for(websocket_completed.wait(), timeout=RUN_TIMEOUT_SECONDS)
                    if websocket_errors:
                        raise websocket_errors[0]
                except Exception as error:
                    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
                    raise AssertionError(f"PowerShell tunnel client failed: {error!r}\npwsh stderr:\n{stderr_text}") from error
                finally:
                    await _stop_client(client)
                    await close_server(websocket_server)
                    await close_server(echo_server)

    asyncio.run(exercise())
