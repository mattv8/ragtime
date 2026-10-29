from __future__ import annotations

import asyncio

import ragtime.remote_client.relay as relay_module
from ragtime.remote_client.relay import RelayManager
from ragtime.remote_client.tunnel_protocol import TunnelMessageType, decode_frame, encode_frame


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.sent: asyncio.Queue[bytes] = asyncio.Queue()
        self.closed = False

    async def receive(self) -> dict[str, object]:
        return await self.incoming.get()

    async def send_bytes(self, data: bytes) -> None:
        await self.sent.put(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True

    async def next_frame(self) -> tuple[TunnelMessageType, int, bytes]:
        return decode_frame(await asyncio.wait_for(self.sent.get(), timeout=1))


def test_relay_forwards_a_multiplexed_stream_and_removes_stale_endpoint() -> None:
    async def exercise() -> None:
        manager = RelayManager()
        await manager.start()
        websocket = FakeWebSocket()
        tunnel_task = asyncio.create_task(manager.attach("pairing", websocket))
        for _ in range(20):
            endpoint = manager.endpoint_for("pairing")
            if endpoint:
                break
            await asyncio.sleep(0.01)
        assert endpoint is not None
        assert endpoint == ("127.0.0.1", endpoint[1])
        reader, writer = await asyncio.open_connection(endpoint[0], endpoint[1])
        kind, stream_id, payload = await websocket.next_frame()
        assert (kind, payload) == (TunnelMessageType.OPEN, b"")

        await websocket.incoming.put({"type": "websocket.receive", "bytes": encode_frame(TunnelMessageType.DATA, stream_id, b"echo")})
        assert await asyncio.wait_for(reader.readexactly(4), timeout=1) == b"echo"

        writer.write(b"request")
        await writer.drain()
        kind, received_stream_id, payload = await websocket.next_frame()
        assert (kind, received_stream_id, payload) == (TunnelMessageType.DATA, stream_id, b"request")

        await websocket.incoming.put({"type": "websocket.receive", "bytes": encode_frame(TunnelMessageType.CLOSE, stream_id)})
        assert await asyncio.wait_for(reader.read(), timeout=1) == b""
        writer.close()
        await writer.wait_closed()
        await manager.revoke("pairing")
        assert manager.endpoint_for("pairing") is None
        await websocket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(tunnel_task, timeout=1)
        await manager.stop()

    asyncio.run(exercise())


def test_replacement_session_closes_old_listener() -> None:
    async def exercise() -> None:
        manager = RelayManager()
        await manager.start()
        first, second = FakeWebSocket(), FakeWebSocket()
        first_task = asyncio.create_task(manager.attach("pairing", first))
        for _ in range(20):
            old_endpoint = manager.endpoint_for("pairing")
            if old_endpoint:
                break
            await asyncio.sleep(0.01)
        second_task = asyncio.create_task(manager.attach("pairing", second))
        for _ in range(20):
            endpoint = manager.endpoint_for("pairing")
            if endpoint and endpoint != old_endpoint:
                break
            await asyncio.sleep(0.01)
        assert endpoint and endpoint != old_endpoint
        assert first.closed
        await manager.stop()
        await asyncio.wait_for(first_task, timeout=1)
        await asyncio.wait_for(second_task, timeout=1)

    asyncio.run(exercise())


def test_revoke_aborts_an_active_local_connection_without_waiting_for_peer() -> None:
    async def exercise() -> None:
        manager = RelayManager()
        await manager.start()
        websocket = FakeWebSocket()
        task = asyncio.create_task(manager.attach("pairing", websocket))
        for _ in range(20):
            endpoint = manager.endpoint_for("pairing")
            if endpoint:
                break
            await asyncio.sleep(0.01)
        assert endpoint is not None
        reader, writer = await asyncio.open_connection(endpoint[0], endpoint[1])
        await websocket.next_frame()
        await asyncio.wait_for(manager.revoke("pairing"), timeout=1)
        assert await asyncio.wait_for(reader.read(), timeout=1) == b""
        assert websocket.closed
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(exercise())


def test_heartbeat_timeout_closes_the_session_without_cancelling_cleanup(monkeypatch) -> None:
    async def exercise() -> None:
        monkeypatch.setattr(relay_module, "HEARTBEAT_SECONDS", 0.01)
        manager = RelayManager()
        await manager.start()
        websocket = FakeWebSocket()
        task = asyncio.create_task(manager.attach("pairing", websocket))
        await asyncio.wait_for(task, timeout=1)
        assert websocket.closed
        assert manager.endpoint_for("pairing") is None

    asyncio.run(exercise())
