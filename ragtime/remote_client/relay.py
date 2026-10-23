"""Single-process, localhost-only WebSocket relay for outbound SSH tunnels."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ragtime.remote_client.tunnel_protocol import MAX_DATA_PAYLOAD, TunnelMessageType, TunnelProtocolError, decode_frame, encode_frame

HOST = "127.0.0.1"
MAX_STREAMS_PER_SESSION = 32
MAX_STREAM_BUFFER_BYTES = 1024 * 1024
HEARTBEAT_SECONDS = 30
MISSED_HEARTBEATS = 2
SEND_TIMEOUT_SECONDS = 10


class RelayUnavailable(RuntimeError):
    """The in-process relay cannot safely be used in this deployment."""


@dataclass
class _Stream:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    outbound_closed: bool = False
    inbound_closed: bool = False
    queued: deque[bytes] = field(default_factory=deque)
    queued_bytes: int = 0
    data_ready: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)


class _Session:
    def __init__(self, manager: "RelayManager", pairing_id: str, websocket: Any) -> None:
        self.manager = manager
        self.pairing_id = pairing_id
        self.websocket = websocket
        self.server: asyncio.AbstractServer | None = None
        self.streams: dict[int, _Stream] = {}
        self.next_stream_id = 1
        self.send_lock = asyncio.Lock()
        self.closed = asyncio.Event()
        self.last_pong = time.monotonic()
        self.tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        if self.closed.is_set():
            return
        self.server = await asyncio.start_server(self._accept_local_connection, HOST, 0, limit=MAX_STREAM_BUFFER_BYTES)
        if self.closed.is_set():
            self.server.close()

    def endpoint(self) -> tuple[str, int] | None:
        if self.closed.is_set() or self.server is None or not self.server.sockets:
            return None
        return HOST, int(self.server.sockets[0].getsockname()[1])

    async def run(self) -> None:
        heartbeat = asyncio.create_task(self._heartbeat())
        self.tasks.add(heartbeat)
        closed_waiter = asyncio.create_task(self.closed.wait())
        try:
            while not self.closed.is_set():
                receive_task = asyncio.create_task(self.websocket.receive())
                done, _ = await asyncio.wait({receive_task, closed_waiter}, return_when=asyncio.FIRST_COMPLETED)
                if closed_waiter in done:
                    receive_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receive_task
                    break
                message = receive_task.result()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("type") != "websocket.receive":
                    continue
                data = message.get("bytes")
                if not isinstance(data, bytes) or message.get("text") is not None:
                    raise TunnelProtocolError("tunnel only accepts binary websocket messages")
                await self._dispatch(data)
        except (TimeoutError, TunnelProtocolError):
            await self.close(code=1002)
        finally:
            closed_waiter.cancel()
            await self.close()

    async def _accept_local_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.closed.is_set() or len(self.streams) >= MAX_STREAMS_PER_SESSION:
            writer.transport.abort()
            return
        stream_id = self.next_stream_id
        self.next_stream_id += 1
        stream = _Stream(reader, writer)
        self.streams[stream_id] = stream
        try:
            await self._send(TunnelMessageType.OPEN, stream_id)
        except (ConnectionError, OSError, RuntimeError, TimeoutError):
            self._abort_stream(stream_id)
            return
        self._create_stream_task(stream, self._read_local(stream_id, stream))
        self._create_stream_task(stream, self._write_local(stream_id, stream))

    def _create_stream_task(self, stream: _Stream, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        stream.tasks.add(task)
        task.add_done_callback(stream.tasks.discard)

    async def _read_local(self, stream_id: int, stream: _Stream) -> None:
        try:
            while data := await stream.reader.read(MAX_DATA_PAYLOAD):
                await self._send(TunnelMessageType.DATA, stream_id, data)
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            await self._send_close(stream_id)

    async def _write_local(self, stream_id: int, stream: _Stream) -> None:
        try:
            while True:
                await stream.data_ready.wait()
                while stream.queued:
                    data = stream.queued.popleft()
                    stream.queued_bytes -= len(data)
                    stream.writer.write(data)
                    await stream.writer.drain()
                stream.data_ready.clear()
                if stream.queued:
                    stream.data_ready.set()
                    continue
                if stream.inbound_closed:
                    if stream.writer.can_write_eof():
                        stream.writer.write_eof()
                        await stream.writer.drain()
                    return
        except (ConnectionError, OSError, RuntimeError):
            await self._send_close(stream_id)

    async def _dispatch(self, raw_frame: bytes) -> None:
        message_type, stream_id, payload = decode_frame(raw_frame)
        if message_type is TunnelMessageType.PING:
            await self._send(TunnelMessageType.PONG)
            return
        if message_type is TunnelMessageType.PONG:
            self.last_pong = time.monotonic()
            return
        if message_type is TunnelMessageType.OPEN:
            raise TunnelProtocolError("remote peers may not open relay streams")
        stream = self.streams.get(stream_id)
        if stream is None:
            return
        if message_type is TunnelMessageType.DATA:
            if stream.inbound_closed:
                raise TunnelProtocolError("data after directional close")
            if stream.queued_bytes + len(payload) > MAX_STREAM_BUFFER_BYTES:
                await self._send_close(stream_id)
                self._abort_stream(stream_id)
                return
            stream.queued.append(payload)
            stream.queued_bytes += len(payload)
            stream.data_ready.set()
            return
        stream.inbound_closed = True
        stream.data_ready.set()
        await self._discard_if_finished(stream_id)

    async def _send_close(self, stream_id: int) -> None:
        stream = self.streams.get(stream_id)
        if stream is None or stream.outbound_closed:
            return
        stream.outbound_closed = True
        with contextlib.suppress(ConnectionError, OSError, RuntimeError, TimeoutError):
            await self._send(TunnelMessageType.CLOSE, stream_id)
        await self._discard_if_finished(stream_id)

    async def _discard_if_finished(self, stream_id: int) -> None:
        stream = self.streams.get(stream_id)
        if stream is not None and stream.inbound_closed and stream.outbound_closed:
            self._abort_stream(stream_id)

    def _abort_stream(self, stream_id: int) -> None:
        stream = self.streams.pop(stream_id, None)
        if stream is None:
            return
        for task in stream.tasks:
            task.cancel()
        stream.writer.transport.abort()

    async def _send(self, message_type: TunnelMessageType, stream_id: int = 0, payload: bytes = b"") -> None:
        frame = encode_frame(message_type, stream_id, payload)

        async def send() -> None:
            async with self.send_lock:
                await self.websocket.send_bytes(frame)

        await asyncio.wait_for(send(), timeout=SEND_TIMEOUT_SECONDS)

    async def _heartbeat(self) -> None:
        while not self.closed.is_set():
            await asyncio.sleep(HEARTBEAT_SECONDS)
            if time.monotonic() - self.last_pong > HEARTBEAT_SECONDS * MISSED_HEARTBEATS:
                await self.close()
                return
            try:
                await self._send(TunnelMessageType.PING)
            except (ConnectionError, OSError, TimeoutError):
                await self.close()
                return

    async def close(self, code: int = 1000) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        if self.server is not None:
            self.server.close()
        current_task = asyncio.current_task()
        for task in self.tasks:
            if task is not current_task:
                task.cancel()
        for stream_id in list(self.streams):
            self._abort_stream(stream_id)
        if self.server is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.server.wait_closed(), timeout=1)
        await self.manager._remove(self)
        with contextlib.suppress(Exception):
            await self.websocket.close(code=code)


class RelayManager:
    """Owns transient per-pairing localhost listeners in one application worker."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._revoked: set[str] = set()
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        workers = os.environ.get("WEB_CONCURRENCY") or os.environ.get("UVICORN_WORKERS") or "1"
        try:
            worker_count = int(workers)
        except ValueError as error:
            raise RelayUnavailable("WEB_CONCURRENCY must be a positive integer when remote tunnels are enabled") from error
        if worker_count != 1:
            raise RelayUnavailable("remote tunnels require exactly one application worker")
        self._started = True

    async def stop(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._started = False
        await asyncio.gather(*(session.close() for session in sessions), return_exceptions=True)

    async def attach(self, pairing_id: str, websocket: Any) -> None:
        if not self._started:
            raise RelayUnavailable("remote relay has not been started")
        session = _Session(self, pairing_id, websocket)
        try:
            await session.start()
        except Exception:
            await session.close(code=1011)
            raise
        async with self._lock:
            rejected = not self._started or pairing_id in self._revoked
            previous = None if rejected else self._sessions.get(pairing_id)
            if not rejected:
                self._sessions[pairing_id] = session
        if rejected:
            await session.close(code=4403)
            return
        if previous is not None:
            await previous.close()
        await session.run()

    def endpoint_for(self, pairing_id: str) -> tuple[str, int] | None:
        session = self._sessions.get(pairing_id)
        return session.endpoint() if session is not None else None

    async def revoke(self, pairing_id: str) -> None:
        async with self._lock:
            self._revoked.add(pairing_id)
            session = self._sessions.pop(pairing_id, None)
        if session is not None:
            await session.close()

    async def _remove(self, session: _Session) -> None:
        async with self._lock:
            if self._sessions.get(session.pairing_id) is session:
                self._sessions.pop(session.pairing_id, None)


relay_manager = RelayManager()
