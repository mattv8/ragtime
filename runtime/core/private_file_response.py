"""Streaming response for runtime-private files with reliable async cleanup."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import AsyncContextManager, TypeVar

from starlette.responses import Response, StreamingResponse

_T = TypeVar("_T")


class _PrivateFileResponse(StreamingResponse):
    def __init__(
        self,
        path: Path,
        *,
        media_type: str,
        filename: str | None,
        cleanup: Callable[[], Awaitable[None]] | None,
        lifetime: Callable[[], AsyncContextManager[None]] | None,
    ) -> None:
        self._path, self._cleanup, self._lifetime = path, cleanup, lifetime
        super().__init__(self._stream(), media_type=media_type)
        if filename is not None:
            self.headers["content-disposition"] = f'attachment; filename="{filename}"'

    @staticmethod
    async def _drain(task: asyncio.Task[_T]) -> _T:
        """Wait through repeated cancellation until a thread operation ends."""
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        return task.result()

    async def _stream(self) -> AsyncIterator[bytes]:
        opening = asyncio.create_task(asyncio.to_thread(self._path.open, "rb"))
        try:
            source = await asyncio.shield(opening)
        except asyncio.CancelledError:
            source = await self._drain(opening)
            closing = asyncio.create_task(asyncio.to_thread(source.close))
            await self._drain(closing)
            raise
        try:
            while True:
                reading = asyncio.create_task(asyncio.to_thread(source.read, 64 * 1024))
                try:
                    chunk = await asyncio.shield(reading)
                except asyncio.CancelledError:
                    await self._drain(reading)
                    raise
                if not chunk:
                    break
                yield chunk
        finally:
            closing = asyncio.create_task(asyncio.to_thread(source.close))
            await self._drain(closing)

    async def __call__(self, scope, receive, send) -> None:
        # StreamingResponse owns h11 framing and disconnect monitoring.  In
        # particular it avoids Response(content=b"")'s Content-Length: 0.
        lifetime = self._lifetime() if self._lifetime is not None else contextlib.nullcontext()
        try:
            async with lifetime:
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Starlette cancels body iteration on a disconnect.  Close
                    # it explicitly while the liveness pin is still held;
                    # async-generator finalization alone is not sufficient for
                    # an interrupted off-loop read.
                    close = getattr(self.body_iterator, "aclose", None)
                    if close is not None:
                        iterator_close = asyncio.create_task(close())
                        await self._drain(iterator_close)
        finally:
            if self._cleanup is not None:
                await asyncio.shield(self._cleanup())


def private_file_response(
    path: Path,
    *,
    media_type: str,
    filename: str | None = None,
    cleanup: Callable[[], Awaitable[None]] | None = None,
    lifetime: Callable[[], AsyncContextManager[None]] | None = None,
) -> Response:
    """Stream ``path`` with optional ASGI-lifetime resource pin and cleanup.

    ``lifetime`` is entered only when ASGI starts, so download liveness pins do
    not leak if a route constructs a response that is never called.
    """
    return _PrivateFileResponse(path, media_type=media_type, filename=filename, cleanup=cleanup, lifetime=lifetime)
