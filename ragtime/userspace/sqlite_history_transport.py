"""Lifecycle-safe streaming helpers for SQLite history proxy responses."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import StreamingResponse


class ClosingStreamingResponse(StreamingResponse):
    """Run cleanup even when ASGI never starts the response body."""

    def __init__(self, *args: Any, cleanup: Callable[[], Awaitable[None]], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup
        self._cleaned = False

    async def _close(self) -> None:
        if not self._cleaned:
            self._cleaned = True
            await self._cleanup()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._close()
