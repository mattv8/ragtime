from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from http.cookiejar import CookieJar

import httpx


class RejectResponseCookies(CookieJar):
    """Cookie jar that deliberately ignores every upstream Set-Cookie header."""

    def extract_cookies(self, response, request) -> None:  # type: ignore[override]
        """Discard every response cookie without storing it."""


class WorkerProxyHTTPClientPool:
    """One connection pool per event loop, without upstream cookie persistence."""

    def __init__(
        self,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._new_client
        # Keep a strong loop reference until explicit shutdown: a weak entry
        # could disappear after a test/reload loop closes, leaking its client.
        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _new_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=30.0, write=30.0, pool=5.0),
            follow_redirects=False,
            cookies=RejectResponseCookies(),
        )

    def get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._lock:
            client = self._clients.get(loop)
            if client is None or client.is_closed:
                client = self._client_factory()
                self._clients[loop] = client
            return client

    async def close(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)


_preview_http_client_pool = WorkerProxyHTTPClientPool()


def get_preview_http_client() -> httpx.AsyncClient:
    return _preview_http_client_pool.get_client()


async def close_preview_http_clients() -> None:
    await _preview_http_client_pool.close()
