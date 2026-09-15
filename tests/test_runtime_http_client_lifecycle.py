import asyncio
import unittest
from unittest import mock

import httpx
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse

from runtime.manager import api as manager_api
from runtime.worker import api as worker_api
from runtime.worker.http_client import RejectResponseCookies, WorkerProxyHTTPClientPool


def _request(*, cookie: str | None = None) -> Request:
    headers = [(b"host", b"preview.test"), (b"accept", b"application/octet-stream")]
    if cookie:
        headers.append((b"cookie", cookie.encode()))

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": headers},
        receive=receive,
    )


async def _body(response: StreamingResponse) -> bytes:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunks.append(chunk)
        elif isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(bytes(chunk))
    return b"".join(chunks)


class RuntimeProxyCookieIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_cookies_are_not_replayed_but_explicit_cookies_are_forwarded(self) -> None:
        received_cookies: list[str | None] = []

        class OneChunkStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"ok"

            async def aclose(self) -> None:
                pass

        def upstream(request: httpx.Request) -> httpx.Response:
            received_cookies.append(request.headers.get("cookie"))
            headers = {"content-type": "application/octet-stream"}
            if len(received_cookies) == 1:
                headers["set-cookie"] = "upstream-session=first; Path=/"
            return httpx.Response(200, headers=headers, stream=OneChunkStream())

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream),
            cookies=RejectResponseCookies(),
        )
        with mock.patch.object(worker_api, "get_preview_http_client", return_value=client):
            first = await worker_api._proxy_preview_request(_request(), "http://upstream.test/")
            assert isinstance(first, StreamingResponse)
            self.assertEqual(await _body(first), b"ok")
            second = await worker_api._proxy_preview_request(_request(), "http://upstream.test/")
            assert isinstance(second, StreamingResponse)
            self.assertEqual(await _body(second), b"ok")
            explicit = await worker_api._proxy_preview_request(_request(cookie="explicit=value"), "http://upstream.test/")
            assert isinstance(explicit, StreamingResponse)
            self.assertEqual(await _body(explicit), b"ok")
        await client.aclose()

        self.assertEqual(received_cookies, [None, None, "explicit=value"])
        self.assertIn("__ragtime_app_cookie_", first.headers["set-cookie"])

    async def test_stream_cancellation_closes_response_without_closing_shared_client(self) -> None:
        stream_closed = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"first"
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                stream_closed.set()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"content-type": "application/octet-stream"},
                    stream=BlockingStream(),
                )
            ),
            cookies=RejectResponseCookies(),
        )
        with mock.patch.object(worker_api, "get_preview_http_client", return_value=client):
            response = await worker_api._proxy_preview_request(_request(), "http://upstream.test/")
            assert isinstance(response, StreamingResponse)
            iterator = response.body_iterator  # type: ignore[attr-defined]
            self.assertEqual(await iterator.__anext__(), b"first")  # type: ignore[attr-defined]
            await iterator.aclose()  # type: ignore[attr-defined]

        self.assertTrue(stream_closed.is_set())
        self.assertFalse(client.is_closed)
        await client.aclose()


class RuntimeProxyClientPoolLifecycleTests(unittest.TestCase):
    def test_pool_isolated_per_event_loop_and_closes_all_clients(self) -> None:
        clients: list[httpx.AsyncClient] = []

        def new_client() -> httpx.AsyncClient:
            client = httpx.AsyncClient(cookies=RejectResponseCookies())
            clients.append(client)
            return client

        pool = WorkerProxyHTTPClientPool(new_client)

        async def use_pool() -> None:
            pool.get_client()

        asyncio.run(use_pool())
        asyncio.run(use_pool())
        asyncio.run(pool.close())

        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.is_closed for client in clients))


class RuntimeWorkerShutdownLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_worker_lifespan_uses_shared_shutdown_helper(self) -> None:
        with mock.patch.object(worker_api, "shutdown_worker_resources", new_callable=mock.AsyncMock) as shutdown:
            async with worker_api._worker_lifespan(FastAPI()):
                pass
        shutdown.assert_awaited_once()

    async def test_combined_manager_lifespan_uses_worker_shutdown_helper(self) -> None:
        manager = mock.AsyncMock()
        manager.startup = mock.AsyncMock()
        manager.shutdown = mock.AsyncMock()
        with (
            mock.patch.object(manager_api, "SessionManager", return_value=manager),
            mock.patch("runtime.worker.api.shutdown_worker_resources", new_callable=mock.AsyncMock) as shutdown,
        ):
            application = manager_api.create_app()
            async with application.router.lifespan_context(application):
                pass

        manager.shutdown.assert_awaited_once()
        shutdown.assert_awaited_once()
