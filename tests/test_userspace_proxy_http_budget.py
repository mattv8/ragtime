from __future__ import annotations

import asyncio
import time
import unittest
import warnings
from unittest import mock

import httpx
from starlette.requests import Request

import ragtime.userspace.runtime_routes as routes


def _request(*, body: bytes = b"", budget: str | None = None, content_length: str | None = None) -> Request:
    headers = [(b"host", b"workspace.preview.test"), (b"content-length", (content_length or str(len(body))).encode())]
    if budget is not None:
        headers.append((b"x-ragtime-internal-http-budget-ms", budget.encode()))
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/", "headers": headers}, receive)


class ProxyHttpBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_plane_replaces_spoofed_budget_and_streams_body(self) -> None:
        seen_headers: dict[str, str] = {}
        seen_body = b""

        async def upstream(request: httpx.Request) -> httpx.Response:
            nonlocal seen_body
            seen_headers.update(dict(request.headers))
            seen_body = await request.aread()
            return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"ok")

        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            with (
                mock.patch.object(routes, "_get_proxy_client", return_value=client),
                mock.patch.object(routes, "get_http_proxy_safe_timeout_seconds", new=mock.AsyncMock(return_value=1.0)),
            ):
                response = await routes._proxy_http_request(_request(body=b"payload", budget="999999999"), "http://worker.test/")
            self.assertEqual(seen_body, b"payload")
            self.assertIn("x-ragtime-internal-http-budget-ms", seen_headers)
            self.assertLessEqual(int(seen_headers["x-ragtime-internal-http-budget-ms"]), 1000)
            self.assertEqual(response.status_code, 200)
        finally:
            await client.aclose()

    async def test_expired_budget_returns_structured_gateway_timeout(self) -> None:
        async def slow_send(*_args: object, **_kwargs: object) -> httpx.Response:
            await __import__("asyncio").sleep(0.02)
            return httpx.Response(200)

        client = mock.Mock()
        client.build_request.return_value = object()
        client.send = slow_send
        with (
            mock.patch.object(routes, "_get_proxy_client", return_value=client),
            mock.patch.object(routes, "get_http_proxy_safe_timeout_seconds", new=mock.AsyncMock(return_value=0.001)),
        ):
            response = await routes._proxy_http_request(_request(), "http://worker.test/")
        self.assertEqual(response.status_code, 504)
        self.assertIn(b"timed out", response.body)

    async def test_request_stream_is_chunk_bounded(self) -> None:
        chunks = [chunk async for chunk in routes._bounded_request_stream(_request(body=b"x" * 130_000), 10**12)]
        self.assertEqual(b"".join(chunks), b"x" * 130_000)
        self.assertLessEqual(max(map(len, chunks)), 64 * 1024)

    async def test_request_stream_rejects_invalid_content_length_framing(self) -> None:
        for length in ("3", "8", "not-a-number"):
            with self.subTest(length=length):
                with self.assertRaises(routes._InvalidProxyRequestBody):
                    _ = [chunk async for chunk in routes._bounded_request_stream(_request(body=b"payload", content_length=length), 10**12)]

    async def test_expired_awaitable_is_closed_without_runtime_warning(self) -> None:
        async def pending() -> None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(routes._ProxyDeadlineExceeded):
                await routes._within_proxy_deadline(pending(), time.monotonic() - 1)

    async def test_expired_before_asgi_headers_returns_structured_timeout_without_reading_body(self) -> None:
        closed = False

        async def close_upstream() -> None:
            nonlocal closed
            closed = True

        response = routes._CancellationSafeStreamingResponse(iter([b"app-bytes"]), deadline=time.monotonic() - 1, on_close=close_upstream)
        sent: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        self.assertEqual(sent[0]["status"], 504)
        timeout_body = sent[1]["body"]
        assert isinstance(timeout_body, bytes)
        self.assertIn(b"timed out", timeout_body)
        self.assertTrue(closed)

    async def test_expired_after_headers_terminates_without_injecting_timeout_body(self) -> None:
        closed = False

        async def close_upstream() -> None:
            nonlocal closed
            closed = True

        response = routes._CancellationSafeStreamingResponse(iter([b"app-bytes"]), deadline=time.monotonic() + 1, on_close=close_upstream)
        sent: list[dict[str, object]] = []

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.start":
                response._deadline = time.monotonic() - 1

        async def receive() -> dict[str, object]:
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        self.assertEqual(sent, [sent[0]])
        self.assertEqual(sent[0]["status"], 200)
        self.assertTrue(closed)
