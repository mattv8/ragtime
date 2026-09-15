from __future__ import annotations

import asyncio
import unittest

import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse

from runtime.worker import api


def _request(*, budget: str | None = None, body: bytes = b"", content_length: str | None = None) -> Request:
    headers = [(b"host", b"worker.test")]
    if content_length is not None:
        headers.append((b"content-length", content_length.encode()))
    if budget is not None:
        headers.append((b"x-ragtime-internal-http-budget-ms", budget.encode()))

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers}, receive)


class RuntimeProxyStreamingTests(unittest.IsolatedAsyncioTestCase):
    def test_worker_budget_rejects_nonfinite_and_clamps_large_values(self) -> None:
        self.assertEqual(api._worker_proxy_budget_seconds(_request(budget="NaN")), 90.0)
        self.assertEqual(api._worker_proxy_budget_seconds(_request(budget="-1")), 90.0)
        self.assertEqual(api._worker_proxy_budget_seconds(_request(budget="999999999")), 3420.0)

    def test_internal_budget_never_reaches_devserver(self) -> None:
        headers = api._preview_request_headers(_request(budget="123"))
        self.assertNotIn("x-ragtime-internal-http-budget-ms", headers)

    async def test_worker_preserves_raw_gzip_response_metadata(self) -> None:
        raw = b"not-decoded-gzip-bytes"

        class RawStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield raw

            async def aclose(self) -> None:
                return None

        async def upstream(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html", "content-encoding": "gzip", "content-length": str(len(raw))}, stream=RawStream())

        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        try:
            from unittest import mock

            with mock.patch.object(api, "get_preview_http_client", return_value=client):
                response = await api._proxy_preview_request(_request(), "http://app.test/")
            self.assertIsInstance(response, StreamingResponse)
            assert isinstance(response, StreamingResponse)
            self.assertEqual(response.headers["content-encoding"], "gzip")
            self.assertEqual(response.headers["content-length"], str(len(raw)))
            body = bytearray()
            async for chunk in response.body_iterator:
                body.extend(chunk.encode() if isinstance(chunk, str) else bytes(chunk))
            self.assertEqual(bytes(body), raw)
        finally:
            await client.aclose()

    async def test_slow_asgi_downstream_send_respects_deadline(self) -> None:
        deadline = asyncio.get_running_loop().time() + 0.001
        response = api._DeadlineStreamingResponse(iter([b"app-bytes"]), deadline=deadline)
        messages: list[object] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        async def send(message: object) -> None:
            await asyncio.sleep(0.02)
            messages.append(message)

        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        self.assertEqual(messages, [])

    async def test_worker_rejects_truncated_oversized_and_malformed_framing(self) -> None:
        for length in ("3", "8", "bogus"):
            with self.subTest(length=length):
                with self.assertRaises(api._InvalidProxyRequestBody):
                    _ = [chunk async for chunk in api._bounded_request_stream(_request(body=b"payload", content_length=length), 10**12)]
