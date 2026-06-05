from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from starlette.requests import Request
from starlette.responses import StreamingResponse

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES

_should_clear_site_data_for_proxy_response = getattr(
    _RUNTIME_ROUTES,
    "_should_clear_site_data_for_proxy_response",
)
_CancellationSafeStreamingResponse = getattr(
    _RUNTIME_ROUTES,
    "_CancellationSafeStreamingResponse",
)


def _build_request(scheme: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": headers or [(b"host", b"workspace.userspace-preview.lvh.me:8000")],
            "scheme": scheme,
            "server": ("workspace.userspace-preview.lvh.me", 8000),
            "client": ("127.0.0.1", 12345),
        }
    )


class UserSpaceRuntimeProxyHeaderTests(unittest.TestCase):
    def test_does_not_clear_site_data_for_local_http_preview_html(self) -> None:
        request = _build_request("http")

        self.assertFalse(_should_clear_site_data_for_proxy_response(request, "text/html; charset=utf-8"))

    def test_clears_site_data_for_https_preview_html(self) -> None:
        request = _build_request("https")

        self.assertTrue(_should_clear_site_data_for_proxy_response(request, "text/html"))

    def test_clears_site_data_when_forwarded_proto_is_https(self) -> None:
        request = _build_request(
            "http",
            [
                (b"host", b"workspace.example.test"),
                (b"x-forwarded-proto", b"https"),
            ],
        )

        self.assertTrue(_should_clear_site_data_for_proxy_response(request, "text/html"))

    def test_does_not_clear_site_data_for_non_html_over_https(self) -> None:
        request = _build_request("https")

        self.assertFalse(_should_clear_site_data_for_proxy_response(request, "application/javascript"))


class CancellationSafeStreamingResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_swallows_cancelled_error_from_streaming_response_call(self) -> None:
        response = _CancellationSafeStreamingResponse(iter([b"chunk"]))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [],
        }
        messages: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            raise asyncio.CancelledError()

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await response(scope, receive, send)

        self.assertTrue(messages)

    async def test_swallows_cancellation_only_exception_group(self) -> None:
        response = _CancellationSafeStreamingResponse(iter([b"chunk"]))

        async def raise_group(*_args: object, **_kwargs: object) -> None:
            raise BaseExceptionGroup(
                "shutdown",
                [asyncio.CancelledError("timeout graceful shutdown exceeded")],
            )

        with mock.patch.object(StreamingResponse, "__call__", raise_group):
            await response({}, None, None)

    async def test_reraises_mixed_exception_group(self) -> None:
        response = _CancellationSafeStreamingResponse(iter([b"chunk"]))

        async def raise_group(*_args: object, **_kwargs: object) -> None:
            raise BaseExceptionGroup(
                "mixed",
                [asyncio.CancelledError("shutdown"), RuntimeError("real failure")],
            )

        with mock.patch.object(StreamingResponse, "__call__", raise_group):
            with self.assertRaises(BaseExceptionGroup):
                await response({}, None, None)


if __name__ == "__main__":
    unittest.main()
