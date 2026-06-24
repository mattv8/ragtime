from __future__ import annotations

import asyncio
import base64
import unittest
from unittest import mock

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
import runtime.worker.api as _WORKER_API

_should_clear_site_data_for_proxy_response = getattr(
    _RUNTIME_ROUTES,
    "_should_clear_site_data_for_proxy_response",
)
_CancellationSafeStreamingResponse = getattr(
    _RUNTIME_ROUTES,
    "_CancellationSafeStreamingResponse",
)
_proxy_request_headers = getattr(_RUNTIME_ROUTES, "_proxy_request_headers")
_proxy_response_headers = getattr(_RUNTIME_ROUTES, "_proxy_response_headers")
_worker_preview_request_headers = getattr(_WORKER_API, "_preview_request_headers")
_worker_preview_response_headers = getattr(_WORKER_API, "_preview_response_headers")


def _browser_app_cookie_name(name: str) -> str:
    encoded = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii").rstrip("=")
    return f"__ragtime_app_cookie_{encoded}"


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

    def test_control_plane_proxy_blocks_cookies_by_default(self) -> None:
        request = _build_request(
            "https",
            [
                (b"host", b"workspace.example.test"),
                (b"cookie", f"{_browser_app_cookie_name('custom_bid_manager_session')}=app".encode("utf-8")),
            ],
        )

        headers = _proxy_request_headers(request)

        self.assertNotIn("cookie", {key.lower(): value for key, value in headers.items()})

    def test_control_plane_forwards_only_namespaced_user_app_cookies_when_allowed(self) -> None:
        app_cookie_name = _browser_app_cookie_name("custom_bid_manager_session")
        request = _build_request(
            "https",
            [
                (b"host", b"workspace.example.test"),
                (
                    b"cookie",
                    f"{app_cookie_name}=app; userspace_preview_session=platform; ragtime_session=secret".encode("utf-8"),
                ),
            ],
        )

        # _proxy_request_headers trusts its gated boolean; the preview-host gate
        # itself lives in _proxy_http_request and is exercised separately below.
        headers = _proxy_request_headers(request, allow_user_cookies=True)

        self.assertEqual(headers.get("cookie"), "custom_bid_manager_session=app")

    def test_control_plane_preview_host_forwards_only_namespaced_user_app_set_cookie(self) -> None:
        app_cookie_name = _browser_app_cookie_name("custom_bid_manager_session")
        upstream_headers = httpx.Headers(
            [
                ("set-cookie", f"{app_cookie_name}=app; Path=/; Secure; SameSite=None"),
                ("set-cookie", "userspace_preview_session=platform; Path=/"),
            ]
        )

        headers, set_cookie_headers = _proxy_response_headers(upstream_headers, allow_user_cookies=True)

        self.assertNotIn("set-cookie", {key.lower(): value for key, value in headers.items()})
        self.assertEqual(set_cookie_headers, [f"{app_cookie_name}=app; Path=/; Secure; SameSite=None"])

    def test_worker_preview_proxy_forwards_cookie_header_from_control_plane(self) -> None:
        request = _build_request(
            "http",
            [
                (b"host", b"runtime-worker.test"),
                (b"authorization", b"Bearer worker-token"),
                (b"cookie", b"custom_bid_manager_session=app"),
            ],
        )

        headers = _worker_preview_request_headers(request)

        self.assertEqual(headers.get("cookie"), "custom_bid_manager_session=app")
        self.assertNotIn("authorization", {key.lower(): value for key, value in headers.items()})

    def test_worker_preview_proxy_namespaces_user_app_set_cookie(self) -> None:
        app_cookie_name = _browser_app_cookie_name("custom_bid_manager_session")
        upstream_headers = httpx.Headers(
            [
                ("set-cookie", "custom_bid_manager_session=app; Path=/; Secure; SameSite=None"),
                ("set-cookie", "userspace_preview_session=app-owned-value; Path=/"),
            ]
        )

        headers, set_cookie_headers = _worker_preview_response_headers(upstream_headers)

        self.assertNotIn("set-cookie", {key.lower(): value for key, value in headers.items()})
        self.assertEqual(
            set_cookie_headers,
            [
                f"{app_cookie_name}=app; Path=/; Secure; SameSite=None",
                f"{_browser_app_cookie_name('userspace_preview_session')}=app-owned-value; Path=/",
            ],
        )


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


class SanitizeUserAppCookieHeaderTests(unittest.TestCase):
    _sanitize = staticmethod(getattr(_RUNTIME_ROUTES, "_sanitize_user_app_cookie_header"))

    def test_returns_none_for_empty_header(self) -> None:
        self.assertIsNone(self._sanitize(None))
        self.assertIsNone(self._sanitize(""))

    def test_returns_none_when_no_namespaced_cookies(self) -> None:
        self.assertIsNone(self._sanitize("userspace_preview_session=x; ragtime_session=y"))

    def test_drops_malformed_namespaced_cookie_name(self) -> None:
        # Prefix present but the encoded segment is not valid base64.
        self.assertIsNone(self._sanitize("__ragtime_app_cookie_!!!=value"))

    def test_decodes_only_namespaced_cookies_and_preserves_value(self) -> None:
        header = f"{_browser_app_cookie_name('sid')}=a=b=c; platform=secret; {_browser_app_cookie_name('csrf')}=tok"
        # The value "a=b=c" exercises split("=", 1) value preservation.
        self.assertEqual(self._sanitize(header), "sid=a=b=c; csrf=tok")


class _FakeUpstreamResponse:
    def __init__(self, headers: httpx.Headers, body: bytes = b"<html><body>ok</body></html>") -> None:
        self.headers = headers
        self.status_code = 200
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aclose(self) -> None:
        return None


class _FakeUpstreamClient:
    def __init__(self, response: _FakeUpstreamResponse) -> None:
        self._response = response
        self.sent_headers: dict[str, str] | None = None

    def build_request(self, *, headers: dict[str, str], **_kwargs: object) -> dict[str, object]:
        self.sent_headers = {key.lower(): value for key, value in headers.items()}
        return {"headers": headers}

    async def send(self, _request: object, stream: bool = False) -> _FakeUpstreamResponse:
        return self._response

    async def aclose(self) -> None:
        return None


def _build_proxy_request(host: str, cookie: str) -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/dashboard",
            "raw_path": b"/dashboard",
            "query_string": b"",
            "headers": [
                (b"host", host.encode("ascii")),
                (b"accept", b"text/html"),
                (b"cookie", cookie.encode("ascii")),
            ],
            "scheme": "https",
            "server": (host, 443),
            "client": ("127.0.0.1", 12345),
        },
        receive,
    )


class ProxyHttpRequestPreviewHostGateTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end gate: app cookies cross the proxy only on a preview host."""

    async def _run(self, *, is_preview_host: bool) -> tuple[_FakeUpstreamClient, Response]:
        app_cookie = f"{_browser_app_cookie_name('sid')}=app-value"
        upstream = _FakeUpstreamResponse(httpx.Headers([("content-type", "text/html"), ("set-cookie", f"{_browser_app_cookie_name('sid')}=rotated; Path=/")]))
        client = _FakeUpstreamClient(upstream)
        request = _build_proxy_request("workspace.preview.test", app_cookie)

        with (
            mock.patch.object(_RUNTIME_ROUTES.httpx, "AsyncClient", return_value=client),
            mock.patch.object(_RUNTIME_ROUTES, "get_app_settings", new=mock.AsyncMock(return_value={"userspace_preview_sandbox_flags": []})),
            mock.patch.object(_RUNTIME_ROUTES, "_is_preview_host_request", return_value=is_preview_host),
        ):
            response = await _RUNTIME_ROUTES._proxy_http_request(
                request,
                "http://runtime/dashboard",
                allow_user_cookies=True,
            )
        return client, response

    async def test_preview_host_forwards_and_returns_app_cookies(self) -> None:
        client, response = await self._run(is_preview_host=True)

        # Decoded app cookie reaches the upstream devserver.
        self.assertEqual((client.sent_headers or {}).get("cookie"), "sid=app-value")
        # Namespaced rotated cookie comes back to the browser.
        set_cookies = [value for key, value in response.raw_headers if key == b"set-cookie"]
        self.assertEqual(set_cookies, [f"{_browser_app_cookie_name('sid')}=rotated; Path=/".encode("latin-1")])

    async def test_non_preview_host_strips_app_cookies_despite_flag(self) -> None:
        client, response = await self._run(is_preview_host=False)

        # No cookie reaches the upstream devserver.
        self.assertNotIn("cookie", client.sent_headers or {})
        # No Set-Cookie is returned to the browser.
        set_cookies = [value for key, value in response.raw_headers if key == b"set-cookie"]
        self.assertEqual(set_cookies, [])


if __name__ == "__main__":
    unittest.main()
