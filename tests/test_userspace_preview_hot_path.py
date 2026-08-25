from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

import httpx
from fastapi import HTTPException
from starlette.requests import Request

import ragtime.userspace.preview_host as _PREVIEW_HOST
import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES


def _build_request(path: str, cookie: str | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = [(b"host", b"workspace-a.ragtime.test")]
    if cookie:
        headers.append((b"cookie", f"userspace_preview_session={cookie}".encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "scheme": "https",
            "server": ("workspace-a.ragtime.test", 443),
            "client": ("127.0.0.1", 1234),
        }
    )


class _CountingRuntimeService:
    def __init__(self, exp_offset_seconds: float = 300.0) -> None:
        self.verify_calls = 0
        self.exp_offset_seconds = exp_offset_seconds

    def verify_preview_token(self, token: str, *, expected_kind: str) -> dict[str, Any]:
        self.verify_calls += 1
        if token == "invalid":
            raise HTTPException(status_code=401, detail="Invalid preview token")
        return {
            "kind": expected_kind,
            "workspace_id": "workspace-a",
            "sub": "user-1",
            "preview_mode": "workspace",
            "exp": time.time() + self.exp_offset_seconds,
        }


class SessionClaimsCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _PREVIEW_HOST._session_claims_cache.clear()

    async def test_repeat_verification_of_same_token_hits_cache(self) -> None:
        fake = _CountingRuntimeService()
        with mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake):
            first = _PREVIEW_HOST._verify_preview_session_token("token-1")
            second = _PREVIEW_HOST._verify_preview_session_token("token-1")
        self.assertEqual(fake.verify_calls, 1)
        self.assertEqual(second["workspace_id"], "workspace-a")
        self.assertIsNot(first, second)

    async def test_expired_cache_entry_reverifies_and_fails_like_today(self) -> None:
        fake = _CountingRuntimeService(exp_offset_seconds=-5.0)
        with mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake):
            _PREVIEW_HOST._verify_preview_session_token("token-2")
            fake.verify_calls = 0

            def now_expired(token: str, *, expected_kind: str) -> dict[str, Any]:
                fake.verify_calls += 1
                raise HTTPException(status_code=401, detail="Invalid preview token")

            fake.verify_preview_token = now_expired  # type: ignore[method-assign]
            with self.assertRaises(HTTPException):
                _PREVIEW_HOST._verify_preview_session_token("token-2")
        self.assertEqual(fake.verify_calls, 1)

    async def test_invalid_token_is_not_cached(self) -> None:
        fake = _CountingRuntimeService()
        with mock.patch.object(_PREVIEW_HOST, "_runtime_service", return_value=fake):
            for _ in range(2):
                with self.assertRaises(HTTPException):
                    _PREVIEW_HOST._verify_preview_session_token("invalid")
        self.assertEqual(fake.verify_calls, 2)

    async def test_missing_token_still_raises_401(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            _PREVIEW_HOST._verify_preview_session_token(None)
        self.assertEqual(raised.exception.status_code, 401)


class PooledProxyClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await _RUNTIME_ROUTES.close_proxy_client()

    async def asyncTearDown(self) -> None:
        await _RUNTIME_ROUTES.close_proxy_client()

    async def test_proxy_client_is_a_reused_singleton(self) -> None:
        first = _RUNTIME_ROUTES._get_proxy_client()
        second = _RUNTIME_ROUTES._get_proxy_client()
        self.assertIs(first, second)
        self.assertFalse(first.is_closed)

    async def test_proxy_client_has_no_total_connection_cap(self) -> None:
        created_client = mock.Mock(is_closed=False)
        created_client.aclose = mock.AsyncMock()  # type: ignore[method-assign]
        with mock.patch.object(_RUNTIME_ROUTES.httpx, "AsyncClient", return_value=created_client) as async_client:
            client = _RUNTIME_ROUTES._get_proxy_client()

        self.assertIs(client, created_client)
        async_client.assert_called_once()
        limits = async_client.call_args.kwargs.get("limits")
        self.assertIsInstance(limits, httpx.Limits)
        assert isinstance(limits, httpx.Limits)
        self.assertIsNone(limits.max_connections)
        self.assertFalse(async_client.call_args.kwargs["follow_redirects"])

    async def test_close_proxy_client_closes_and_resets(self) -> None:
        client = _RUNTIME_ROUTES._get_proxy_client()
        await _RUNTIME_ROUTES.close_proxy_client()
        self.assertTrue(client.is_closed)
        self.assertIsNot(_RUNTIME_ROUTES._get_proxy_client(), client)

    async def test_shutdown_hook_is_registered_on_preview_host_app(self) -> None:
        self.assertIn(
            _RUNTIME_ROUTES.close_proxy_client,
            _PREVIEW_HOST.preview_host_app.router.on_shutdown,
        )


class BridgeContentCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _PREVIEW_HOST._bridge_content_cache.clear()

    async def test_bridge_content_is_cached_within_ttl(self) -> None:
        build = mock.AsyncMock(return_value="// bridge v1")
        fake_service = SimpleNamespace(build_runtime_bridge_content=build)
        claims = {"workspace_id": "workspace-a", "preview_mode": "workspace", "sub": "user-1"}
        with (
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(_PREVIEW_HOST, "_verify_preview_session_cookie", new=mock.AsyncMock(return_value=claims)),
        ):
            first = await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))
            second = await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))
        build.assert_awaited_once_with("workspace-a")
        self.assertEqual(first.body, second.body)
        self.assertEqual(first.headers["cache-control"], "no-store")

    async def test_bridge_content_rebuilds_after_ttl(self) -> None:
        build = mock.AsyncMock(return_value="// bridge v1")
        fake_service = SimpleNamespace(build_runtime_bridge_content=build)
        claims = {"workspace_id": "workspace-a", "preview_mode": "workspace", "sub": "user-1"}
        with (
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(_PREVIEW_HOST, "_verify_preview_session_cookie", new=mock.AsyncMock(return_value=claims)),
        ):
            await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))
            key = next(iter(_PREVIEW_HOST._bridge_content_cache))
            content, _expires = _PREVIEW_HOST._bridge_content_cache[key]
            _PREVIEW_HOST._bridge_content_cache[key] = (content, time.monotonic() - 1.0)
            await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))
        self.assertEqual(build.await_count, 2)

    async def test_bridge_content_cache_misses_when_runtime_bridge_version_changes(self) -> None:
        build = mock.AsyncMock(side_effect=["// bridge v1", "// bridge v2"])
        fake_service = SimpleNamespace(build_runtime_bridge_content=build)
        claims = {"workspace_id": "workspace-a", "preview_mode": "workspace", "sub": "user-1"}
        with (
            mock.patch.object(_PREVIEW_HOST, "_userspace_service", return_value=fake_service),
            mock.patch.object(_PREVIEW_HOST, "_verify_preview_session_cookie", new=mock.AsyncMock(return_value=claims)),
            mock.patch("ragtime.userspace.service._RUNTIME_BRIDGE_VERSION", 18),
        ):
            first = await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))
            with mock.patch("ragtime.userspace.service._RUNTIME_BRIDGE_VERSION", 19):
                second = await _PREVIEW_HOST.preview_bridge_script(_build_request("/__ragtime/bridge.js", cookie="t"))

        self.assertEqual(build.await_count, 2)
        self.assertNotEqual(first.body, second.body)


if __name__ == "__main__":
    unittest.main()
