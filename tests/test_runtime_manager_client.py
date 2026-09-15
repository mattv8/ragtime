from __future__ import annotations

import asyncio
import functools
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import HTTPException

from ragtime.core import runtime_manager_client


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "userspace_runtime_manager_url": "http://runtime.example",
        "userspace_runtime_auth_token": "first-token",
        "userspace_runtime_manager_timeout_seconds": 12.0,
        "userspace_runtime_manager_retry_attempts": 3,
        "userspace_runtime_manager_retry_delay_seconds": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeClient:
    def __init__(self, responses: list[object], **kwargs: object) -> None:
        self.responses = responses
        self.kwargs = kwargs
        self.requests: list[dict[str, object]] = []
        self.closed = False

    async def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        self.requests.append({"method": method, "url": url, **kwargs})
        next_response = self.responses.pop(0)
        if isinstance(next_response, BaseException):
            raise next_response
        return next_response  # type: ignore[return-value]

    async def aclose(self) -> None:
        self.closed = True


class RuntimeManagerClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await runtime_manager_client.close_runtime_manager_client()
        self.addAsyncCleanup(runtime_manager_client.close_runtime_manager_client)

    async def test_reuses_pool_and_resolves_token_for_each_request(self) -> None:
        configured_settings = _settings()
        clients: list[_FakeClient] = []

        def new_client(**kwargs: object) -> _FakeClient:
            client = _FakeClient([httpx.Response(200, json={"first": True}), httpx.Response(200, json={"second": True})], **kwargs)
            clients.append(client)
            return client

        with (
            mock.patch.object(runtime_manager_client, "settings", configured_settings),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", side_effect=new_client),
        ):
            self.assertEqual(await runtime_manager_client.runtime_manager_request("GET", "/status"), {"first": True})
            configured_settings.userspace_runtime_auth_token = "rotated-token"
            self.assertEqual(await runtime_manager_client.runtime_manager_request("GET", "/status"), {"second": True})

        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0].requests[0]["headers"], {"Authorization": "Bearer first-token"})
        self.assertEqual(clients[0].requests[1]["headers"], {"Authorization": "Bearer rotated-token"})

    async def test_preserves_post_retry_and_error_behavior(self) -> None:
        client = _FakeClient([httpx.ConnectError("offline"), httpx.Response(503, text="unavailable"), httpx.Response(200, json={"ok": True})])
        with (
            mock.patch.object(runtime_manager_client, "settings", _settings()),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", return_value=client),
            mock.patch.object(runtime_manager_client.asyncio, "sleep", new=mock.AsyncMock()) as sleep,
        ):
            result = await runtime_manager_client.runtime_manager_request("POST", "/sessions", json_payload={"workspace_id": "one"})

        self.assertEqual(result, {"ok": True})
        self.assertEqual([request["method"] for request in client.requests], ["POST", "POST", "POST"])
        self.assertEqual(sleep.await_count, 2)

        await runtime_manager_client.close_runtime_manager_client()
        failed_client = _FakeClient([httpx.Response(401, text="denied")])
        with (
            mock.patch.object(runtime_manager_client, "settings", _settings(retry_attempts=1)),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", return_value=failed_client),
        ):
            with self.assertRaises(HTTPException) as raised:
                await runtime_manager_client.runtime_manager_request("POST", "/sessions", request_failed_detail_prefix="Manager failed")

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(raised.exception.detail, "Manager failed (401): denied")

    async def test_per_request_timeout_override_does_not_recreate_client(self) -> None:
        client = _FakeClient([httpx.Response(200, json={}), httpx.Response(200, json={})])
        with (
            mock.patch.object(runtime_manager_client, "settings", _settings()),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", return_value=client),
        ):
            await runtime_manager_client.runtime_manager_request("GET", "/one")
            await runtime_manager_client.runtime_manager_request("GET", "/two", timeout_override_seconds=1.5)

        first_timeout = client.requests[0]["timeout"]
        second_timeout = client.requests[1]["timeout"]
        assert isinstance(first_timeout, httpx.Timeout)
        assert isinstance(second_timeout, httpx.Timeout)
        self.assertEqual(first_timeout.connect, 12.0)
        self.assertEqual(second_timeout.connect, 1.5)

    async def test_shutdown_closes_current_loop_client_and_next_request_recreates_it(self) -> None:
        clients: list[_FakeClient] = []

        def new_client(**kwargs: object) -> _FakeClient:
            client = _FakeClient([httpx.Response(200, json={"ok": True})], **kwargs)
            clients.append(client)
            return client

        with (
            mock.patch.object(runtime_manager_client, "settings", _settings()),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", side_effect=new_client),
        ):
            await runtime_manager_client.runtime_manager_request("GET", "/one")
            await runtime_manager_client.close_runtime_manager_client()
            await runtime_manager_client.runtime_manager_request("GET", "/two")

        self.assertEqual(len(clients), 2)
        self.assertTrue(clients[0].closed)
        self.assertFalse(clients[1].closed)

    def test_clients_are_not_reused_across_event_loops(self) -> None:
        clients: list[_FakeClient] = []

        def new_client(**kwargs: object) -> _FakeClient:
            client = _FakeClient([httpx.Response(200, json={"ok": True})], **kwargs)
            clients.append(client)
            return client

        async def request_from_new_loop() -> None:
            await runtime_manager_client.runtime_manager_request("GET", "/status")
            await runtime_manager_client.close_runtime_manager_client()

        with (
            mock.patch.object(runtime_manager_client, "settings", _settings()),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", side_effect=new_client),
        ):
            asyncio.run(request_from_new_loop())
            asyncio.run(request_from_new_loop())

        self.assertEqual(len(clients), 2)
        self.assertTrue(all(client.closed for client in clients))

    async def test_response_cookies_are_not_replayed_and_redirects_remain_safe(self) -> None:
        real_async_client = httpx.AsyncClient
        received: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(request)
            if request.url.path == "/sets-cookie":
                return httpx.Response(200, headers={"set-cookie": "upstream_session=secret; Path=/"}, json={})
            if request.url.path == "/redirect":
                return httpx.Response(302, headers={"location": "https://other.example/final"})
            return httpx.Response(200, json={"ok": True})

        factory = functools.partial(real_async_client, transport=httpx.MockTransport(handler))
        with (
            mock.patch.object(runtime_manager_client, "settings", _settings()),
            mock.patch.object(runtime_manager_client.httpx, "AsyncClient", side_effect=factory),
        ):
            await runtime_manager_client.runtime_manager_request("GET", "/sets-cookie")
            await runtime_manager_client.runtime_manager_request("GET", "/next")
            self.assertEqual(await runtime_manager_client.runtime_manager_request("GET", "/redirect"), {"ok": True})

        next_request = next(request for request in received if request.url.path == "/next")
        redirected_request = next(request for request in received if request.url.host == "other.example")
        self.assertNotIn("cookie", next_request.headers)
        self.assertNotIn("authorization", redirected_request.headers)


if __name__ == "__main__":
    unittest.main()
