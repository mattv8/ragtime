from __future__ import annotations

import unittest
from base64 import urlsafe_b64encode

import httpx

import ragtime.userspace.runtime_routes as runtime_routes
from ragtime.core.http_client import RejectResponseCookies


class RejectResponseCookiesTests(unittest.IsolatedAsyncioTestCase):
    async def test_pooled_client_discards_upstream_cookies_but_keeps_explicit_cookies(self) -> None:
        seen_cookies: list[str | None] = []

        async def upstream(request: httpx.Request) -> httpx.Response:
            seen_cookies.append(request.headers.get("cookie"))
            if len(seen_cookies) == 1:
                return httpx.Response(200, headers={"set-cookie": "worker_session=user-a; Path=/"})
            return httpx.Response(200)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(upstream),
            cookies=RejectResponseCookies(),
        ) as client:
            first = client.build_request("GET", "http://workspace-a.test/", headers={"cookie": "app_session=explicit-a"})
            await client.send(first)
            second = client.build_request("GET", "http://workspace-b.test/")
            await client.send(second)
            third = client.build_request("GET", "http://workspace-b.test/", headers={"cookie": "app_session=explicit-b"})
            await client.send(third)

        self.assertEqual(seen_cookies, ["app_session=explicit-a", None, "app_session=explicit-b"])

    async def test_proxy_pool_uses_cookie_rejecting_jar(self) -> None:
        await runtime_routes.close_proxy_client()
        try:
            client = runtime_routes._get_proxy_client()
            self.assertIsInstance(client.cookies.jar, RejectResponseCookies)
        finally:
            await runtime_routes.close_proxy_client()

    async def test_response_keeps_only_namespaced_app_cookies(self) -> None:
        app_cookie_name = "__ragtime_app_cookie_" + urlsafe_b64encode(b"session").decode("ascii").rstrip("=")
        headers, set_cookies = runtime_routes._proxy_response_headers(
            httpx.Headers(
                [
                    ("set-cookie", f"{app_cookie_name}=rotated; Path=/; Secure"),
                    ("set-cookie", "userspace_preview_session=platform; Path=/; Secure"),
                ]
            ),
            allow_user_cookies=True,
        )

        self.assertNotIn("set-cookie", headers)
        self.assertEqual(set_cookies, [f"{app_cookie_name}=rotated; Path=/; Secure"])
