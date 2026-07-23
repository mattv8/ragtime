import ipaddress
import unittest

import httpx

from ragtime.http_api.security import (
    HTTP_API_BLOCKED_HEADER_NAMES,
    HttpApiSecurityError,
    PinnedAsyncTransport,
    build_pinned_base_url,
    build_trusted_configured_headers,
    normalize_relative_path,
    validate_resolved_addresses,
)


class HttpApiSecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_normalize_relative_path_rejects_origin_replacement_targets(self) -> None:
        for value in (
            "https://evil.example/path",
            "//evil.example/path",
            "/safe#frag",
            "/%2f%2fevil.example/path",
            "/%2e%2e//evil.example/path",
            "/user@evil.example/path",
        ):
            with self.subTest(value=value):
                with self.assertRaises(HttpApiSecurityError):
                    normalize_relative_path(value)

    def test_normalize_relative_path_accepts_relative_target(self) -> None:
        self.assertEqual(normalize_relative_path("v1/items?limit=2"), "/v1/items?limit=2")
        self.assertEqual(normalize_relative_path("/v1//items?next=https://example.com/a//b"), "/v1//items?next=https://example.com/a//b")

    def test_normalize_relative_path_rejects_traversal_backslash_and_encoded_origin_prefix_only(self) -> None:
        for value in (
            "/../etc/passwd",
            "/a/../b",
            r"/a\b",
            "/%5cwindows",
            "/%2F%2Fevil.example/path",
        ):
            with self.subTest(value=value):
                with self.assertRaises(HttpApiSecurityError):
                    normalize_relative_path(value)

    def test_validate_resolved_addresses_blocks_private_addresses_in_production(self) -> None:
        with self.assertRaises(HttpApiSecurityError):
            validate_resolved_addresses(["127.0.0.1"], debug_mode=False)

        resolved = validate_resolved_addresses(["127.0.0.1", "::1"], debug_mode=True)
        self.assertEqual(resolved, [ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")])

    async def test_pinned_transport_preserves_host_and_sni_while_connecting_to_pinned_address(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["host"] = request.url.host
            captured["header_host"] = request.headers.get("host")
            captured["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, json={"ok": True})

        transport = PinnedAsyncTransport(
            httpx.MockTransport(handler),
            pinned_host="203.0.113.10",
            original_host="api.example.com",
            original_port=443,
        )
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get(build_pinned_base_url("https://api.example.com", "203.0.113.10") + "/v1/items")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["host"], "203.0.113.10")
        self.assertEqual(captured["header_host"], "api.example.com")
        self.assertEqual(captured["sni"], "api.example.com")

    async def test_pinned_transport_and_url_support_ipv6(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["host"] = request.url.host
            captured["header_host"] = request.headers.get("host")
            captured["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, json={"ok": True})

        transport = PinnedAsyncTransport(
            httpx.MockTransport(handler),
            pinned_host="2001:db8::10",
            original_host="api.example.com",
            original_port=443,
        )
        base_url = build_pinned_base_url("https://api.example.com/base", "2001:db8::10")
        self.assertEqual(base_url, "https://[2001:db8::10]/base")

        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get(base_url + "/v1/items")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["host"], "2001:db8::10")
        self.assertEqual(captured["header_host"], "api.example.com")
        self.assertEqual(captured["sni"], "api.example.com")

    def test_blocked_headers_cover_auth_proxy_and_hop_by_hop_headers(self) -> None:
        for header in (
            "host",
            "authorization",
            "cookie",
            "proxy-authorization",
            "content-length",
            "transfer-encoding",
            "forwarded",
            "x-forwarded-for",
            "connection",
        ):
            self.assertIn(header, HTTP_API_BLOCKED_HEADER_NAMES)

    def test_build_trusted_configured_headers_allows_admin_auth_headers(self) -> None:
        headers = build_trusted_configured_headers(
            [
                {"name": "Authorization", "value": "Bearer fake-token"},
                {"name": "Authentication", "value": "Bearer fake-token"},
                {"name": "X-Tenant", "value": "tenant-a"},
            ]
        )

        self.assertEqual(
            headers,
            {
                "Authorization": "Bearer fake-token",
                "Authentication": "Bearer fake-token",
                "X-Tenant": "tenant-a",
            },
        )

    def test_build_trusted_configured_headers_rejects_blocked_and_blank_values(self) -> None:
        for rows in (
            [{"name": "Host", "value": "api.example.com"}],
            [{"name": "Cookie", "value": "session=fake"}],
            [{"name": "Content-Length", "value": "12"}],
            [{"name": "X-Forwarded-For", "value": "203.0.113.10"}],
            [{"name": "Authorization", "value": ""}],
        ):
            with self.subTest(rows=rows):
                with self.assertRaises(HttpApiSecurityError):
                    build_trusted_configured_headers(rows)

    def test_build_trusted_configured_headers_rejects_raw_none_and_crlf_values(self) -> None:
        invalid_rows = (
            [{"name": None, "value": "x"}],
            [{"name": "X-Test", "value": None}],
            [{"name": "Bad\r\nHeader", "value": "x"}],
            [{"name": "X-Test", "value": "line1\nline2"}],
        )

        for rows in invalid_rows:
            with self.subTest(rows=rows):
                with self.assertRaises(HttpApiSecurityError):
                    build_trusted_configured_headers(rows)


if __name__ == "__main__":
    unittest.main()
