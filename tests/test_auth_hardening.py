"""
Tests for scoped auth hardening:

1. GET /v1/models protected by verify_api_key when API_KEY is configured.
2. MCP token endpoint brute-force throttling.
3. MCP OAuth metadata base URL uses EXTERNAL_BASE_URL when configured.
"""

import base64
import json
import unittest
from typing import Optional
from unittest import mock

from ragtime.mcp import oauth

# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------


def _basic(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _scope(
    *,
    method: str = "GET",
    path: str = "/mcp/cowork",
    headers: Optional[dict[bytes, bytes]] = None,
    client: Optional[tuple[str, int]] = None,
) -> dict:
    scope: dict = {
        "type": "http",
        "method": method,
        "scheme": "https",
        "path": path,
        "headers": list((headers or {b"host": b"ragtime.example"}).items()),
    }
    if client is not None:
        scope["client"] = client
    return scope


def _form_receive(body: bytes):
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _capture_response(call) -> tuple[int, dict[str, str], dict]:
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    await call(send)
    status = messages[0]["status"]
    headers = {key.decode("latin1"): value.decode("latin1") for key, value in messages[0].get("headers", [])}
    body = json.loads(messages[-1].get("body", b"{}"))
    return status, headers, body


# ---------------------------------------------------------------------------
# 1. GET /v1/models — verify_api_key dependency
# ---------------------------------------------------------------------------


class ModelsEndpointAuthTests(unittest.TestCase):
    """Verify that verify_api_key is wired to GET /v1/models."""

    def test_verify_api_key_is_dependency_of_list_models(self) -> None:
        """list_models must declare verify_api_key as a FastAPI dependency."""
        # FastAPI stores dependencies on the endpoint function's __wrapped__
        # or on the route object. The simplest portable check is to inspect
        # the route registered on the router.
        from ragtime.api.routes import list_models, router, verify_api_key

        route = next(
            (r for r in router.routes if getattr(r, "path", None) == "/v1/models"),
            None,
        )
        self.assertIsNotNone(route, "GET /v1/models route not found on router")

        # FastAPI stores Depends() objects in route.dependencies
        dep_calls = [dep.dependency for dep in getattr(route, "dependencies", [])]
        self.assertIn(
            verify_api_key,
            dep_calls,
            "verify_api_key must be listed as a dependency of GET /v1/models",
        )

    def test_verify_api_key_raises_when_key_configured_and_missing(self) -> None:
        """verify_api_key raises 401 when API_KEY is set and no Authorization header."""
        import asyncio

        from fastapi import HTTPException

        from ragtime.api.routes import verify_api_key

        with mock.patch("ragtime.api.routes.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(verify_api_key(authorization=None))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_verify_api_key_raises_on_wrong_key(self) -> None:
        """verify_api_key raises 401 when the provided key does not match."""
        import asyncio

        from fastapi import HTTPException

        from ragtime.api.routes import verify_api_key

        with mock.patch("ragtime.api.routes.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            with self.assertRaises(HTTPException) as ctx:
                asyncio.get_event_loop().run_until_complete(verify_api_key(authorization="Bearer wrong-key"))
        self.assertEqual(ctx.exception.status_code, 401)

    def test_verify_api_key_passes_with_correct_bearer(self) -> None:
        """verify_api_key does not raise when the correct Bearer token is supplied."""
        import asyncio

        from ragtime.api.routes import verify_api_key

        with mock.patch("ragtime.api.routes.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            # Should not raise
            asyncio.get_event_loop().run_until_complete(verify_api_key(authorization="Bearer secret-key"))

    def test_verify_api_key_passes_when_no_key_configured(self) -> None:
        """verify_api_key is a no-op when API_KEY is not configured."""
        import asyncio

        from ragtime.api.routes import verify_api_key

        with mock.patch("ragtime.api.routes.settings") as mock_settings:
            mock_settings.api_key = ""
            # Should not raise even without an Authorization header
            asyncio.get_event_loop().run_until_complete(verify_api_key(authorization=None))


# ---------------------------------------------------------------------------
# 2. MCP token endpoint brute-force throttling
# ---------------------------------------------------------------------------


class McpThrottleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # Reset throttle state before each test so tests are independent.
        oauth._auth_failure_timestamps.clear()
        oauth._auth_blocked_until.clear()

    def tearDown(self) -> None:
        oauth._auth_failure_timestamps.clear()
        oauth._auth_blocked_until.clear()

    def test_is_throttled_returns_false_initially(self) -> None:
        self.assertFalse(oauth._is_throttled("1.2.3.4"))

    def test_record_failure_increments_counter(self) -> None:
        oauth._record_auth_failure("1.2.3.4")
        self.assertEqual(len(oauth._auth_failure_timestamps["1.2.3.4"]), 1)

    def test_block_triggered_after_max_failures(self) -> None:
        for _ in range(oauth._THROTTLE_MAX_FAILURES):
            oauth._record_auth_failure("1.2.3.4")
        self.assertTrue(oauth._is_throttled("1.2.3.4"))

    def test_success_clears_failure_counter(self) -> None:
        for _ in range(oauth._THROTTLE_MAX_FAILURES - 1):
            oauth._record_auth_failure("1.2.3.4")
        oauth._record_auth_success("1.2.3.4")
        self.assertFalse(oauth._is_throttled("1.2.3.4"))
        self.assertNotIn("1.2.3.4", oauth._auth_failure_timestamps)

    def test_get_client_ip_prefers_client_tuple_over_x_forwarded_for(self) -> None:
        scope = _scope(
            headers={b"x-forwarded-for": b"10.0.0.1, 192.168.1.1", b"host": b"ragtime.example"},
            client=("172.16.0.5", 54321),
        )
        self.assertEqual(oauth._get_client_ip(scope), "172.16.0.5")

    def test_get_client_ip_uses_x_forwarded_for_without_client_tuple(self) -> None:
        scope = _scope(headers={b"x-forwarded-for": b"10.0.0.1, 192.168.1.1", b"host": b"ragtime.example"})
        self.assertEqual(oauth._get_client_ip(scope), "10.0.0.1")

    def test_get_client_ip_falls_back_to_client_tuple(self) -> None:
        scope = _scope(client=("172.16.0.5", 54321))
        self.assertEqual(oauth._get_client_ip(scope), "172.16.0.5")

    def test_get_client_ip_returns_unknown_when_no_info(self) -> None:
        scope = _scope()
        self.assertEqual(oauth._get_client_ip(scope), "unknown")

    async def test_token_endpoint_returns_429_when_throttled(self) -> None:
        """Token endpoint must return 429 when the source IP is blocked."""
        # Pre-block the IP.
        oauth._auth_blocked_until["1.2.3.4"] = 9_999_999_999.0

        scope = _scope(
            method="POST",
            headers={
                b"host": b"ragtime.example",
                b"authorization": _basic("cid", "secret").encode("ascii"),
            },
            client=("1.2.3.4", 12345),
        )
        status, _headers, body = await _capture_response(
            lambda send: oauth.handle_token_request(
                scope,
                _form_receive(b"grant_type=client_credentials"),
                send,
                "cowork",
            )
        )
        self.assertEqual(status, 429)
        self.assertEqual(body["error"], "too_many_requests")

    async def test_token_endpoint_records_failure_on_bad_credentials(self) -> None:
        """Failed credential check must increment the failure counter."""
        scope = _scope(
            method="POST",
            headers={
                b"host": b"ragtime.example",
                b"authorization": _basic("cid", "wrong").encode("ascii"),
            },
            client=("5.6.7.8", 12345),
        )

        with (
            mock.patch.object(
                oauth,
                "_get_route_client_credentials",
                new=mock.AsyncMock(return_value=("cid", "encrypted-secret")),
            ),
            mock.patch.object(
                oauth,
                "_credentials_match",
                return_value=False,
            ),
        ):
            status, _headers, body = await _capture_response(
                lambda send: oauth.handle_token_request(
                    scope,
                    _form_receive(b"grant_type=client_credentials"),
                    send,
                    "cowork",
                )
            )

        self.assertEqual(status, 401)
        self.assertGreater(len(oauth._auth_failure_timestamps.get("5.6.7.8", [])), 0)

    async def test_token_endpoint_clears_counter_on_success(self) -> None:
        """Successful token issuance must clear the failure counter."""
        # Seed a prior failure.
        oauth._auth_failure_timestamps["9.9.9.9"] = [1.0]

        scope = _scope(
            method="POST",
            headers={
                b"host": b"ragtime.example",
                b"authorization": _basic("cid", "correct-secret").encode("ascii"),
            },
            client=("9.9.9.9", 12345),
        )

        with (
            mock.patch.object(
                oauth,
                "_get_route_client_credentials",
                new=mock.AsyncMock(return_value=("cid", "encrypted-secret")),
            ),
            mock.patch.object(
                oauth,
                "_credentials_match",
                return_value=True,
            ),
            mock.patch.object(
                oauth,
                "issue_client_credentials_token",
                return_value={"access_token": "tok", "token_type": "Bearer", "expires_in": 3600, "scope": "mcp:cowork"},
            ),
        ):
            status, _headers, body = await _capture_response(
                lambda send: oauth.handle_token_request(
                    scope,
                    _form_receive(b"grant_type=client_credentials"),
                    send,
                    "cowork",
                )
            )

        self.assertEqual(status, 200)
        self.assertNotIn("9.9.9.9", oauth._auth_failure_timestamps)

    async def test_basic_auth_returns_false_when_throttled(self) -> None:
        """validate_client_credentials_basic must return False when IP is blocked."""
        oauth._auth_blocked_until["2.3.4.5"] = 9_999_999_999.0

        scope = _scope(
            headers={
                b"host": b"ragtime.example",
                b"authorization": _basic("cid", "secret").encode("ascii"),
            },
            client=("2.3.4.5", 12345),
        )
        result = await oauth.validate_client_credentials_basic(scope, "cid", "encrypted-secret")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# 3. MCP OAuth metadata base URL — EXTERNAL_BASE_URL takes precedence
# ---------------------------------------------------------------------------


class McpResourceBaseTests(unittest.TestCase):
    def test_resource_base_uses_external_base_url_when_configured(self) -> None:
        """_resource_base must return EXTERNAL_BASE_URL when it is set."""
        scope = _scope(
            headers={
                b"host": b"internal.host",
                b"x-forwarded-host": b"attacker.example",
                b"x-forwarded-proto": b"https",
            }
        )
        with mock.patch.object(oauth.settings, "external_base_url", "https://ragtime.example.com"):
            result = oauth._resource_base(scope)
        self.assertEqual(result, "https://ragtime.example.com")

    def test_resource_base_strips_trailing_slash_from_external_base_url(self) -> None:
        scope = _scope()
        with mock.patch.object(oauth.settings, "external_base_url", "https://ragtime.example.com/"):
            result = oauth._resource_base(scope)
        self.assertEqual(result, "https://ragtime.example.com")

    def test_resource_base_falls_back_to_headers_when_no_external_url(self) -> None:
        """Without EXTERNAL_BASE_URL, _resource_base uses request headers."""
        scope = _scope(
            headers={
                b"host": b"ragtime.example",
                b"x-forwarded-proto": b"https",
            }
        )
        with mock.patch.object(oauth.settings, "external_base_url", ""):
            result = oauth._resource_base(scope)
        self.assertEqual(result, "https://ragtime.example")

    def test_resource_base_ignores_mismatched_x_forwarded_host_when_no_external_url(self) -> None:
        """Without EXTERNAL_BASE_URL, hostile X-Forwarded-Host must not replace Host."""
        scope = _scope(
            headers={
                b"host": b"internal.host",
                b"x-forwarded-host": b"public.ragtime.example",
                b"x-forwarded-proto": b"https",
            }
        )
        with mock.patch.object(oauth.settings, "external_base_url", ""):
            result = oauth._resource_base(scope)
        self.assertEqual(result, "https://internal.host")

    def test_resource_base_accepts_matching_x_forwarded_host_port(self) -> None:
        """Matching forwarded host may preserve proxy-visible port."""
        scope = _scope(
            headers={
                b"host": b"ragtime.example",
                b"x-forwarded-host": b"ragtime.example:8443",
                b"x-forwarded-proto": b"https",
            }
        )
        with mock.patch.object(oauth.settings, "external_base_url", ""):
            result = oauth._resource_base(scope)
        self.assertEqual(result, "https://ragtime.example:8443")

    def test_authorization_server_metadata_uses_external_base_url(self) -> None:
        """handle_authorization_server_metadata must embed EXTERNAL_BASE_URL in metadata."""
        import asyncio

        scope = _scope(
            headers={
                b"host": b"internal.host",
                b"x-forwarded-host": b"attacker.example",
            }
        )

        async def run():
            with (
                mock.patch.object(oauth.settings, "external_base_url", "https://ragtime.example.com"),
                mock.patch.object(
                    oauth,
                    "_get_route_client_credentials",
                    new=mock.AsyncMock(return_value=("cid", "enc-secret")),
                ),
            ):
                _status, _headers, body = await _capture_response(
                    lambda send: oauth.handle_authorization_server_metadata(scope, _form_receive(b""), send, "cowork")
                )
            return body

        body = asyncio.run(run())
        self.assertIn("ragtime.example.com", body["token_endpoint"])
        self.assertNotIn("attacker.example", body["token_endpoint"])
        self.assertNotIn("internal.host", body["token_endpoint"])


if __name__ == "__main__":
    unittest.main()
