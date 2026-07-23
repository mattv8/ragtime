import asyncio
import json
import logging
import unittest
from unittest import mock

import httpx

from ragtime.http_api.models import (
    HttpApiApiKeyLocation,
    HttpApiAuthMode,
    HttpApiBodyFormat,
    HttpApiConfiguredHeader,
    HttpApiConnectionConfig,
    HttpApiLoginBodyFormat,
    HttpApiMethod,
    HttpApiMethodPolicy,
    HttpApiRequest,
    HttpApiTokenField,
)
from ragtime.http_api.security import HttpApiSecurityError
from ragtime.http_api.service import (
    HttpApiBroker,
    HttpApiConfigurationError,
    _sanitize_json_value,
    http_api_broker,
    resolve_http_api_hostname,
)


def _header(name: str, value: str) -> HttpApiConfiguredHeader:
    return HttpApiConfiguredHeader(name=name, value=value)


def _token_field(name: str, value: str, *, secret: bool) -> HttpApiTokenField:
    return HttpApiTokenField(name=name, value=value, secret=secret)


class _TrackedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], closed_flags: list[bool]) -> None:
        self._chunks = chunks
        self._closed_flags = closed_flags
        self._closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self._closed = True
        self._closed_flags.append(True)


class HttpApiServiceTests(unittest.IsolatedAsyncioTestCase):
    def _broker(self, handler, *, resolver=None, clock=None) -> HttpApiBroker:
        options = {"clock": clock} if clock is not None else {}
        return HttpApiBroker(
            resolver=resolver or (lambda _host: ["8.8.8.8"]),
            base_transport=httpx.MockTransport(handler),
            **options,
        )

    async def _execute(self, broker: HttpApiBroker, config: HttpApiConnectionConfig, request: HttpApiRequest | None = None):
        return await broker.execute(
            "tool-1",
            config,
            request or HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=10,
        )

    async def test_resolve_http_api_hostname_deduplicates_stream_addresses(self) -> None:
        loop = asyncio.get_running_loop()

        async def fake_getaddrinfo(*_args, **_kwargs):
            return [
                (0, 0, 0, "", ("8.8.8.8", 443)),
                (0, 0, 0, "", ("2606:4700:4700::1111", 443, 0, 0)),
                (0, 0, 0, "", ("8.8.8.8", 443)),
            ]

        with mock.patch.object(loop, "getaddrinfo", side_effect=fake_getaddrinfo):
            resolved = await resolve_http_api_hostname("api.example.com")

        self.assertEqual(resolved, ["8.8.8.8", "2606:4700:4700::1111"])

    async def test_shared_http_api_broker_uses_process_resolver_function(self) -> None:
        self.assertIs(http_api_broker._resolver, resolve_http_api_hostname)

    async def test_execute_resolves_target_once_and_reuses_pinned_address_for_login_and_retry(self) -> None:
        resolver_calls: list[str] = []
        seen_hosts: list[str] = []
        seen_auth: list[str | None] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            if request.url.path == "/auth/token":
                token = "token-1" if seen_auth == [] else "token-2"
                return httpx.Response(200, json={"access_token": token, "expires_in": 60})
            seen_auth.append(request.headers.get("authorization"))
            if request.headers.get("authorization") == "Bearer token-1":
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(200, json=[{"id": 1}])

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["8.8.8.8", "8.8.4.4"] if len(resolver_calls) == 1 else ["8.8.4.4"]

        broker = self._broker(handler, resolver=resolver, clock=lambda: 1000.0)

        result = await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="/auth/token",
                login_username="demo",
                login_password="super-secret",
            ),
        )

        self.assertEqual(resolver_calls, ["api.example.com"])
        self.assertEqual(seen_hosts, ["8.8.8.8", "8.8.8.8", "8.8.8.8", "8.8.8.8"])
        self.assertEqual(result.status, 200)

    async def test_validate_configuration_without_login_sends_no_request(self) -> None:
        calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append("request")
            return httpx.Response(200, json={})

        broker = self._broker(handler)

        result = await broker.validate_configuration(HttpApiConnectionConfig(base_url="https://api.example.com"), perform_login=False)

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Configuration is valid - no live request was sent.")
        self.assertEqual(calls, [])

    async def test_validate_configuration_with_login_performs_exchange(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = self._broker(handler)

        result = await broker.validate_configuration(
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="/auth/token",
                login_username="demo",
                login_password="super-secret",
            ),
            perform_login=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Login exchange succeeded - token received.")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/auth/token")

    async def test_login_exchange_injects_api_key_server_side_for_header_location(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = self._broker(handler)

        result = await broker.validate_configuration(
            HttpApiConnectionConfig(
                base_url="https://api.example.com/base/",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="auth/token",
                login_username="demo",
                login_password="super-secret",
                api_key="secret-key",
                api_key_location=HttpApiApiKeyLocation.HEADER,
                api_key_name="X-API-Key",
                api_key_prefix="Token",
                send_api_key_to_login=True,
            ),
            perform_login=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(requests[0].headers.get("X-API-Key"), "Token secret-key")
        self.assertEqual(requests[0].url.path, "/base/auth/token")

    async def test_login_exchange_injects_api_key_server_side_for_query_location(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = self._broker(handler)

        await broker.validate_configuration(
            HttpApiConnectionConfig(
                base_url="https://api.example.com/base",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="/auth/token",
                login_username="demo",
                login_password="super-secret",
                api_key="secret-key",
                api_key_location=HttpApiApiKeyLocation.QUERY,
                api_key_name="api_key",
                send_api_key_to_login=True,
            ),
            perform_login=True,
        )

        self.assertEqual(dict(requests[0].url.params), {"api_key": "secret-key"})

    async def test_validate_configuration_with_token_exchange_performs_exchange(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(handler),
        )

        result = await broker.validate_configuration(
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                login_path="/oauth/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[
                    _token_field("grant_type", "client_credentials", secret=False),
                    _token_field("client_id", "client-id", secret=False),
                    _token_field("client_secret", "client-secret", secret=True),
                ],
            ),
            perform_login=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Token exchange succeeded - token received.")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/oauth/token")

    async def test_validate_configuration_preflights_token_exchange_conflict_before_resolver_or_network(self) -> None:
        resolver_calls: list[str] = []
        network_calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            network_calls.append("request")
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["8.8.8.8"]

        broker = self._broker(handler, resolver=resolver)

        with self.assertRaisesRegex(HttpApiSecurityError, "token_header_name"):
            await broker.validate_configuration(
                HttpApiConnectionConfig(
                    base_url="https://api.example.com",
                    auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                    login_path="/oauth/token",
                    login_body_format=HttpApiLoginBodyFormat.FORM,
                    request_headers=[_header("Authorization", "Bearer preset")],
                    token_request_fields=[
                        _token_field("grant_type", "client_credentials", secret=False),
                    ],
                    token_header_name="authorization",
                ),
                perform_login=True,
            )

        self.assertEqual(resolver_calls, [])
        self.assertEqual(network_calls, [])

    async def test_validate_configuration_preflights_blank_token_request_header_before_resolver_or_network(self) -> None:
        resolver_calls: list[str] = []
        network_calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            network_calls.append("request")
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["8.8.8.8"]

        broker = HttpApiBroker(resolver=resolver, base_transport=httpx.MockTransport(handler))

        with self.assertRaisesRegex(HttpApiSecurityError, "X-Token-Key"):
            await broker.validate_configuration(
                HttpApiConnectionConfig(
                    base_url="https://api.example.com",
                    auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                    login_path="/oauth/token",
                    login_body_format=HttpApiLoginBodyFormat.FORM,
                    token_request_headers=[_header("X-Token-Key", "")],
                    token_request_fields=[
                        _token_field("grant_type", "client_credentials", secret=False),
                    ],
                ),
                perform_login=True,
            )

        self.assertEqual(resolver_calls, [])
        self.assertEqual(network_calls, [])

    async def test_execute_token_exchange_posts_exact_form_payload_uses_cache_and_injects_authorization_header(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/oauth/token":
                return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})
            return httpx.Response(200, json=[{"id": 1}])

        broker = self._broker(handler, clock=lambda: 1000.0)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
            login_path="/oauth/token",
            login_body_format=HttpApiLoginBodyFormat.FORM,
            request_headers=[_header("X-Tenant", "tenant-a")],
            token_request_headers=[_header("X-Token-Key", "endpoint-secret")],
            token_request_fields=[
                _token_field("grant_type", "client_credentials", secret=False),
                _token_field("client_id", "client-id", secret=False),
                _token_field("client_secret", "client-secret", secret=True),
            ],
            token_header_name="Authorization",
            token_prefix="Bearer",
            token_expires_in_path="expires_in",
        )

        first = await broker.execute(
            "tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10
        )
        second = await broker.execute(
            "tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10
        )

        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 200)
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].headers.get("content-type"), "application/x-www-form-urlencoded")
        self.assertEqual(requests[0].headers.get("X-Token-Key"), "endpoint-secret")
        self.assertEqual(requests[0].content.decode("utf-8"), "grant_type=client_credentials&client_id=client-id&client_secret=client-secret")
        self.assertEqual(requests[1].headers.get("Authorization"), "Bearer token-123")
        self.assertEqual(requests[1].headers.get("X-Tenant"), "tenant-a")
        self.assertEqual(requests[2].headers.get("Authorization"), "Bearer token-123")

    async def test_execute_token_exchange_supports_authentication_header_name_and_refreshes_once_after_401(self) -> None:
        seen_resource_headers: list[str | None] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/oauth/token":
                token = "token-1" if seen_resource_headers == [] else "token-2"
                return httpx.Response(200, json={"access_token": token, "expires_in": 60})
            seen_resource_headers.append(request.headers.get("Authentication"))
            if request.headers.get("Authentication") == "Bearer token-1":
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(200, json=[{"id": 1}])

        broker = self._broker(handler, clock=lambda: 1000.0)

        result = await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                login_path="/oauth/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[
                    _token_field("grant_type", "client_credentials", secret=False),
                    _token_field("client_id", "client-id", secret=False),
                    _token_field("client_secret", "client-secret", secret=True),
                ],
                token_header_name="Authentication",
                token_prefix="Bearer",
            ),
        )

        self.assertEqual(result.status, 200)
        self.assertEqual(seen_resource_headers, ["Bearer token-1", "Bearer token-2"])

    async def test_token_exchange_reports_non_success_status_before_token_selection(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(400, json={"error": "invalid_client"})
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
            token_url="/oauth2/token",
            login_body_format=HttpApiLoginBodyFormat.FORM,
            token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
        )

        with self.assertRaisesRegex(HttpApiConfigurationError, r"^Token exchange failed with HTTP 400$"):
            await self._execute(broker, config)

    async def test_token_exchange_ignores_resource_default_selector(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(200, json={"access_token": "access-1"})
            return httpx.Response(200, json={"data": {"items": [{"id": 1}]}})

        broker = self._broker(handler)
        result = await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                token_url="/oauth2/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
                default_response_selector="data.items",
            ),
        )

        self.assertEqual(result.status, 200)
        self.assertEqual(result.output, [{"id": 1}])
        self.assertEqual(result.rows, [{"id": 1}])

    async def test_token_exchange_success_missing_token_path_preserves_selector_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(200, json={"expires_in": 60})
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
            token_url="/oauth2/token",
            login_body_format=HttpApiLoginBodyFormat.FORM,
            token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
        )

        with self.assertRaisesRegex(ValueError, r"Response selector segment not found: access_token"):
            await self._execute(broker, config)

    async def test_execute_token_exchange_rejects_resource_header_conflict_before_network_io(self) -> None:
        calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append("request")
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = self._broker(handler)

        with self.assertRaisesRegex(ValueError, "token_header_name"):
            await broker.execute(
                "tool-1",
                HttpApiConnectionConfig(
                    base_url="https://api.example.com",
                    auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                    login_path="/oauth/token",
                    login_body_format=HttpApiLoginBodyFormat.FORM,
                    request_headers=[_header("authorization", "Bearer preset")],
                    token_request_fields=[
                        _token_field("grant_type", "client_credentials", secret=False),
                        _token_field("client_id", "client-id", secret=False),
                        _token_field("client_secret", "client-secret", secret=True),
                    ],
                    token_header_name="Authorization",
                ),
                HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )

        self.assertEqual(calls, [])

    async def test_execute_token_exchange_rejects_blank_configured_header_value_before_network_io(self) -> None:
        calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append("request")
            return httpx.Response(200, json={"access_token": "token-123", "expires_in": 60})

        broker = self._broker(handler)

        with self.assertRaisesRegex(ValueError, "Header 'X-Tenant'"):
            await broker.execute(
                "tool-1",
                HttpApiConnectionConfig(
                    base_url="https://api.example.com",
                    auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                    login_path="/oauth/token",
                    login_body_format=HttpApiLoginBodyFormat.FORM,
                    request_headers=[_header("X-Tenant", "")],
                    token_request_fields=[
                        _token_field("grant_type", "client_credentials", secret=False),
                        _token_field("client_id", "client-id", secret=False),
                        _token_field("client_secret", "client-secret", secret=True),
                    ],
                ),
                HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )

        self.assertEqual(calls, [])

    async def test_execute_enforces_disabled_and_write_method_policies(self) -> None:
        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"])
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            method_policies={"POST": HttpApiMethodPolicy.WRITE, "DELETE": HttpApiMethodPolicy.DISABLED},
        )

        with self.assertRaises(ValueError):
            await broker.execute(
                "tool-1", config, HttpApiRequest(method=HttpApiMethod.DELETE, path="/items/1"), allow_write=False, timeout_seconds=5, max_results=10
            )

        with self.assertRaises(PermissionError):
            await broker.execute(
                "tool-1",
                config,
                HttpApiRequest(method=HttpApiMethod.POST, path="/items", json_body={"x": 1}),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )

    async def test_execute_headers_mode_injects_configured_request_headers_server_side(self) -> None:
        captured: dict[str, str | None] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("Authorization")
            captured["authentication"] = request.headers.get("Authentication")
            captured["tenant"] = request.headers.get("X-Tenant")
            return httpx.Response(200, json={"ok": True})

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(handler),
        )

        result = await broker.execute(
            "tool-1",
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.HEADERS,
                request_headers=[
                    _header("Authorization", "Bearer fake-token"),
                    _header("Authentication", "Bearer fake-authentication-token"),
                    _header("X-Tenant", "tenant-a"),
                ],
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=10,
        )

        self.assertEqual(result.output, {"ok": True})
        self.assertEqual(captured, {"authorization": "Bearer fake-token", "authentication": "Bearer fake-authentication-token", "tenant": "tenant-a"})

    async def test_execute_merges_configured_json_fields_with_precedence(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["content_type"] = request.headers["content-type"]
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.HEADERS,
                method_policies={"POST": HttpApiMethodPolicy.READ},
                request_body_fields=[_token_field("tenant", "configured", secret=False)],
            ),
            HttpApiRequest(method=HttpApiMethod.POST, path="/items", json_body={"tenant": "operation", "name": "Ada"}),
        )
        self.assertEqual(captured, {"content_type": "application/json", "body": {"tenant": "configured", "name": "Ada"}})

    async def test_execute_encodes_configured_form_and_multipart_fields(self) -> None:
        captured: list[tuple[str, str, bytes]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append((request.headers["content-type"], request.url.path, request.content))
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        form_config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.HEADERS,
            request_body_format=HttpApiBodyFormat.FORM,
            method_policies={"POST": HttpApiMethodPolicy.READ},
            request_body_fields=[_token_field("tenant", "configured", secret=False)],
        )
        await self._execute(broker, form_config, HttpApiRequest(method=HttpApiMethod.POST, path="/form", form_body={"name": "Ada"}))
        multipart_config = form_config.model_copy(update={"request_body_format": HttpApiBodyFormat.MULTIPART})
        await self._execute(broker, multipart_config, HttpApiRequest(method=HttpApiMethod.POST, path="/multipart"))
        self.assertEqual(captured[0], ("application/x-www-form-urlencoded", "/form", b"name=Ada&tenant=configured"))
        self.assertIn("multipart/form-data; boundary=", captured[1][0])
        self.assertIn(b'name="tenant"', captured[1][2])
        self.assertIn(b"configured", captured[1][2])

    async def test_execute_rejects_body_format_mismatch_and_non_object_json(self) -> None:
        broker = self._broker(lambda _request: httpx.Response(200, json={}))
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.HEADERS,
            method_policies={"POST": HttpApiMethodPolicy.READ},
            request_body_fields=[_token_field("tenant", "configured", secret=False)],
        )
        with self.assertRaisesRegex(HttpApiConfigurationError, "JSON fields require"):
            await self._execute(broker, config, HttpApiRequest(method=HttpApiMethod.POST, path="/items", json_body=["not", "object"]))
        with self.assertRaisesRegex(HttpApiConfigurationError, "does not match"):
            await self._execute(broker, config, HttpApiRequest(method=HttpApiMethod.POST, path="/items", form_body={"x": "1"}))

    async def test_execute_enforces_encoded_multipart_size_limit(self) -> None:
        calls: list[str] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            calls.append("sent")
            return httpx.Response(200, json={})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.HEADERS,
            request_body_format=HttpApiBodyFormat.MULTIPART,
            method_policies={"POST": HttpApiMethodPolicy.READ},
            request_body_fields=[_token_field("payload", "x" * (256 * 1024), secret=False)],
        )
        with self.assertRaisesRegex(HttpApiConfigurationError, "exceeds"):
            await self._execute(broker, config, HttpApiRequest(method=HttpApiMethod.POST, path="/items"))
        self.assertEqual(calls, [])

    async def test_headers_connectivity_is_bodyless_and_credential_free(self) -> None:
        captured: httpx.Request | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = request
            return httpx.Response(401)

        broker = self._broker(handler)
        result = await broker.validate_connectivity(
            HttpApiConnectionConfig(
                base_url="https://api.example.com/base",
                auth_mode=HttpApiAuthMode.HEADERS,
                request_headers=[_header("X-Api-Key", "secret")],
                request_body_fields=[_token_field("tenant", "north", secret=False)],
                method_policies={"HEAD": HttpApiMethodPolicy.DISABLED},
            )
        )
        assert captured is not None
        self.assertEqual(captured.method, "HEAD")
        self.assertEqual(captured.url.path, "/base")
        self.assertEqual(captured.content, b"")
        self.assertNotIn("x-api-key", captured.headers)
        self.assertTrue(result.success)
        self.assertEqual(result.details, {"status": 401, "auth_tested": False})

    async def test_absolute_token_endpoint_uses_separate_pinned_client_and_no_credential_crossover(self) -> None:
        resolver_calls: list[str] = []
        seen: list[tuple[str, str | None, str | None]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.host, request.headers.get("x-resource-secret"), request.headers.get("x-token-secret")))
            if request.url.path == "/oauth2/token":
                return httpx.Response(200, json={"access_token": "access-1"})
            return httpx.Response(200, json={"ok": True})

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return {"api.example.com": ["8.8.8.8"], "auth.example.com": ["8.8.4.4"]}[host]

        broker = self._broker(handler, resolver=resolver)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com/base",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                token_url="https://auth.example.com/oauth2/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
                token_request_headers=[_header("X-Token-Secret", "token-secret")],
                request_headers=[_header("X-Resource-Secret", "resource-secret")],
            ),
        )
        self.assertEqual(resolver_calls, ["api.example.com", "auth.example.com"])
        self.assertEqual(seen, [("8.8.4.4", None, "token-secret"), ("8.8.8.8", "resource-secret", None)])

    async def test_same_origin_absolute_token_url_ignores_resource_base_path_and_reuses_pin(self) -> None:
        resolver_calls: list[str] = []
        seen_paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if request.url.path == "/oauth2/token":
                return httpx.Response(200, json={"access_token": "access-1"})
            return httpx.Response(200, json={"ok": True})

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["8.8.8.8"]

        broker = self._broker(handler, resolver=resolver)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com/base",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                token_url="https://api.example.com:443/oauth2/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
            ),
        )
        self.assertEqual(resolver_calls, ["api.example.com"])
        self.assertEqual(seen_paths, ["/oauth2/token", "/base/items"])

    async def test_absolute_token_url_supports_public_ipv6_origin_with_bracketed_wire_url(self) -> None:
        resolver_calls: list[str] = []
        seen_urls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_urls.append(str(request.url))
            if request.url.path == "/oauth2/token":
                return httpx.Response(200, json={"access_token": "access-1"})
            return httpx.Response(200, json={"ok": True})

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["2606:4700:4700::1111"] if host == "2606:4700:4700::1111" else ["8.8.8.8"]

        broker = self._broker(handler, resolver=resolver)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                token_url="https://[2606:4700:4700::1111]/oauth2/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
            ),
        )
        self.assertEqual(resolver_calls, ["api.example.com", "2606:4700:4700::1111"])
        self.assertTrue(seen_urls[0].startswith("https://[2606:4700:4700::1111]/oauth2/token"))

    async def test_token_target_validation_error_does_not_include_endpoint_value(self) -> None:
        broker = self._broker(lambda _request: httpx.Response(200, json={"access_token": "unused"}))
        secret_endpoint = "https://api.example.com:token-secret/oauth2/token"
        with self.assertRaisesRegex(HttpApiSecurityError, "Token target validation failed") as caught:
            await self._execute(
                broker,
                HttpApiConnectionConfig(
                    base_url="https://api.example.com",
                    auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                    token_url=secret_endpoint,
                    token_request_fields=[_token_field("grant_type", "client_credentials", secret=False)],
                ),
            )
        self.assertNotIn(secret_endpoint, str(caught.exception))

    async def test_get_with_configured_json_body_sends_body(self) -> None:
        captured: httpx.Request | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = request
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.HEADERS,
                request_body_fields=[_token_field("tenant", "configured", secret=False)],
            ),
        )
        assert captured is not None
        self.assertEqual(captured.method, "GET")
        self.assertEqual(json.loads(captured.content), {"tenant": "configured"})

    async def test_ukg_shaped_token_exchange_keeps_token_and_resource_credentials_separate(self) -> None:
        requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path.endswith("/oauth2/token"):
                return httpx.Response(200, json={"access_token": "access-1"})
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://secure6.saashr.com",
                auth_mode=HttpApiAuthMode.TOKEN_EXCHANGE,
                token_url="/ta/rest/v2/companies/100713024/oauth2/token",
                login_body_format=HttpApiLoginBodyFormat.FORM,
                token_request_fields=[
                    _token_field("grant_type", "client_credentials", secret=False),
                    _token_field("client_id", "client-id", secret=True),
                    _token_field("client_secret", "client-secret", secret=True),
                ],
                token_header_name="Authentication",
                token_prefix="Bearer",
                request_headers=[_header("Content-Type", "application/json")],
            ),
        )
        self.assertEqual(requests[0].headers["content-type"], "application/x-www-form-urlencoded")
        self.assertEqual(requests[1].headers["authentication"], "Bearer access-1")
        self.assertNotIn("client-secret", requests[1].content.decode())

    async def test_execute_blocks_unapproved_and_sensitive_headers(self) -> None:
        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"])
        config = HttpApiConnectionConfig(base_url="https://api.example.com", approved_request_headers=["x-trace-id"])

        with self.assertRaises(ValueError):
            await broker.execute(
                "tool-1",
                config,
                HttpApiRequest(method=HttpApiMethod.GET, path="/items", headers={"Authorization": "Bearer secret"}),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )

        with self.assertRaises(ValueError):
            await broker.execute(
                "tool-1",
                config,
                HttpApiRequest(method=HttpApiMethod.GET, path="/items", headers={"Accept": "application/json"}),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )

    async def test_execute_injects_query_api_key_and_normalizes_selected_rows(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["query"] = dict(request.url.params)
            captured["host"] = request.headers.get("host")
            return httpx.Response(200, json={"data": {"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]}})

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(handler),
        )
        result = await broker.execute(
            "tool-1",
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.API_KEY,
                api_key="secret-key",
                api_key_location=HttpApiApiKeyLocation.QUERY,
                api_key_name="api_key",
                send_api_key_to_requests=True,
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items", response_selector="data.items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=1,
        )

        self.assertEqual(captured["query"], {"api_key": "secret-key"})
        self.assertEqual(result.status, 200)
        self.assertEqual(result.rows, [{"id": 1, "name": "A"}])
        self.assertEqual(result.columns, ["id", "name"])
        self.assertEqual(result.row_count, 1)

    async def test_execute_does_not_send_api_key_to_resource_requests_without_opt_in(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["query"] = dict(request.url.params)
            captured["header"] = request.headers.get("X-API-Key")
            return httpx.Response(200, json={"ok": True})

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(handler),
        )
        result = await broker.execute(
            "tool-1",
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.API_KEY,
                api_key="secret-key",
                api_key_location=HttpApiApiKeyLocation.HEADER,
                api_key_name="X-API-Key",
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=1,
        )

        self.assertEqual(captured["query"], {})
        self.assertIsNone(captured["header"])
        self.assertEqual(result.output, {"ok": True})

    def test_sanitize_json_value_redacts_generic_token_fields_recursively(self) -> None:
        value = {
            "safe": "visible",
            "token": "also-visible",
            "AcCeSs_ToKeN": "access-secret",
            "nested": [
                {"REFRESH_TOKEN": "refresh-secret", "safe_nested": True},
                {"Client_Secret": "client-secret", "OAuth_Access_Token": "oauth-secret"},
            ],
        }

        sanitized = _sanitize_json_value(value)

        self.assertEqual(
            sanitized,
            {
                "safe": "visible",
                "token": "also-visible",
                "nested": [{"safe_nested": True}, {}],
            },
        )

    async def test_execute_refreshes_login_token_once_after_401(self) -> None:
        seen_auth: list[str | None] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/auth/token":
                token = "token-1" if seen_auth == [] else "token-2"
                return httpx.Response(200, json={"access_token": token, "expires_in": 60})
            seen_auth.append(request.headers.get("authorization"))
            if request.headers.get("authorization") == "Bearer token-1":
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(200, json=[{"id": 1}])

        broker = self._broker(handler, clock=lambda: 1000.0)
        result = await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="/auth/token",
                login_username="demo",
                login_password="super-secret",
            ),
        )

        self.assertEqual(seen_auth, ["Bearer token-1", "Bearer token-2"])
        self.assertEqual(result.status, 200)
        self.assertEqual(result.rows, [{"id": 1}])

    async def test_execute_oauth_injects_access_token_and_refreshes_after_401(self) -> None:
        seen: list[str | None] = []
        refresh_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal refresh_calls
            if request.url.path == "/token":
                refresh_calls += 1
                return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600})
            seen.append(request.headers.get("authorization"))
            return httpx.Response(401 if len(seen) == 1 else 200, json={"ok": True})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_access_token="stale",
            oauth_refresh_token="refresh",
            oauth_token_expires_at="2999-01-01T00:00:00Z",
        )

        result = await self._execute(broker, config)

        self.assertEqual(result.status, 200)
        self.assertEqual(seen, ["Bearer stale", "Bearer fresh"])
        self.assertEqual(refresh_calls, 1)
        self.assertEqual(config.oauth_refresh_token, "rotated")

    async def test_oauth_resource_headers_and_body_fields_are_applied_with_precedence(self) -> None:
        captured: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_access_token="access",
            oauth_token_expires_at="2999-01-01T00:00:00Z",
            request_headers=[_header("X-Tenant", "configured")],
            request_body_fields=[_token_field("tenant", "configured", secret=False)],
        )

        await self._execute(
            broker,
            config,
            HttpApiRequest(method=HttpApiMethod.GET, path="/items", headers={}, json_body={"tenant": "operation", "name": "Ada"}),
        )

        headers = captured["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["x-tenant"], "configured")
        self.assertEqual(headers["authorization"], "Bearer access")
        self.assertEqual(captured["body"], {"tenant": "configured", "name": "Ada"})

    async def test_oauth_resource_body_fields_encode_form_data(self) -> None:
        captured: httpx.Request | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = request
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        await self._execute(
            broker,
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.OAUTH2,
                oauth_client_id="client",
                oauth_token_url="https://api.example.com/token",
                oauth_device_authorization_url="https://api.example.com/device",
                oauth_access_token="access",
                oauth_token_expires_at="2999-01-01T00:00:00Z",
                request_body_format=HttpApiBodyFormat.FORM,
                request_body_fields=[_token_field("tenant", "configured", secret=False)],
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items", form_body={"tenant": "operation", "name": "Ada"}),
        )
        assert captured is not None
        self.assertEqual(captured.headers["content-type"], "application/x-www-form-urlencoded")
        self.assertEqual(captured.content, b"tenant=configured&name=Ada")

    async def test_oauth_resource_token_header_conflict_is_rejected(self) -> None:
        broker = self._broker(lambda _request: httpx.Response(200, json={"ok": True}))
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_access_token="access",
            request_headers=[_header("Authorization", "preset")],
        )
        with self.assertRaisesRegex(HttpApiSecurityError, "token_header_name"):
            await self._execute(broker, config)

    async def test_oauth_validate_configuration_without_login_validates_endpoints_without_provider_call(self) -> None:
        calls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_access_token="access",
        )

        result = await broker.validate_configuration(config, perform_login=False)

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Configuration is valid - no live request was sent.")
        self.assertEqual(calls, [])

    async def test_oauth_validate_configuration_login_requires_stored_unexpired_token_without_refresh(self) -> None:
        broker = self._broker(lambda _request: httpx.Response(500, json={"error": "must not call"}))
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_access_token="access",
            oauth_token_expires_at="2999-01-01T00:00:00Z",
        )

        result = await broker.validate_configuration(config, perform_login=True)

        self.assertTrue(result.success)
        self.assertEqual(result.message, "OAuth configuration validated - stored access token is available.")

    async def test_oauth_validate_configuration_login_rejects_missing_or_expired_access_token_without_provider_call(self) -> None:
        provider_calls: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            provider_calls.append(request)
            return httpx.Response(200, json={"access_token": "must-not-be-used"})

        broker = self._broker(handler)
        base_config = {
            "base_url": "https://api.example.com",
            "auth_mode": HttpApiAuthMode.OAUTH2,
            "oauth_client_id": "client",
            "oauth_token_url": "https://api.example.com/token",
            "oauth_device_authorization_url": "https://api.example.com/device",
            "oauth_refresh_token": "refresh",
        }

        for oauth_fields in (
            {"oauth_access_token": ""},
            {"oauth_access_token": "expired", "oauth_token_expires_at": "2000-01-01T00:00:00Z"},
        ):
            with self.subTest(oauth_fields=oauth_fields):
                config = HttpApiConnectionConfig.model_validate({**base_config, **oauth_fields})
                with self.assertRaisesRegex(ValueError, "OAuth validation requires a usable stored access token"):
                    await broker.validate_configuration(config, perform_login=True)

        self.assertEqual(provider_calls, [])

    async def test_oauth_refresh_persists_rotation_only_through_callback(self) -> None:
        updates: list[dict] = []

        async def updater(update: dict) -> None:
            updates.append(update)

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/token":
                return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600})
            return httpx.Response(200, json={"ok": True})

        broker = self._broker(handler)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.OAUTH2,
            oauth_client_id="client",
            oauth_token_url="https://api.example.com/token",
            oauth_device_authorization_url="https://api.example.com/device",
            oauth_refresh_token="refresh",
        )
        await broker.execute(
            "tool-1",
            config,
            HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=10,
            oauth_credential_updater=updater,
        )
        self.assertEqual(updates[0]["oauth_refresh_token"], "rotated")
        self.assertEqual(updates[0]["oauth_refresh_token_used"], "refresh")

    async def test_execute_login_token_cache_is_concurrency_safe(self) -> None:
        login_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal login_calls
            if request.url.path == "/auth/token":
                login_calls += 1
                return httpx.Response(200, json={"access_token": "token-1", "expires_in": 60})
            return httpx.Response(200, json=[{"id": 1}])

        broker = self._broker(handler, clock=lambda: 1000.0)
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
            login_path="/auth/token",
            login_username="demo",
            login_password="super-secret",
            token_expires_in_path="expires_in",
        )

        await asyncio.gather(
            broker.execute("tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10),
            broker.execute("tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10),
        )

        self.assertEqual(login_calls, 1)

    async def test_execute_login_token_cache_respects_expiry(self) -> None:
        current_time = {"value": 1000.0}
        login_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal login_calls
            if request.url.path == "/auth/token":
                login_calls += 1
                return httpx.Response(200, json={"access_token": f"token-{login_calls}", "expires_in": 1})
            return httpx.Response(200, json=[{"id": login_calls}])

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(handler),
            clock=lambda: current_time["value"],
        )
        config = HttpApiConnectionConfig(
            base_url="https://api.example.com",
            auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
            login_path="/auth/token",
            login_username="demo",
            login_password="super-secret",
            token_expires_in_path="expires_in",
        )

        await broker.execute("tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10)
        current_time["value"] = 1002.0
        await broker.execute("tool-1", config, HttpApiRequest(method=HttpApiMethod.GET, path="/items"), allow_write=False, timeout_seconds=5, max_results=10)

        self.assertEqual(login_calls, 2)

    async def test_execute_rejects_selector_wildcard_and_bracket_syntax(self) -> None:
        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": [{"id": 1}]})),
        )
        config = HttpApiConnectionConfig(base_url="https://api.example.com")

        for selector in ("data[*]", "data..items", "data[0]", "$.data", "data?(x)"):
            with self.subTest(selector=selector):
                with self.assertRaises(ValueError):
                    await broker.execute(
                        "tool-1",
                        config,
                        HttpApiRequest(method=HttpApiMethod.GET, path="/items", response_selector=selector),
                        allow_write=False,
                        timeout_seconds=5,
                        max_results=10,
                    )

    async def test_execute_rejects_http_and_private_targets_in_production(self) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        broker = HttpApiBroker(resolver=lambda _host: ["127.0.0.1"], base_transport=httpx.MockTransport(handler))
        with mock.patch("ragtime.http_api.service.settings.debug_mode", False):
            with self.assertRaises(ValueError):
                await broker.validate_configuration(HttpApiConnectionConfig(base_url="http://api.example.com"), perform_login=False)
            with self.assertRaises(ValueError):
                await broker.validate_configuration(HttpApiConnectionConfig(base_url="https://api.example.com"), perform_login=False)

    async def test_execute_enforces_request_body_limit(self) -> None:
        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"], base_transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})))
        huge_value = "x" * (256 * 1024)
        with self.assertRaises(ValueError):
            await broker.execute(
                "tool-1",
                HttpApiConnectionConfig(base_url="https://api.example.com", method_policies={"POST": HttpApiMethodPolicy.WRITE}),
                HttpApiRequest(method=HttpApiMethod.POST, path="/items", json_body={"payload": huge_value}),
                allow_write=True,
                timeout_seconds=5,
                max_results=10,
            )

    async def test_execute_enforces_response_body_limit_and_closes_response(self) -> None:
        closed_flags: list[bool] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            stream = _TrackedAsyncByteStream([b"a" * (1024 * 1024), b"b"], closed_flags)
            return httpx.Response(200, stream=stream)

        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"], base_transport=httpx.MockTransport(handler))
        with self.assertRaises(ValueError):
            await broker.execute(
                "tool-1",
                HttpApiConnectionConfig(base_url="https://api.example.com"),
                HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
                allow_write=False,
                timeout_seconds=5,
                max_results=10,
            )
        self.assertEqual(closed_flags, [True])

    async def test_execute_closes_401_response_before_retry(self) -> None:
        responses: list[httpx.Response] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/auth/token":
                return httpx.Response(200, json={"access_token": "token-1", "expires_in": 60})
            status_code = 401 if not responses else 200
            response = httpx.Response(status_code, json=[{"id": 1}])
            responses.append(response)
            return response

        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"], base_transport=httpx.MockTransport(handler), clock=lambda: 1000.0)
        await broker.execute(
            "tool-1",
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.LOGIN_EXCHANGE,
                login_path="/auth/token",
                login_username="demo",
                login_password="super-secret",
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=10,
        )
        self.assertTrue(responses[0].is_closed)

    async def test_build_client_disables_redirects_and_trust_env(self) -> None:
        recorded: dict[str, object] = {}

        class RecordingClient(httpx.AsyncClient):
            def __init__(self, *args, **kwargs):
                recorded["follow_redirects"] = kwargs.get("follow_redirects")
                recorded["trust_env"] = kwargs.get("trust_env")
                super().__init__(*args, **kwargs)

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
            client_factory=RecordingClient,
        )
        await broker.validate_configuration(HttpApiConnectionConfig(base_url="https://api.example.com"), perform_login=False)
        self.assertEqual(recorded, {"follow_redirects": False, "trust_env": False})

    async def test_oauth_absolute_requests_reject_private_targets_outside_debug(self) -> None:
        broker = self._broker(lambda _request: httpx.Response(200, json={}), resolver=lambda _host: ["127.0.0.1"])
        with mock.patch("ragtime.http_api.service.settings.debug_mode", False):
            with self.assertRaisesRegex(HttpApiSecurityError, "target"):
                await broker.oauth_request_json("https://auth.example.com/token")

    async def test_fetch_openapi_document_uses_single_resolution_and_response_cap(self) -> None:
        resolver_calls: list[str] = []
        seen_hosts: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_hosts.append(request.url.host)
            return httpx.Response(200, text='{"openapi":"3.1.0","info":{"title":"Demo","version":"1"},"paths":{}}')

        def resolver(host: str) -> list[str]:
            resolver_calls.append(host)
            return ["8.8.8.8", "8.8.4.4"] if len(resolver_calls) == 1 else ["8.8.4.4"]

        broker = HttpApiBroker(resolver=resolver, base_transport=httpx.MockTransport(handler))
        document = await broker.fetch_openapi_document("https://api.example.com/openapi.json", timeout_seconds=5)
        self.assertIn('"openapi":"3.1.0"', document)
        self.assertEqual(resolver_calls, ["api.example.com"])
        self.assertEqual(seen_hosts, ["8.8.8.8"])

    async def test_fetch_openapi_document_rejects_non_https_in_production(self) -> None:
        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"], base_transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="{}")))
        with mock.patch("ragtime.http_api.service.settings.debug_mode", False):
            with self.assertRaises(ValueError):
                await broker.fetch_openapi_document("http://api.example.com/openapi.json", timeout_seconds=5)

    async def test_validate_configuration_rejects_base_url_with_query_string(self) -> None:
        broker = HttpApiBroker(resolver=lambda _host: ["8.8.8.8"])
        with self.assertRaises(ValueError):
            await broker.validate_configuration(HttpApiConnectionConfig(base_url="https://api.example.com/base?x=1"), perform_login=False)

    async def test_execute_does_not_leak_secrets_to_errors_or_logs(self) -> None:
        logger = logging.getLogger("http-api-test")
        messages: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                messages.append(record.getMessage())

        handler = Capture()
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        async def failing_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal", "token": "server-secret"})

        broker = HttpApiBroker(
            resolver=lambda _host: ["8.8.8.8"],
            base_transport=httpx.MockTransport(failing_handler),
            logger=logger,
        )

        result = await broker.execute(
            "tool-1",
            HttpApiConnectionConfig(
                base_url="https://api.example.com",
                auth_mode=HttpApiAuthMode.BEARER,
                bearer_token="client-secret",
            ),
            HttpApiRequest(method=HttpApiMethod.GET, path="/items"),
            allow_write=False,
            timeout_seconds=5,
            max_results=10,
        )

        self.assertEqual(result.status, 500)
        self.assertNotIn("client-secret", str(result.output))
        for message in messages:
            self.assertNotIn("client-secret", message)
            self.assertNotIn("server-secret", message)


if __name__ == "__main__":
    unittest.main()
