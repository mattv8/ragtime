"""HTTP rate-limit contracts for interactive OAuth token exchanges."""

import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from starlette.types import Message

from ragtime import main
from ragtime.api import auth as api_auth
from ragtime.core import rate_limit
from ragtime.core.rate_limit import MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES, OAuthTokenRateLimitMiddleware, limiter
from ragtime.userspace import preview_host


class OAuthTokenRateLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.limiter_enabled = mock.patch.object(limiter, "enabled", True)
        self.limiter_enabled.start()
        limiter._storage.reset()
        api_auth._auth_codes.clear()

    def tearDown(self) -> None:
        limiter._storage.reset()
        api_auth._auth_codes.clear()
        self.limiter_enabled.stop()

    def _assert_quota_response(self, response: httpx.Response) -> None:
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"], "too_many_requests")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["pragma"], "no-cache")
        self.assertGreater(int(response.headers["retry-after"]), 0)

    def _store_code(self, code: str) -> None:
        api_auth._auth_codes[code] = {
            "user_id": "user-1",
            "username": "alice",
            "role": "user",
            "client_id": "client-a",
            "redirect_uri": "https://client.example/callback",
            "code_challenge": "challenge",
            "security_generation": 0,
            "resource": "mcp-service",
            "scope": "",
            "expires": 4_102_444_800,
        }

    async def _exchange(self, client: httpx.AsyncClient, path: str, code: str) -> httpx.Response:
        self._store_code(code)
        return await client.post(
            path,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": "verifier",
                "client_id": "client-a",
            },
        )

    async def test_code_exchange_quota_allows_more_than_five_successes_and_is_shared_by_aliases(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        token_pair = {"access_token": "token", "token_type": "Bearer", "expires_in": 60, "scope": ""}
        with (
            mock.patch.object(api_auth, "_verify_pkce", return_value=True),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "_issue_mcp_token_pair", new=mock.AsyncMock(return_value=token_pair)),
            mock.patch.object(api_auth, "_refresh_mcp_token_pair", new=mock.AsyncMock(return_value=token_pair)),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.10", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for attempt in range(59):
                    response = await self._exchange(client, "/token" if attempt % 2 else "/auth/oauth2/token", f"code-{attempt}")
                    self.assertEqual(response.status_code, 200)
                refresh = await client.post(
                    "/auth/oauth2/token",
                    data={"grant_type": "refresh_token", "refresh_token": "refresh-token", "client_id": "client-a"},
                )
                self.assertEqual(refresh.status_code, 200)
                limited = await self._exchange(client, "/token", "code-limited")

        self._assert_quota_response(limited)

    async def test_password_attempts_use_a_separate_five_per_minute_bucket(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")
        token_pair = {"access_token": "token", "token_type": "Bearer", "expires_in": 60, "scope": ""}
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="invalid-session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
            mock.patch.object(api_auth, "_verify_pkce", return_value=True),
            mock.patch.object(api_auth, "_issue_mcp_token_pair", new=mock.AsyncMock(return_value=token_pair)),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.20", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for _ in range(5):
                    response = await client.post("/token", data={"grant_type": "password", "username": "alice", "password": "correct-password"})
                    self.assertEqual(response.status_code, 200)
                password_limited = await client.post("/auth/oauth2/token", data={"grant_type": "password", "username": "alice", "password": "correct-password"})
                exchange = await self._exchange(client, "/token", "code-after-password-limit")

        self._assert_quota_response(password_limited)
        self.assertEqual(exchange.status_code, 200)

    async def test_malformed_grants_share_the_exchange_bucket_without_client_id_partitioning(self) -> None:
        transport = httpx.ASGITransport(app=main.app, client=("198.51.100.30", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
            for attempt in range(60):
                response = await client.post(
                    "/token" if attempt % 2 else "/auth/oauth2/token",
                    data={"grant_type": "not-supported", "client_id": f"untrusted-{attempt}"},
                )
                self.assertEqual(response.status_code, 400)
            limited = await client.post("/token", data={"grant_type": "not-supported", "client_id": "untrusted-limited"})

        self._assert_quota_response(limited)

    async def test_exchange_quota_is_isolated_by_remote_address(self) -> None:
        first_transport = httpx.ASGITransport(app=main.app, client=("198.51.100.40", 12345))
        second_transport = httpx.ASGITransport(app=main.app, client=("198.51.100.41", 12345))
        async with (
            httpx.AsyncClient(transport=first_transport, base_url="http://ragtime.test") as first_client,
            httpx.AsyncClient(transport=second_transport, base_url="http://ragtime.test") as second_client,
        ):
            for _ in range(60):
                response = await first_client.post("/token", data={"grant_type": "unsupported"})
                self.assertEqual(response.status_code, 400)
            first_limited = await first_client.post("/token", data={"grant_type": "unsupported"})
            second_allowed = await second_client.post("/auth/oauth2/token", data={"grant_type": "unsupported"})

        self._assert_quota_response(first_limited)
        self.assertEqual(second_allowed.status_code, 400)

    async def test_ambiguous_password_requests_cannot_bypass_the_password_bucket(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")
        variants = ("duplicate", "multipart", "invalid_utf8")

        async def post_variant(client: httpx.AsyncClient, path: str, variant: str) -> httpx.Response:
            if variant == "duplicate":
                return await client.post(
                    path,
                    content=b"grant_type=x&grant_type=password&username=alice&password=correct",
                    headers={"content-type": "application/x-www-form-urlencoded"},
                )
            if variant == "multipart":
                return await client.post(
                    path,
                    files={
                        "grant_type": (None, "password"),
                        "username": (None, "alice"),
                        "password": (None, "correct"),
                    },
                )
            return await client.post(
                path,
                content=b"grant_type=password&username=alice&password=correct&noise=\xff",
                headers={"content-type": "application/x-www-form-urlencoded"},
            )

        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)) as authenticate,
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
        ):
            for variant_index, variant in enumerate(variants):
                for alias_index, path in enumerate(("/token", "/auth/oauth2/token")):
                    limiter._storage.reset()
                    authenticate.reset_mock()
                    transport = httpx.ASGITransport(app=main.app, client=(f"198.51.100.{50 + variant_index * 2 + alias_index}", 12345))
                    async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                        for _ in range(5):
                            response = await post_variant(client, path, variant)
                            self.assertEqual(response.status_code, 200)
                        limited = await post_variant(client, path, variant)

                    self._assert_quota_response(limited)
                    self.assertEqual(authenticate.await_count, 5)

    async def test_missing_grant_validation_error_consumes_the_password_bucket(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.60", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                invalid = await client.post("/token", data={})
                for _ in range(4):
                    response = await client.post("/auth/oauth2/token", data={"grant_type": "password", "username": "alice", "password": "correct"})
                    self.assertEqual(response.status_code, 200)
                limited = await client.post("/token", data={"grant_type": "password", "username": "alice", "password": "correct"})

        self.assertEqual(invalid.status_code, 422)
        self._assert_quota_response(limited)

    async def test_disabled_limiter_preserves_debug_token_behavior(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")
        with (
            mock.patch.object(limiter, "enabled", False),
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)) as authenticate,
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.61", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for path in ("/token", "/auth/oauth2/token"):
                    for _ in range(3):
                        response = await client.post(path, data={"grant_type": "password", "username": "alice", "password": "correct"})
                        self.assertEqual(response.status_code, 200)

        self.assertEqual(authenticate.await_count, 6)

    async def test_preview_host_token_route_bypasses_the_control_plane_quota(self) -> None:
        received_requests = 0
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")

        async def preview_app(_scope, _receive, send) -> None:
            nonlocal received_requests
            received_requests += 1
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        with (
            mock.patch.object(preview_host, "is_preview_host", return_value=True),
            mock.patch.object(preview_host, "_workspace_service_guard_response", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(preview_host, "preview_host_app", new=preview_app),
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.70", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for _ in range(6):
                    response = await client.post("/token", headers={"host": "workspace-a.ragtime.test"}, data={"grant_type": "password"})
                    self.assertEqual(response.status_code, 204)
                with mock.patch.object(preview_host, "is_preview_host", return_value=False):
                    for _ in range(5):
                        response = await client.post("/token", data={"grant_type": "password", "username": "alice", "password": "correct"})
                        self.assertEqual(response.status_code, 200)
                    limited = await client.post("/token", data={"grant_type": "password", "username": "alice", "password": "correct"})

        self.assertEqual(received_requests, 6)
        self._assert_quota_response(limited)

    async def test_cors_wraps_token_quota_errors(self) -> None:
        user = SimpleNamespace(id="user-1", securityGeneration=0)
        db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)))
        auth_result = SimpleNamespace(success=True, user_id="user-1", username="alice", role="user")
        with (
            mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
            mock.patch.object(api_auth, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(api_auth, "_issue_login_session", new=mock.AsyncMock(return_value="session-token")),
            mock.patch.object(api_auth, "decode_jwt_payload", return_value={"exp": 4_102_444_800}),
        ):
            transport = httpx.ASGITransport(app=main.app, client=("198.51.100.80", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://ragtime.test") as client:
                for _ in range(5):
                    response = await client.post(
                        "/token", headers={"origin": "http://localhost:3000"}, data={"grant_type": "password", "username": "alice", "password": "correct"}
                    )
                    self.assertEqual(response.status_code, 200)
                limited = await client.post(
                    "/token", headers={"origin": "http://localhost:3000"}, data={"grant_type": "password", "username": "alice", "password": "correct"}
                )

        self._assert_quota_response(limited)
        self.assertEqual(limited.headers["access-control-allow-origin"], "http://localhost:3000")

    async def test_oversized_chunked_token_body_is_rejected_before_the_downstream_app(self) -> None:
        downstream = mock.AsyncMock()
        middleware = OAuthTokenRateLimitMiddleware(downstream)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/token",
            "headers": [(b"content-type", b"application/x-www-form-urlencoded"), (b"content-length", b"1")],
            "client": ("198.51.100.90", 12345),
        }
        messages = iter(
            [
                {"type": "http.request", "body": b"x" * MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES, "more_body": True},
                {"type": "http.request", "body": b"x", "more_body": False},
            ]
        )
        sent: list[Message] = []

        async def receive() -> Message:
            return next(messages)

        async def send(message: Message) -> None:
            sent.append(message)

        await middleware(scope, receive, send)

        downstream.assert_not_awaited()
        self.assertEqual(sent[0]["status"], 413)

    async def test_oversized_declared_token_body_is_rejected_without_reading_it(self) -> None:
        downstream = mock.AsyncMock()
        middleware = OAuthTokenRateLimitMiddleware(downstream)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/token",
            "headers": [(b"content-length", str(MAX_OAUTH_TOKEN_REQUEST_BODY_BYTES + 1).encode())],
            "client": ("198.51.100.91", 12345),
        }
        sent: list[Message] = []

        async def receive() -> Message:
            raise AssertionError("oversized declared body must not be read")

        async def send(message: Message) -> None:
            sent.append(message)

        await middleware(scope, receive, send)

        downstream.assert_not_awaited()
        self.assertEqual(sent[0]["status"], 413)

    async def test_disconnected_token_request_is_not_dispatched_or_rate_limited(self) -> None:
        downstream = mock.AsyncMock()
        middleware = OAuthTokenRateLimitMiddleware(downstream)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/token",
            "headers": [(b"content-type", b"application/x-www-form-urlencoded")],
            "client": ("198.51.100.92", 12345),
        }
        messages: list[Message] = [
            {"type": "http.request", "body": b"grant_type=pass", "more_body": True},
            {"type": "http.disconnect"},
        ]
        sent: list[Message] = []

        async def receive() -> Message:
            return messages.pop(0)

        async def send(message: Message) -> None:
            sent.append(message)

        with mock.patch.object(rate_limit, "consume_oauth_token_rate_limit") as consume_quota:
            await middleware(scope, receive, send)

        downstream.assert_not_awaited()
        consume_quota.assert_not_called()
        self.assertEqual(sent[0]["status"], 400)

    async def test_main_app_returns_oauth_error_for_disconnected_token_requests(self) -> None:
        for path in ("/token", "/auth/oauth2/token"):
            messages: list[Message] = [
                {"type": "http.request", "body": b"grant_type=pass", "more_body": True},
                {"type": "http.disconnect"},
            ]
            sent: list[Message] = []

            async def receive() -> Message:
                return messages.pop(0)

            async def send(message: Message) -> None:
                sent.append(message)

            scope = {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": [(b"host", b"ragtime.test"), (b"content-type", b"application/x-www-form-urlencoded")],
                "client": ("198.51.100.93", 12345),
                "server": ("ragtime.test", 80),
            }
            with (
                mock.patch.object(rate_limit, "consume_oauth_token_rate_limit") as consume_quota,
                mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock()) as authenticate,
            ):
                await main.app(scope, receive, send)

            self.assertEqual(sent[0]["status"], 400)
            consume_quota.assert_not_called()
            authenticate.assert_not_awaited()
