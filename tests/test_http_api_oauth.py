import asyncio
import unittest
from typing import cast
from unittest import mock

from ragtime.http_api.models import HttpApiAuthMode, HttpApiConnectionConfig, HttpApiOAuthFlow
from ragtime.http_api.oauth import HttpApiOAuthManager


class OAuthManagerTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, **updates):
        values = {
            "auth_mode": HttpApiAuthMode.OAUTH2,
            "oauth_flow": HttpApiOAuthFlow.DEVICE_CODE,
            "oauth_client_id": "client",
            "oauth_device_authorization_url": "https://auth.example.test/device",
            "oauth_token_url": "https://auth.example.test/token",
        }
        values.update(updates)
        return HttpApiConnectionConfig(**values)

    async def test_device_flow_polls_once_and_returns_credentials(self) -> None:
        calls = []

        async def request(url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/device"):
                return 200, {"device_code": "device", "user_code": "ABCD", "verification_uri": "https://auth.example.test/verify"}
            return 200, {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}

        broker = mock.AsyncMock()
        broker.oauth_request_json.side_effect = request
        manager = HttpApiOAuthManager(broker)
        started = await manager.start("admin", self._config(), "https://ragtime.test/callback")
        polled = await manager.poll("admin", started.session_id)

        self.assertEqual(polled.status, "connected")
        credentials = await manager.peek_credentials("admin", started.session_id)
        self.assertEqual(credentials.access_token, "access")
        self.assertEqual(len(calls), 2)

    async def test_pkce_start_contains_s256_and_state(self) -> None:
        broker = mock.AsyncMock()
        manager = HttpApiOAuthManager(broker)
        config = self._config(
            oauth_flow=HttpApiOAuthFlow.AUTHORIZATION_CODE_PKCE,
            oauth_authorization_url="https://auth.example.test/authorize",
        )

        result = await manager.start("admin", config, "https://ragtime.test/indexes/tools/http-api/oauth/callback")

        self.assertIn("code_challenge_method=S256", result.authorization_url)
        self.assertIn("redirect_uri=https%3A%2F%2Fragtime.test%2Findexes%2Ftools%2Fhttp-api%2Foauth%2Fcallback", result.authorization_url)

    async def test_discovery_uses_rfc8414_path_then_oidc_fallback_and_validates_endpoints(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.side_effect = [
            (404, {}),
            (
                200,
                {
                    "issuer": "https://auth.example.test/tenant",
                    "authorization_endpoint": "https://auth.example.test/authorize",
                    "device_authorization_endpoint": "https://auth.example.test/device",
                    "token_endpoint": "https://auth.example.test/token",
                    "grant_types_supported": ["authorization_code"],
                    "code_challenge_methods_supported": ["S256"],
                    "scopes_supported": ["read"],
                    "token_endpoint_auth_methods_supported": ["none"],
                },
            ),
        ]
        manager = HttpApiOAuthManager(broker)

        result = await manager.discover("https://auth.example.test/tenant/")

        calls = broker.oauth_request_json.await_args_list
        self.assertTrue(calls[0].args[0].endswith("/.well-known/oauth-authorization-server/tenant"))
        self.assertTrue(calls[1].args[0].endswith("/tenant/.well-known/openid-configuration"))
        self.assertEqual(broker.validate_oauth_url.await_count, 3)
        self.assertEqual(result.issuer, "https://auth.example.test/tenant")

    async def test_discovery_rejects_issuer_mismatch_and_endpoint_security_failure(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.return_value = (200, {"issuer": "https://other.example.test"})
        with self.assertRaisesRegex(ValueError, "issuer"):
            await HttpApiOAuthManager(broker).discover("https://auth.example.test")

        broker.oauth_request_json.return_value = (200, {"issuer": "https://auth.example.test", "token_endpoint": "https://127.0.0.1/token"})
        broker.validate_oauth_url.side_effect = ValueError("target validation failed")
        with self.assertRaisesRegex(ValueError, "target validation"):
            await HttpApiOAuthManager(broker).discover("https://auth.example.test")

    async def test_device_statuses_and_interval_clamping(self) -> None:
        now = [100.0]
        broker = mock.AsyncMock()
        broker.oauth_request_json.side_effect = [
            (200, {"device_code": "device", "interval": 999}),
            (400, {"error": "authorization_pending"}),
            (400, {"error": "slow_down"}),
            (400, {"error": "access_denied"}),
        ]
        manager = HttpApiOAuthManager(broker, clock=lambda: now[0])
        started = await manager.start("admin", self._config(), "https://ragtime.test/callback")
        self.assertEqual(started.interval, 60)
        self.assertEqual((await manager.poll("admin", started.session_id)).status, "pending")
        now[0] += 60
        pending = await manager.poll("admin", started.session_id)
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.retry_after, 60)
        now[0] += 60
        self.assertEqual((await manager.poll("admin", started.session_id)).status, "failed")
        self.assertEqual(broker.oauth_request_json.await_count, 4)

    async def test_device_interval_clamps_to_one_second(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.return_value = (200, {"device_code": "device", "interval": 0})
        result = await HttpApiOAuthManager(broker).start("admin", self._config(), "https://ragtime.test/callback")
        self.assertEqual(result.interval, 1)

    async def test_device_expired_token_and_safe_provider_errors(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.side_effect = [(200, {"device_code": "device"}), (400, {"error": "expired_token", "error_description": "secret details"})]
        manager = HttpApiOAuthManager(broker)
        started = await manager.start("admin", self._config(), "https://ragtime.test/callback")
        result = await manager.poll("admin", started.session_id)
        self.assertEqual(result.status, "expired")
        self.assertNotIn("secret details", repr(result))

    async def test_duplicate_polls_are_single_flight_and_owner_is_bound(self) -> None:
        release = asyncio.Event()
        calls = 0

        async def request(url, **_kwargs):
            nonlocal calls
            if url.endswith("/device"):
                return 200, {"device_code": "device"}
            calls += 1
            await release.wait()
            return 400, {"error": "authorization_pending"}

        broker = mock.AsyncMock()
        broker.oauth_request_json.side_effect = request
        manager = HttpApiOAuthManager(broker)
        started = await manager.start("admin", self._config(), "https://ragtime.test/callback")
        first = asyncio.create_task(manager.poll("admin", started.session_id))
        await asyncio.sleep(0)
        second = asyncio.create_task(manager.poll("admin", started.session_id))
        await asyncio.sleep(0)
        self.assertEqual(calls, 1)
        release.set()
        await asyncio.gather(first, second)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await manager.poll("other", started.session_id)

    async def test_pkce_state_mismatch_replay_and_atomic_consumption(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.return_value = (200, {"access_token": "access"})
        manager = HttpApiOAuthManager(broker)
        config = self._config(oauth_flow=HttpApiOAuthFlow.AUTHORIZATION_CODE_PKCE, oauth_authorization_url="https://auth.example.test/authorize")
        started = await manager.start("admin", config, "https://ragtime.test/callback")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await manager.peek_credentials("admin", "wrong")
        state = next(session.state for session in manager._sessions.values() if session.session_id == started.session_id)
        mismatch = await manager.complete_authorization_code("wrong", "code", None)
        self.assertEqual(mismatch.status, "failed")
        connected = await manager.complete_authorization_code(state, "code", None)
        self.assertEqual(connected.status, "connected")
        replay = await manager.complete_authorization_code(state, "code", None)
        self.assertEqual(replay.status, "failed")

    async def test_authorization_callback_lookup_is_serialized_with_session_eviction(self) -> None:
        broker = mock.AsyncMock()
        broker.oauth_request_json.return_value = (200, {"access_token": "access"})
        manager = HttpApiOAuthManager(broker)
        config = self._config(
            oauth_flow=HttpApiOAuthFlow.AUTHORIZATION_CODE_PKCE,
            oauth_authorization_url="https://auth.example.test/authorize",
        )
        sessions = [await manager.start(str(index), config, "https://ragtime.test/callback") for index in range(128)]
        state = next(session.state for session in manager._sessions.values() if session.session_id == sessions[0].session_id)

        lookup_entered = asyncio.Event()
        release_lookup = asyncio.Event()

        class ManagerLockGate:
            async def __aenter__(self):
                lookup_entered.set()
                await release_lookup.wait()
                return self

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

        manager._lock = cast(asyncio.Lock, ManagerLockGate())
        callback = asyncio.create_task(manager.complete_authorization_code(state, "code", None))
        await asyncio.wait_for(lookup_entered.wait(), timeout=1)
        eviction = asyncio.create_task(manager.start("new", config, "https://ragtime.test/callback"))
        await asyncio.sleep(0)
        release_lookup.set()
        callback_result, _ = await asyncio.gather(callback, eviction)

        self.assertEqual(callback_result.status, "connected")
        self.assertEqual(len(manager._sessions), 128)

    async def test_sessions_expire_and_evict_at_128(self) -> None:
        now = [100.0]
        broker = mock.AsyncMock()
        broker.oauth_request_json.return_value = (200, {"device_code": "device"})
        manager = HttpApiOAuthManager(broker, clock=lambda: now[0])
        sessions = [await manager.start(str(index), self._config(), "https://ragtime.test/callback") for index in range(128)]
        self.assertEqual(len(manager._sessions), 128)
        oldest = sessions[0].session_id
        await manager.start("new", self._config(), "https://ragtime.test/callback")
        self.assertEqual(len(manager._sessions), 128)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await manager.poll("0", oldest)
        now[0] += 901
        with self.assertRaisesRegex(ValueError, "unavailable"):
            await manager.poll("new", sessions[-1].session_id)


if __name__ == "__main__":
    unittest.main()
