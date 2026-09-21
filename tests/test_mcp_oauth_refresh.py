"""Live PostgreSQL integration coverage for refreshable interactive MCP OAuth.

Run inside ``ragtime-dev`` only after the auth-timeout migration and generated
Prisma client are current:

    RAGTIME_AUTH_INTEGRATION=1 python -m pytest tests/test_mcp_oauth_refresh.py -q
"""

import asyncio
import base64
import hashlib
import hmac
import os
import re
import unittest
from datetime import timedelta
from typing import Any
from unittest import mock
from uuid import uuid4

from prisma import Prisma
from prisma.enums import AuthProvider, McpAuthMethod
from prisma.types import AuthProviderConfigCreateInput, AuthProviderConfigUpsertInput, McpRouteConfigCreateInput, UserCreateInput
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ragtime.api import auth as api_auth
from ragtime.config.settings import settings
from ragtime.core.auth import AuthResult, decode_jwt_payload, encode_jwt_payload, hash_token, validate_session
from ragtime.core.database import _manager
from ragtime.core.datetimes import utc_now
from ragtime.core.oauth_grants import (
    create_grant,
    get_active_grant,
    revoke_grant,
    revoke_user_auth,
)
from ragtime.mcp.user_oauth import (
    McpOAuthError,
    derive_mcp_refresh_successor,
    issue_mcp_token_pair,
    refresh_mcp_token_pair,
    revoke_mcp_token,
    validate_mcp_token_and_fetch_user,
)


class McpRefreshSuccessorTests(unittest.TestCase):
    def test_successor_has_frozen_known_answer(self) -> None:
        with mock.patch.object(settings, "encryption_key", "test-encryption-key"):
            successor = derive_mcp_refresh_successor("refresh-token")

        self.assertEqual(successor, "VOJVggUXt6Yrf-eJevewmgccVT8jFIA4KN8hLRYke1o")

    def test_successor_changes_with_key_and_domain(self) -> None:
        refresh_token = "refresh-token"
        with mock.patch.object(settings, "encryption_key", "test-encryption-key"):
            successor = derive_mcp_refresh_successor(refresh_token)
        with mock.patch.object(settings, "encryption_key", "another-encryption-key"):
            changed_key_successor = derive_mcp_refresh_successor(refresh_token)

        old_domain_successor = (
            base64.urlsafe_b64encode(
                hmac.new(
                    b"test-encryption-key",
                    f"ragtime.mcp.refresh:{refresh_token}".encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        self.assertNotEqual(successor, changed_key_successor)
        self.assertNotEqual(successor, old_domain_successor)


@unittest.skipUnless(
    os.environ.get("RAGTIME_AUTH_INTEGRATION") == "1",
    "set RAGTIME_AUTH_INTEGRATION=1 to run against local Prisma/Postgres",
)
class McpOAuthRefreshDatabaseTests(unittest.IsolatedAsyncioTestCase):
    """Use real grant and refresh-token records; never replace Prisma with a mock."""

    issuer = "http://ragtime.test"
    audience = "http://ragtime.test/mcp"
    client_id = "refresh-integration-client"

    async def asyncSetUp(self) -> None:
        self.db = Prisma()
        self.restarted_db = Prisma()
        await self.db.connect()
        await self.restarted_db.connect()
        self.previous_db = _manager._db
        _manager._db = self.db
        self.previous_auth_config = await self.db.authproviderconfig.find_unique(where={"id": "default"})
        self.user_id = str(uuid4())
        self.username = f"oauth-refresh-{self.user_id}"
        user_data = UserCreateInput(
            id=self.user_id,
            username=self.username,
            authProvider=AuthProvider.local,
        )
        await self.db.user.create(data=user_data)
        self.route_path = f"refresh-route-{uuid4().hex}"
        route_data = McpRouteConfigCreateInput(
            name=self.route_path,
            routePath=self.route_path,
            enabled=True,
            requireAuth=True,
            authMethod=McpAuthMethod.oauth2,
        )
        await self.db.mcprouteconfig.create(data=route_data)
        self.audience = f"{self.issuer}/mcp/{self.route_path}"
        self.mcp_settings = {
            "mcp_enabled": True,
            "mcp_default_route_auth": True,
            "mcp_default_route_auth_method": "oauth2",
        }

    async def asyncTearDown(self) -> None:
        # Grant records cascade from this uniquely-prefixed disposable user.
        await self.db.mcprouteconfig.delete_many(where={"routePath": self.route_path})
        await self.db.user.delete_many(where={"id": self.user_id})
        if self.previous_auth_config is None:
            await self.db.authproviderconfig.delete_many(where={"id": "default"})
        else:
            await self.db.authproviderconfig.update(
                where={"id": "default"},
                data={"webSessionHours": self.previous_auth_config.webSessionHours},
            )
        _manager._db = self.previous_db
        await self.db.disconnect()
        await self.restarted_db.disconnect()

    async def test_issued_pair_is_grant_backed_and_creates_no_web_session(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="tools.read",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )

        payload = decode_jwt_payload(pair["access_token"], audience=self.audience)
        assert payload is not None
        self.assertEqual(payload["token_use"], "mcp_access")
        self.assertEqual(payload["sub"], self.user_id)
        self.assertEqual(payload["client_id"], self.client_id)
        self.assertEqual(payload["scope"], "tools.read")
        self.assertEqual(payload["iss"], self.issuer)
        self.assertIn("grant_id", payload)
        self.assertGreater(pair["expires_in"], 0)
        self.assertEqual(pair["token_type"], "Bearer")
        self.assertEqual(
            await self.db.session.count(where={"userId": self.user_id}),
            0,
            "authorization-code MCP issuance must not create an application session",
        )
        grant = await get_active_grant(payload["grant_id"])
        assert grant is not None
        self.assertEqual(grant.audience, self.audience)
        self.assertEqual(grant.scope, "tools.read")

    async def test_password_grant_expires_in_matches_issued_session_lifetime_override(self) -> None:
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "path": "/token",
                "headers": [(b"host", b"ragtime.test")],
            }
        )
        auth_result = AuthResult(
            success=True,
            user_id=self.user_id,
            username=self.username,
            role="user",
        )

        for override_hours in (48, 12):
            auth_config_data = AuthProviderConfigUpsertInput(
                create=AuthProviderConfigCreateInput(id="default", webSessionHours=override_hours),
                update={"webSessionHours": override_hours},
            )
            await self.db.authproviderconfig.upsert(
                where={"id": "default"},
                data=auth_config_data,
            )
            response = Response()
            with (
                mock.patch.object(api_auth.settings, "jwt_expire_hours", 24),
                mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)),
                mock.patch.object(
                    api_auth,
                    "get_auth_provider_config",
                    new=mock.AsyncMock(side_effect=AssertionError("password response must not refetch auth configuration")),
                ),
                mock.patch.object(api_auth, "mfa_needed_for_user", new=mock.AsyncMock(return_value=False)),
            ):
                result = await api_auth.oauth2_token(
                    request=request,
                    response=response,
                    grant_type="password",
                    username=self.username,
                    password="correct-password",
                    totp_code=None,
                    remember_device=False,
                    code=None,
                    code_verifier=None,
                    redirect_uri=None,
                    client_id=None,
                    scope=None,
                    resource=None,
                    refresh_token=None,
                )

            self.assertIsInstance(result, api_auth.OAuth2TokenResponse)
            assert isinstance(result, api_auth.OAuth2TokenResponse)
            token_payload = decode_jwt_payload(result.access_token)
            assert token_payload is not None
            session = await self.db.session.find_first(where={"tokenHash": hash_token(result.access_token)})
            assert session is not None
            max_age_match = re.search(r"Max-Age=(\d+)", response.headers["set-cookie"])
            assert max_age_match is not None

            expected_lifetime = override_hours * 3600
            self.assertAlmostEqual(token_payload["exp"] - utc_now().timestamp(), expected_lifetime, delta=1)
            self.assertAlmostEqual(session.expiresAt.timestamp(), token_payload["exp"], delta=1)
            self.assertEqual(int(max_age_match.group(1)), expected_lifetime)
            self.assertAlmostEqual(result.expires_in, token_payload["exp"] - utc_now().timestamp(), delta=1)

    async def test_refresh_replay_inside_overlap_returns_same_active_successor_after_client_restart(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )
        refreshed = await refresh_mcp_token_pair(
            refresh_token=pair["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertNotEqual(refreshed["refresh_token"], pair["refresh_token"])
        self.assertGreater(refreshed["expires_in"], 0)

        # A new Prisma client represents a process/client restart: duplicate
        # resolution must depend only on persisted grant and refresh records.
        _manager._db = self.restarted_db
        replayed = await refresh_mcp_token_pair(
            refresh_token=pair["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertEqual(replayed["refresh_token"], refreshed["refresh_token"])
        payload = decode_jwt_payload(refreshed["access_token"], audience=self.audience)
        assert payload is not None
        self.assertEqual(await self.restarted_db.oauthrefreshtoken.count(where={"grantId": payload["grant_id"]}), 2)

        for result in (refreshed, replayed):
            token_data, user = await validate_mcp_token_and_fetch_user(
                result["access_token"],
                resource=self.audience,
                issuer=self.issuer,
            )
            self.assertIsNotNone(token_data)
            self.assertIsNotNone(user)

        successor = await refresh_mcp_token_pair(
            refresh_token=refreshed["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertNotEqual(successor["refresh_token"], refreshed["refresh_token"])

    async def test_refresh_replay_after_fixed_overlap_revokes_winner_access(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )
        refreshed = await refresh_mcp_token_pair(
            refresh_token=pair["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )

        # Do not sleep: first model an in-window retry, then move the original
        # consumption outside the fixed (and non-sliding) overlap interval.
        consumed_at = utc_now().replace(microsecond=0, tzinfo=None) - timedelta(seconds=5)
        await self.db.execute_raw(
            'UPDATE "oauth_refresh_tokens" SET "consumed_at" = $1::timestamp WHERE "token_hash" = $2',
            consumed_at,
            hash_token(pair["refresh_token"]),
        )
        replayed = await refresh_mcp_token_pair(
            refresh_token=pair["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertEqual(replayed["refresh_token"], refreshed["refresh_token"])
        original_record = await self.db.oauthrefreshtoken.find_unique(where={"tokenHash": hash_token(pair["refresh_token"])})
        assert original_record is not None
        assert original_record.consumedAt is not None
        self.assertEqual(original_record.consumedAt.replace(tzinfo=None), consumed_at)

        await self.db.execute_raw(
            'UPDATE "oauth_refresh_tokens" SET "consumed_at" = $1::timestamp WHERE "token_hash" = $2',
            (utc_now() - timedelta(seconds=11)).replace(tzinfo=None),
            hash_token(pair["refresh_token"]),
        )
        with self.assertRaises(McpOAuthError) as replay:
            await refresh_mcp_token_pair(
                refresh_token=pair["refresh_token"],
                client_id=self.client_id,
                resource=None,
                scope=None,
                issuer=self.issuer,
            )
        self.assertEqual(replay.exception.error, "invalid_grant")

        token_data, user = await validate_mcp_token_and_fetch_user(
            refreshed["access_token"],
            resource=self.audience,
            issuer=self.issuer,
        )
        self.assertIsNone(token_data)
        self.assertIsNone(user)

    async def test_revoked_grant_is_rejected_before_successor_derivation_or_refresh_mutation(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )
        payload = decode_jwt_payload(pair["access_token"], audience=self.audience)
        assert payload is not None
        await revoke_grant(payload["grant_id"])
        before = await self.db.oauthrefreshtoken.find_unique(where={"tokenHash": hash_token(pair["refresh_token"])})
        assert before is not None
        self.assertIsNone(before.consumedAt)

        with mock.patch(
            "ragtime.mcp.user_oauth.derive_mcp_refresh_successor",
            side_effect=AssertionError("revoked grants must not derive successors"),
        ) as derive:
            with self.assertRaises(McpOAuthError) as rejected:
                await refresh_mcp_token_pair(
                    refresh_token=pair["refresh_token"],
                    client_id=self.client_id,
                    resource=None,
                    scope=None,
                    issuer=self.issuer,
                )
        self.assertEqual(rejected.exception.error, "invalid_grant")
        derive.assert_not_called()

        after = await self.db.oauthrefreshtoken.find_unique(where={"tokenHash": hash_token(pair["refresh_token"])})
        assert after is not None
        self.assertIsNone(after.consumedAt)
        self.assertEqual(await self.db.oauthrefreshtoken.count(where={"grantId": payload["grant_id"]}), 1)

    async def test_concurrent_refreshes_share_successor_and_leave_grant_active(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="tools.read",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )

        async def refresh() -> dict[str, Any]:
            return await refresh_mcp_token_pair(
                refresh_token=pair["refresh_token"],
                client_id=self.client_id,
                resource=None,
                scope=None,
                issuer=self.issuer,
            )

        results = await asyncio.gather(refresh(), refresh(), return_exceptions=True)
        self.assertTrue(all(not isinstance(result, Exception) for result in results), [type(result).__name__ for result in results])
        first, second = results
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        first_access_token = first["access_token"]
        first_refresh_token = first["refresh_token"]
        second_access_token = second["access_token"]
        second_refresh_token = second["refresh_token"]
        assert isinstance(first_access_token, str)
        assert isinstance(first_refresh_token, str)
        assert isinstance(second_access_token, str)
        assert isinstance(second_refresh_token, str)
        self.assertEqual(first_refresh_token, second_refresh_token)

        payload = decode_jwt_payload(first_access_token, audience=self.audience)
        assert payload is not None
        self.assertEqual(await self.db.oauthrefreshtoken.count(where={"grantId": payload["grant_id"]}), 2)
        for result in (first, second):
            access_token = result["access_token"]
            assert isinstance(access_token, str)
            token_data, user = await validate_mcp_token_and_fetch_user(access_token, resource=self.audience, issuer=self.issuer)
            self.assertIsNotNone(token_data)
            self.assertIsNotNone(user)

        successor = await refresh_mcp_token_pair(
            refresh_token=first_refresh_token,
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertNotEqual(successor["refresh_token"], first_refresh_token)

    async def test_authorization_code_http_response_has_no_cookie_and_no_store(self) -> None:
        verifier = "refresh-integration-verifier"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        code = f"refresh-integration-code-{uuid4()}"
        api_auth._auth_codes[code] = {
            "user_id": self.user_id,
            "username": self.username,
            "role": "user",
            "client_id": self.client_id,
            "redirect_uri": "https://client.example/callback",
            "code_challenge": challenge,
            "expires": utc_now().timestamp() + 60,
            "mfa_verified": True,
            "auth_methods": ["password"],
            "security_generation": 0,
            "resource": self.audience,
            "scope": "tools.read",
        }
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "path": "/token",
                "headers": [(b"host", b"ragtime.test")],
            }
        )

        response = Response()
        result = await api_auth.oauth2_token(
            request=request,
            response=response,
            grant_type="authorization_code",
            code=code,
            code_verifier=verifier,
            redirect_uri="https://client.example/callback",
            client_id=self.client_id,
            scope=None,
            resource=None,
            username=None,
            password=None,
            totp_code=None,
            remember_device=False,
            refresh_token=None,
        )
        assert isinstance(result, api_auth.OAuth2TokenResponse)
        self.assertIsNotNone(result.refresh_token)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["pragma"], "no-cache")
        self.assertNotIn("set-cookie", response.headers)

    async def test_refresh_client_and_valid_resource_mismatch_do_not_revoke_family(self) -> None:
        with mock.patch(
            "ragtime.mcp.user_oauth.get_app_settings",
            new=mock.AsyncMock(return_value=self.mcp_settings),
        ):
            pair = await issue_mcp_token_pair(
                user_id=self.user_id,
                client_id=self.client_id,
                audience=self.audience,
                scope="tools.read",
                security_generation=0,
                mfa_verified=True,
                auth_methods=["password"],
                issuer=self.issuer,
            )

            for client_id, resource in (("other-client", None), (self.client_id, f"{self.issuer}/mcp")):
                with self.assertRaises(McpOAuthError) as denied:
                    await refresh_mcp_token_pair(
                        refresh_token=pair["refresh_token"],
                        client_id=client_id,
                        resource=resource,
                        scope=None,
                        issuer=self.issuer,
                    )
                self.assertEqual(denied.exception.error, "invalid_grant")

            with self.assertRaises(McpOAuthError) as invalid_target:
                await refresh_mcp_token_pair(
                    refresh_token=pair["refresh_token"],
                    client_id=self.client_id,
                    resource=f"{self.issuer}/mcp/not-an-enabled-route",
                    scope=None,
                    issuer=self.issuer,
                )
            self.assertEqual(invalid_target.exception.error, "invalid_target")

            refreshed = await refresh_mcp_token_pair(
                refresh_token=pair["refresh_token"],
                client_id=self.client_id,
                resource=None,
                scope=None,
                issuer=self.issuer,
            )
            self.assertIn("access_token", refreshed)

    async def test_refresh_lifetime_is_capped_at_absolute_grant_expiry(self) -> None:
        expires_at = utc_now() + timedelta(seconds=30)
        refresh_token = f"fixture-refresh-{uuid4()}"
        grant = await create_grant(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="tools.read",
            expires_at=expires_at,
            security_generation=0,
            mfa_verified_at=utc_now(),
            auth_methods=["password"],
            refresh_hash=hash_token(refresh_token),
        )

        pair = await refresh_mcp_token_pair(
            refresh_token=refresh_token,
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        replayed = await refresh_mcp_token_pair(
            refresh_token=refresh_token,
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertEqual(replayed["refresh_token"], pair["refresh_token"])
        for result in (pair, replayed):
            payload = decode_jwt_payload(result["access_token"], audience=self.audience)
            assert payload is not None
            self.assertLessEqual(payload["exp"], int(expires_at.timestamp()))
            self.assertLessEqual(result["expires_in"], 30)
        self.assertIsNotNone(await get_active_grant(grant.id))

    async def test_wrong_issuer_or_route_rejects_valid_access_token(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password"],
            issuer=self.issuer,
        )
        payload = decode_jwt_payload(pair["access_token"], audience=self.audience)
        assert payload is not None

        wrong_issuer_token_data, wrong_issuer_user = await validate_mcp_token_and_fetch_user(
            pair["access_token"],
            resource=self.audience,
            issuer="http://another-issuer.test",
        )
        self.assertIsNone(wrong_issuer_token_data)
        self.assertIsNone(wrong_issuer_user)
        wrong_route_token_data, wrong_route_user = await validate_mcp_token_and_fetch_user(
            pair["access_token"],
            resource=f"{self.issuer}/mcp/another-route",
            issuer=self.issuer,
        )
        self.assertIsNone(wrong_route_token_data)
        self.assertIsNone(wrong_route_user)

    async def test_access_and_refresh_revocation_each_kill_the_grant_family(self) -> None:
        for token_kind in ("access_token", "refresh_token"):
            pair = await issue_mcp_token_pair(
                user_id=self.user_id,
                client_id=self.client_id,
                audience=self.audience,
                scope="",
                security_generation=0,
                mfa_verified=True,
                auth_methods=["password"],
                issuer=self.issuer,
            )
            await revoke_mcp_token(token=pair[token_kind], client_id=self.client_id, issuer=self.issuer)
            token_data, user = await validate_mcp_token_and_fetch_user(pair["access_token"], resource=self.audience, issuer=self.issuer)
            self.assertIsNone(token_data)
            self.assertIsNone(user)

    async def test_expired_access_can_refresh_but_is_never_a_web_session(self) -> None:
        pair = await issue_mcp_token_pair(
            user_id=self.user_id,
            client_id=self.client_id,
            audience=self.audience,
            scope="",
            security_generation=0,
            mfa_verified=True,
            auth_methods=["password", "totp"],
            issuer=self.issuer,
        )
        payload = decode_jwt_payload(pair["access_token"], audience=self.audience)
        assert payload is not None
        payload["exp"] = int((utc_now() - timedelta(minutes=1)).timestamp())
        expired_access = encode_jwt_payload(payload)
        self.assertIsNone(await validate_session(expired_access))
        refreshed = await refresh_mcp_token_pair(
            refresh_token=pair["refresh_token"],
            client_id=self.client_id,
            resource=None,
            scope=None,
            issuer=self.issuer,
        )
        self.assertGreater(refreshed["expires_in"], 0)

    async def test_mfa_policy_tightening_rejects_existing_non_mfa_grant(self) -> None:
        with mock.patch("ragtime.mcp.user_oauth.mfa_needed_for_user", new=mock.AsyncMock(return_value=False)):
            pair = await issue_mcp_token_pair(
                user_id=self.user_id,
                client_id=self.client_id,
                audience=self.audience,
                scope="",
                security_generation=0,
                mfa_verified=False,
                auth_methods=["password"],
                issuer=self.issuer,
            )
        with mock.patch("ragtime.mcp.user_oauth.mfa_needed_for_user", new=mock.AsyncMock(return_value=True)):
            token_data, user = await validate_mcp_token_and_fetch_user(
                pair["access_token"],
                resource=self.audience,
                issuer=self.issuer,
            )
        self.assertIsNone(token_data)
        self.assertIsNone(user)

    async def test_stale_authorization_code_is_denied_after_user_auth_revocation(self) -> None:
        verifier = "refresh-integration-stale-verifier"
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        code = f"refresh-integration-stale-code-{uuid4()}"
        api_auth._auth_codes[code] = {
            "user_id": self.user_id,
            "username": self.username,
            "role": "user",
            "client_id": self.client_id,
            "redirect_uri": "https://client.example/callback",
            "code_challenge": challenge,
            "expires": utc_now().timestamp() + 60,
            "mfa_verified": True,
            "auth_methods": ["password"],
            "security_generation": 0,
            "resource": self.audience,
            "scope": "",
        }
        await revoke_user_auth(self.user_id)
        response = await api_auth.oauth2_token(
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/token",
                    "headers": [(b"host", b"ragtime.test")],
                }
            ),
            response=Response(),
            grant_type="authorization_code",
            code=code,
            code_verifier=verifier,
            redirect_uri="https://client.example/callback",
            client_id=self.client_id,
            scope=None,
            resource=None,
            username=None,
            password=None,
            totp_code=None,
            remember_device=False,
            refresh_token=None,
        )
        assert isinstance(response, JSONResponse)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.body, b'{"error":"invalid_grant","error_description":"authorization is no longer valid"}')
