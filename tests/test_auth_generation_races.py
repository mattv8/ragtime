import os
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from ragtime.api import auth as api_auth
from ragtime.core import mfa, webauthn_mfa
from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.oauth_grants import OAuthGrantError, revoke_user_auth


@unittest.skipUnless(
    os.getenv("RAGTIME_AUTH_INTEGRATION") == "1",
    "requires RAGTIME_AUTH_INTEGRATION=1 and the additive OAuth grant migration",
)
class AuthGenerationRaceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await connect_db()
        self.db = await get_db()
        self.user_id = str(uuid.uuid4())
        self.username = f"local:auth-generation-race-{self.user_id}"
        await self.db.execute_raw(
            'INSERT INTO "users" ("id", "username", "auth_provider") VALUES ($1, $2, \'local_managed\'::"AuthProvider")',
            self.user_id,
            self.username,
        )

    async def asyncTearDown(self) -> None:
        await self.db.execute_raw('DELETE FROM "users" WHERE "id" = $1', self.user_id)
        await disconnect_db()

    def _user_snapshot(self, generation: int = 0) -> SimpleNamespace:
        return SimpleNamespace(
            id=self.user_id,
            username=self.username,
            role="user",
            securityGeneration=generation,
        )

    def _request(self) -> Request:
        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/auth/mfa/enroll/complete",
                "headers": [(b"host", b"ragtime.test")],
                "scheme": "https",
            }
        )

    async def _session_count(self) -> int:
        rows = await self.db.query_raw('SELECT COUNT(*)::int AS "count" FROM "sessions" WHERE "user_id" = $1', self.user_id)
        return int(rows[0]["count"])

    async def test_reset_after_confirm_rejects_conditional_invalidation_without_session(self) -> None:
        async def confirm_then_admin_reset(*_args, **_kwargs):
            await revoke_user_auth(self.user_id)
            return True, []

        with mock.patch.object(api_auth, "confirm_totp_enrollment", new=confirm_then_admin_reset):
            with self.assertRaises(HTTPException) as raised:
                await api_auth.complete_mfa_enrollment(
                    self._request(),
                    Response(),
                    api_auth.MfaEnrollCompleteRequest(code="123456", enrollment_token="test"),
                    self._user_snapshot(),
                )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(await self._session_count(), 0)

    async def test_reset_after_conditional_bump_rejects_replacement_session(self) -> None:
        async def confirmed(*_args, **_kwargs):
            return True, []

        original_issue = api_auth._issue_login_session

        async def reset_before_issue(*args, **kwargs):
            await revoke_user_auth(self.user_id)
            return await original_issue(*args, **kwargs)

        with (
            mock.patch.object(api_auth, "confirm_totp_enrollment", new=confirmed),
            mock.patch.object(api_auth, "_issue_login_session", new=reset_before_issue),
        ):
            with self.assertRaises(HTTPException) as raised:
                await api_auth.complete_mfa_enrollment(
                    self._request(),
                    Response(),
                    api_auth.MfaEnrollCompleteRequest(code="123456", enrollment_token="test"),
                    self._user_snapshot(),
                )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(await self._session_count(), 0)

    async def test_stale_snapshot_cannot_start_totp_or_webauthn_continuations(self) -> None:
        await self.db.execute_raw('UPDATE "users" SET "security_generation" = 4 WHERE "id" = $1', self.user_id)
        stale_user = self._user_snapshot(3)

        with self.assertRaisesRegex(ValueError, "Authentication is no longer valid"):
            await mfa.begin_totp_enrollment(stale_user, security_generation=3)
        with self.assertRaisesRegex(webauthn_mfa.WebauthnError, "Authentication is no longer valid"):
            await webauthn_mfa.begin_webauthn_registration(stale_user, self._request(), security_generation=3)
        with self.assertRaisesRegex(webauthn_mfa.WebauthnError, "Authentication is no longer valid"):
            await webauthn_mfa.begin_webauthn_authentication(stale_user, self._request(), security_generation=3)

    async def test_nonzero_snapshot_enrollment_and_conditional_reset_return_written_generation(self) -> None:
        await self.db.execute_raw('UPDATE "users" SET "security_generation" = 7 WHERE "id" = $1', self.user_id)
        setup = await mfa.begin_totp_enrollment(self._user_snapshot(7), security_generation=7)
        claims = mfa.decode_totp_enrollment_token(setup["enrollment_token"])
        self.assertIsNotNone(claims)
        assert claims is not None
        self.assertEqual(claims.security_generation, 7)

        self.assertEqual(await revoke_user_auth(self.user_id, expected_generation=7), 8)
        with self.assertRaisesRegex(OAuthGrantError, "security_generation_mismatch"):
            await revoke_user_auth(self.user_id, expected_generation=7)
        rows = await self.db.query_raw('SELECT "security_generation" FROM "users" WHERE "id" = $1', self.user_id)
        self.assertEqual(int(rows[0]["security_generation"]), 8)
