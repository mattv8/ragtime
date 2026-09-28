import asyncio
import json
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI
from prisma.enums import AuthProvider
from starlette.requests import Request

from ragtime.api import auth as api_auth
from ragtime.core import mfa_recovery
from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.mfa import PendingMfaPurpose


class RecoveryPassPrimitiveTests(unittest.TestCase):
    def test_pass_is_high_entropy_hash_only_and_constant_time_verifiable(self):
        value = mfa_recovery.generate_pass()
        stored = mfa_recovery.hash_pass(value)
        self.assertGreaterEqual(len(value), 32)
        self.assertNotIn(value, stored)
        self.assertTrue(mfa_recovery.verify_pass(value, stored))
        self.assertFalse(mfa_recovery.verify_pass("wrong", stored))

    def test_recovery_continuation_cannot_be_used_as_totp_enrollment_token(self):
        expires = mfa_recovery.utcnow() + timedelta(minutes=1)
        token = mfa_recovery.create_recovery_token(user_id="u", grant_id="g", generation=3, expires=expires)
        self.assertIsNotNone(mfa_recovery.decode_token(token, purpose="mfa:recovery_continuation"))
        self.assertIsNone(mfa_recovery.decode_token(token, purpose="mfa:recovery_totp_enrollment"))

    def test_status_derives_expiry_without_a_write(self):
        now = mfa_recovery.utcnow()
        self.assertEqual(mfa_recovery.recovery_status({"expires_at": now - timedelta(seconds=1)}), "expired")
        self.assertEqual(mfa_recovery.recovery_status({"expires_at": now + timedelta(seconds=1), "redeemed_at": now}), "redeemed")
        self.assertEqual(mfa_recovery.recovery_status({"expires_at": now - timedelta(seconds=1), "redeemed_at": now}), "expired")

    def test_enrollment_token_is_bound_to_grant_and_generation(self):
        token = mfa_recovery.create_totp_token(user_id="u", grant_id="g", generation=5, secret="secret", expires=mfa_recovery.utcnow() + timedelta(minutes=1))
        claims = mfa_recovery.decode_token(token, purpose="mfa:recovery_totp_enrollment")
        self.assertEqual(claims and claims["grant_id"], "g")
        self.assertEqual(claims and claims["generation"], 5)


@unittest.skipUnless(
    os.getenv("RAGTIME_AUTH_INTEGRATION") == "1",
    "requires RAGTIME_AUTH_INTEGRATION=1 and the recovery migration",
)
class RecoveryRouteAndTransactionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await connect_db()
        self.db = await get_db()
        suffix = uuid.uuid4().hex
        self.actor_id = f"actor-{suffix}"
        self.user_id = f"target-{suffix}"
        self.other_id = f"other-{suffix}"
        self.client_host = f"198.51.{int(suffix[:2], 16)}.{max(1, int(suffix[2:4], 16))}"
        for user_id, username, role in (
            (self.actor_id, f"admin-{suffix}", "admin"),
            (self.user_id, f"target-{suffix}", "user"),
            (self.other_id, f"other-{suffix}", "user"),
        ):
            await self.db.execute_raw(
                """INSERT INTO "users" ("id", "username", "auth_provider", "source_provider", "source_id", "role")
                   VALUES ($1, $2, 'local_managed'::"AuthProvider", 'local_managed'::"AuthProvider", $2, $3::"UserRole")""",
                user_id,
                username,
                role,
            )
        self.target = await self.db.user.find_unique(where={"id": self.user_id})
        self.actor = await self.db.user.find_unique(where={"id": self.actor_id})
        assert self.target is not None and self.actor is not None

    async def asyncTearDown(self):
        await self.db.execute_raw('DELETE FROM "users" WHERE "id" = ANY($1::text[])', [self.actor_id, self.user_id, self.other_id])
        await disconnect_db()

    def _request(self, method: str = "POST") -> Request:
        return Request(
            {
                "type": "http",
                "method": method,
                "path": "/auth/test",
                "headers": [],
                "scheme": "https",
                "client": (self.client_host, 12345),
            }
        )

    def _app(self, actor=None) -> FastAPI:
        app = FastAPI()
        app.include_router(api_auth.router)
        selected_actor = actor or self.actor
        app.dependency_overrides[api_auth.require_admin] = lambda: selected_actor
        app.dependency_overrides[api_auth.get_current_user] = lambda: selected_actor
        return app

    def _pending_token(self, *, purpose: PendingMfaPurpose = "challenge", generation: int = 0) -> str:
        return api_auth.create_pending_mfa_token(
            user_id=self.user_id,
            username=self.target.username,
            role="user",
            purpose=purpose,
            security_generation=generation,
        )

    async def _seed_issued_grant(self, grant_id: str, raw_pass: str, *, expires_delta: timedelta = timedelta(minutes=30)) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.execute_raw(
            """INSERT INTO "user_mfa_recovery_passes"
               ("id", "user_id", "pass_hash", "security_generation", "attempts", "created_at", "expires_at")
               VALUES ($1, $2, $3, 0, 0, $4::timestamp, $5::timestamp)""",
            grant_id,
            self.user_id,
            mfa_recovery.hash_pass(raw_pass),
            now,
            now + expires_delta,
        )

    async def _seed_redeemed_grant(self, grant_id: str) -> None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.execute_raw(
            """INSERT INTO "user_mfa_recovery_passes"
               ("id", "user_id", "pass_hash", "security_generation", "attempts", "created_at", "expires_at", "redeemed_at")
               VALUES ($1, $2, 'hash', 0, 0, $3::timestamp, $3::timestamp + interval '30 minutes', $3::timestamp)""",
            grant_id,
            self.user_id,
            now,
        )

    async def test_verify_cookie_dependency_and_issue_guard_bind_real_session(self):
        app = self._app()
        transport = httpx.ASGITransport(app=app, client=(self.client_host, 12345))
        auth_result = SimpleNamespace(success=True, user_id=self.actor_id)
        with mock.patch.object(api_auth, "authenticate", new=mock.AsyncMock(return_value=auth_result)):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://test",
                cookies={"ragtime_session": "real-session-cookie"},
            ) as client:
                verified = await client.post("/auth/admin/security/verify", json={"password": "correct-password"})
                self.assertEqual(verified.status_code, 200, verified.text)
                token = verified.json()["verification_token"]
                issued = await client.post(
                    f"/auth/users/{self.user_id}/mfa/recovery-pass",
                    json={"verification_token": token},
                )
        self.assertEqual(issued.status_code, 200, issued.text)
        self.assertIn("pass", issued.json())
        self.assertNotIn("set-cookie", {key.lower() for key in issued.headers})
        rows = await self.db.query_raw(
            """SELECT "user_id", "detail" FROM "auth_sync_events"
               WHERE "user_id" = $1 AND "action" = 'mfa_recovery_pass_issued'
               ORDER BY "created_at" DESC LIMIT 1""",
            self.user_id,
        )
        self.assertEqual(rows[0]["user_id"], self.user_id)
        self.assertEqual(json.loads(rows[0]["detail"]), {"actor_id": self.actor_id})

    async def test_issue_route_rejects_token_bound_to_wrong_actor_with_real_cookie_dependency(self):
        token, _expires = api_auth._create_admin_security_token(actor=self.actor, session_token="real-session-cookie")
        wrong_actor = await self.db.user.find_unique(where={"id": self.other_id})
        assert wrong_actor is not None
        app = self._app(wrong_actor)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
            cookies={"ragtime_session": "real-session-cookie"},
        ) as client:
            response = await client.post(
                f"/auth/users/{self.user_id}/mfa/recovery-pass",
                json={"verification_token": token},
            )
        self.assertEqual(response.status_code, 403, response.text)
        rows = await self.db.query_raw(
            'SELECT COUNT(*)::int AS "count" FROM "user_mfa_recovery_passes" WHERE "user_id" = $1',
            self.user_id,
        )
        self.assertEqual(int(rows[0]["count"]), 0)

    async def test_redeem_route_handles_raw_timestamp_and_sets_no_session(self):
        raw_pass = "valid-recovery-pass"
        grant_id = str(uuid.uuid4())
        await self._seed_issued_grant(grant_id, raw_pass)
        app = self._app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)), base_url="https://test") as client:
            response = await client.post(
                "/auth/mfa/recovery-pass/redeem",
                json={"mfa_challenge_token": self._pending_token(), "pass": raw_pass},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("recovery_token", response.json())
        self.assertNotIn("set-cookie", {key.lower() for key in response.headers})
        rows = await self.db.query_raw(
            'SELECT "redeemed_at" FROM "user_mfa_recovery_passes" WHERE "id" = $1',
            grant_id,
        )
        self.assertIsNotNone(rows[0]["redeemed_at"])
        self.assertEqual(await self.db.session.count(where={"userId": self.user_id}), 0)

    async def test_redeem_rejects_enroll_expiry_and_stale_generation_and_commits_attempt_limit(self):
        app = self._app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)), base_url="https://test") as client:
            grant_id = str(uuid.uuid4())
            await self._seed_issued_grant(grant_id, "pass-one")
            with self.assertRaises(api_auth.HTTPException) as enroll_error:
                await api_auth.redeem_recovery_pass(
                    self._request(),
                    api_auth.RecoveryPassRedeemRequest(
                        mfa_challenge_token=self._pending_token(purpose="enroll"),
                        **{"pass": "pass-one"},
                    ),
                )
            self.assertEqual(enroll_error.exception.status_code, 401)

            for _attempt in range(api_auth.MAX_PASS_ATTEMPTS):
                wrong = await client.post(
                    "/auth/mfa/recovery-pass/redeem",
                    json={"mfa_challenge_token": self._pending_token(), "pass": "wrong"},
                )
                self.assertEqual(wrong.status_code, 401)
            rows = await self.db.query_raw(
                'SELECT "attempts", "revoked_at" FROM "user_mfa_recovery_passes" WHERE "id" = $1',
                grant_id,
            )
            self.assertEqual(int(rows[0]["attempts"]), api_auth.MAX_PASS_ATTEMPTS)
            self.assertIsNotNone(rows[0]["revoked_at"])

            await self.db.execute_raw('DELETE FROM "user_mfa_recovery_passes" WHERE "user_id" = $1', self.user_id)
            expired_id = str(uuid.uuid4())
            await self._seed_issued_grant(expired_id, "expired-pass", expires_delta=timedelta(seconds=-1))
            with self.assertRaises(api_auth.HTTPException) as expired_error:
                await api_auth.redeem_recovery_pass(
                    self._request(),
                    api_auth.RecoveryPassRedeemRequest(
                        mfa_challenge_token=self._pending_token(),
                        **{"pass": "expired-pass"},
                    ),
                )
            self.assertEqual(expired_error.exception.status_code, 401)

            await self.db.execute_raw('DELETE FROM "user_mfa_recovery_passes" WHERE "user_id" = $1', self.user_id)
            stale_id = str(uuid.uuid4())
            await self._seed_issued_grant(stale_id, "stale-pass")
            await self.db.execute_raw('UPDATE "users" SET "security_generation" = 1 WHERE "id" = $1', self.user_id)
            with self.assertRaises(api_auth.HTTPException) as stale_error:
                await api_auth.redeem_recovery_pass(
                    self._request(),
                    api_auth.RecoveryPassRedeemRequest(
                        mfa_challenge_token=self._pending_token(),
                        **{"pass": "stale-pass"},
                    ),
                )
            self.assertEqual(stale_error.exception.status_code, 401)

    async def test_concurrent_redeem_has_one_winner(self):
        raw_pass = "race-pass"
        await self._seed_issued_grant(str(uuid.uuid4()), raw_pass)
        app = self._app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)), base_url="https://test") as client:
            payload = {"mfa_challenge_token": self._pending_token(), "pass": raw_pass}
            responses = await asyncio.gather(
                client.post("/auth/mfa/recovery-pass/redeem", json=payload),
                client.post("/auth/mfa/recovery-pass/redeem", json=payload),
            )
        self.assertEqual(sorted(response.status_code for response in responses), [200, 401])

    async def test_reissue_and_revoke_invalidate_old_continuation(self):
        raw_pass = "continuation-pass"
        await self._seed_issued_grant(str(uuid.uuid4()), raw_pass)
        app = self._app()
        admin_token, _ = api_auth._create_admin_security_token(actor=self.actor, session_token="admin-cookie")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
            cookies={"ragtime_session": "admin-cookie"},
        ) as client:
            redeemed = await client.post(
                "/auth/mfa/recovery-pass/redeem",
                json={"mfa_challenge_token": self._pending_token(), "pass": raw_pass},
            )
            self.assertEqual(redeemed.status_code, 200, redeemed.text)
            continuation = redeemed.json()["recovery_token"]
            reissued = await client.post(
                f"/auth/users/{self.user_id}/mfa/recovery-pass",
                json={"verification_token": admin_token},
            )
            self.assertEqual(reissued.status_code, 200, reissued.text)
            stale = await client.post(
                "/auth/mfa/recovery-pass/totp/start",
                json={"recovery_token": continuation},
            )
            self.assertEqual(stale.status_code, 401)
            replacement_redeem = await client.post(
                "/auth/mfa/recovery-pass/redeem",
                json={
                    "mfa_challenge_token": self._pending_token(),
                    "pass": reissued.json()["pass"],
                },
            )
            self.assertEqual(replacement_redeem.status_code, 200, replacement_redeem.text)
            replacement_continuation = replacement_redeem.json()["recovery_token"]
            revoked = await client.request(
                "DELETE",
                f"/auth/users/{self.user_id}/mfa/recovery-pass",
                json={"verification_token": admin_token},
            )
            self.assertEqual(revoked.status_code, 200, revoked.text)
            revoked_continuation = await client.post(
                "/auth/mfa/recovery-pass/totp/start",
                json={"recovery_token": replacement_continuation},
            )
            self.assertEqual(revoked_continuation.status_code, 401)
            status_response = await client.get(f"/auth/users/{self.user_id}/mfa/recovery-pass")
            self.assertEqual(status_response.json()["grant"]["status"], "revoked")

    async def test_verify_rejects_wrong_identity_role_and_generation_with_403(self):
        app = self._app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
            cookies={"ragtime_session": "real-cookie"},
        ) as client:
            with mock.patch.object(
                api_auth,
                "authenticate",
                new=mock.AsyncMock(return_value=SimpleNamespace(success=True, user_id=self.other_id)),
            ):
                wrong_actor = await client.post("/auth/admin/security/verify", json={"password": "password"})
            self.assertEqual(wrong_actor.status_code, 403)

            await self.db.execute_raw('UPDATE "users" SET "role" = \'user\'::"UserRole" WHERE "id" = $1', self.actor_id)
            with mock.patch.object(
                api_auth,
                "authenticate",
                new=mock.AsyncMock(return_value=SimpleNamespace(success=True, user_id=self.actor_id)),
            ):
                wrong_role = await client.post("/auth/admin/security/verify", json={"password": "password"})
            self.assertEqual(wrong_role.status_code, 403)

            await self.db.execute_raw(
                'UPDATE "users" SET "role" = \'admin\'::"UserRole", "security_generation" = 1 WHERE "id" = $1',
                self.actor_id,
            )
            with mock.patch.object(
                api_auth,
                "authenticate",
                new=mock.AsyncMock(return_value=SimpleNamespace(success=True, user_id=self.actor_id)),
            ):
                stale_generation = await client.post("/auth/admin/security/verify", json={"password": "password"})
            self.assertEqual(stale_generation.status_code, 403)

    async def test_profile_explicit_null_clears_email_and_name_falls_back_then_get_user_refreshes(self):
        app = self._app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
        ) as client:
            renamed = await client.patch(
                f"/auth/local/users/{self.user_id}",
                json={"display_name": "Renamed"},
            )
            self.assertEqual(renamed.status_code, 200, renamed.text)
            self.assertEqual(renamed.json()["email"], self.target.email)
            cleared = await client.patch(
                f"/auth/local/users/{self.user_id}",
                json={"display_name": None, "email": None},
            )
            self.assertEqual(cleared.status_code, 200, cleared.text)
            self.assertEqual(cleared.json()["display_name"], self.target.username)
            self.assertIsNone(cleared.json()["email"])
            refreshed = await client.get(f"/auth/users/{self.user_id}")
            directory = await client.get("/auth/users/directory")
        self.assertEqual(refreshed.status_code, 200, refreshed.text)
        self.assertEqual(refreshed.json()["display_name"], self.target.username)
        self.assertEqual(directory.status_code, 200, directory.text)

    async def test_password_path_explicit_null_uses_same_canonical_profile_values(self):
        token, _ = api_auth._create_admin_security_token(actor=self.actor, session_token="admin-cookie")
        app = self._app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
            cookies={"ragtime_session": "admin-cookie"},
        ) as client:
            response = await client.patch(
                f"/auth/local/users/{self.user_id}",
                json={
                    "password": "replacement-password",
                    "verification_token": token,
                    "display_name": None,
                    "email": None,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["display_name"], self.target.username)
        self.assertIsNone(response.json()["email"])

    async def test_missing_issue_target_is_404_and_revoke_audits_only_actual_change(self):
        token, _ = api_auth._create_admin_security_token(actor=self.actor, session_token="admin-cookie")
        app = self._app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
            cookies={"ragtime_session": "admin-cookie"},
        ) as client:
            missing = await client.post(
                f"/auth/users/missing-{uuid.uuid4()}/mfa/recovery-pass",
                json={"verification_token": token},
            )
            self.assertEqual(missing.status_code, 404, missing.text)
            await self._seed_issued_grant(str(uuid.uuid4()), "pass")
            for _ in range(2):
                revoked = await client.request(
                    "DELETE",
                    f"/auth/users/{self.user_id}/mfa/recovery-pass",
                    json={"verification_token": token},
                )
                self.assertEqual(revoked.status_code, 200, revoked.text)
        rows = await self.db.query_raw(
            """SELECT COUNT(*)::int AS "count" FROM "auth_sync_events"
               WHERE "user_id" = $1 AND "action" = 'mfa_recovery_pass_revoked' """,
            self.user_id,
        )
        self.assertEqual(int(rows[0]["count"]), 1)

    async def test_status_marks_generation_stale_grant_revoked(self):
        await self._seed_issued_grant(str(uuid.uuid4()), "pass")
        await self.db.execute_raw('UPDATE "users" SET "security_generation" = 1 WHERE "id" = $1', self.user_id)
        app = self._app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=(self.client_host, 12345)),
            base_url="https://test",
        ) as client:
            response = await client.get(f"/auth/users/{self.user_id}/mfa/recovery-pass")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["grant"]["status"], "revoked")

    async def test_finalize_rolls_back_grant_and_old_factor_on_credential_conflict(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        await self.db.usermfafactor.create(
            data={
                "userId": self.user_id,
                "factorType": "totp",
                "secretEncrypted": "old",
                "enabled": True,
            }
        )
        await self.db.userwebauthncredential.create(
            data={
                "userId": self.other_id,
                "credentialId": "duplicate",
                "publicKey": "pk",
                "name": "existing",
            }
        )
        claims = {"grant_id": grant_id, "generation": 0}
        factor = {
            "type": "webauthn",
            "jti": str(uuid.uuid4()),
            "expires_at": datetime.now(timezone.utc),
            "credential_id": "duplicate",
            "public_key": "pk2",
            "sign_count": 0,
            "transports": [],
            "aaguid": None,
            "name": "new",
        }
        with self.assertRaises(api_auth.HTTPException) as raised:
            await api_auth._finalize_recovery_factor(user=self.target, claims=claims, factor=factor, codes=["code"])
        self.assertEqual(raised.exception.status_code, 409)
        factors = await self.db.usermfafactor.find_many(where={"userId": self.user_id})
        rows = await self.db.query_raw(
            'SELECT "completed_at" FROM "user_mfa_recovery_passes" WHERE "id" = $1',
            grant_id,
        )
        user = await self.db.user.find_unique(where={"id": self.user_id})
        self.assertEqual(len(factors), 1)
        self.assertEqual(factors[0].secretEncrypted, "old")
        self.assertIsNone(rows[0]["completed_at"])
        self.assertEqual(user.securityGeneration, 0)

    async def test_admin_reset_waits_for_security_lock_before_deleting_factors(self):
        await self.db.usermfafactor.create(
            data={
                "userId": self.user_id,
                "factorType": "totp",
                "secretEncrypted": "old",
                "enabled": True,
            }
        )
        entered = asyncio.Event()
        real_revoke = api_auth.revoke_user_auth_in_transaction

        async def observed_revoke(tx, user_id, **kwargs):
            entered.set()
            return await real_revoke(tx, user_id, **kwargs)

        with (
            mock.patch.object(api_auth, "_require_admin_security_token", new=mock.AsyncMock()),
            mock.patch.object(api_auth, "revoke_user_auth_in_transaction", new=observed_revoke),
        ):
            async with self.db.tx() as blocker:
                await api_auth.lock_user_security_generation(blocker, self.user_id)
                task = asyncio.create_task(
                    api_auth.reset_user_mfa_by_admin(
                        self.user_id,
                        api_auth.VerificationTokenRequest(verification_token="verified"),
                        self._request("DELETE"),
                        self.actor,
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=2)
                factors = await self.db.usermfafactor.find_many(where={"userId": self.user_id})
                self.assertEqual(len(factors), 1, "factor deletion happened before the security row lock")
            await task
        self.assertEqual(await self.db.usermfafactor.count(where={"userId": self.user_id}), 0)
        audit = await self.db.query_raw(
            """SELECT "detail" FROM "auth_sync_events"
               WHERE "user_id" = $1 AND "action" = 'mfa_reset' ORDER BY "created_at" DESC LIMIT 1""",
            self.user_id,
        )
        self.assertEqual(json.loads(audit[0]["detail"]), {"actor_id": self.actor_id})

    async def test_password_update_waits_for_security_lock_before_writing_hash(self):
        await self.db.execute_raw('UPDATE "users" SET "password_hash" = \'old-hash\' WHERE "id" = $1', self.user_id)
        entered = asyncio.Event()
        real_revoke = api_auth.revoke_user_auth_in_transaction

        async def observed_revoke(tx, user_id, **kwargs):
            entered.set()
            return await real_revoke(tx, user_id, **kwargs)

        with (
            mock.patch.object(api_auth, "_require_admin_security_token", new=mock.AsyncMock()),
            mock.patch.object(api_auth, "revoke_user_auth_in_transaction", new=observed_revoke),
            mock.patch.object(api_auth, "_user_response", new=mock.AsyncMock(return_value={"ok": True})),
        ):
            async with self.db.tx() as blocker:
                await api_auth.lock_user_security_generation(blocker, self.user_id)
                task = asyncio.create_task(
                    api_auth.update_local_user(
                        self.user_id,
                        api_auth.LocalUserUpdateRequest(password="new-password", verification_token="verified"),
                        self._request("PATCH"),
                        self.actor,
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=2)
                rows = await self.db.query_raw('SELECT "password_hash" FROM "users" WHERE "id" = $1', self.user_id)
                self.assertEqual(rows[0]["password_hash"], "old-hash", "password changed before the security row lock")
            await task
        rows = await self.db.query_raw('SELECT "password_hash" FROM "users" WHERE "id" = $1', self.user_id)
        self.assertNotEqual(rows[0]["password_hash"], "old-hash")

    async def test_totp_start_and_valid_complete_routes_replace_factor(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        await self.db.usermfafactor.create(
            data={
                "userId": self.user_id,
                "factorType": "totp",
                "secretEncrypted": "old",
                "enabled": True,
            }
        )
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        await self.db.session.create(
            data={
                "userId": self.user_id,
                "tokenHash": f"session-{grant_id}",
                "expiresAt": future,
            }
        )
        await self.db.usermfatrusteddevice.create(
            data={
                "userId": self.user_id,
                "tokenHash": f"trusted-{grant_id}",
                "expiresAt": future,
            }
        )
        oauth_id = str(uuid.uuid4())
        await self.db.execute_raw(
            """INSERT INTO "oauth_grants"
               ("id", "user_id", "client_id", "audience", "scope", "expires_at", "security_generation", "auth_methods")
               VALUES ($1, $2, 'client', 'audience', '', $3::timestamp, 0, '[]'::jsonb)""",
            oauth_id,
            self.user_id,
            future.replace(tzinfo=None),
        )
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        recovery_token = mfa_recovery.create_recovery_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            expires=expires,
        )
        with (
            mock.patch.object(api_auth, "get_allowed_mfa_methods", new=mock.AsyncMock(return_value=["totp"])),
            mock.patch.object(api_auth, "get_app_settings", new=mock.AsyncMock(return_value={"server_name": "Recovery Test"})),
        ):
            start = await api_auth.start_recovery_totp(
                api_auth.RecoveryTotpStartRequest(recovery_token=recovery_token),
            )
            start_body = json.loads(bytes(start.body))
            self.assertIn("issuer=Recovery%20Test", start_body["otpauth_uri"])
            code = api_auth.generate_totp_code(start_body["secret"])
            complete = await api_auth.complete_recovery_totp(
                self._request(),
                api_auth.RecoveryTotpCompleteRequest(
                    recovery_token=recovery_token,
                    enrollment_token=start_body["enrollment_token"],
                    code=code,
                ),
            )
        complete_body = json.loads(bytes(complete.body))
        self.assertTrue(complete_body["success"])
        self.assertEqual(len(complete_body["recovery_codes"]), api_auth.RECOVERY_CODE_COUNT)
        self.assertNotIn("set-cookie", {key.lower() for key in complete.headers})
        self.assertEqual(await self.db.session.count(where={"userId": self.user_id}), 0)
        self.assertEqual(await self.db.usermfatrusteddevice.count(where={"userId": self.user_id}), 0)
        oauth_rows = await self.db.query_raw('SELECT "revoked_at" FROM "oauth_grants" WHERE "id" = $1', oauth_id)
        self.assertIsNotNone(oauth_rows[0]["revoked_at"])
        factors = await self.db.usermfafactor.find_many(where={"userId": self.user_id})
        self.assertEqual(len(factors), 1)
        self.assertNotEqual(factors[0].secretEncrypted, "old")
        rows = await self.db.query_raw(
            'SELECT "completed_at" FROM "user_mfa_recovery_passes" WHERE "id" = $1',
            grant_id,
        )
        self.assertIsNotNone(rows[0]["completed_at"])

    async def test_webauthn_start_route_returns_recovery_bound_registration_token(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        recovery_token = mfa_recovery.create_recovery_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            expires=expires,
        )
        challenge = SimpleNamespace(
            challenge=b"challenge",
            jti=str(uuid.uuid4()),
            exp=expires,
        )
        with (
            mock.patch.object(api_auth, "get_allowed_mfa_methods", new=mock.AsyncMock(return_value=["webauthn"])),
            mock.patch.object(
                api_auth,
                "begin_webauthn_registration",
                new=mock.AsyncMock(return_value=({"challenge": "options"}, "normal-token")),
            ),
            mock.patch.object(api_auth, "decode_registration_challenge", return_value=challenge),
        ):
            response = await api_auth.start_recovery_webauthn(
                self._request(),
                api_auth.RecoveryWebauthnStartRequest(recovery_token=recovery_token),
            )
        body = json.loads(bytes(response.body))
        claims = mfa_recovery.decode_token(
            body["registration_token"],
            purpose="mfa:recovery_webauthn_enrollment",
        )
        self.assertEqual(body["options"], {"challenge": "options"})
        self.assertEqual(claims and claims["sub"], self.user_id)
        self.assertEqual(claims and claims["grant_id"], grant_id)
        self.assertEqual(claims and claims["generation"], 0)

    async def test_webauthn_complete_bad_proof_preserves_then_valid_proof_replaces(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        await self.db.usermfafactor.create(
            data={
                "userId": self.user_id,
                "factorType": "totp",
                "secretEncrypted": "old",
                "enabled": True,
            }
        )
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        recovery_token = mfa_recovery.create_recovery_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            expires=expires,
        )
        registration_token = mfa_recovery.create_webauthn_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            challenge="Y2hhbGxlbmdl",
            jti=str(uuid.uuid4()),
            expires=expires,
        )
        verified = {
            "jti": str(uuid.uuid4()),
            "expires_at": expires,
            "credential_id": f"credential-{grant_id}",
            "public_key": "public-key",
            "sign_count": 0,
            "transports": [],
            "aaguid": None,
        }
        with (
            mock.patch.object(api_auth, "get_allowed_mfa_methods", new=mock.AsyncMock(return_value=["webauthn"])),
            mock.patch.object(api_auth, "verify_webauthn_registration_pure", side_effect=api_auth.WebauthnError("bad proof")),
        ):
            with self.assertRaises(api_auth.HTTPException) as bad:
                await api_auth.complete_recovery_webauthn(
                    self._request(),
                    api_auth.RecoveryWebauthnCompleteRequest(
                        recovery_token=recovery_token,
                        registration_token=registration_token,
                        credential={"id": "bad"},
                        name="Recovered key",
                    ),
                )
        self.assertEqual(bad.exception.status_code, 401)
        self.assertEqual(await self.db.usermfafactor.count(where={"userId": self.user_id}), 1)
        rows = await self.db.query_raw(
            'SELECT "completed_at" FROM "user_mfa_recovery_passes" WHERE "id" = $1',
            grant_id,
        )
        self.assertIsNone(rows[0]["completed_at"])

        with (
            mock.patch.object(api_auth, "get_allowed_mfa_methods", new=mock.AsyncMock(return_value=["webauthn"])),
            mock.patch.object(api_auth, "verify_webauthn_registration_pure", return_value=verified),
        ):
            completed = await api_auth.complete_recovery_webauthn(
                self._request(),
                api_auth.RecoveryWebauthnCompleteRequest(
                    recovery_token=recovery_token,
                    registration_token=registration_token,
                    credential={"id": "valid"},
                    name="Recovered key",
                ),
            )
        self.assertTrue(json.loads(bytes(completed.body))["success"])
        self.assertEqual(await self.db.usermfafactor.count(where={"userId": self.user_id}), 0)
        self.assertEqual(await self.db.userwebauthncredential.count(where={"userId": self.user_id}), 1)

    async def test_concurrent_totp_finalize_has_one_winner(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        recovery_token = mfa_recovery.create_recovery_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            expires=expires,
        )
        secret = api_auth.generate_totp_secret()
        enrollment_token = mfa_recovery.create_totp_token(
            user_id=self.user_id,
            grant_id=grant_id,
            generation=0,
            secret=secret,
            expires=expires,
        )
        body = api_auth.RecoveryTotpCompleteRequest(
            recovery_token=recovery_token,
            enrollment_token=enrollment_token,
            code=api_auth.generate_totp_code(secret),
        )
        with mock.patch.object(api_auth, "get_allowed_mfa_methods", new=mock.AsyncMock(return_value=["totp"])):
            results = await asyncio.gather(
                api_auth.complete_recovery_totp(self._request(), body),
                api_auth.complete_recovery_totp(self._request(), body),
                return_exceptions=True,
            )
        successes = [result for result in results if not isinstance(result, Exception)]
        failures = [result for result in results if isinstance(result, api_auth.HTTPException)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].status_code, 401)
        self.assertEqual(await self.db.usermfafactor.count(where={"userId": self.user_id}), 1)

    async def test_normal_enrollment_starts_reject_recovery_continuation_purpose(self):
        token = mfa_recovery.create_recovery_token(
            user_id=self.user_id,
            grant_id=str(uuid.uuid4()),
            generation=0,
            expires=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        with self.assertRaises(api_auth.HTTPException) as totp_error:
            await api_auth.start_mfa_enrollment(
                api_auth.MfaEnrollStartRequest(mfa_challenge_token=token),
                current_user=None,
            )
        with self.assertRaises(api_auth.HTTPException) as webauthn_error:
            await api_auth.start_webauthn_registration(
                self._request(),
                api_auth.WebauthnRegisterStartRequest(mfa_challenge_token=token),
                current_user=None,
            )
        self.assertEqual(totp_error.exception.status_code, 401)
        self.assertEqual(webauthn_error.exception.status_code, 401)

    async def test_stale_finalize_cas_preserves_existing_factor(self):
        grant_id = str(uuid.uuid4())
        await self._seed_redeemed_grant(grant_id)
        await self.db.usermfafactor.create(
            data={
                "userId": self.user_id,
                "factorType": "totp",
                "secretEncrypted": "old",
                "enabled": True,
            }
        )
        await self.db.execute_raw('UPDATE "users" SET "security_generation" = 1 WHERE "id" = $1', self.user_id)
        with self.assertRaises(api_auth.HTTPException) as raised:
            await api_auth._finalize_recovery_factor(
                user=self.target,
                claims={"grant_id": grant_id, "generation": 0},
                factor={"type": "totp", "secret": "new", "time_step": 1},
                codes=["code"],
            )
        self.assertEqual(raised.exception.status_code, 401)
        factors = await self.db.usermfafactor.find_many(where={"userId": self.user_id})
        self.assertEqual([factor.secretEncrypted for factor in factors], ["old"])
