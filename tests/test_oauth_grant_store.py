import asyncio
import os
import unittest
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.oauth_grants import (
    MAX_DUPLICATE_GRACE_SECONDS,
    OAuthGrantError,
    cleanup_expired_grants,
    create_grant,
    get_active_grant,
    lookup_refresh_grant,
    revoke_user_auth,
    rotate_refresh_token,
)


@unittest.skipUnless(
    os.getenv("RAGTIME_AUTH_INTEGRATION") == "1",
    "requires RAGTIME_AUTH_INTEGRATION=1 and the additive OAuth grant migration",
)
class OAuthGrantStoreIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await connect_db()
        self.db = await get_db()
        self.user_id = str(uuid.uuid4())
        await self.db.execute_raw(
            'INSERT INTO "users" ("id", "username", "auth_provider") VALUES ($1, $2, \'local_managed\'::"AuthProvider")',
            self.user_id,
            f"local:oauth-grant-test-{self.user_id}",
        )

    async def asyncTearDown(self) -> None:
        await self.db.execute_raw('DELETE FROM "users" WHERE "id" = $1', self.user_id)
        await disconnect_db()

    async def _create(self, *, expires_at: datetime | None = None, refresh_hash: str = "refresh-1"):
        return await create_grant(
            user_id=self.user_id,
            client_id="public-client",
            audience="https://ragtime.test/mcp",
            scope="read",
            expires_at=expires_at or datetime.now(timezone.utc) + timedelta(days=1),
            security_generation=0,
            mfa_verified_at=None,
            auth_methods=["password"],
            refresh_hash=refresh_hash,
        )

    async def test_create_rotate_and_replay_revokes_the_family(self) -> None:
        expiry = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0)
        grant = await self._create(expires_at=expiry)
        self.assertIsInstance(grant.expires_at, datetime)
        self.assertEqual(grant.expires_at.tzinfo, timezone.utc)
        self.assertEqual(grant.expires_at, expiry)
        looked_up = await lookup_refresh_grant("refresh-1")
        self.assertIsNotNone(looked_up)
        assert looked_up is not None
        self.assertEqual(looked_up.expires_at, expiry)
        rotated = await rotate_refresh_token(refresh_hash="refresh-1", next_refresh_hash="refresh-2", client_id="public-client")
        self.assertEqual(rotated.id, grant.id)

        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(refresh_hash="refresh-1", next_refresh_hash="refresh-3", client_id="public-client")
        self.assertIsNone(await get_active_grant(grant.id))

    async def test_expired_grant_cannot_rotate(self) -> None:
        grant = await self._create()
        await self.db.execute_raw(
            'UPDATE "oauth_grants" SET "expires_at" = $1::timestamp WHERE "id" = $2',
            (datetime.now(timezone.utc) - timedelta(seconds=1)).replace(tzinfo=None),
            grant.id,
        )
        with self.assertRaisesRegex(OAuthGrantError, "expired_grant"):
            await rotate_refresh_token(refresh_hash="refresh-1", next_refresh_hash="refresh-2", client_id="public-client")

    async def test_wrong_client_resource_and_scope_leave_grant_active(self) -> None:
        grant = await self._create()
        for client_id, audience, scope in (
            ("other-client", None, None),
            ("public-client", "https://ragtime.test/mcp/other", None),
            ("public-client", None, "write"),
        ):
            with self.assertRaises(OAuthGrantError):
                await rotate_refresh_token(
                    refresh_hash="refresh-1",
                    next_refresh_hash=f"unused-{uuid.uuid4()}",
                    client_id=client_id,
                    audience=audience,
                    scope=scope,
                )
            self.assertIsNotNone(await get_active_grant(grant.id))

        await rotate_refresh_token(refresh_hash="refresh-1", next_refresh_hash="refresh-2", client_id="public-client")

    async def test_concurrent_refresh_replay_revokes_the_family(self) -> None:
        grant = await self._create()

        async def rotate(next_hash: str) -> str:
            try:
                await rotate_refresh_token(refresh_hash="refresh-1", next_refresh_hash=next_hash, client_id="public-client")
                return "rotated"
            except OAuthGrantError as error:
                return error.reason

        results = await asyncio.gather(rotate("refresh-2"), rotate("refresh-3"))
        self.assertIn("rotated", results)
        self.assertIn("replayed_refresh_token", results)
        self.assertIsNone(await get_active_grant(grant.id))

    async def test_opt_in_duplicate_grace_returns_existing_successor_before_fixed_boundary(self) -> None:
        grant = await self._create()
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)

        await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        duplicate = await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at + timedelta(seconds=9, milliseconds=999),
        )

        self.assertEqual(duplicate.id, grant.id)
        rows = await self.db.query_raw(
            'SELECT "token_hash", "consumed_at" FROM "oauth_refresh_tokens" WHERE "grant_id" = $1 ORDER BY "token_hash"',
            grant.id,
        )
        self.assertEqual([row["token_hash"] for row in rows], ["refresh-1", "refresh-2"])
        self.assertIsNotNone(rows[0]["consumed_at"])
        self.assertIsNone(rows[1]["consumed_at"])
        self.assertIsNotNone(await get_active_grant(grant.id))

    async def test_duplicate_grace_at_exact_ten_seconds_revokes_the_family(self) -> None:
        grant = await self._create()
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)
        await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )

        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="refresh-1",
                next_refresh_hash="refresh-2",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=10),
            )
        self.assertIsNone(await get_active_grant(grant.id))

    async def test_duplicate_grace_is_non_sliding_and_rejects_negative_age(self) -> None:
        grant = await self._create()
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)
        await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at + timedelta(seconds=9),
        )

        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="refresh-1",
                next_refresh_hash="refresh-2",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=10, milliseconds=1),
            )
        self.assertIsNone(await get_active_grant(grant.id))

        negative_age_grant = await self._create(refresh_hash="negative-age-refresh")
        await rotate_refresh_token(
            refresh_hash="negative-age-refresh",
            next_refresh_hash="negative-age-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="negative-age-refresh",
                next_refresh_hash="negative-age-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at - timedelta(milliseconds=1),
            )
        self.assertIsNone(await get_active_grant(negative_age_grant.id))

    async def test_duplicate_grace_rejects_wrong_or_consumed_successor(self) -> None:
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)
        wrong_successor_grant = await self._create(refresh_hash="wrong-successor-refresh")
        await rotate_refresh_token(
            refresh_hash="wrong-successor-refresh",
            next_refresh_hash="expected-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="wrong-successor-refresh",
                next_refresh_hash="different-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=1),
            )
        self.assertIsNone(await get_active_grant(wrong_successor_grant.id))

        consumed_successor_grant = await self._create(refresh_hash="consumed-successor-refresh")
        await rotate_refresh_token(
            refresh_hash="consumed-successor-refresh",
            next_refresh_hash="already-consumed-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        await self.db.execute_raw(
            'UPDATE "oauth_refresh_tokens" SET "consumed_at" = $1::timestamp WHERE "token_hash" = $2',
            (consumed_at + timedelta(milliseconds=1)).replace(tzinfo=None),
            "already-consumed-successor",
        )
        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="consumed-successor-refresh",
                next_refresh_hash="already-consumed-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=1),
            )
        self.assertIsNone(await get_active_grant(consumed_successor_grant.id))

    async def test_duplicate_grace_binding_failures_leave_grant_active(self) -> None:
        grant = await self._create()
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)
        await rotate_refresh_token(
            refresh_hash="refresh-1",
            next_refresh_hash="refresh-2",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )

        for client_id, audience, scope in (
            ("other-client", None, None),
            ("public-client", "https://ragtime.test/mcp/other", None),
            ("public-client", None, "write"),
        ):
            with self.assertRaises(OAuthGrantError):
                await rotate_refresh_token(
                    refresh_hash="refresh-1",
                    next_refresh_hash="refresh-2",
                    client_id=client_id,
                    audience=audience,
                    scope=scope,
                    duplicate_grace_seconds=10,
                    now=consumed_at + timedelta(seconds=1),
                )
            self.assertIsNotNone(await get_active_grant(grant.id))

    async def test_duplicate_grace_rechecks_expiry_revocation_and_security_generation(self) -> None:
        consumed_at = datetime.now(timezone.utc).replace(microsecond=0)
        expired = await self._create(refresh_hash="expired-duplicate-refresh")
        await rotate_refresh_token(
            refresh_hash="expired-duplicate-refresh",
            next_refresh_hash="expired-duplicate-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        await self.db.execute_raw(
            'UPDATE "oauth_grants" SET "expires_at" = $1::timestamp WHERE "id" = $2',
            (consumed_at - timedelta(seconds=1)).replace(tzinfo=None),
            expired.id,
        )
        with self.assertRaisesRegex(OAuthGrantError, "expired_grant"):
            await rotate_refresh_token(
                refresh_hash="expired-duplicate-refresh",
                next_refresh_hash="expired-duplicate-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=1),
            )

        revoked = await self._create(refresh_hash="revoked-duplicate-refresh")
        await rotate_refresh_token(
            refresh_hash="revoked-duplicate-refresh",
            next_refresh_hash="revoked-duplicate-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        await self.db.execute_raw(
            'UPDATE "oauth_grants" SET "revoked_at" = $1::timestamp WHERE "id" = $2',
            (consumed_at + timedelta(milliseconds=1)).replace(tzinfo=None),
            revoked.id,
        )
        with self.assertRaisesRegex(OAuthGrantError, "revoked_grant"):
            await rotate_refresh_token(
                refresh_hash="revoked-duplicate-refresh",
                next_refresh_hash="revoked-duplicate-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=1),
            )

        security_invalid = await self._create(refresh_hash="security-duplicate-refresh")
        await rotate_refresh_token(
            refresh_hash="security-duplicate-refresh",
            next_refresh_hash="security-duplicate-successor",
            client_id="public-client",
            duplicate_grace_seconds=10,
            now=consumed_at,
        )
        await self.db.execute_raw('UPDATE "users" SET "security_generation" = 1 WHERE "id" = $1', self.user_id)
        with self.assertRaisesRegex(OAuthGrantError, "security_generation_mismatch"):
            await rotate_refresh_token(
                refresh_hash="security-duplicate-refresh",
                next_refresh_hash="security-duplicate-successor",
                client_id="public-client",
                duplicate_grace_seconds=10,
                now=consumed_at + timedelta(seconds=1),
            )
        self.assertIsNone(await get_active_grant(security_invalid.id))

    async def test_duplicate_grace_defaults_to_strict_and_rejects_invalid_values(self) -> None:
        self.assertEqual(MAX_DUPLICATE_GRACE_SECONDS, 10)
        strict_grant = await self._create(refresh_hash="strict-duplicate-refresh")
        await rotate_refresh_token(
            refresh_hash="strict-duplicate-refresh",
            next_refresh_hash="strict-duplicate-successor",
            client_id="public-client",
        )
        with self.assertRaisesRegex(OAuthGrantError, "replayed_refresh_token"):
            await rotate_refresh_token(
                refresh_hash="strict-duplicate-refresh",
                next_refresh_hash="strict-duplicate-successor",
                client_id="public-client",
            )
        self.assertIsNone(await get_active_grant(strict_grant.id))

        for invalid_grace in (True, -1, 11, 1.5, "10"):
            with self.subTest(invalid_grace=invalid_grace):
                with self.assertRaises(ValueError):
                    await rotate_refresh_token(
                        refresh_hash="unknown-refresh",
                        next_refresh_hash="unused-successor",
                        client_id="public-client",
                        duplicate_grace_seconds=invalid_grace,  # type: ignore[arg-type]
                    )

    async def test_security_reset_serializes_with_grant_creation_and_cleans_sessions(self) -> None:
        await self.db.execute_raw(
            'INSERT INTO "sessions" ("id", "user_id", "token_hash", "expires_at") VALUES ($1, $2, $3, $4::timestamp)',
            str(uuid.uuid4()),
            self.user_id,
            "session-hash",
            (datetime.now(timezone.utc) + timedelta(days=1)).replace(tzinfo=None),
        )

        results = await asyncio.gather(self._create(refresh_hash="refresh-race"), revoke_user_auth(self.user_id), return_exceptions=True)
        self.assertEqual(await self.db.query_raw('SELECT 1 FROM "sessions" WHERE "user_id" = $1', self.user_id), [])
        active = await self.db.query_raw('SELECT 1 FROM "oauth_grants" WHERE "user_id" = $1 AND "revoked_at" IS NULL', self.user_id)
        self.assertEqual(active, [])
        self.assertTrue(any(isinstance(result, OAuthGrantError) or isinstance(result, int) for result in results))

    async def test_cleanup_removes_only_expired_grants_inside_a_rollback_fixture(self) -> None:
        grant = await self._create()
        rollback = _RollbackCleanupFixture()
        try:
            async with self.db.tx() as tx:
                await tx.execute_raw(
                    'UPDATE "oauth_grants" SET "expires_at" = $1::timestamp WHERE "id" = $2',
                    (datetime.now(timezone.utc) - timedelta(seconds=1)).replace(tzinfo=None),
                    grant.id,
                )
                with mock.patch("ragtime.core.oauth_grants.get_db", mock.AsyncMock(return_value=_TransactionDb(tx))):
                    deleted = await cleanup_expired_grants()
                self.assertGreaterEqual(deleted, 1)
                self.assertEqual(await tx.query_raw('SELECT 1 FROM "oauth_grants" WHERE "id" = $1', grant.id), [])
                raise rollback
        except _RollbackCleanupFixture:
            pass

        self.assertIsNotNone(await get_active_grant(grant.id))


class _RollbackCleanupFixture(Exception):
    pass


class _TransactionDb:
    """Lets cleanup use the test's real transaction, which is rolled back afterwards."""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    @asynccontextmanager
    async def tx(self):
        yield self._tx
