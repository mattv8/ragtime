"""Transactional, hash-only persistence for interactive MCP OAuth grants."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ragtime.core.database import get_db


@dataclass(frozen=True)
class OAuthGrantData:
    id: str
    user_id: str
    client_id: str
    audience: str
    scope: str
    expires_at: datetime
    security_generation: int
    mfa_verified_at: datetime | None
    auth_methods: list[str]
    revoked_at: datetime | None


class OAuthGrantError(Exception):
    """A safe internal category for OAuth protocol-layer error mapping."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    return (value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    """Serialize a UTC instant for PostgreSQL ``TIMESTAMP WITHOUT TIME ZONE`` columns."""
    return _now(value).replace(tzinfo=None).isoformat(sep=" ", timespec="milliseconds")


def _aware(value: datetime | str | None) -> datetime | None:
    """Normalize Prisma raw-query timestamp strings to aware UTC datetimes."""
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return (parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _grant_from_row(row: dict[str, Any]) -> OAuthGrantData:
    methods = row.get("auth_methods") or []
    if isinstance(methods, str):
        methods = json.loads(methods)
    return OAuthGrantData(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        client_id=str(row["client_id"]),
        audience=str(row["audience"]),
        scope=str(row["scope"]),
        expires_at=_aware(row["expires_at"]),  # type: ignore[arg-type]
        security_generation=int(row["security_generation"]),
        mfa_verified_at=_aware(row.get("mfa_verified_at")),  # type: ignore[arg-type]
        auth_methods=[str(method) for method in methods],
        revoked_at=_aware(row.get("revoked_at")),  # type: ignore[arg-type]
    )


_GRANT_COLUMNS = """
    "id", "user_id", "client_id", "audience", "scope", "expires_at",
    "security_generation", "mfa_verified_at", "auth_methods", "revoked_at"
"""


async def lock_user_security_generation(tx: Any, user_id: str) -> int:
    """Lock a user row and return its current security generation.

    Callers sharing auth invalidation serialization must invoke this first, before
    locking grants or refresh rows, inside an existing Prisma transaction.
    """
    rows = await tx.query_raw('SELECT "security_generation" FROM "users" WHERE "id" = $1 FOR UPDATE', user_id)
    if not rows:
        raise OAuthGrantError("user_not_found")
    return int(rows[0]["security_generation"])


async def _find_grant(tx: Any, grant_id: str, *, lock: bool) -> OAuthGrantData | None:
    suffix = " FOR UPDATE" if lock else ""
    rows = await tx.query_raw(f'SELECT {_GRANT_COLUMNS} FROM "oauth_grants" WHERE "id" = $1{suffix}', grant_id)
    return _grant_from_row(rows[0]) if rows else None


def _assert_active(grant: OAuthGrantData, *, current_generation: int, now: datetime) -> None:
    if grant.security_generation != current_generation:
        raise OAuthGrantError("security_generation_mismatch")
    if grant.revoked_at is not None:
        raise OAuthGrantError("revoked_grant")
    if grant.expires_at <= now:
        raise OAuthGrantError("expired_grant")


async def create_grant(
    *,
    user_id: str,
    client_id: str,
    audience: str,
    scope: str,
    expires_at: datetime,
    security_generation: int,
    mfa_verified_at: datetime | None,
    auth_methods: list[str],
    refresh_hash: str,
    now: datetime | None = None,
) -> OAuthGrantData:
    expires_at = _now(expires_at)
    grant_id = str(uuid.uuid4())
    refresh_id = str(uuid.uuid4())
    db = await get_db()
    async with db.tx() as tx:
        current_generation = await lock_user_security_generation(tx, user_id)
        if current_generation != security_generation:
            raise OAuthGrantError("security_generation_mismatch")
        # Capture production time after waiting for the serialization lock.
        issued_at = _now(now)
        if expires_at <= issued_at:
            raise OAuthGrantError("expired_grant")
        rows = await tx.query_raw(
            f"""INSERT INTO "oauth_grants" ({_GRANT_COLUMNS}, "created_at")
                VALUES ($1, $2, $3, $4, $5, $6::timestamp, $7, $8::timestamp, $9::jsonb, NULL, $10::timestamp)
                RETURNING {_GRANT_COLUMNS}""",
            grant_id,
            user_id,
            client_id,
            audience,
            scope,
            _timestamp(expires_at),
            security_generation,
            _timestamp(mfa_verified_at) if mfa_verified_at is not None else None,
            json.dumps(auth_methods),
            _timestamp(issued_at),
        )
        await tx.execute_raw(
            """INSERT INTO "oauth_refresh_tokens" ("id", "grant_id", "token_hash", "created_at")
               VALUES ($1, $2, $3, $4::timestamp)""",
            refresh_id,
            grant_id,
            refresh_hash,
            _timestamp(issued_at),
        )
    return _grant_from_row(rows[0])


async def lookup_refresh_grant(refresh_hash: str) -> OAuthGrantData | None:
    db = await get_db()
    rows = await db.query_raw(
        """SELECT g."id", g."user_id", g."client_id", g."audience", g."scope",
                  g."expires_at", g."security_generation", g."mfa_verified_at",
                  g."auth_methods", g."revoked_at"
            FROM "oauth_grants" g
            JOIN "oauth_refresh_tokens" r ON r."grant_id" = g."id"
            WHERE r."token_hash" = $1""",
        refresh_hash,
    )
    return _grant_from_row(rows[0]) if rows else None


async def rotate_refresh_token(
    *,
    refresh_hash: str,
    next_refresh_hash: str,
    client_id: str,
    audience: str | None = None,
    scope: str | None = None,
    now: datetime | None = None,
) -> OAuthGrantData:
    db = await get_db()
    key_rows = await db.query_raw(
        """SELECT g."id" AS "grant_id", g."user_id" AS "user_id"
           FROM "oauth_grants" g JOIN "oauth_refresh_tokens" r ON r."grant_id" = g."id"
           WHERE r."token_hash" = $1""",
        refresh_hash,
    )
    if not key_rows:
        raise OAuthGrantError("unknown_refresh_token")

    replayed = False
    async with db.tx() as tx:
        current_generation = await lock_user_security_generation(tx, str(key_rows[0]["user_id"]))
        grant = await _find_grant(tx, str(key_rows[0]["grant_id"]), lock=True)
        if grant is None:
            raise OAuthGrantError("unknown_refresh_token")
        # Do not let time spent waiting for locks renew an already-expired grant.
        current_time = _now(now)
        _assert_active(grant, current_generation=current_generation, now=current_time)
        # Binding failures deliberately happen before consuming/revoking a valid family.
        if grant.client_id != client_id:
            raise OAuthGrantError("client_mismatch")
        if audience is not None and grant.audience != audience:
            raise OAuthGrantError("audience_mismatch")
        if scope is not None and grant.scope != scope:
            raise OAuthGrantError("scope_mismatch")
        refresh_rows = await tx.query_raw(
            """SELECT "id", "consumed_at" FROM "oauth_refresh_tokens"
               WHERE "token_hash" = $1 FOR UPDATE""",
            refresh_hash,
        )
        if not refresh_rows:
            raise OAuthGrantError("unknown_refresh_token")
        if refresh_rows[0]["consumed_at"] is not None:
            await tx.execute_raw(
                'UPDATE "oauth_grants" SET "revoked_at" = COALESCE("revoked_at", $1::timestamp) WHERE "id" = $2',
                _timestamp(current_time),
                grant.id,
            )
            replayed = True
        else:
            await tx.execute_raw(
                'UPDATE "oauth_refresh_tokens" SET "consumed_at" = $1::timestamp WHERE "id" = $2',
                _timestamp(current_time),
                str(refresh_rows[0]["id"]),
            )
            await tx.execute_raw(
                """INSERT INTO "oauth_refresh_tokens" ("id", "grant_id", "token_hash", "created_at")
                   VALUES ($1, $2, $3, $4::timestamp)""",
                str(uuid.uuid4()),
                grant.id,
                next_refresh_hash,
                _timestamp(current_time),
            )
    if replayed:
        # Raised after the context exits so the family revocation is committed.
        raise OAuthGrantError("replayed_refresh_token")
    return grant


async def get_active_grant(grant_id: str, *, now: datetime | None = None) -> OAuthGrantData | None:
    db = await get_db()
    key_rows = await db.query_raw('SELECT "user_id" FROM "oauth_grants" WHERE "id" = $1', grant_id)
    if not key_rows:
        return None
    try:
        async with db.tx() as tx:
            current_generation = await lock_user_security_generation(tx, str(key_rows[0]["user_id"]))
            grant = await _find_grant(tx, grant_id, lock=True)
            if grant is None:
                return None
            _assert_active(grant, current_generation=current_generation, now=_now(now))
            return grant
    except OAuthGrantError:
        return None


async def revoke_grant(grant_id: str, *, client_id: str | None = None) -> None:
    db = await get_db()
    keys = await db.query_raw('SELECT "user_id" FROM "oauth_grants" WHERE "id" = $1', grant_id)
    if not keys:
        return
    async with db.tx() as tx:
        await lock_user_security_generation(tx, str(keys[0]["user_id"]))
        grant = await _find_grant(tx, grant_id, lock=True)
        if grant is None or (client_id is not None and grant.client_id != client_id):
            return
        await tx.execute_raw(
            'UPDATE "oauth_grants" SET "revoked_at" = COALESCE("revoked_at", $1::timestamp) WHERE "id" = $2',
            _timestamp(_now()),
            grant_id,
        )


async def revoke_refresh_token(refresh_hash: str, *, client_id: str | None = None) -> None:
    db = await get_db()
    keys = await db.query_raw(
        """SELECT g."id" AS "grant_id", g."user_id" AS "user_id"
           FROM "oauth_grants" g JOIN "oauth_refresh_tokens" r ON r."grant_id" = g."id"
           WHERE r."token_hash" = $1""",
        refresh_hash,
    )
    if not keys:
        return
    async with db.tx() as tx:
        await lock_user_security_generation(tx, str(keys[0]["user_id"]))
        grant = await _find_grant(tx, str(keys[0]["grant_id"]), lock=True)
        if grant is None or (client_id is not None and grant.client_id != client_id):
            return
        # Preserve global lock ordering even though only the owning grant is revoked.
        await tx.query_raw('SELECT "id" FROM "oauth_refresh_tokens" WHERE "token_hash" = $1 FOR UPDATE', refresh_hash)
        await tx.execute_raw(
            'UPDATE "oauth_grants" SET "revoked_at" = COALESCE("revoked_at", $1::timestamp) WHERE "id" = $2',
            _timestamp(_now()),
            grant.id,
        )


async def revoke_user_auth(user_id: str, *, expected_generation: int | None = None) -> int:
    """Atomically invalidate user auth and return the generation written.

    When an authentication snapshot is supplied, it must still be current while
    holding the user row lock.  This prevents a stale continuation from
    invalidating a newer administrator security reset and then inheriting its
    generation.
    """
    db = await get_db()
    async with db.tx() as tx:
        current_generation = await lock_user_security_generation(tx, user_id)
        if expected_generation is not None and int(expected_generation) != current_generation:
            raise OAuthGrantError("security_generation_mismatch")
        rows = await tx.query_raw(
            'UPDATE "users" SET "security_generation" = "security_generation" + 1 WHERE "id" = $1 RETURNING "security_generation"',
            user_id,
        )
        await tx.execute_raw('DELETE FROM "sessions" WHERE "user_id" = $1', user_id)
        await tx.execute_raw(
            'UPDATE "oauth_grants" SET "revoked_at" = COALESCE("revoked_at", $1::timestamp) WHERE "user_id" = $2',
            _timestamp(_now()),
            user_id,
        )
    return int(rows[0]["security_generation"])


async def cleanup_expired_grants(*, now: datetime | None = None) -> int:
    """Remove expired grant families; return grant-family count."""
    db = await get_db()
    current_time = _now(now)
    async with db.tx() as tx:
        deleted = await tx.query_raw('DELETE FROM "oauth_grants" WHERE "expires_at" <= $1::timestamp RETURNING "id"', _timestamp(current_time))
    return len(deleted)
