"""Durable PostgreSQL store for queued User Space SQLite backups."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import HTTPException

from ragtime.core.database import get_db
from ragtime.userspace.sqlite_history import _validate_database_name

_ACTIVE = ("pending", "running")
_TERMINAL = ("completed", "failed", "cancelled", "interrupted")
_TRIGGERS = {"manual", "snapshot", "scheduled"}
_STATUSES = set(_ACTIVE) | set(_TERMINAL)
_QUEUE_LOCK = "userspace-sqlite-backup-queue"
_COLUMNS = """id, workspace_id, requested_by_id, trigger, database_names, snapshot_id,
snapshot_git_commit_hash, request_key, request_hash, status, created_at, available_at,
started_at, finished_at, updated_at, heartbeat_at, owner_token, cancel_requested,
completed_databases, total_databases, backup_ids, error_message"""


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {key: row.get(key) for key in _COLUMNS.replace("\n", " ").replace(",", " ").split()}
    for key in ("created_at", "available_at", "started_at", "finished_at", "updated_at", "heartbeat_at"):
        value = payload[key]
        if isinstance(value, str):
            payload[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return payload


def _names(database_names: list[str] | set[str] | tuple[str, ...] | None) -> list[str]:
    if database_names is None:
        return []
    names = {_validate_database_name(name) for name in database_names}
    if not names:
        raise HTTPException(status_code=400, detail="At least one database name is required")
    return sorted(names)


def _qualified_columns(alias: str) -> str:
    return ", ".join(f"{alias}.{column.strip()}" for column in _COLUMNS.replace("\n", " ").split(","))


class SqliteBackupQueueStore:
    """Queue metadata operations serialized by a transaction-scoped advisory lock."""

    async def enqueue(
        self,
        workspace_id: str,
        *,
        trigger: str,
        database_names: list[str] | set[str] | tuple[str, ...] | None = None,
        snapshot_id: str | None = None,
        snapshot_git_commit_hash: str | None = None,
        requested_by_id: str | None = None,
        request_key: str | None = None,
    ) -> dict[str, Any]:
        if trigger not in _TRIGGERS:
            raise ValueError("trigger must be manual, snapshot, or scheduled")
        names = _names(database_names)
        request_key = request_key or str(uuid.uuid4())
        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "trigger": trigger,
                    "database_names": names,
                    "snapshot_id": snapshot_id,
                    "snapshot_git_commit_hash": snapshot_git_commit_hash,
                    "requested_by_id": requested_by_id,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        db = await get_db()
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", _QUEUE_LOCK)
            existing = await tx.query_raw(
                f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND request_key = $2", workspace_id, request_key
            )
            if existing:
                row = _payload(existing[0])
                if row["request_hash"] != request_hash:
                    raise HTTPException(status_code=409, detail="Idempotency key conflicts with a different backup request")
                return row
            if trigger == "scheduled":
                active = await tx.query_raw(
                    f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND trigger = 'scheduled' AND status IN ('pending', 'running') ORDER BY created_at, id LIMIT 1",
                    workspace_id,
                )
                if active:
                    return _payload(active[0])
            counts = await tx.query_raw(
                "SELECT count(*) FILTER (WHERE status IN ('pending', 'running'))::int AS global_count, count(*) FILTER (WHERE workspace_id = $1 AND status IN ('pending', 'running'))::int AS workspace_count FROM workspace_sqlite_backup_jobs",
                workspace_id,
            )
            if int(counts[0]["global_count"]) >= 256 or int(counts[0]["workspace_count"]) >= 32:
                raise HTTPException(status_code=503, detail="SQLite backup queue is full")
            rows = await tx.query_raw(
                f"""INSERT INTO workspace_sqlite_backup_jobs
                (workspace_id, requested_by_id, trigger, database_names, snapshot_id, snapshot_git_commit_hash, request_key, request_hash)
                VALUES ($1, $2, $3, $4::text[], $5, $6, $7, $8) RETURNING {_COLUMNS}""",
                workspace_id,
                requested_by_id,
                trigger,
                names,
                snapshot_id,
                snapshot_git_commit_hash,
                request_key,
                request_hash,
            )
            return _payload(rows[0])

    async def list_jobs(self, workspace_id: str, *, database_name: str | None = None, snapshot_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 50))
        db = await get_db()
        rows = await db.query_raw(
            f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND ($2::text IS NULL OR cardinality(database_names) = 0 OR $2 = ANY(database_names)) AND ($3::text IS NULL OR snapshot_id = $3) ORDER BY CASE WHEN status IN ('pending', 'running') THEN 0 ELSE 1 END, created_at DESC, id DESC LIMIT $4",
            workspace_id,
            database_name,
            snapshot_id,
            limit,
        )
        return [_payload(row) for row in rows]

    async def get_job(self, workspace_id: str, job_id: str) -> dict[str, Any] | None:
        db = await get_db()
        rows = await db.query_raw(f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND id = $2", workspace_id, job_id)
        return _payload(rows[0]) if rows else None

    async def claim_next(self, owner_token: str) -> dict[str, Any] | None:
        db = await get_db()
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", _QUEUE_LOCK)
            claimed = await tx.query_raw(
                f"""WITH next_job AS (
                SELECT candidate.id FROM workspace_sqlite_backup_jobs candidate
                WHERE candidate.status = 'pending' AND candidate.available_at <= NOW()
                  AND NOT EXISTS (SELECT 1 FROM workspace_sqlite_backup_jobs running WHERE running.workspace_id = candidate.workspace_id AND running.status = 'running')
                  AND NOT EXISTS (SELECT 1 FROM workspace_sqlite_backup_jobs running WHERE running.status = 'running')
                ORDER BY candidate.created_at, candidate.id FOR UPDATE SKIP LOCKED LIMIT 1
            ) UPDATE workspace_sqlite_backup_jobs job SET status = 'running', owner_token = $1,
                started_at = NOW(), heartbeat_at = NOW(), updated_at = NOW(), cancel_requested = FALSE
            FROM next_job WHERE job.id = next_job.id RETURNING {_qualified_columns("job")}""",
                owner_token,
            )
            return _payload(claimed[0]) if claimed else None

    async def heartbeat(self, job_id: str, owner_token: str) -> bool:
        return await self._owned_update(job_id, owner_token, "heartbeat_at = NOW(), updated_at = NOW()")

    async def progress(self, job_id: str, owner_token: str, *, completed_databases: int, total_databases: int) -> bool:
        db = await get_db()
        updated = await db.execute_raw(
            "UPDATE workspace_sqlite_backup_jobs SET completed_databases = $1, total_databases = $2, heartbeat_at = NOW(), updated_at = NOW() WHERE id = $3 AND status = 'running' AND owner_token = $4",
            completed_databases,
            total_databases,
            job_id,
            owner_token,
        )
        return updated == 1

    async def is_cancel_requested(self, job_id: str, owner_token: str) -> bool:
        db = await get_db()
        rows = await db.query_raw(
            "SELECT cancel_requested FROM workspace_sqlite_backup_jobs WHERE id = $1 AND status = 'running' AND owner_token = $2", job_id, owner_token
        )
        return not rows or bool(rows[0]["cancel_requested"])

    async def finish(self, job_id: str, owner_token: str, *, status: str, backup_ids: list[str], error_message: str | None = None) -> bool:
        if status not in _TERMINAL:
            raise ValueError("finish status must be terminal")
        db = await get_db()
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", _QUEUE_LOCK)
            rows = await tx.query_raw(
                f"UPDATE workspace_sqlite_backup_jobs SET status = CASE WHEN cancel_requested THEN 'cancelled' ELSE $1 END, backup_ids = $2::text[], error_message = $3, finished_at = NOW(), updated_at = NOW(), heartbeat_at = NOW() WHERE id = $4 AND status = 'running' AND owner_token = $5 RETURNING {_COLUMNS}",
                status,
                backup_ids,
                error_message,
                job_id,
                owner_token,
            )
            if not rows:
                return False
            if (
                str(rows[0].get("status")) == "failed"
                and str(rows[0].get("trigger")) == "scheduled"
                and not str(rows[0].get("request_key") or "").startswith("scheduled-retry:")
                and not bool(rows[0].get("cancel_requested"))
            ):
                retry_key = f"scheduled-retry:{job_id}"
                await tx.query_raw(
                    """INSERT INTO workspace_sqlite_backup_jobs
                    (workspace_id, requested_by_id, trigger, database_names, snapshot_id, snapshot_git_commit_hash, request_key, request_hash, available_at)
                    SELECT $1, $2, 'scheduled', $3::text[], $4, $5, $6, $7, NOW() + INTERVAL '5 minutes'
                    WHERE NOT EXISTS (SELECT 1 FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND trigger = 'scheduled' AND status IN ('pending', 'running'))
                    ON CONFLICT (workspace_id, request_key) DO NOTHING""",
                    rows[0]["workspace_id"],
                    rows[0].get("requested_by_id"),
                    rows[0].get("database_names") or [],
                    rows[0].get("snapshot_id"),
                    rows[0].get("snapshot_git_commit_hash"),
                    retry_key,
                    rows[0]["request_hash"],
                )
            return True

    async def cancel(self, workspace_id: str, job_id: str) -> dict[str, Any] | None:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw(
                f"UPDATE workspace_sqlite_backup_jobs SET status = CASE WHEN status = 'pending' THEN 'cancelled' ELSE status END, cancel_requested = CASE WHEN status = 'running' THEN TRUE ELSE FALSE END, finished_at = CASE WHEN status = 'pending' THEN NOW() ELSE finished_at END, updated_at = NOW() WHERE workspace_id = $1 AND id = $2 AND status IN ('pending', 'running') RETURNING {_COLUMNS}",
                workspace_id,
                job_id,
            )
            if rows:
                return _payload(rows[0])
            existing = await tx.query_raw(f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE workspace_id = $1 AND id = $2", workspace_id, job_id)
            return _payload(existing[0]) if existing else None

    async def stale_running(self, *, older_than_seconds: int = 120, limit: int = 50) -> list[dict[str, Any]]:
        db = await get_db()
        rows = await db.query_raw(
            f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE status = 'running' AND COALESCE(heartbeat_at, started_at, created_at) < NOW() - ($1 * INTERVAL '1 second') ORDER BY COALESCE(heartbeat_at, started_at, created_at) LIMIT $2",
            older_than_seconds,
            max(1, min(limit, 50)),
        )
        return [_payload(row) for row in rows]

    async def reconcilable(self, *, older_than_seconds: int = 120, limit: int = 50) -> list[dict[str, Any]]:
        """Include interrupted projections: runtime receipts may finish later."""
        db = await get_db()
        rows = await db.query_raw(
            f"SELECT {_COLUMNS} FROM workspace_sqlite_backup_jobs WHERE status IN ('running', 'interrupted') AND COALESCE(heartbeat_at, finished_at, started_at, created_at) < NOW() - ($1 * INTERVAL '1 second') ORDER BY COALESCE(heartbeat_at, finished_at, started_at, created_at) LIMIT $2",
            older_than_seconds,
            max(1, min(limit, 50)),
        )
        return [_payload(row) for row in rows]

    async def project_runtime_terminal(self, job_id: str, *, status: str, backup_ids: list[str], error_message: str | None) -> bool:
        if status not in _TERMINAL:
            raise ValueError("runtime projection must be terminal")
        db = await get_db()
        updated = await db.execute_raw(
            "UPDATE workspace_sqlite_backup_jobs SET status = $1, backup_ids = $2::text[], error_message = $3, finished_at = NOW(), updated_at = NOW(), heartbeat_at = NOW() WHERE id = $4 AND status IN ('running', 'interrupted')",
            status,
            backup_ids,
            error_message,
            job_id,
        )
        return updated == 1

    async def interrupt(self, job_id: str, owner_token: str, *, error_message: str) -> bool:
        db = await get_db()
        updated = await db.execute_raw(
            "UPDATE workspace_sqlite_backup_jobs SET status = 'interrupted', error_message = $1, finished_at = NOW(), updated_at = NOW() WHERE id = $2 AND status = 'running' AND owner_token = $3",
            error_message,
            job_id,
            owner_token,
        )
        return updated == 1

    async def takeover_observer(self, job_id: str, previous_owner_token: str, observer_token: str) -> dict[str, Any] | None:
        """CAS transfer an expired projection lease without changing runtime work.

        This is intentionally only queue-observer ownership.  The stable job ID
        remains the runtime operation ID, so a controller restart cannot create
        a second capture.
        """
        db = await get_db()
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", _QUEUE_LOCK)
            rows = await tx.query_raw(
                f"UPDATE workspace_sqlite_backup_jobs SET owner_token = $1, heartbeat_at = NOW(), updated_at = NOW() WHERE id = $2 AND status = 'running' AND owner_token = $3 RETURNING {_COLUMNS}",
                observer_token,
                job_id,
                previous_owner_token,
            )
            return _payload(rows[0]) if rows else None

    async def prune_terminal(self, *, older_than_days: int = 30, limit: int = 100) -> list[str]:
        db = await get_db()
        rows = await db.query_raw(
            "DELETE FROM workspace_sqlite_backup_jobs WHERE id IN (SELECT id FROM workspace_sqlite_backup_jobs WHERE status IN ('completed', 'failed', 'cancelled', 'interrupted') AND finished_at < NOW() - ($1 * INTERVAL '1 day') ORDER BY finished_at LIMIT $2) RETURNING id",
            older_than_days,
            max(1, min(limit, 100)),
        )
        return [str(row["id"]) for row in rows]

    async def _owned_update(self, job_id: str, owner_token: str, assignments: str) -> bool:
        db = await get_db()
        updated = await db.execute_raw(
            f"UPDATE workspace_sqlite_backup_jobs SET {assignments} WHERE id = $1 AND status = 'running' AND owner_token = $2", job_id, owner_token
        )
        return updated == 1
