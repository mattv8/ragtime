"""Opt-in PostgreSQL integration coverage for the durable SQLite backup queue."""

import asyncio
import contextvars
import os
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException
from prisma import Prisma
from prisma.enums import AuthProvider

from ragtime.core.datetimes import utc_now
from ragtime.userspace.sqlite_backup_queue import SqliteBackupQueueService
from ragtime.userspace.sqlite_backup_queue_store import SqliteBackupQueueStore
from ragtime.userspace.sqlite_history import SqliteHistoryService

_task_db: contextvars.ContextVar[Prisma] = contextvars.ContextVar("sqlite_backup_queue_test_db")


async def _get_task_db() -> Prisma:
    return _task_db.get()


@unittest.skipUnless(os.environ.get("RAGTIME_SQLITE_QUEUE_DATABASE_INTEGRATION") == "1", "disposable queue Postgres opt-in")
class SqliteBackupQueueDatabaseTests(unittest.IsolatedAsyncioTestCase):
    """Use independent clients so database serialization is the system under test."""

    async def asyncSetUp(self) -> None:
        self.first, self.second = Prisma(), Prisma()
        await self.first.connect()
        await self.second.connect()
        self._get_db_patch = mock.patch("ragtime.userspace.sqlite_backup_queue_store.get_db", new=_get_task_db)
        self._get_db_patch.start()
        self.user_id = str(uuid4())
        self.workspace_ids: list[str] = []
        self._sqlite_connections: list[sqlite3.Connection] = []
        await self.first.user.create(data={"id": self.user_id, "username": f"sqlite-queue-db-{self.user_id}", "authProvider": AuthProvider.local})

    async def asyncTearDown(self) -> None:
        for workspace_id in self.workspace_ids:
            await self.first.workspace.delete_many(where={"id": workspace_id})
        await self.first.user.delete_many(where={"id": self.user_id})
        for connection in self._sqlite_connections:
            connection.close()
        self._get_db_patch.stop()
        await self.first.disconnect()
        await self.second.disconnect()

    async def _workspace(self) -> str:
        workspace_id = str(uuid4())
        self.workspace_ids.append(workspace_id)
        await self.first.workspace.create(data={"id": workspace_id, "name": f"sqlite-queue-db-{workspace_id}", "ownerUserId": self.user_id})
        return workspace_id

    async def _call(self, db: Prisma, method: str, *args: object, **kwargs: object) -> object:
        token = _task_db.set(db)
        try:
            return await getattr(SqliteBackupQueueStore(), method)(*args, **kwargs)
        finally:
            _task_db.reset(token)

    async def _enqueue(self, db: Prisma, workspace_id: str, key: str, **kwargs: object) -> dict:
        result = await self._call(db, "enqueue", workspace_id, trigger="manual", database_names=["app.sqlite3"], requested_by_id=self.user_id, request_key=key, **kwargs)
        assert isinstance(result, dict)
        return result

    async def _finish(self, db: Prisma, job: dict, owner_token: str, status: str = "completed") -> None:
        result = await self._call(db, "finish", job["id"], owner_token, status=status, backup_ids=[], error_message="capture failed" if status == "failed" else None)
        self.assertTrue(result)

    async def test_atomic_claim_enforces_global_cap_and_workspace_exclusion(self) -> None:
        first_workspace, second_workspace = await self._workspace(), await self._workspace()
        first_job = await self._enqueue(self.first, first_workspace, "first")
        second_job = await self._enqueue(self.second, second_workspace, "second")
        await self.first.execute_raw("UPDATE workspace_sqlite_backup_jobs SET created_at = NOW() - ($1 * INTERVAL '1 second') WHERE id = $2", 60, first_job["id"])
        claimed = await self._call(self.first, "claim_next", "owner-first")
        self.assertEqual(claimed["id"], first_job["id"])
        self.assertIsNone(await self._call(self.second, "claim_next", "owner-second"))
        await self._finish(self.first, claimed, "owner-first")
        next_claim = await self._call(self.second, "claim_next", "owner-second")
        self.assertEqual(next_claim["id"], second_job["id"])
        await self._enqueue(self.first, second_workspace, "same-workspace-pending")
        self.assertIsNone(await self._call(self.first, "claim_next", "owner-third"))
        await self._finish(self.second, next_claim, "owner-second")
        self.assertEqual((await self._call(self.first, "claim_next", "owner-third"))["workspace_id"], second_workspace)

    async def test_concurrent_idempotency_hash_conflict_and_fifo_eligibility(self) -> None:
        first_workspace, second_workspace = await self._workspace(), await self._workspace()
        same_key = await asyncio.gather(self._enqueue(self.first, first_workspace, "idempotent-key"), self._enqueue(self.second, first_workspace, "idempotent-key"))
        self.assertEqual(same_key[0]["id"], same_key[1]["id"])
        with self.assertRaises(HTTPException) as conflict:
            await self._call(self.second, "enqueue", first_workspace, trigger="manual", database_names=["other.sqlite3"], requested_by_id=self.user_id, request_key="idempotent-key")
        self.assertEqual(conflict.exception.status_code, 409)
        await self.first.execute_raw("UPDATE workspace_sqlite_backup_jobs SET created_at = NOW() - ($1 * INTERVAL '1 second') WHERE id = $2", 60, same_key[0]["id"])
        future_job = await self._enqueue(self.first, second_workspace, "future")
        ready_job = await self._enqueue(self.second, await self._workspace(), "ready")
        await self.first.execute_raw("UPDATE workspace_sqlite_backup_jobs SET available_at = NOW() + ($1 * INTERVAL '1 second') WHERE id = $2", 600, future_job["id"])
        claimed = await self._call(self.first, "claim_next", "fifo-owner")
        self.assertEqual(claimed["id"], same_key[0]["id"])
        await self._finish(self.first, claimed, "fifo-owner")
        self.assertEqual((await self._call(self.second, "claim_next", "fifo-owner-two"))["id"], ready_job["id"])

    async def test_scheduled_coalescing_recurrence_and_failed_retry_delay(self) -> None:
        workspace_id = await self._workspace()
        first = await self._call(self.first, "enqueue", workspace_id, trigger="scheduled", request_key=f"scheduled:{workspace_id}:claim-one")
        coalesced = await self._call(self.second, "enqueue", workspace_id, trigger="scheduled", request_key=f"scheduled:{workspace_id}:claim-two")
        self.assertEqual(first["id"], coalesced["id"])
        claimed = await self._call(self.first, "claim_next", "scheduled-owner")
        await self._finish(self.first, claimed, "scheduled-owner", status="failed")
        jobs = await self._call(self.second, "list_jobs", workspace_id, limit=10)
        retry = next(job for job in jobs if job["request_key"] == f"scheduled-retry:{first['id']}")
        self.assertEqual(retry["status"], "pending")
        self.assertGreaterEqual(retry["available_at"], utc_now() + timedelta(minutes=4, seconds=50))
        self.assertIsNone(await self._call(self.second, "claim_next", "too-early"))
        self.assertEqual((await self._call(self.second, "cancel", workspace_id, retry["id"]))["status"], "cancelled")
        recurrence = await self._call(self.first, "enqueue", workspace_id, trigger="scheduled", request_key=f"scheduled:{workspace_id}:claim-three")
        self.assertNotEqual(recurrence["id"], first["id"])

    async def test_cancel_claim_race_owner_fencing_and_workspace_scoped_history(self) -> None:
        workspace_id, other_workspace = await self._workspace(), await self._workspace()
        job, other = await self._enqueue(self.first, workspace_id, "race"), await self._enqueue(self.first, other_workspace, "other")
        await self.first.execute_raw("UPDATE workspace_sqlite_backup_jobs SET created_at = NOW() - ($1 * INTERVAL '1 second') WHERE id = $2", 60, job["id"])
        claimed, cancelled = await asyncio.gather(self._call(self.first, "claim_next", "actual-owner"), self._call(self.second, "cancel", workspace_id, job["id"]))
        stored = await self._call(self.first, "get_job", workspace_id, job["id"])
        self.assertIn(stored["status"], {"cancelled", "running"})
        if claimed is not None:
            self.assertFalse(await self._call(self.second, "heartbeat", claimed["id"], "wrong-owner"))
            self.assertFalse(await self._call(self.second, "progress", claimed["id"], "wrong-owner", completed_databases=1, total_databases=1))
            self.assertFalse(await self._call(self.second, "finish", claimed["id"], "wrong-owner", status="completed", backup_ids=[]))
            await self._finish(self.first, claimed, "actual-owner", status="cancelled" if claimed["id"] == job["id"] else "completed")
        self.assertIsNotNone(cancelled)
        target = await self._call(self.first, "get_job", workspace_id, job["id"])
        self.assertEqual(target["status"], "cancelled")
        other_job = await self._call(self.first, "get_job", other_workspace, other["id"])
        if other_job["status"] == "pending":
            other_claim = await self._call(self.second, "claim_next", "other-owner")
            self.assertEqual(other_claim["id"], other["id"])
            await self._finish(self.second, other_claim, "other-owner")
        other_history = await self._call(self.second, "list_jobs", other_workspace, limit=10)
        self.assertEqual([row["id"] for row in other_history], [other["id"]])
        self.assertEqual(other_history[0]["status"], "completed")
        self.assertIsNone(await self._call(self.second, "get_job", other_workspace, job["id"]))

    async def test_nonterminal_cap_limits_and_migration_indexes(self) -> None:
        workspace_id = await self._workspace()
        for number in range(32):
            await self._enqueue(self.first, workspace_id, f"workspace-cap-{number}")
        with self.assertRaises(HTTPException) as workspace_cap:
            await self._enqueue(self.first, workspace_id, "workspace-cap-overflow")
        self.assertEqual(workspace_cap.exception.status_code, 503)
        for number in range(7):
            for offset in range(32):
                await self._enqueue(self.second, await self._workspace() if offset == 0 else self.workspace_ids[-1], f"global-cap-{number}-{offset}")
        with self.assertRaises(HTTPException) as global_cap:
            await self._enqueue(self.second, await self._workspace(), "global-cap-overflow")
        self.assertEqual(global_cap.exception.status_code, 503)
        indexes = await self.first.query_raw("SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND tablename = 'workspace_sqlite_backup_jobs'")
        names = {str(row["indexname"]) for row in indexes}
        self.assertIn("workspace_sqlite_backup_jobs_status_created_at_idx", names)
        self.assertIn("workspace_sqlite_backup_jobs_workspace_id_status_idx", names)
        self.assertIn("workspace_sqlite_backup_jobs_workspace_id_created_at_idx", names)
        constraints = await self.first.query_raw("SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint WHERE conrelid = 'workspace_sqlite_backup_jobs'::regclass")
        self.assertTrue(any("FOREIGN KEY" in str(row["definition"]) and "ON DELETE CASCADE" in str(row["definition"]) for row in constraints))

    async def test_two_real_runners_capture_wal_databases_serially(self) -> None:
        """Exercise the runner, durable store, Landlock capture, and catalog together."""
        workspace_ids = [await self._workspace() for _ in range(3)]
        with tempfile.TemporaryDirectory() as temporary_root:
            root = Path(temporary_root)
            for workspace_id in workspace_ids:
                database_dir = root / "_userspace" / "workspaces" / workspace_id / "files" / ".ragtime" / "db"
                database_dir.mkdir(parents=True)
                connection = sqlite3.connect(database_dir / "app.sqlite3")
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, value TEXT)")
                connection.execute("INSERT INTO events (value) VALUES ('queued')")
                connection.commit()
                self._sqlite_connections.append(connection)

            history = SqliteHistoryService(lambda workspace_id: root / "_userspace" / "workspaces" / workspace_id / "files")
            first_runner, second_runner = SqliteBackupQueueService(), SqliteBackupQueueService()
            active_captures = peak_captures = 0

            async def capture_with_peak(original: object, *args: object) -> list[dict]:
                nonlocal active_captures, peak_captures
                active_captures += 1
                peak_captures = max(peak_captures, active_captures)
                try:
                    return await original(*args)
                finally:
                    active_captures -= 1

            first_capture, second_capture = first_runner._capture, second_runner._capture
            async def first_wrapped(*args: object) -> list[dict]:
                return await capture_with_peak(first_capture, *args)
            async def second_wrapped(*args: object) -> list[dict]:
                return await capture_with_peak(second_capture, *args)

            with (
                mock.patch("ragtime.userspace.sqlite_runtime.runtime_manager_enabled", return_value=False),
                mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
                mock.patch("ragtime.userspace.sqlite_backup_queue.settings.index_data_path", str(root)),
                mock.patch.object(first_runner, "_capture", side_effect=first_wrapped),
                mock.patch.object(second_runner, "_capture", side_effect=second_wrapped),
            ):
                first_token = _task_db.set(self.first)
                try:
                    await first_runner.start()
                    first_snapshot = await first_runner.enqueue(workspace_ids[0], trigger="snapshot", snapshot_id="snapshot-one", snapshot_git_commit_hash="commit-one", requested_by_id=self.user_id, request_key="snapshot-one")
                    second_snapshot = await first_runner.enqueue(workspace_ids[0], trigger="snapshot", snapshot_id="snapshot-two", snapshot_git_commit_hash="commit-two", request_key="snapshot-two")
                    third_manual = await first_runner.enqueue(workspace_ids[1], trigger="manual", database_names=["app.sqlite3"], requested_by_id=self.user_id, request_key="manual-two")
                finally:
                    _task_db.reset(first_token)
                second_token = _task_db.set(self.second)
                try:
                    await second_runner.start()
                    fourth_manual = await second_runner.enqueue(workspace_ids[2], trigger="manual", database_names=["app.sqlite3"], requested_by_id=self.user_id, request_key="manual-three")
                finally:
                    _task_db.reset(second_token)

                async def wait(db: Prisma, runner: SqliteBackupQueueService, workspace_id: str, job_id: str) -> dict:
                    token = _task_db.set(db)
                    try:
                        return await asyncio.wait_for(runner.wait_for_job(workspace_id, job_id), timeout=90)
                    finally:
                        _task_db.reset(token)

                try:
                    completed = await asyncio.gather(
                        wait(self.first, first_runner, workspace_ids[0], first_snapshot["id"]),
                        wait(self.first, first_runner, workspace_ids[0], second_snapshot["id"]),
                        wait(self.first, first_runner, workspace_ids[1], third_manual["id"]),
                        wait(self.second, second_runner, workspace_ids[2], fourth_manual["id"]),
                    )
                finally:
                    first_token, second_token = _task_db.set(self.first), _task_db.set(self.second)
                    try:
                        await asyncio.gather(first_runner.stop(), second_runner.stop())
                    finally:
                        _task_db.reset(second_token)
                        _task_db.reset(first_token)

            self.assertEqual({"completed"}, {job["status"] for job in completed})
            self.assertEqual(1, peak_captures)
            self.assertTrue(all(job["backup_ids"] for job in completed))
            first_rows = await history.list_backups(workspace_ids[0])
            snapshot_rows = {row["snapshot_id"]: row for row in first_rows if row.get("snapshot_id") in {"snapshot-one", "snapshot-two"}}
            self.assertEqual({"snapshot-one", "snapshot-two"}, set(snapshot_rows))
            self.assertEqual(snapshot_rows["snapshot-one"]["blob"], snapshot_rows["snapshot-two"]["blob"])
            self.assertEqual(first_snapshot["id"], snapshot_rows["snapshot-one"]["capture_job_id"])
            self.assertEqual(second_snapshot["id"], snapshot_rows["snapshot-two"]["capture_job_id"])
