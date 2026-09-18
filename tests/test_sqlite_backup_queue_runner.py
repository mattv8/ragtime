from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

from ragtime.userspace import sqlite_backup_queue as queue
from ragtime.userspace.sqlite_backup_queue import SqliteBackupQueueService


class _Store:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.finished: list[dict] = []

    async def get_job(self, workspace_id: str, job_id: str):
        return self.job

    async def heartbeat(self, job_id: str, owner_token: str) -> bool:
        return True

    async def is_cancel_requested(self, job_id: str, owner_token: str) -> bool:
        return False

    async def progress(self, job_id: str, owner_token: str, **kwargs: int) -> bool:
        return True

    async def finish(self, job_id: str, owner_token: str, **kwargs):
        self.finished.append(kwargs)
        return True

    async def interrupt(self, job_id: str, owner_token: str, **kwargs):
        raise AssertionError("successful capture must not be interrupted")


class _History:
    async def capture_workspace_databases(self, workspace_id: str, **kwargs):
        self.kwargs = kwargs
        await kwargs["progress_callback"](1, 1)
        self.assert_cancel_check = await kwargs["cancel_check"]()
        return [{"id": "ready-backup", "status": "ready"}]


class SqliteBackupQueueRunnerTests(unittest.IsolatedAsyncioTestCase):
    def _job(self) -> dict:
        return {
            "id": str(uuid4()),
            "workspace_id": "workspace-a",
            "owner_token": str(uuid4()),
            "status": "running",
            "trigger": "manual",
            "database_names": [],
        }

    async def test_claimed_job_captures_then_fences_completion(self) -> None:
        job_id = str(uuid4())
        owner = str(uuid4())
        job = {
            "id": job_id,
            "workspace_id": "workspace-a",
            "owner_token": owner,
            "status": "running",
            "trigger": "manual",
            "database_names": ["app.sqlite3"],
            "snapshot_id": None,
            "snapshot_git_commit_hash": None,
        }
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        history = _History()
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "ragtime.userspace.sqlite_backup_queue.settings.index_data_path", temp
        ), mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history):
            await service._run_claimed(job)

        self.assertEqual([{"status": "completed", "backup_ids": ["ready-backup"], "error_message": None}], store.finished)
        self.assertEqual({"app.sqlite3"}, history.kwargs["database_names"])
        self.assertFalse(history.assert_cancel_check)

    async def test_cancelled_capture_is_terminal_cancelled_after_results(self) -> None:
        job_id = str(uuid4())
        owner = str(uuid4())
        job = {"id": job_id, "workspace_id": "workspace-a", "owner_token": owner, "status": "running", "trigger": "manual", "database_names": []}
        store = _Store(job)
        store.is_cancel_requested = mock.AsyncMock(return_value=True)
        service = SqliteBackupQueueService(store)
        history = _History()
        with tempfile.TemporaryDirectory() as temp, mock.patch("ragtime.userspace.sqlite_backup_queue.settings.index_data_path", temp), mock.patch(
            "ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history
        ):
            await service._run_claimed(job)
        self.assertEqual("cancelled", store.finished[0]["status"])

    async def test_known_failed_outcomes_have_safe_summary(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        history = _History()
        history.capture_workspace_databases = mock.AsyncMock(return_value=[{"id": "failed-backup", "status": "failed"}])
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp), mock.patch(
            "ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history
        ):
            await service._run_claimed(job)
        self.assertEqual("failed", store.finished[0]["status"])
        self.assertEqual("One or more SQLite databases could not be captured", store.finished[0]["error_message"])
        self.assertEqual(["failed-backup"], store.finished[0]["backup_ids"])

    async def test_heartbeat_database_error_interrupts_after_capture_drains(self) -> None:
        job = self._job()
        store = _Store(job)
        heartbeat_started = asyncio.Event()

        async def failed_heartbeat(*_args) -> bool:
            heartbeat_started.set()
            raise RuntimeError("database unavailable")

        store.heartbeat = failed_heartbeat
        service = SqliteBackupQueueService(store)

        async def drained_capture(*_args):
            await heartbeat_started.wait()
            return [{"id": "ready-before-error", "status": "ready"}]

        service._capture = drained_capture  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp), mock.patch.object(
            queue, "_HEARTBEAT_SECONDS", 0
        ):
            await service._run_claimed(job)
        self.assertEqual("interrupted", store.finished[0]["status"])
        self.assertEqual(["ready-before-error"], store.finished[0]["backup_ids"])
        self.assertNotIn("database unavailable", store.finished[0]["error_message"])

    async def test_lock_acquisition_error_cleans_owned_job_and_interrupts(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        service._capture_job_backup_ids = mock.AsyncMock(return_value=[])  # type: ignore[method-assign]
        with mock.patch.object(queue, "_try_job_lock", side_effect=OSError("unsafe lock")):
            await service._run_claimed(job)
        self.assertEqual({}, service._owned_jobs)
        self.assertEqual("interrupted", store.finished[0]["status"])

    async def test_live_lock_owned_elsewhere_leaves_claim_untouched_and_cleans_local_state(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        service._capture = mock.AsyncMock()  # type: ignore[method-assign]
        with mock.patch.object(queue, "_try_job_lock", return_value=None):
            await service._run_claimed(job)
        service._capture.assert_not_awaited()
        self.assertEqual([], store.finished)
        self.assertEqual({}, service._owned_jobs)

    async def test_stop_claim_race_finishes_claimed_job_without_capture(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        service._stopping = True
        service._capture = mock.AsyncMock()  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp):
            await service._run_claimed(job)
        service._capture.assert_not_awaited()
        self.assertEqual("cancelled", store.finished[0]["status"])
