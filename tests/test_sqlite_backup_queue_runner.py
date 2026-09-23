from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
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
    async def runtime_history_active(self) -> bool:
        return False

    async def capture_workspace_databases(self, workspace_id: str, **kwargs):
        self.kwargs = kwargs
        await kwargs["progress_callback"](1, 1)
        self.assert_cancel_check = await kwargs["cancel_check"]()
        return [{"id": "ready-backup", "status": "ready"}]


class _RunnerStore:
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str]] = []
        self.enqueued: list[str] = []

    async def stale_running(self, **kwargs):
        return []

    async def prune_terminal(self, **kwargs):
        return []

    async def claim_next(self, owner_token: str):
        return None

    async def cancel(self, workspace_id: str, job_id: str):
        self.cancelled.append((workspace_id, job_id))
        return {"id": job_id}

    async def enqueue(self, workspace_id: str, **kwargs):
        self.enqueued.append(workspace_id)
        return {"id": "enqueued"}


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
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch("ragtime.userspace.sqlite_backup_queue.settings.index_data_path", temp),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
        ):
            await service._run_claimed(job)

        self.assertEqual([{"status": "completed", "backup_ids": ["ready-backup"], "error_message": None}], store.finished)
        self.assertEqual({"app.sqlite3"}, history.kwargs["database_names"])
        self.assertFalse(history.assert_cancel_check)

    async def test_cancelled_capture_is_terminal_cancelled_after_results(self) -> None:
        job_id = str(uuid4())
        owner = str(uuid4())
        job = {"id": job_id, "workspace_id": "workspace-a", "owner_token": owner, "status": "running", "trigger": "manual", "database_names": []}
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        history = _History()
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch("ragtime.userspace.sqlite_backup_queue.settings.index_data_path", temp),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(store, "is_cancel_requested", new=mock.AsyncMock(return_value=True)),
        ):
            await service._run_claimed(job)
        self.assertEqual("cancelled", store.finished[0]["status"])

    async def test_runtime_shutdown_during_initial_get_detaches_before_capture(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        get_started = asyncio.Event()

        async def delayed_get(_workspace_id: str, _job_id: str) -> dict:
            get_started.set()
            await asyncio.Event().wait()
            return job

        history = _History()
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(queue.settings, "index_data_path", temp),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(store, "get_job", new=delayed_get),
            mock.patch.object(history, "capture_workspace_databases", new=mock.AsyncMock()) as capture,
        ):
            activation = Path(temp) / "_userspace" / "sqlite-history-runtime-activation-v2.json"
            activation.parent.mkdir()
            activation.write_text("{}")
            service._worker_task = asyncio.create_task(service._run_claimed(job))
            await get_started.wait()
            await service.stop()

        self.assertFalse(store.finished)
        capture.assert_not_awaited()

    async def test_configured_but_inactive_runtime_capture_remains_legacy_and_does_not_ack(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        history = _History()
        ack = mock.AsyncMock()
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(queue.settings, "index_data_path", temp),
            mock.patch.object(queue.settings, "userspace_runtime_manager_url", "http://runtime:8090"),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(queue, "runtime_manager_request", ack),
        ):
            await service._run_claimed(job)

        self.assertEqual("completed", store.finished[0]["status"])
        ack.assert_not_awaited()

    async def test_known_failed_outcomes_have_safe_summary(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        history = _History()
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(queue.settings, "index_data_path", temp),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(history, "capture_workspace_databases", new=mock.AsyncMock(return_value=[{"id": "failed-backup", "status": "failed"}])),
        ):
            await service._run_claimed(job)
        self.assertEqual("failed", store.finished[0]["status"])
        self.assertEqual("One or more SQLite databases could not be captured", store.finished[0]["error_message"])
        self.assertEqual(["failed-backup"], store.finished[0]["backup_ids"])

    async def test_heartbeat_database_error_interrupts_after_capture_drains(self) -> None:
        job = self._job()
        store = _Store(job)
        heartbeat_started = asyncio.Event()

        async def failed_heartbeat(job_id: str, owner_token: str) -> bool:
            heartbeat_started.set()
            raise RuntimeError("database unavailable")

        service = SqliteBackupQueueService(store)

        async def drained_capture(_job: dict[str, object], _owner_token: str, _lock_fd: int) -> list[dict[str, str]]:
            await heartbeat_started.wait()
            return [{"id": "ready-before-error", "status": "ready"}]

        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(queue.settings, "index_data_path", temp),
            mock.patch.object(queue, "_HEARTBEAT_SECONDS", 0),
            mock.patch.object(store, "heartbeat", new=failed_heartbeat),
            mock.patch.object(service, "_capture", new=drained_capture),
        ):
            await service._run_claimed(job)
        self.assertEqual("interrupted", store.finished[0]["status"])
        self.assertEqual(["ready-before-error"], store.finished[0]["backup_ids"])
        self.assertNotIn("database unavailable", store.finished[0]["error_message"])

    async def test_lock_acquisition_error_cleans_owned_job_and_interrupts(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        with (
            mock.patch.object(queue, "_try_job_lock", side_effect=OSError("unsafe lock")),
            mock.patch.object(service, "_capture_job_backup_ids", new=mock.AsyncMock(return_value=[])),
        ):
            await service._run_claimed(job)
        self.assertEqual({}, service._owned_jobs)
        self.assertEqual("interrupted", store.finished[0]["status"])

    async def test_live_lock_owned_elsewhere_leaves_claim_untouched_and_cleans_local_state(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        capture = mock.AsyncMock()
        with mock.patch.object(queue, "_try_job_lock", return_value=None), mock.patch.object(service, "_capture", new=capture):
            await service._run_claimed(job)
        capture.assert_not_awaited()
        self.assertEqual([], store.finished)
        self.assertEqual({}, service._owned_jobs)

    async def test_stop_claim_race_finishes_claimed_job_without_capture(self) -> None:
        job = self._job()
        store = _Store(job)
        service = SqliteBackupQueueService(store)
        service._stopping = True
        capture = mock.AsyncMock()
        with (
            tempfile.TemporaryDirectory() as temp,
            mock.patch.object(queue.settings, "index_data_path", temp),
            mock.patch.object(service, "_capture", new=capture),
        ):
            await service._run_claimed(job)
        capture.assert_not_awaited()
        self.assertEqual("cancelled", store.finished[0]["status"])

    async def test_idle_claim_waits_back_off_to_capped_interval(self) -> None:
        service = SqliteBackupQueueService(_RunnerStore())
        timeouts: list[float] = []

        async def timeout_wait(awaitable, *, timeout: float):
            awaitable.close()
            timeouts.append(timeout)
            if len(timeouts) == 5:
                service._stopping = True
            raise TimeoutError

        with (
            mock.patch.object(service._store, "claim_next", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(queue.asyncio, "wait_for", new=timeout_wait),
        ):
            await service._run()

        self.assertEqual([1.0, 2.0, 4.0, 5.0, 5.0], timeouts)

    async def test_cancellation_wake_during_claim_is_not_lost_and_resets_idle_backoff(self) -> None:
        store = _RunnerStore()
        service = SqliteBackupQueueService(store)
        timeouts: list[float] = []
        claim_count = 0

        async def claim_next(_owner_token: str):
            nonlocal claim_count
            claim_count += 1
            if claim_count == 2:
                await service.cancel("workspace-a", "job-a")
            return None

        async def wait_for_wake(awaitable, *, timeout: float):
            awaitable.close()
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise TimeoutError
            if len(timeouts) == 2:
                self.assertTrue(service._wake.is_set())
                return None
            service._stopping = True
            raise TimeoutError

        with (
            mock.patch.object(service._store, "claim_next", new=claim_next),
            mock.patch.object(queue.asyncio, "wait_for", new=wait_for_wake),
        ):
            await service._run()

        self.assertEqual([1.0, 2.0, 1.0], timeouts)
        self.assertEqual([("workspace-a", "job-a")], store.cancelled)

    async def test_enqueue_wake_during_claim_is_not_lost(self) -> None:
        store = _RunnerStore()
        service = SqliteBackupQueueService(store)

        async def enqueue_during_claim(_owner_token: str):
            await service.enqueue("workspace-a", trigger="manual")
            return None

        async def wait_for_wake(awaitable, *, timeout: float):
            awaitable.close()
            self.assertEqual(1.0, timeout)
            self.assertTrue(service._wake.is_set())
            service._stopping = True

        with (
            mock.patch.object(service._store, "claim_next", new=enqueue_during_claim),
            mock.patch.object(queue.asyncio, "wait_for", new=wait_for_wake),
        ):
            await service._run()

        self.assertEqual(["workspace-a"], store.enqueued)

    async def test_successful_claim_resets_idle_backoff(self) -> None:
        service = SqliteBackupQueueService(_RunnerStore())
        job = {"id": "claimed"}
        claims = iter((None, job, None))
        timeouts: list[float] = []

        async def timeout_wait(awaitable, *, timeout: float):
            awaitable.close()
            timeouts.append(timeout)
            if len(timeouts) == 2:
                service._stopping = True
            raise TimeoutError

        with (
            mock.patch.object(service._store, "claim_next", new=mock.AsyncMock(side_effect=claims)),
            mock.patch.object(service, "_run_claimed", new=mock.AsyncMock()),
            mock.patch.object(queue.asyncio, "wait_for", new=timeout_wait),
        ):
            await service._run()

        self.assertEqual([1.0, 1.0], timeouts)

    async def test_error_iteration_resets_idle_backoff_and_uses_legacy_poll_delay(self) -> None:
        service = SqliteBackupQueueService(_RunnerStore())
        claims = iter((None, RuntimeError("store unavailable"), None))
        timeouts: list[float] = []
        sleeps: list[float] = []

        async def timeout_wait(awaitable, *, timeout: float):
            awaitable.close()
            timeouts.append(timeout)
            if len(timeouts) == 2:
                service._stopping = True
            raise TimeoutError

        async def record_sleep(delay: float):
            sleeps.append(delay)

        with (
            mock.patch.object(service._store, "claim_next", new=mock.AsyncMock(side_effect=claims)),
            mock.patch.object(queue.asyncio, "wait_for", new=timeout_wait),
            mock.patch.object(queue.asyncio, "sleep", new=record_sleep),
        ):
            await service._run()

        self.assertEqual([1.0, 1.0], timeouts)
        self.assertEqual([queue._POLL_SECONDS], sleeps)

    async def test_stop_wake_during_claim_exits_without_another_wait(self) -> None:
        service = SqliteBackupQueueService(_RunnerStore())

        async def stop_during_claim(_owner_token: str):
            service._stopping = True
            service._wake.set()
            return None

        wait_for = mock.AsyncMock()
        with (
            mock.patch.object(service._store, "claim_next", new=stop_during_claim),
            mock.patch.object(queue.asyncio, "wait_for", new=wait_for),
        ):
            await service._run()

        wait_for.assert_not_awaited()
