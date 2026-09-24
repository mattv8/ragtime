from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException

from ragtime.userspace import sqlite_backup_queue as queue
from ragtime.userspace.sqlite_history import SqliteHistoryService


class _RecoveryStore:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.interrupted: list[str] = []

    async def stale_running(self, **kwargs):
        return [{"id": self.job_id, "owner_token": "dead-owner"}]

    async def interrupt(self, job_id: str, owner_token: str, **kwargs):
        self.interrupted.append(job_id)
        return True


class _RuntimeRecoveryStore:
    def __init__(self, job: dict[str, object], *, projected: bool) -> None:
        self.job = job
        self.projected = projected
        self.projections: list[dict[str, object]] = []

    async def reconcilable(self, **kwargs):
        return [self.job]

    async def project_runtime_terminal(self, job_id: str, **kwargs):
        self.projections.append({"job_id": job_id, **kwargs})
        return self.projected

    async def takeover_observer(self, *args, **kwargs):
        raise AssertionError("terminal receipt must not become an observer")


class SqliteBackupQueueLivenessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        runtime_active_patch = mock.patch.object(
            SqliteHistoryService,
            "runtime_history_active",
            new=mock.AsyncMock(return_value=False),
        )
        runtime_active_patch.start()
        self.addCleanup(runtime_active_patch.stop)

    async def test_empty_recovery_does_not_consult_runtime_activation(self) -> None:
        store = SimpleNamespace(reconcilable=mock.AsyncMock(return_value=[]))
        history = SimpleNamespace(runtime_history_active=mock.AsyncMock(side_effect=RuntimeError("runtime unavailable")))
        service = queue.SqliteBackupQueueService(store)
        with mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history):
            self.assertEqual([], await service.recover_stale())

        history.runtime_history_active.assert_not_awaited()

    async def test_recovery_does_not_interrupt_when_another_process_holds_job_lock(self) -> None:
        job_id = str(uuid4())
        store = _RecoveryStore(job_id)
        service = queue.SqliteBackupQueueService(store)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp):
            fd = queue._try_job_lock(job_id)
            assert fd is not None
            try:
                self.assertEqual([], await service.recover_stale())
            finally:
                queue._release_job_lock(fd)
            self.assertEqual([job_id], await service.recover_stale())
        self.assertEqual([job_id], store.interrupted)

    async def test_child_inherits_context_job_fd(self) -> None:
        job_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp):
            fd = queue._try_job_lock(job_id)
            assert fd is not None
            try:
                # A real subprocess observes the inherited descriptor; this is
                # the fence that survives a controller process crash.
                from ragtime.userspace.sqlite_capture_admission import inherit_capture_fds, run_admitted_subprocess

                with inherit_capture_fds((fd,)):
                    result = run_admitted_subprocess([sys.executable, "-c", "import os,sys; os.fstat(int(sys.argv[1]))", str(fd)], check=True)
                self.assertEqual(0, result.returncode)
            finally:
                queue._release_job_lock(fd)

    async def test_runtime_recovery_replays_missing_running_receipt_with_same_id_and_payload(self) -> None:
        job_id = str(uuid4())
        job: dict[str, object] = {
            "id": job_id,
            "workspace_id": "workspace/one",
            "owner_token": "dead-owner",
            "requested_by_id": "creator-1",
            "trigger": "scheduled",
            "database_names": ["app.sqlite3"],
            "snapshot_id": "snapshot-1",
            "snapshot_git_commit_hash": "commit-1",
            "status": "running",
        }
        store = _RuntimeRecoveryStore(job, projected=True)
        history = SimpleNamespace(
            runtime_history_active=mock.AsyncMock(return_value=True),
            runtime_capture_payload=SqliteHistoryService.runtime_capture_payload,
        )
        missing = HTTPException(status_code=404, detail="missing")
        request = mock.AsyncMock(
            side_effect=[
                missing,
                {"operation_id": job_id, "phase": "completed", "database_outcomes": {}},
                {"acknowledged": True},
            ]
        )
        service = queue.SqliteBackupQueueService(store)
        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(queue, "runtime_manager_request", request),
        ):
            self.assertEqual([job_id], await service.recover_stale())

        replay = request.await_args_list[1]
        self.assertEqual("POST", replay.args[0])
        self.assertEqual(job_id, replay.kwargs["json_payload"]["operation_id"])
        self.assertEqual("creator-1", replay.kwargs["json_payload"]["creator_id"])
        self.assertEqual(["app.sqlite3"], replay.kwargs["json_payload"]["database_names"])
        self.assertIn("workspace%2Fone", replay.args[1])
        self.assertEqual("POST", request.await_args_list[2].args[0])
        history.runtime_history_active.assert_awaited_once_with()

    async def test_terminal_runtime_row_retries_ack_without_rewriting_projection(self) -> None:
        job_id = str(uuid4())
        job: dict[str, object] = {
            "id": job_id,
            "workspace_id": "workspace-1",
            "owner_token": "dead-owner",
            "status": "completed",
        }
        store = _RuntimeRecoveryStore(job, projected=False)
        history = SimpleNamespace(runtime_history_active=mock.AsyncMock(return_value=True))
        request = mock.AsyncMock(
            side_effect=[
                {"operation_id": job_id, "phase": "completed", "database_outcomes": {}},
                RuntimeError("ack unavailable"),
                {"operation_id": job_id, "phase": "completed", "database_outcomes": {}},
                {"acknowledged": True},
            ]
        )
        service = queue.SqliteBackupQueueService(store)
        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(queue, "runtime_manager_request", request),
        ):
            self.assertEqual([job_id], await service.recover_stale())
            self.assertEqual([job_id], await service.recover_stale())

        self.assertEqual(2, len(store.projections))
        self.assertEqual("POST", request.await_args_list[1].args[0])
        self.assertEqual("POST", request.await_args_list[3].args[0])

    async def test_retired_runtime_receipt_is_not_resubmitted_after_not_found(self) -> None:
        job_id = str(uuid4())
        job: dict[str, object] = {
            "id": job_id,
            "workspace_id": "workspace-1",
            "owner_token": "dead-owner",
            "status": "completed",
        }
        store = _RuntimeRecoveryStore(job, projected=False)
        history = SimpleNamespace(runtime_history_active=mock.AsyncMock(return_value=True))
        request = mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="retired"))
        service = queue.SqliteBackupQueueService(store)
        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(queue, "runtime_manager_request", request),
        ):
            self.assertEqual([], await service.recover_stale())

        request.assert_awaited_once()
        self.assertFalse(store.projections)
