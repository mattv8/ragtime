from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from uuid import uuid4

from fastapi import HTTPException

from ragtime.userspace.sqlite_history import SqliteHistoryService


class RuntimeHistoryFacadeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = mock.patch("ragtime.userspace.sqlite_history.settings.index_data_path", temporary.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_list_uses_runtime_without_opening_local_catalog(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: (_ for _ in ()).throw(AssertionError("local catalog opened")))
        request = mock.AsyncMock(side_effect=[{"active": True, "capability": True}, {"backups": [{"id": "backup-1"}]}])
        with (
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_enabled", return_value=True),
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_request", request),
        ):
            backups = await service.list_backups("workspace-1")

        self.assertEqual([{"id": "backup-1"}], backups)
        self.assertEqual(request.await_count, 2)
        self.assertTrue(request.await_args_list[0].args[1].endswith("/activation"))
        self.assertTrue(request.await_args_list[1].args[1].endswith("/sqlite-history"))
        self.assertEqual("GET", request.await_args_list[1].args[0])

    async def test_capture_reuses_parent_queue_job_id_and_returns_receipt_outcomes(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: Path("/must-not-open"))
        job_id = str(uuid4())
        request = mock.AsyncMock(
            side_effect=[
                {"active": True, "capability": True},
                {
                    "operation_id": job_id,
                    "phase": "completed",
                    "database_outcomes": {"app.sqlite3": {"operation_id": str(uuid4()), "results": [{"id": "backup-1", "status": "ready"}]}},
                },
            ]
        )
        with (
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_enabled", return_value=True),
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_request", request),
        ):
            results = await service.capture_workspace_databases(
                "workspace-1",
                trigger="manual",
                database_names={"app.sqlite3"},
                capture_job_id=job_id,
            )

        self.assertEqual([{"id": "backup-1", "status": "ready"}], results)
        self.assertIsNotNone(request.await_args)
        await_args = request.await_args
        assert await_args is not None
        payload = await_args.kwargs["json_payload"]
        self.assertEqual(job_id, payload["operation_id"])
        self.assertEqual(["app.sqlite3"], payload["database_names"])

    async def test_immediately_completed_runtime_capture_reports_terminal_progress(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: Path("/must-not-open"))
        job_id = str(uuid4())
        callback = mock.AsyncMock()
        request = mock.AsyncMock(
            side_effect=[
                {"active": True, "capability": True},
                {
                    "operation_id": job_id,
                    "phase": "completed",
                    "database_outcomes": {
                        "progress": {"completed": 1, "total": 1},
                        "app.sqlite3": {"operation_id": str(uuid4()), "results": [{"id": "backup-1", "status": "ready"}]},
                    },
                },
            ]
        )
        with (
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_enabled", return_value=True),
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_request", request),
        ):
            results = await service.capture_workspace_databases(
                "workspace-1",
                trigger="manual",
                database_names={"app.sqlite3"},
                capture_job_id=job_id,
                progress_callback=callback,
            )

        self.assertEqual([{"id": "backup-1", "status": "ready"}], results)
        callback.assert_awaited_once_with(1, 1)

    async def test_runtime_snapshot_metadata_never_invokes_local_workspace_callback(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: (_ for _ in ()).throw(AssertionError("local workspace opened")))
        request = mock.AsyncMock(side_effect=[{"active": True, "capability": True}, {"backups": [{"snapshot_id": "snapshot-1"}, {"id": "backup-2"}]}])
        with (
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_enabled", return_value=True),
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_request", request),
        ):
            snapshot_ids = await service.get_snapshot_ids_with_backups("workspace-1")

        self.assertEqual({"snapshot-1"}, snapshot_ids)

    async def test_runtime_scheduler_enqueues_idempotently_then_acknowledges_due_claim(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: (_ for _ in ()).throw(AssertionError("local catalog opened")))
        queue = SimpleNamespace(enqueue=mock.AsyncMock(return_value={"id": "scheduled-job"}))
        service._runtime_due_receipts[("workspace-1", "occurrence-1")] = "scheduled-job"
        history_request = mock.AsyncMock(
            side_effect=[
                {"acknowledged": True},
                {"claims": [{"workspace_id": "workspace-1", "occurrence_id": "occurrence-2"}]},
                {"acknowledged": True},
            ]
        )
        queue_module = ModuleType("ragtime.userspace.sqlite_backup_queue")
        setattr(queue_module, "get_sqlite_backup_queue_service", lambda: queue)
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[{"id": "workspace-1"}]))
        with (
            mock.patch.dict("sys.modules", {"ragtime.userspace.sqlite_backup_queue": queue_module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(service, "runtime_history_active", new_callable=mock.AsyncMock, return_value=True),
            mock.patch.object(service, "_runtime_history_request", history_request),
        ):
            await service.run_maintenance_once()

        self.assertEqual(2, queue.enqueue.await_count)
        self.assertEqual("scheduled:workspace-1:occurrence-1", queue.enqueue.await_args_list[0].kwargs["request_key"])
        self.assertEqual("scheduled:workspace-1:occurrence-2", queue.enqueue.await_args_list[1].kwargs["request_key"])
        self.assertEqual({}, service._runtime_due_receipts)
        self.assertEqual("/sqlite-history/due/claim", history_request.await_args_list[1].args[1])
        self.assertEqual("/workspaces/workspace-1/sqlite-history/due/ack", history_request.await_args_list[2].args[1])

    async def test_runtime_capture_404_replays_exact_operation_payload(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: Path("/must-not-open"))
        operation_id = str(uuid4())
        missing = HTTPException(status_code=404, detail="not found")
        request = mock.AsyncMock(
            side_effect=[
                missing,
                missing,
                {"operation_id": operation_id, "phase": "completed", "database_outcomes": {}},
            ]
        )
        with (
            mock.patch.object(service, "runtime_history_active", new_callable=mock.AsyncMock, return_value=True),
            mock.patch.object(service, "_runtime_history_request", request),
        ):
            await service.capture_workspace_databases("workspace/one", trigger="manual", capture_job_id=operation_id)

        self.assertEqual(3, request.await_count)
        self.assertEqual(request.await_args_list[0].kwargs["payload"], request.await_args_list[2].kwargs["payload"])
        self.assertEqual(operation_id, request.await_args_list[2].kwargs["payload"]["operation_id"])
        self.assertIn("workspace%2Fone", request.await_args_list[0].args[1])

    async def test_runtime_scheduler_pages_database_membership_in_eight_workspace_rotations(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: Path("/must-not-open"))
        db = SimpleNamespace(
            query_raw=mock.AsyncMock(
                side_effect=[
                    [{"id": f"workspace-{number}"} for number in range(8)],
                    [{"id": "workspace-8"}],
                ]
            )
        )
        queue = SimpleNamespace(enqueue=mock.AsyncMock(return_value={"id": "scheduled-job"}))
        queue_module = ModuleType("ragtime.userspace.sqlite_backup_queue")
        setattr(queue_module, "get_sqlite_backup_queue_service", lambda: queue)
        request = mock.AsyncMock(return_value={"claims": []})
        with (
            mock.patch.dict("sys.modules", {"ragtime.userspace.sqlite_backup_queue": queue_module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(service, "runtime_history_active", new_callable=mock.AsyncMock, return_value=True),
            mock.patch.object(service, "_runtime_history_request", request),
        ):
            await service.run_maintenance_once()
            await service.run_maintenance_once()

        self.assertEqual(
            [
                mock.call("SELECT id FROM workspaces ORDER BY id LIMIT $1 OFFSET $2", 8, 0),
                mock.call("SELECT id FROM workspaces ORDER BY id LIMIT $1 OFFSET $2", 8, 8),
            ],
            db.query_raw.await_args_list,
        )
        self.assertEqual([f"workspace-{number}" for number in range(8)], request.await_args_list[0].kwargs["payload"]["workspace_ids"])
        self.assertEqual(["workspace-8"], request.await_args_list[1].kwargs["payload"]["workspace_ids"])

    async def test_configured_runtime_with_unknown_activation_fails_closed(self) -> None:
        service = SqliteHistoryService(lambda _workspace_id: Path("/must-not-open"))
        with (
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_enabled", return_value=True),
            mock.patch("ragtime.userspace.sqlite_history.runtime_manager_request", new=mock.AsyncMock(return_value={"capability": True})),
            self.assertRaises(HTTPException) as raised,
        ):
            await service.get_snapshot_ids_with_backups("workspace-1")

        self.assertEqual(503, raised.exception.status_code)
