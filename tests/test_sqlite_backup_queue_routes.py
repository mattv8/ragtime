from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import sqlite_history_routes as routes
from ragtime.userspace.sqlite_history_models import SqliteHistoryCaptureJobRequest


def _job(**changes):
    now = datetime.now(timezone.utc)
    job = {
        "id": "job-1",
        "workspace_id": "workspace-1",
        "trigger": "manual",
        "database_names": ["app.sqlite3"],
        "snapshot_id": None,
        "snapshot_git_commit_hash": None,
        "status": "pending",
        "created_at": now,
        "available_at": now,
        "started_at": None,
        "finished_at": None,
        "updated_at": now,
        "completed_databases": 0,
        "total_databases": 0,
        "backup_ids": [],
        "error_message": None,
        "cancel_requested": False,
        "owner_token": "private-owner-token",
        "request_hash": "private-request-hash",
        "internal_path": "/private/path",
    }
    job.update(changes)
    return job


class SqliteBackupQueueRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.queue = SimpleNamespace(
            enqueue=mock.AsyncMock(return_value=_job()),
            list_jobs=mock.AsyncMock(return_value=[_job()]),
            get_job=mock.AsyncMock(return_value=_job()),
            cancel=mock.AsyncMock(return_value=_job(status="cancelled", finished_at=datetime.now(timezone.utc))),
        )
        self.queue_module = SimpleNamespace(get_sqlite_backup_queue_service=lambda: self.queue)
        self.manage = mock.patch.object(routes, "_manage", new_callable=mock.AsyncMock)
        self.manage.start()
        self.modules = mock.patch.dict(
            sys.modules,
            {"ragtime.userspace.sqlite_backup_queue": self.queue_module},
        )
        self.modules.start()
        self.user = SimpleNamespace(id="owner-1", role="user")

    async def asyncTearDown(self) -> None:
        self.modules.stop()
        self.manage.stop()

    async def test_enqueue_passes_actor_and_hides_queue_internals(self) -> None:
        result = await routes.enqueue_sqlite_history_capture_job(
            "workspace-1",
            SqliteHistoryCaptureJobRequest(database_name="app.sqlite3", request_id="request-123"),
            self.user,
        )

        self.queue.enqueue.assert_awaited_once_with(
            "workspace-1",
            trigger="manual",
            database_names={"app.sqlite3"},
            requested_by_id="owner-1",
            request_key="request-123",
        )
        self.assertNotIn("owner_token", result["job"])
        self.assertNotIn("request_hash", result["job"])
        self.assertNotIn("internal_path", result["job"])

    async def test_list_get_and_cancel_are_workspace_scoped(self) -> None:
        await routes.list_sqlite_history_capture_jobs("workspace-1", "app.sqlite3", "snapshot-1", self.user)
        await routes.get_sqlite_history_capture_job("workspace-1", "job-1", self.user)
        await routes.cancel_sqlite_history_capture_job("workspace-1", "job-1", self.user)

        self.queue.list_jobs.assert_awaited_once_with(
            "workspace-1", database_name="app.sqlite3", snapshot_id="snapshot-1", limit=50
        )
        self.queue.get_job.assert_awaited_once_with("workspace-1", "job-1")
        self.queue.cancel.assert_awaited_once_with("workspace-1", "job-1")

    async def test_missing_job_is_not_exposed_across_workspaces(self) -> None:
        self.queue.get_job.return_value = None
        with self.assertRaises(HTTPException) as error:
            await routes.get_sqlite_history_capture_job("foreign-workspace", "job-1", self.user)
        self.assertEqual(404, error.exception.status_code)
        self.queue.get_job.assert_awaited_once_with("foreign-workspace", "job-1")

    async def test_owner_and_admin_management_checks_use_the_actor_identity(self) -> None:
        self.manage.stop()
        try:
            with mock.patch(
                "ragtime.userspace.service.userspace_service._enforce_workspace_access",
                new_callable=mock.AsyncMock,
            ) as enforce:
                await routes._manage("workspace-1", SimpleNamespace(id="owner-1", role="user"))
                await routes._manage("workspace-2", SimpleNamespace(id="admin-1", role="admin"))
        finally:
            self.manage.start()

        self.assertEqual(
            [
                mock.call("workspace-1", "owner-1", required_role="owner", is_admin=False),
                mock.call("workspace-2", "admin-1", required_role="owner", is_admin=True),
            ],
            enforce.await_args_list,
        )

    async def test_foreign_workspace_access_stops_queue_call(self) -> None:
        self.manage.stop()
        try:
            with mock.patch.object(
                routes,
                "_manage",
                new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden")),
            ):
                with self.assertRaises(HTTPException) as error:
                    await routes.enqueue_sqlite_history_capture_job(
                        "foreign-workspace",
                        SqliteHistoryCaptureJobRequest(database_name="app.sqlite3"),
                        self.user,
                    )
        finally:
            self.manage.start()

        self.assertEqual(403, error.exception.status_code)
        self.queue.enqueue.assert_not_awaited()

    def test_capture_job_routes_precede_dynamic_backup_routes(self) -> None:
        paths = [route.path for route in routes.router.routes]
        self.assertLess(
            paths.index("/indexes/userspace/workspaces/{workspace_id}/sqlite-history/capture-jobs"),
            paths.index("/indexes/userspace/workspaces/{workspace_id}/sqlite-history/{backup_id}/download"),
        )
