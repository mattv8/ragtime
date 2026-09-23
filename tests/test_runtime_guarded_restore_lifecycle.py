"""Guarded SQLite restore contexts must live in their dedicated task."""

from __future__ import annotations

import asyncio
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from runtime.core.sqlite_workspace_state import MARKER_NAME, workspace_operation
from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService


class _Worker:
    def __init__(self, files: Path) -> None:
        self.files = files
        self.acquired = asyncio.Event()
        self.released = asyncio.Event()

    async def acquire_sqlite_workspace_access(self, workspace_id: str, lease_id: str, *, maintenance: bool) -> dict[str, str]:
        self.acquired.set()
        return {"authoritative_root": str(self.files)}

    async def release_sqlite_workspace_access(self, workspace_id: str, lease_id: str) -> None:
        self.released.set()


class GuardedRestoreLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = self.root / "workspaces" / "workspace" / "files"
        database = self.files / ".ragtime" / "db" / "app.sqlite3"
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE items (value TEXT)")
            connection.execute("INSERT INTO items VALUES ('preserved')")
        self.worker = _Worker(self.files)
        self.service = RuntimeSqliteHistoryService(self.root, self.worker)

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def _begin_in_independent_task(self) -> dict[str, object]:
        begin = asyncio.create_task(self.service.begin_guarded_code_restore("workspace", "operation", "user"))
        await self.worker.acquired.wait()
        result = await begin
        self.assertIsNot(begin, self.service._guarded_restores["operation"]["task"])
        return result

    async def test_begin_and_finish_from_independent_tasks_keep_one_lifecycle_owner(self) -> None:
        begun = await self._begin_in_independent_task()
        self.assertEqual("active", begun["status"])
        self.assertTrue((self.files.parent / "sqlite_backups" / MARKER_NAME).exists())

        finish = asyncio.create_task(self.service.finish_guarded_code_restore("workspace", "operation", "user"))
        self.assertEqual("completed", (await finish)["status"])
        await self.worker.released.wait()

        self.assertNotIn("operation", self.service._guarded_restores)
        self.assertFalse((self.files.parent / "sqlite_backups" / MARKER_NAME).exists())
        with workspace_operation(self.root, "workspace", exclusive=True, nonblocking=True):
            pass

    async def test_cancelled_begin_does_not_cancel_dedicated_guard(self) -> None:
        begin = asyncio.create_task(self.service.begin_guarded_code_restore("workspace", "operation", "user"))
        await self.worker.acquired.wait()
        begin.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await begin

        begun = await self.service.begin_guarded_code_restore("workspace", "operation", "user")
        self.assertEqual("active", begun["status"])
        self.assertFalse(self.service._guarded_restores["operation"]["task"].done())
        await self.service.stop()
        self.assertFalse(self.service._guarded_restores["operation"]["task"].done())
        await self.service.finish_guarded_code_restore("workspace", "operation", "user")

    async def test_prepublication_failure_keeps_marker_but_releases_live_lock(self) -> None:
        with mock.patch.object(self.service, "_capture_workspace_databases", new=mock.AsyncMock(side_effect=OSError("capture failed"))):
            with self.assertRaisesRegex(OSError, "capture failed"):
                await asyncio.create_task(self.service.begin_guarded_code_restore("workspace", "operation", "user"))

        self.assertTrue((self.files.parent / "sqlite_backups" / MARKER_NAME).exists())
        await self.worker.released.wait()
        with workspace_operation(self.root, "workspace", exclusive=True, nonblocking=True):
            pass

    async def test_finish_publication_error_leaves_durable_marker_for_recovery(self) -> None:
        await self._begin_in_independent_task()
        with mock.patch.object(self.service, "_publish_verified_candidate", side_effect=OSError("publication failed")):
            with self.assertRaisesRegex(OSError, "publication failed"):
                await asyncio.create_task(self.service.finish_guarded_code_restore("workspace", "operation", "user"))

        marker = self.files.parent / "sqlite_backups" / MARKER_NAME
        self.assertTrue(marker.exists())
        recovered = RuntimeSqliteHistoryService(self.root, self.worker)
        self.assertEqual("completed", (await recovered.recover_guarded_code_restore("workspace", "operation"))["status"])
        self.assertFalse(marker.exists())
        with workspace_operation(self.root, "workspace", exclusive=True, nonblocking=True):
            pass
