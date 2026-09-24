import asyncio
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator


class _History:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def capture_workspace_databases(self, _workspace_id, **kwargs):
        self.started.set()
        await self.release.wait()
        return [{"database_name": "app.sqlite3", "status": "ready"}]


class _CompletedHistory:
    async def capture_workspace_databases(self, _workspace_id, **_kwargs):
        return [{"database_name": "app.sqlite3", "status": "skipped_unchanged"}]


class RuntimeSqliteHistoryCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_durable_receipt_before_background_capture_and_cancels_after_drain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = _History()
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=lambda: history)
            coordinator.activate()
            operation_id = str(uuid4())
            receipt = await coordinator.accept_capture(
                "workspace-a",
                {"operation_id": operation_id, "creator_id": "local:admin", "database_names": ["app.sqlite3"]},
            )
            self.assertEqual(receipt["phase"], "accepted")
            await asyncio.wait_for(history.started.wait(), timeout=1)
            self.assertEqual((await coordinator.get_operation("workspace-a", operation_id))["phase"], "running")
            self.assertEqual((await coordinator.cancel("workspace-a", operation_id))["phase"], "cancelling")
            history.release.set()
            for _ in range(20):
                receipt = await coordinator.get_operation("workspace-a", operation_id)
                if receipt["phase"] == "cancelled":
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(receipt["phase"], "cancelled")

    async def test_inactive_capability_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=lambda: _History())
            self.assertFalse(coordinator.capability())
            with self.assertRaises(Exception):
                await coordinator.list_backups("workspace-a")

    async def test_completed_receipt_retains_authoritative_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=_CompletedHistory)
            coordinator.activate()
            operation_id = str(uuid4())
            await coordinator.accept_capture(
                "workspace-a",
                {"operation_id": operation_id, "creator_id": "local:admin", "database_names": ["app.sqlite3"]},
            )
            await coordinator._tasks[operation_id]

            receipt = await coordinator.get_operation("workspace-a", operation_id)

        self.assertEqual("completed", receipt["phase"])
        self.assertEqual({"completed": 1, "total": 1}, receipt["database_outcomes"]["progress"])
