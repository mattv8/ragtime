from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from runtime import main as runtime_main
from runtime.manager import api as manager_api
from runtime.worker import api as worker_api
from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator


class _HistoryService:
    def __init__(self, root: Path) -> None:
        self._runtime = SimpleNamespace(root=root)
        self.reconciled: list[str] = []
        self.maintenance_ran = asyncio.Event()

    async def reconcile_capture_operation(self, receipt: dict[str, str]) -> dict[str, str]:
        self.reconciled.append(receipt["operation_id"])
        return {"phase": "interrupted", "error": "recovered after manager restart"}

    async def run_maintenance_once(self) -> None:
        self.maintenance_ran.set()


class _WorkerService:
    def __init__(self, coordinator: SqliteHistoryCoordinator) -> None:
        self._coordinator = coordinator

    def sqlite_history_coordinator(self) -> SqliteHistoryCoordinator:
        return self._coordinator


class RuntimeManagerHistoryLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_manager_lifespan_starts_and_stops_embedded_history_coordinator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_sqlite_history").mkdir()
            (root / "_sqlite_history" / "activation-v1.json").write_text('{"version":2,"active":true}', encoding="utf-8")
            history = _HistoryService(root)
            coordinator = SqliteHistoryCoordinator(root, object(), history_factory=lambda: history)
            operation_id = str(uuid4())
            coordinator._store.accept(
                operation_id=operation_id,
                workspace_id="workspace-lifecycle",
                creator_id="local:owner",
                request_digest="request-digest",
                kind="capture",
                accepted_payload={},
            )
            # A released liveness lock represents work left by a dead process.
            with coordinator._store.hold_liveness(operation_id):
                pass
            worker = _WorkerService(coordinator)
            manager = mock.AsyncMock()
            manager.startup = mock.AsyncMock()
            manager.shutdown = mock.AsyncMock()

            async def shutdown_worker_resources() -> None:
                await coordinator.shutdown()

            with (
                mock.patch.dict(os.environ, {"RUNTIME_SERVICE_MODE": "manager"}),
                mock.patch.object(manager_api, "SessionManager", return_value=manager),
                mock.patch.object(manager_api, "get_worker_service", return_value=worker),
                mock.patch.object(worker_api, "shutdown_worker_resources", side_effect=shutdown_worker_resources) as shutdown,
            ):
                application = runtime_main.create_app()
                async with application.router.lifespan_context(application):
                    await asyncio.wait_for(history.maintenance_ran.wait(), timeout=1)
                    self.assertEqual(history.reconciled, [operation_id])
                    self.assertEqual(coordinator._store.get(operation_id)["phase"], "interrupted")
                    self.assertTrue(coordinator._started)
                    self.assertIsNotNone(coordinator._maintenance)

            manager.startup.assert_awaited_once()
            manager.shutdown.assert_awaited_once()
            shutdown.assert_awaited_once()
            self.assertFalse(coordinator._started)
            self.assertIsNone(coordinator._maintenance_task)
