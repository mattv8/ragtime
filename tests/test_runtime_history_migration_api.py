from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from runtime.worker.sqlite_history.coordinator import SqliteHistoryCoordinator
from runtime.worker.sqlite_history.migration import LegacyHistoryMigration


class _History:
    def __init__(self) -> None:
        self.calls = 0

    async def migrate_legacy_backups(self, workspace_id: str, *, pass_fds: tuple[int, ...] = (), cancel_check=None) -> int:
        self.calls += 1
        return 3


class RuntimeHistoryMigrationApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_migration_completes_and_exact_replay_does_not_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = _History()
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=lambda: history)
            coordinator.activate()
            migrations = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            accepted = await migrations.accept("workspace-a", {"operation_id": operation_id, "user_id": "local:admin"})
            replay = await migrations.accept("workspace-a", {"operation_id": operation_id, "user_id": "local:admin"})
            self.assertEqual("completed", accepted["phase"])
            self.assertEqual(accepted, replay)
            self.assertEqual(1, history.calls)
            self.assertEqual(3, accepted["database_outcomes"]["legacy_migration"]["migrated_backup_count"])

    async def test_workspace_scope_hides_other_workspace_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=_History)
            coordinator.activate()
            migrations = LegacyHistoryMigration(coordinator)
            operation_id = str(uuid4())
            await migrations.accept("workspace-a", {"operation_id": operation_id, "user_id": "local:admin"})
            with self.assertRaises(HTTPException) as raised:
                await migrations.get("workspace-b", operation_id)
            self.assertEqual(404, raised.exception.status_code)

    async def test_failure_is_durably_reported(self) -> None:
        class FailingHistory:
            async def migrate_legacy_backups(self, workspace_id: str, *, pass_fds: tuple[int, ...] = (), cancel_check=None) -> int:
                raise RuntimeError("repository unavailable")

        with tempfile.TemporaryDirectory() as directory:
            coordinator = SqliteHistoryCoordinator(Path(directory), object(), history_factory=FailingHistory)
            coordinator.activate()
            operation_id = str(uuid4())
            receipt = await LegacyHistoryMigration(coordinator).accept("workspace-a", {"operation_id": operation_id, "user_id": "local:admin"})
            self.assertEqual("failed", receipt["phase"])
            self.assertEqual("SQLite history legacy migration failed", receipt["error"])
