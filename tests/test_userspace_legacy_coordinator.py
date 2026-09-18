import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.object_storage.legacy_coordinator import LegacyObjectStorageCoordinator
from ragtime.userspace.object_storage.legacy_migration import LegacyObjectStorageMigrator


class _Migrator(LegacyObjectStorageMigrator):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.receipts: dict[str, list[dict[str, object]]] = {}
        self.reconcile_calls: list[tuple[str, bool]] = []

    def load_receipts(self, workspace_id: str) -> list[dict[str, object]]:
        return self.receipts.get(workspace_id, [])

    def source_buckets(self, workspace_id: str) -> Path:
        return self.root / workspace_id / "s3" / "buckets"

    async def reconcile(self, workspace_id: str, *, runtime_active: bool = False) -> bool:
        self.reconcile_calls.append((workspace_id, runtime_active))
        return False


class LegacyObjectStorageCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.migrator = _Migrator(self.root)
        self.runtime_active = mock.AsyncMock(return_value=False)
        self.coordinator = LegacyObjectStorageCoordinator(
            self.migrator,
            workspace_ids=mock.AsyncMock(return_value=[]),
            runtime_active=self.runtime_active,
            legacy_payload=mock.Mock(return_value={"buckets": []}),
            config_path=lambda workspace_id: self.root / workspace_id / "config.json",
            workspaces_dir=self.root,
            logger=mock.Mock(),
        )

    async def asyncTearDown(self) -> None:
        await self.coordinator.shutdown()
        self.tempdir.cleanup()

    async def test_pending_workspace_retains_single_staging_admission(self) -> None:
        self.migrator.receipts["first"] = [{"cleanup_state": "published"}]
        self.migrator.receipts["second"] = [{"cleanup_state": "published"}]

        self.assertFalse(await self.coordinator.reconcile("first"))
        self.assertFalse(await self.coordinator.reconcile("second"))
        self.assertEqual([("first", False)], self.migrator.reconcile_calls)

    async def test_terminal_compact_receipt_skips_source_and_manifest_work(self) -> None:
        self.migrator.receipts["workspace"] = [
            {
                "workspace_id": "workspace",
                "generation": "a" * 32,
                "manifest_sha256": "b" * 64,
                "cleanup_state": "consumed",
                "retained_count": 0,
            }
        ]
        self.assertFalse(self.coordinator.needs_reconciliation("workspace"))

    async def test_lazy_read_returns_503_and_schedules_background_work(self) -> None:
        self.migrator.receipts["workspace"] = [{"cleanup_state": "published"}]
        with mock.patch(
            "ragtime.userspace.object_storage.legacy_coordinator.control.get_workspace",
            new=mock.AsyncMock(return_value={"state": "importing", "legacy_import_state": "pending"}),
        ):
            with self.assertRaises(HTTPException) as failure:
                await self.coordinator.ensure_managed("workspace")
        self.assertEqual(503, failure.exception.status_code)
        self.assertIn("workspace", self.coordinator.workspace_tasks)

    async def test_shutdown_cancels_and_drains_lazy_task(self) -> None:
        blocker = asyncio.Event()

        async def blocked(workspace_id: str, receipts: list[dict[str, Any]] | None = None) -> bool:
            del workspace_id, receipts
            await blocker.wait()
            return False

        with mock.patch.object(self.coordinator, "process", side_effect=blocked):
            self.coordinator.enqueue("workspace")
            await asyncio.sleep(0)
            await self.coordinator.shutdown()
        self.assertFalse(self.coordinator.workspace_tasks)
