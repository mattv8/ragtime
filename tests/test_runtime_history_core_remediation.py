"""Regression coverage for runtime-owned SQLite-history remediation."""

from __future__ import annotations

import asyncio
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

from fastapi import HTTPException

from runtime.core.sqlite_workspace_state import MARKER_NAME, claim_marker
from runtime.worker.sqlite_history.maintenance import RuntimeHistoryMaintenance
from runtime.worker.sqlite_history.service import RuntimeSqliteHistoryService, SqliteHistoryService, _now, sqlite_workspace_access


class _Worker:
    def __init__(self, files: Path) -> None:
        self.files = files

    async def acquire_sqlite_workspace_access(self, workspace_id: str, lease_id: str, *, maintenance: bool) -> dict[str, str]:
        return {"authoritative_root": str(self.files)}

    async def release_sqlite_workspace_access(self, workspace_id: str, lease_id: str) -> None:
        return None


class RuntimeHistoryCoreRemediationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.files = self.root / "workspaces" / "workspace" / "files"
        (self.files / ".ragtime" / "db").mkdir(parents=True)
        self.service = RuntimeSqliteHistoryService(self.root, _Worker(self.files))

    async def asyncTearDown(self) -> None:
        self.temporary.cleanup()

    async def test_existing_marker_rejects_ordinary_access_inside_workspace_fence(self) -> None:
        marker = self.files.parent / "sqlite_backups" / MARKER_NAME
        claim_marker(marker, {"lease_id": "other", "workspace_id": "workspace"})
        async with self.service._installed():
            with self.assertRaises(HTTPException) as raised:
                async with sqlite_workspace_access("workspace"):
                    pass
        self.assertEqual(423, raised.exception.status_code)

    async def test_due_claim_replays_same_occurrence_id(self) -> None:
        root = self.files.parent / "sqlite_backups"
        root.mkdir()
        manifest = self.service._load(root, "workspace")
        manifest["next_scheduled_at"] = (_now() - timedelta(seconds=1)).isoformat()
        self.service._save(root, manifest)
        first = await self.service.claim_due(["workspace"])
        second = await self.service.claim_due(["workspace"])
        self.assertEqual(first, second)
        self.assertEqual("workspace", first[0]["workspace_id"])

    async def test_guarded_git_admission_drains_before_finish(self) -> None:
        # This exercises the guard bookkeeping without invoking SQLite capture.
        event = asyncio.Event()
        held: dict[str, Any] = {
            "workspace_id": "workspace",
            "user_id": "user",
            "closing": False,
            "git_substeps": 0,
            "git_drained": asyncio.Event(),
            "task": asyncio.current_task(),
        }
        held["git_drained"].set()
        self.service._guarded_restores["operation"] = held
        with mock.patch.object(self.service, "verify_guarded_code_restore_lease", new=mock.AsyncMock()):

            async def admitted() -> None:
                async with self.service.guarded_git_operation("workspace", "operation"):
                    event.set()
                    await asyncio.sleep(0.05)

            task = asyncio.create_task(admitted())
            await event.wait()
            held["closing"] = True
            with self.assertRaises(HTTPException):
                async with self.service.guarded_git_operation("workspace", "operation"):
                    pass
            self.assertFalse(held["git_drained"].is_set())
            await task
            self.assertTrue(held["git_drained"].is_set())

    async def test_busy_maintenance_withdraws_writer_while_guarded_restore_allows_read_and_finish(self) -> None:
        """A real begun guard remains usable while maintenance skips this pass."""
        (self.root / "workspaces" / "other" / "files" / ".ragtime" / "db").mkdir(parents=True)

        class _Repository:
            async def check(self, *, read_data: bool) -> None:
                raise AssertionError("busy maintenance must not reach repository work")

            async def prune(self, *, max_repack_size: int) -> None:
                raise AssertionError("busy maintenance must not reach repository work")

        class _MaintenanceService:
            def __init__(self, runtime: RuntimeSqliteHistoryService) -> None:
                self._runtime = runtime._runtime
                self.repository = _Repository()

            async def run_maintenance_once(self) -> None:
                return None

            async def drain_restic_forget_tombstones(self) -> list[str]:
                raise AssertionError("busy maintenance must not drain tombstones")

        # The capture boundary is outside this admission/lifecycle test.  The
        # resulting begun guard, its repository gate, marker, Git admission and
        # finish path are all the concrete production implementations.
        with mock.patch.object(self.service, "_capture_workspace_databases", new=mock.AsyncMock(return_value=[])):
            begun = await self.service.begin_guarded_code_restore("workspace", "maintenance-busy", "user")
            self.assertEqual("active", begun["status"])
            await RuntimeHistoryMaintenance(_MaintenanceService(self.service)).run_once()
            self.assertEqual([], await asyncio.wait_for(self.service.list_backups("other"), timeout=0.2))
            async with self.service.guarded_git_operation("workspace", "maintenance-busy"):
                pass
            finished = await self.service.finish_guarded_code_restore("workspace", "maintenance-busy", "user")
        self.assertEqual("completed", finished["status"])

    async def test_live_recovery_waits_for_admitted_git_before_finish(self) -> None:
        marker = self.files.parent / "sqlite_backups" / MARKER_NAME
        claim_marker(marker, {"lease_id": "lease", "workspace_id": "workspace"})
        held: dict[str, Any] = {
            "workspace_id": "workspace",
            "user_id": "user",
            "lease_id": "lease",
            "files": self.files,
            "closing": False,
            "git_substeps": 0,
            "git_drained": asyncio.Event(),
            "task": asyncio.current_task(),
        }
        held["git_drained"].set()
        self.service._guarded_restores["recovery"] = held
        entered = asyncio.Event()
        release = asyncio.Event()
        with (
            mock.patch.object(self.service, "verify_guarded_code_restore_lease", new=mock.AsyncMock()),
            mock.patch.object(self.service, "finish_guarded_code_restore", new=mock.AsyncMock(return_value={"status": "completed"})) as finish,
        ):

            async def git() -> None:
                async with self.service.guarded_git_operation("workspace", "recovery"):
                    entered.set()
                    await release.wait()

            git_task = asyncio.create_task(git())
            await entered.wait()
            recovery = asyncio.create_task(self.service.recover_guarded_code_restore("workspace", "recovery"))
            await asyncio.sleep(0)
            finish.assert_not_awaited()
            release.set()
            await git_task
            self.assertEqual("completed", (await recovery)["status"])
            finish.assert_awaited_once()

    async def test_guard_verification_does_not_reacquire_repository_gate(self) -> None:
        marker = self.files.parent / "sqlite_backups" / MARKER_NAME
        claim_marker(marker, {"lease_id": "lease", "workspace_id": "workspace"})
        self.service._guarded_restores["operation"] = {
            "workspace_id": "workspace",
            "lease_id": "lease",
            "files": self.files,
        }
        with mock.patch.object(self.service, "_installed", side_effect=AssertionError("must reuse held guard authority")):
            verified = await self.service.verify_guarded_code_restore_lease("workspace", "operation")
        self.assertEqual("lease", verified["lease_id"])

    async def test_wrong_guard_finisher_cannot_consume_lifecycle(self) -> None:
        begun = await self.service.begin_guarded_code_restore("workspace", "finisher", "creator")
        self.assertEqual("active", begun["status"])
        with self.assertRaises(HTTPException) as raised:
            await self.service.finish_guarded_code_restore("workspace", "finisher", "other")
        self.assertEqual(403, raised.exception.status_code)
        self.assertFalse(self.service._guarded_restores["finisher"]["finish"].done())
        await self.service.finish_guarded_code_restore("workspace", "finisher", "creator")


class RuntimeHistoryQuotaRemediationTests(unittest.TestCase):
    def test_quota_eviction_queues_restic_tombstone(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "sqlite_backups"
            root.mkdir()
            service = SqliteHistoryService(lambda _workspace: root.parent / "files")
            manifest = service._load(root, "workspace")
            manifest["backups"] = [
                {
                    "id": "old",
                    "database_name": "app.sqlite3",
                    "status": "ready",
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "size_bytes": 10,
                    "sha256": "a",
                    "storage": {"kind": "restic", "snapshot_id": "old-snapshot"},
                    "blob": None,
                },
                {
                    "id": "new",
                    "database_name": "app.sqlite3",
                    "status": "ready",
                    "created_at": "2021-01-01T00:00:00+00:00",
                    "size_bytes": 10,
                    "sha256": "b",
                    "storage": {"kind": "restic", "snapshot_id": "new-snapshot"},
                    "blob": None,
                },
            ]
            with mock.patch("runtime.worker.sqlite_history.service._MAX_WORKSPACE_BYTES", 25):
                service._enforce_quota(root, manifest, 10)
            self.assertEqual(["new"], [row["id"] for row in manifest["backups"]])
            self.assertEqual(["old-snapshot"], manifest["restic_forget_tombstones"])
