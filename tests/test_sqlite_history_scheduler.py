import asyncio
import sys
import tempfile
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from ragtime.userspace.sqlite_history import SqliteHistoryService, _now


class SqliteHistorySchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        self.files.mkdir(parents=True)
        self.service = SqliteHistoryService(lambda _: self.files)
        self.root = self.files.parent / "sqlite_backups"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_first_seen_schedule_is_jittered_and_claim_completion_is_owned(self) -> None:
        self.assertFalse(self.service._cleanup_and_due_sync(self.root, "workspace"))
        manifest = self.service._load(self.root, "workspace")
        due = _now() + timedelta(minutes=5)
        self.assertLessEqual(_now(), due)
        self.assertIsNotNone(manifest["next_scheduled_at"])
        self.assertFalse(self.service._claim_scheduled_due_sync(self.root, "workspace"))
        manifest["next_scheduled_at"] = (_now() - timedelta(seconds=1)).isoformat()
        self.service._save(self.root, manifest)
        claim = self.service._claim_scheduled_due_sync(self.root, "workspace")
        self.assertIsInstance(claim, str)
        self.assertFalse(self.service._complete_scheduled_attempt_sync(self.root, "workspace", "wrong", success=True))
        self.assertTrue(self.service._complete_scheduled_attempt_sync(self.root, "workspace", str(claim), success=True))

    def test_liveness_proven_orphan_claim_is_replaced_and_failure_retries_soon(self) -> None:
        self.service._cleanup_and_due_sync(self.root, "workspace")
        manifest = self.service._load(self.root, "workspace")
        manifest["next_scheduled_at"] = (_now() - timedelta(seconds=1)).isoformat()
        manifest["scheduled_claim"] = {"claim_id": "crashed", "claimed_at": (_now() - timedelta(days=1)).isoformat()}
        self.service._save(self.root, manifest)
        handle = self.service._try_scheduled_liveness_lock(self.root)
        self.assertIsNotNone(handle)
        try:
            claim = self.service._claim_scheduled_due_sync(self.root, "workspace", try_lock=True, recover_orphan_claim=True)
        finally:
            self.service._release_scheduled_liveness_lock(handle)
        self.assertIsInstance(claim, str)
        self.assertTrue(self.service._complete_scheduled_attempt_sync(self.root, "workspace", str(claim), success=False))
        retry = self.service._next_scheduled_due_sync(self.root, "workspace")
        self.assertLessEqual(retry, _now() + timedelta(minutes=5, seconds=1))

    def test_nonblocking_catalog_lock_reports_busy(self) -> None:
        from ragtime.userspace.sqlite_history import _catalog_lock

        with _catalog_lock(self.root):
            with self.assertRaises(Exception):
                self.service._cleanup_and_due_sync(self.root, "workspace", try_lock=True)

    def test_maintenance_enqueues_due_workspace_without_capturing(self) -> None:
        workspace_root = Path(self.temp.name) / "scheduled" / "workspaces" / "workspace"
        files = workspace_root / "files"
        files.mkdir(parents=True)
        service = SqliteHistoryService(lambda _: files)
        root = files.parent / "sqlite_backups"
        service._cleanup_and_due_sync(root, "workspace")
        manifest = service._load(root, "workspace")
        manifest["next_scheduled_at"] = (_now() - timedelta(seconds=1)).isoformat()
        service._save(root, manifest)
        module = ModuleType("ragtime.userspace.service")
        setattr(module, "userspace_service", SimpleNamespace(root_path=workspace_root.parent.parent))
        queue = SimpleNamespace(enqueue=mock.AsyncMock(return_value={"id": "scheduled-job", "status": "pending"}))
        queue_module = ModuleType("ragtime.userspace.sqlite_backup_queue")
        setattr(queue_module, "get_sqlite_backup_queue_service", lambda: queue)
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "ragtime.userspace.service": module,
                    "ragtime.userspace.sqlite_backup_queue": queue_module,
                },
            ),
            mock.patch.object(service, "capture_workspace_databases", new_callable=mock.AsyncMock) as capture,
        ):
            asyncio.run(service.run_maintenance_once())
            with mock.patch.object(service, "_root", side_effect=AssertionError("cached no-due workspace should not touch storage")):
                asyncio.run(service.run_maintenance_once())
        capture.assert_not_awaited()
        queue.enqueue.assert_awaited_once()
        _, kwargs = queue.enqueue.await_args
        self.assertEqual("scheduled", kwargs["trigger"])
        self.assertEqual("scheduled:workspace:", kwargs["request_key"][:20])

    def test_cancelled_liveness_acquisition_drains_and_releases_handle(self) -> None:
        entered = threading.Event()
        unblock = threading.Event()
        original = self.service._try_scheduled_liveness_lock

        def blocked(root: Path):
            entered.set()
            unblock.wait(timeout=1)
            return original(root)

        async def cancel() -> None:
            with mock.patch.object(self.service, "_try_scheduled_liveness_lock", side_effect=blocked):
                task = asyncio.create_task(self.service._acquire_scheduled_liveness(self.root))
                await asyncio.to_thread(entered.wait)
                task.cancel()
                unblock.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        asyncio.run(cancel())
        handle = self.service._try_scheduled_liveness_lock(self.root)
        self.assertIsNotNone(handle)
        self.service._release_scheduled_liveness_lock(handle)

    def test_failed_save_keeps_expired_preview_candidate_and_manifest(self) -> None:
        candidate_dir = self.root / "candidates"
        candidate_dir.mkdir(parents=True)
        candidate = candidate_dir / "expired.sqlite3"
        candidate.write_bytes(b"candidate")
        manifest = {
            "version": 1,
            "workspace_id": "workspace",
            "backups": [],
            "operations": {},
            "last_scheduled_at": None,
            "previews": {"expired": {"candidate": "candidates/expired.sqlite3", "expires_at": (_now() - timedelta(seconds=1)).isoformat()}},
        }
        self.service._save(self.root, manifest)
        with mock.patch.object(self.service, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.service._cleanup_and_due_sync(self.root, "workspace")
        self.assertTrue(candidate.exists())
        self.assertIn("expired", self.service._load(self.root, "workspace")["previews"])
