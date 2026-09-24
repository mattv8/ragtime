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
        runtime_active_patch = mock.patch.object(
            SqliteHistoryService,
            "runtime_history_active",
            new=mock.AsyncMock(return_value=False),
        )
        runtime_active_patch.start()
        self.addCleanup(runtime_active_patch.stop)
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
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[{"id": "workspace"}]))
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
            mock.patch("ragtime.userspace.sqlite_history.get_db", return_value=db),
            mock.patch("ragtime.userspace.sqlite_history_confinement.confinement_available", return_value=True),
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

    def test_maintenance_cleans_up_but_does_not_enqueue_when_confinement_is_unavailable(self) -> None:
        workspace_root = Path(self.temp.name) / "scheduled" / "workspaces" / "workspace"
        files = workspace_root / "files"
        files.mkdir(parents=True)
        service = SqliteHistoryService(lambda _: files)
        root = files.parent / "sqlite_backups"
        service._cleanup_and_due_sync(root, "workspace")
        candidate = root / "candidates" / "expired.sqlite3"
        candidate.parent.mkdir()
        candidate.write_bytes(b"expired")
        manifest = service._load(root, "workspace")
        due_at = (_now() - timedelta(seconds=1)).isoformat()
        manifest["next_scheduled_at"] = due_at
        manifest["previews"] = {
            "expired": {"candidate": "candidates/expired.sqlite3", "expires_at": due_at},
        }
        service._save(root, manifest)
        module = ModuleType("ragtime.userspace.service")
        setattr(module, "userspace_service", SimpleNamespace(root_path=workspace_root.parent.parent))
        queue = SimpleNamespace(enqueue=mock.AsyncMock())
        queue_module = ModuleType("ragtime.userspace.sqlite_backup_queue")
        setattr(queue_module, "get_sqlite_backup_queue_service", lambda: queue)
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[{"id": "workspace"}]))
        with (
            mock.patch.dict(sys.modules, {"ragtime.userspace.service": module, "ragtime.userspace.sqlite_backup_queue": queue_module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", return_value=db),
            mock.patch("ragtime.userspace.sqlite_history_confinement.confinement_available", return_value=False),
        ):
            asyncio.run(service.run_maintenance_once())
        queue.enqueue.assert_not_awaited()
        unavailable_manifest = service._load(root, "workspace")
        self.assertEqual(due_at, unavailable_manifest["next_scheduled_at"])
        self.assertNotIn("scheduled_claim", unavailable_manifest)
        self.assertNotIn("expired", unavailable_manifest["previews"])
        self.assertFalse(candidate.exists())
        with (
            mock.patch.dict(sys.modules, {"ragtime.userspace.service": module, "ragtime.userspace.sqlite_backup_queue": queue_module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", return_value=db),
            mock.patch("ragtime.userspace.sqlite_history_confinement.confinement_available", return_value=True),
        ):
            asyncio.run(service.run_maintenance_once())
        queue.enqueue.assert_awaited_once()

    def test_maintenance_only_schedules_catalog_workspaces(self) -> None:
        workspace_root = Path(self.temp.name) / "scheduled" / "workspaces"
        valid_files = workspace_root / "valid" / "files"
        orphan_files = workspace_root / "chat-diag" / "files"
        valid_files.mkdir(parents=True)
        orphan_files.mkdir(parents=True)
        service = SqliteHistoryService(lambda workspace_id: workspace_root / workspace_id / "files")
        valid_root = valid_files.parent / "sqlite_backups"
        orphan_root = orphan_files.parent / "sqlite_backups"
        for workspace_id, root in (("valid", valid_root), ("chat-diag", orphan_root)):
            service._cleanup_and_due_sync(root, workspace_id)
            manifest = service._load(root, workspace_id)
            manifest["next_scheduled_at"] = (_now() - timedelta(seconds=1)).isoformat()
            service._save(root, manifest)
        orphan_due = service._load(orphan_root, "chat-diag")["next_scheduled_at"]

        module = ModuleType("ragtime.userspace.service")
        setattr(module, "userspace_service", SimpleNamespace(root_path=workspace_root.parent))
        queue = SimpleNamespace(enqueue=mock.AsyncMock(return_value={"id": "scheduled-job", "status": "pending"}))
        queue_module = ModuleType("ragtime.userspace.sqlite_backup_queue")
        setattr(queue_module, "get_sqlite_backup_queue_service", lambda: queue)
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[{"id": "valid"}]))
        with (
            mock.patch.dict(sys.modules, {"ragtime.userspace.service": module, "ragtime.userspace.sqlite_backup_queue": queue_module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", return_value=db),
            mock.patch("ragtime.userspace.sqlite_history_confinement.confinement_available", return_value=True),
        ):
            asyncio.run(service.run_maintenance_once())

        queue.enqueue.assert_awaited_once()
        self.assertEqual("valid", queue.enqueue.await_args.args[0])
        self.assertEqual([], service._load(orphan_root, "chat-diag")["backups"])
        self.assertEqual(orphan_due, service._load(orphan_root, "chat-diag")["next_scheduled_at"])

    def test_maintenance_does_not_mask_catalog_lookup_failure_when_workspace_root_is_missing(self) -> None:
        module = ModuleType("ragtime.userspace.service")
        setattr(module, "userspace_service", SimpleNamespace(root_path=Path(self.temp.name) / "missing"))
        db = SimpleNamespace(query_raw=mock.AsyncMock(side_effect=RuntimeError("database unavailable")))
        with (
            mock.patch.dict(sys.modules, {"ragtime.userspace.service": module}),
            mock.patch("ragtime.userspace.sqlite_history.get_db", return_value=db),
            self.assertRaisesRegex(RuntimeError, "database unavailable"),
        ):
            asyncio.run(self.service.run_maintenance_once())

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

    def test_unchanged_cleanup_does_not_save_catalog_again(self) -> None:
        self.service._cleanup_and_due_sync(self.root, "workspace")

        with mock.patch.object(self.service, "_save", wraps=self.service._save) as save:
            self.service._cleanup_and_due_sync(self.root, "workspace")

        save.assert_not_called()
