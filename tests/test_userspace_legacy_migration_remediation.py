import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest import mock

from ragtime.core.file_lock import backup_restore_lock
from ragtime.userspace.object_storage.legacy_migration import LegacyObjectStorageMigrator


class LegacyMigrationRemediationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.workspace = self.root / "workspaces" / "ws"
        self.source = self.workspace / "s3" / "buckets" / "assets"
        self.source.mkdir(parents=True)
        self.migrator = LegacyObjectStorageMigrator(self.root / "storage", lambda _: self.workspace)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stage_globally_sorts_nested_and_unicode_manifest_paths(self) -> None:
        (self.source / "dir").mkdir()
        (self.source / "dir" / "file.txt").write_bytes(b"nested")
        (self.source / "dir.txt").write_bytes(b"sibling")
        (self.source / "\U00010000.txt").write_bytes(b"supplementary")
        (self.source / "\ue000.txt").write_bytes(b"private-use")

        receipt = self.migrator.stage("ws")
        paths = [entry["path"] for entry in receipt["manifest"]["files"]]

        self.assertEqual(sorted(paths), paths)
        self.assertLess(paths.index("buckets/assets/dir.txt"), paths.index("buckets/assets/dir/file.txt"))
        self.assertLess(paths.index("buckets/assets/\ue000.txt"), paths.index("buckets/assets/\U00010000.txt"))

    def test_v1_consumed_receipt_compacts_and_restored_active_replaces_it(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        receipt = self.migrator.stage("ws")
        receipt["cleanup_state"] = "consumed"
        receipt["retained_files"] = {"buckets/assets/a.txt": "source changed"}
        path = self.migrator._receipt_path("ws", receipt["generation"])
        self.migrator._write_json_atomic(path, receipt)

        compact = self.migrator.load_receipts("ws")[0]
        self.assertEqual(2, compact["version"])
        self.assertNotIn("manifest", compact)
        self.assertEqual({"source changed": 1}, compact["retained_reasons"])

        restored_active = dict(receipt, cleanup_state="published")
        self.migrator._write_json_atomic(path, restored_active)
        loaded = self.migrator.load_receipts("ws")

        self.assertEqual("published", loaded[0]["cleanup_state"])
        self.assertEqual("published", json.loads(path.read_text())["cleanup_state"])

    def test_compaction_waits_for_restore_flock_before_opening_receipt(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        active = self.migrator.stage("ws")
        path = self.migrator._receipt_path("ws", active["generation"])
        self.migrator._write_json_atomic(path, dict(active, cleanup_state="consumed"))
        lock_path = self.root / "backup.lock"
        loader_waiting = threading.Event()
        loaded: list[dict[str, object]] = []
        errors: list[BaseException] = []

        @contextmanager
        def observing_lock() -> Iterator[None]:
            loader_waiting.set()
            with backup_restore_lock(lock_path):
                yield

        def load() -> None:
            try:
                loaded.extend(self.migrator.load_receipts("ws"))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with mock.patch("ragtime.userspace.object_storage.legacy_migration.locked_operation", observing_lock):
            with backup_restore_lock(lock_path):
                worker = threading.Thread(target=load)
                worker.start()
                self.assertTrue(loader_waiting.wait(1))
                self.migrator._write_json_atomic(path, active)
            worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertEqual([], errors)
        self.assertEqual("published", loaded[0]["cleanup_state"])
        self.assertEqual("published", json.loads(path.read_text())["cleanup_state"])

    async def test_bound_receipt_progresses_with_orphan_through_ack_and_restart(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        receipt = self.migrator.stage("ws")
        orphan = self.migrator.receipt_dir("ws") / ("f" * 32)
        orphan.mkdir()
        (orphan / "partial").write_bytes(b"partial")
        pending = {
            "state": "copying",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
        }
        completed = {
            "state": "completed",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
            "verified_files": ["buckets/assets/a.txt"],
            "gc_completed": False,
        }
        acknowledged = {**completed, "gc_completed": True}
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import",
                new=mock.AsyncMock(side_effect=[pending, completed, acknowledged]),
            ),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.acknowledge_legacy_gc", new=mock.AsyncMock()),
        ):
            self.assertFalse(await self.migrator.reconcile("ws"))
            self.assertTrue(orphan.exists())
            self.assertTrue(await self.migrator.reconcile("ws"))

        self.assertFalse(orphan.exists())
        restarted = LegacyObjectStorageMigrator(self.migrator.storage_root, lambda _: self.workspace)
        self.assertFalse(restarted.has_unfinished_receipt("ws"))

    async def test_consumed_receipt_prevents_submitting_peer_published_generation(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        consumed = self.migrator.stage("ws")
        consumed["cleanup_state"] = "consumed"
        self.migrator._write_json_atomic(self.migrator._receipt_path("ws", consumed["generation"]), consumed)
        published = self.migrator.stage("ws")

        with mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()) as submit:
            self.assertTrue(await self.migrator.reconcile("ws"))

        submit.assert_not_awaited()
        self.assertTrue(self.migrator._receipt_path("ws", published["generation"]).exists())
