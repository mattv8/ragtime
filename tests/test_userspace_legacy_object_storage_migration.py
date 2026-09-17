import asyncio
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.object_storage.legacy_migration import LegacyMigrationError, LegacyObjectStorageMigrator


class LegacyObjectStorageMigrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.workspace = self.root / "workspaces" / "ws"
        self.source = self.workspace / "s3" / "buckets" / "assets"
        self.source.mkdir(parents=True)
        self.storage = self.root / "_object_storage"
        self.migrator = LegacyObjectStorageMigrator(self.storage, lambda _: self.workspace)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_stage_publishes_sorted_atomic_manifest_and_receipt(self) -> None:
        (self.source / "z.txt").write_bytes(b"z")
        (self.source / "a.txt").write_bytes(b"abc")
        receipt = self.migrator.stage("ws")
        generation = receipt["generation"]
        manifest_path = self.storage / "_legacy_imports" / "ws" / generation / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(["buckets/assets/a.txt", "buckets/assets/z.txt"], [entry["path"] for entry in manifest["files"]])
        self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), receipt["manifest_sha256"])
        self.assertTrue((self.storage / "_legacy_imports" / "ws" / f"receipt-{generation}.json").is_file())

    def test_missing_source_is_not_an_empty_import(self) -> None:
        self.source.rmdir()
        (self.workspace / "s3" / "buckets").rmdir()
        with self.assertRaises(LegacyMigrationError):
            self.migrator.stage("ws")

    def test_source_symlink_is_rejected(self) -> None:
        target = self.root / "outside"
        target.write_bytes(b"outside")
        (self.source / "escape").symlink_to(target)
        with self.assertRaises(LegacyMigrationError):
            self.migrator.stage("ws")

    def test_symlinked_workspace_s3_or_storage_parent_is_rejected(self) -> None:
        self.source.rmdir()
        (self.workspace / "s3" / "buckets").rmdir()
        (self.workspace / "s3").rmdir()
        external = self.root / "external"
        (external / "buckets" / "assets").mkdir(parents=True)
        (self.workspace / "s3").symlink_to(external)
        with self.assertRaises(LegacyMigrationError):
            self.migrator.stage("ws")

        # A separate source proves storage ancestry is also traversed no-follow.
        self.tempdir.cleanup()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.workspace = self.root / "workspaces" / "ws"
        self.source = self.workspace / "s3" / "buckets" / "assets"
        self.source.mkdir(parents=True)
        self.source.joinpath("a").write_bytes(b"a")
        target = self.root / "real-storage"
        target.mkdir()
        storage_link = self.root / "storage-link"
        storage_link.symlink_to(target, target_is_directory=True)
        self.migrator = LegacyObjectStorageMigrator(storage_link, lambda _: self.workspace)
        with self.assertRaises(LegacyMigrationError):
            self.migrator.stage("ws")

    async def test_completed_job_gc_removes_only_verified_unchanged_file(self) -> None:
        verified = self.source / "verified.txt"
        retained = self.source / "unfinished.part"
        verified.write_bytes(b"verified")
        retained.write_bytes(b"retain")
        receipt = self.migrator.stage("ws")
        complete = {
            "state": "completed",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
            "verified_files": ["buckets/assets/verified.txt"],
        }
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import", new=mock.AsyncMock(return_value=complete)),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.acknowledge_legacy_gc", new=mock.AsyncMock()),
        ):
            self.assertTrue(await self.migrator.reconcile("ws"))
        self.assertFalse(verified.exists())
        self.assertTrue(retained.exists())

    async def test_changed_verified_file_is_retained(self) -> None:
        file = self.source / "a.txt"
        file.write_bytes(b"before")
        receipt = self.migrator.stage("ws")
        file.write_bytes(b"after")
        complete = {
            "state": "completed",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
            "verified_files": ["buckets/assets/a.txt"],
        }
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import", new=mock.AsyncMock(return_value=complete)),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.acknowledge_legacy_gc", new=mock.AsyncMock()),
        ):
            await self.migrator.reconcile("ws")
        self.assertTrue(file.exists())

    async def test_active_runtime_defers_gc(self) -> None:
        file = self.source / "a.txt"
        file.write_bytes(b"a")
        receipt = self.migrator.stage("ws")
        complete = {
            "state": "completed",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
            "verified_files": ["buckets/assets/a.txt"],
        }
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import", new=mock.AsyncMock(return_value=complete)),
        ):
            self.assertTrue(await self.migrator.reconcile("ws", runtime_active=True))
        self.assertTrue(file.exists())

    async def test_consumed_receipt_is_final_and_never_restaged(self) -> None:
        file = self.source / "a.txt"
        file.write_bytes(b"a")
        receipt = self.migrator.stage("ws")
        receipt["cleanup_state"] = "consumed"
        self.migrator._write_json_atomic(self.migrator._receipt_path("ws", receipt["generation"]), receipt)
        with mock.patch.object(self.migrator, "stage") as stage:
            self.assertTrue(await self.migrator.reconcile("ws"))
        stage.assert_not_called()

    async def test_forged_receipt_is_rejected_before_restage(self) -> None:
        generation = "a" * 32
        forged = {
            "version": 1,
            "workspace_id": "ws",
            "generation": generation,
            "manifest": {"version": 1, "workspace_id": "ws", "generation": generation, "files": [{"path": "buckets/../escape", "size": 1, "sha256": "0" * 64}]},
            "manifest_sha256": "0" * 64,
            "source_files": {},
            "source_root": {"device": 1, "inode": 1},
            "cleanup_state": "published",
        }
        self.migrator._write_json_atomic(self.migrator._receipt_path("ws", generation), forged)
        with mock.patch.object(self.migrator, "stage") as stage, self.assertRaises(LegacyMigrationError):
            await self.migrator.reconcile("ws")
        stage.assert_not_called()

    async def test_replaced_verified_entry_is_retained_and_receipt_records_reason(self) -> None:
        file = self.source / "a.txt"
        file.write_bytes(b"original")
        receipt = self.migrator.stage("ws")
        file.unlink()
        file.write_bytes(b"original")
        complete = {
            "state": "completed",
            "generation": receipt["generation"],
            "manifest_sha256": receipt["manifest_sha256"],
            "verified_files": ["buckets/assets/a.txt"],
        }
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import", new=mock.AsyncMock(return_value=complete)),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.acknowledge_legacy_gc", new=mock.AsyncMock()),
        ):
            await self.migrator.reconcile("ws")
        self.assertTrue(file.exists())
        stored = self.migrator._load_receipts("ws")[0]
        self.assertEqual("source changed", stored["retained_files"]["buckets/assets/a.txt"])

    async def test_cancelled_filesystem_waits_for_worker_before_returning(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocked() -> str:
            started.set()
            release.wait(2)
            return "done"

        task = asyncio.create_task(self.migrator._filesystem(blocked))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_interrupted_tmp_generation_is_reclaimed_without_touching_unknown_tmp(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        receipt = self.migrator.stage("ws")
        receipt["cleanup_state"] = "consumed"
        self.migrator._write_json_atomic(self.migrator._receipt_path("ws", receipt["generation"]), receipt)
        root = self.migrator.receipt_dir("ws")
        interrupted = root / f"tmp-{'b' * 32}"
        interrupted.mkdir()
        (interrupted / "partial").write_bytes(b"partial")
        unknown = root / "tmp-keep-me"
        unknown.mkdir()

        self.assertTrue(await self.migrator.reconcile("ws"))
        self.assertFalse(interrupted.exists())
        self.assertTrue(unknown.exists())

    async def test_unreceipted_published_generation_is_reclaimed_only_after_absent_job(self) -> None:
        (self.source / "a.txt").write_bytes(b"a")
        root = self.migrator.receipt_dir("ws")
        orphan = root / ("c" * 32)
        orphan.mkdir(parents=True)
        (orphan / "partial").write_bytes(b"partial")
        with (
            mock.patch(
                "ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import",
                new=mock.AsyncMock(side_effect=[HTTPException(status_code=404), {}]),
            ),
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.submit_legacy_import", new=mock.AsyncMock()),
        ):
            self.assertFalse(await self.migrator.reconcile("ws"))
        self.assertFalse(orphan.exists())
        self.assertEqual(1, len(self.migrator._load_receipts("ws")))

    async def test_bound_unreceipted_generation_is_preserved_and_fails_closed(self) -> None:
        root = self.migrator.receipt_dir("ws")
        generation = "e" * 32
        orphan = root / generation
        orphan.mkdir(parents=True)
        (orphan / "manifest.json").write_bytes(b"incomplete")
        status = {"state": "copying", "generation": generation, "manifest_sha256": "0" * 64}
        with (
            mock.patch("ragtime.userspace.object_storage.legacy_migration.control.get_legacy_import", new=mock.AsyncMock(return_value=status)),
            mock.patch.object(self.migrator, "stage") as stage,
        ):
            self.assertFalse(await self.migrator.reconcile("ws"))
        self.assertTrue(orphan.exists())
        stage.assert_not_called()

    def test_noncanonical_cleanup_path_is_rejected(self) -> None:
        from ragtime.userspace.object_storage.legacy_migration import _safe_relative

        for value in ("buckets//a", "buckets/./a", "buckets/a/"):
            with self.assertRaises(LegacyMigrationError):
                _safe_relative(value)
