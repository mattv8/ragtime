"""Crash-safe collection of confined SQLite history scratch directories."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime.core.sqlite_history_scratch import SCRATCH_PREFIX, _owner_lock, scratch_owner_lock
from runtime.worker.sqlite_history.export import RuntimeHistoryExporter
from runtime.worker.sqlite_history.service import SqliteHistoryService


class RuntimeHistoryScratchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "sqlite_backups"
        self.root.mkdir()
        (self.root / "blobs").mkdir()
        self.service = SqliteHistoryService(lambda _workspace_id: self.root.parent / "files")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _scratch(self, parent: Path, name: str) -> Path:
        scratch = parent / f"{SCRATCH_PREFIX}{name}"
        scratch.mkdir()
        return scratch

    def test_live_child_scratch_survives_then_dead_child_scratch_is_removed(self) -> None:
        scratch = self._scratch(self.root, "live")
        owner = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; from runtime.core.sqlite_history_scratch import scratch_owner_lock; "
                "import sys, time; p=Path(sys.argv[1]); "
                "\nwith scratch_owner_lock(p):\n print('locked', flush=True); time.sleep(30)",
                os.fspath(scratch),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert owner.stdout is not None
        self.assertEqual("locked\n", owner.stdout.readline())
        self.service._cleanup_and_due_sync(self.root, "workspace")
        self.assertTrue(scratch.exists())
        owner.kill()
        owner.wait(timeout=10)
        owner.stdout.close()
        self.service._cleanup_and_due_sync(self.root, "workspace")
        self.assertFalse(scratch.exists())

    def test_unmarked_and_symlinked_scratch_are_never_followed(self) -> None:
        legacy = self._scratch(self.root, "legacy")
        external = Path(self.temporary.name) / "external"
        external.mkdir()
        (external / "keep").write_text("keep")
        managed = self._scratch(self.root / "blobs", "managed")
        with scratch_owner_lock(managed):
            (managed / "payload").write_bytes(b"1234")
        (managed / "outside").symlink_to(external, target_is_directory=True)
        self.service._cleanup_and_due_sync(self.root, "workspace")
        self.assertTrue(legacy.exists())
        self.assertTrue((external / "keep").exists())
        self.assertFalse(managed.exists())

    def test_quota_usage_includes_managed_root_and_blob_scratch_once(self) -> None:
        root_scratch = self._scratch(self.root, "root")
        blob_scratch = self._scratch(self.root / "blobs", "blob")
        with scratch_owner_lock(root_scratch):
            (root_scratch / "one").write_bytes(b"123")
        with scratch_owner_lock(blob_scratch):
            nested = blob_scratch / "nested"
            nested.mkdir()
            (nested / "two").write_bytes(b"12345")
            (blob_scratch / "external").symlink_to(self.temporary.name, target_is_directory=True)
        self.assertEqual(8, self.service._history_disk_usage(self.root))

    def test_contended_owner_probe_closes_its_file_descriptor(self) -> None:
        scratch = self._scratch(self.root, "contended")
        with scratch_owner_lock(scratch):
            with (
                mock.patch("runtime.core.sqlite_history_scratch.fcntl.flock", side_effect=BlockingIOError),
                mock.patch("runtime.core.sqlite_history_scratch.os.close", wraps=os.close) as close,
            ):
                self.assertIsNone(_owner_lock(scratch, nonblocking=True))
                close.assert_called_once()

    def test_quota_usage_counts_candidates_and_downloads_without_backup_double_charge(self) -> None:
        for name in ("candidates", "downloads"):
            directory = self.root / name
            directory.mkdir()
            (directory / "copy.sqlite3").write_bytes(b"12345")
        (self.root / "blobs" / "retained.sqlite3").write_bytes(b"already-logically-charged")
        self.assertEqual(10, self.service._history_disk_usage(self.root))

    def test_export_tree_excludes_marker_owned_scratch(self) -> None:
        source = Path(self.temporary.name) / "source"
        destination = Path(self.temporary.name) / "destination"
        source.mkdir()
        (source / "catalog").write_text("catalog")
        scratch = self._scratch(source, "export")
        with scratch_owner_lock(scratch):
            (scratch / "private").write_text("private")
        RuntimeHistoryExporter._copy_regular_tree(source, destination)
        self.assertTrue((destination / "catalog").is_file())
        self.assertFalse((destination / scratch.name).exists())
