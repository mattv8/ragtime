"""Real-Landlock regressions for SQLite temporary-file spills."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux Landlock")
class RuntimeSqliteHistorySpillTests(unittest.TestCase):
    ROW_COUNT = 8_192
    BLOB_SIZE = 1_024

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        self.source.mkdir()
        (self.source / "migrations").mkdir()
        (self.destination / "blobs").mkdir(parents=True)
        (self.destination / "candidates").mkdir()
        self.current = self.source / "app.sqlite3"
        with sqlite3.connect(self.current) as connection:
            connection.execute("CREATE TABLE payloads (id INTEGER PRIMARY KEY, body BLOB NOT NULL)")
            connection.executemany(
                "INSERT INTO payloads (id, body) VALUES (?, randomblob(?))",
                ((row_id, self.BLOB_SIZE) for row_id in range(self.ROW_COUNT)),
            )
        shutil.copyfile(self.current, self.destination / "blobs" / "backup.sqlite3")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run_child(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        source_fd = os.open(self.source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        destination_fd = os.open(self.destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            return subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "runtime.worker.sqlite_history.capture_child",
                    "--source-fd",
                    str(source_fd),
                    "--destination-fd",
                    str(destination_fd),
                    "--source-name",
                    "app.sqlite3",
                    "--destination-name",
                    "candidates/candidate.sqlite3",
                    *arguments,
                ],
                pass_fds=(source_fd, destination_fd),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        finally:
            os.close(destination_fd)
            os.close(source_fd)

    def test_preview_large_blob_database_keeps_sqlite_spill_inside_landlock(self) -> None:
        completed = self._run_child(
            "--preview-backup",
            "blobs/backup.sqlite3",
            "--preview-current",
            "app.sqlite3",
            "--preview-candidate",
            "candidates/candidate.sqlite3",
            "--mode",
            "overwrite",
            "--conflict-policy",
            "keep_current",
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["can_apply"], result)
        self.assertTrue((self.destination / "candidates" / "candidate.sqlite3").is_file())

    def test_drift_large_blob_database_keeps_sqlite_spill_inside_landlock(self) -> None:
        completed = self._run_child(
            "--fingerprint-current",
            "app.sqlite3",
            "--fingerprint-migrations",
            "migrations",
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsInstance(result["current_fingerprint"], str)
        self.assertEqual(64, len(result["current_fingerprint"]))

    def test_private_sqlite_temp_directory_restores_process_state(self) -> None:
        from runtime.worker.sqlite_history import capture_child

        previous_tmpdir = os.environ.get("TMPDIR")
        previous_sqlite_tmpdir = os.environ.get("SQLITE_TMPDIR")
        previous_tempdir = tempfile.tempdir
        previous_cwd = os.stat(".")
        destination_fd = os.open(self.destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        previous_cwd_fd = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
        try:
            with capture_child._confined_sqlite_temp(destination_fd, previous_cwd_fd) as scratch:
                scratch_name = scratch.name
                self.assertEqual(os.fspath(scratch), os.environ["TMPDIR"])
                self.assertEqual(os.fspath(scratch), os.environ["SQLITE_TMPDIR"])
                self.assertEqual(os.fspath(scratch), tempfile.tempdir)
                self.assertTrue(scratch.is_dir())
        finally:
            os.close(previous_cwd_fd)
            os.close(destination_fd)
        self.assertEqual(previous_tmpdir, os.environ.get("TMPDIR"))
        self.assertEqual(previous_sqlite_tmpdir, os.environ.get("SQLITE_TMPDIR"))
        self.assertEqual(previous_tempdir, tempfile.tempdir)
        self.assertEqual((previous_cwd.st_dev, previous_cwd.st_ino), (os.stat(".").st_dev, os.stat(".").st_ino))
        self.assertFalse((self.destination / scratch_name).exists())
