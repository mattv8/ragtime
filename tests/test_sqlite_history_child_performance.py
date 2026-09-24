from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.userspace import sqlite_history_child


class SqliteHistoryChildPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_dir = self.root / "source"
        self.destination_dir = self.root / "destination"
        self.source_dir.mkdir()
        self.destination_dir.mkdir()
        self.current = self.source_dir / "current.sqlite3"
        with sqlite3.connect(self.current) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item VALUES (1, 'current')")
        self.backup = self.destination_dir / "backup.sqlite3"
        with sqlite3.connect(self.backup) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item VALUES (1, 'backup')")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preview_captures_and_fingerprints_current_once(self) -> None:
        source_fd = os.open(self.source_dir, os.O_RDONLY)
        destination_fd = os.open(self.destination_dir, os.O_RDONLY)
        try:
            argv = [
                "sqlite_history_child",
                "--source-fd",
                str(source_fd),
                "--destination-fd",
                str(destination_fd),
                "--source-name",
                "current.sqlite3",
                "--destination-name",
                "ignored.sqlite3",
                "--preview-backup",
                "backup.sqlite3",
                "--preview-current",
                "current.sqlite3",
                "--preview-migrations",
                "migrations",
                "--preview-candidate",
                "candidate.sqlite3",
                "--mode",
                "overwrite",
                "--conflict-policy",
                "keep_current",
            ]
            original_tempdir = tempfile.tempdir
            original_tmpdir = os.environ.get("TMPDIR")
            with (
                mock.patch.dict(os.environ, {}),
                mock.patch.object(tempfile, "tempdir", original_tempdir),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(sqlite_history_child, "_install_sqlite_landlock"),
                mock.patch.object(sqlite_history_child, "capture_database", wraps=sqlite_history_child.capture_database) as capture,
                mock.patch.object(sqlite_history_child, "database_fingerprint", wraps=sqlite_history_child.database_fingerprint) as fingerprint,
                mock.patch("builtins.print") as printed,
            ):
                sqlite_history_child.main()
        finally:
            os.close(source_fd)
            os.close(destination_fd)

        payload = json.loads(printed.call_args.args[0])
        self.assertTrue(payload["can_apply"], payload["blockers"])
        self.assertEqual(tempfile.tempdir, original_tempdir)
        self.assertEqual(os.environ.get("TMPDIR"), original_tmpdir)
        with sqlite3.connect(self.destination_dir / "candidate.sqlite3") as connection:
            self.assertEqual(connection.execute("SELECT value FROM item WHERE id = 1").fetchone(), ("backup",))
        self.assertEqual(payload["current_fingerprint"], sqlite_history_child.database_fingerprint(self.current))
        self.assertEqual(capture.call_count, 1)
        self.assertFalse(capture.call_args.kwargs["include_fingerprint"])
        self.assertEqual(fingerprint.call_count, 1)

    def test_drift_fingerprint_captures_and_fingerprints_current_once(self) -> None:
        source_fd = os.open(self.source_dir, os.O_RDONLY)
        destination_fd = os.open(self.destination_dir, os.O_RDONLY)
        try:
            argv = [
                "sqlite_history_child",
                "--source-fd",
                str(source_fd),
                "--destination-fd",
                str(destination_fd),
                "--source-name",
                "current.sqlite3",
                "--destination-name",
                "ignored.sqlite3",
                "--fingerprint-current",
                "current.sqlite3",
            ]
            original_tempdir = tempfile.tempdir
            original_tmpdir = os.environ.get("TMPDIR")
            with (
                mock.patch.dict(os.environ, {}),
                mock.patch.object(tempfile, "tempdir", original_tempdir),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(sqlite_history_child, "_install_sqlite_landlock"),
                mock.patch.object(sqlite_history_child, "capture_database", wraps=sqlite_history_child.capture_database) as capture,
                mock.patch.object(sqlite_history_child, "database_fingerprint", wraps=sqlite_history_child.database_fingerprint) as fingerprint,
                mock.patch("builtins.print") as printed,
            ):
                sqlite_history_child.main()
        finally:
            os.close(source_fd)
            os.close(destination_fd)

        payload = json.loads(printed.call_args.args[0])
        self.assertEqual(tempfile.tempdir, original_tempdir)
        self.assertEqual(os.environ.get("TMPDIR"), original_tmpdir)
        self.assertEqual(payload["current_fingerprint"], sqlite_history_child.database_fingerprint(self.current))
        self.assertEqual(capture.call_count, 1)
        self.assertFalse(capture.call_args.kwargs["include_fingerprint"])
        self.assertEqual(fingerprint.call_count, 1)
