from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ragtime.userspace.sqlite_history_child import _freeze_migrations, _require_regular
from runtime.core.secure_files import SecureFileError, publish_regular_file
from runtime.core.workspace_ops import iter_managed_sqlite_database_paths


class SqliteHistoryConfinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.files = self.root / "files"
        self.database_dir = self.files / ".ragtime" / "db"
        self.database_dir.mkdir(parents=True)
        self.victim = self.root / "victim.sqlite3"
        self.victim.write_bytes(b"do-not-read-or-write")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_database_enumeration_rejects_symlinked_ragtime_parent(self) -> None:
        (self.files / ".ragtime").rename(self.files / ".ragtime-safe")
        (self.files / ".ragtime").symlink_to(self.root)

        self.assertEqual([], iter_managed_sqlite_database_paths(self.files))
        self.assertEqual(b"do-not-read-or-write", self.victim.read_bytes())

    def test_database_enumeration_ignores_symlinked_database_leaf(self) -> None:
        (self.database_dir / "app.sqlite3").symlink_to(self.victim)

        self.assertEqual([], iter_managed_sqlite_database_paths(self.files))
        self.assertEqual(b"do-not-read-or-write", self.victim.read_bytes())

    def test_migration_freeze_rejects_symlink_and_special_entries(self) -> None:
        migrations = self.database_dir / "migrations"
        migrations.mkdir()
        (migrations / "001.sql").write_text("CREATE TABLE stable (id INTEGER);", encoding="utf-8")
        (migrations / "002.sql").symlink_to(self.victim)
        scratch = self.root / "scratch"
        scratch.mkdir()
        source_fd = os.open(self.database_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            with self.assertRaises(ValueError):
                _freeze_migrations(source_fd, scratch)
        finally:
            os.close(source_fd)
        self.assertEqual(b"do-not-read-or-write", self.victim.read_bytes())

    def test_regular_source_check_rejects_swapped_sidecar_before_open(self) -> None:
        sidecar = self.database_dir / "app.sqlite3-shm"
        sidecar.write_bytes(b"safe")
        source_fd = os.open(self.database_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            sidecar.unlink()
            sidecar.symlink_to(self.victim)
            with self.assertRaises(OSError):
                _require_regular(source_fd, "app.sqlite3-shm")
        finally:
            os.close(source_fd)
        self.assertEqual(b"do-not-read-or-write", self.victim.read_bytes())

    def test_descriptor_publication_rejects_target_parent_swap(self) -> None:
        staging = self.root / "staging"
        (staging / ".ragtime" / "db").mkdir(parents=True)
        with sqlite3.connect(staging / ".ragtime" / "db" / "app.sqlite3") as connection:
            connection.execute("CREATE TABLE item (value TEXT)")
        (self.files / ".ragtime").rename(self.files / ".ragtime-safe")
        (self.files / ".ragtime").symlink_to(self.root)

        with self.assertRaises(SecureFileError):
            publish_regular_file(staging, ".ragtime/db/app.sqlite3", self.files, ".ragtime/db/app.sqlite3")
        self.assertEqual(b"do-not-read-or-write", self.victim.read_bytes())
