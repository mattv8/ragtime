from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from ragtime.userspace.sqlite_history import SqliteHistoryService
from runtime.core.sqlite_capture_state import pinned_source_state, source_state_token
from runtime.core.sqlite_recovery import _connect_readonly, capture_database


class SqliteCaptureStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database = self.root / "app.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item VALUES (1, 'original')")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _directory_fd(self) -> int:
        return os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)

    def test_token_changes_for_wal_only_commit(self) -> None:
        """A missing WAL component in the token would miss committed WAL-only data."""
        directory_fd = self._directory_fd()
        try:
            with sqlite3.connect(self.database) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.commit()
                before = source_state_token(directory_fd, self.database.name)
                main_size = self.database.stat().st_size
                connection.execute("INSERT INTO item VALUES (2, 'wal-only')")
                connection.commit()
                after = source_state_token(directory_fd, self.database.name)
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            self.assertEqual(self.database.stat().st_size, main_size)
            self.assertNotEqual(before, after)
        finally:
            os.close(directory_fd)

    def test_pinned_state_rejects_main_replacement_after_initial_token(self) -> None:
        """A capture redirected through a replacement leaf must not be reusable."""
        directory_fd = self._directory_fd()
        try:
            with pinned_source_state(directory_fd, self.database.name) as state:
                self.assertIsNotNone(state.token())
                replacement = self.root / "replacement.sqlite3"
                with sqlite3.connect(replacement) as connection:
                    connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
                    connection.execute("INSERT INTO item VALUES (1, 'replacement')")
                os.replace(replacement, self.database)
                self.assertIsNone(state.token())
        finally:
            os.close(directory_fd)

    def test_pinned_main_fd_captures_committed_wal_state(self) -> None:
        """The descriptor capability used by capture must retain SQLite's WAL view."""
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("INSERT INTO item VALUES (2, 'wal-state')")
            connection.commit()
            directory_fd = self._directory_fd()
            try:
                with pinned_source_state(directory_fd, self.database.name) as state:
                    before = state.token()
                    self.assertIsNotNone(before)
                    capture_database(Path(f"/proc/self/fd/{state.main_fd}"), self.root / "capture.sqlite3", include_fingerprint=False)
                    # SQLite may checkpoint while serving the read snapshot; a
                    # changed post-token is conservatively reported as unknown.
                    self.assertIn(state.token(), {before, None})
                with sqlite3.connect(self.root / "capture.sqlite3") as captured:
                    self.assertEqual(captured.execute("SELECT value FROM item WHERE id=2").fetchone()[0], "wal-state")
            finally:
                os.close(directory_fd)

    def test_warmup_refresh_allows_stable_wal_capture_without_relaxing_post_check(self) -> None:
        """SQLite's one-time readonly warmup ctime update must not prevent source reuse."""
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("INSERT INTO item VALUES (2, 'wal-state')")
            connection.commit()
            directory_fd = self._directory_fd()
            try:
                with pinned_source_state(directory_fd, self.database.name) as state:
                    source = Path(f"/proc/self/fd/{state.main_fd}")
                    warmed = _connect_readonly(source)
                    try:
                        warmed.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                        self.assertTrue(state.refresh_after_warmup())
                        before = state.token()
                        capture_database(source, self.root / "warmed-capture.sqlite3", include_fingerprint=False, source_connection=warmed)
                        self.assertEqual(state.token(), before)
                    finally:
                        warmed.close()
            finally:
                os.close(directory_fd)

    def test_nonempty_journal_and_unsafe_entries_fail_closed(self) -> None:
        """Rollback recovery or a symlink/FIFO must never be treated as unchanged."""
        directory_fd = self._directory_fd()
        try:
            journal = self.root / f"{self.database.name}-journal"
            journal.write_bytes(b"hot rollback journal")
            self.assertIsNone(source_state_token(directory_fd, self.database.name))
            journal.unlink()
            self.database.unlink()
            self.database.symlink_to("replacement.sqlite3")
            self.assertIsNone(source_state_token(directory_fd, self.database.name))
        finally:
            os.close(directory_fd)

    def test_confined_child_keeps_reusable_token_for_unchanged_live_wal(self) -> None:
        """A child reopening a warmed WAL reader must allow catalog pre-copy reuse."""
        files = self.root / "workspace" / "files"
        database_dir = files / ".ragtime" / "db"
        database_dir.mkdir(parents=True)
        database = database_dir / "app.sqlite3"
        service = SqliteHistoryService(lambda _: files)
        blobs = files.parent / "sqlite_backups" / "blobs"
        blobs.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA wal_autocheckpoint=0")
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item VALUES (1, 'original')")
            connection.commit()
            first = service._capture_confined(files, database.name, blobs, "first.sqlite3")
            first_probe = service._probe_confined(files, database.name)
            second = service._capture_confined(files, database.name, blobs, "second.sqlite3")
            second_probe = service._probe_confined(files, database.name)
            self.assertIsInstance(first.get("source_token"), str)
            self.assertEqual(first["source_token"], first_probe)
            self.assertEqual(second.get("source_token"), second_probe)
            connection.execute("UPDATE item SET value='changed' WHERE id=1")
            connection.commit()
            changed = service._capture_confined(files, database.name, blobs, "changed.sqlite3")
        self.assertNotEqual(first["source_token"], changed.get("source_token"))
        with sqlite3.connect(blobs / "changed.sqlite3") as captured:
            self.assertEqual(captured.execute("SELECT value FROM item").fetchone()[0], "changed")
