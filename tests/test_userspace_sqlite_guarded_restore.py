from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from unittest import mock

from ragtime.userspace import sqlite_inspector
from ragtime.userspace.service import UserSpaceService


class _RestoreService:
    _fsync_file = staticmethod(UserSpaceService._fsync_file)
    _fsync_directory = staticmethod(UserSpaceService._fsync_directory)

    def __init__(self, files: Path) -> None:
        self.files = files


class GuardedRestoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "files"
        self.db = self.files / ".ragtime" / "db" / "app.sqlite3"
        self.db.parent.mkdir(parents=True)
        with sqlite3.connect(self.db) as connection:
            connection.execute("CREATE TABLE items (value TEXT)")
            connection.execute("INSERT INTO items VALUES ('live')")
        self.service = _RestoreService(self.files)

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    def _history(self):
        return mock.Mock(
            runtime_history_active=mock.AsyncMock(return_value=False),
            capture_workspace_databases=mock.AsyncMock(return_value=[]),
        )

    async def test_git_failure_preserves_database_and_releases_fence(self) -> None:
        released = False

        @asynccontextmanager
        async def access(workspace_id: str, *, maintenance: bool = False):
            nonlocal released
            yield self.files
            released = True

        with (
            mock.patch("ragtime.userspace.service.sqlite_workspace_access", access),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=self._history()),
        ):
            with self.assertRaisesRegex(RuntimeError, "git failed"):
                async with UserSpaceService._guarded_code_restore(cast(UserSpaceService, self.service), "workspace"):
                    self.db.unlink()
                    raise RuntimeError("git failed")

        with sqlite3.connect(self.db) as connection:
            self.assertEqual("live", connection.execute("SELECT value FROM items").fetchone()[0])
        self.assertTrue(released)

    async def test_publication_failure_crosses_maintenance_fence(self) -> None:
        released = False

        @asynccontextmanager
        async def access(workspace_id: str, *, maintenance: bool = False):
            nonlocal released
            try:
                yield self.files
            except OSError:
                raise
            else:
                released = True

        with (
            mock.patch("ragtime.userspace.service.sqlite_workspace_access", access),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=self._history()),
            mock.patch("runtime.core.secure_files.publish_regular_file", side_effect=OSError("publish failed")),
        ):
            with self.assertRaisesRegex(OSError, "publish failed"):
                async with UserSpaceService._guarded_code_restore(cast(UserSpaceService, self.service), "workspace"):
                    pass
        self.assertFalse(released)

    async def test_real_git_checkout_cannot_restore_historical_database_blob(self) -> None:
        subprocess.run(["git", "init"], cwd=self.files, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.files, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=self.files, check=True)
        migrations = self.db.parent / "migrations"
        migrations.mkdir()
        (migrations / "001_old.sql").write_text("CREATE TABLE old_value (id INTEGER);", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.files, check=True)
        subprocess.run(["git", "commit", "-m", "old snapshot"], cwd=self.files, check=True, capture_output=True)
        old_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.files, text=True).strip()
        with sqlite3.connect(self.db) as connection:
            connection.execute("DELETE FROM items")
            connection.execute("INSERT INTO items VALUES ('current')")
        (migrations / "002_target.sql").write_text("CREATE TABLE target_value (id INTEGER);", encoding="utf-8")

        @asynccontextmanager
        async def access(workspace_id: str, *, maintenance: bool = False):
            yield self.files

        with (
            mock.patch("ragtime.userspace.service.sqlite_workspace_access", access),
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=self._history()),
        ):
            async with UserSpaceService._guarded_code_restore(cast(UserSpaceService, self.service), "workspace"):
                subprocess.run(["git", "checkout", "-f", old_commit], cwd=self.files, check=True, capture_output=True)
                subprocess.run(
                    ["git", "clean", "-fd", "--exclude=/.ragtime/db/app.sqlite3"],
                    cwd=self.files,
                    check=True,
                    capture_output=True,
                )

        with sqlite3.connect(self.db) as connection:
            self.assertEqual("current", connection.execute("SELECT value FROM items").fetchone()[0])
        self.assertTrue((migrations / "001_old.sql").exists())
        self.assertFalse((migrations / "002_target.sql").exists())


class InspectorAtomicImportTests(unittest.TestCase):
    def test_invalid_import_does_not_replace_live_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            files = Path(directory) / "files"
            summary = sqlite_inspector.initialize_database(files)
            target = files / sqlite_inspector.MANAGED_DB_DIRNAME / summary.name
            with sqlite3.connect(target) as connection:
                connection.execute("CREATE TABLE item (value TEXT)")
                connection.execute("INSERT INTO item VALUES ('live')")
            source = Path(directory) / "bad.sqlite3"
            source.write_bytes(b"not sqlite")

            with self.assertRaises(Exception):
                sqlite_inspector.import_database_file(files, summary.name, source)

            with sqlite3.connect(target) as connection:
                self.assertEqual("live", connection.execute("SELECT value FROM item").fetchone()[0])
