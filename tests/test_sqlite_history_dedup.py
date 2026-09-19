import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_history import SqliteHistoryService


class SqliteHistoryDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        (self.files / ".ragtime" / "db").mkdir(parents=True)
        self.service = SqliteHistoryService(lambda _: self.files)
        self.root = self.files.parent / "sqlite_backups"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _capture(self, destination: Path, *, token: str = "state") -> dict[str, object]:
        destination.write_bytes(b"sqlite backup")
        return {
            "size_bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_token": token,
        }

    def test_manual_unchanged_creates_logical_alias_but_scheduled_skips(self) -> None:
        (self.files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"source")
        with (
            mock.patch.object(self.service, "_probe_confined", return_value="state"),
            mock.patch.object(
                self.service, "_capture_confined", side_effect=lambda _files, _name, blob_dir, destination: self._capture(blob_dir / destination)
            ) as capture,
        ):
            first = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "manual", "one", None)
            second = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "snapshot", "two", None)
            scheduled = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "scheduled", None, None)
        self.assertEqual(first["blob"], second["blob"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual("skipped_unchanged", scheduled["outcome"])
        self.assertEqual(1, capture.call_count)

    def test_deleting_alias_keeps_shared_blob_until_last_reference(self) -> None:
        (self.files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"source")
        with (
            mock.patch.object(self.service, "_probe_confined", return_value="state"),
            mock.patch.object(
                self.service, "_capture_confined", side_effect=lambda _files, _name, blob_dir, destination: self._capture(blob_dir / destination)
            ),
        ):
            first = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "manual", None, None)
            second = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "snapshot", "two", None)
        blob = self.root / str(first["blob"])
        manifest = self.service._load(self.root, "workspace")
        manifest["backups"] = [row for row in manifest["backups"] if row["id"] != first["id"]]
        self.service._save(self.root, manifest)
        self.assertTrue(blob.exists())
        manifest["backups"] = [row for row in manifest["backups"] if row["id"] != second["id"]]
        self.service._save(self.root, manifest)
        self.service._unlink_unreferenced(self.root, manifest, {str(second["blob"])})
        self.assertFalse(blob.exists())

    def test_real_confined_capture_returns_reusable_source_token(self) -> None:
        database = self.files / ".ragtime" / "db" / "app.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE item (value TEXT)")
            connection.execute("INSERT INTO item VALUES ('saved')")
        blob_dir = self.root / "blobs"
        blob_dir.mkdir(parents=True)
        result = self.service._capture_confined(self.files, "app.sqlite3", blob_dir, "captured.sqlite3")
        self.assertTrue((blob_dir / "captured.sqlite3").is_file())
        self.assertIsInstance(result.get("source_token"), str)

    def test_real_wal_mutation_creates_a_new_scheduled_blob(self) -> None:
        database = self.files / ".ragtime" / "db" / "app.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE item (value TEXT)")
            connection.execute("INSERT INTO item VALUES ('first')")
            connection.commit()
            first = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "manual", None, None)
            connection.execute("INSERT INTO item VALUES ('wal change')")
            connection.commit()
            second = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "scheduled", None, None)
        self.assertEqual("ready", second["status"])
        self.assertNotEqual(first["blob"], second["blob"])

    def test_corrupt_cached_blob_cannot_skip_and_is_recaptured(self) -> None:
        (self.files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"source")
        with (
            mock.patch.object(self.service, "_probe_confined", return_value="state"),
            mock.patch.object(
                self.service, "_capture_confined", side_effect=lambda _files, _name, blob_dir, destination: self._capture(blob_dir / destination)
            ) as capture,
        ):
            first = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "manual", None, None)
            (self.root / str(first["blob"])).write_bytes(b"corrupt")
            scheduled = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "scheduled", None, None)
        self.assertEqual("ready", scheduled["status"])
        self.assertEqual(2, capture.call_count)

    def test_pre_restore_never_reuses_source_token_before_full_capture(self) -> None:
        (self.files / ".ragtime" / "db" / "app.sqlite3").write_bytes(b"source")
        with (
            mock.patch.object(self.service, "_probe_confined", return_value="state"),
            mock.patch.object(
                self.service, "_capture_confined", side_effect=lambda _files, _name, blob_dir, destination: self._capture(blob_dir / destination)
            ) as capture,
        ):
            self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "manual", None, None)
            safety = self.service._capture_one("workspace", self.root, self.files, "app.sqlite3", "pre_restore", None, None)
        self.assertEqual("pre_restore", safety["trigger"])
        self.assertEqual(2, capture.call_count)

    def test_impossible_quota_reservation_leaves_manifest_and_blob_intact(self) -> None:
        blob_dir = self.root / "blobs"
        blob_dir.mkdir(parents=True)
        blob = blob_dir / "only.sqlite3"
        blob.write_bytes(b"too large")
        manifest = {
            "version": 1,
            "workspace_id": "workspace",
            "previews": {},
            "operations": {},
            "last_scheduled_at": None,
            "backups": [
                {
                    "id": "only",
                    "database_name": "app.sqlite3",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "status": "ready",
                    "trigger": "manual",
                    "blob": "blobs/only.sqlite3",
                }
            ],
        }
        with mock.patch("ragtime.userspace.sqlite_history._MAX_WORKSPACE_BYTES", 1):
            with self.assertRaises(HTTPException):
                self.service._enforce_quota(self.root, manifest, 1)
        backups = cast(list[dict[str, object]], manifest["backups"])
        self.assertEqual(["only"], [row["id"] for row in backups])
        self.assertTrue(blob.exists())
