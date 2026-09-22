import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_history import SqliteHistoryService


class SqliteHistoryCatalogPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        self.files.mkdir(parents=True)
        self.root = self.files.parent / "sqlite_backups"
        self.service = SqliteHistoryService(lambda _: self.files)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _row(row_id: str, blob: str, created_at: str, *, database: str = "app.sqlite3") -> dict[str, object]:
        return {
            "id": row_id,
            "database_name": database,
            "created_at": created_at,
            "status": "ready",
            "trigger": "manual",
            "blob": blob,
            "size_bytes": 10,
            "sha256": hashlib.sha256(b"0123456789").hexdigest(),
            "source_token": "state",
        }

    def test_quota_does_not_credit_blob_still_referenced_by_protected_alias(self) -> None:
        blob_dir = self.root / "blobs"
        blob_dir.mkdir(parents=True)
        (blob_dir / "shared.sqlite3").write_bytes(b"0123456789")
        manifest: dict[str, Any] = {
            "version": 1,
            "workspace_id": "workspace",
            "previews": {},
            "operations": {},
            "backups": [
                self._row("candidate", "blobs/shared.sqlite3", "2026-01-01T00:00:00+00:00", database="old.sqlite3"),
                self._row("protected", "blobs/shared.sqlite3", "2026-01-02T00:00:00+00:00", database="live.sqlite3"),
            ],
        }

        with mock.patch("ragtime.userspace.sqlite_history._MAX_WORKSPACE_BYTES", 10):
            with self.assertRaises(HTTPException):
                self.service._enforce_quota(self.root, manifest, 1, protected_ids={"protected"})

        self.assertEqual(["candidate", "protected"], [row["id"] for row in manifest["backups"]])

    def test_reusable_aliases_hash_once_per_expected_artifact(self) -> None:
        blob_dir = self.root / "blobs"
        blob_dir.mkdir(parents=True)
        blob = blob_dir / "shared.sqlite3"
        blob.write_bytes(b"corrupted!")
        manifest = {
            "backups": [
                self._row("newest", "blobs/shared.sqlite3", "2026-01-02T00:00:00+00:00"),
                self._row("older", "blobs/shared.sqlite3", "2026-01-01T00:00:00+00:00"),
            ]
        }

        with mock.patch(
            "ragtime.userspace.sqlite_history._sha256",
            side_effect=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        ) as sha256:
            reusable = self.service._latest_reusable_blob(self.root, manifest, "app.sqlite3", "state")

        self.assertIsNone(reusable)
        self.assertEqual(1, sha256.call_count)
