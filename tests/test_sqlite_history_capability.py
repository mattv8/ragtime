import asyncio
import errno
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import sqlite_history_confinement
from ragtime.userspace.sqlite_history import SqliteHistoryService


class SqliteHistoryCapabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.confinement_cache = mock.patch.multiple(
            sqlite_history_confinement,
            _cached_at=None,
            _cached_available=None,
            _reported_unavailable=False,
        )
        self.confinement_cache.start()
        self.temp = tempfile.TemporaryDirectory()
        self.files = Path(self.temp.name) / "workspace" / "files"
        database_dir = self.files / ".ragtime" / "db"
        database_dir.mkdir(parents=True)
        (database_dir / "one.sqlite3").write_bytes(b"one")
        (database_dir / "two.sqlite3").write_bytes(b"two")
        self.service = SqliteHistoryService(lambda _: self.files)

    def tearDown(self) -> None:
        self.temp.cleanup()
        self.confinement_cache.stop()

    def test_unavailable_preflight_records_failures_without_spawning_children(self) -> None:
        unavailable = OSError(errno.EOPNOTSUPP, "Landlock ABI 3 is required")
        with (
            mock.patch.object(sqlite_history_confinement._landlock, "trial_ruleset", side_effect=unavailable) as trial,
            mock.patch.object(self.service, "_capture_confined") as capture,
            mock.patch.object(sqlite_history_confinement.logger, "warning") as warning,
        ):
            rows = asyncio.run(self.service.capture_workspace_databases("workspace", trigger="manual", files_dir=self.files))
        self.assertEqual(["failed", "failed"], [row["status"] for row in rows])
        trial.assert_called_once()
        capture.assert_not_called()
        warning.assert_called_once()

    def test_cached_preflight_recovers_after_ttl(self) -> None:
        monotonic = mock.Mock(side_effect=[0.0, 1.0, 61.0])
        unavailable = OSError(errno.EOPNOTSUPP, "Landlock ABI 3 is required")
        with (
            mock.patch.object(sqlite_history_confinement, "monotonic", monotonic),
            mock.patch.object(sqlite_history_confinement._landlock, "trial_ruleset", side_effect=[unavailable, None]) as trial,
        ):
            self.assertFalse(sqlite_history_confinement.confinement_available())
            self.assertFalse(sqlite_history_confinement.confinement_available())
            self.assertTrue(sqlite_history_confinement.confinement_available())
        self.assertEqual(2, trial.call_count)

    def test_unsupported_capture_does_not_evict_ready_backup_and_mandatory_capture_blocks(self) -> None:
        root = self.files.parent / "sqlite_backups"
        blob_dir = root / "blobs"
        blob_dir.mkdir(parents=True)
        blob = blob_dir / "ready.sqlite3"
        blob.write_bytes(b"ready")
        manifest = self.service._load(root, "workspace")
        manifest["backups"] = [
            {
                "id": "ready",
                "database_name": "one.sqlite3",
                "created_at": "2026-01-01T00:00:00+00:00",
                "status": "ready",
                "trigger": "manual",
                "blob": "blobs/ready.sqlite3",
            }
        ]
        self.service._save(root, manifest)
        with (
            mock.patch.object(sqlite_history_confinement._landlock, "trial_ruleset", side_effect=OSError(errno.EOPNOTSUPP, "Landlock ABI 3 is required")),
            mock.patch("ragtime.userspace.sqlite_history.assert_sqlite_workspace_maintenance_held", new_callable=mock.AsyncMock),
        ):
            with self.assertRaises(HTTPException) as blocked:
                asyncio.run(self.service.capture_workspace_databases("workspace", trigger="pre_restore", mandatory=True, files_dir=self.files))
        self.assertEqual(409, blocked.exception.status_code)
        self.assertTrue(blob.exists())
        rows = self.service._load(root, "workspace")["backups"]
        self.assertEqual("ready", rows[0]["id"])
        self.assertEqual("failed", rows[1]["status"])

    def test_unsupported_preflight_blocks_preview_and_drift_before_children(self) -> None:
        root = self.files.parent / "sqlite_backups"
        with (
            mock.patch.object(sqlite_history_confinement, "confinement_available", return_value=False),
            mock.patch("ragtime.userspace.sqlite_history.run_admitted_subprocess") as child,
        ):
            for operation in (
                lambda: self.service._preview_confined(
                    self.files, root, "blobs/ready.sqlite3", "one.sqlite3", "candidates/preview.sqlite3", "replace", "abort", {}
                ),
                lambda: self.service._drift_confined(self.files, root, "one.sqlite3"),
            ):
                with self.assertRaises(HTTPException) as blocked:
                    operation()
                self.assertEqual(503, blocked.exception.status_code)
        child.assert_not_called()

    def test_positive_preflight_does_not_mask_drift_child_failure(self) -> None:
        root = self.files.parent / "sqlite_backups"
        root.mkdir()
        with (
            mock.patch.object(sqlite_history_confinement, "confinement_available", return_value=True),
            mock.patch(
                "ragtime.userspace.sqlite_history.run_admitted_subprocess",
                return_value=mock.Mock(returncode=1, stdout="", stderr="child failed"),
            ) as child,
            self.assertRaises(HTTPException) as blocked,
        ):
            self.service._drift_confined(self.files, root, "one.sqlite3")
        self.assertEqual(503, blocked.exception.status_code)
        child.assert_called_once()
