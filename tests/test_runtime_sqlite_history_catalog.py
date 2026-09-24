"""Runtime-private SQLite history catalog contracts."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class HistoryCatalogTests(unittest.TestCase):
    def test_legacy_and_restic_references_preserve_logical_backup_ids(self) -> None:
        from runtime.worker.sqlite_history.catalog import HistoryCatalog

        with TemporaryDirectory() as temporary:
            catalog = HistoryCatalog(Path(temporary), "workspace-a")
            catalog.add_ready(
                {"id": "legacy-id", "database_name": "app.sqlite3", "created_at": "2026-09-22T00:00:00+00:00", "status": "ready"},
                {"kind": "legacy_file", "blob": "blobs/legacy.sqlite3"},
            )
            catalog.add_ready(
                {"id": "restic-id", "database_name": "app.sqlite3", "created_at": "2026-09-22T01:00:00+00:00", "status": "ready"},
                {"kind": "restic", "repository_id": "repository", "snapshot_id": "snapshot", "path": "/database.sqlite3"},
            )

            rows = catalog.list_ready()

            self.assertEqual([row["id"] for row in rows], ["restic-id", "legacy-id"])
            self.assertEqual(rows[0]["storage"]["kind"], "restic")
            self.assertFalse((Path(temporary) / "manifest-v2.json").exists())
            self.assertTrue((Path(temporary) / "manifest-v1.json").exists())

    def test_existing_v1_manifest_keeps_legacy_fields_and_lock_name(self) -> None:
        from runtime.worker.sqlite_history.catalog import HistoryCatalog

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest-v1.json").write_text(
                '{"version":1,"workspace_id":"workspace-a","backups":[{"id":"old","status":"ready","blob":"blobs/old.sqlite3","database_name":"app.sqlite3","created_at":"2026-09-22T00:00:00+00:00"}],"previews":{},"operations":{}}',
                encoding="utf-8",
            )
            catalog = HistoryCatalog(root, "workspace-a")
            row = catalog.list_ready()[0]

            self.assertEqual({"kind": "legacy_file", "blob": "blobs/old.sqlite3"}, catalog.storage(row))
            self.assertFalse((root / "manifest-v2.json").exists())
