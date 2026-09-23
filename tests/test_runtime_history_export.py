from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime.worker.sqlite_history.export import RuntimeHistoryExporter


class RuntimeHistoryExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_export_materializes_only_requested_workspace(self) -> None:
        class Service:
            async def list_backups(self, workspace_id):
                return [{"id": "backup", "status": "ready", "sha256": "x", "size_bytes": 1, "storage": {"kind": "restic"}}]

            async def download_to_path(self, workspace_id, backup_id, destination):
                destination.write_bytes(b"x")
                return destination

            def _root(self, workspace_id):
                return Path("/")

        with TemporaryDirectory() as temporary:
            exported = await RuntimeHistoryExporter(Service()).stage_workspace_export("workspace", Path(temporary) / "export")
            self.assertTrue((exported / "blobs/backup.sqlite3").is_file())
            self.assertFalse((exported / "secrets").exists())
