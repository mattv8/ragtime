from __future__ import annotations

import errno
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

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

    def test_immutable_pack_staging_hardlinks_with_exdev_copy_fallback(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "restic"
            pack = source / "data" / "pack"
            pack.parent.mkdir(parents=True)
            pack.write_bytes(b"immutable")
            linked = root / "linked"
            RuntimeHistoryExporter._copy_regular_tree(source, linked, immutable_packs=True)
            self.assertEqual((linked / "data" / "pack").stat().st_ino, pack.stat().st_ino)
            copied = root / "copied"
            with mock.patch("runtime.worker.sqlite_history.export.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
                RuntimeHistoryExporter._copy_regular_tree(source, copied, immutable_packs=True)
            self.assertEqual((copied / "data" / "pack").read_bytes(), b"immutable")
            self.assertNotEqual((copied / "data" / "pack").stat().st_ino, pack.stat().st_ino)
