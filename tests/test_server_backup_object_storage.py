import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.core import server_backup


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class ServerBackupObjectStorageTests(unittest.TestCase):
    def test_backup_skips_only_unpublished_legacy_staging_generation(self) -> None:
        self.assertTrue(server_backup._should_skip_data_path(Path("_userspace/_object_storage/_legacy_imports/ws/tmp-0123/buckets/a")))
        self.assertFalse(server_backup._should_skip_data_path(Path("_userspace/_object_storage/_legacy_imports/ws/0123/manifest.json")))

    def test_database_restore_with_managed_key_requires_offline_confirmation_when_destination_storage_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "backup.tar.gz"
            staging = root / "staging"
            (staging / "data").mkdir(parents=True)
            (staging / "data" / ".encryption_key").write_text("restored-key")
            (staging / "database.dump").write_bytes(b"database")
            (staging / "backup-meta.json").write_text(
                json.dumps(
                    {
                        "format": "tar.gz",
                        "version": 1,
                        "created_at": "",
                        "scope": "database",
                        "ragtime_version": "test",
                        "schema_version": "test",
                        "encrypted": True,
                        "includes_managed_key": True,
                    }
                )
            )
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(staging / "data", arcname="data")
                handle.add(staging / "database.dump", arcname="database.dump")
                handle.add(staging / "backup-meta.json", arcname="backup-meta.json")
            data_dir = root / "live-data"
            storage_root = data_dir / "_userspace" / "_object_storage"
            storage_root.mkdir(parents=True)

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.dict(
                    os.environ,
                    {"OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED": "", "STORAGE_ROOT": str(storage_root)},
                    clear=False,
                ),
            ):
                with self.assertRaisesRegex(server_backup.BackupValidationError, "gateway to be stopped"):
                    server_backup.restore_backup(server_backup.RestoreOptions(archive_path=archive, restore_confirmation="RESTORE ragtime"))

    def test_files_restore_requires_offline_confirmation_when_destination_storage_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "backup.tar.gz"
            staging = root / "staging"
            (staging / "data" / "indexes").mkdir(parents=True)
            (staging / "data" / "indexes" / "index.txt").write_text("data")
            (staging / "backup-meta.json").write_text(
                json.dumps(
                    {
                        "format": "tar.gz",
                        "version": 1,
                        "created_at": "",
                        "scope": "files",
                        "ragtime_version": "test",
                        "schema_version": "test",
                        "encrypted": False,
                        "includes_managed_key": False,
                    }
                )
            )
            with tarfile.open(archive, "w:gz") as handle:
                handle.add(staging / "data", arcname="data")
                handle.add(staging / "backup-meta.json", arcname="backup-meta.json")
            data_dir = root / "live-data"
            storage_root = data_dir / "_userspace" / "_object_storage"
            storage_root.mkdir(parents=True)

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.dict(
                    os.environ,
                    {"OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED": "", "STORAGE_ROOT": str(storage_root)},
                    clear=False,
                ),
            ):
                with self.assertRaisesRegex(server_backup.BackupValidationError, "gateway to be stopped"):
                    server_backup.restore_backup(server_backup.RestoreOptions(archive_path=archive, restore_confirmation="RESTORE ragtime"))

    def test_initialized_storage_is_quiesced_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            storage_root = data_dir / "_userspace" / "_object_storage"
            storage_root.mkdir(parents=True)
            (storage_root / "registry.db").write_bytes(b"state")
            output = Path(tmpdir) / "backup.tar.gz"
            responses = [_Response({"consistent": True, "lease_id": "lease-1"}), _Response({"success": True})]

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=lambda path: path.write_bytes(b"db")),
                mock.patch.object(server_backup, "urlopen", side_effect=responses) as urlopen,
            ):
                server_backup.create_backup(server_backup.BackupOptions(scope=server_backup.BackupScope.FULL, output_path=output))

            self.assertEqual(urlopen.call_count, 2)
            self.assertTrue(urlopen.call_args_list[0].args[0].full_url.endswith("/v1/backup/prepare"))
            self.assertTrue(urlopen.call_args_list[1].args[0].full_url.endswith("/v1/backup/release"))

    def test_initialized_storage_fails_closed_when_prepare_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            storage_root = data_dir / "_userspace" / "_object_storage"
            storage_root.mkdir(parents=True)
            (storage_root / "registry.db").write_bytes(b"state")

            with mock.patch.object(server_backup, "DATA_DIR", data_dir), mock.patch.object(server_backup, "urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(server_backup.BackupError, "could not be quiesced"):
                    server_backup.create_backup(server_backup.BackupOptions(scope=server_backup.BackupScope.FILES))
