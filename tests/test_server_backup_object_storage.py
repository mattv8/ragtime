import json
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
