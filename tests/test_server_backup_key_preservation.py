import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.core import server_backup


def _make_archive(path: Path, scope: str, files: dict[str, bytes], *, encrypted: bool = False, includes_key: bool = False) -> None:
    staging = path.parent / "staging"
    for name, content in files.items():
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (staging / "backup-meta.json").write_text(
        json.dumps(
            {
                "format": "ragbak" if encrypted else "tar.gz",
                "version": 1,
                "created_at": "",
                "scope": scope,
                "ragtime_version": "test",
                "schema_version": "test",
                "encrypted": encrypted,
                "includes_managed_key": includes_key,
            }
        ),
        encoding="utf-8",
    )
    with tarfile.open(path, "w:gz") as archive:
        for child in staging.rglob("*"):
            archive.add(child, arcname=child.relative_to(staging).as_posix())


class ServerBackupKeyPreservationTests(unittest.TestCase):
    def _restore_files(self, archive: Path, data_dir: Path, *, scope: str = "files") -> None:
        with mock.patch.object(server_backup, "DATA_DIR", data_dir):
            server_backup.restore_backup(
                server_backup.RestoreOptions(
                    archive_path=archive,
                    restore_confirmation="RESTORE ragtime",
                    replace_data=True,
                    scope_override=server_backup.BackupScope(scope),
                    skip_migrations=True,
                )
            )

    def test_keyless_replace_preserves_existing_regular_keys_for_files_and_full(self) -> None:
        for scope in ("files", "full"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                archive = root / "backup.tar.gz"
                files = {"data/restored.txt": b"restored"}
                if scope == "full":
                    files["database.dump"] = b"database"
                _make_archive(archive, scope, files)
                data_dir = root / "live"
                data_dir.mkdir()
                (data_dir / ".encryption_key").write_text("existing-key", encoding="utf-8")
                (data_dir / ".jwt_secret").write_text("existing-jwt", encoding="utf-8")
                (data_dir / "old.txt").write_text("old", encoding="utf-8")

                patches = (
                    mock.patch.object(server_backup, "_create_database_safety_dump", side_effect=lambda path: path.write_bytes(b"safety")),
                    mock.patch.object(server_backup, "_terminate_other_database_connections"),
                    mock.patch.object(server_backup, "_restore_database"),
                    mock.patch.object(server_backup, "_invalidate_restored_runtime_sessions"),
                )
                with patches[0], patches[1], patches[2], patches[3]:
                    self._restore_files(archive, data_dir, scope=scope)

                self.assertEqual((data_dir / ".encryption_key").read_text(encoding="utf-8"), "existing-key")
                self.assertEqual((data_dir / ".jwt_secret").read_text(encoding="utf-8"), "existing-jwt")
                self.assertEqual((data_dir / "restored.txt").read_text(encoding="utf-8"), "restored")
                self.assertFalse((data_dir / "old.txt").exists())

    def test_encrypted_archived_managed_key_replaces_existing_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "backup.tar.gz"
            _make_archive(
                archive,
                "files",
                {"data/restored.txt": b"restored", "data/.encryption_key": b"archived-key"},
                encrypted=True,
                includes_key=True,
            )
            data_dir = root / "live"
            data_dir.mkdir()
            (data_dir / ".encryption_key").write_text("existing-key", encoding="utf-8")

            self._restore_files(archive, data_dir)

            self.assertEqual((data_dir / ".encryption_key").read_text(encoding="utf-8"), "archived-key")

    def test_keyless_initialized_storage_restore_without_existing_key_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "backup.tar.gz"
            _make_archive(
                archive,
                "files",
                {"data/_userspace/_object_storage/registry.db": b"initialized", "data/restored.txt": b"restored"},
            )
            data_dir = root / "live"
            data_dir.mkdir()
            (data_dir / "old.txt").write_text("old", encoding="utf-8")

            with mock.patch.dict(os.environ, {"OBJECT_STORAGE_RESTORE_OFFLINE_CONFIRMED": "true"}, clear=False):
                with self.assertRaisesRegex(server_backup.BackupValidationError, "authoritative key"):
                    self._restore_files(archive, data_dir)

            self.assertEqual((data_dir / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((data_dir / "restored.txt").exists())

    def test_unsafe_preserved_key_path_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "backup.tar.gz"
            _make_archive(archive, "files", {"data/restored.txt": b"restored"})
            data_dir = root / "live"
            data_dir.mkdir()
            outside = root / "outside-key"
            outside.write_text("key", encoding="utf-8")
            (data_dir / ".encryption_key").symlink_to(outside)
            (data_dir / "old.txt").write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(server_backup.BackupValidationError, "unsafe existing key"):
                self._restore_files(archive, data_dir)

            self.assertTrue((data_dir / ".encryption_key").is_symlink())
            self.assertEqual((data_dir / "old.txt").read_text(encoding="utf-8"), "old")
