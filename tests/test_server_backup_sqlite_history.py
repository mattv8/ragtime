import json
import os
import tarfile
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from ragtime.core import server_backup


class ServerBackupSqliteHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = mock.patch.object(server_backup, "DATA_DIR", Path(temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_portable_export_excludes_live_history_and_marks_key_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            history_root = data / "_userspace" / "_sqlite_history"
            (history_root / "restic").mkdir(parents=True)
            (history_root / "restic" / "secret-pack").write_bytes(b"live")
            for directory_name in ("secrets", "cache", "scratch", "transfers"):
                directory_path = history_root / directory_name
                directory_path.mkdir()
                (directory_path / "repository-password").write_bytes(b"must-not-export")
            workspace_history = data / "_userspace" / "workspaces" / "workspace" / "sqlite_backups"
            workspace_history.mkdir(parents=True)
            (workspace_history / "catalog.json").write_bytes(b"live")
            (data / "keep.txt").write_text("keep", encoding="utf-8")
            output = root / "backup.tar.gz"

            def export(destination: Path, *, include_repository_key: bool):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"immutable-runtime-export")
                return {
                    "version": 1,
                    "repository_id": "a" * 64,
                    "includes_repository_key": include_repository_key,
                    "requires_original_repository_key": not include_repository_key,
                    "bundle": "runtime-sqlite-history/export.bundle",
                }

            with (
                mock.patch.object(server_backup, "DATA_DIR", data),
                mock.patch.object(server_backup, "_runtime_history_status", return_value={"active": True}),
                mock.patch.object(server_backup, "_runtime_history_export", side_effect=export),
            ):
                manifest = server_backup.create_backup(server_backup.BackupOptions(scope=server_backup.BackupScope.FILES, output_path=output))

            assert manifest.sqlite_history is not None
            self.assertEqual(manifest.sqlite_history["repository_id"], "a" * 64)
            self.assertTrue(manifest.sqlite_history["requires_original_repository_key"])
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("runtime-sqlite-history/export.bundle", names)
            self.assertIn("data/keep.txt", names)
            self.assertFalse(any(name.startswith("data/_userspace/_sqlite_history/") for name in names))
            self.assertNotIn("data/_userspace/workspaces/workspace/sqlite_backups/catalog.json", names)

            encrypted = root / "backup.ragbak"
            read_fd, write_fd = os.pipe()
            os.write(write_fd, b"backup-password")
            os.close(write_fd)
            with (
                mock.patch.object(server_backup, "DATA_DIR", data),
                mock.patch.object(server_backup, "_runtime_history_status", return_value={"active": True}),
                mock.patch.object(server_backup, "_runtime_history_export", side_effect=export) as exported,
            ):
                encrypted_manifest = server_backup.create_backup(
                    server_backup.BackupOptions(
                        scope=server_backup.BackupScope.FILES,
                        output_path=encrypted,
                        encrypt=True,
                        password_fd=read_fd,
                    )
                )
            os.close(read_fd)
            assert encrypted_manifest.sqlite_history is not None
            self.assertTrue(encrypted_manifest.sqlite_history["includes_repository_key"])
            exported.assert_called_once_with(mock.ANY, include_repository_key=True)

    def test_runtime_private_secrets_never_enter_generic_walker(self) -> None:
        for active in (False, True):
            for directory in ("secrets", "cache", "scratch", "transfers"):
                with self.subTest(active=active, directory=directory):
                    self.assertTrue(
                        server_backup._should_skip_data_path(
                            Path("_userspace") / "_sqlite_history" / directory / "repository-password",
                            skip_runtime_history=active,
                        )
                    )

    def test_restore_import_runs_after_files_and_legacy_archive_skips_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "backup.tar.gz"
            staging = root / "staging"
            (staging / "data").mkdir(parents=True)
            (staging / "data" / "restored.txt").write_text("restored", encoding="utf-8")
            bundle = staging / "runtime-sqlite-history" / "export.bundle"
            bundle.parent.mkdir()
            bundle.write_bytes(b"bundle")
            (staging / "backup-meta.json").write_text(
                json.dumps(
                    {
                        "format": "tar.gz",
                        "version": 2,
                        "scope": "files",
                        "encrypted": False,
                        "includes_managed_key": False,
                        "sqlite_history": {
                            "version": 1,
                            "repository_id": "b" * 64,
                            "includes_repository_key": False,
                            "requires_original_repository_key": True,
                            "bundle": "runtime-sqlite-history/export.bundle",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with tarfile.open(archive, "w:gz") as tar:
                for child in staging.rglob("*"):
                    tar.add(child, child.relative_to(staging).as_posix())

            destination = root / "destination"
            destination.mkdir()
            events: list[str] = []
            original_copy = server_backup._copy_tree_contents

            def copy(*args, **kwargs):
                result = original_copy(*args, **kwargs)
                events.append("files")
                return result

            def imported(*_args):
                self.assertEqual((destination / "restored.txt").read_text(encoding="utf-8"), "restored")
                events.append("history")

            with (
                mock.patch.object(server_backup, "DATA_DIR", destination),
                mock.patch.object(server_backup, "_copy_tree_contents", side_effect=copy),
                mock.patch.object(server_backup, "_runtime_history_import", side_effect=imported),
                mock.patch.object(server_backup, "_preflight_runtime_history_import"),
                mock.patch.object(server_backup, "_invalidate_restored_workspace_runtime_artifacts"),
            ):
                server_backup.restore_backup(
                    server_backup.RestoreOptions(
                        archive_path=archive,
                        scope_override=server_backup.BackupScope.FILES,
                        replace_data=True,
                        restore_confirmation="RESTORE ragtime",
                    )
                )
            # The restore safety snapshot also uses the shared copy helper;
            # activation must still happen after the actual files restore.
            self.assertEqual(events[-2:], ["files", "history"])

            # Version-1 archives have no runtime metadata and remain readable.
            legacy = root / "legacy.tar.gz"
            with tarfile.open(legacy, "w:gz") as tar:
                tar.add(staging / "data" / "restored.txt", "data/restored.txt")
            with mock.patch.object(server_backup, "DATA_DIR", destination), mock.patch.object(server_backup, "_runtime_history_import") as imported:
                server_backup.restore_backup(server_backup.RestoreOptions(archive_path=legacy, restore_confirmation="RESTORE ragtime"))
            imported.assert_not_called()

    def test_known_active_runtime_status_failure_is_not_silently_skipped(self) -> None:
        class Response:
            def __init__(self, payload: dict[str, object]) -> None:
                self.payload = payload

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with (
            mock.patch.object(server_backup, "_runtime_history_endpoint", return_value=("http://runtime", "token")),
            mock.patch.object(
                server_backup,
                "urlopen",
                side_effect=[
                    Response({"version": 2, "active": True, "capability": True}),
                    HTTPError("http://runtime/sqlite-history/exports/status", 503, "down", Message(), None),
                ],
            ),
        ):
            with self.assertRaisesRegex(server_backup.BackupError, "status could not be determined"):
                server_backup._runtime_history_status()

        with (
            mock.patch.object(server_backup, "_runtime_history_endpoint", return_value=("http://runtime", "token")),
            mock.patch.object(server_backup, "urlopen", return_value=Response({"version": 2, "active": False, "capability": True})) as request,
        ):
            self.assertIsNone(server_backup._runtime_history_status())
        request.assert_called_once()

    def test_generic_replace_preserves_managed_history_inode_in_place(self) -> None:
        """Generic replacement must not unlink a runtime-owned live lock."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "destination"
            lock = destination / "_userspace" / "_sqlite_history" / "restic" / "lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("live", encoding="utf-8")
            inode = lock.stat().st_ino
            (destination / "replace-me.txt").write_text("old", encoding="utf-8")
            source = root / "source"
            (source / "replace-me.txt").parent.mkdir(parents=True)
            (source / "replace-me.txt").write_text("new", encoding="utf-8")
            with mock.patch.object(server_backup, "DATA_DIR", destination):
                server_backup._copy_tree_contents(source, destination, replace=True, preserve_runtime_history=True)
            self.assertEqual(lock.stat().st_ino, inode)
            self.assertEqual(lock.read_text(encoding="utf-8"), "live")
            self.assertEqual((destination / "replace-me.txt").read_text(encoding="utf-8"), "new")

    def test_restore_failure_rollback_preserves_runtime_history_key_and_lock_inodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "destination"
            history = destination / "_userspace" / "_sqlite_history"
            lock = history / "restic" / "lock"
            key = history / "secrets" / "repository-password"
            lock.parent.mkdir(parents=True)
            key.parent.mkdir(parents=True)
            lock.write_text("live-lock", encoding="utf-8")
            key.write_text("live-key", encoding="utf-8")
            workspace_lock = destination / "_userspace" / "workspaces" / "ws" / "sqlite_backups" / ".lock"
            workspace_lock.parent.mkdir(parents=True)
            workspace_lock.write_text("workspace-lock", encoding="utf-8")
            original_inodes = {path: path.stat().st_ino for path in (lock, key, workspace_lock)}
            (destination / "old.txt").write_text("old", encoding="utf-8")
            staging = root / "staging"
            (staging / "data").mkdir(parents=True)
            (staging / "data" / "new.txt").write_text("new", encoding="utf-8")
            bundle = staging / "runtime-sqlite-history" / "export.bundle"
            bundle.parent.mkdir()
            bundle.write_bytes(b"bundle")
            (staging / "backup-meta.json").write_text(
                json.dumps(
                    {
                        "format": "tar.gz",
                        "version": 2,
                        "scope": "files",
                        "encrypted": False,
                        "includes_managed_key": False,
                        "sqlite_history": {
                            "version": 1,
                            "repository_id": "c" * 64,
                            "includes_repository_key": False,
                            "requires_original_repository_key": True,
                            "bundle": "runtime-sqlite-history/export.bundle",
                        },
                    }
                ),
                encoding="utf-8",
            )
            archive = root / "backup.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for child in staging.rglob("*"):
                    tar.add(child, child.relative_to(staging).as_posix())
            with (
                mock.patch.object(server_backup, "DATA_DIR", destination),
                mock.patch.object(server_backup, "_preflight_runtime_history_import"),
                mock.patch.object(server_backup, "_runtime_history_import", side_effect=RuntimeError("activation failed")),
                mock.patch.object(server_backup, "_invalidate_restored_workspace_runtime_artifacts"),
            ):
                with self.assertRaises(server_backup.BackupMutationError):
                    server_backup.restore_backup(
                        server_backup.RestoreOptions(
                            archive_path=archive,
                            scope_override=server_backup.BackupScope.FILES,
                            replace_data=True,
                            restore_confirmation="RESTORE ragtime",
                        )
                    )
            self.assertEqual((destination / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((destination / "new.txt").exists())
            for path, inode in original_inodes.items():
                self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(key.read_text(encoding="utf-8"), "live-key")

    def test_runtime_history_path_is_exact_workspace_catalog_location(self) -> None:
        self.assertTrue(server_backup._is_runtime_history_path(Path("_userspace/workspaces/ws/sqlite_backups")))
        self.assertTrue(server_backup._is_runtime_history_path(Path("_userspace/workspaces/ws/sqlite_backups/manifest-v1.json")))
        self.assertFalse(server_backup._is_runtime_history_path(Path("_userspace/workspaces/ws/files/sqlite_backups/legacy.db")))
