import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _write_password_fd(password: str) -> int:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, password.encode("utf-8"))
    os.close(write_fd)
    return read_fd


def _make_tar(path: Path, members: dict[str, bytes], *, symlink: tuple[str, str] | None = None, member_type: bytes | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            archive.addfile(info)
        if member_type is not None:
            info = tarfile.TarInfo("special")
            info.type = member_type
            archive.addfile(info)


class _NonSeekableInput(io.RawIOBase):
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class ServerBackupTests(unittest.TestCase):
    def test_create_plaintext_backup_excludes_managed_key_without_bytesio_tar_buffer(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / ".encryption_key").write_text("managed-key", encoding="utf-8")
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "ENCRYPTION_KEY_FILE", data_dir / ".encryption_key"),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
            ):
                manifest = create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path))

            self.assertFalse(manifest.encrypted)
            self.assertFalse(hasattr(server_backup, "io"))
            with tarfile.open(output_path, "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("data/document.txt", names)
            self.assertNotIn("data/.encryption_key", names)

    def test_restore_wrong_password_fails_before_mutation(self) -> None:
        from ragtime.core.server_backup import BackupOptions, BackupScope, BackupValidationError, RestoreOptions, create_backup, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            archive_path = Path(tmpdir) / "backup.ragbak"
            password_fd = _write_password_fd("secret-password")
            wrong_fd = _write_password_fd("wrong-password")

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            try:
                with (
                    mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                    mock.patch("ragtime.core.server_backup._dump_database", side_effect=fake_dump),
                ):
                    create_backup(BackupOptions(scope=BackupScope.DATABASE, output_path=archive_path, encrypt=True, password_fd=password_fd))

                with (
                    mock.patch("ragtime.core.server_backup._restore_database") as restore_db,
                    mock.patch("ragtime.core.server_backup._copy_tree_contents") as restore_files,
                ):
                    with self.assertRaises(BackupValidationError):
                        restore_backup(
                            RestoreOptions(
                                archive_path=archive_path,
                                password_fd=wrong_fd,
                                restore_confirmation="RESTORE ragtime",
                            )
                        )
                restore_db.assert_not_called()
                restore_files.assert_not_called()
            finally:
                os.close(password_fd)
                os.close(wrong_fd)

    def test_inspect_backup_rejects_unsafe_symlink_and_special_member(self) -> None:
        from ragtime.core.server_backup import BackupValidationError, inspect_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "unsafe-link.tar.gz"
            _make_tar(
                archive_path,
                {"backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode()},
                symlink=("data/link", "/etc/passwd"),
            )
            with self.assertRaises(BackupValidationError):
                inspect_backup(archive_path)

            special_path = Path(tmpdir) / "special.tar.gz"
            _make_tar(
                special_path, {"backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode()}, member_type=tarfile.FIFOTYPE
            )
            with self.assertRaises(BackupValidationError):
                inspect_backup(special_path)

    def test_restore_supports_legacy_faiss_directory(self) -> None:
        from ragtime.core.server_backup import RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "legacy.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode(),
                    "faiss/file.txt": b"legacy",
                },
            )
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir):
                manifest = restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"))

            self.assertEqual(manifest.scope.value, "files")
            self.assertEqual((data_dir / "file.txt").read_text(encoding="utf-8"), "legacy")

    def test_restore_merge_rollback_restores_previous_files(self) -> None:
        from ragtime.core.server_backup import BackupError, BackupMutationError, RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "files.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode(),
                    "data/new.txt": b"new",
                },
            )
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "old.txt").write_text("old", encoding="utf-8")

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._invalidate_restored_workspace_runtime_artifacts", side_effect=BackupError("boom")),
            ):
                with self.assertRaises(BackupMutationError):
                    restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"))

            self.assertEqual((data_dir / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse((data_dir / "new.txt").exists())

    def test_restore_uses_single_transaction_and_preserves_mutation_error_type(self) -> None:
        from ragtime.core.server_backup import BackupMutationError, RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "db.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "database"}).encode(),
                    "database.dump": b"dump",
                },
            )
            commands = []

            def fake_run_command(command, **_kwargs):
                commands.append(command)
                raise RuntimeError("restore failed after mutation start")

            def fake_safety_dump(destination: Path) -> None:
                destination.write_bytes(b"safety")

            with (
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=fake_safety_dump),
                mock.patch("ragtime.core.server_backup._run_command", side_effect=fake_run_command),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
            ):
                with self.assertRaises(BackupMutationError):
                    restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"))

            self.assertTrue(any("--single-transaction" in cmd for cmd in commands))

    def test_full_restore_rolls_back_database_when_files_step_fails(self) -> None:
        from ragtime.core.server_backup import BackupMutationError, RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "full.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "full"}).encode(),
                    "database.dump": b"dump",
                    "data/new.txt": b"new",
                },
            )
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "old.txt").write_text("old", encoding="utf-8")
            events = []

            def fake_safety_dump(destination: Path) -> None:
                events.append(("safety_dump", destination.name))
                destination.write_bytes(b"safety")

            def fake_restore_database(archive: Path, *, data_only: bool) -> None:
                del data_only
                events.append(("restore_db", archive.name))

            def fake_copy_tree(_source: Path, _destination: Path, *, replace: bool) -> None:
                del replace
                events.append(("files", "fail"))
                raise RuntimeError("files failed")

            def fake_db_rollback(archive: Path) -> None:
                events.append(("rollback_db", archive.name))

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=fake_safety_dump),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
                mock.patch("ragtime.core.server_backup._restore_database", side_effect=fake_restore_database),
                mock.patch("ragtime.core.server_backup._run_migrations", return_value=None),
                mock.patch("ragtime.core.server_backup._invalidate_restored_runtime_sessions", return_value=None),
                mock.patch("ragtime.core.server_backup._copy_tree_contents", side_effect=fake_copy_tree),
                mock.patch("ragtime.core.server_backup._restore_database_from_safety_dump", side_effect=fake_db_rollback),
            ):
                with self.assertRaises(BackupMutationError):
                    restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"))

            self.assertEqual(events[0][0], "safety_dump")
            self.assertEqual(events[1], ("restore_db", "database.dump"))
            self.assertEqual(events[2], ("files", "fail"))
            self.assertEqual(events[3][0], "rollback_db")
            self.assertEqual((data_dir / "old.txt").read_text(encoding="utf-8"), "old")

    def test_full_restore_rolls_back_database_when_migration_fails(self) -> None:
        from ragtime.core.server_backup import BackupMutationError, RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "full.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "full"}).encode(),
                    "database.dump": b"dump",
                    "data/new.txt": b"new",
                },
            )
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "old.txt").write_text("old", encoding="utf-8")
            events = []

            def fake_safety_dump(destination: Path) -> None:
                events.append(("safety_dump", destination.name))
                destination.write_bytes(b"safety")

            def fake_restore_database(archive: Path, *, data_only: bool) -> None:
                del data_only
                events.append(("restore_db", archive.name))

            def fake_migrations() -> None:
                events.append(("migrations", "fail"))
                raise RuntimeError("migrations failed")

            def fake_db_rollback(archive: Path) -> None:
                events.append(("rollback_db", archive.name))

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=fake_safety_dump),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
                mock.patch("ragtime.core.server_backup._restore_database", side_effect=fake_restore_database),
                mock.patch("ragtime.core.server_backup._run_migrations", side_effect=fake_migrations),
                mock.patch("ragtime.core.server_backup._restore_database_from_safety_dump", side_effect=fake_db_rollback),
            ):
                with self.assertRaises(BackupMutationError):
                    restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"))

            self.assertEqual(events[0][0], "safety_dump")
            self.assertEqual(events[1], ("restore_db", "database.dump"))
            self.assertEqual(events[2], ("migrations", "fail"))
            self.assertEqual(events[3][0], "rollback_db")

    def test_local_admin_mirroring_uses_psql_variable_binding_and_role_precedence(self) -> None:
        from ragtime.core.server_backup import _mirror_local_admin_access

        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            stdout = "1\n"
            if "candidate_scores" in command[-1]:
                stdout = "safe-source\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        target_username = "local:ad'min"
        source_username = "src'; drop table users; --"
        with mock.patch("ragtime.core.server_backup.subprocess.run", side_effect=fake_run):
            _mirror_local_admin_access(source_username, target_username)

        sql_texts = [command[-1] for command in calls]
        flat_args = [arg for command in calls for arg in command]
        self.assertIn(f"target_username={target_username}", flat_args)
        self.assertIn(f"source_username={source_username}", flat_args)
        self.assertTrue(any(":'target_username'" in sql for sql in sql_texts))
        self.assertTrue(any("CASE EXCLUDED.role" in sql for sql in sql_texts))
        self.assertFalse(any(source_username in sql for sql in sql_texts))

    def test_local_admin_mirroring_auto_mode_skips_when_no_eligible_source(self) -> None:
        from ragtime.core.server_backup import _mirror_local_admin_access

        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            stdout = ""
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        with mock.patch("ragtime.core.server_backup.subprocess.run", side_effect=fake_run):
            _mirror_local_admin_access("auto", "admin")

        self.assertEqual(len(calls), 2)
        self.assertTrue(any("candidate_scores" in command[-1] for command in calls))

    def test_restore_confirmation_phrase_respects_env_override_and_case(self) -> None:
        from ragtime.core.server_backup import RestoreOptions, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "files.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode(),
                    "data/file.txt": b"ok",
                },
            )
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            with (
                mock.patch.dict(os.environ, {"RESTORE_CONFIRMATION_PHRASE": "Restore ragtime"}, clear=False),
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
            ):
                restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="Restore ragtime"))

            self.assertEqual((data_dir / "file.txt").read_text(encoding="utf-8"), "ok")

    def test_confirmation_phrase_preserves_postgres_db_case(self) -> None:
        from ragtime.core.server_backup import _confirmation_phrase

        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_URL": "",
                "POSTGRES_DB": "RagTimeProd",
                "POSTGRES_USER": "ragtime",
                "POSTGRES_PASSWORD": "pw",
                "POSTGRES_HOST": "localhost",
            },
            clear=False,
        ):
            self.assertEqual(_confirmation_phrase(), "RESTORE RagTimeProd")

    def test_main_passes_stdin_stream_and_legacy_aliases(self) -> None:
        import sys

        from ragtime.core.server_backup import BackupScope, main

        captured = {}

        def fake_restore(options, progress=None):
            del progress
            captured["scope"] = options.scope_override
            captured["stream"] = options.archive_stream
            return None

        fake_stdin = SimpleNamespace(buffer=_NonSeekableInput(b"archive"))
        with (
            mock.patch("ragtime.core.server_backup.restore_backup", side_effect=fake_restore),
            mock.patch.object(sys, "stdin", fake_stdin),
        ):
            result = main(["restore", "--faiss-only", "--include-secret", "--confirm-restore", "RESTORE RAGTIME", "-"])

        self.assertEqual(result, 0)
        self.assertEqual(captured["scope"], BackupScope.FILES)
        self.assertIs(captured["stream"], fake_stdin.buffer)


if __name__ == "__main__":
    unittest.main()
