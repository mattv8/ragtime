import io
import json
import os
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from ragtime.core.server_backup import decrypt_stream, encrypt_stream


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


def _read_tar_members(path: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(path, "r:gz") as archive:
        return {member.name: member for member in archive.getmembers()}


def _decrypt_backup_to_tarball(path: Path, password: str) -> Path:
    decrypted_dir = Path(tempfile.mkdtemp())
    decrypted_path = decrypted_dir / "backup.tar.gz"
    with path.open("rb") as encrypted_source, decrypted_path.open("wb") as decrypted_destination:
        decrypt_stream(encrypted_source, decrypted_destination, password)
    return decrypted_path


def _read_tar_text(path: Path, member_name: str) -> str:
    with tarfile.open(path, "r:gz") as archive:
        extracted = archive.extractfile(member_name)
        assert extracted is not None
        with extracted:
            return extracted.read().decode("utf-8")


def _make_tar_entries(
    path: Path,
    entries: list[dict[str, object]],
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for entry in entries:
            info = tarfile.TarInfo(str(entry["name"]))
            info.type = entry.get("type", tarfile.REGTYPE)
            info.linkname = str(entry.get("linkname", ""))
            content = entry.get("content", b"")
            if not isinstance(content, bytes):
                raise TypeError("tar entry content must be bytes")
            info.size = int(entry.get("size", len(content)))
            if info.isreg():
                archive.addfile(info, io.BytesIO(content))
            else:
                archive.addfile(info)


def _encrypt_tarball(source_path: Path, encrypted_path: Path, password: str) -> None:
    with source_path.open("rb") as source_handle, encrypted_path.open("wb") as destination_handle:
        encrypt_stream(source_handle, destination_handle, password)


def _make_encrypted_tar_entries(path: Path, entries: list[dict[str, object]], password: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        plaintext_path = Path(tmpdir) / "payload.tar.gz"
        _make_tar_entries(plaintext_path, entries)
        _encrypt_tarball(plaintext_path, path, password)


class _NonSeekableInput(io.RawIOBase):
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


class ServerBackupTests(unittest.TestCase):
    def _assert_relative_item_progress(self, events: list[dict[str, object]], expected_items: set[str]) -> None:
        item_events = [event for event in events if "current_item" in event]
        self.assertTrue(item_events)
        self.assertTrue(all(not Path(str(event["current_item"])).is_absolute() for event in item_events))
        self.assertEqual(item_events[-1]["processed_items"], item_events[-1]["total_items"])
        progress_values = [cast(int, event["progress"]) for event in events]
        self.assertEqual(progress_values, sorted(progress_values))
        self.assertTrue(expected_items.issubset({str(event["current_item"]) for event in item_events}))

    def test_create_backup_emits_active_messages_before_long_operations(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"
            events: list[dict[str, object]] = []

            def current_phase() -> str:
                return str(events[-1]["phase"])

            def fake_dump(destination: Path) -> None:
                self.assertEqual(current_phase(), "database_dump_start")
                self.assertEqual(events[-1]["message"], "Dumping database")
                destination.write_bytes(b"db-dump")

            def fake_copy_tree(source: Path, destination: Path, **_kwargs) -> int:
                self.assertEqual(current_phase(), "files_collect_start")
                self.assertEqual(events[-1]["message"], "Copying data files")
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "document.txt").write_text((source / "document.txt").read_text(encoding="utf-8"), encoding="utf-8")
                return 1

            def fake_build_tarball(root: Path, archive_path: Path, **_kwargs) -> int:
                self.assertEqual(current_phase(), "archive_build_start")
                self.assertEqual(
                    events[-1]["message"],
                    "Compressing staged database dump, data files, and manifest into a gzip-compressed tar archive",
                )
                archive_path.write_bytes(b"archive")
                return 4

            def fake_copyfileobj(_src, dst) -> None:
                self.assertEqual(current_phase(), "output_write_start")
                self.assertEqual(events[-1]["message"], "Writing backup artifact")
                dst.write(b"archive")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_copy_backup_data_tree", side_effect=fake_copy_tree),
                mock.patch.object(server_backup, "_build_tarball", side_effect=fake_build_tarball),
                mock.patch.object(server_backup.shutil, "copyfileobj", side_effect=fake_copyfileobj),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path), progress=events.append)

            self.assertIn("database_dump_complete", [event["phase"] for event in events])
            self.assertIn("files_collect_complete", [event["phase"] for event in events])
            self.assertIn("archive_build_complete", [event["phase"] for event in events])
            self.assertIn("output_write_complete", [event["phase"] for event in events])

    def test_create_backup_emits_monotonic_progress_with_real_file_counts(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "notes.md").write_text("world", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"
            events: list[dict[str, object]] = []

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
            ):
                create_backup(
                    BackupOptions(scope=BackupScope.FULL, output_path=output_path),
                    progress=events.append,
                )

            phase_events = [event for event in events if "current_item" not in event]
            self.assertEqual(
                [event["phase"] for event in phase_events],
                [
                    "start",
                    "database_dump_start",
                    "database_dump_complete",
                    "files_collect_start",
                    "files_collect_complete",
                    "manifest_write",
                    "archive_build_start",
                    "archive_build_complete",
                    "output_write_start",
                    "output_write_complete",
                    "complete",
                ],
            )
            progress_values = [cast(int, event["progress"]) for event in events]
            self.assertEqual(progress_values, sorted(progress_values))
            self.assertEqual(events[-1]["progress"], 100)
            self.assertEqual(phase_events[4]["item_count"], 2)
            self.assertEqual(phase_events[7]["item_count"], 4)

    def test_create_backup_emits_relative_item_progress_for_copy_and_archive(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            nested_dir = data_dir / "nested"
            nested_dir.mkdir()
            (nested_dir / "notes.md").write_text("world", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"
            events: list[dict[str, object]] = []

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
                mock.patch("ragtime.core.server_backup.time.monotonic", side_effect=range(1, 50)),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path), progress=events.append)

            self._assert_relative_item_progress(
                events,
                {"document.txt", "nested/notes.md", "database.dump", "data/document.txt", "backup-meta.json"},
            )
            archive_start = next(event for event in events if event["phase"] == "archive_build_start")
            self.assertEqual(
                archive_start["message"],
                "Compressing staged database dump, data files, and manifest into a gzip-compressed tar archive",
            )

    def test_create_backup_preserves_root_dotfile_name_in_current_item(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / ".env.example").write_text("value", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"
            events: list[dict[str, object]] = []

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
                mock.patch("ragtime.core.server_backup.time.monotonic", side_effect=range(1, 50)),
            ):
                create_backup(BackupOptions(scope=BackupScope.FILES, output_path=output_path), progress=events.append)

            item_names = {str(event["current_item"]) for event in events if "current_item" in event}
            self.assertIn(".env.example", item_names)

    def test_emit_progress_callback_failures_do_not_abort_backup(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                mock.patch.object(server_backup, "_get_current_schema_version", return_value="20260716000000"),
                mock.patch.object(server_backup, "_get_ragtime_version", return_value="test-version"),
                mock.patch.object(server_backup.logger, "warning") as logger_warning,
            ):
                manifest = create_backup(
                    BackupOptions(scope=BackupScope.FULL, output_path=output_path),
                    progress=lambda _event: (_ for _ in ()).throw(RuntimeError("callback boom")),
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(manifest.scope, BackupScope.FULL)
            logger_warning.assert_called()

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

    def test_create_backup_excludes_server_backups_subtree(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backups_dir = data_dir / "_server_backups"
            backups_dir.mkdir(parents=True)
            (backups_dir / "old.ragbak").write_text("old-backup", encoding="utf-8")
            (data_dir / "keep.txt").write_text("keep", encoding="utf-8")
            (data_dir / "_tmp").mkdir()
            (data_dir / "_tmp" / "scratch.txt").write_text("scratch", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path))

            names = set(_read_tar_members(output_path))
            self.assertIn("data/keep.txt", names)
            self.assertNotIn("data/_server_backups/old.ragbak", names)
            self.assertNotIn("data/_tmp/scratch.txt", names)

    def test_create_backup_keeps_existing_tmp_prefix_exclusion_semantics(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "_tmpcache.txt").write_text("skip-me", encoding="utf-8")
            (data_dir / "keep.txt").write_text("keep", encoding="utf-8")
            output_path = Path(tmpdir) / "backup.tar.gz"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path))

            names = set(_read_tar_members(output_path))
            self.assertIn("data/keep.txt", names)
            self.assertNotIn("data/_tmpcache.txt", names)

    def test_create_backup_preserves_directory_symlink_without_archiving_target_contents_twice(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            target_dir = data_dir / "nested"
            target_dir.mkdir()
            (target_dir / "inside.txt").write_text("inside", encoding="utf-8")
            (data_dir / "linked-nested").symlink_to(target_dir, target_is_directory=True)
            output_path = Path(tmpdir) / "backup.tar.gz"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=output_path))

            members = _read_tar_members(output_path)
            self.assertIn("data/nested/inside.txt", members)
            self.assertIn("data/linked-nested", members)
            self.assertTrue(members["data/linked-nested"].issym())
            self.assertEqual(members["data/linked-nested"].linkname, str(target_dir))
            self.assertNotIn("data/linked-nested/inside.txt", members)

    def test_create_backup_includes_deployment_environment_only_for_encrypted_full_backups(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            plaintext_full_path = Path(tmpdir) / "full.tar.gz"
            encrypted_database_path = Path(tmpdir) / "database.ragbak"
            encrypted_files_path = Path(tmpdir) / "files.ragbak"
            encrypted_full_path = Path(tmpdir) / "full.ragbak"
            password = "secret-password"

            env = {
                "PORT": "",
                "LOCAL_ADMIN_PASSWORD": 'line1\nline"2\\end',
                "EXTERNAL_BASE_URL": "https://example.com/path",
                "DATABASE_URL": "postgres://user:pw@example/db",
                "POSTGRES_PASSWORD": "db-password",
                "DEBUG_MODE": "true",
                "CUSTOM_FUTURE_FLAG": "future-value",
                "BASH_FUNC_which%%": "() {  echo /usr/bin/which; }",
                "HOME": "/home/ragtime",
                "PYTHONPATH": "/venv/lib/python",
            }

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
            ):
                create_backup(BackupOptions(scope=BackupScope.FULL, output_path=plaintext_full_path))
                create_backup(
                    BackupOptions(
                        scope=BackupScope.DATABASE,
                        output_path=encrypted_database_path,
                        encrypt=True,
                        password_fd=_write_password_fd(password),
                    )
                )
                create_backup(
                    BackupOptions(
                        scope=BackupScope.FILES,
                        output_path=encrypted_files_path,
                        encrypt=True,
                        password_fd=_write_password_fd(password),
                    )
                )
                manifest = create_backup(
                    BackupOptions(
                        scope=BackupScope.FULL,
                        output_path=encrypted_full_path,
                        encrypt=True,
                        password_fd=_write_password_fd(password),
                    )
                )

            plaintext_full_members = set(_read_tar_members(plaintext_full_path))
            self.assertNotIn("deployment-env.json", plaintext_full_members)

            decrypted_database_members = set(_read_tar_members(_decrypt_backup_to_tarball(encrypted_database_path, password)))
            self.assertNotIn("deployment-env.json", decrypted_database_members)

            decrypted_files_members = set(_read_tar_members(_decrypt_backup_to_tarball(encrypted_files_path, password)))
            self.assertNotIn("deployment-env.json", decrypted_files_members)

            decrypted_full_path = _decrypt_backup_to_tarball(encrypted_full_path, password)
            decrypted_full_members = set(_read_tar_members(decrypted_full_path))
            self.assertIn("deployment-env.json", decrypted_full_members)
            deployment_env_payload = json.loads(_read_tar_text(decrypted_full_path, "deployment-env.json"))

            manifest_payload = json.loads(_read_tar_text(decrypted_full_path, "backup-meta.json"))
            self.assertEqual(
                manifest_payload["deployment_environment_variables"],
                [
                    "CUSTOM_FUTURE_FLAG",
                    "DATABASE_URL",
                    "DEBUG_MODE",
                    "EXTERNAL_BASE_URL",
                    "LOCAL_ADMIN_PASSWORD",
                    "PORT",
                    "POSTGRES_PASSWORD",
                ],
            )
            self.assertEqual(
                deployment_env_payload["variables"],
                {
                    "CUSTOM_FUTURE_FLAG": "future-value",
                    "DATABASE_URL": "postgres://user:pw@example/db",
                    "DEBUG_MODE": "true",
                    "EXTERNAL_BASE_URL": "https://example.com/path",
                    "LOCAL_ADMIN_PASSWORD": 'line1\nline"2\\end',
                    "PORT": "",
                    "POSTGRES_PASSWORD": "db-password",
                },
            )
            self.assertNotIn("HOME", deployment_env_payload["variables"])
            self.assertNotIn("BASH_FUNC_which%%", deployment_env_payload["variables"])
            self.assertNotIn("PYTHONPATH", deployment_env_payload["variables"])
            self.assertNotIn("BASH_FUNC_which%%", manifest_payload["deployment_environment_variables"])
            manifest_json = json.dumps(manifest_payload, sort_keys=True)
            self.assertNotIn(env["CUSTOM_FUTURE_FLAG"], manifest_json)
            self.assertNotIn(env["DATABASE_URL"], manifest_json)
            self.assertNotIn(env["LOCAL_ADMIN_PASSWORD"], manifest_json)
            self.assertEqual(
                manifest.deployment_environment_variables,
                manifest_payload["deployment_environment_variables"],
            )

    def test_recover_deployment_environment_returns_sorted_variables_and_warnings(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, create_backup, recover_deployment_environment

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            (data_dir / "document.txt").write_text("hello", encoding="utf-8")
            archive_path = Path(tmpdir) / "full.ragbak"
            password = "secret-password"
            env = {
                "PORT": "",
                "LOCAL_ADMIN_PASSWORD": 'line1\nline"2\\end',
                "EXTERNAL_BASE_URL": "https://example.com/path",
                "DATABASE_URL": "postgres://user:pw@example/db",
                "POSTGRES_PASSWORD": "db-password",
                "UNAPPROVED_SECRET": "should-not-leak",
            }

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(server_backup, "DATA_DIR", data_dir),
                mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
            ):
                create_backup(
                    BackupOptions(
                        scope=BackupScope.FULL,
                        output_path=archive_path,
                        encrypt=True,
                        password_fd=_write_password_fd(password),
                    )
                )

            recovery = recover_deployment_environment(archive_path, password)
            self.assertEqual(
                recovery.variable_names,
                [
                    "DATABASE_URL",
                    "EXTERNAL_BASE_URL",
                    "LOCAL_ADMIN_PASSWORD",
                    "PORT",
                    "POSTGRES_PASSWORD",
                    "UNAPPROVED_SECRET",
                ],
            )
            self.assertEqual(
                recovery.variables,
                {
                    "DATABASE_URL": "postgres://user:pw@example/db",
                    "EXTERNAL_BASE_URL": "https://example.com/path",
                    "LOCAL_ADMIN_PASSWORD": 'line1\nline"2\\end',
                    "PORT": "",
                    "POSTGRES_PASSWORD": "db-password",
                    "UNAPPROVED_SECRET": "should-not-leak",
                },
            )
            self.assertEqual(len(recovery.warnings), 2)
            self.assertTrue(any("DATABASE_URL" in warning for warning in recovery.warnings))
            self.assertTrue(any("POSTGRES_PASSWORD" in warning for warning in recovery.warnings))

    def test_recover_deployment_environment_preserves_older_now_ignored_names(self) -> None:
        from ragtime.core.server_backup import DEPLOYMENT_ENV_NAME, recover_deployment_environment

        password = "secret-password"

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "ignored-name.ragbak"
            _make_encrypted_tar_entries(
                archive_path,
                [
                    {
                        "name": DEPLOYMENT_ENV_NAME,
                        "content": json.dumps(
                            {
                                "version": 1,
                                "variables": {
                                    "CUSTOM_FUTURE_FLAG": "future-value",
                                    "HOME": "/home/legacy",
                                },
                            }
                        ).encode("utf-8"),
                    }
                ],
                password,
            )

            recovery = recover_deployment_environment(archive_path, password)

            self.assertEqual(recovery.variable_names, ["CUSTOM_FUTURE_FLAG", "HOME"])
            self.assertEqual(
                recovery.variables,
                {
                    "CUSTOM_FUTURE_FLAG": "future-value",
                    "HOME": "/home/legacy",
                },
            )

    def test_recover_deployment_environment_rejects_invalid_payload_variants_without_secret_leak(self) -> None:
        from ragtime.core.server_backup import DEPLOYMENT_ENV_NAME, BackupValidationError, recover_deployment_environment

        password = "secret-password"
        fixture_secret = "fixture-secret-value"

        def payload_bytes(payload: object) -> bytes:
            return json.dumps(payload).encode("utf-8")

        cases = [
            {
                "name": "duplicate payload",
                "entries": [
                    {"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"PORT": fixture_secret}})},
                    {"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"PORT": "other"}})},
                ],
                "expected": "duplicate",
            },
            {
                "name": "non-regular payload member",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "type": tarfile.SYMTYPE, "linkname": "relative-target"}],
                "expected": "invalid deployment environment payload",
            },
            {
                "name": "declared payload over limit",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": b"x" * ((1024 * 1024) + 1)}],
                "expected": "maximum supported size",
            },
            {
                "name": "missing payload",
                "entries": [{"name": "backup-meta.json", "content": b"{}"}],
                "expected": "does not contain a deployment environment payload",
            },
            {
                "name": "malformed json",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": b'{"version": 1, "variables": {"PORT": "fixture-secret-value"'}],
                "expected": "invalid deployment environment payload",
            },
            {
                "name": "invalid version",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 2, "variables": {"PORT": fixture_secret}})}],
                "expected": "version is invalid",
            },
            {
                "name": "variables must be object",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": [fixture_secret]})}],
                "expected": "variables are invalid",
            },
            {
                "name": "non-string key",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": b"{}"}],
                "expected": "variables are invalid",
                "patch": mock.patch(
                    "ragtime.core.server_backup._load_deployment_environment_payload",
                    side_effect=lambda _payload: __import__(
                        "ragtime.core.server_backup", fromlist=["_parse_deployment_environment_payload"]
                    )._parse_deployment_environment_payload({"version": 1, "variables": {1: fixture_secret}}),
                ),
            },
            {
                "name": "non-string value",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"PORT": [fixture_secret]}})}],
                "expected": "variables are invalid",
            },
            {
                "name": "unicode variable name",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"NÁME": fixture_secret}})}],
                "expected": "invalid name",
            },
            {
                "name": "hyphenated variable name",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"BAD-NAME": fixture_secret}})}],
                "expected": "invalid name",
            },
            {
                "name": "digit-prefixed variable name",
                "entries": [{"name": DEPLOYMENT_ENV_NAME, "content": payload_bytes({"version": 1, "variables": {"1NAME": fixture_secret}})}],
                "expected": "invalid name",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for case in cases:
                with self.subTest(case=case["name"]):
                    archive_path = Path(tmpdir) / f"{str(case['name']).replace(' ', '-')}.ragbak"
                    _make_encrypted_tar_entries(archive_path, case["entries"], password)
                    with self.assertRaises(BackupValidationError) as ctx:
                        with ExitStack() as stack:
                            patcher = case.get("patch")
                            if patcher is not None:
                                stack.enter_context(patcher)
                            recover_deployment_environment(archive_path, password)
                    message = str(ctx.exception)
                    self.assertIn(str(case["expected"]), message)
                    self.assertNotIn(fixture_secret, message)

    def test_recover_deployment_environment_rejects_actual_payload_over_limit_without_secret_leak(self) -> None:
        from ragtime.core.server_backup import DEPLOYMENT_ENV_NAME, BackupValidationError, recover_deployment_environment

        password = "secret-password"
        fixture_secret = "fixture-secret-value"

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "actual-over-limit.ragbak"
            payload = json.dumps({"version": 1, "variables": {"PORT": fixture_secret}}).encode("utf-8")
            _make_encrypted_tar_entries(archive_path, [{"name": DEPLOYMENT_ENV_NAME, "content": payload}], password)

            original_extractfile = tarfile.TarFile.extractfile

            def fake_extractfile(self, member, *args, **kwargs):
                extracted = original_extractfile(self, member, *args, **kwargs)
                if getattr(member, "name", "") != DEPLOYMENT_ENV_NAME:
                    return extracted
                return io.BytesIO(b"x" * ((1024 * 1024) + 1))

            with (
                mock.patch("tarfile.TarFile.extractfile", side_effect=fake_extractfile, autospec=True),
                self.assertRaises(BackupValidationError) as ctx,
            ):
                recover_deployment_environment(archive_path, password)

            message = str(ctx.exception)
            self.assertIn("maximum supported size", message)
            self.assertNotIn(fixture_secret, message)

    def test_restore_database_only_installs_managed_key_from_backup(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupOptions, BackupScope, RestoreOptions, create_backup, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            source_data_dir = Path(tmpdir) / "source-data"
            source_data_dir.mkdir()
            source_key_path = source_data_dir / ".encryption_key"
            source_key_path.write_bytes(b"backup-key")
            archive_path = Path(tmpdir) / "database.ragbak"
            password = "secret-password"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            password_fd = _write_password_fd(password)
            try:
                with (
                    mock.patch.object(server_backup, "DATA_DIR", source_data_dir),
                    mock.patch.object(server_backup, "ENCRYPTION_KEY_FILE", source_key_path),
                    mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                ):
                    create_backup(
                        BackupOptions(
                            scope=BackupScope.DATABASE,
                            output_path=archive_path,
                            encrypt=True,
                            password_fd=password_fd,
                        )
                    )
            finally:
                os.close(password_fd)

            restore_data_dir = Path(tmpdir) / "restore-data"
            restore_data_dir.mkdir()
            restore_key_path = restore_data_dir / ".encryption_key"
            restore_key_path.write_bytes(b"old-key")
            restore_key_path.chmod(0o644)

            with (
                mock.patch.object(server_backup, "DATA_DIR", restore_data_dir),
                mock.patch.object(server_backup, "_create_database_safety_dump", side_effect=lambda destination: destination.write_bytes(b"safety")),
                mock.patch.object(server_backup, "_terminate_other_database_connections", return_value=None),
                mock.patch.object(server_backup, "_restore_database", return_value=None),
                mock.patch.object(server_backup, "_run_migrations", return_value=None),
                mock.patch.object(server_backup, "_invalidate_restored_runtime_sessions", return_value=None),
            ):
                restore_backup(
                    RestoreOptions(
                        archive_path=archive_path,
                        password_fd=_write_password_fd(password),
                        scope_override=BackupScope.DATABASE,
                        restore_confirmation="RESTORE ragtime",
                    )
                )

            self.assertEqual(restore_key_path.read_bytes(), b"backup-key")
            self.assertEqual(restore_key_path.stat().st_mode & 0o777, 0o600)

    def test_restore_database_only_managed_key_failure_keeps_original_key(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import BackupMutationError, BackupOptions, BackupScope, RestoreOptions, create_backup, restore_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            source_data_dir = Path(tmpdir) / "source-data"
            source_data_dir.mkdir()
            source_key_path = source_data_dir / ".encryption_key"
            source_key_path.write_bytes(b"backup-key")
            archive_path = Path(tmpdir) / "database.ragbak"
            password = "secret-password"

            def fake_dump(destination: Path) -> None:
                destination.write_bytes(b"db-dump")

            password_fd = _write_password_fd(password)
            try:
                with (
                    mock.patch.object(server_backup, "DATA_DIR", source_data_dir),
                    mock.patch.object(server_backup, "ENCRYPTION_KEY_FILE", source_key_path),
                    mock.patch.object(server_backup, "_dump_database", side_effect=fake_dump),
                ):
                    create_backup(
                        BackupOptions(
                            scope=BackupScope.DATABASE,
                            output_path=archive_path,
                            encrypt=True,
                            password_fd=password_fd,
                        )
                    )
            finally:
                os.close(password_fd)

            restore_data_dir = Path(tmpdir) / "restore-data"
            restore_data_dir.mkdir()
            restore_key_path = restore_data_dir / ".encryption_key"
            restore_key_path.write_bytes(b"old-key")

            with (
                mock.patch.object(server_backup, "DATA_DIR", restore_data_dir),
                mock.patch.object(server_backup, "_create_database_safety_dump", side_effect=lambda destination: destination.write_bytes(b"safety")),
                mock.patch.object(server_backup, "_terminate_other_database_connections", return_value=None),
                mock.patch.object(server_backup, "_restore_database", return_value=None),
                mock.patch.object(server_backup, "_run_migrations", return_value=None),
                mock.patch.object(server_backup, "_invalidate_restored_runtime_sessions", side_effect=RuntimeError("stop before key replacement")),
                mock.patch.object(server_backup, "_restore_database_from_safety_dump", return_value=None),
            ):
                with self.assertRaises(BackupMutationError):
                    restore_backup(
                        RestoreOptions(
                            archive_path=archive_path,
                            password_fd=_write_password_fd(password),
                            scope_override=BackupScope.DATABASE,
                            restore_confirmation="RESTORE ragtime",
                        )
                    )

            self.assertEqual(restore_key_path.read_bytes(), b"old-key")

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

    def test_inspect_backup_emits_relative_item_progress_for_archive_extraction(self) -> None:
        from ragtime.core.server_backup import inspect_backup

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "inspect.tar.gz"
            _make_tar(
                archive_path,
                {
                    "backup-meta.json": json.dumps({"format": "tar.gz", "version": 1, "scope": "files"}).encode(),
                    "data/document.txt": b"hello",
                },
            )
            events: list[dict[str, object]] = []

            with mock.patch("ragtime.core.server_backup.time.monotonic", side_effect=range(1, 50)):
                inspect_backup(archive_path, progress=events.append)

            self._assert_relative_item_progress(events, {"backup-meta.json", "data/document.txt"})

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

    def test_restore_emits_validation_and_commit_progress_without_early_100(self) -> None:
        from ragtime.core.server_backup import RestoreOptions, restore_backup

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
            events: list[dict[str, object]] = []

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=lambda destination: destination.write_bytes(b"safety")),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
                mock.patch("ragtime.core.server_backup._restore_database", return_value=None),
                mock.patch("ragtime.core.server_backup._run_migrations", return_value=None),
                mock.patch("ragtime.core.server_backup._invalidate_restored_runtime_sessions", return_value=None),
                mock.patch("ragtime.core.server_backup._invalidate_restored_workspace_runtime_artifacts", return_value=None),
                mock.patch("ragtime.core.server_backup.time.monotonic", side_effect=range(1, 100)),
            ):
                restore_backup(
                    RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"),
                    progress=events.append,
                )

            phase_events = [event for event in events if "current_item" not in event]
            self.assertEqual(
                [event["phase"] for event in phase_events],
                [
                    "archive_prepare",
                    "archive_extract_start",
                    "archive_extract_complete",
                    "validation_complete",
                    "start",
                    "data_snapshot_start",
                    "data_snapshot_complete",
                    "database_safety_dump_start",
                    "database_safety_dump_complete",
                    "database_restore_start",
                    "database_restore_complete",
                    "database_migrations_start",
                    "database_migrations_complete",
                    "runtime_sessions_invalidate",
                    "files_restore_start",
                    "files_restore_complete",
                    "workspace_runtime_invalidate",
                    "complete",
                ],
            )
            self.assertTrue(all(cast(int, event["progress"]) < 100 for event in events[:-1]))
            self.assertEqual(phase_events[0]["progress"], 41)
            self.assertEqual(phase_events[3]["progress"], 44)
            self.assertEqual(phase_events[8]["progress"], 60)
            self.assertEqual(events[-1]["progress"], 100)

    def test_restore_emits_relative_item_progress_for_extract_snapshot_and_copy(self) -> None:
        from ragtime.core.server_backup import RestoreOptions, restore_backup

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
            (data_dir / "existing.txt").write_text("old", encoding="utf-8")
            snapshot_file = data_dir / "nested.txt"
            snapshot_file.write_text("nested", encoding="utf-8")
            events: list[dict[str, object]] = []

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=lambda destination: destination.write_bytes(b"safety")),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
                mock.patch("ragtime.core.server_backup._restore_database", return_value=None),
                mock.patch("ragtime.core.server_backup._run_migrations", return_value=None),
                mock.patch("ragtime.core.server_backup._invalidate_restored_runtime_sessions", return_value=None),
                mock.patch("ragtime.core.server_backup._invalidate_restored_workspace_runtime_artifacts", return_value=None),
                mock.patch("ragtime.core.server_backup.time.monotonic", side_effect=range(1, 100)),
            ):
                restore_backup(
                    RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"),
                    progress=events.append,
                )

            self._assert_relative_item_progress(
                events,
                {"database.dump", "data/new.txt", "existing.txt", "nested.txt", "new.txt"},
            )

    def test_restore_emits_active_messages_before_long_operations(self) -> None:
        from ragtime.core import server_backup
        from ragtime.core.server_backup import RestoreOptions, restore_backup

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
            events: list[dict[str, object]] = []

            def current_phase() -> str:
                return str(events[-1]["phase"])

            original_extract = server_backup._extract_archive

            def fake_extract(archive: Path, destination: Path, **kwargs):
                self.assertEqual(current_phase(), "archive_extract_start")
                self.assertEqual(events[-1]["message"], "Extracting restore archive")
                return original_extract(archive, destination, **kwargs)

            def fake_snapshot(target: Path, snapshot_root: Path, **_kwargs):
                self.assertEqual(current_phase(), "data_snapshot_start")
                self.assertEqual(events[-1]["message"], "Creating data safety snapshot")
                return (snapshot_root / "snap", 0)

            def fake_safety_dump(destination: Path) -> None:
                self.assertEqual(current_phase(), "database_safety_dump_start")
                self.assertEqual(events[-1]["message"], "Creating database safety dump")
                destination.write_bytes(b"safety")

            def fake_restore_database(_archive: Path, *, data_only: bool) -> None:
                del data_only
                self.assertEqual(current_phase(), "database_restore_start")
                self.assertEqual(events[-1]["message"], "Restoring database")

            def fake_run_migrations() -> None:
                self.assertEqual(current_phase(), "database_migrations_start")
                self.assertEqual(events[-1]["message"], "Applying database migrations")

            def fake_copy_tree(_source: Path, _destination: Path, *, replace: bool, **_kwargs) -> int:
                del replace
                self.assertEqual(current_phase(), "files_restore_start")
                self.assertEqual(events[-1]["message"], "Restoring data files")
                return 1

            with (
                mock.patch("ragtime.core.server_backup.DATA_DIR", data_dir),
                mock.patch("ragtime.core.server_backup._extract_archive", side_effect=fake_extract),
                mock.patch("ragtime.core.server_backup._snapshot_directory", side_effect=fake_snapshot),
                mock.patch("ragtime.core.server_backup._create_database_safety_dump", side_effect=fake_safety_dump),
                mock.patch("ragtime.core.server_backup._terminate_other_database_connections", return_value=None),
                mock.patch("ragtime.core.server_backup._restore_database", side_effect=fake_restore_database),
                mock.patch("ragtime.core.server_backup._run_migrations", side_effect=fake_run_migrations),
                mock.patch("ragtime.core.server_backup._invalidate_restored_runtime_sessions", return_value=None),
                mock.patch("ragtime.core.server_backup._copy_tree_contents", side_effect=fake_copy_tree),
                mock.patch("ragtime.core.server_backup._invalidate_restored_workspace_runtime_artifacts", return_value=None),
            ):
                restore_backup(RestoreOptions(archive_path=archive_path, restore_confirmation="RESTORE ragtime"), progress=events.append)

            self.assertIn("archive_extract_complete", [event["phase"] for event in events])
            self.assertIn("data_snapshot_complete", [event["phase"] for event in events])
            self.assertIn("database_safety_dump_complete", [event["phase"] for event in events])
            self.assertIn("database_restore_complete", [event["phase"] for event in events])
            self.assertIn("database_migrations_complete", [event["phase"] for event in events])
            self.assertIn("files_restore_complete", [event["phase"] for event in events])

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

            original_copy_tree = __import__("ragtime.core.server_backup", fromlist=["_copy_tree_contents"])._copy_tree_contents

            def fake_copy_tree(_source: Path, _destination: Path, *, replace: bool, **_kwargs) -> int:
                del replace
                if _destination != data_dir:
                    return original_copy_tree(_source, _destination, replace=False, **_kwargs)
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
