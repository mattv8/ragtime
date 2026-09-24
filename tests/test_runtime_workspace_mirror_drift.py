from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime.worker import sandbox


class RuntimeWorkspaceMirrorDriftTests(unittest.TestCase):
    def _create_mirrored_sqlite_database(self, canonical: Path, mirrored: Path) -> tuple[Path, Path]:
        database = canonical / ".ragtime" / "db" / "app.sqlite3"
        database.parent.mkdir(parents=True)
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE entries (value TEXT)")
        connection.execute("INSERT INTO entries VALUES ('preserved')")
        connection.commit()
        connection.close()
        wal = database.with_name(f"{database.name}-wal")
        wal.write_bytes(b"sqlite WAL sidecar")
        mirrored_database = mirrored / ".ragtime" / "db" / "app.sqlite3"
        mirrored_database.parent.mkdir(parents=True)
        shutil.copy2(database, mirrored_database)
        shutil.copy2(wal, mirrored_database.with_name(f"{mirrored_database.name}-wal"))
        return database, wal

    def test_conflicting_api_and_shell_edits_preserve_canonical_and_archive_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "app.txt").write_text("base")
            (mirrored / "app.txt").write_text("base")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            (canonical / "app.txt").write_text("api")
            (mirrored / "app.txt").write_text("shell")

            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=False)

            self.assertEqual((canonical / "app.txt").read_text(), "api")
            archives = list((root / sandbox._WORKSPACE_LEGACY_RECOVERY_DIR).iterdir())
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / "app.txt").read_text(), "shell")

    def test_conflict_retirement_invalidates_baseline_before_a_second_reconcile(self) -> None:
        """A retired conflict mirror must not publish an empty-mirror deletion baseline."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "app.txt").write_text("base")
            (mirrored / "app.txt").write_text("base")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            (canonical / "app.txt").write_text("api")
            (mirrored / "app.txt").write_text("shell")

            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=False)
            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=False)

            self.assertFalse(sandbox._workspace_mirror_hash_path(spec).exists())
            self.assertEqual((canonical / "app.txt").read_text(), "api")

    def test_source_wins_retirement_invalidates_baseline_before_a_second_reconcile(self) -> None:
        """A legacy-source archive must not turn its new empty mirror into a delete source."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "app.txt").write_text("stale")
            (mirrored / "app.txt").write_text("legacy")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)

            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=True)
            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=False)

            self.assertFalse(sandbox._workspace_mirror_hash_path(spec).exists())
            self.assertEqual((canonical / "app.txt").read_text(), "legacy")

    def test_shell_deletion_is_reconciled_when_canonical_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "app.txt").write_text("base")
            (mirrored / "app.txt").write_text("base")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            (mirrored / "app.txt").unlink()

            sandbox._reconcile_workspace_copy(spec, label="test", prefer_source=False)

            self.assertFalse((canonical / "app.txt").exists())

    def test_stop_reconcile_preserves_artifact_sidecar_create_update_move_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "dashboard").mkdir()
            (mirrored / "dashboard").mkdir()
            (canonical / "dashboard" / "main.ts").write_text("base")
            (mirrored / "dashboard" / "main.ts").write_text("base")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")

            sandbox._write_workspace_mirror_hashes(spec)
            sidecar = mirrored / "dashboard" / "main.ts.artifact.json"
            sidecar.write_text('{"artifact_type":"module_ts","revision":1}')
            sandbox._reconcile_workspace_copy(spec, label="create", prefer_source=False)
            self.assertEqual((canonical / "dashboard" / "main.ts.artifact.json").read_text(), sidecar.read_text())

            sandbox._write_workspace_mirror_hashes(spec)
            sidecar.write_text('{"artifact_type":"module_ts","revision":2}')
            sandbox._reconcile_workspace_copy(spec, label="update", prefer_source=False)
            self.assertIn("2", (canonical / "dashboard" / "main.ts.artifact.json").read_text())

            sandbox._write_workspace_mirror_hashes(spec)
            moved = mirrored / "dashboard" / "app.ts"
            moved_sidecar = mirrored / "dashboard" / "app.ts.artifact.json"
            (mirrored / "dashboard" / "main.ts").rename(moved)
            sidecar.rename(moved_sidecar)
            sandbox._reconcile_workspace_copy(spec, label="move", prefer_source=False)
            self.assertFalse((canonical / "dashboard" / "main.ts").exists())
            self.assertFalse((canonical / "dashboard" / "main.ts.artifact.json").exists())
            self.assertEqual((canonical / "dashboard" / "app.ts.artifact.json").read_text(), moved_sidecar.read_text())

            sandbox._write_workspace_mirror_hashes(spec)
            moved.unlink()
            moved_sidecar.unlink()
            sandbox._reconcile_workspace_copy(spec, label="delete", prefer_source=False)
            self.assertFalse((canonical / "dashboard" / "app.ts").exists())
            self.assertFalse((canonical / "dashboard" / "app.ts.artifact.json").exists())

    def test_restart_reconcile_removes_mirror_file_deleted_while_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "deleted.txt").write_text("baseline")
            (mirrored / "deleted.txt").write_text("baseline")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            (canonical / "deleted.txt").unlink()

            sandbox._reconcile_workspace_copy(spec, label="restart", prefer_source=False)

            self.assertFalse((canonical / "deleted.txt").exists())
            self.assertFalse((mirrored / "deleted.txt").exists())

    def test_restart_reconcile_preserves_snapshot_restore_over_stale_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirrored = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirrored.mkdir(parents=True)
            (canonical / "app.txt").write_text("second")
            (mirrored / "app.txt").write_text("second")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            # Stop records the clean B baseline; inactive snapshot restore then
            # changes only canonical storage back to A.
            sandbox._write_workspace_mirror_hashes(spec)
            (canonical / "app.txt").write_text("first")

            sandbox._reconcile_workspace_copy(spec, label="restart", prefer_source=False)

            self.assertEqual((canonical / "app.txt").read_text(), "first")

    def test_archive_invalidates_baseline_when_mirror_is_already_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            canonical.mkdir()
            (canonical / "app.txt").write_text("canonical")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)

            sandbox.archive_workspace_mirror(spec)

            self.assertFalse(sandbox._workspace_mirror_hash_path(spec).exists())

    def test_archive_interruption_after_rename_does_not_leave_a_deletion_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirror = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirror.mkdir(parents=True)
            (canonical / "app.txt").write_text("canonical")
            (mirror / "app.txt").write_text("mirror")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)

            with (
                self.assertRaisesRegex(RuntimeError, "interrupted"),
                mock.patch.object(
                    sandbox,
                    "_ensure_real_directory",
                    side_effect=RuntimeError("interrupted"),
                ),
            ):
                sandbox.archive_workspace_mirror(spec)

            self.assertFalse(sandbox._workspace_mirror_hash_path(spec).exists())
            self.assertFalse(mirror.exists())
            archives = list((root / sandbox._WORKSPACE_LEGACY_RECOVERY_DIR).glob("sqlite-maintenance-*"))
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / "app.txt").read_text(), "mirror")
            mirror.mkdir()

            sandbox._reconcile_workspace_copy(spec, label="restart", prefer_source=False)

            self.assertEqual((canonical / "app.txt").read_text(), "canonical")

    def test_archive_setup_failure_does_not_touch_baseline_or_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirror = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirror.mkdir(parents=True)
            (canonical / "app.txt").write_text("canonical")
            (mirror / "app.txt").write_text("mirror")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            baseline = sandbox._workspace_mirror_hash_path(spec)
            baseline_contents = baseline.read_bytes()

            with (
                mock.patch.object(sandbox, "_safe_legacy_archive_path", side_effect=OSError("setup unavailable")),
                self.assertRaisesRegex(OSError, "setup unavailable"),
            ):
                sandbox.archive_workspace_mirror(spec)

            self.assertEqual(baseline.read_bytes(), baseline_contents)
            self.assertTrue((mirror / "app.txt").is_file())

    def test_baseline_stash_failure_aborts_archive_before_mirror_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirror = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirror.mkdir(parents=True)
            (canonical / "app.txt").write_text("canonical")
            (mirror / "app.txt").write_text("mirror")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            baseline = sandbox._workspace_mirror_hash_path(spec)
            baseline_contents = baseline.read_bytes()
            replace = Path.replace

            def reject_baseline_stash(path: Path, target: Path) -> Path:
                if path == baseline:
                    raise OSError("stash unavailable")
                return replace(path, target)

            with mock.patch.object(Path, "replace", new=reject_baseline_stash), self.assertRaisesRegex(OSError, "stash unavailable"):
                sandbox.archive_workspace_mirror(spec)

            self.assertEqual(baseline.read_bytes(), baseline_contents)
            self.assertTrue(mirror.is_dir())
            self.assertTrue((mirror / "app.txt").is_file())
            self.assertEqual(list(baseline.parent.glob(f".{baseline.name}.retiring-*")), [])

    def test_archive_rename_failure_restores_baseline_and_does_not_resurrect_deleted_sqlite_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirror = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirror.mkdir(parents=True)
            database, wal = self._create_mirrored_sqlite_database(canonical, mirror)
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            baseline = sandbox._workspace_mirror_hash_path(spec)
            baseline_contents = baseline.read_bytes()
            rename = Path.rename
            database.unlink(missing_ok=True)
            wal.unlink(missing_ok=True)

            def reject_mirror_rename(path: Path, target: Path) -> Path:
                if path == mirror:
                    raise OSError("rename unavailable")
                return rename(path, target)

            with mock.patch.object(Path, "rename", new=reject_mirror_rename), self.assertRaisesRegex(OSError, "rename unavailable"):
                sandbox.archive_workspace_mirror(spec)

            self.assertEqual(baseline.read_bytes(), baseline_contents)
            sandbox._reconcile_workspace_copy(spec, label="restart", prefer_source=False)

            self.assertFalse(database.exists())
            self.assertFalse(wal.exists())
            self.assertFalse((mirror / ".ragtime" / "db" / "app.sqlite3").exists())
            self.assertFalse((mirror / ".ragtime" / "db" / "app.sqlite3-wal").exists())

    def test_archive_rollback_failure_propagates_and_keeps_stashed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "files"
            mirror = root / "rootfs" / "workspace"
            canonical.mkdir(parents=True)
            mirror.mkdir(parents=True)
            (canonical / "app.txt").write_text("canonical")
            (mirror / "app.txt").write_text("mirror")
            spec = sandbox.SandboxSpec(workspace_id="workspace", workspace_files_path=canonical, rootfs_path=root / "rootfs")
            sandbox._write_workspace_mirror_hashes(spec)
            baseline = sandbox._workspace_mirror_hash_path(spec)
            baseline_contents = baseline.read_bytes()
            replace = Path.replace
            replace_calls = 0

            def fail_rollback(path: Path, target: Path) -> Path:
                nonlocal replace_calls
                replace_calls += 1
                if replace_calls == 2:
                    raise OSError("rollback unavailable")
                return replace(path, target)

            with (
                mock.patch.object(Path, "rename", side_effect=OSError("rename unavailable")),
                mock.patch.object(Path, "replace", new=fail_rollback),
                self.assertRaisesRegex(OSError, "rollback unavailable"),
            ):
                sandbox.archive_workspace_mirror(spec)

            stashes = list(baseline.parent.glob(f".{baseline.name}.retiring-*"))
            self.assertFalse(baseline.exists())
            self.assertEqual(len(stashes), 1)
            self.assertEqual(stashes[0].read_bytes(), baseline_contents)
            self.assertTrue((mirror / "app.txt").is_file())
