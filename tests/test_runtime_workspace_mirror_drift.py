from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime.worker import sandbox


class RuntimeWorkspaceMirrorDriftTests(unittest.TestCase):
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
