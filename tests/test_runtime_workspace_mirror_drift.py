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
