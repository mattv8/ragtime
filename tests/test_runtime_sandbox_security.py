import errno
import os
import tempfile
import unittest
from pathlib import Path

from runtime.worker import sandbox


class PinnedMountSecurityIntegrationTests(unittest.TestCase):
    def test_pinned_bind_remount_read_only_and_detach_preserve_underlay(self) -> None:
        """Requires a disposable container with SYS_ADMIN and unconfined mount policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temporary_root = Path(tmpdir)
            rootfs = temporary_root / "rootfs"
            source = temporary_root / "source"
            source.mkdir()
            (source / "visible.txt").write_text("mounted\n", encoding="utf-8")
            destination = rootfs / "mounted"
            destination.mkdir(parents=True)
            (destination / "underlay.txt").write_text("underlay\n", encoding="utf-8")

            try:
                with sandbox._pinned_directory(rootfs, ("mounted",), create=True) as target:
                    sandbox._syscall_mount(str(source), target, None, sandbox.MS_BIND | sandbox.MS_REC)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EPERM}:
                    self.skipTest(f"mount authority is unavailable: {exc}")
                raise
            try:
                with sandbox._pinned_directory(rootfs, ("mounted",)) as target:
                    sandbox._syscall_mount(
                        str(source),
                        target,
                        None,
                        sandbox.MS_BIND | sandbox.MS_REMOUNT | sandbox.MS_RDONLY | sandbox.MS_REC,
                    )
                    self.assertEqual((Path(target) / "visible.txt").read_text(encoding="utf-8"), "mounted\n")
                    with self.assertRaises(OSError) as blocked_write:
                        (Path(target) / "visible.txt").write_text("blocked\n", encoding="utf-8")
                    self.assertEqual(blocked_write.exception.errno, errno.EROFS)
            finally:
                sandbox._unmount_pinned_directory(rootfs, ("mounted",))

            self.assertEqual((destination / "underlay.txt").read_text(encoding="utf-8"), "underlay\n")

    def test_clear_pinned_existing_directory_uses_descriptor_relative_removal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            rootfs = Path(tmpdir) / "rootfs"
            target = rootfs / "workspace" / "stale"
            (target / "nested").mkdir(parents=True)
            (target / "nested" / "file.txt").write_text("stale\n", encoding="utf-8")

            sandbox._clear_pinned_directory(rootfs, ("workspace", "stale"))

            self.assertTrue((rootfs / "workspace").is_dir())
            self.assertFalse(target.exists())
