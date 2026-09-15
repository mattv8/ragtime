"""Linux/image integration coverage for copied-mount rsync materialization."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from runtime.worker.mount_sync import MountSyncCancelled, MountSyncError, mount_sync_available, sync_copied_mount


@unittest.skipUnless(sys.platform == "linux", "mount sync integration requires Linux")
class MountSyncIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        if not mount_sync_available():
            self.skipTest("rsync or Landlock confinement is unavailable")
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        self.source.mkdir()
        self.destination.mkdir()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _sync(self, *, protected_paths: tuple[str, ...] = (), timeout_seconds: float = 180.0) -> None:
        source_fd = os.open(self.source, os.O_RDONLY | os.O_DIRECTORY)
        destination_fd = os.open(self.destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            sync_copied_mount(
                source_fd,
                destination_fd,
                protected_paths=protected_paths,
                timeout_seconds=timeout_seconds,
            )
        finally:
            os.close(destination_fd)
            os.close(source_fd)

    def test_recursive_checksum_refresh_preserves_unchanged_identity_and_metadata(self) -> None:
        nested = self.source / "nested"
        nested.mkdir()
        changed = nested / "same-size.txt"
        changed.write_bytes(b"first-content")
        unchanged = nested / "unchanged.txt"
        unchanged.write_bytes(b"unchanged-content")
        os.chmod(unchanged, 0o640)
        timestamp_ns = 1_700_000_000_123_456_789
        os.utime(unchanged, ns=(timestamp_ns, timestamp_ns))
        self._set_user_xattr_if_available(unchanged)

        self._sync()
        copied_unchanged = self.destination / "nested" / "unchanged.txt"
        copied_changed = self.destination / "nested" / "same-size.txt"
        copied_inode = copied_unchanged.stat().st_ino
        self.assertNotEqual(copied_inode, unchanged.stat().st_ino)
        self.assertEqual(stat.S_IMODE(copied_unchanged.stat().st_mode), 0o640)
        self.assertEqual(copied_unchanged.stat().st_mtime_ns, timestamp_ns)
        self._assert_user_xattr_parity(unchanged, copied_unchanged)

        original_changed_mtime = changed.stat().st_mtime_ns
        changed.write_bytes(b"other-content")
        os.utime(changed, ns=(original_changed_mtime, original_changed_mtime))
        self._sync()

        self.assertEqual(copied_changed.read_bytes(), b"other-content")
        self.assertEqual(copied_unchanged.stat().st_ino, copied_inode)

    def test_symlinks_deletions_type_changes_and_literal_nested_protection(self) -> None:
        (self.source / "empty").mkdir()
        (self.source / "entry").write_text("file", encoding="utf-8")
        os.symlink("missing-target", self.source / "dangling")
        os.symlink("/outside/not-followed", self.source / "external")
        literal_protected = "nested[keep]*"
        (self.destination / "stale.txt").write_text("stale", encoding="utf-8")
        (self.destination / literal_protected).mkdir()
        (self.destination / literal_protected / "child.txt").write_text("child", encoding="utf-8")
        self._sync(protected_paths=(literal_protected,))

        self.assertFalse((self.destination / "stale.txt").exists())
        self.assertTrue((self.destination / "empty").is_dir())
        self.assertEqual(os.readlink(self.destination / "dangling"), "missing-target")
        self.assertEqual(os.readlink(self.destination / "external"), "/outside/not-followed")
        self.assertEqual((self.destination / literal_protected / "child.txt").read_text(encoding="utf-8"), "child")

        (self.source / "entry").unlink()
        (self.source / "entry").mkdir()
        (self.source / "entry" / "child").write_text("directory", encoding="utf-8")
        (self.source / "dangling").unlink()
        (self.source / "dangling").write_text("regular", encoding="utf-8")
        self._sync(protected_paths=(literal_protected,))

        self.assertTrue((self.destination / "entry").is_dir())
        self.assertEqual((self.destination / "entry" / "child").read_text(encoding="utf-8"), "directory")
        self.assertTrue((self.destination / "dangling").is_file())

    def test_destination_symlink_cannot_redirect_content_outside_target(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "sentinel.txt"
        sentinel.write_text("untouched", encoding="utf-8")
        (self.source / "redirect").mkdir()
        (self.source / "redirect" / "inside.txt").write_text("inside", encoding="utf-8")
        os.symlink(outside, self.destination / "redirect")

        self._sync()

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")
        self.assertEqual((self.destination / "redirect" / "inside.txt").read_text(encoding="utf-8"), "inside")
        self.assertEqual((self.source / "redirect" / "inside.txt").read_text(encoding="utf-8"), "inside")

    def test_cancel_and_timeout_return_visible_errors(self) -> None:
        payload = self.source / "large.bin"
        payload.write_bytes(b"x" * (32 * 1024 * 1024))
        with self.assertRaises(MountSyncError):
            self._sync(timeout_seconds=0.0001)

        cancellation = threading.Event()
        cancellation.set()
        source_fd = os.open(self.source, os.O_RDONLY | os.O_DIRECTORY)
        destination_fd = os.open(self.destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with self.assertRaises(MountSyncCancelled):
                sync_copied_mount(source_fd, destination_fd, cancel_event=cancellation)
        finally:
            os.close(destination_fd)
            os.close(source_fd)

    def _set_user_xattr_if_available(self, path: Path) -> None:
        if not hasattr(os, "setxattr"):
            self.skipTest("Python xattr support is unavailable")
        try:
            os.setxattr(path, "user.ragtime-test", b"metadata")
        except OSError as exc:
            self.skipTest(f"user xattrs are unavailable: {exc}")

    def _assert_user_xattr_parity(self, source: Path, destination: Path) -> None:
        self.assertEqual(os.getxattr(destination, "user.ragtime-test"), os.getxattr(source, "user.ragtime-test"))


if __name__ == "__main__":
    unittest.main()
