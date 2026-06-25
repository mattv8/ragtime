"""Tests for symlink-escape containment in resolve_workspace_mount_source_path."""

import os
import tempfile
import unittest
from pathlib import Path

import pytest

workspace_ops = pytest.importorskip("runtime.core.workspace_ops")
resolve_workspace_mount_source_path = workspace_ops.resolve_workspace_mount_source_path
_is_path_contained_under = workspace_ops._is_path_contained_under


def _mount(target: str, source: str, *, read_only: bool = False) -> dict:
    return {
        "target_path": target,
        "source_local_path": source,
        "read_only": read_only,
    }


class IsPathContainedUnderTests(unittest.TestCase):
    def test_direct_child_is_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            child = parent / "subdir" / "file.txt"
            child.parent.mkdir(parents=True, exist_ok=True)
            child.touch()
            self.assertTrue(_is_path_contained_under(child, parent))

    def test_parent_itself_is_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            self.assertTrue(_is_path_contained_under(parent, parent))

    def test_sibling_directory_is_not_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            sibling = root / "sibling"
            source.mkdir()
            sibling.mkdir()
            self.assertFalse(_is_path_contained_under(sibling, source))

    def test_symlink_within_root_is_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            target_dir = root / "target"
            source.mkdir()
            target_dir.mkdir()
            real_file = target_dir / "data.txt"
            real_file.write_text("data", encoding="utf-8")
            # Symlink inside source pointing to a file also inside root/target
            link = source / "link.txt"
            link.symlink_to(real_file)
            # The symlink resolves to root/target/data.txt which is NOT under source
            self.assertFalse(_is_path_contained_under(link, source))

    def test_symlink_pointing_outside_root_is_not_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            link = source / "escape.txt"
            link.symlink_to(secret)
            self.assertFalse(_is_path_contained_under(link, source))

    def test_symlink_pointing_to_parent_is_not_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            link = source / "up"
            link.symlink_to(root)
            self.assertFalse(_is_path_contained_under(link, source))

    def test_symlink_loop_is_not_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            first = source / "first"
            second = source / "second"
            first.symlink_to(second)
            second.symlink_to(first)
            self.assertFalse(_is_path_contained_under(first, source))

    def test_nonexistent_path_under_root_is_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir)
            nonexistent = parent / "new_dir" / "new_file.txt"
            # Path does not exist yet — should still be considered contained
            self.assertTrue(_is_path_contained_under(nonexistent, parent))

    def test_nonexistent_path_outside_root_is_not_contained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            outside_nonexistent = root / "other" / "file.txt"
            self.assertFalse(_is_path_contained_under(outside_nonexistent, source))


class ResolveMountSourcePathSymlinkTests(unittest.TestCase):
    def test_normal_path_resolves_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            real_file = source / "report.csv"
            real_file.write_text("data", encoding="utf-8")

            mounts = [_mount("/workspace/data", str(source))]
            result = resolve_workspace_mount_source_path(mounts, "data/report.csv")

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[0].resolve(), real_file.resolve())

    def test_symlink_escaping_mount_root_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            # Plant a symlink inside the mount source that points outside
            link = source / "escape.txt"
            link.symlink_to(secret)

            mounts = [_mount("/workspace/data", str(source))]
            result = resolve_workspace_mount_source_path(mounts, "data/escape.txt")

            self.assertIsNone(result)

    def test_symlink_directory_escaping_mount_root_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            (outside / "file.txt").write_text("secret", encoding="utf-8")
            # Symlink directory inside source pointing to outside dir
            link_dir = source / "linked_dir"
            link_dir.symlink_to(outside)

            mounts = [_mount("/workspace/data", str(source))]
            result = resolve_workspace_mount_source_path(mounts, "data/linked_dir/file.txt")

            self.assertIsNone(result)

    def test_nonexistent_path_within_root_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()

            mounts = [_mount("/workspace/data", str(source))]
            result = resolve_workspace_mount_source_path(mounts, "data/new_file.txt")

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[0], source / "new_file.txt")

    def test_deepest_prefix_wins_and_containment_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_shallow = root / "shallow"
            source_deep = root / "deep"
            outside = root / "outside"
            source_shallow.mkdir()
            source_deep.mkdir()
            outside.mkdir()
            secret = outside / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            # Plant escape symlink in the deeper mount
            link = source_deep / "escape.txt"
            link.symlink_to(secret)

            mounts = [
                _mount("/workspace/data", str(source_shallow)),
                _mount("/workspace/data/sub", str(source_deep)),
            ]
            result = resolve_workspace_mount_source_path(mounts, "data/sub/escape.txt")

            self.assertIsNone(result)

    def test_read_only_flag_preserved_for_safe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "file.txt").write_text("data", encoding="utf-8")

            mounts = [_mount("/workspace/data", str(source), read_only=True)]
            result = resolve_workspace_mount_source_path(mounts, "data/file.txt")

            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result[1])

    def test_no_mounts_returns_none(self) -> None:
        result = resolve_workspace_mount_source_path([], "data/file.txt")
        self.assertIsNone(result)

    def test_unmatched_path_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            mounts = [_mount("/workspace/data", str(source))]
            result = resolve_workspace_mount_source_path(mounts, "other/file.txt")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
