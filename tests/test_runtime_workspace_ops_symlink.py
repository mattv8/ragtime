"""Tests for symlink-escape containment in resolve_workspace_mount_source_path."""

import os
import tempfile
import unittest
from pathlib import Path

import pytest

workspace_ops = pytest.importorskip("runtime.core.workspace_ops")
resolve_workspace_mount_source_path = workspace_ops.resolve_workspace_mount_source_path
_is_path_contained_under = workspace_ops._is_path_contained_under
list_workspace_tree_entries = workspace_ops.list_workspace_tree_entries
list_mount_source_tree_entries = workspace_ops.list_mount_source_tree_entries


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


class TreeEntrySymlinkVisibilityTests(unittest.TestCase):
    """Symlinked directories are by-design workspace mounts (cloud mounts,
    docker volumes, local paths) and must appear in the runtime tree UI."""

    def test_contained_symlinked_directory_and_contents_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            mount_src = files_dir / ".data" / "mount-src"
            (mount_src / "sub").mkdir(parents=True)
            (mount_src / "data.csv").write_text("a,b\n", encoding="utf-8")
            (mount_src / "sub" / "deep.txt").write_text("x", encoding="utf-8")
            (files_dir / "mounted").symlink_to(mount_src)

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path: entry.entry_type for entry in entries}

            self.assertEqual(paths.get("mounted"), "directory")
            self.assertEqual(paths.get("mounted/data.csv"), "file")
            self.assertEqual(paths.get("mounted/sub"), "directory")
            self.assertEqual(paths.get("mounted/sub/deep.txt"), "file")

    def test_contained_symlinked_file_is_listed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            files_dir.mkdir()
            target = files_dir / "real.txt"
            target.write_text("data", encoding="utf-8")
            (files_dir / "link.txt").symlink_to(target)

            entries = list_workspace_tree_entries(files_dir, include_dirs=False)
            paths = {entry.path for entry in entries}

            self.assertIn("link.txt", paths)
            self.assertIn("real.txt", paths)

    def test_symlink_escaping_walk_root_is_not_listed_or_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files_dir = root / "files"
            outside = root / "outside"
            files_dir.mkdir()
            outside.mkdir()
            (outside / "secret.txt").write_text("secret", encoding="utf-8")
            (files_dir / "escape-dir").symlink_to(outside)
            (files_dir / "escape.txt").symlink_to(outside / "secret.txt")
            (files_dir / "kept.txt").write_text("x", encoding="utf-8")

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path for entry in entries}

            self.assertIn("kept.txt", paths)
            self.assertNotIn("escape-dir", paths)
            self.assertNotIn("escape-dir/secret.txt", paths)
            self.assertNotIn("escape.txt", paths)

    def test_sibling_symlink_does_not_shadow_real_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            (files_dir / "docs").mkdir(parents=True)
            (files_dir / "docs" / "a.txt").write_text("x", encoding="utf-8")
            # Name sorts before "docs" so a naive global visited-set would
            # consume the inode first and leave the real directory empty.
            (files_dir / "alias").symlink_to(files_dir / "docs")

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path for entry in entries}

            self.assertIn("docs/a.txt", paths)
            self.assertIn("alias", paths)

    def test_symlink_loop_terminates_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            (files_dir / "child").mkdir(parents=True)
            (files_dir / "child" / "up").symlink_to(files_dir)
            (files_dir / "self").symlink_to(files_dir)

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = [entry.path for entry in entries]

            # Loop links appear as directory entries but are never descended.
            self.assertIn("child", paths)
            self.assertIn("child/up", paths)
            self.assertIn("self", paths)
            self.assertNotIn("child/up/child", paths)
            self.assertNotIn("self/child", paths)
            self.assertEqual(len(paths), len(set(paths)))

    def test_dangling_symlink_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            files_dir.mkdir()
            (files_dir / "gone").symlink_to(Path(tmpdir) / "missing")
            (files_dir / "kept.txt").write_text("x", encoding="utf-8")

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path for entry in entries}

            self.assertIn("kept.txt", paths)
            self.assertNotIn("gone", paths)

    def test_hidden_dirs_pruned_inside_symlinked_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            mount_src = files_dir / ".data" / "mount-src"
            (mount_src / "node_modules" / "pkg").mkdir(parents=True)
            (mount_src / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
            (mount_src / "src").mkdir()
            (mount_src / "src" / "app.ts").write_text("x", encoding="utf-8")
            (files_dir / "mounted").symlink_to(mount_src)

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path for entry in entries}

            self.assertIn("mounted/src/app.ts", paths)
            self.assertNotIn("mounted/node_modules", paths)
            self.assertNotIn("mounted/node_modules/pkg/index.js", paths)

    def test_duplicate_symlink_target_lists_both_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            files_dir = Path(tmpdir) / "files"
            (files_dir / "docs").mkdir(parents=True)
            (files_dir / "docs" / "a.txt").write_text("x", encoding="utf-8")
            (files_dir / "docs-link").symlink_to(files_dir / "docs")

            entries = list_workspace_tree_entries(files_dir, include_dirs=True)
            paths = {entry.path for entry in entries}

            # Both the canonical dir and the contained alias are browsable.
            self.assertIn("docs/a.txt", paths)
            self.assertIn("docs-link", paths)
            self.assertIn("docs-link/a.txt", paths)

    def test_mount_source_with_internal_symlinked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "volume"
            shared = source / "releases" / "v2"
            outside = root / "outside"
            (source / "reports").mkdir(parents=True)
            (source / "reports" / "q1.csv").write_text("x", encoding="utf-8")
            shared.mkdir(parents=True)
            (shared / "ref.txt").write_text("x", encoding="utf-8")
            outside.mkdir()
            (outside / "secret.txt").write_text("x", encoding="utf-8")
            # Contained internal symlink (common volume layout) stays visible;
            # a link escaping the mount source root is hidden, matching the
            # access-time containment in resolve_workspace_mount_source_path.
            (source / "current").symlink_to(shared)
            (source / "leak").symlink_to(outside)

            mounts = [{"target_path": "/workspace/data", "source_local_path": str(source)}]
            entries = list_mount_source_tree_entries(mounts, include_dirs=True)
            paths = {entry.path: entry.entry_type for entry in entries}

            self.assertEqual(paths.get("data"), "directory")
            self.assertEqual(paths.get("data/reports/q1.csv"), "file")
            self.assertEqual(paths.get("data/current"), "directory")
            self.assertEqual(paths.get("data/current/ref.txt"), "file")
            self.assertNotIn("data/leak", paths)
            self.assertNotIn("data/leak/secret.txt", paths)


if __name__ == "__main__":
    unittest.main()
