from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from ragtime.core import workspace_ops as ragtime_ops
from runtime.core import workspace_ops as runtime_ops


class WorkspaceOpsContractTests(unittest.TestCase):
    def test_ragtime_facade_reexports_runtime_implementations(self) -> None:
        exported_names = (
            "normalize_relative_file_path",
            "enforce_sqlite_managed_path",
            "normalize_runtime_file_path",
            "compute_file_hash",
            "workspace_mount_target_repo_relative_path",
            "workspace_path_matches_mount_prefix",
            "deduplicate_ancestor_paths",
            "sync_scope_relative_paths",
        )
        for name in exported_names:
            self.assertIs(getattr(ragtime_ops, name), getattr(runtime_ops, name))
        self.assertIs(ragtime_ops.SQLITE_FILE_EXTENSIONS, runtime_ops.SQLITE_FILE_EXTENSIONS)
        self.assertEqual(ragtime_ops.PLATFORM_MANAGED_GITIGNORE_PATTERNS, runtime_ops.PLATFORM_MANAGED_GITIGNORE_PATTERNS)

    def test_normalizes_paths_and_rejects_traversal_and_reserved_paths(self) -> None:
        self.assertEqual(ragtime_ops.normalize_relative_file_path(r" /docs\\readme.md "), "docs/readme.md")
        for path in ("", ".", "../secret", "docs/../../secret"):
            with self.subTest(path=path), self.assertRaises(HTTPException) as error:
                ragtime_ops.normalize_relative_file_path(path)
            self.assertEqual(error.exception.status_code, 400)
            self.assertEqual(error.exception.detail, "Invalid file path")
        with self.assertRaises(HTTPException) as error:
            ragtime_ops.normalize_runtime_file_path(".ragtime/config.json", check_reserved=True)
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Invalid file path")
        with self.assertRaises(HTTPException) as error:
            ragtime_ops.normalize_runtime_file_path("blocked.txt", is_reserved_path=lambda path: path == "blocked.txt")
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(error.exception.detail, "Invalid file path")

    def test_enforces_sqlite_location_and_preserves_allowed_suffixes(self) -> None:
        self.assertEqual(
            ragtime_ops.normalize_runtime_file_path(".ragtime/db/app.SQLITE3", enforce_sqlite_managed=True),
            ".ragtime/db/app.SQLITE3",
        )
        self.assertEqual(ragtime_ops.normalize_runtime_file_path("notes.db-journal", enforce_sqlite_managed=True), "notes.db-journal")
        with self.assertRaises(HTTPException) as error:
            ragtime_ops.normalize_runtime_file_path("data/app.db", enforce_sqlite_managed=True)
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(
            error.exception.detail,
            "SQLite persistence files must be managed under .ragtime/db/. Use paths like .ragtime/db/app.sqlite3.",
        )

    def test_hash_mount_and_ancestor_helpers_preserve_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "payload.txt"
            file_path.write_bytes(b"workspace contract")
            self.assertEqual(ragtime_ops.compute_file_hash(file_path), hashlib.sha256(b"workspace contract").hexdigest())
        self.assertEqual(ragtime_ops.workspace_mount_target_repo_relative_path(" /workspace/data/reports "), "data/reports")
        self.assertIsNone(ragtime_ops.workspace_mount_target_repo_relative_path("/workspace/../secret"))
        self.assertTrue(ragtime_ops.workspace_path_matches_mount_prefix("data/reports/q1.csv", "data/reports"))
        self.assertFalse(ragtime_ops.workspace_path_matches_mount_prefix("data/reporting/q1.csv", "data/reports"))
        self.assertEqual(ragtime_ops.deduplicate_ancestor_paths(["src/app", "src", "docs", "src/app"]), ["docs", "src"])

    def test_sync_scope_excludes_git_and_exact_ignored_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / ".git").mkdir()
            (root / "src" / "app.py").write_text("ok", encoding="utf-8")
            (root / "ignored.txt").write_text("ignore", encoding="utf-8")
            (root / ".git" / "config").write_text("hidden", encoding="utf-8")
            paths = ragtime_ops.sync_scope_relative_paths(root, ignored_relative_paths={"ignored.txt"})
        self.assertEqual(set(paths), {"src/app.py"})
        self.assertEqual(paths["src/app.py"].name, "app.py")
