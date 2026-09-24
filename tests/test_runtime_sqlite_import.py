"""Focused contracts for runtime-owned SQLite inspector imports."""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from runtime.worker.sqlite_history.inspector_import import RuntimeInspectorImport


class RuntimeInspectorImportTests(unittest.TestCase):
    def test_accepts_managed_sqlite_filenames_only(self) -> None:
        self.assertEqual(RuntimeInspectorImport._validate_name("app.sqlite3"), "app.sqlite3")

    def test_rejects_path_and_non_sqlite_filenames(self) -> None:
        for name in ("../app.sqlite3", "app.sqlite3/extra", "app.txt", ".sqlite3"):
            with self.subTest(name=name), self.assertRaises(HTTPException) as raised:
                RuntimeInspectorImport._validate_name(name)
            self.assertEqual(raised.exception.status_code, 400)
