from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime.core import sqlite_recovery
from runtime.core.sqlite_recovery import SqliteRecoveryError, capture_database


class SqliteCaptureLightweightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.sqlite3"
        with sqlite3.connect(self.source) as connection:
            connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO item VALUES (1, 'captured')")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_lightweight_capture_omits_logical_fingerprint_but_keeps_valid_backup(self) -> None:
        """Scheduled capture must avoid the expensive logical row scan without weakening output validity."""
        destination = self.root / "lightweight.sqlite3"
        with mock.patch("runtime.core.sqlite_recovery.database_fingerprint", side_effect=AssertionError("logical scan")):
            result = capture_database(self.source, destination, include_fingerprint=False)
        self.assertNotIn("fingerprint", result)
        self.assertTrue(result["sha256"])
        with sqlite3.connect(destination) as connection:
            self.assertEqual(connection.execute("SELECT value FROM item").fetchone()[0], "captured")

    def test_default_capture_retains_logical_fingerprint_contract(self) -> None:
        """Restore callers still receive a logical fingerprint unless opting out."""
        result = capture_database(self.source, self.root / "default.sqlite3")
        self.assertEqual(len(result["fingerprint"]), 64)

    def test_lightweight_capture_skips_redundant_pre_normalization_integrity_scan(self) -> None:
        """The lightweight path must retain final validation while avoiding duplicate validation."""
        with mock.patch("runtime.core.sqlite_recovery._check_capture_database", wraps=sqlite_recovery._check_capture_database) as check:
            capture_database(self.source, self.root / "one-check.sqlite3", include_fingerprint=False)
        self.assertEqual(check.call_count, 1)

    def test_caller_owned_source_connection_remains_open_without_a_transaction(self) -> None:
        """Warmed capture callers retain their connection for post-capture source validation."""
        source_connection = sqlite_recovery._connect_readonly(self.source)
        try:
            source_connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
            capture_database(self.source, self.root / "caller-owned.sqlite3", include_fingerprint=False, source_connection=source_connection)
            self.assertFalse(source_connection.in_transaction)
            self.assertEqual(source_connection.execute("SELECT value FROM item").fetchone()[0], "captured")
        finally:
            source_connection.close()

    def test_capture_rejects_an_existing_caller_transaction_without_rolling_it_back(self) -> None:
        """Engine rollback must be limited to the transaction it started."""
        source_connection = sqlite_recovery._connect_readonly(self.source)
        try:
            source_connection.execute("BEGIN")
            with self.assertRaises(SqliteRecoveryError):
                capture_database(self.source, self.root / "rejected.sqlite3", source_connection=source_connection)
            self.assertTrue(source_connection.in_transaction)
        finally:
            source_connection.rollback()
            source_connection.close()
