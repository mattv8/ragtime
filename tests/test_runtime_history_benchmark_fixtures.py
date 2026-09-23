"""Unit tests for SQLite benchmark fixture generators.

The fixture class is defined in scripts.sqlite_history_benchmark_fixtures
to keep it runtime-independent (no Ragtime imports).
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sqlite_history_benchmark_fixtures import SQLiteBenchmarkFixture


class BenchmarkFixtureTests(unittest.TestCase):
    """Verify fixture generator produces deterministic, sized databases."""

    def test_fixture_generation_is_deterministic(self) -> None:
        """Same seed + size produces identical SHA."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)

            path1 = tmpdir_path / "fixture1.sqlite3"
            path2 = tmpdir_path / "fixture2.sqlite3"

            sha1, size1 = fixture.create_fixture(path1)
            sha2, size2 = fixture.create_fixture(path2)

            self.assertEqual(sha1, sha2)
            self.assertEqual(size1, size2)
            self.assertGreater(size1, 1024 * 1024)  # At least 1 MiB

    def test_fixture_size_is_approximately_target(self) -> None:
        """Generated database is approximately the target size."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)
            path = tmpdir_path / "fixture.sqlite3"

            _, size = fixture.create_fixture(path)

            # Allow 10% variance due to SQLite overhead
            target = 1024 * 1024
            self.assertGreater(size, target * 0.9)
            self.assertLess(size, target * 1.1)

    def test_sparse_mutation_changes_content(self) -> None:
        """Sparse mutation produces a different SHA."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)
            path = tmpdir_path / "fixture.sqlite3"

            sha_original, _ = fixture.create_fixture(path)
            size_before = path.stat().st_size

            fixture.mutate_sparse(path, mutation_ratio=0.1)
            sha_mutated = hashlib.sha256(path.read_bytes()).hexdigest()
            size_after = path.stat().st_size

            self.assertNotEqual(sha_original, sha_mutated)
            # Size should remain approximately the same
            self.assertAlmostEqual(size_before, size_after, delta=size_before * 0.05)

    def test_append_mutation_increases_size(self) -> None:
        """Append mutation grows the database."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)
            path = tmpdir_path / "fixture.sqlite3"

            _, size_original = fixture.create_fixture(path)

            fixture.mutate_append(path, append_ratio=0.2)
            size_after = path.stat().st_size

            self.assertGreater(size_after, size_original)
            # Expect roughly 20% growth
            self.assertGreater(size_after, size_original * 1.15)
            self.assertLess(size_after, size_original * 1.3)

    def test_vacuum_compacts_database(self) -> None:
        """VACUUM may reduce database size after mutation."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)
            path = tmpdir_path / "fixture.sqlite3"

            _, _ = fixture.create_fixture(path)
            fixture.mutate_sparse(path, mutation_ratio=0.5)
            size_before_vacuum = path.stat().st_size

            fixture.mutate_vacuum(path)
            size_after_vacuum = path.stat().st_size

            # VACUUM may or may not reduce size, but it should be deterministic
            self.assertGreater(size_after_vacuum, 0)
            # Size change should be modest
            self.assertLess(abs(size_before_vacuum - size_after_vacuum), size_before_vacuum * 0.2)

    def test_wal_mode_creates_separate_files(self) -> None:
        """WAL mutation keeps its writer open so a capture sees WAL content."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fixture = SQLiteBenchmarkFixture(size_mib=1, seed=42)
            path = tmpdir_path / "fixture.sqlite3"

            _, _ = fixture.create_fixture(path)
            connection = fixture.mutate_wal_only(path)
            try:
                wal_path = tmpdir_path / "fixture.sqlite3-wal"
                self.assertTrue(wal_path.exists(), "WAL file should exist while writer is open")
                self.assertGreater(wal_path.stat().st_size, 0)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
