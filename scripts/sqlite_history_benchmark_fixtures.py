"""Deterministic reusable fixture generators for SQLite history benchmarking.

This module is runtime-independent (no Ragtime imports) and can be used by
both test suites and standalone benchmark scripts.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class SQLiteBenchmarkFixture:
    """Generate deterministic SQLite database fixtures for benchmarking."""

    def __init__(self, size_mib: int, seed: int = 42) -> None:
        """Initialize fixture generator.

        Args:
            size_mib: Target database size in MiB (1, 64, 256 typical).
            seed: Deterministic seed for payload generation.
        """
        self.size_mib = size_mib
        self.seed = seed
        self.target_bytes = size_mib * 1024 * 1024

    def _deterministic_payload(self, offset: int, *, phase: str = "original") -> bytes:
        """Return one deterministic, high-entropy, phase-separated payload.

        The counter is expanded through SHA-256 rather than repeating one digest.
        Size, seed, and mutation phase are all part of the domain separator, so a
        mutation cannot reuse an original payload merely by choosing an offset.
        """
        identity = (f"sqlite-history-benchmark-v1\0{self.seed}\0{self.size_mib}\0{phase}\0{offset}\0").encode()
        return b"".join(hashlib.sha256(identity + block.to_bytes(8, "little")).digest() for block in range(8192 // hashlib.sha256().digest_size))

    def create_fixture(self, path: Path) -> tuple[str, int]:
        """Create a deterministic database at the target size.

        Returns:
            (sha256_hexdigest, size_bytes) of the created database.
        """
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA synchronous=OFF")  # Speed up fixture creation
            conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
            conn.execute("CREATE INDEX idx_records_id ON records(id)")

            # Fill with deterministic payloads
            bytes_written = 0
            offset = 0
            row_id = 1
            while bytes_written < self.target_bytes:
                payload = self._deterministic_payload(offset)
                conn.execute("INSERT INTO records(payload) VALUES (?)", (payload,))
                bytes_written += len(payload)
                row_id += 1
                offset += 1

            # Commit to flush to disk
            conn.commit()
            # Checkpoint and switch to DELETE journal mode for stable format
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Calculate hash
        sha256 = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)

        size_bytes = path.stat().st_size
        return sha256.hexdigest(), size_bytes

    def mutate_sparse(self, path: Path, mutation_ratio: float = 0.01) -> None:
        """Apply sparse mutations: update ~mutation_ratio of rows."""
        with sqlite3.connect(path) as conn:
            # Get total row count
            (total_rows,) = conn.execute("SELECT COUNT(*) FROM records").fetchone()  # type: ignore
            mutations = max(1, int(total_rows * mutation_ratio))

            # Update rows at deterministic offsets
            for i in range(mutations):
                row_id = (i * (total_rows // mutations)) + 1
                new_payload = self._deterministic_payload(offset=i, phase="sparse")
                conn.execute("UPDATE records SET payload = ? WHERE id = ?", (new_payload, row_id))

            conn.commit()
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def mutate_append(self, path: Path, append_ratio: float = 0.1) -> None:
        """Append new rows (~append_ratio of original size)."""
        with sqlite3.connect(path) as conn:
            # Get current max ID
            (max_id,) = conn.execute("SELECT MAX(id) FROM records").fetchone() or (0,)
            max_id = max_id or 0

            # Calculate target byte growth
            bytes_before = path.stat().st_size
            target_growth_bytes = max(self.target_bytes // 10, int(bytes_before * append_ratio))

            bytes_added = 0
            new_id = max_id + 1

            while bytes_added < target_growth_bytes:
                payload = self._deterministic_payload(offset=new_id, phase="append")
                conn.execute("INSERT INTO records(payload) VALUES (?)", (payload,))
                bytes_added += len(payload)
                new_id += 1

            conn.commit()
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def mutate_wal_only(self, path: Path) -> sqlite3.Connection:
        """Apply uncheckpointed WAL writes and return the open writer connection.

        The caller must keep this connection open until after the online capture;
        closing the final WAL connection may checkpoint/remove the sidecar.
        """
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        (max_id,) = conn.execute("SELECT MAX(id) FROM records").fetchone() or (0,)
        max_id = max_id or 0
        payload = self._deterministic_payload(offset=max_id, phase="wal")
        conn.execute("INSERT INTO records(payload) VALUES (?)", (payload,))
        conn.commit()
        return conn

    def mutate_vacuum(self, path: Path) -> None:
        """Apply VACUUM to reorganize the database."""
        with sqlite3.connect(path) as conn:
            conn.execute("VACUUM")
