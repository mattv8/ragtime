#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _sorted_sql_files(migrations_dir: Path) -> list[Path]:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        return []
    return sorted(
        p
        for p in migrations_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".sql"
    )


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _ragtime_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _load_applied(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        "SELECT filename, checksum FROM _ragtime_migrations ORDER BY filename"
    ).fetchall()
    return {str(filename): str(checksum) for filename, checksum in rows}


def _apply_migration(
    conn: sqlite3.Connection,
    migration_path: Path,
    checksum: str,
) -> None:
    sql = migration_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO _ragtime_migrations (filename, checksum, applied_at) VALUES (?, ?, ?)",
        (
            migration_path.name,
            checksum,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def run(db_path: Path, migrations_dir: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migrations_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _ensure_meta_table(conn)
        applied = _load_applied(conn)

        for migration in _sorted_sql_files(migrations_dir):
            checksum = _checksum(migration)
            existing_checksum = applied.get(migration.name)
            if existing_checksum is not None:
                if existing_checksum != checksum:
                    raise RuntimeError(
                        "Detected modified applied migration "
                        f"'{migration.name}'. Create a new migration instead of editing history."
                    )
                continue

            with conn:
                _apply_migration(conn, migration, checksum)

        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply deterministic SQLite migrations."
    )
    parser.add_argument("--db", required=True, help="SQLite database file path")
    parser.add_argument(
        "--migrations",
        required=True,
        help="Directory containing lexically ordered .sql migration files",
    )
    args = parser.parse_args()
    return run(Path(args.db), Path(args.migrations))


if __name__ == "__main__":
    raise SystemExit(main())
