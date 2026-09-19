#!/usr/bin/env python3
"""Apply numbered SQL migrations to a workspace SQLite database.

This runner treats SQL files under the migrations directory as append-only history.
Each file checksum is recorded in `_ragtime_migrations`; changing an applied file
raises an error to avoid silent drift.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_migration_files(migrations_dir: Path) -> list[Path]:
    if not migrations_dir.exists() or not migrations_dir.is_dir():
        return []
    return sorted(path for path in migrations_dir.iterdir() if path.is_file() and path.suffix.lower() == ".sql")


def _ensure_tracking_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _ragtime_migrations (
            filename TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _get_applied_checksum(conn: sqlite3.Connection, filename: str) -> str | None:
    row = conn.execute(
        "SELECT checksum FROM _ragtime_migrations WHERE filename = ?",
        (filename,),
    ).fetchone()
    if not row:
        return None
    return str(row[0])


def _record_applied(conn: sqlite3.Connection, filename: str, checksum: str) -> None:
    conn.execute(
        """
        INSERT INTO _ragtime_migrations (filename, checksum, applied_at)
        VALUES (?, ?, ?)
        """,
        (filename, checksum, datetime.now(timezone.utc).isoformat()),
    )


def _leading_sql_token(statement: str) -> str | None:
    index = 0
    while index < len(statement):
        if statement[index].isspace() or statement[index] == ";":
            index += 1
        elif statement.startswith("--", index):
            newline = statement.find("\n", index + 2)
            index = len(statement) if newline < 0 else newline + 1
        elif statement.startswith("/*", index):
            end = statement.find("*/", index + 2)
            if end < 0:
                raise RuntimeError("Migration contains an unterminated SQL comment")
            index = end + 2
        else:
            end = index
            while end < len(statement) and (statement[end].isalnum() or statement[end] == "_"):
                end += 1
            return statement[index:end].upper() or None
    return None


def _statements(sql: str) -> list[str]:
    """Split SQLite statements at complete semicolon boundaries."""
    statements: list[str] = []
    start = 0
    for index, char in enumerate(sql):
        if char != ";":
            continue
        candidate = sql[start : index + 1]
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement:
                statements.append(statement)
            start = index + 1
    trailing = sql[start:]
    if trailing.strip() and _leading_sql_token(trailing) is not None:
        raise RuntimeError("Migration contains an incomplete SQL statement")
    return statements


def apply_migrations(db_path: Path, migrations_dir: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    migration_files = _iter_migration_files(migrations_dir)

    if not migration_files:
        print(f"No SQL migrations found in {migrations_dir}")
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_tracking_table(conn)
        applied = 0
        skipped = 0

        for migration_file in migration_files:
            sql_text = migration_file.read_text(encoding="utf-8")
            checksum = _sha256_text(sql_text)
            filename = migration_file.name

            existing_checksum = _get_applied_checksum(conn, filename)
            if existing_checksum is not None:
                if existing_checksum != checksum:
                    raise RuntimeError(f"Applied migration '{filename}' was modified. Create a new migration file instead of editing history.")
                skipped += 1
                continue

            try:
                # executescript commits any pending transaction before running;
                # execute complete statements instead so SQL and ledger are atomic.
                statements = _statements(sql_text)
                conn.execute("BEGIN")
                for statement in statements:
                    conn.execute(statement)
                _record_applied(conn, filename, checksum)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            applied += 1
            print(f"Applied migration: {filename}")

        print(f"SQLite migration complete: applied={applied}, skipped={skipped}, total={len(migration_files)}")
        return applied
    finally:
        conn.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply workspace SQLite migrations")
    parser.add_argument("--db", required=True, help="Path to SQLite database file")
    parser.add_argument(
        "--migrations",
        required=True,
        help="Directory containing numbered .sql migration files",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    db_path = Path(args.db)
    migrations_dir = Path(args.migrations)
    apply_migrations(db_path, migrations_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
