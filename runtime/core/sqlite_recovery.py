"""Safe, stdlib-only SQLite capture and disposable restore preparation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote


class SqliteRecoveryError(ValueError):
    pass


_LEDGER = "_ragtime_migrations"
_INTERNAL_TABLES = {_LEDGER, "sqlite_sequence"}
_FORBIDDEN_MIGRATION_TOKENS = frozenset({"BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "PRAGMA", "ATTACH", "DETACH"})
_MAX_PREVIEW_STRING_CHARS = 1_024
_MAX_PREVIEW_BLOB_BASE64_CHARS = 1_024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _durable_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    # Callers that need a pinned descriptor capability may pass /proc/self/fd
    # paths.  Resolving here would turn that capability back into a mutable name.
    return sqlite3.connect(f"file:{quote(os.fspath(path), safe='/:')}?mode=ro", uri=True, timeout=5)


def _check_database(conn: sqlite3.Connection) -> None:
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        raise SqliteRecoveryError(f"SQLite integrity check failed: {integrity[0] if integrity else 'no result'}")
    foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        raise SqliteRecoveryError("SQLite foreign key check failed")


def _check_capture_database(conn: sqlite3.Connection) -> None:
    """Check physical structure without rejecting a pre-existing FK violation."""
    integrity = conn.execute("PRAGMA integrity_check").fetchone()
    if not integrity or integrity[0] != "ok":
        raise SqliteRecoveryError(f"SQLite integrity check failed: {integrity[0] if integrity else 'no result'}")


def _normalize_single_file_output(conn: sqlite3.Connection) -> None:
    """Make publication independent of SQLite sidecar lifetime."""
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE").fetchone()


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$blob": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return {"$float": repr(value)}
    return value


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != ? ORDER BY name", (_LEDGER,))
    ]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_tokens(sql: str) -> tuple[str, ...]:
    """Return a formatting-insensitive SQLite DDL representation.

    SQLite has no structured pragma for CHECK expressions, collations, or
    trigger bodies.  A small lexical representation keeps those semantics
    while making keyword case, comments, and insignificant whitespace irrelevant.
    """
    tokens: list[str] = []
    index = 0
    while index < len(sql):
        char = sql[index]
        if char.isspace():
            index += 1
        elif sql.startswith("--", index):
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline < 0 else newline + 1
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                raise SqliteRecoveryError("Unterminated SQL comment in schema")
            index = end + 2
        elif char in "'\"`":
            quote = char
            end = index + 1
            while end < len(sql):
                if sql[end] == quote:
                    if end + 1 < len(sql) and sql[end + 1] == quote:
                        end += 2
                        continue
                    end += 1
                    break
                end += 1
            if end > len(sql) or sql[end - 1] != quote:
                raise SqliteRecoveryError("Unterminated SQL string in schema")
            tokens.append(sql[index:end] if quote == "'" else sql[index:end].lower())
            index = end
        elif char in "[]":
            if char == "[":
                end = sql.find("]", index + 1)
                if end < 0:
                    raise SqliteRecoveryError("Unterminated quoted identifier in schema")
                tokens.append(sql[index : end + 1].lower())
                index = end + 1
            else:
                tokens.append(char)
                index += 1
        elif char.isalnum() or char in "_$.":
            end = index + 1
            while end < len(sql) and (sql[end].isalnum() or sql[end] in "_$."):
                end += 1
            tokens.append(sql[index:end].lower())
            index = end
        else:
            tokens.append(char)
            index += 1
    return tuple(tokens)


def _schema_shape(conn: sqlite3.Connection) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name in _table_names(conn):
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0] or ""
        columns = [tuple(row) for row in conn.execute(f"PRAGMA table_xinfo({_quote(name)})")]
        indexes = []
        for index in conn.execute(f"PRAGMA index_list({_quote(name)})"):
            index_name = index[1]
            index_columns = []
            for row in conn.execute(f"PRAGMA index_xinfo({_quote(index_name)})"):
                values = list(row)
                # Collation identifiers are case-insensitive in SQLite, but
                # PRAGMA preserves their spelling from the source DDL.
                if isinstance(values[4], str):
                    values[4] = values[4].lower()
                index_columns.append(tuple(values))
            indexes.append((tuple(index), index_columns))
        triggers = [
            _sql_tokens(row[0] or "") for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=? ORDER BY name", (name,))
        ]
        tables[name] = {
            "columns": columns,
            "indexes": indexes,
            "triggers": triggers,
            "virtual": bool(re.match(r"\s*CREATE\s+VIRTUAL\s+TABLE\b", sql, re.I)),
            "definition": _sql_tokens(sql),
        }
    return tables


def _schema_hash(conn: sqlite3.Connection) -> str:
    encoded = json.dumps(_schema_shape(conn), sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def database_fingerprint(path: Path) -> str:
    """Hash database schema and logical rows using streaming deterministic SHA256.

    Uses framed typed values with SQL-driven deterministic ordering.
    Text columns are not cast to BLOB in SELECT to preserve invalid UTF-8.
    Visible generated columns (xinfo row[6] in 0,2,3) are included.
    Shadow tables are excluded.
    """
    digest = hashlib.sha256()

    def _frame(payload: bytes) -> None:
        """Frame a payload: 8-byte big-endian length + payload."""
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    with _connect_readonly(path) as conn:
        # Version prefix
        digest.update(b"v1")

        # Frame schema
        schema = _schema_shape(conn)
        schema_json = json.dumps(schema, sort_keys=True, default=str, separators=(",", ":")).encode()
        _frame(schema_json)

        # Identify shadow tables
        shadow_names = {row[1] for row in conn.execute("PRAGMA table_list") if row[2] == "shadow"}

        # Stream rows per non-shadow table
        for table in _table_names(conn):
            if table in shadow_names:
                continue

            # Get visible columns: xinfo row[6] in (0, 2, 3)
            xinfo = list(conn.execute(f"PRAGMA table_xinfo({_quote(table)})"))
            visible_columns = [row[1] for row in xinfo if row[6] in (0, 2, 3)]

            if not visible_columns:
                continue

            # Frame table name
            _frame(table.encode("utf-8"))

            # Build SELECT with pairs: typeof(col), CASE WHEN ...
            select_parts = []
            for col in visible_columns:
                select_parts.append(f"typeof({_quote(col)})")
                select_parts.append(f"CASE WHEN typeof({_quote(col)})='text' THEN CAST({_quote(col)} AS BLOB) ELSE {_quote(col)} END")

            query = (
                f"SELECT {', '.join(select_parts)} FROM {_quote(table)} "
                f"ORDER BY {', '.join(f'typeof({_quote(col)}), {_quote(col)} COLLATE BINARY' for col in visible_columns)}"
            )

            for row in conn.execute(query):
                digest.update(b"R")

                # Process pairs: (type_string, value)
                for i in range(0, len(row), 2):
                    type_str = row[i]
                    value = row[i + 1]

                    # Frame type as ASCII
                    _frame(type_str.encode("ascii"))

                    # Frame value
                    if value is None:
                        _frame(b"")
                    elif isinstance(value, int):
                        _frame(str(value).encode("ascii"))
                    elif isinstance(value, float):
                        _frame(struct.pack(">d", value))
                    elif isinstance(value, bytes):
                        _frame(value)
                    elif isinstance(value, str):
                        # Text from CAST(col AS BLOB) in CASE returns bytes
                        _frame(value.encode("utf-8"))
                    else:
                        _frame(repr(value).encode("utf-8"))

            digest.update(b"E")

    return digest.hexdigest()


def migration_fingerprint(migrations_dir: Path) -> str:
    items = []
    if migrations_dir.is_dir():
        for path in sorted(migrations_dir.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() == ".sql":
                text = path.read_text(encoding="utf-8")
                items.append((path.name, hashlib.sha256(text.encode()).hexdigest()))
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def capture_database(
    source: Path,
    destination: Path,
    *,
    timeout_seconds: float = 60,
    include_fingerprint: bool = True,
    source_connection: sqlite3.Connection | None = None,
) -> dict:
    if timeout_seconds <= 0:
        raise SqliteRecoveryError("Capture timeout must be positive")
    if not source.is_file():
        raise SqliteRecoveryError("SQLite source database does not exist")
    if destination.exists():
        raise SqliteRecoveryError("Capture destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(temp_fd)
    temp = Path(temp_name)
    temp.unlink()
    started = time.monotonic()
    try:
        owns_source_connection = source_connection is None
        source_conn = source_connection or _connect_readonly(source)
        capture_transaction_started = False
        try:
            if source_conn.in_transaction:
                raise SqliteRecoveryError("Caller-owned SQLite source connection has an active transaction")
            source_conn.execute("BEGIN")  # Establish a WAL read snapshot before backup.
            capture_transaction_started = True
            try:
                source_conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
                output_conn = sqlite3.connect(temp)
                try:

                    def progress(_: int, __: int, ___: int) -> None:
                        if time.monotonic() - started > timeout_seconds:
                            raise SqliteRecoveryError("SQLite capture timed out")

                    source_conn.backup(output_conn, pages=128, progress=progress, sleep=0.001)
                    if include_fingerprint:
                        _check_capture_database(output_conn)
                    # A single-file artifact must not depend on a journal or WAL
                    # whose lifetime would otherwise end with connection GC.
                    _normalize_single_file_output(output_conn)
                    _check_capture_database(output_conn)
                finally:
                    output_conn.close()
            finally:
                if capture_transaction_started:
                    source_conn.rollback()
        finally:
            if owns_source_connection:
                source_conn.close()
        _durable_file(temp)
        os.replace(temp, destination)
        _durable_file(destination)
        with _connect_readonly(destination) as conn:
            schema_hash = _schema_hash(conn)
        result = {"sha256": _sha256_file(destination), "size_bytes": destination.stat().st_size, "schema_hash": schema_hash}
        if include_fingerprint:
            result["fingerprint"] = database_fingerprint(destination)
        return result
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _migration_files(directory: Path) -> dict[str, tuple[str, str]]:
    result = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() == ".sql":
                text = path.read_text(encoding="utf-8")
                result[path.name] = (text, hashlib.sha256(text.encode()).hexdigest())
    return result


def _ledger(conn: sqlite3.Connection) -> dict[str, str] | None:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_LEDGER,)).fetchone()
    if not exists:
        return None
    return {row[0]: row[1] for row in conn.execute(f"SELECT filename, checksum FROM {_quote(_LEDGER)}")}


def _statements(sql: str) -> list[str]:
    """Split at complete semicolon boundaries, including multiple per line."""
    pending, statements, start = "", [], 0
    for index, char in enumerate(sql):
        if char != ";":
            continue
        candidate = sql[start : index + 1]
        if sqlite3.complete_statement(candidate):
            statement = candidate.strip()
            if statement:
                statements.append(statement)
            start = index + 1
    pending = sql[start:]
    # Comments after a statement are valid trailing content.  SQLite accepts
    # them without a semicolon, but they are not independently executable.
    if pending.strip() and _leading_sql_token(pending) is not None:
        raise SqliteRecoveryError("Migration contains an incomplete SQL statement")
    return statements


def _leading_sql_token(statement: str) -> str | None:
    """Return the first top-level SQL keyword, skipping leading comments."""
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
                raise SqliteRecoveryError("Migration contains an unterminated SQL comment")
            index = end + 2
        else:
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", statement[index:])
            return match.group(0).upper() if match else None
    return None


def _apply_forward_migrations(
    candidate: Path, backup_ledger: dict[str, str] | None, current_ledger: dict[str, str] | None, files: dict[str, tuple[str, str]], *, current_exists: bool
) -> list[str]:
    if backup_ledger is None:
        raise SqliteRecoveryError("Schema conversion requires a _ragtime_migrations ledger in the backup")
    if current_exists and current_ledger is None:
        raise SqliteRecoveryError("Schema conversion requires a _ragtime_migrations ledger in the current database")
    ledgers = [backup_ledger] + ([current_ledger] if current_ledger is not None else [])
    for ledger in ledgers:
        for name, checksum in ledger.items():
            if name not in files or files[name][1] != checksum:
                raise SqliteRecoveryError(f"Migration lineage is missing or changed: {name}")
    wanted = set(current_ledger) if current_exists else set(files)
    forward = sorted(wanted - set(backup_ledger))
    with sqlite3.connect(candidate) as conn:
        conn.execute("PRAGMA foreign_keys=ON")

        def authorizer(action: int, arg1: str | None, arg2: str | None, _: str | None, __: str | None) -> int:
            if action in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and (arg1 or arg2 or "").lower() == "load_extension":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)
        for name in forward:
            sql, checksum = files[name]
            statements = _statements(sql)
            if any(_leading_sql_token(statement) in _FORBIDDEN_MIGRATION_TOKENS for statement in statements):
                raise SqliteRecoveryError(f"Migration {name} contains forbidden transaction-control or PRAGMA SQL")
            try:
                conn.execute("BEGIN")
                for statement in statements:
                    conn.execute(statement)
                conn.execute(f"INSERT INTO {_quote(_LEDGER)} (filename, checksum, applied_at) VALUES (?, ?, datetime('now'))", (name, checksum))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        conn.set_authorizer(None)
        _check_database(conn)
    return forward


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_xinfo({_quote(table)})") if row[6] in (0, 2, 3)]


def _row_query(table: str, columns: list[str], *, where: str = "", order_by: bool = False) -> str:
    selected = ", ".join(_quote(column) for column in columns)
    query = f"SELECT {selected} FROM {_quote(table)}"
    if where:
        query += f" WHERE {where}"
    if order_by:
        query += " ORDER BY " + ", ".join(f"typeof({_quote(column)}), {_quote(column)} COLLATE BINARY" for column in columns)
    return query


def _assert_unambiguous_primary_keys(conn: sqlite3.Connection, table: str, pk: list[str]) -> None:
    null_check = " OR ".join(f"{_quote(column)} IS NULL" for column in pk)
    if conn.execute(f"SELECT 1 FROM {_quote(table)} WHERE {null_check} LIMIT 1").fetchone():
        raise SqliteRecoveryError(f"Merge blocked: table {table} has NULL or ambiguous primary keys")
    grouped = ", ".join(_quote(column) for column in pk)
    if conn.execute(f"SELECT 1 FROM {_quote(table)} GROUP BY {grouped} HAVING COUNT(*) > 1 LIMIT 1").fetchone():
        raise SqliteRecoveryError(f"Merge blocked: table {table} has NULL or ambiguous primary keys")


def _same_rows(left: sqlite3.Connection, right: sqlite3.Connection, table: str, columns: list[str]) -> tuple[bool, int]:
    """Compare no-PK tables as ordered cursors without retaining their rows."""
    left_rows = left.execute(_row_query(table, columns, order_by=True))
    right_rows = right.execute(_row_query(table, columns, order_by=True))
    count = 0
    while True:
        left_row = left_rows.fetchone()
        right_row = right_rows.fetchone()
        if left_row is None or right_row is None:
            return left_row is right_row, count
        if tuple(left_row) != tuple(right_row):
            return False, count
        count += 1


def _primary_key(conn: sqlite3.Connection, table: str) -> list[str]:
    columns = [(row[5], row[1]) for row in conn.execute(f"PRAGMA table_info({_quote(table)})") if row[5]]
    return [name for _, name in sorted(columns)]


def _report(name: str) -> dict[str, Any]:
    return {"name": name, "inserted": 0, "updated": 0, "deleted": 0, "unchanged": 0, "conflicts": 0, "conflict_samples": []}


def _safe_row(row: dict[str, Any]) -> dict[str, Any]:
    return {name: _json_value(value) for name, value in row.items()}


def _preview_value(value: Any) -> Any:
    """Serialize a row value for bounded conflict-preview payloads only."""
    if isinstance(value, bytes):
        encoded = base64.b64encode(value).decode("ascii")
        if len(encoded) > _MAX_PREVIEW_BLOB_BASE64_CHARS:
            return {
                "$blob": encoded[:_MAX_PREVIEW_BLOB_BASE64_CHARS],
                "$truncated": True,
                "$size_bytes": len(value),
            }
        return {"$blob": encoded}
    if isinstance(value, str) and len(value) > _MAX_PREVIEW_STRING_CHARS:
        return value[:_MAX_PREVIEW_STRING_CHARS] + "…"
    return _json_value(value)


def _preview_row(row: dict[str, Any]) -> dict[str, Any]:
    return {name: _preview_value(value) for name, value in row.items()}


def _copy_input(source: Path, destination: Path) -> None:
    capture_database(source, destination)


def prepare_restore(
    backup_path: Path,
    current_path: Path | None,
    migrations_dir: Path,
    output_path: Path,
    *,
    mode: str,
    conflict_policy: str = "keep_current",
    table_policies: dict[str, str] | None = None,
) -> dict:
    result: dict[str, Any] = {"mode": mode, "migrations_applied": [], "tables": [], "warnings": [], "blockers": [], "can_apply": False}
    table_policies = table_policies or {}
    if (
        mode not in {"merge", "overwrite"}
        or conflict_policy not in {"keep_current", "use_backup"}
        or any(value not in {"keep_current", "use_backup"} for value in table_policies.values())
    ):
        result["blockers"].append("Invalid restore mode or conflict policy")
        return result
    if output_path.exists():
        result["blockers"].append("Restore candidate destination already exists")
        return result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.is_file() or (mode == "merge" and (current_path is None or not current_path.is_file())):
        result["blockers"].append("Required backup or current database is missing")
        return result
    temp_dir = Path(tempfile.mkdtemp(prefix="sqlite-restore-", dir=output_path.parent))
    candidate = temp_dir / "candidate.sqlite3"
    try:
        _copy_input(backup_path, candidate)
        files = _migration_files(migrations_dir)
        with _connect_readonly(candidate) as backup_conn:
            backup_schema = _schema_hash(backup_conn)
            backup_ledger = _ledger(backup_conn)
        current_schema = None
        current_ledger = None
        if current_path is not None and current_path.is_file():
            current_copy = temp_dir / "current.sqlite3"
            _copy_input(current_path, current_copy)
            with _connect_readonly(current_copy) as current_conn:
                current_schema = _schema_hash(current_conn)
                current_ledger = _ledger(current_conn)
        if current_schema != backup_schema and (current_schema is not None or files):
            result["migrations_applied"] = _apply_forward_migrations(candidate, backup_ledger, current_ledger, files, current_exists=current_schema is not None)
        if current_schema is not None:
            with _connect_readonly(candidate) as candidate_conn:
                if _schema_hash(candidate_conn) != current_schema:
                    raise SqliteRecoveryError("Forward migrations did not produce the current semantic schema")
        with _connect_readonly(candidate) as candidate_conn:
            unknown_policies = sorted(set(table_policies) - set(_table_names(candidate_conn)))
        if unknown_policies:
            raise SqliteRecoveryError(f"Unknown table policy names: {', '.join(unknown_policies)}")
        if mode == "merge":
            assert current_path is not None
            backup_copy = temp_dir / "backup.sqlite3"
            os.replace(candidate, backup_copy)
            _copy_input(current_path, candidate)
            _merge(candidate, backup_copy, result, conflict_policy, table_policies)
        else:
            if current_path is not None and current_path.is_file():
                _overwrite_report(temp_dir / "current.sqlite3", candidate, result)
        conn = sqlite3.connect(candidate)
        try:
            _check_database(conn)
            _normalize_single_file_output(conn)
        finally:
            conn.close()
        _durable_file(candidate)
        os.replace(candidate, output_path)
        _durable_file(output_path)
        result["candidate_sha256"] = _sha256_file(output_path)
        result["can_apply"] = True
    except (sqlite3.Error, sqlite3.Warning, OSError, SqliteRecoveryError, UnicodeError, Warning) as exc:
        result["blockers"].append(str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return result


def _overwrite_report(current: Path, candidate: Path, result: dict[str, Any]) -> None:
    with _connect_readonly(current) as old, _connect_readonly(candidate) as new:
        old_tables = set(_table_names(old))
        for table in _table_names(new):
            report = _report(table)
            result["tables"].append(report)
            columns = _columns(new, table)
            if table not in old_tables:
                report["inserted"] = new.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
                continue
            pk = _primary_key(new, table)
            if not pk:
                report["deleted"] = old.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
                report["inserted"] = new.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
                continue
            _assert_unambiguous_primary_keys(old, table, pk)
            _assert_unambiguous_primary_keys(new, table, pk)
            where = " AND ".join(f"{_quote(column)}=?" for column in pk)
            for backup_row in new.execute(_row_query(table, columns)):
                values = tuple(backup_row)
                key = tuple(values[columns.index(column)] for column in pk)
                current_row = old.execute(_row_query(table, columns, where=where), key).fetchone()
                if current_row is None:
                    report["inserted"] += 1
                elif tuple(current_row) == values:
                    report["unchanged"] += 1
                else:
                    report["updated"] += 1
            report["deleted"] = old.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0] - report["unchanged"] - report["updated"]


def _merge(candidate: Path, backup: Path, result: dict[str, Any], default_policy: str, policies: dict[str, str]) -> None:
    # backup is created as a consistent private copy above.
    with sqlite3.connect(candidate) as current_conn, _connect_readonly(backup) as backup_conn:
        current_conn.execute("PRAGMA foreign_keys=ON")
        current_tables = _table_names(current_conn)
        backup_tables = _table_names(backup_conn)
        if current_tables != backup_tables or _schema_shape(current_conn) != _schema_shape(backup_conn):
            raise SqliteRecoveryError("Merge requires matching semantic schemas")
        shapes = _schema_shape(current_conn)
        plans: list[tuple[str, list[str], list[str], str]] = []
        for table in current_tables:
            report = _report(table)
            result["tables"].append(report)
            columns = _columns(current_conn, table)
            pk = _primary_key(current_conn, table)
            if not pk:
                same, count = _same_rows(current_conn, backup_conn, table, columns)
                if not same:
                    raise SqliteRecoveryError(f"Merge blocked: table {table} has no declared primary key")
                report["unchanged"] = count
                continue
            _assert_unambiguous_primary_keys(current_conn, table, pk)
            _assert_unambiguous_primary_keys(backup_conn, table, pk)
            policy = policies.get(table, default_policy)
            where = " AND ".join(f"{_quote(column)}=?" for column in pk)
            for backup_row in backup_conn.execute(_row_query(table, columns)):
                backup_values = tuple(backup_row)
                key = tuple(backup_values[columns.index(column)] for column in pk)
                current_row = current_conn.execute(_row_query(table, columns, where=where), key).fetchone()
                if current_row is None:
                    report["inserted"] += 1
                elif tuple(current_row) == backup_values:
                    report["unchanged"] += 1
                else:
                    report["conflicts"] += 1
                    if len(report["conflict_samples"]) < 20:
                        report["conflict_samples"].append(
                            {
                                "key": _preview_row({name: value for name, value in zip(pk, key)}),
                                "current": _preview_row(dict(zip(columns, current_row))),
                                "backup": _preview_row(dict(zip(columns, backup_values))),
                            }
                        )
                    if policy == "use_backup":
                        report["updated"] += 1
            writes = report["inserted"] + report["updated"]
            shape = shapes[table]
            if writes and shape["virtual"]:
                raise SqliteRecoveryError(f"Merge blocked: virtual table {table} would receive writes")
            if writes and shape["triggers"]:
                raise SqliteRecoveryError(f"Merge blocked: trigger-bearing table {table} would receive writes")
            writable = [row[1] for row in current_conn.execute(f"PRAGMA table_xinfo({_quote(table)})") if row[6] == 0]
            plans.append((table, writable, pk, policy))
        result["warnings"].append("Merge keeps current-only rows; backup-only rows may revive deliberate deletions.")
        try:
            current_conn.execute("BEGIN")
            current_conn.execute("PRAGMA defer_foreign_keys=ON")
            for table, writable, pk, policy in plans:
                columns = _columns(backup_conn, table)
                where = " AND ".join(f"{_quote(column)}=?" for column in pk)
                for backup_row in backup_conn.execute(_row_query(table, columns)):
                    backup_values = tuple(backup_row)
                    values_by_column = dict(zip(columns, backup_values))
                    key = tuple(values_by_column[column] for column in pk)
                    current_row = current_conn.execute(_row_query(table, columns, where=where), key).fetchone()
                    if current_row is None:
                        current_conn.execute(
                            f"INSERT INTO {_quote(table)} ({', '.join(_quote(col) for col in writable)}) VALUES ({', '.join('?' for _ in writable)})",
                            tuple(values_by_column[col] for col in writable),
                        )
                    elif tuple(current_row) != backup_values and policy == "use_backup":
                        changed = [col for col in writable if col not in pk]
                        if changed:
                            current_conn.execute(
                                f"UPDATE {_quote(table)} SET {', '.join(f'{_quote(col)}=?' for col in changed)} WHERE {' AND '.join(f'{_quote(col)}=?' for col in pk)}",
                                tuple(values_by_column[col] for col in changed) + key,
                            )
            _repair_sequences(current_conn, backup_conn, plans)
            _check_database(current_conn)
            current_conn.commit()
        except Exception:
            current_conn.rollback()
            raise


def _repair_sequences(conn: sqlite3.Connection, backup: sqlite3.Connection, plans: list[tuple[str, list[str], list[str], str]]) -> None:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'").fetchone():
        return
    for table, _, pk, _ in plans:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0] or ""
        if "AUTOINCREMENT" not in sql.upper() or len(pk) != 1:
            continue
        value = backup.execute(f"SELECT MAX({_quote(pk[0])}) FROM {_quote(table)} WHERE typeof({_quote(pk[0])})='integer'").fetchone()[0]
        if value is not None:
            conn.execute("UPDATE sqlite_sequence SET seq=MAX(seq, ?) WHERE name=?", (value, table))
