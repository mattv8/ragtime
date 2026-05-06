"""Import SQL dumps (PostgreSQL, MySQL, generic) into a workspace SQLite database.

Uses *sqlglot* for robust SQL transpilation between dialects instead of
hand-rolled regex transforms.  Only non-SQL constructs (psql protocol
commands, MySQL directives, binary format detection) are handled with
lightweight pre-processing before handing off to sqlglot.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError

SqlImportDialect = Literal["postgresql", "mysql", "sqlite", "generic"]
SqlImportProgressCallback = Callable[[dict[str, Any]], None]

# Magic bytes for binary pg_dump custom format.
_PG_DUMP_MAGIC = b"PGDMP"
_PG_RESTORE_TIMEOUT_SECONDS = 600
_SQLITE_INSERT_BATCH_ROWS = 1000
_PROGRESS_REPORT_INTERVAL_BYTES = 4 * 1024 * 1024
_RESTORING_DUMP_PROGRESS = 0.15
_TRANSPILING_SQL_PROGRESS = 0.3
_IMPORTING_SQL_BASE_PROGRESS = 0.35
_IMPORTING_SQL_PROGRESS_SPAN = 0.6
_FINALIZING_SQLITE_PROGRESS = 0.97

# Map our dialect names -> sqlglot dialect identifiers.
_SQLGLOT_READ_DIALECT: dict[SqlImportDialect, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "generic": "sqlite",  # best-effort: already close to SQLite
}

_MAX_REPORTED_ERRORS = 100
_MAX_REPORTED_WARNINGS = 100


def _emit_progress(
    progress_callback: SqlImportProgressCallback | None,
    payload: dict[str, Any],
) -> None:
    if progress_callback:
        progress_callback(payload)


def _should_report_byte_progress(
    processed_bytes: int, total_bytes: int, last_chunk_bytes: int
) -> bool:
    return (
        processed_bytes == total_bytes
        or processed_bytes % _PROGRESS_REPORT_INTERVAL_BYTES < last_chunk_bytes
    )


def _byte_progress(
    processed_bytes: int,
    total_bytes: int,
    *,
    base_progress: float = _IMPORTING_SQL_BASE_PROGRESS,
    progress_span: float = _IMPORTING_SQL_PROGRESS_SPAN,
) -> float:
    return base_progress + progress_span * min(
        1.0, processed_bytes / max(1, total_bytes)
    )


@dataclass
class SqlImportResult:
    success: bool = False
    dialect: SqlImportDialect = "generic"
    tables_created: int = 0
    rows_inserted: int = 0
    statements_executed: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binary format detection
# ---------------------------------------------------------------------------


def detect_binary_pg_dump(content: bytes) -> bool:
    return content[:5] == _PG_DUMP_MAGIC


def detect_binary_pg_dump_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(5) == _PG_DUMP_MAGIC


def convert_binary_pg_dump_to_sql(
    content: bytes,
    progress_callback: SqlImportProgressCallback | None = None,
) -> str:
    """Convert PostgreSQL custom-format dumps to plain SQL using pg_restore."""
    if not detect_binary_pg_dump(content):
        raise ValueError("Content is not a PostgreSQL custom-format dump")

    with tempfile.NamedTemporaryFile(suffix=".dump") as dump_file:
        dump_file.write(content)
        dump_file.flush()
        _emit_progress(
            progress_callback,
            {"phase": "restoring_dump", "processed_bytes": len(content)},
        )
        try:
            completed = subprocess.run(
                [
                    "pg_restore",
                    "--file=-",
                    "--no-owner",
                    "--no-privileges",
                    dump_file.name,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_PG_RESTORE_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Binary PostgreSQL dump detected, but pg_restore is not installed. "
                "Install PostgreSQL client tools or re-export using: pg_dump --format=plain"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "pg_restore timed out while converting binary PostgreSQL dump"
            ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown pg_restore error"
        raise RuntimeError(
            f"pg_restore failed while converting binary PostgreSQL dump: {stderr}"
        )

    _emit_progress(
        progress_callback,
        {"phase": "transpiling_sql", "processed_bytes": len(content)},
    )
    return completed.stdout


def restore_binary_pg_dump_to_sql_file(
    dump_path: Path,
    sql_path: Path,
    progress_callback: SqlImportProgressCallback | None = None,
) -> None:
    """Convert a PostgreSQL custom-format dump to plain SQL on disk."""
    _restore_binary_pg_dump_section_to_file(
        dump_path,
        sql_path,
        [],
        progress_callback,
        "restoring_dump",
    )
    _emit_progress(
        progress_callback,
        {
            "phase": "transpiling_sql",
            "dialect": "postgresql",
            "processed_bytes": dump_path.stat().st_size,
            "progress": _TRANSPILING_SQL_PROGRESS,
        },
    )


def _restore_binary_pg_dump_section_to_file(
    dump_path: Path,
    sql_path: Path,
    section_args: list[str],
    progress_callback: SqlImportProgressCallback | None = None,
    phase: str = "restoring_dump",
) -> None:
    if not detect_binary_pg_dump_file(dump_path):
        raise ValueError("Content is not a PostgreSQL custom-format dump")

    _emit_progress(
        progress_callback,
        {
            "phase": phase,
            "dialect": "postgresql",
            "processed_bytes": dump_path.stat().st_size,
            "progress": _RESTORING_DUMP_PROGRESS,
        },
    )

    try:
        completed = subprocess.run(
            [
                "pg_restore",
                f"--file={sql_path}",
                "--no-owner",
                "--no-privileges",
                *section_args,
                str(dump_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PG_RESTORE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Binary PostgreSQL dump detected, but pg_restore is not installed. "
            "Install PostgreSQL client tools or re-export using: pg_dump --format=plain"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "pg_restore timed out while converting binary PostgreSQL dump"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown pg_restore error"
        raise RuntimeError(
            f"pg_restore failed while converting binary PostgreSQL dump: {stderr}"
        )


def decode_sql_dump_bytes(content: bytes) -> str:
    """Decode a text SQL dump with common BOM and encoding fallbacks."""
    for encoding in ("utf-8-sig", "utf-16", "utf-32"):
        try:
            return content.decode(encoding)
        except UnicodeError:
            pass

    try:
        import chardet

        detected = chardet.detect(content)
        encoding = detected.get("encoding")
        confidence = float(detected.get("confidence") or 0)
        if encoding and confidence >= 0.6:
            return content.decode(encoding)
    except Exception:
        pass

    # Latin-1 is byte-preserving, so it gives users a best-effort import instead
    # of rejecting uncommon legacy encodings outright.
    return content.decode("latin-1")


# ---------------------------------------------------------------------------
# Dialect detection (heuristic on raw text before parsing)
# ---------------------------------------------------------------------------


def detect_sql_dialect(text: str, filename: str | None = None) -> SqlImportDialect:
    suffix = Path(filename).suffix.lower() if filename else ""
    if suffix in {".pg", ".pgsql"}:
        return "postgresql"
    if suffix == ".mysql":
        return "mysql"

    sample = text[:8000].lower()
    pg_signals = (
        "postgresql database dump",
        "dumped from database version",
        "dumped by pg_dump",
        "pg_catalog",
        "pg_dump",
        "\\connect",
        "create sequence",
        "alter table only",
        " owner to ",
        "nextval(",
        "set search_path",
        "set statement_timeout",
        "set lock_timeout",
        "set idle_in_transaction_session_timeout",
        "set row_security",
        "set default_tablespace",
        "copy ",
        "copy public.",
        "create table public.",
        "::text",
        "::integer",
        "::boolean",
        "::timestamp",
        "without time zone",
        "with time zone",
    )
    mysql_signals = (
        "mysql dump",
        "mariadb dump",
        "auto_increment",
        "engine=innodb",
        "engine=myisam",
        "default charset",
        "lock tables",
        "unlock tables",
        "/*!",
        "set @@",
    )
    pg_score = sum(1 for s in pg_signals if s in sample)
    mysql_score = sum(1 for s in mysql_signals if s in sample)

    if re.search(r"\bcopy\s+.+?\s+from\s+stdin\b", sample, re.IGNORECASE | re.DOTALL):
        return "postgresql"
    if "postgresql database dump" in sample or "mysql dump" in sample:
        return "postgresql" if "postgresql database dump" in sample else "mysql"
    if pg_score >= 2:
        return "postgresql"
    if mysql_score >= 2:
        return "mysql"
    if "sqlite" in sample:
        return "sqlite"
    return "generic"


# ---------------------------------------------------------------------------
# Pre-processing: strip non-SQL constructs that sqlglot cannot parse
# ---------------------------------------------------------------------------

# PostgreSQL COPY ... FROM stdin blocks -> INSERT statements.
_PG_COPY_RE = re.compile(
    r"COPY\s+"
    r"((?:\"(?:\"\"|[^\"])+\"|[A-Za-z_][\w$]*)(?:\s*\.\s*(?:\"(?:\"\"|[^\"])+\"|[A-Za-z_][\w$]*))*)"
    r"\s*\((.*?)\)\s+FROM\s+stdin\s*(?:WITH\s*(?:\([^;\n]*\)|[^;\n]*))?;?",
    re.IGNORECASE | re.DOTALL,
)

# Psql backslash commands and pg_catalog function calls (match entire line).
_PG_STRIP_LINE_RE = re.compile(
    r"^\s*(?:\\.*|SELECT\s+pg_catalog\..*)$",
    re.IGNORECASE | re.MULTILINE,
)

# MySQL conditional comments: /*!40101 SET ... */;
_MYSQL_CONDITIONAL_RE = re.compile(r"/\*![\d ]*.*?\*/;?", re.DOTALL)
_MYSQL_DELIMITER_RE = re.compile(r"^\s*DELIMITER\b.*$", re.IGNORECASE | re.MULTILINE)
_MYSQL_ON_DUPLICATE_RE = re.compile(
    r"\s+ON\s+DUPLICATE\s+KEY\s+UPDATE\s+(?P<updates>.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Security: filesystem-touching SQLite statements must never reach the executor.
_DANGEROUS_SQLITE_RE = re.compile(
    r"^\s*(?:ATTACH(?:\s+DATABASE)?|VACUUM\s+INTO)\b",
    re.IGNORECASE,
)
_ATTACH_DB_RE = re.compile(r"\bATTACH(?:\s+DATABASE)?\b", re.IGNORECASE)

# Post-processing: sqlglot quirks that need patching for valid SQLite.
# UBIGINT / USMALLINT / UTINYINT are not real SQLite types.
_UNSIGNED_INT_TYPE_RE = re.compile(r"\bU(?:BIG|SMALL|TINY)?INT\b", re.IGNORECASE)
# NULLS FIRST / NULLS LAST not supported in older SQLite.
_NULLS_ORDER_RE = re.compile(r"\s+NULLS\s+(?:FIRST|LAST)\b", re.IGNORECASE)


def _add_limited_message(messages: list[str], message: str, limit: int) -> None:
    if len(messages) < limit:
        messages.append(message)
    elif len(messages) == limit:
        messages.append("Additional messages suppressed.")


def _strip_identifier_quotes(identifier: str) -> str:
    identifier = identifier.strip()
    if identifier.startswith('"') and identifier.endswith('"'):
        return identifier[1:-1].replace('""', '"')
    if identifier.startswith("`") and identifier.endswith("`"):
        return identifier[1:-1].replace("``", "`")
    return identifier


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _split_identifier_parts(identifier: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_double_quote = False
    index = 0
    while index < len(identifier):
        char = identifier[index]
        if char == '"':
            current.append(char)
            if (
                in_double_quote
                and index + 1 < len(identifier)
                and identifier[index + 1] == '"'
            ):
                current.append(identifier[index + 1])
                index += 2
                continue
            in_double_quote = not in_double_quote
        elif char == "." and not in_double_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    if current:
        parts.append("".join(current).strip())
    return [_strip_identifier_quotes(part) for part in parts if part.strip()]


def _sqlite_table_name(identifier: str, warnings: list[str]) -> str:
    parts = _split_identifier_parts(identifier)
    if not parts:
        return _quote_sqlite_identifier(identifier.strip())
    if len(parts) > 1:
        _add_limited_message(
            warnings,
            f"Dropped source schema qualifier for table {identifier.strip()}",
            _MAX_REPORTED_WARNINGS,
        )
    return _quote_sqlite_identifier(parts[-1])


def _split_identifier_list(text: str) -> list[str]:
    identifiers: list[str] = []
    current: list[str] = []
    in_double_quote = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            current.append(char)
            if in_double_quote and index + 1 < len(text) and text[index + 1] == '"':
                current.append(text[index + 1])
                index += 2
                continue
            in_double_quote = not in_double_quote
        elif char == "," and not in_double_quote:
            value = "".join(current).strip()
            if value:
                identifiers.append(value)
            current = []
        else:
            current.append(char)
        index += 1
    value = "".join(current).strip()
    if value:
        identifiers.append(value)
    return identifiers


def _sqlite_column_list(columns: str) -> str:
    return ", ".join(
        _quote_sqlite_identifier(_strip_identifier_quotes(column))
        for column in _split_identifier_list(columns)
    )


def _ensure_copy_target_table(
    cursor: sqlite3.Cursor,
    table_name: str,
    columns: list[str],
) -> None:
    column_defs = ", ".join(
        f"{_quote_sqlite_identifier(_strip_identifier_quotes(column))} TEXT"
        for column in columns
    )
    if column_defs:
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs})")


def _sqlite_literal(value: str | None) -> str:
    if value is None:
        return "NULL"
    return "'" + value.replace("'", "''") + "'"


def _decode_pg_copy_text_value(value: str) -> str | None:
    if value == r"\N":
        return None

    escapes = {
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
    }
    result: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char != "\\" or index + 1 >= len(value):
            result.append(char)
            index += 1
            continue

        next_char = value[index + 1]
        if next_char in escapes:
            result.append(escapes[next_char])
            index += 2
            continue
        if next_char in "01234567":
            end = index + 1
            while end < len(value) and end < index + 4 and value[end] in "01234567":
                end += 1
            result.append(chr(int(value[index + 1 : end], 8)))
            index = end
            continue
        if next_char == "x":
            end = index + 2
            while (
                end < len(value)
                and end < index + 4
                and value[end] in "0123456789abcdefABCDEF"
            ):
                end += 1
            if end > index + 2:
                result.append(chr(int(value[index + 2 : end], 16)))
                index = end
                continue

        result.append(next_char)
        index += 2

    return "".join(result)


def _copy_options_are_csv(copy_clause: str) -> bool:
    return bool(re.search(r"\b(?:FORMAT\s+CSV|CSV)\b", copy_clause, re.IGNORECASE))


def _convert_pg_copy_blocks(sql: str, warnings: list[str]) -> str:
    """Convert PostgreSQL COPY ... FROM stdin blocks to INSERT statements."""
    result_parts: list[str] = []
    pos = 0

    while pos < len(sql):
        match = _PG_COPY_RE.search(sql, pos)
        if not match:
            result_parts.append(sql[pos:])
            break

        result_parts.append(sql[pos : match.start()])

        table_name = _sqlite_table_name(match.group(1), warnings)
        columns = _sqlite_column_list(match.group(2).strip())
        copy_clause = match.group(0)

        # Find end of COPY data block (line that is just \.)
        try:
            data_start = sql.index("\n", match.end()) + 1
        except ValueError:
            result_parts.append(f"-- Skipped COPY for {table_name} (no data)\n")
            pos = match.end()
            continue

        end_marker_re = re.compile(r"^\\\.\s*$", re.MULTILINE)
        end_match = end_marker_re.search(sql, data_start)
        if not end_match:
            _add_limited_message(
                warnings,
                f"Unterminated COPY block for {table_name}; skipped",
                _MAX_REPORTED_WARNINGS,
            )
            result_parts.append(f"-- Skipped unterminated COPY for {table_name}\n")
            pos = data_start
            continue

        data_block = sql[data_start : end_match.start()]
        pos = end_match.end()

        row_count = 0
        for line in data_block.splitlines():
            line = line.rstrip("\r")
            if not line:
                continue
            if _copy_options_are_csv(copy_clause):
                # PostgreSQL CSV COPY uses the csv module's escaping rules closely enough
                # for import purposes once the outer COPY protocol is removed.
                import csv

                values = next(csv.reader([line]))
                parsed_values = [None if value == r"\N" else value for value in values]
            else:
                parsed_values = [
                    _decode_pg_copy_text_value(value) for value in line.split("\t")
                ]
            escaped = [_sqlite_literal(value) for value in parsed_values]
            result_parts.append(
                f"INSERT INTO {table_name} ({columns}) VALUES ({', '.join(escaped)});\n"
            )
            row_count += 1

        if row_count:
            _add_limited_message(
                warnings,
                f"Converted COPY block for {table_name}: {row_count} rows",
                _MAX_REPORTED_WARNINGS,
            )

    return "".join(result_parts)


def _preprocess(sql: str, dialect: SqlImportDialect, warnings: list[str]) -> str:
    """Strip non-SQL constructs that sqlglot cannot parse.

    Returns cleaned SQL text ready for transpilation.
    """
    if dialect == "postgresql":
        sql = _convert_pg_copy_blocks(sql, warnings)
        sql = _PG_STRIP_LINE_RE.sub("", sql)

    if dialect == "mysql":
        sql = _MYSQL_CONDITIONAL_RE.sub("", sql)
        sql = _MYSQL_DELIMITER_RE.sub("", sql)

    # Comment obvious ATTACH uses early; the executor also blocks parsed statements.
    sql = _ATTACH_DB_RE.sub("-- blocked: ATTACH", sql)

    return sql


# ---------------------------------------------------------------------------
# Transpilation via sqlglot
# ---------------------------------------------------------------------------

# Statement prefixes that sqlglot won't transpile meaningfully and that
# we should skip rather than feed to SQLite.
_SKIP_PREFIXES = (
    "set ",
    "lock tables",
    "unlock tables",
    "grant ",
    "revoke ",
    "create sequence",
    "alter sequence",
    "alter table only",
    "alter table if exists",
    "create extension",
    "comment on",
    "create schema",
    "create function",
    "create or replace function",
    "create trigger",
    "drop trigger",
    "create type",
    "alter type",
    "create index concurrently",
    "-- blocked:",
    "begin",
    "commit",
    "start transaction",
    "delimiter ",
)

_SKIP_PATTERNS = (
    re.compile(r"^\s*ALTER\s+TABLE\s+.+\s+OWNER\s+TO\b", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"^\s*ALTER\s+TABLE\s+(?:ONLY\s+)?\S+\s+ADD\s+CONSTRAINT\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"^\s*SELECT\s+pg_catalog\.setval\b", re.IGNORECASE | re.DOTALL),
)


# Inline INDEX / FULLTEXT INDEX inside CREATE TABLE (invalid in SQLite).
_INLINE_INDEX_RE = re.compile(
    r",\s*(?:UNIQUE\s+|FULLTEXT\s+)?INDEX\s+\"?\w+\"?\s*\([^)]*\)",
    re.IGNORECASE,
)


def _post_process_sqlite(stmt: str) -> str:
    """Fix sqlglot output quirks that produce invalid SQLite SQL."""
    # Replace unsigned int types with INTEGER.
    stmt = _UNSIGNED_INT_TYPE_RE.sub("INTEGER", stmt)
    # Remove NULLS FIRST/LAST (unsupported in SQLite).
    stmt = _NULLS_ORDER_RE.sub("", stmt)
    # Remove inline INDEX declarations from CREATE TABLE (SQLite doesn't support them).
    if "INDEX" in stmt.upper() and "CREATE TABLE" in stmt.upper():
        stmt = _INLINE_INDEX_RE.sub("", stmt)
    # Fix AUTOINCREMENT: SQLite only allows INTEGER PRIMARY KEY AUTOINCREMENT.
    if "AUTOINCREMENT" in stmt.upper():
        stmt = _fix_autoincrement(stmt)
    if "ON DUPLICATE KEY UPDATE" in stmt.upper():
        stmt = _fix_mysql_on_duplicate_key(stmt)
    return stmt.strip()


def _fix_mysql_on_duplicate_key(stmt: str) -> str:
    match = _MYSQL_ON_DUPLICATE_RE.search(stmt)
    if not match:
        return stmt
    updates = match.group("updates")
    updates = re.sub(r"^\s*SET\s+", "", updates, flags=re.IGNORECASE)
    updates = re.sub(
        r"VALUES\(\s*([`\"\[]?)(\w+)\1\s*\)",
        lambda m: f"excluded.{_quote_sqlite_identifier(m.group(2))}",
        updates,
        flags=re.IGNORECASE,
    )
    return stmt[: match.start()] + f" ON CONFLICT DO UPDATE SET {updates}"


def _fix_autoincrement(stmt: str) -> str:
    """Rewrite AUTOINCREMENT columns to valid SQLite INTEGER PRIMARY KEY AUTOINCREMENT."""
    # Column names may be bare (id) or double-quoted ("id") after sqlglot.
    ai_re = re.compile(
        r'("?\b\w+"?)'  # column name, possibly quoted
        r"\s+(INTEGER|INT|BIGINT|SMALLINT|TINYINT|\w*INT\w*)"  # type
        r"((?:\s+NOT\s+NULL|\s+NULL)*)"  # optional NOT NULL
        r"\s+AUTOINCREMENT"
        r"(\s+PRIMARY\s+KEY)?",
        re.IGNORECASE,
    )

    def _repl(m: re.Match) -> str:
        col = m.group(1)
        not_null = m.group(3).strip()
        nn = " NOT NULL" if "NOT" in not_null.upper() else ""
        return f"{col} INTEGER{nn} PRIMARY KEY AUTOINCREMENT"

    result = ai_re.sub(_repl, stmt)

    # Also handle: "col" INTEGER PRIMARY KEY AUTOINCREMENT — already valid,
    # but normalise type to INTEGER just in case.
    result = re.sub(
        r'("?\b\w+"?)\s+\w*INT\w*\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
        r"\1 INTEGER PRIMARY KEY AUTOINCREMENT",
        result,
        flags=re.IGNORECASE,
    )

    # If AUTOINCREMENT still present but no PRIMARY KEY, strip it
    # (SQLite requires PRIMARY KEY for AUTOINCREMENT).
    if "AUTOINCREMENT" in result.upper() and "PRIMARY KEY" not in result.upper():
        result = re.sub(r"\s*AUTOINCREMENT\b", "", result, flags=re.IGNORECASE)

    return result


def _should_skip_statement(stmt: str) -> bool:
    lower = stmt.strip().lower()
    if any(lower.startswith(prefix) for prefix in _SKIP_PREFIXES):
        return True
    return any(pattern.match(stmt) for pattern in _SKIP_PATTERNS)


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    dollar_quote: str | None = None
    index = 0

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if in_line_comment:
            current.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                in_block_comment = False
                index += 2
            else:
                index += 1
            continue

        if dollar_quote:
            if sql.startswith(dollar_quote, index):
                current.append(dollar_quote)
                index += len(dollar_quote)
                dollar_quote = None
            else:
                current.append(char)
                index += 1
            continue

        if in_single_quote:
            current.append(char)
            if char == "'":
                if next_char == "'":
                    current.append(next_char)
                    index += 2
                    continue
                in_single_quote = False
            elif char == "\\" and next_char:
                current.append(next_char)
                index += 2
                continue
            index += 1
            continue

        if in_double_quote:
            current.append(char)
            if char == '"':
                if next_char == '"':
                    current.append(next_char)
                    index += 2
                    continue
                in_double_quote = False
            index += 1
            continue

        if char == "-" and next_char == "-":
            current.append(char)
            current.append(next_char)
            in_line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            current.append(char)
            current.append(next_char)
            in_block_comment = True
            index += 2
            continue
        if char == "'":
            current.append(char)
            in_single_quote = True
            index += 1
            continue
        if char == '"':
            current.append(char)
            in_double_quote = True
            index += 1
            continue
        if char == "$":
            match = re.match(r"\$[A-Za-z_][\w$]*\$|\$\$", sql[index:])
            if match:
                dollar_quote = match.group(0)
                current.append(dollar_quote)
                index += len(dollar_quote)
                continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    statement = "".join(current).strip()
    if statement:
        statements.append(statement)
    return statements


def _strip_schema_qualifiers(
    expression: exp.Expression, warnings: list[str]
) -> exp.Expression:
    stripped_any = False

    def _transform(node: exp.Expression) -> exp.Expression:
        nonlocal stripped_any
        if isinstance(node, exp.Table) and (
            node.args.get("db") or node.args.get("catalog")
        ):
            node.set("db", None)
            node.set("catalog", None)
            stripped_any = True
        return node

    transformed = expression.transform(_transform)
    if stripped_any:
        _add_limited_message(
            warnings,
            "Dropped source schema qualifiers for SQLite compatibility",
            _MAX_REPORTED_WARNINGS,
        )
    return transformed


def _transpile_statements(
    sql: str, dialect: SqlImportDialect, warnings: list[str]
) -> list[str]:
    """Transpile SQL text from source dialect to SQLite using sqlglot.

    Returns a list of SQLite-compatible SQL strings.
    """
    read_dialect = _SQLGLOT_READ_DIALECT.get(dialect, "sqlite")

    output: list[str] = []
    for raw_stmt in _split_sql_statements(sql):
        if _should_skip_statement(raw_stmt):
            continue
        if _DANGEROUS_SQLITE_RE.match(raw_stmt):
            _add_limited_message(
                warnings,
                "Blocked filesystem-touching SQLite statement",
                _MAX_REPORTED_WARNINGS,
            )
            continue

        try:
            parsed = sqlglot.parse(
                raw_stmt, read=read_dialect, error_level=ErrorLevel.WARN
            )
            transpiled = [
                _strip_schema_qualifiers(expression, warnings).sql(
                    dialect="sqlite",
                    unsupported_level=ErrorLevel.IGNORE,
                )
                for expression in parsed
                if expression is not None
            ]
        except ParseError as exc:
            _add_limited_message(
                warnings,
                f"sqlglot parse warning; using raw statement fallback: {exc}",
                _MAX_REPORTED_WARNINGS,
            )
            transpiled = [raw_stmt]

        for stmt in transpiled:
            stripped = _post_process_sqlite(stmt.strip())
            if stripped and not _should_skip_statement(stripped):
                output.append(stripped)

    return output


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def import_sql_to_sqlite(
    sqlite_path: Path,
    sql_text: str,
    dialect: SqlImportDialect | None = None,
    progress_callback: SqlImportProgressCallback | None = None,
    rollback_on_error: bool = True,
) -> SqlImportResult:
    """Parse, transpile, and execute a SQL dump into the target SQLite database.

    Runs all statements inside a single transaction.  Per-statement errors
    are collected (best-effort) rather than aborting the entire import.
    """
    if dialect is None:
        dialect = detect_sql_dialect(sql_text)

    result = SqlImportResult(dialect=dialect)
    _emit_progress(progress_callback, {"phase": "transpiling_sql", "dialect": dialect})

    # Pre-process: remove non-SQL constructs.
    cleaned = _preprocess(sql_text, dialect, result.warnings)

    # Transpile via sqlglot.
    statements = _transpile_statements(cleaned, dialect, result.warnings)
    _emit_progress(
        progress_callback,
        {
            "phase": "importing_sql",
            "dialect": dialect,
            "total_statements": len(statements),
            "statements_executed": 0,
            "rows_inserted": 0,
            "tables_created": 0,
        },
    )

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=OFF")

        cursor = conn.cursor()
        cursor.execute("BEGIN")

        tables_before = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if not row[0].startswith("sqlite_")
        }

        for stmt in statements:
            # Final safety check -- never execute ATTACH.
            if _DANGEROUS_SQLITE_RE.match(stmt) or _ATTACH_DB_RE.search(stmt):
                _add_limited_message(
                    result.warnings,
                    "Blocked filesystem-touching SQLite statement",
                    _MAX_REPORTED_WARNINGS,
                )
                continue

            try:
                cursor.execute(stmt)
                result.statements_executed += 1

                lower = stmt.lower().lstrip()
                if lower.startswith(("insert", "replace")):
                    result.rows_inserted += max(cursor.rowcount, 0)
                if progress_callback and (
                    result.statements_executed == len(statements)
                    or result.statements_executed % 100 == 0
                ):
                    _emit_progress(
                        progress_callback,
                        {
                            "phase": "importing_sql",
                            "dialect": dialect,
                            "total_statements": len(statements),
                            "statements_executed": result.statements_executed,
                            "rows_inserted": result.rows_inserted,
                        },
                    )
            except sqlite3.Error as exc:
                _add_limited_message(
                    result.errors,
                    f"Statement error: {exc}",
                    _MAX_REPORTED_ERRORS,
                )
                continue

        if result.errors and rollback_on_error:
            conn.rollback()
            result.tables_created = 0
            _add_limited_message(
                result.warnings,
                "Rolled back SQL import because one or more statements failed",
                _MAX_REPORTED_WARNINGS,
            )
            result.success = False
        else:
            tables_after = {
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not row[0].startswith("sqlite_")
            }
            result.tables_created = len(tables_after - tables_before)
            _emit_progress(
                progress_callback,
                {
                    "phase": "finalizing_sqlite",
                    "dialect": dialect,
                    "total_statements": len(statements),
                    "statements_executed": result.statements_executed,
                    "rows_inserted": result.rows_inserted,
                    "tables_created": result.tables_created,
                    "progress": _FINALIZING_SQLITE_PROGRESS,
                },
            )
            conn.commit()
            result.success = not result.errors or not rollback_on_error
        _emit_progress(
            progress_callback,
            {
                "phase": "completed" if result.success else "failed",
                "dialect": dialect,
                "total_statements": len(statements),
                "statements_executed": result.statements_executed,
                "rows_inserted": result.rows_inserted,
                "tables_created": result.tables_created,
                "warnings": result.warnings,
                "errors": result.errors,
                "success": result.success,
            },
        )

    except Exception as exc:
        _add_limited_message(
            result.errors, f"Fatal import error: {exc}", _MAX_REPORTED_ERRORS
        )
        result.success = False
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    return result


def _flush_copy_batch(
    cursor: sqlite3.Cursor, insert_sql: str, batch: list[list[Any]]
) -> int:
    if not batch:
        return 0
    cursor.executemany(insert_sql, batch)
    count = len(batch)
    batch.clear()
    return count


def _stream_postgres_data_file_to_sqlite(
    sqlite_path: Path,
    data_path: Path,
    progress_callback: SqlImportProgressCallback | None = None,
    base_progress: float = _IMPORTING_SQL_BASE_PROGRESS,
    progress_span: float = _IMPORTING_SQL_PROGRESS_SPAN,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    rows_inserted = 0
    total_bytes = max(1, data_path.stat().st_size)
    processed_bytes = 0
    current_insert_sql: str | None = None
    current_batch: list[list[Any]] = []
    current_csv_mode = False
    current_table = ""
    expected_columns = 0

    conn = sqlite3.connect(str(sqlite_path))
    conn.isolation_level = None
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.execute("BEGIN")
        with data_path.open("rb") as handle:
            for raw_line in handle:
                processed_bytes += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if line.endswith("\r"):
                    line = line[:-1]

                if current_insert_sql:
                    if line == r"\.":
                        rows_inserted += _flush_copy_batch(
                            cursor, current_insert_sql, current_batch
                        )
                        current_insert_sql = None
                        current_csv_mode = False
                        current_table = ""
                        expected_columns = 0
                    elif line:
                        if current_csv_mode:
                            values = next(csv.reader([line]))
                            parsed_values = [
                                None if value == r"\N" else value for value in values
                            ]
                        else:
                            parsed_values = [
                                _decode_pg_copy_text_value(value)
                                for value in line.split("\t")
                            ]
                        if len(parsed_values) != expected_columns:
                            raise RuntimeError(
                                f"COPY row for {current_table} had {len(parsed_values)} value(s); expected {expected_columns}"
                            )
                        current_batch.append(parsed_values)
                        if len(current_batch) >= _SQLITE_INSERT_BATCH_ROWS:
                            rows_inserted += _flush_copy_batch(
                                cursor, current_insert_sql, current_batch
                            )
                    if progress_callback and _should_report_byte_progress(
                        processed_bytes, total_bytes, len(raw_line)
                    ):
                        _emit_progress(
                            progress_callback,
                            {
                                "phase": "importing_sql",
                                "dialect": "postgresql",
                                "processed_bytes": processed_bytes,
                                "rows_inserted": rows_inserted + len(current_batch),
                                "progress": _byte_progress(
                                    processed_bytes,
                                    total_bytes,
                                    base_progress=base_progress,
                                    progress_span=progress_span,
                                ),
                            },
                        )
                    continue

                copy_match = _PG_COPY_RE.match(line)
                if copy_match:
                    table_name = _sqlite_table_name(copy_match.group(1), warnings)
                    columns = _split_identifier_list(copy_match.group(2).strip())
                    expected_columns = len(columns)
                    sqlite_columns = _sqlite_column_list(copy_match.group(2).strip())
                    _ensure_copy_target_table(cursor, table_name, columns)
                    placeholders = ", ".join("?" for _ in columns)
                    current_insert_sql = f"INSERT INTO {table_name} ({sqlite_columns}) VALUES ({placeholders})"
                    current_csv_mode = _copy_options_are_csv(copy_match.group(0))
                    current_table = table_name
                    current_batch = []

        if current_insert_sql:
            raise RuntimeError(f"Unterminated COPY block for {current_table}")
        _emit_progress(
            progress_callback,
            {
                "phase": "finalizing_sqlite",
                "dialect": "postgresql",
                "processed_bytes": processed_bytes,
                "rows_inserted": rows_inserted + len(current_batch),
                "progress": _FINALIZING_SQLITE_PROGRESS,
            },
        )
        cursor.execute("COMMIT")
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    return rows_inserted, warnings


def _stream_restored_postgres_sql_file_to_sqlite(
    sqlite_path: Path,
    restored_sql_path: Path,
    progress_callback: SqlImportProgressCallback | None = None,
) -> SqlImportResult:
    result = SqlImportResult(dialect="postgresql")
    total_bytes = max(1, restored_sql_path.stat().st_size)
    processed_bytes = 0
    ddl_lines: list[str] = []
    schema_loaded = False
    current_insert_sql: str | None = None
    current_batch: list[list[Any]] = []
    current_csv_mode = False
    current_table = ""
    expected_columns = 0

    def load_schema() -> bool:
        nonlocal schema_loaded
        if schema_loaded:
            return True
        schema_loaded = True
        schema_sql = "".join(ddl_lines)
        if not schema_sql.strip():
            return True
        schema_result = import_sql_to_sqlite(
            sqlite_path, schema_sql, "postgresql", rollback_on_error=False
        )
        result.tables_created = schema_result.tables_created
        result.statements_executed = schema_result.statements_executed
        result.warnings.extend(schema_result.warnings)
        result.warnings.extend(schema_result.errors)
        return schema_result.success and schema_result.tables_created > 0

    conn: sqlite3.Connection | None = None
    try:
        with restored_sql_path.open("rb") as handle:
            for raw_line in handle:
                processed_bytes += len(raw_line)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if line.endswith("\r"):
                    line = line[:-1]

                copy_match = _PG_COPY_RE.match(line)
                if not schema_loaded and not copy_match:
                    ddl_lines.append(line + "\n")
                    continue

                if not schema_loaded:
                    if not load_schema():
                        result.success = False
                        return result
                    conn = sqlite3.connect(str(sqlite_path))
                    conn.isolation_level = None
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys=OFF")
                    cursor.execute("BEGIN")
                elif conn is None:
                    conn = sqlite3.connect(str(sqlite_path))
                    conn.isolation_level = None
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys=OFF")
                    cursor.execute("BEGIN")

                if current_insert_sql:
                    if line == r"\.":
                        result.rows_inserted += _flush_copy_batch(
                            cursor, current_insert_sql, current_batch
                        )
                        current_insert_sql = None
                        current_csv_mode = False
                        current_table = ""
                        expected_columns = 0
                    elif line:
                        if current_csv_mode:
                            values = next(csv.reader([line]))
                            parsed_values = [
                                None if value == r"\N" else value for value in values
                            ]
                        else:
                            parsed_values = [
                                _decode_pg_copy_text_value(value)
                                for value in line.split("\t")
                            ]
                        if len(parsed_values) != expected_columns:
                            raise RuntimeError(
                                f"COPY row for {current_table} had {len(parsed_values)} value(s); expected {expected_columns}"
                            )
                        current_batch.append(parsed_values)
                        if len(current_batch) >= _SQLITE_INSERT_BATCH_ROWS:
                            result.rows_inserted += _flush_copy_batch(
                                cursor, current_insert_sql, current_batch
                            )
                elif copy_match:
                    table_name = _sqlite_table_name(
                        copy_match.group(1), result.warnings
                    )
                    columns = _split_identifier_list(copy_match.group(2).strip())
                    expected_columns = len(columns)
                    sqlite_columns = _sqlite_column_list(copy_match.group(2).strip())
                    _ensure_copy_target_table(cursor, table_name, columns)
                    placeholders = ", ".join("?" for _ in columns)
                    current_insert_sql = f"INSERT INTO {table_name} ({sqlite_columns}) VALUES ({placeholders})"
                    current_csv_mode = _copy_options_are_csv(copy_match.group(0))
                    current_table = table_name
                    current_batch = []

                if progress_callback and _should_report_byte_progress(
                    processed_bytes, total_bytes, len(raw_line)
                ):
                    _emit_progress(
                        progress_callback,
                        {
                            "phase": "importing_sql",
                            "dialect": "postgresql",
                            "processed_bytes": processed_bytes,
                            "rows_inserted": result.rows_inserted + len(current_batch),
                            "tables_created": result.tables_created,
                            "total_statements": result.statements_executed,
                            "statements_executed": result.statements_executed,
                            "progress": _byte_progress(processed_bytes, total_bytes),
                        },
                    )

        if not schema_loaded and not load_schema():
            result.success = False
            return result
        if current_insert_sql:
            raise RuntimeError(f"Unterminated COPY block for {current_table}")
        if conn is not None:
            _emit_progress(
                progress_callback,
                {
                    "phase": "finalizing_sqlite",
                    "dialect": "postgresql",
                    "processed_bytes": processed_bytes,
                    "rows_inserted": result.rows_inserted + len(current_batch),
                    "tables_created": result.tables_created,
                    "total_statements": result.statements_executed,
                    "statements_executed": result.statements_executed,
                    "progress": _FINALIZING_SQLITE_PROGRESS,
                },
            )
            cursor.execute("COMMIT")
        result.success = len(result.errors) == 0
        return result
    except Exception as exc:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        _add_limited_message(
            result.errors, f"Fatal import error: {exc}", _MAX_REPORTED_ERRORS
        )
        result.success = False
        return result
    finally:
        if conn is not None:
            conn.close()


def import_postgres_custom_dump_to_sqlite(
    sqlite_path: Path,
    dump_path: Path,
    progress_callback: SqlImportProgressCallback | None = None,
) -> SqlImportResult:
    task_dir = dump_path.parent
    restored_sql_path = task_dir / "restored.sql"
    temp_sqlite_path = sqlite_path.with_suffix(sqlite_path.suffix + ".importing")
    result = SqlImportResult(dialect="postgresql")

    try:
        restored_sql_path.unlink(missing_ok=True)
        temp_sqlite_path.unlink(missing_ok=True)
        restore_binary_pg_dump_to_sql_file(
            dump_path, restored_sql_path, progress_callback
        )
        _emit_progress(
            progress_callback,
            {
                "phase": "importing_sql",
                "dialect": "postgresql",
                "processed_bytes": 0,
                "rows_inserted": 0,
                "progress": _IMPORTING_SQL_BASE_PROGRESS,
            },
        )
        streamed_result = _stream_restored_postgres_sql_file_to_sqlite(
            temp_sqlite_path, restored_sql_path, progress_callback
        )
        result.tables_created = streamed_result.tables_created
        result.rows_inserted = streamed_result.rows_inserted
        result.statements_executed = streamed_result.statements_executed
        result.warnings.extend(streamed_result.warnings)
        result.errors.extend(streamed_result.errors)
        result.success = streamed_result.success
        if not result.success:
            return result
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_progress(
            progress_callback,
            {
                "phase": "finalizing_sqlite",
                "dialect": "postgresql",
                "processed_bytes": dump_path.stat().st_size,
                "tables_created": result.tables_created,
                "rows_inserted": result.rows_inserted,
                "total_statements": result.statements_executed,
                "statements_executed": result.statements_executed,
                "warnings": result.warnings,
                "errors": result.errors,
                "success": True,
                "progress": 0.99,
            },
        )
        temp_sqlite_path.replace(sqlite_path)
        _emit_progress(
            progress_callback,
            {
                "phase": "completed",
                "dialect": "postgresql",
                "processed_bytes": dump_path.stat().st_size,
                "tables_created": result.tables_created,
                "rows_inserted": result.rows_inserted,
                "total_statements": result.statements_executed,
                "statements_executed": result.statements_executed,
                "warnings": result.warnings,
                "errors": result.errors,
                "success": True,
                "progress": 1.0,
            },
        )
        return result
    except Exception as exc:
        _add_limited_message(
            result.errors, f"Fatal import error: {exc}", _MAX_REPORTED_ERRORS
        )
        result.success = False
        return result
    finally:
        restored_sql_path.unlink(missing_ok=True)
        if not result.success:
            temp_sqlite_path.unlink(missing_ok=True)


def _write_progress(progress_path: Path | None, payload: dict[str, Any]) -> None:
    if progress_path is None:
        return
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(progress_path)


def import_dump_file_to_sqlite(
    sqlite_path: Path,
    dump_path: Path,
    filename: str | None = None,
    progress_path: Path | None = None,
) -> SqlImportResult:
    """Import a text SQL dump or PostgreSQL custom-format dump file."""
    total_bytes = dump_path.stat().st_size if dump_path.exists() else 0

    def _progress(update: dict[str, Any]) -> None:
        payload = {
            "phase": update.get("phase") or "queued",
            "total_bytes": total_bytes,
            "processed_bytes": update.get("processed_bytes", total_bytes),
            "dialect": update.get("dialect") or "generic",
            "total_statements": update.get("total_statements") or 0,
            "statements_executed": update.get("statements_executed") or 0,
            "rows_inserted": update.get("rows_inserted") or 0,
            "tables_created": update.get("tables_created") or 0,
            "warnings": update.get("warnings") or [],
            "errors": update.get("errors") or [],
            "success": bool(update.get("success")),
            "progress": update.get("progress"),
        }
        _write_progress(progress_path, payload)

    try:
        _progress({"phase": "staging_upload", "processed_bytes": total_bytes})
        if detect_binary_pg_dump_file(dump_path):
            result = import_postgres_custom_dump_to_sqlite(
                sqlite_path, dump_path, _progress
            )
        else:
            content = dump_path.read_bytes()
            sql_text = decode_sql_dump_bytes(content)
            dialect = detect_sql_dialect(sql_text, filename or dump_path.name)
            result = import_sql_to_sqlite(sqlite_path, sql_text, dialect, _progress)
        _progress(
            {
                "phase": "completed" if result.success else "failed",
                "dialect": result.dialect,
                "total_statements": result.statements_executed,
                "statements_executed": result.statements_executed,
                "rows_inserted": result.rows_inserted,
                "tables_created": result.tables_created,
                "warnings": result.warnings,
                "errors": result.errors,
                "success": result.success,
            }
        )
        return result
    except Exception as exc:
        result = SqlImportResult(success=False, errors=[f"Fatal import error: {exc}"])
        _progress({"phase": "failed", "errors": result.errors})
        return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="Import a SQL dump into SQLite")
    parser.add_argument("--sqlite-path", required=True)
    parser.add_argument("--dump-path", required=True)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--progress-path", default=None)
    args = parser.parse_args()

    result = import_dump_file_to_sqlite(
        Path(args.sqlite_path),
        Path(args.dump_path),
        args.filename,
        Path(args.progress_path) if args.progress_path else None,
    )
    print(json.dumps(asdict(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
