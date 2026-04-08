"""Import SQL dumps (PostgreSQL, MySQL, generic) into a workspace SQLite database.

Uses *sqlglot* for robust SQL transpilation between dialects instead of
hand-rolled regex transforms.  Only non-SQL constructs (psql protocol
commands, MySQL directives, binary format detection) are handled with
lightweight pre-processing before handing off to sqlglot.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import sqlglot
from sqlglot.errors import ErrorLevel

SqlImportDialect = Literal["postgresql", "mysql", "sqlite", "generic"]

_MAX_IMPORT_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# Magic bytes for binary pg_dump custom format.
_PG_DUMP_MAGIC = b"PGDMP"

# Map our dialect names -> sqlglot dialect identifiers.
_SQLGLOT_READ_DIALECT: dict[SqlImportDialect, str] = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "generic": "sqlite",  # best-effort: already close to SQLite
}


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


# ---------------------------------------------------------------------------
# Dialect detection (heuristic on raw text before parsing)
# ---------------------------------------------------------------------------


def detect_sql_dialect(text: str) -> SqlImportDialect:
    sample = text[:8000].lower()
    pg_signals = (
        "pg_catalog",
        "pg_dump",
        "\\connect",
        "create sequence",
        "nextval(",
        "set search_path",
        "set default_tablespace",
        "copy ",
        "::text",
        "::integer",
        "::boolean",
        "::timestamp",
        "without time zone",
        "with time zone",
    )
    mysql_signals = (
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
    r"COPY\s+(?:\"?(\w+)\"?)\s*\(([^)]+)\)\s+FROM\s+stdin",
    re.IGNORECASE,
)

# Psql backslash commands and pg_catalog function calls (match entire line).
_PG_STRIP_LINE_RE = re.compile(
    r"^(?:\\connect\b.*|\\\..*|SELECT\s+pg_catalog\..*)$",
    re.IGNORECASE | re.MULTILINE,
)

# MySQL conditional comments: /*!40101 SET ... */;
_MYSQL_CONDITIONAL_RE = re.compile(r"/\*![\d ]*.*?\*/;?", re.DOTALL)

# Security: ATTACH DATABASE must never reach the executor.
_ATTACH_DB_RE = re.compile(r"\bATTACH\s+(?:DATABASE\s+)?'", re.IGNORECASE)

# Post-processing: sqlglot quirks that need patching for valid SQLite.
# UBIGINT / USMALLINT / UTINYINT are not real SQLite types.
_UNSIGNED_INT_TYPE_RE = re.compile(r"\bU(?:BIG|SMALL|TINY)?INT\b", re.IGNORECASE)
# NULLS FIRST / NULLS LAST not supported in older SQLite.
_NULLS_ORDER_RE = re.compile(r"\s+NULLS\s+(?:FIRST|LAST)\b", re.IGNORECASE)


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

        table_name = match.group(1)
        columns = match.group(2).strip()

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
            warnings.append(f"Unterminated COPY block for {table_name}; skipped")
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
            values = line.split("\t")
            escaped: list[str] = []
            for v in values:
                if v == "\\N":
                    escaped.append("NULL")
                else:
                    escaped.append(
                        "'" + v.replace("'", "''").replace("\\\\", "\\") + "'"
                    )
            result_parts.append(
                f'INSERT INTO "{table_name}" ({columns}) VALUES ({", ".join(escaped)});\n'
            )
            row_count += 1

        if row_count:
            warnings.append(f"Converted COPY block for {table_name}: {row_count} rows")

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

    # Remove ATTACH DATABASE (security).
    sql = _ATTACH_DB_RE.sub("-- blocked: ATTACH ", sql)

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
    return stmt.strip()


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


def _transpile_statements(
    sql: str, dialect: SqlImportDialect, warnings: list[str]
) -> list[str]:
    """Transpile SQL text from source dialect to SQLite using sqlglot.

    Returns a list of SQLite-compatible SQL strings.
    """
    read_dialect = _SQLGLOT_READ_DIALECT.get(dialect, "sqlite")

    # sqlglot.transpile handles splitting *and* translating in one pass.
    # error_level=WARN collects unsupported-syntax warnings instead of raising.
    try:
        transpiled = sqlglot.transpile(
            sql,
            read=read_dialect,
            write="sqlite",
            error_level=ErrorLevel.WARN,
        )
    except sqlglot.errors.ParseError as exc:
        warnings.append(f"sqlglot parse error (falling back to raw split): {exc}")
        # Fall back: split on semicolons and pass through as-is.
        transpiled = [s.strip() for s in sql.split(";") if s.strip()]

    output: list[str] = []
    for stmt in transpiled:
        stripped = stmt.strip()
        if not stripped or stripped == ";":
            continue
        lower = stripped.lower()
        if any(lower.startswith(p) for p in _SKIP_PREFIXES):
            continue
        # Post-process sqlglot output for SQLite compatibility.
        stripped = _post_process_sqlite(stripped)
        if stripped:
            output.append(stripped)

    return output


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def import_sql_to_sqlite(
    sqlite_path: Path,
    sql_text: str,
    dialect: SqlImportDialect | None = None,
) -> SqlImportResult:
    """Parse, transpile, and execute a SQL dump into the target SQLite database.

    Runs all statements inside a single transaction.  Per-statement errors
    are collected (best-effort) rather than aborting the entire import.
    """
    if dialect is None:
        dialect = detect_sql_dialect(sql_text)

    result = SqlImportResult(dialect=dialect)

    # Pre-process: remove non-SQL constructs.
    cleaned = _preprocess(sql_text, dialect, result.warnings)

    # Transpile via sqlglot.
    statements = _transpile_statements(cleaned, dialect, result.warnings)

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
        }

        for stmt in statements:
            # Final safety check -- never execute ATTACH.
            if _ATTACH_DB_RE.search(stmt):
                result.warnings.append("Blocked ATTACH DATABASE statement")
                continue

            try:
                cursor.execute(stmt)
                result.statements_executed += 1

                lower = stmt.lower().lstrip()
                if lower.startswith(("insert", "replace")):
                    result.rows_inserted += max(cursor.rowcount, 0)
            except sqlite3.Error as exc:
                result.errors.append(f"Statement error: {exc}")
                continue

        tables_after = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        result.tables_created = len(tables_after - tables_before)

        conn.commit()
        result.success = len(result.errors) == 0

    except Exception as exc:
        result.errors.append(f"Fatal import error: {exc}")
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
