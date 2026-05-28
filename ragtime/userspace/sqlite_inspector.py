"""SQLite database inspector primitives for workspace-managed databases.

Pure (synchronous) helpers operating on a workspace's files directory. They
introspect and mutate SQLite databases stored under `.ragtime/db/` using only
the standard library `sqlite3` module. All identifier and type values are
validated against strict allowlists so they can be safely interpolated into
generated DDL; row values always use parameterized statements.

These helpers do not perform authentication or workspace-membership checks.
They are intended to be called from the async service layer
(`ragtime.userspace.service`) which is responsible for enforcing access
control and running the helpers via `asyncio.to_thread`.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

# Managed database location inside `<workspace>/files/`.
MANAGED_DB_DIRNAME = ".ragtime/db"
DEFAULT_DATABASE_NAME = "app.sqlite3"
ALLOWED_DB_EXTENSIONS: frozenset[str] = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})

# Maximum rows returned in a single page; guard against unbounded scans.
MAX_ROW_PAGE_SIZE = 200
DEFAULT_ROW_PAGE_SIZE = 50

# SQLite type affinities we expose in the guided UI. Other declarations
# (e.g. user-supplied "VARCHAR(64)") are tolerated when reading existing
# schemas but cannot be created through the guided forms.
SUPPORTED_COLUMN_TYPES: tuple[str, ...] = ("TEXT", "INTEGER", "REAL", "NUMERIC", "BLOB")
SUPPORTED_COLUMN_TYPES_SET: frozenset[str] = frozenset(SUPPORTED_COLUMN_TYPES)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_IDENTIFIER_PREFIX = "sqlite_"


# ---------------------------------------------------------------------------
# Dataclasses returned to the service layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseSummary:
    name: str
    relative_path: str
    size_bytes: int
    table_count: int
    last_modified_ms: int


@dataclass(frozen=True)
class TableSummary:
    name: str
    type: str  # "table" or "view"
    row_count: int


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    not_null: bool
    primary_key: bool
    primary_key_position: int  # 0 when not a primary key
    default_value: str | None


@dataclass(frozen=True)
class IndexInfo:
    name: str
    unique: bool
    origin: str  # "c" created, "u" unique constraint, "pk" primary key
    columns: list[str]


@dataclass(frozen=True)
class ForeignKeyInfo:
    id: int
    seq: int
    from_column: str
    to_table: str
    to_column: str
    on_update: str
    on_delete: str


@dataclass(frozen=True)
class TableSchema:
    name: str
    type: str
    columns: list[ColumnInfo]
    indexes: list[IndexInfo]
    foreign_keys: list[ForeignKeyInfo]
    sql: str | None


@dataclass(frozen=True)
class RowPage:
    columns: list[ColumnInfo]
    rows: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


@dataclass(frozen=True)
class ColumnDefinition:
    """Guided column specification used for CREATE TABLE / ADD COLUMN."""

    name: str
    type: str
    not_null: bool = False
    primary_key: bool = False
    default_value: str | None = None  # raw SQL literal; validated separately


@dataclass
class TableAlteration:
    """Single guided ALTER TABLE step."""

    op: str  # rename_table | add_column | rename_column | drop_column | change_column_type
    new_table_name: str | None = None
    column: ColumnDefinition | None = None
    column_name: str | None = None
    new_column_name: str | None = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_identifier(name: str, *, kind: str = "identifier") -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{kind} is required")
    if len(cleaned) > 64:
        raise HTTPException(status_code=400, detail=f"{kind} must be 64 characters or fewer")
    if not _IDENTIFIER_RE.match(cleaned):
        raise HTTPException(
            status_code=400,
            detail=(f"{kind} must start with a letter or underscore and contain only letters, digits, or underscores"),
        )
    if cleaned.lower().startswith(_RESERVED_IDENTIFIER_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=f"{kind} cannot start with the reserved prefix 'sqlite_'",
        )
    return cleaned


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def validate_column_type(declared_type: str) -> str:
    cleaned = (declared_type or "").strip().upper()
    if cleaned not in SUPPORTED_COLUMN_TYPES_SET:
        raise HTTPException(
            status_code=400,
            detail=("Column type must be one of: " + ", ".join(SUPPORTED_COLUMN_TYPES)),
        )
    return cleaned


def _validate_default_literal(default_value: str | None) -> str | None:
    """Allow a tightly-scoped set of literal defaults.

    Accepts: numeric literals, NULL, CURRENT_TIMESTAMP / CURRENT_TIME /
    CURRENT_DATE, or single-quoted string literals (without embedded quotes).
    Anything else is rejected so the guided UI cannot inject arbitrary SQL.
    """

    if default_value is None:
        return None
    cleaned = default_value.strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    if upper in {"NULL", "CURRENT_TIMESTAMP", "CURRENT_TIME", "CURRENT_DATE"}:
        return upper
    # Numeric (int / float, optional sign)
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned):
        return cleaned
    # Single-quoted string literal with no embedded quotes/backslashes
    if re.fullmatch(r"'[^'\\\r\n]*'", cleaned):
        return cleaned
    raise HTTPException(
        status_code=400,
        detail=("Default value must be NULL, a number, a single-quoted string, or one of CURRENT_TIMESTAMP/CURRENT_TIME/CURRENT_DATE"),
    )


def _render_column_definition(column: ColumnDefinition) -> str:
    name = validate_identifier(column.name, kind="Column name")
    type_ = validate_column_type(column.type)
    pieces = [quote_identifier(name), type_]
    if column.primary_key:
        pieces.append("PRIMARY KEY")
    if column.not_null:
        pieces.append("NOT NULL")
    default = _validate_default_literal(column.default_value)
    if default is not None:
        pieces.append(f"DEFAULT {default}")
    return " ".join(pieces)


def _render_existing_column_definition(column: ColumnInfo, *, override_type: str | None = None) -> str:
    declared_type = validate_column_type(override_type or column.type or "TEXT")
    spec = ColumnDefinition(
        name=column.name,
        type=declared_type,
        not_null=column.not_null,
        primary_key=column.primary_key,
        default_value=column.default_value,
    )
    return _render_column_definition(spec)


# ---------------------------------------------------------------------------
# Database path / connection helpers
# ---------------------------------------------------------------------------


def _managed_db_root(workspace_files_dir: Path) -> Path:
    return workspace_files_dir / MANAGED_DB_DIRNAME


def _resolve_database_path(workspace_files_dir: Path, db_name: str) -> Path:
    cleaned = (db_name or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Database name is required")
    if "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid database name")
    suffix = Path(cleaned).suffix.lower()
    if suffix not in ALLOWED_DB_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=("Database file must end with one of: " + ", ".join(sorted(ALLOWED_DB_EXTENSIONS))),
        )
    root = _managed_db_root(workspace_files_dir).resolve()
    candidate = (root / cleaned).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid database name") from exc
    return candidate


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(
        str(db_path),
        timeout=5.0,
        detect_types=0,
        isolation_level=None,  # autocommit; explicit transactions where needed
    )
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_table_exists(conn: sqlite3.Connection, table_name: str) -> None:
    if not _table_exists(conn, table_name):
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")


# ---------------------------------------------------------------------------
# Databases
# ---------------------------------------------------------------------------


def list_databases(workspace_files_dir: Path) -> tuple[list[DatabaseSummary], int]:
    """List managed databases under `.ragtime/db/` and total bytes consumed."""

    root = _managed_db_root(workspace_files_dir)
    if not root.exists() or not root.is_dir():
        return [], 0

    summaries: list[DatabaseSummary] = []
    total_bytes = 0
    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in ALLOWED_DB_EXTENSIONS:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        size_bytes = stat.st_size
        total_bytes += size_bytes
        table_count = 0
        try:
            with _connect(entry) as conn:
                row = conn.execute("SELECT count(*) AS cnt FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'").fetchone()
                table_count = int(row["cnt"]) if row else 0
        except sqlite3.DatabaseError:
            table_count = 0
        summaries.append(
            DatabaseSummary(
                name=entry.name,
                relative_path=f"{MANAGED_DB_DIRNAME}/{entry.name}",
                size_bytes=size_bytes,
                table_count=table_count,
                last_modified_ms=int(stat.st_mtime * 1000),
            )
        )
    return summaries, total_bytes


def initialize_database(workspace_files_dir: Path, db_name: str = DEFAULT_DATABASE_NAME) -> DatabaseSummary:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        # Open and close to materialize an empty SQLite file.
        with _connect(db_path):
            pass
    stat = db_path.stat()
    return DatabaseSummary(
        name=db_path.name,
        relative_path=f"{MANAGED_DB_DIRNAME}/{db_path.name}",
        size_bytes=stat.st_size,
        table_count=0,
        last_modified_ms=int(stat.st_mtime * 1000),
    )


def delete_database(workspace_files_dir: Path, db_name: str) -> None:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    db_path.unlink()
    # Best-effort cleanup of WAL/SHM sidecar files.
    for sidecar_suffix in ("-wal", "-shm", "-journal"):
        sidecar = db_path.with_name(db_path.name + sidecar_suffix)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def export_database_copy(workspace_files_dir: Path, db_name: str) -> Path:
    """Return a temporary consistent copy of a managed database."""

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    fd, temp_name = tempfile.mkstemp(prefix="ragtime-sqlite-", suffix=db_path.suffix)
    Path(temp_name).unlink(missing_ok=True)
    # Close the fd from mkstemp before sqlite creates the destination.
    try:
        import os

        os.close(fd)
    except OSError:
        pass
    out_path = Path(temp_name)
    with _connect(db_path) as source:
        dest = sqlite3.connect(str(out_path), timeout=5.0, isolation_level=None)
        try:
            source.backup(dest)
        finally:
            dest.close()
    return out_path


def import_database_file(workspace_files_dir: Path, db_name: str, source_path: Path) -> DatabaseSummary:
    """Validate and copy an uploaded SQLite database into the managed DB dir."""

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=400, detail="Uploaded database file was not readable")
    try:
        with _connect(source_path) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            if row is None or str(row[0]).lower() != "ok":
                raise HTTPException(status_code=400, detail="Uploaded file failed SQLite integrity check")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
    except HTTPException:
        raise
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a valid SQLite database: {exc}") from exc

    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, db_path)
    for sidecar_suffix in ("-wal", "-shm", "-journal"):
        sidecar = db_path.with_name(db_path.name + sidecar_suffix)
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return initialize_database(workspace_files_dir, db_name)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _row_count(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT count(*) AS cnt FROM {quote_identifier(table_name)}").fetchone()
    return int(row["cnt"]) if row else 0


def list_tables(workspace_files_dir: Path, db_name: str) -> list[TableSummary]:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    out: list[TableSummary] = []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY type, name"
        ).fetchall()
        for entry in rows:
            tname = str(entry["name"])
            try:
                count = _row_count(conn, tname)
            except sqlite3.DatabaseError:
                count = 0
            out.append(TableSummary(name=tname, type=str(entry["type"]), row_count=count))
    return out


def _column_info(conn: sqlite3.Connection, table_name: str) -> list[ColumnInfo]:
    rows = conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    columns: list[ColumnInfo] = []
    for row in rows:
        columns.append(
            ColumnInfo(
                name=str(row["name"]),
                type=str(row["type"] or ""),
                not_null=bool(row["notnull"]),
                primary_key=bool(row["pk"]),
                primary_key_position=int(row["pk"]),
                default_value=(str(row["dflt_value"]) if row["dflt_value"] is not None else None),
            )
        )
    return columns


def _index_info(conn: sqlite3.Connection, table_name: str) -> list[IndexInfo]:
    out: list[IndexInfo] = []
    rows = conn.execute(f"PRAGMA index_list({quote_identifier(table_name)})").fetchall()
    for row in rows:
        idx_name = str(row["name"])
        cols_rows = conn.execute(f"PRAGMA index_info({quote_identifier(idx_name)})").fetchall()
        columns = [str(cr["name"]) for cr in cols_rows]
        out.append(
            IndexInfo(
                name=idx_name,
                unique=bool(row["unique"]),
                origin=str(row["origin"]),
                columns=columns,
            )
        )
    return out


def _foreign_keys(conn: sqlite3.Connection, table_name: str) -> list[ForeignKeyInfo]:
    rows = conn.execute(f"PRAGMA foreign_key_list({quote_identifier(table_name)})").fetchall()
    return [
        ForeignKeyInfo(
            id=int(row["id"]),
            seq=int(row["seq"]),
            from_column=str(row["from"]),
            to_table=str(row["table"]),
            to_column=str(row["to"]),
            on_update=str(row["on_update"]),
            on_delete=str(row["on_delete"]),
        )
        for row in rows
    ]


def get_table_schema(workspace_files_dir: Path, db_name: str, table_name: str) -> TableSchema:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        master = conn.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ?",
            (table_name,),
        ).fetchone()
        if master is None:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        return TableSchema(
            name=table_name,
            type=str(master["type"]),
            columns=_column_info(conn, table_name),
            indexes=_index_info(conn, table_name),
            foreign_keys=_foreign_keys(conn, table_name),
            sql=(str(master["sql"]) if master["sql"] is not None else None),
        )


def create_table(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    columns: list[ColumnDefinition],
    *,
    without_rowid: bool = False,
) -> TableSummary:
    if not columns:
        raise HTTPException(status_code=400, detail="At least one column is required")
    table_name = validate_identifier(table_name, kind="Table name")
    seen: set[str] = set()
    pk_count = 0
    rendered_columns: list[str] = []
    for col in columns:
        validated_name = validate_identifier(col.name, kind="Column name")
        lowered = validated_name.lower()
        if lowered in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate column '{validated_name}'")
        seen.add(lowered)
        if col.primary_key:
            pk_count += 1
        rendered_columns.append(_render_column_definition(col))
    if pk_count > 1:
        raise HTTPException(
            status_code=400,
            detail="Only one column-level PRIMARY KEY is supported (composite keys are not exposed yet)",
        )

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = f"CREATE TABLE {quote_identifier(table_name)} (" + ", ".join(rendered_columns) + ")" + (" WITHOUT ROWID" if without_rowid else "")
    with _connect(db_path) as conn:
        if _table_exists(conn, table_name):
            raise HTTPException(status_code=409, detail=f"Table '{table_name}' already exists")
        try:
            conn.execute(sql)
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to create table: {exc}") from exc
        return TableSummary(name=table_name, type="table", row_count=0)


def drop_table(workspace_files_dir: Path, db_name: str, table_name: str) -> None:
    table_name = validate_identifier(table_name, kind="Table name")
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        conn.execute(f"DROP TABLE {quote_identifier(table_name)}")


def apply_table_alterations(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    alterations: list[TableAlteration],
) -> TableSchema:
    if not alterations:
        raise HTTPException(status_code=400, detail="At least one alteration is required")
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    current_table = validate_identifier(table_name, kind="Table name")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, current_table)
        for step in alterations:
            op = (step.op or "").strip().lower()
            if op == "rename_table":
                new_name = validate_identifier(step.new_table_name or "", kind="New table name")
                conn.execute(f"ALTER TABLE {quote_identifier(current_table)} RENAME TO {quote_identifier(new_name)}")
                current_table = new_name
            elif op == "add_column":
                if step.column is None:
                    raise HTTPException(status_code=400, detail="add_column requires a column definition")
                col_sql = _render_column_definition(step.column)
                if step.column.primary_key:
                    raise HTTPException(
                        status_code=400,
                        detail="SQLite cannot add a PRIMARY KEY column to an existing table",
                    )
                conn.execute(f"ALTER TABLE {quote_identifier(current_table)} ADD COLUMN {col_sql}")
            elif op == "rename_column":
                old_col = validate_identifier(step.column_name or "", kind="Column name")
                new_col = validate_identifier(step.new_column_name or "", kind="New column name")
                try:
                    conn.execute(f"ALTER TABLE {quote_identifier(current_table)} RENAME COLUMN {quote_identifier(old_col)} TO {quote_identifier(new_col)}")
                except sqlite3.OperationalError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"RENAME COLUMN requires SQLite 3.25 or newer; the embedded engine reported: {exc}"),
                    ) from exc
            elif op == "drop_column":
                col_name = validate_identifier(step.column_name or "", kind="Column name")
                try:
                    conn.execute(f"ALTER TABLE {quote_identifier(current_table)} DROP COLUMN {quote_identifier(col_name)}")
                except sqlite3.OperationalError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"DROP COLUMN requires SQLite 3.35 or newer and the column must not be referenced by indexes/foreign keys: {exc}"),
                    ) from exc
            elif op == "change_column_type":
                col_name = validate_identifier(step.column_name or "", kind="Column name")
                if step.column is None:
                    raise HTTPException(status_code=400, detail="change_column_type requires a column definition with the new type")
                new_type = validate_column_type(step.column.type)
                if _row_count(conn, current_table) > 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Column type changes are only allowed on empty tables to avoid destructive data conversion",
                    )
                if _foreign_keys(conn, current_table):
                    raise HTTPException(status_code=400, detail="Column type changes are not supported for tables with foreign keys")
                user_indexes = [idx for idx in _index_info(conn, current_table) if idx.origin != "pk"]
                if user_indexes:
                    raise HTTPException(status_code=400, detail="Column type changes are not supported for tables with indexes")
                columns = _column_info(conn, current_table)
                if not any(col.name == col_name for col in columns):
                    raise HTTPException(status_code=404, detail=f"Column '{col_name}' not found")
                rendered = [_render_existing_column_definition(col, override_type=(new_type if col.name == col_name else None)) for col in columns]
                tmp_name = validate_identifier(f"__rt_tmp_{current_table[:48]}", kind="Temporary table name")
                counter = 1
                while _table_exists(conn, tmp_name):
                    tmp_name = validate_identifier(
                        f"__rt_tmp_{counter}_{current_table[:40]}",
                        kind="Temporary table name",
                    )
                    counter += 1
                conn.execute(f"CREATE TABLE {quote_identifier(tmp_name)} (" + ", ".join(rendered) + ")")
                conn.execute(f"DROP TABLE {quote_identifier(current_table)}")
                conn.execute(f"ALTER TABLE {quote_identifier(tmp_name)} RENAME TO {quote_identifier(current_table)}")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported alteration '{step.op}'")

        return TableSchema(
            name=current_table,
            type="table",
            columns=_column_info(conn, current_table),
            indexes=_index_info(conn, current_table),
            foreign_keys=_foreign_keys(conn, current_table),
            sql=_table_sql(conn, current_table),
        )


def _table_sql(conn: sqlite3.Connection, table_name: str) -> str | None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?",
        (table_name,),
    ).fetchone()
    return None if row is None or row["sql"] is None else str(row["sql"])


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _primary_key_columns(columns: list[ColumnInfo]) -> list[ColumnInfo]:
    return sorted(
        [c for c in columns if c.primary_key],
        key=lambda c: c.primary_key_position,
    )


def _row_identifier(
    columns: list[ColumnInfo],
    row_key: dict[str, Any] | None,
) -> tuple[str, list[Any]]:
    """Build a WHERE clause for a single row, using primary key when available.

    When no PRIMARY KEY exists we fall back to SQLite's `rowid` (the caller is
    expected to include it under the special key `_rowid` in `row_key`).
    """

    pks = _primary_key_columns(columns)
    if pks:
        if not row_key or not all(pk.name in row_key for pk in pks):
            raise HTTPException(
                status_code=400,
                detail="Primary key values for all key columns are required",
            )
        clauses = [f"{quote_identifier(pk.name)} = ?" for pk in pks]
        params = [row_key[pk.name] for pk in pks]
        return " AND ".join(clauses), params
    if not row_key or "_rowid" not in row_key:
        raise HTTPException(
            status_code=400,
            detail="Table has no PRIMARY KEY; supply '_rowid' to identify the row",
        )
    return "rowid = ?", [row_key["_rowid"]]


def list_rows(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    *,
    limit: int = DEFAULT_ROW_PAGE_SIZE,
    offset: int = 0,
    order_by: str | None = None,
    order_direction: str = "asc",
) -> RowPage:
    started = time.perf_counter()
    if limit <= 0:
        limit = DEFAULT_ROW_PAGE_SIZE
    if limit > MAX_ROW_PAGE_SIZE:
        limit = MAX_ROW_PAGE_SIZE
    if offset < 0:
        offset = 0
    direction = (order_direction or "asc").strip().lower()
    if direction not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="order_direction must be 'asc' or 'desc'")

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")

    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        columns = _column_info(conn, table_name)
        column_names = {c.name for c in columns}
        order_clause = ""
        if order_by:
            if order_by not in column_names:
                raise HTTPException(status_code=400, detail=f"Unknown order_by column '{order_by}'")
            order_clause = f" ORDER BY {quote_identifier(order_by)} {direction.upper()}"
        elif not _primary_key_columns(columns):
            order_clause = " ORDER BY rowid ASC"
        else:
            pk_cols = ", ".join(quote_identifier(pk.name) for pk in _primary_key_columns(columns))
            order_clause = f" ORDER BY {pk_cols} {direction.upper()}"

        total = _row_count(conn, table_name)
        select_columns = ", ".join(quote_identifier(c.name) for c in columns)
        has_pk = bool(_primary_key_columns(columns))
        rowid_clause = "" if has_pk else ", rowid AS _rowid"
        sql = f"SELECT {select_columns}{rowid_clause} FROM {quote_identifier(table_name)}{order_clause} LIMIT ? OFFSET ?"
        rows = conn.execute(sql, (limit, offset)).fetchall()
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            record: dict[str, Any] = {}
            for col in columns:
                value = row[col.name]
                record[col.name] = _coerce_value(value)
            if not has_pk:
                record["_rowid"] = int(row["_rowid"])
            out_rows.append(record)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RowPage(columns=columns, rows=out_rows, total=total, limit=limit, offset=offset, elapsed_ms=elapsed_ms)


def _coerce_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__blob__": True, "size": len(bytes(value))}
    return value


def insert_row(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        columns = _column_info(conn, table_name)
        column_names = {c.name for c in columns}
        provided = {k: v for k, v in (values or {}).items() if k in column_names}
        if not provided:
            # Insert with all defaults
            try:
                cur = conn.execute(f"INSERT INTO {quote_identifier(table_name)} DEFAULT VALUES")
            except sqlite3.DatabaseError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            inserted_rowid = cur.lastrowid
        else:
            cols_sql = ", ".join(quote_identifier(k) for k in provided.keys())
            placeholders = ", ".join("?" for _ in provided)
            try:
                cur = conn.execute(
                    f"INSERT INTO {quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders})",
                    list(provided.values()),
                )
            except sqlite3.DatabaseError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            inserted_rowid = cur.lastrowid

        return _fetch_row_by_rowid(conn, table_name, columns, inserted_rowid)


def update_row(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    row_key: dict[str, Any],
    updates: dict[str, Any],
) -> dict[str, Any]:
    if not updates:
        raise HTTPException(status_code=400, detail="No column updates supplied")
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        columns = _column_info(conn, table_name)
        column_names = {c.name for c in columns}
        applicable = {k: v for k, v in updates.items() if k in column_names}
        if not applicable:
            raise HTTPException(status_code=400, detail="No updatable columns supplied")
        where_clause, where_params = _row_identifier(columns, row_key)
        set_clause = ", ".join(f"{quote_identifier(k)} = ?" for k in applicable.keys())
        try:
            cur = conn.execute(
                f"UPDATE {quote_identifier(table_name)} SET {set_clause} WHERE {where_clause}",
                list(applicable.values()) + list(where_params),
            )
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Row not found")
        # Re-fetch row via the same identifier
        select_cols = ", ".join(quote_identifier(c.name) for c in columns)
        has_pk = bool(_primary_key_columns(columns))
        rowid_clause = "" if has_pk else ", rowid AS _rowid"
        row = conn.execute(
            f"SELECT {select_cols}{rowid_clause} FROM {quote_identifier(table_name)} WHERE {where_clause}",
            where_params,
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Row not found after update")
        return _row_to_dict(row, columns, include_rowid=not has_pk)


def delete_row(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    row_key: dict[str, Any],
) -> None:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        columns = _column_info(conn, table_name)
        where_clause, where_params = _row_identifier(columns, row_key)
        cur = conn.execute(
            f"DELETE FROM {quote_identifier(table_name)} WHERE {where_clause}",
            where_params,
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Row not found")


def export_table_csv(workspace_files_dir: Path, db_name: str, table_name: str) -> str:
    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        _ensure_table_exists(conn, table_name)
        columns = _column_info(conn, table_name)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow([c.name for c in columns])
        select_columns = ", ".join(quote_identifier(c.name) for c in columns)
        for row in conn.execute(f"SELECT {select_columns} FROM {quote_identifier(table_name)}"):
            writer.writerow([row[c.name] for c in columns])
        return output.getvalue()


def import_table_csv(
    workspace_files_dir: Path,
    db_name: str,
    table_name: str,
    csv_text: str,
    *,
    replace: bool = False,
) -> TableSummary:
    table_name = validate_identifier(table_name, kind="Table name")
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must include a header row")
    field_names = [validate_identifier(name or "", kind="CSV column name") for name in reader.fieldnames]
    rows = list(reader)

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        if _table_exists(conn, table_name):
            if replace:
                conn.execute(f"DROP TABLE {quote_identifier(table_name)}")
            else:
                columns = _column_info(conn, table_name)
                existing = {c.name for c in columns}
                missing = [name for name in field_names if name not in existing]
                if missing:
                    raise HTTPException(status_code=400, detail=f"CSV column(s) not found in table: {', '.join(missing)}")
        if not _table_exists(conn, table_name):
            column_sql = ", ".join(_render_column_definition(ColumnDefinition(name=name, type="TEXT")) for name in field_names)
            conn.execute(f"CREATE TABLE {quote_identifier(table_name)} ({column_sql})")

        if rows:
            cols_sql = ", ".join(quote_identifier(name) for name in field_names)
            placeholders = ", ".join("?" for _ in field_names)
            conn.executemany(
                f"INSERT INTO {quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders})",
                [[row.get(name) for name in field_names] for row in rows],
            )
        return TableSummary(name=table_name, type="table", row_count=_row_count(conn, table_name))


def execute_readonly_query(
    workspace_files_dir: Path,
    db_name: str,
    sql: str,
    *,
    max_rows: int = 200,
) -> QueryResult:
    cleaned = (sql or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="SQL query is required")
    statement = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if ";" in statement:
        raise HTTPException(status_code=400, detail="Only one SQL statement can be executed at a time")
    first_token = statement.split(None, 1)[0].lower() if statement.split(None, 1) else ""
    if first_token not in {"select", "with", "pragma", "explain"}:
        raise HTTPException(status_code=400, detail="The SQL console only supports read-only SELECT, WITH, PRAGMA, and EXPLAIN statements")
    if max_rows <= 0 or max_rows > 500:
        max_rows = 200

    db_path = _resolve_database_path(workspace_files_dir, db_name)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    with _connect(db_path) as conn:
        conn.execute("PRAGMA query_only = ON")
        try:
            cursor = conn.execute(statement)
            names = [str(desc[0]) for desc in (cursor.description or [])]
            raw_rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.DatabaseError as exc:
            raise HTTPException(status_code=400, detail=f"Query failed: {exc}") from exc
        truncated = len(raw_rows) > max_rows
        rows = raw_rows[:max_rows]
        records = [{name: _coerce_value(row[name]) for name in names} for row in rows]
        return QueryResult(columns=names, rows=records, row_count=len(records), truncated=truncated)


def _fetch_row_by_rowid(
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[ColumnInfo],
    rowid: int | None,
) -> dict[str, Any]:
    if rowid is None:
        raise HTTPException(status_code=500, detail="Inserted row id was not returned by SQLite")
    select_cols = ", ".join(quote_identifier(c.name) for c in columns)
    has_pk = bool(_primary_key_columns(columns))
    rowid_clause = "" if has_pk else ", rowid AS _rowid"
    row = conn.execute(
        f"SELECT {select_cols}{rowid_clause} FROM {quote_identifier(table_name)} WHERE rowid = ?",
        (rowid,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=500, detail="Inserted row could not be retrieved")
    return _row_to_dict(row, columns, include_rowid=not has_pk)


def _row_to_dict(
    row: sqlite3.Row,
    columns: list[ColumnInfo],
    *,
    include_rowid: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {col.name: _coerce_value(row[col.name]) for col in columns}
    if include_rowid:
        record["_rowid"] = int(row["_rowid"])
    return record


# Placeholder to silence unused-import linting in some toolchains.
_FIELD = field
