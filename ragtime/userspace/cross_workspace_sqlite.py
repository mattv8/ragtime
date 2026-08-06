from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

from fastapi import HTTPException

from ragtime.core.workspace_ops import normalize_runtime_file_path
from ragtime.userspace import sqlite_inspector

LOGGER = logging.getLogger(__name__)

_SQLITE_PATH = f"{sqlite_inspector.MANAGED_DB_DIRNAME}/{sqlite_inspector.DEFAULT_DATABASE_NAME}"
_SQLITE_URI_SUFFIX = "?mode=ro"
_QUERY_MAX_SQL_BYTES = 32 * 1024
_QUERY_MAX_RESPONSE_BYTES = 1024 * 1024
_MUTATION_MAX_PAYLOAD_BYTES = 1024 * 1024
_MAX_BIND_PARAMETERS = 500
_DEFAULT_QUERY_TIMEOUT_SECONDS = 2.0
_DEFAULT_BUSY_TIMEOUT_MS = 5000
_RETRY_DELAYS_SECONDS = (0.1, 0.25)
_AUTHORIZE_DENY = getattr(sqlite3, "SQLITE_DENY", 1)
_AUTHORIZE_OK = getattr(sqlite3, "SQLITE_OK", 0)
_ALLOWED_LEADING_TOKENS = {"SELECT", "WITH", "EXPLAIN"}
_DENIED_FUNCTIONS = {"load_extension", "writefile", "readfile", "fts3_tokenizer"}
MutationKind = Literal["insert", "upsert", "update", "delete"]
_MUTATION_KINDS: frozenset[MutationKind] = frozenset(("insert", "upsert", "update", "delete"))
_SUPPORTED_VALUE_TYPES = (type(None), bool, int, float, str, bytes)
_SQLITE_BUSY_CODES = {
    getattr(sqlite3, "SQLITE_BUSY", 5),
    getattr(sqlite3, "SQLITE_LOCKED", 6),
}
_SQLITE_INTERRUPT = getattr(sqlite3, "SQLITE_INTERRUPT", 9)
_SQLITE_CANTOPEN = getattr(sqlite3, "SQLITE_CANTOPEN", 14)
_SQLITE_INT64_MIN = -(2**63)
_SQLITE_INT64_MAX = 2**63 - 1
_MUTATION_LOCKS: weakref.WeakValueDictionary[Path, "_PathLockBox"] = weakref.WeakValueDictionary()

_DENIED_ACTION_CODES = {
    code
    for code in (
        getattr(sqlite3, "SQLITE_INSERT", None),
        getattr(sqlite3, "SQLITE_UPDATE", None),
        getattr(sqlite3, "SQLITE_DELETE", None),
        getattr(sqlite3, "SQLITE_CREATE_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", None),
        getattr(sqlite3, "SQLITE_CREATE_VTABLE", None),
        getattr(sqlite3, "SQLITE_DROP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_VIEW", None),
        getattr(sqlite3, "SQLITE_DROP_VTABLE", None),
        getattr(sqlite3, "SQLITE_ALTER_TABLE", None),
        getattr(sqlite3, "SQLITE_TRANSACTION", None),
        getattr(sqlite3, "SQLITE_SAVEPOINT", None),
        getattr(sqlite3, "SQLITE_ATTACH", None),
        getattr(sqlite3, "SQLITE_DETACH", None),
        getattr(sqlite3, "SQLITE_PRAGMA", None),
        getattr(sqlite3, "SQLITE_REINDEX", None),
        getattr(sqlite3, "SQLITE_ANALYZE", None),
    )
    if code is not None
}

_SAFE_MESSAGES = {
    "invalid_sql": "SQL input is invalid.",
    "invalid_parameters": "Parameters are invalid.",
    "sql_too_large": "SQL input exceeds the maximum size.",
    "payload_too_large": "Mutation payload exceeds the maximum size.",
    "response_too_large": "Query response exceeds the maximum size.",
    "row_limit_invalid": "Row limit must be between 1 and 500.",
    "database_not_found": "Managed SQLite database is unavailable.",
    "sqlite_busy": "Managed SQLite database is busy.",
    "query_timeout": "SQLite operation timed out.",
    "sql_not_allowed": "SQL statement is not allowed.",
    "query_failed": "SQLite query failed.",
    "mutation_failed": "SQLite mutation failed.",
    "audit_unavailable": "Mutation audit could not be recorded.",
}


@dataclass
class CrossWorkspaceSqlitePolicy:
    query_timeout_seconds: float = _DEFAULT_QUERY_TIMEOUT_SECONDS
    busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False


@dataclass(frozen=True)
class MutationOperation:
    kind: MutationKind
    table: str
    values: Mapping[str, Any] | None = None
    where: Mapping[str, Any] | None = None
    conflict_columns: Sequence[str] | None = None


@dataclass(frozen=True)
class MutationOperationResult:
    kind: MutationKind
    rowcount: int
    lastrowid: int | None = None


@dataclass(frozen=True)
class MutationResult:
    operations: list[MutationOperationResult]
    fingerprint: str


@dataclass(frozen=True)
class AuditIdentityContext:
    actor_id: str
    actor_type: str = "user"
    request_id: str | None = None


@dataclass(frozen=True)
class AuditIntent:
    fingerprint: str
    operation_count: int
    identity_context: AuditIdentityContext


@dataclass(frozen=True)
class AuditOutcome:
    fingerprint: str
    operation_count: int
    identity_context: AuditIdentityContext
    status: str
    error_code: str | None = None


@dataclass
class _PreparedMutationOperation:
    kind: MutationKind
    table: str
    values: dict[str, Any] = field(default_factory=dict)
    where: dict[str, Any] = field(default_factory=dict)
    conflict_columns: list[str] = field(default_factory=list)


@dataclass
class _PathLockBox:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CrossWorkspaceSqliteError(Exception):
    def __init__(self, code: str):
        self.code = code
        self.safe_message = _SAFE_MESSAGES[code]
        super().__init__(self.safe_message)


class CrossWorkspaceSqliteBroker:
    def __init__(self, policy: CrossWorkspaceSqlitePolicy):
        self.policy = policy

    async def query(
        self,
        workspace_files_dir: Path,
        sql: str,
        *,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
        max_rows: int = 200,
    ) -> QueryResult:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1 or max_rows > 500:
            raise CrossWorkspaceSqliteError("row_limit_invalid")
        statement = _validate_query_sql(sql)
        bindings = _normalize_bind_parameters(parameters)
        db_path = _resolve_managed_db_path(workspace_files_dir)
        return await asyncio.to_thread(
            _execute_query,
            db_path,
            statement,
            bindings,
            max_rows,
            self.policy,
        )

    async def mutate(
        self,
        workspace_files_dir: Path,
        operations: Sequence[MutationOperation],
        *,
        audit_context: AuditIdentityContext,
        audit_intent_callback: Callable[[AuditIntent], Any],
        audit_outcome_callback: Callable[[AuditOutcome], Any] | None = None,
    ) -> MutationResult:
        prepared = _prepare_mutation_operations(operations)
        _validate_mutation_payload_size(prepared)
        fingerprint = self.fingerprint_operations(prepared)
        intent = AuditIntent(
            fingerprint=fingerprint,
            operation_count=len(prepared),
            identity_context=audit_context,
        )
        if not await _call_audit_intent(audit_intent_callback, intent):
            raise CrossWorkspaceSqliteError("audit_unavailable")

        db_path = _resolve_managed_db_path(workspace_files_dir)
        lock_box = _mutation_lock_for(db_path)
        error: CrossWorkspaceSqliteError | None = None
        results: list[MutationOperationResult] | None = None
        async with lock_box.lock:
            try:
                results = await asyncio.to_thread(_execute_mutation_transaction, db_path, prepared, self.policy)
            except CrossWorkspaceSqliteError as exc:
                error = exc
            except Exception as exc:
                error = CrossWorkspaceSqliteError("mutation_failed")
                error.__cause__ = exc

        if error is not None:
            await _call_audit_outcome(
                audit_outcome_callback,
                AuditOutcome(
                    fingerprint=fingerprint,
                    operation_count=len(prepared),
                    identity_context=audit_context,
                    status=_outcome_status_for_error(error.code),
                    error_code=error.code,
                ),
            )
            raise error

        await _call_audit_outcome(
            audit_outcome_callback,
            AuditOutcome(
                fingerprint=fingerprint,
                operation_count=len(prepared),
                identity_context=audit_context,
                status="committed",
                error_code=None,
            ),
        )
        return MutationResult(operations=results or [], fingerprint=fingerprint)

    async def checkpoint(self, workspace_files_dir: Path) -> None:
        db_path = _resolve_managed_db_path(workspace_files_dir)
        lock_box = _mutation_lock_for(db_path)
        async with lock_box.lock:
            await asyncio.to_thread(_execute_wal_checkpoint, db_path, self.policy)

    def fingerprint_operations(self, operations: Sequence[MutationOperation | _PreparedMutationOperation]) -> str:
        canonical = {
            "version": 1,
            "operations": [
                {
                    "index": index,
                    "kind": operation.kind,
                    "table": operation.table,
                    "value_columns": sorted((operation.values or {}).keys()),
                    "predicate_columns": sorted((operation.where or {}).keys()),
                    "conflict_columns": sorted(operation.conflict_columns or []),
                }
                for index, operation in enumerate(operations)
            ],
        }
        payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _mutation_lock_for(db_path: Path) -> _PathLockBox:
    canonical = db_path.resolve()
    lock_box = _MUTATION_LOCKS.get(canonical)
    if lock_box is None:
        lock_box = _PathLockBox()
        _MUTATION_LOCKS[canonical] = lock_box
    return lock_box


def _resolve_managed_db_path(workspace_files_dir: Path) -> Path:
    clean_path = normalize_runtime_file_path(_SQLITE_PATH, enforce_sqlite_managed=True)
    workspace_root = workspace_files_dir.resolve()
    db_path = (workspace_root / clean_path).resolve()
    try:
        db_path.relative_to(workspace_root)
    except ValueError as exc:
        raise CrossWorkspaceSqliteError("database_not_found") from exc
    return db_path


def _validate_query_sql(sql: str) -> str:
    if not isinstance(sql, str):
        raise CrossWorkspaceSqliteError("invalid_sql")
    encoded = (sql or "").encode("utf-8")
    if len(encoded) > _QUERY_MAX_SQL_BYTES:
        raise CrossWorkspaceSqliteError("sql_too_large")
    cleaned = (sql or "").strip()
    if not cleaned:
        raise CrossWorkspaceSqliteError("invalid_sql")
    statement = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if not statement or ";" in statement:
        raise CrossWorkspaceSqliteError("invalid_sql")
    leading_token = statement.split(None, 1)[0].upper() if statement.split(None, 1) else ""
    if leading_token not in _ALLOWED_LEADING_TOKENS:
        raise CrossWorkspaceSqliteError("sql_not_allowed")
    return statement


def _normalize_bind_parameters(parameters: Sequence[Any] | Mapping[str, Any] | None) -> Sequence[Any] | Mapping[str, Any]:
    if parameters is None:
        return ()
    if isinstance(parameters, Mapping):
        normalized: dict[str, Any] = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise CrossWorkspaceSqliteError("invalid_parameters")
            normalized[key] = _validate_bind_value(value)
        if len(normalized) > _MAX_BIND_PARAMETERS:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        return normalized
    if isinstance(parameters, (str, bytes, bytearray)) or not isinstance(parameters, Sequence):
        raise CrossWorkspaceSqliteError("invalid_parameters")
    normalized_values = tuple(_validate_bind_value(value) for value in parameters)
    if len(normalized_values) > _MAX_BIND_PARAMETERS:
        raise CrossWorkspaceSqliteError("invalid_parameters")
    return normalized_values


def _validate_bind_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < _SQLITE_INT64_MIN or value > _SQLITE_INT64_MAX:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        return value
    if isinstance(value, _SUPPORTED_VALUE_TYPES):
        return value
    raise CrossWorkspaceSqliteError("invalid_parameters")


def _authorizer(action_code: int, param1: str | None, param2: str | None, _db_name: str | None, _trigger: str | None) -> int:
    try:
        if action_code in _DENIED_ACTION_CODES:
            return _AUTHORIZE_DENY
        if action_code == getattr(sqlite3, "SQLITE_FUNCTION", -1):
            names = {str(value or "").lower() for value in (param1, param2)}
            if names & _DENIED_FUNCTIONS:
                return _AUTHORIZE_DENY
    except Exception:
        return _AUTHORIZE_DENY
    return _AUTHORIZE_OK


def _deadline_progress_handler(deadline: float) -> Callable[[], int]:
    def _progress() -> int:
        try:
            return 1 if time.monotonic() >= deadline else 0
        except Exception:
            return 1

    return _progress


def _connect_readonly(db_path: Path, policy: CrossWorkspaceSqlitePolicy) -> sqlite3.Connection:
    if not db_path.exists():
        raise CrossWorkspaceSqliteError("database_not_found")
    try:
        connection = sqlite3.connect(
            db_path.resolve().as_uri() + _SQLITE_URI_SUFFIX,
            uri=True,
            timeout=max(policy.busy_timeout_ms, 1) / 1000,
            isolation_level=None,
            detect_types=0,
        )
    except sqlite3.Error as exc:
        raise _map_sqlite_error(exc, default_code="database_not_found") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _execute_query(
    db_path: Path,
    statement: str,
    parameters: Sequence[Any] | Mapping[str, Any],
    max_rows: int,
    policy: CrossWorkspaceSqlitePolicy,
) -> QueryResult:
    connection = _connect_readonly(db_path, policy)
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(policy.busy_timeout_ms)}")
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_authorizer)
        connection.set_progress_handler(_deadline_progress_handler(time.monotonic() + float(policy.query_timeout_seconds)), 1000)
        try:
            cursor = connection.execute(statement, parameters)
            column_names = _deduplicate_column_names([str(description[0]) for description in (cursor.description or ())])
            rows: list[dict[str, Any]] = []
            response_size = _serialized_utf8_size(column_names)
            truncated = False
            for raw_row in cursor:
                row = {name: _sanitize_cell_value(raw_row[index]) for index, name in enumerate(column_names)}
                response_size += _serialized_utf8_size(row)
                if response_size > _QUERY_MAX_RESPONSE_BYTES:
                    raise CrossWorkspaceSqliteError("response_too_large")
                if len(rows) >= max_rows:
                    truncated = True
                    break
                rows.append(row)
        except CrossWorkspaceSqliteError:
            raise
        except sqlite3.Error as exc:
            raise _map_sqlite_error(exc, default_code="query_failed") from exc
        return QueryResult(columns=column_names, rows=rows, row_count=len(rows), truncated=truncated)
    finally:
        connection.close()


def _sanitize_cell_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__blob__": True, "size": len(bytes(value))}
    return value


def _serialized_utf8_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _deduplicate_column_names(column_names: Sequence[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    duplicate_counts: dict[str, int] = {}
    for raw_name in column_names:
        base_name = raw_name or "column"
        candidate = base_name
        if candidate in seen:
            duplicate_counts[base_name] = duplicate_counts.get(base_name, 1) + 1
            candidate = f"{base_name}__{duplicate_counts[base_name]}"
            while candidate in seen:
                duplicate_counts[base_name] += 1
                candidate = f"{base_name}__{duplicate_counts[base_name]}"
        else:
            duplicate_counts.setdefault(base_name, 1)
        seen.add(candidate)
        deduplicated.append(candidate)
    return deduplicated


def _prepare_mutation_operations(operations: Sequence[MutationOperation]) -> list[_PreparedMutationOperation]:
    if not 1 <= len(operations) <= 500:
        raise CrossWorkspaceSqliteError("invalid_parameters")
    prepared: list[_PreparedMutationOperation] = []
    for operation in operations:
        raw_kind = str(operation.kind or "").strip().lower()
        if raw_kind not in _MUTATION_KINDS:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        kind = cast(MutationKind, raw_kind)
        table = _validate_identifier(operation.table)
        values = _validate_value_mapping(operation.values or {})
        where = _validate_value_mapping(operation.where or {})
        conflict_columns = _validate_identifier_list(operation.conflict_columns or [])
        if kind == "insert" and not values:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        if kind == "upsert":
            if not values or not conflict_columns:
                raise CrossWorkspaceSqliteError("invalid_parameters")
        if kind in {"update", "delete"} and not where:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        if kind == "update" and not values:
            raise CrossWorkspaceSqliteError("invalid_parameters")
        prepared.append(
            _PreparedMutationOperation(
                kind=kind,
                table=table,
                values=values,
                where=where,
                conflict_columns=conflict_columns,
            )
        )
    return prepared


def _validate_value_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    if not values:
        return {}
    normalized: dict[str, Any] = {}
    for key in sorted(values.keys()):
        normalized[_validate_identifier(key)] = _validate_bind_value(values[key])
    return normalized


def _validate_identifier(name: str) -> str:
    try:
        return sqlite_inspector.validate_identifier(name)
    except HTTPException as exc:
        raise CrossWorkspaceSqliteError("invalid_sql") from exc


def _validate_identifier_list(values: Sequence[str]) -> list[str]:
    normalized = [_validate_identifier(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise CrossWorkspaceSqliteError("invalid_sql")
    return sorted(normalized)


def _validate_mutation_payload_size(operations: Sequence[_PreparedMutationOperation]) -> None:
    bytes_total = 2
    for operation in operations:
        payload_shape = {
            "kind": operation.kind,
            "table": operation.table,
            "values": {key: _payload_value_shape(value) for key, value in operation.values.items()},
            "where": {key: _payload_value_shape(value) for key, value in operation.where.items()},
            "conflict_columns": list(operation.conflict_columns),
        }
        if bytes_total > 2:
            bytes_total += 1
        bytes_total += len(json.dumps(payload_shape, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        bytes_total += _raw_bytes_size(operation.values.values())
        bytes_total += _raw_bytes_size(operation.where.values())
        if bytes_total > _MUTATION_MAX_PAYLOAD_BYTES:
            raise CrossWorkspaceSqliteError("payload_too_large")


def _payload_value_shape(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes__": len(value)}
    return value


def _raw_bytes_size(values: Sequence[Any] | Any) -> int:
    total = 0
    for value in values:
        if isinstance(value, bytes):
            total += len(value)
    return total


def _execute_mutation_transaction(
    db_path: Path,
    operations: Sequence[_PreparedMutationOperation],
    policy: CrossWorkspaceSqlitePolicy,
) -> list[MutationOperationResult]:
    if not db_path.exists():
        raise CrossWorkspaceSqliteError("database_not_found")
    attempts = len(_RETRY_DELAYS_SECONDS) + 1
    for attempt in range(attempts):
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                str(db_path),
                timeout=max(policy.busy_timeout_ms, 1) / 1000,
                isolation_level=None,
                detect_types=0,
            )
            connection.execute(f"PRAGMA busy_timeout = {int(policy.busy_timeout_ms)}")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.set_progress_handler(_deadline_progress_handler(time.monotonic() + float(policy.query_timeout_seconds)), 1000)
            connection.execute("BEGIN IMMEDIATE")
            results = [_execute_prepared_mutation(connection, operation) for operation in operations]
            connection.execute("COMMIT")
            return results
        except CrossWorkspaceSqliteError:
            if connection is not None:
                _safe_rollback(connection)
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                _safe_rollback(connection)
            mapped = _map_sqlite_error(exc, default_code="mutation_failed")
            if mapped.code == "sqlite_busy" and attempt < attempts - 1:
                if connection is not None:
                    connection.close()
                time.sleep(_RETRY_DELAYS_SECONDS[attempt])
                continue
            raise mapped from exc
        finally:
            if connection is not None:
                connection.close()
    raise CrossWorkspaceSqliteError("sqlite_busy")


def _safe_rollback(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        return


def _execute_prepared_mutation(connection: sqlite3.Connection, operation: _PreparedMutationOperation) -> MutationOperationResult:
    if operation.kind == "insert":
        sql, binds = _render_insert(operation)
    elif operation.kind == "upsert":
        sql, binds = _render_upsert(operation)
    elif operation.kind == "update":
        sql, binds = _render_update(operation)
    elif operation.kind == "delete":
        sql, binds = _render_delete(operation)
    else:
        raise CrossWorkspaceSqliteError("invalid_parameters")
    try:
        cursor = connection.execute(sql, binds)
    except sqlite3.Error as exc:
        raise _map_sqlite_error(exc, default_code="mutation_failed") from exc
    lastrowid: int | None = None
    if operation.kind in {"insert", "upsert"} and cursor.lastrowid is not None and cursor.rowcount > 0:
        lastrowid = int(cursor.lastrowid)
    return MutationOperationResult(kind=operation.kind, rowcount=int(cursor.rowcount), lastrowid=lastrowid)


def _render_insert(operation: _PreparedMutationOperation) -> tuple[str, list[Any]]:
    columns = list(operation.values.keys())
    placeholders = ", ".join("?" for _ in columns)
    sql = (
        f"INSERT INTO {sqlite_inspector.quote_identifier(operation.table)} "
        f"({', '.join(sqlite_inspector.quote_identifier(column) for column in columns)}) VALUES ({placeholders})"
    )
    return sql, [operation.values[column] for column in columns]


def _render_upsert(operation: _PreparedMutationOperation) -> tuple[str, list[Any]]:
    insert_sql, binds = _render_insert(operation)
    conflict_sql = ", ".join(sqlite_inspector.quote_identifier(column) for column in operation.conflict_columns)
    update_columns = [column for column in operation.values.keys() if column not in set(operation.conflict_columns)]
    if not update_columns:
        return f"{insert_sql} ON CONFLICT ({conflict_sql}) DO NOTHING", binds
    assignments = ", ".join(f"{sqlite_inspector.quote_identifier(column)} = excluded.{sqlite_inspector.quote_identifier(column)}" for column in update_columns)
    return f"{insert_sql} ON CONFLICT ({conflict_sql}) DO UPDATE SET {assignments}", binds


def _render_update(operation: _PreparedMutationOperation) -> tuple[str, list[Any]]:
    assignments = ", ".join(f"{sqlite_inspector.quote_identifier(column)} = ?" for column in operation.values.keys())
    predicate_sql, predicate_binds = _render_predicates(operation.where)
    sql = f"UPDATE {sqlite_inspector.quote_identifier(operation.table)} SET {assignments} WHERE {predicate_sql}"
    return sql, [operation.values[column] for column in operation.values.keys()] + predicate_binds


def _render_delete(operation: _PreparedMutationOperation) -> tuple[str, list[Any]]:
    predicate_sql, predicate_binds = _render_predicates(operation.where)
    sql = f"DELETE FROM {sqlite_inspector.quote_identifier(operation.table)} WHERE {predicate_sql}"
    return sql, predicate_binds


def _render_predicates(where: Mapping[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    binds: list[Any] = []
    for column, value in where.items():
        quoted = sqlite_inspector.quote_identifier(column)
        if value is None:
            clauses.append(f"{quoted} IS NULL")
        else:
            clauses.append(f"{quoted} = ?")
            binds.append(value)
    return " AND ".join(clauses), binds


async def _call_audit_intent(callback: Callable[[AuditIntent], Any], intent: AuditIntent) -> bool:
    try:
        result = callback(intent)
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)
    except Exception:
        return False


async def _call_audit_outcome(callback: Callable[[AuditOutcome], Any] | None, outcome: AuditOutcome) -> None:
    if callback is None:
        return
    try:
        result = callback(outcome)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        LOGGER.warning("Cross-workspace SQLite audit outcome callback failed", exc_info=True)


def _outcome_status_for_error(code: str) -> str:
    if code == "sqlite_busy":
        return "busy"
    if code == "query_timeout":
        return "timeout"
    return "rolled_back"


def _map_sqlite_error(exc: sqlite3.Error, *, default_code: str) -> CrossWorkspaceSqliteError:
    error_code = getattr(exc, "sqlite_errorcode", None)
    message = str(exc).lower()
    if error_code == _SQLITE_INTERRUPT or "interrupted" in message:
        return CrossWorkspaceSqliteError("query_timeout")
    if error_code == _SQLITE_CANTOPEN or "unable to open database file" in message or "readonly" in message:
        return CrossWorkspaceSqliteError("database_not_found")
    if error_code in _SQLITE_BUSY_CODES or "database is locked" in message or "database is busy" in message:
        return CrossWorkspaceSqliteError("sqlite_busy")
    if "not authorized" in message or "prohibited" in message:
        return CrossWorkspaceSqliteError("sql_not_allowed")
    return CrossWorkspaceSqliteError(default_code)


def _execute_wal_checkpoint(db_path: Path, policy: CrossWorkspaceSqlitePolicy) -> None:
    if not db_path.exists():
        raise CrossWorkspaceSqliteError("database_not_found")
    try:
        connection = sqlite3.connect(
            str(db_path),
            timeout=max(policy.busy_timeout_ms, 1) / 1000,
            isolation_level=None,
            detect_types=0,
        )
    except sqlite3.Error as exc:
        raise _map_sqlite_error(exc, default_code="database_not_found") from exc
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(policy.busy_timeout_ms)}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.Error as exc:
        raise _map_sqlite_error(exc, default_code="mutation_failed") from exc
    finally:
        connection.close()
