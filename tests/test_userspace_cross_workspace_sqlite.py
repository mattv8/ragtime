from __future__ import annotations

import asyncio
import gc
import importlib
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ragtime.userspace.cross_workspace_sqlite import AuditOutcome


def _mod():
    return importlib.import_module("ragtime.userspace.cross_workspace_sqlite")


def _policy(**overrides):
    mod = _mod()
    values = {}
    values.update(overrides)
    return mod.CrossWorkspaceSqlitePolicy(**values)


def _broker(**policy_overrides):
    mod = _mod()
    return mod.CrossWorkspaceSqliteBroker(_policy(**policy_overrides))


def _files_dir(tmp_path: Path) -> Path:
    files_dir = tmp_path / "files"
    files_dir.mkdir()
    return files_dir


def _db_path(files_dir: Path) -> Path:
    db_path = files_dir / ".ragtime" / "db" / "app.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _init_db(files_dir: Path) -> Path:
    db_path = _db_path(files_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, payload BLOB, qty INTEGER DEFAULT 0)")
        conn.commit()
    return db_path


async def _allow_audit(_intent) -> bool:
    return True


def _run(awaitable):
    return asyncio.run(awaitable)


def _assert_broker_error(exc_info: pytest.ExceptionInfo[Exception], code: str) -> None:
    error = exc_info.value
    assert getattr(error, "code") == code
    assert isinstance(str(error), str)


def test_query_allows_reads_and_supports_positional_and_named_parameters(tmp_path: Path) -> None:
    files_dir = _files_dir(tmp_path)
    db_path = _init_db(files_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO items (id, name, payload, qty) VALUES (?, ?, ?, ?)", (1, "Ada", b"abc", 3))
        conn.execute("INSERT INTO items (id, name, qty) VALUES (?, ?, ?)", (2, "Grace", 7))
        conn.commit()

    positional = _run(_broker().query(files_dir, "SELECT name FROM items WHERE id = ?", parameters=[1]))
    assert positional.rows == [{"name": "Ada"}]

    named = _run(
        _broker().query(
            files_dir,
            "SELECT name, payload FROM items WHERE qty = :qty",
            parameters={"qty": 3},
        )
    )
    assert named.rows == [{"name": "Ada", "payload": {"__blob__": True, "size": 3}}]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM items",
        "CREATE TABLE nope (id INTEGER)",
        "PRAGMA user_version",
        "ATTACH DATABASE ':memory:' AS extra",
        "SELECT load_extension('test')",
    ],
)
def test_query_rejects_mutation_ddl_pragma_attach_and_extension_attempts(tmp_path: Path, sql: str) -> None:
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, sql))
    _assert_broker_error(exc_info, "sql_not_allowed")


@pytest.mark.parametrize(
    "sql",
    [
        "WITH cte AS (SELECT 1) DELETE FROM items WHERE id = 1",
        "WITH cte AS (SELECT 1) UPDATE items SET name = 'x' WHERE id = 1",
        "EXPLAIN DELETE FROM items WHERE id = 1",
    ],
)
def test_query_authorizer_blocks_write_ctes_and_explain_writes(tmp_path: Path, sql: str) -> None:
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, sql))
    _assert_broker_error(exc_info, "sql_not_allowed")


def test_query_enforces_sql_row_parameter_and_response_limits(tmp_path: Path) -> None:
    files_dir = _files_dir(tmp_path)
    db_path = _init_db(files_dir)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO items (id, name) VALUES (?, ?)",
            [(idx, "x" * 4096) for idx in range(1, 400)],
        )
        conn.commit()

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT 1", max_rows=0))
    _assert_broker_error(exc_info, "row_limit_invalid")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "S" * (32 * 1024 + 1)))
    _assert_broker_error(exc_info, "sql_too_large")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT ?", parameters=object()))
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT ?", parameters=list(range(501))))
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT ?", parameters={1: "bad-key"}))
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT ?", parameters=[2**63]))
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, 123))
    _assert_broker_error(exc_info, "invalid_sql")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT 1", max_rows="10"))
    _assert_broker_error(exc_info, "row_limit_invalid")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT 1", max_rows=True))
    _assert_broker_error(exc_info, "row_limit_invalid")

    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT name FROM items", max_rows=399))
    _assert_broker_error(exc_info, "response_too_large")


def test_query_missing_database_maps_to_database_not_found(tmp_path: Path) -> None:
    files_dir = _files_dir(tmp_path)
    with pytest.raises(Exception) as exc_info:
        _run(_broker().query(files_dir, "SELECT 1"))
    _assert_broker_error(exc_info, "database_not_found")


def test_query_maps_timeouts_and_supports_readonly_wal_databases(tmp_path: Path) -> None:
    files_dir = _files_dir(tmp_path)
    db_path = _init_db(files_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO items (id, name) VALUES (?, ?)", (1, "wal"))
        conn.commit()

    wal_result = _run(_broker().query(files_dir, "SELECT name FROM items"))
    assert wal_result.rows == [{"name": "wal"}]

    timeout_broker = _broker()
    timeout_broker.policy.query_timeout_seconds = 0.001
    with pytest.raises(Exception) as exc_info:
        _run(
            timeout_broker.query(
                files_dir,
                "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 10000000) SELECT max(x) FROM cnt",
            )
        )
    _assert_broker_error(exc_info, "query_timeout")


def test_query_renames_duplicate_output_columns_without_data_loss(tmp_path: Path) -> None:
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    result = _run(_broker().query(files_dir, "SELECT 1 AS x, 2 AS x, 3 AS x"))

    assert result.columns == ["x", "x__2", "x__3"]
    assert result.rows == [{"x": 1, "x__2": 2, "x__3": 3}]


def test_mutate_runs_atomic_insert_upsert_update_and_delete_transaction(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    result = _run(
        _broker().mutate(
            files_dir,
            [
                mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada", "qty": 1}),
                mod.MutationOperation(kind="upsert", table="items", values={"id": 1, "name": "Ada Lovelace", "qty": 2}, conflict_columns=["id"]),
                mod.MutationOperation(kind="insert", table="items", values={"id": 2, "name": "Grace", "qty": 4}),
                mod.MutationOperation(kind="update", table="items", values={"qty": 5}, where={"id": 2}),
                mod.MutationOperation(kind="delete", table="items", where={"id": 1}),
            ],
            audit_context=mod.AuditIdentityContext(actor_id="user-1"),
            audit_intent_callback=_allow_audit,
        )
    )

    assert [entry.rowcount for entry in result.operations] == [1, 1, 1, 1, 1]
    assert result.operations[0].lastrowid is not None
    assert result.operations[1].lastrowid is not None

    rows = _run(_broker().query(files_dir, "SELECT id, name, qty FROM items ORDER BY id"))
    assert rows.rows == [{"id": 2, "name": "Grace", "qty": 5}]


def test_mutate_rejects_empty_predicates_unsafe_identifiers_operation_count_and_payload_size(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [mod.MutationOperation(kind="update", table="items", values={"name": "Ada"}, where={})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            )
        )
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [mod.MutationOperation(kind="delete", table="items", where={})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            )
        )
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items;drop", values={"id": 1})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            )
        )
    _assert_broker_error(exc_info, "invalid_sql")

    too_many = [mod.MutationOperation(kind="insert", table="items", values={"id": idx}) for idx in range(1, 502)]
    with pytest.raises(Exception) as exc_info:
        _run(_broker().mutate(files_dir, too_many, audit_context=mod.AuditIdentityContext(actor_id="user-1"), audit_intent_callback=_allow_audit))
    _assert_broker_error(exc_info, "invalid_parameters")

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 2**63, "name": "Ada"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            )
        )
    _assert_broker_error(exc_info, "invalid_parameters")

    too_large = [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "x" * (1024 * 1024)})]
    with pytest.raises(Exception) as exc_info:
        _run(_broker().mutate(files_dir, too_large, audit_context=mod.AuditIdentityContext(actor_id="user-1"), audit_intent_callback=_allow_audit))
    _assert_broker_error(exc_info, "payload_too_large")


def test_mutate_rolls_back_on_failure_and_reports_outcomes_without_values(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    outcomes: list[AuditOutcome] = []

    async def outcome_callback(outcome: AuditOutcome) -> None:
        outcomes.append(outcome)

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [
                    mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"}),
                    mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Duplicate"}),
                ],
                audit_context=mod.AuditIdentityContext(actor_id="user-1", request_id="req-1"),
                audit_intent_callback=_allow_audit,
                audit_outcome_callback=outcome_callback,
            )
        )
    _assert_broker_error(exc_info, "mutation_failed")

    rows = _run(_broker().query(files_dir, "SELECT id, name FROM items"))
    assert rows.rows == []
    assert outcomes
    serialized = repr(outcomes[0])
    assert "Ada" not in serialized
    assert "Duplicate" not in serialized


def test_mutate_committed_success_reports_outcome_and_swallowed_callback_failures_do_not_change_result(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    outcomes: list[AuditOutcome] = []

    async def outcome_callback(outcome: AuditOutcome) -> None:
        outcomes.append(outcome)

    result = _run(
        _broker().mutate(
            files_dir,
            [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
            audit_context=mod.AuditIdentityContext(actor_id="user-1"),
            audit_intent_callback=_allow_audit,
            audit_outcome_callback=outcome_callback,
        )
    )

    assert result.operations[0].rowcount == 1
    assert len(outcomes) == 1
    assert outcomes[0].status == "committed"
    assert outcomes[0].error_code is None

    async def failing_outcome_callback(_outcome: AuditOutcome) -> None:
        raise RuntimeError("ignored outcome failure")

    second_result = _run(
        _broker().mutate(
            files_dir,
            [mod.MutationOperation(kind="insert", table="items", values={"id": 2, "name": "Grace"})],
            audit_context=mod.AuditIdentityContext(actor_id="user-1"),
            audit_intent_callback=_allow_audit,
            audit_outcome_callback=failing_outcome_callback,
        )
    )

    assert second_result.operations[0].rowcount == 1
    rows = _run(_broker().query(files_dir, "SELECT id, name FROM items ORDER BY id"))
    assert rows.rows == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]


def test_upsert_do_nothing_reports_zero_rows_when_all_supplied_columns_are_conflict_columns(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)

    _run(
        _broker().mutate(
            files_dir,
            [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
            audit_context=mod.AuditIdentityContext(actor_id="user-1"),
            audit_intent_callback=_allow_audit,
        )
    )

    result = _run(
        _broker().mutate(
            files_dir,
            [mod.MutationOperation(kind="upsert", table="items", values={"id": 1}, conflict_columns=["id"])],
            audit_context=mod.AuditIdentityContext(actor_id="user-1"),
            audit_intent_callback=_allow_audit,
        )
    )

    assert result.operations[0].rowcount == 0
    rows = _run(_broker().query(files_dir, "SELECT id, name FROM items"))
    assert rows.rows == [{"id": 1, "name": "Ada"}]


def test_mutate_maps_busy_errors_after_retries(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    db_path = _init_db(files_dir)

    blocker = sqlite3.connect(db_path, isolation_level=None, timeout=0.1)
    blocker.execute("PRAGMA journal_mode=WAL")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        busy_broker = _broker()
        busy_broker.policy.busy_timeout_ms = 1
        with pytest.raises(Exception) as exc_info:
            _run(
                busy_broker.mutate(
                    files_dir,
                    [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
                    audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                    audit_intent_callback=_allow_audit,
                )
            )
        _assert_broker_error(exc_info, "sqlite_busy")
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_mutate_serializes_writes_to_the_same_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    broker = _broker()
    original = mod._execute_mutation_transaction
    order: list[str] = []

    def slow_execute(*args, **kwargs):
        order.append("start")
        time.sleep(0.15)
        try:
            return original(*args, **kwargs)
        finally:
            order.append("end")

    monkeypatch.setattr(mod, "_execute_mutation_transaction", slow_execute)

    async def run_pair() -> None:
        await asyncio.gather(
            broker.mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            ),
            broker.mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 2, "name": "Grace"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            ),
        )

    _run(run_pair())
    assert order == ["start", "end", "start", "end"]


def test_mutation_outcome_callback_runs_outside_database_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    broker = _broker()
    original = mod._execute_mutation_transaction
    order: list[str] = []
    call_index = 0

    def slow_execute(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        current = call_index
        order.append(f"execute-start:{current}")
        time.sleep(0.05)
        try:
            return original(*args, **kwargs)
        finally:
            order.append(f"execute-end:{current}")

    monkeypatch.setattr(mod, "_execute_mutation_transaction", slow_execute)

    async def outcome_callback(outcome: AuditOutcome) -> None:
        order.append(f"outcome-start:{outcome.identity_context.request_id}")
        if outcome.identity_context.request_id == "first":
            await asyncio.sleep(0.2)
        order.append(f"outcome-end:{outcome.identity_context.request_id}")

    async def run_pair() -> None:
        first = asyncio.create_task(
            broker.mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1", request_id="first"),
                audit_intent_callback=_allow_audit,
                audit_outcome_callback=outcome_callback,
            )
        )
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            broker.mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 2, "name": "Grace"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1", request_id="second"),
                audit_intent_callback=_allow_audit,
                audit_outcome_callback=outcome_callback,
            )
        )
        await asyncio.gather(first, second)

    _run(run_pair())
    assert order.index("execute-start:2") < order.index("outcome-end:first")


def test_checkpoint_uses_safe_worker_for_wal_truncate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    db_path = _init_db(files_dir)
    calls: list[Path] = []

    def fake_execute_wal_checkpoint(path: Path, _policy) -> None:
        calls.append(path)

    monkeypatch.setattr(mod, "_execute_wal_checkpoint", fake_execute_wal_checkpoint, raising=False)

    _run(_broker().checkpoint(files_dir))

    assert calls == [db_path]


def test_checkpoint_and_mutation_share_canonical_path_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    broker = _broker()
    order: list[str] = []
    original_mutation = mod._execute_mutation_transaction

    def slow_execute_mutation(*args, **kwargs):
        order.append("mutation-start")
        time.sleep(0.15)
        try:
            return original_mutation(*args, **kwargs)
        finally:
            order.append("mutation-end")

    def slow_checkpoint(_db_path: Path, _policy) -> None:
        order.append("checkpoint-start")
        order.append("checkpoint-end")

    monkeypatch.setattr(mod, "_execute_mutation_transaction", slow_execute_mutation)
    monkeypatch.setattr(mod, "_execute_wal_checkpoint", slow_checkpoint)

    async def run_pair() -> None:
        mutate_task = asyncio.create_task(
            broker.mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
            )
        )
        await asyncio.sleep(0.01)
        checkpoint_task = asyncio.create_task(broker.checkpoint(files_dir))
        await asyncio.gather(mutate_task, checkpoint_task)

    _run(run_pair())
    assert order == ["mutation-start", "mutation-end", "checkpoint-start", "checkpoint-end"]


def test_mutation_lock_registry_uses_weak_lifetime_storage(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    db_path = mod._resolve_managed_db_path(files_dir)

    box = mod._mutation_lock_for(db_path)
    assert db_path.resolve() in mod._MUTATION_LOCKS

    del box
    gc.collect()
    assert db_path.resolve() not in mod._MUTATION_LOCKS


def test_audit_intent_failure_prevents_writes_and_fingerprint_excludes_values(tmp_path: Path) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    broker = _broker()
    seen_intents: list[object] = []

    async def deny_intent(intent) -> bool:
        seen_intents.append(intent)
        return False

    operation_a = mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})
    operation_b = mod.MutationOperation(kind="insert", table="items", values={"id": 2, "name": "Grace"})

    assert broker.fingerprint_operations([operation_a]) == broker.fingerprint_operations([operation_b])

    with pytest.raises(Exception) as exc_info:
        _run(
            broker.mutate(
                files_dir,
                [operation_a],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=deny_intent,
            )
        )
    _assert_broker_error(exc_info, "audit_unavailable")
    assert seen_intents

    rows = _run(_broker().query(files_dir, "SELECT id, name FROM items"))
    assert rows.rows == []


def test_unexpected_transaction_worker_error_maps_to_mutation_failed_and_reports_failure_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _mod()
    files_dir = _files_dir(tmp_path)
    _init_db(files_dir)
    outcomes: list[AuditOutcome] = []

    async def outcome_callback(outcome: AuditOutcome) -> None:
        outcomes.append(outcome)

    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_execute_mutation_transaction", explode)

    with pytest.raises(Exception) as exc_info:
        _run(
            _broker().mutate(
                files_dir,
                [mod.MutationOperation(kind="insert", table="items", values={"id": 1, "name": "Ada"})],
                audit_context=mod.AuditIdentityContext(actor_id="user-1"),
                audit_intent_callback=_allow_audit,
                audit_outcome_callback=outcome_callback,
            )
        )

    _assert_broker_error(exc_info, "mutation_failed")
    assert len(outcomes) == 1
    assert outcomes[0].status == "rolled_back"
    assert outcomes[0].error_code == "mutation_failed"
    assert "Ada" not in repr(outcomes[0])
