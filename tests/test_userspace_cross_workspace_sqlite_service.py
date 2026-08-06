from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import cross_workspace_sqlite as cross_workspace_sqlite_mod
from ragtime.userspace.models import (
    RuntimeBridgeSqliteMutationOperation,
    RuntimeBridgeSqliteMutationRequest,
    RuntimeBridgeSqliteMutationResponse,
    RuntimeBridgeSqliteQueryRequest,
    RuntimeBridgeSqliteQueryResponse,
)
from ragtime.userspace.service import UserSpaceService

_NOW = datetime(2026, 8, 5, tzinfo=timezone.utc)


@dataclass
class _WorkspaceRow:
    owner_user_id: str
    name: str


class _CapturedAuditEvent(TypedDict):
    workspace_id: str
    user_id: str | None
    event_type: str
    payload: dict[str, object]
    session_id: str | None
    created_at: datetime | None


class _FakeWorkspaceTable:
    def __init__(self, rows: dict[str, _WorkspaceRow]) -> None:
        self.rows = rows
        self.find_unique_calls = 0

    async def find_unique(self, *, where: dict[str, str], include: dict[str, object] | None = None) -> SimpleNamespace | None:
        self.find_unique_calls += 1
        workspace_id = str(where.get("id") or "")
        row = self.rows.get(workspace_id)
        if row is None:
            return None
        return SimpleNamespace(id=workspace_id, ownerUserId=row.owner_user_id, name=row.name)


class _FakeWorkspaceMemberTable:
    def __init__(self, roles: dict[tuple[str, str], str]) -> None:
        self.roles = roles
        self.find_first_calls = 0

    async def find_first(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        self.find_first_calls += 1
        key = (str(where.get("workspaceId") or ""), str(where.get("userId") or ""))
        role = self.roles.get(key)
        if role is None:
            return None
        return SimpleNamespace(workspaceId=key[0], userId=key[1], role=role)


class _FakeGrantTable:
    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = rows or []
        self.find_first_calls = 0

    async def find_first(self, *, where: dict[str, str]) -> SimpleNamespace | None:
        self.find_first_calls += 1
        for row in self.rows:
            if str(getattr(row, "sourceWorkspaceId", "") or "") == str(where.get("sourceWorkspaceId") or "") and str(
                getattr(row, "targetWorkspaceId", "") or ""
            ) == str(where.get("targetWorkspaceId") or ""):
                return row
        return None


class _FakeUserTable:
    async def find_unique(self, **_: object) -> SimpleNamespace | None:  # pragma: no cover - should never be called here
        raise AssertionError("cross-workspace SQLite auth must not query global user role")


class _FakeBroker:
    def __init__(self) -> None:
        self.query_calls: list[dict[str, object]] = []
        self.mutate_calls: list[dict[str, object]] = []
        self.checkpoint_calls: list[Path] = []
        self.query_result = cross_workspace_sqlite_mod.QueryResult(columns=["id"], rows=[{"id": 1}], row_count=1, truncated=False)
        self.mutation_result = cross_workspace_sqlite_mod.MutationResult(
            operations=[cross_workspace_sqlite_mod.MutationOperationResult(kind="insert", rowcount=1, lastrowid=7)],
            fingerprint="fingerprint-1",
        )

    def fingerprint_operations(self, operations: Sequence[cross_workspace_sqlite_mod.MutationOperation]) -> str:
        return cross_workspace_sqlite_mod.CrossWorkspaceSqliteBroker(
            cross_workspace_sqlite_mod.CrossWorkspaceSqlitePolicy()
        ).fingerprint_operations(operations)

    async def query(self, workspace_files_dir: Path, sql: str, *, parameters: object = None, max_rows: int = 200):
        self.query_calls.append(
            {
                "workspace_files_dir": workspace_files_dir,
                "sql": sql,
                "parameters": parameters,
                "max_rows": max_rows,
            }
        )
        return self.query_result

    async def mutate(
        self,
        workspace_files_dir: Path,
        operations: Sequence[cross_workspace_sqlite_mod.MutationOperation],
        *,
        audit_context: cross_workspace_sqlite_mod.AuditIdentityContext,
        audit_intent_callback,
        audit_outcome_callback=None,
    ):
        self.mutate_calls.append(
            {
                "workspace_files_dir": workspace_files_dir,
                "operations": operations,
                "audit_context": audit_context,
            }
        )
        intent = cross_workspace_sqlite_mod.AuditIntent(
            fingerprint=self.mutation_result.fingerprint,
            operation_count=len(operations),
            identity_context=audit_context,
        )
        ok = audit_intent_callback(intent)
        if hasattr(ok, "__await__"):
            ok = await ok
        if not ok:
            raise cross_workspace_sqlite_mod.CrossWorkspaceSqliteError("audit_unavailable")
        if audit_outcome_callback is not None:
            outcome = cross_workspace_sqlite_mod.AuditOutcome(
                fingerprint=self.mutation_result.fingerprint,
                operation_count=len(operations),
                identity_context=audit_context,
                status="committed",
                error_code=None,
            )
            result = audit_outcome_callback(outcome)
            if hasattr(result, "__await__"):
                await result
        return self.mutation_result

    async def checkpoint(self, workspace_files_dir: Path) -> None:
        self.checkpoint_calls.append(workspace_files_dir)


class RuntimeBridgeCrossWorkspaceSqliteServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceService()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.broker = _FakeBroker()
        self.audit_events: list[_CapturedAuditEvent] = []
        self.service._runtime_bridge_sqlite_rate_limits.clear()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _db(
        self,
        *,
        roles: dict[tuple[str, str], str] | None = None,
        grant_rows: list[SimpleNamespace] | None = None,
        workspace_rows: dict[str, _WorkspaceRow] | None = None,
        with_grant_model: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            workspace=_FakeWorkspaceTable(
                workspace_rows
                or {
                    "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source"),
                    "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target"),
                }
            ),
            workspacemember=_FakeWorkspaceMemberTable(roles or {}),
            workspaceagentgrant=_FakeGrantTable(grant_rows) if with_grant_model else None,
            user=_FakeUserTable(),
        )

    def _grant(self, *, sqlite_access_mode: str = "read", expires_at: datetime | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            id="grant-1",
            sourceWorkspaceId="source-ws",
            targetWorkspaceId="target-ws",
            accessMode="read",
            sqliteAccessMode=sqlite_access_mode,
            expiresAt=expires_at,
        )

    def _patch_common(self, fake_db: SimpleNamespace) -> tuple[mock._patch, ...]:
        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        async def _record_runtime_audit_event(
            workspace_id: str,
            user_id: str | None,
            event_type: str,
            payload: dict[str, object],
            *,
            session_id: str | None = None,
            created_at: datetime | None = None,
        ) -> bool:
            self.audit_events.append(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "event_type": event_type,
                    "payload": payload,
                    "session_id": session_id,
                    "created_at": created_at,
                }
            )
            return True

        return (
            mock.patch("ragtime.userspace.service.get_db", new=_fake_get_db),
            mock.patch.object(self.service, "_record_runtime_audit_event", new=_record_runtime_audit_event),
            mock.patch.object(self.service, "_workspace_files_dir", side_effect=lambda workspace_id: self.workspace_root / workspace_id / "files"),
            mock.patch.object(
                self.service,
                "_runtime_bridge_cross_workspace_sqlite_broker",
                new=mock.Mock(return_value=self.broker),
            ),
        )

    @staticmethod
    def _insert_operation(**values: object) -> RuntimeBridgeSqliteMutationOperation:
        return RuntimeBridgeSqliteMutationOperation(kind="insert", table="items", values=values)

    async def test_query_and_mutation_grant_matrix(self) -> None:
        request_query = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1")
        request_mutate = RuntimeBridgeSqliteMutationRequest(
            target_workspace_id="target-ws",
            operations=[self._insert_operation(id=1)],
        )

        cases: list[tuple[str, SimpleNamespace | None, bool, bool]] = [
            ("missing", None, False, False),
            ("expired", self._grant(sqlite_access_mode="read_write", expires_at=_NOW - timedelta(seconds=1)), False, False),
            ("none", self._grant(sqlite_access_mode="none"), False, False),
            ("read", self._grant(sqlite_access_mode="read"), True, False),
            ("read_write", self._grant(sqlite_access_mode="read_write"), True, True),
        ]
        for label, grant_row, expect_query_ok, expect_mutate_ok in cases:
            with self.subTest(case=label):
                fake_db = self._db(
                    roles={
                        ("source-ws", "leased-user"): "viewer",
                        ("target-ws", "leased-user"): "editor",
                    },
                    grant_rows=([grant_row] if grant_row is not None else []),
                )
                with ExitStack() as stack:
                    for patcher in self._patch_common(fake_db):
                        stack.enter_context(patcher)
                    stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
                    if expect_query_ok:
                        query_response = await self.service.query_runtime_bridge_sqlite(
                            workspace_id="source-ws",
                            session_id="sess-1",
                            leased_by_user_id="leased-user",
                            request=request_query,
                        )
                        self.assertEqual(query_response.target_workspace_id, "target-ws")
                    else:
                        with self.assertRaises(HTTPException) as raised:
                            await self.service.query_runtime_bridge_sqlite(
                                workspace_id="source-ws",
                                session_id="sess-1",
                                leased_by_user_id="leased-user",
                                request=request_query,
                            )
                        self.assertEqual(raised.exception.status_code, 403)

                    if expect_mutate_ok:
                        mutation_response = await self.service.mutate_runtime_bridge_sqlite(
                            workspace_id="source-ws",
                            session_id="sess-1",
                            leased_by_user_id="leased-user",
                            request=request_mutate,
                        )
                        self.assertEqual(mutation_response.fingerprint, "fingerprint-1")
                    else:
                        with self.assertRaises(HTTPException) as raised:
                            await self.service.mutate_runtime_bridge_sqlite(
                                workspace_id="source-ws",
                                session_id="sess-1",
                                leased_by_user_id="leased-user",
                                request=request_mutate,
                            )
                        self.assertEqual(raised.exception.status_code, 403)

    async def test_revocation_and_downgrade_apply_on_next_call(self) -> None:
        grant_row = self._grant(sqlite_access_mode="read_write")
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[grant_row],
        )
        request = RuntimeBridgeSqliteMutationRequest(
            target_workspace_id="target-ws",
            operations=[self._insert_operation(id=1)],
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            first = await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=request,
            )
            self.assertEqual(first.fingerprint, "fingerprint-1")
            grant_row.sqliteAccessMode = "read"
            with self.assertRaises(HTTPException) as downgraded:
                await self.service.mutate_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=request,
                )
            self.assertEqual(downgraded.exception.status_code, 403)
            fake_db.workspaceagentgrant.rows.clear()
            with self.assertRaises(HTTPException) as revoked:
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1"),
                )
            self.assertEqual(revoked.exception.status_code, 403)

    async def test_removed_source_or_target_membership_is_denied(self) -> None:
        grant_row = self._grant(sqlite_access_mode="read_write")
        request_query = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1")
        request_mutate = RuntimeBridgeSqliteMutationRequest(
            target_workspace_id="target-ws",
            operations=[self._insert_operation(id=1)],
        )
        cases: list[
            tuple[
                dict[tuple[str, str], str],
                RuntimeBridgeSqliteQueryRequest | RuntimeBridgeSqliteMutationRequest,
            ]
        ] = [
            ({("target-ws", "leased-user"): "editor"}, request_query),
            ({("source-ws", "leased-user"): "viewer"}, request_mutate),
        ]
        for roles, request in cases:
            with self.subTest(roles=roles):
                fake_db = self._db(roles=roles, grant_rows=[grant_row])
                with ExitStack() as stack:
                    for patcher in self._patch_common(fake_db):
                        stack.enter_context(patcher)
                    stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
                    with self.assertRaises(HTTPException) as raised:
                        if isinstance(request, RuntimeBridgeSqliteQueryRequest):
                            await self.service.query_runtime_bridge_sqlite(
                                workspace_id="source-ws",
                                session_id="sess-1",
                                leased_by_user_id="leased-user",
                                request=request,
                            )
                        else:
                            await self.service.mutate_runtime_bridge_sqlite(
                                workspace_id="source-ws",
                                session_id="sess-1",
                                leased_by_user_id="leased-user",
                                request=request,
                            )
                    self.assertEqual(raised.exception.status_code, 403)

    async def test_global_admin_without_memberships_is_denied(self) -> None:
        fake_db = self._db(roles={}, grant_rows=[self._grant(sqlite_access_mode="read_write")])
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            with self.assertRaises(HTTPException) as raised:
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="admin-user",
                    request=RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1"),
                )
        self.assertEqual(raised.exception.status_code, 403)

    async def test_missing_grant_model_fails_closed_with_fixed_403(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
            },
            with_grant_model=False,
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            with self.assertRaises(HTTPException) as raised:
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1"),
                )
        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(str(raised.exception.detail), "Cross-workspace SQLite access denied")

    async def test_target_viewer_can_query_but_cannot_mutate_and_editor_can_mutate(self) -> None:
        query_request = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1")
        mutate_request = RuntimeBridgeSqliteMutationRequest(
            target_workspace_id="target-ws",
            operations=[self._insert_operation(id=1)],
        )
        viewer_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(viewer_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            query_response = await self.service.query_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=query_request,
            )
            self.assertEqual(query_response.row_count, 1)
            with self.assertRaises(HTTPException) as raised:
                await self.service.mutate_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=mutate_request,
                )
            self.assertEqual(raised.exception.status_code, 403)

        editor_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(editor_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            mutation_response = await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=mutate_request,
            )
            self.assertEqual(mutation_response.operations[0].rowcount, 1)

    async def test_session_target_limiter_returns_429_and_separate_keys_do_not_collide(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
                ("target-2", "leased-user"): "viewer",
            },
            grant_rows=[
                self._grant(sqlite_access_mode="read"),
                SimpleNamespace(
                    id="grant-2",
                    sourceWorkspaceId="source-ws",
                    targetWorkspaceId="target-2",
                    accessMode="read",
                    sqliteAccessMode="read",
                    expiresAt=None,
                ),
            ],
            workspace_rows={
                "source-ws": _WorkspaceRow(owner_user_id="owner-source", name="Source"),
                "target-ws": _WorkspaceRow(owner_user_id="owner-target", name="Target"),
                "target-2": _WorkspaceRow(owner_user_id="owner-target-2", name="Target 2"),
            },
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            request = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1")
            for _ in range(120):
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=request,
                )
            with self.assertRaises(HTTPException) as raised:
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=request,
                )
            self.assertEqual(raised.exception.status_code, 429)

            other_key_response = await self.service.query_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-2", sql="SELECT 1"),
            )
            self.assertEqual(other_key_response.target_workspace_id, "target-2")

            other_session_response = await self.service.query_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-2",
                leased_by_user_id="leased-user",
                request=request,
            )
            self.assertEqual(other_session_response.target_workspace_id, "target-ws")

    async def test_rate_limited_denials_audit_best_effort_without_values_and_before_auth_queries(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
            },
            grant_rows=[self._grant(sqlite_access_mode="read")],
        )
        request = RuntimeBridgeSqliteQueryRequest(
            target_workspace_id="target-ws",
            sql="SELECT secret FROM tokens WHERE code = ?",
            parameters=["top-secret"],
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            for _ in range(120):
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=request,
                )
            workspace_calls_before = fake_db.workspace.find_unique_calls
            member_calls_before = fake_db.workspacemember.find_first_calls
            grant_calls_before = fake_db.workspaceagentgrant.find_first_calls
            with self.assertRaises(HTTPException) as raised:
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=request,
                )
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(fake_db.workspace.find_unique_calls, workspace_calls_before)
        self.assertEqual(fake_db.workspacemember.find_first_calls, member_calls_before)
        self.assertEqual(fake_db.workspaceagentgrant.find_first_calls, grant_calls_before)
        denied_event = self.audit_events[-1]
        payload: dict[str, object] = denied_event["payload"]
        self.assertEqual(payload["phase"], "denied")
        self.assertEqual(payload["error_code"], "rate_limited")
        serialized = repr(payload)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("SELECT secret", serialized)

    async def test_query_audit_uses_digest_without_sql_or_parameters(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
            },
            grant_rows=[self._grant(sqlite_access_mode="read")],
        )
        sql = "SELECT secret FROM tokens WHERE code = ?"
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            await self.service.query_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteQueryRequest(
                    target_workspace_id="target-ws",
                    sql=sql,
                    parameters=["top-secret"],
                ),
            )
        payload: dict[str, object] = self.audit_events[-1]["payload"]
        self.assertEqual(payload["sql_digest"], hashlib.sha256(sql.encode("utf-8")).hexdigest())
        serialized = repr(payload)
        self.assertNotIn(sql, serialized)
        self.assertNotIn("top-secret", serialized)
        self.assertNotIn("parameters", payload)

    async def test_unexpected_query_broker_exception_audits_generic_safe_code_without_raw_text(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "viewer",
            },
            grant_rows=[self._grant(sqlite_access_mode="read")],
        )

        async def explode(*_args, **_kwargs):
            raise RuntimeError("boom top-secret")

        self.broker.query = explode  # type: ignore[method-assign]
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            with self.assertRaises(RuntimeError):
                await self.service.query_runtime_bridge_sqlite(
                    workspace_id="source-ws",
                    session_id="sess-1",
                    leased_by_user_id="leased-user",
                    request=RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="SELECT 1"),
                )
        payload: dict[str, object] = self.audit_events[-1]["payload"]
        self.assertEqual(payload["error_code"], "query_failed")
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("top-secret", repr(payload))

    async def test_mutation_audit_contains_no_values(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteMutationRequest(
                    target_workspace_id="target-ws",
                    operations=[self._insert_operation(name="Ada", secret="42")],
                ),
            )
        serialized = repr([event["payload"] for event in self.audit_events])
        self.assertNotIn("Ada", serialized)
        self.assertNotIn("42", serialized)

    async def test_checkpoint_occurs_after_commit_and_before_snapshot(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )
        order: list[str] = []

        async def mutate(*args, **kwargs):
            order.append("mutate")
            return await _FakeBroker.mutate(self.broker, *args, **kwargs)

        async def checkpoint(_workspace_files_dir: Path) -> None:
            order.append("checkpoint")

        async def create_snapshot(*_args, **_kwargs) -> SimpleNamespace:
            order.append("snapshot")
            return SimpleNamespace(id="snapshot-1")

        self.broker.mutate = mutate  # type: ignore[method-assign]
        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            stack.enter_context(mock.patch.object(self.broker, "checkpoint", new=checkpoint))
            stack.enter_context(mock.patch.object(self.service, "create_snapshot", new=create_snapshot))
            await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteMutationRequest(
                    target_workspace_id="target-ws",
                    operations=[self._insert_operation(id=1)],
                ),
            )
        self.assertEqual(order, ["mutate", "checkpoint", "snapshot"])

    async def test_checkpoint_failure_skips_snapshot_and_audits_checkpoint_failed(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )
        snapshot_calls = 0

        async def checkpoint(_workspace_files_dir: Path) -> None:
            raise cross_workspace_sqlite_mod.CrossWorkspaceSqliteError("sqlite_busy")

        async def create_snapshot(*_args, **_kwargs) -> SimpleNamespace:
            nonlocal snapshot_calls
            snapshot_calls += 1
            return SimpleNamespace(id="snapshot-1")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            stack.enter_context(mock.patch.object(self.broker, "checkpoint", new=checkpoint))
            stack.enter_context(mock.patch.object(self.service, "create_snapshot", new=create_snapshot))
            mutation_response = await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteMutationRequest(
                    target_workspace_id="target-ws",
                    operations=[self._insert_operation(id=1)],
                ),
            )
        self.assertEqual(mutation_response.fingerprint, "fingerprint-1")
        self.assertEqual(snapshot_calls, 0)
        self.assertTrue(any(event["payload"].get("error_code") == "checkpoint_failed" for event in self.audit_events))

    async def test_snapshot_failure_audits_snapshot_failed_and_returns_committed_result(self) -> None:
        fake_db = self._db(
            roles={
                ("source-ws", "leased-user"): "viewer",
                ("target-ws", "leased-user"): "editor",
            },
            grant_rows=[self._grant(sqlite_access_mode="read_write")],
        )

        async def create_snapshot(*_args, **_kwargs) -> SimpleNamespace:
            raise RuntimeError("snapshot failed")

        with ExitStack() as stack:
            for patcher in self._patch_common(fake_db):
                stack.enter_context(patcher)
            stack.enter_context(mock.patch("ragtime.userspace.service.utc_now", return_value=_NOW))
            stack.enter_context(mock.patch.object(self.service, "create_snapshot", new=create_snapshot))
            mutation_response = await self.service.mutate_runtime_bridge_sqlite(
                workspace_id="source-ws",
                session_id="sess-1",
                leased_by_user_id="leased-user",
                request=RuntimeBridgeSqliteMutationRequest(
                    target_workspace_id="target-ws",
                    operations=[self._insert_operation(id=1)],
                ),
            )
        self.assertEqual(mutation_response.fingerprint, "fingerprint-1")
        self.assertTrue(any(event["payload"].get("error_code") == "snapshot_failed" for event in self.audit_events))


if __name__ == "__main__":
    unittest.main()
