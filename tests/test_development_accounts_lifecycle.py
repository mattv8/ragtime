"""Failure-boundary coverage for account workspace creation and discovery."""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase, mock
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_accounts import (
    ACCOUNT_LIST_MAX_BYTES,
    AccountDevelopmentService,
    AccountWorkspaceCreateRequest,
    AccountWorkspaceListRequest,
)
from ragtime.userspace.development_creation_receipts import CompletionUnconfirmed, receipt_path
from ragtime.userspace.service import userspace_service


class _Ledger:
    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = rows or []

    async def find_first(self, *, where: dict[str, Any]) -> SimpleNamespace | None:
        return next(
            (
                row
                for row in self.rows
                if row.userId == where["userId"] and row.idempotencyKey == where["idempotencyKey"]
            ),
            None,
        )

    async def find_unique(self, *, where: dict[str, Any]) -> SimpleNamespace | None:
        return next((row for row in self.rows if row.id == where["id"]), None)

    async def count(self, *, where: dict[str, Any]) -> int:
        return len(self.rows)

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        row = SimpleNamespace(**data)
        self.rows.append(row)
        return row

    async def update_many(self, *, where: dict[str, Any], data: dict[str, Any]) -> int:
        row = await self.find_unique(where={"id": where["id"]})
        states = where["state"]["in"]
        if row is None or row.state not in states:
            return 0
        for key, value in data.items():
            setattr(row, key, value)
        return 1


class _Database:
    def __init__(self, ledger: _Ledger, workspace: Any | None = None) -> None:
        self.workspacedevelopmentcreaterequest = ledger
        self.workspace = workspace or mock.AsyncMock()

    @asynccontextmanager
    async def tx(self):
        yield self

    async def query_raw(self, *_: object) -> None:
        return None


def _principal() -> DevelopmentPrincipal:
    return DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")


def _request(key: str = "request-key-0001", **changes: Any) -> AccountWorkspaceCreateRequest:
    payload: dict[str, Any] = {"name": "A workspace", "idempotency_key": key}
    payload.update(changes)
    return AccountWorkspaceCreateRequest(**payload)


def _claim(state: str = "accepted") -> SimpleNamespace:
    return SimpleNamespace(
        id="request-1",
        userId="user-1",
        idempotencyKey="request-key-0001",
        payloadHash=_request().payload_hash(),
        workspaceId="workspace-1",
        state=state,
        errorCode=None,
    )


class AccountCreateLifecycleTests(IsolatedAsyncioTestCase):
    async def _create(self, ledger: _Ledger, request: AccountWorkspaceCreateRequest, *, core: Any) -> dict[str, Any]:
        database = _Database(ledger)
        with (
            mock.patch("ragtime.userspace.development_accounts.get_db", new=mock.AsyncMock(return_value=database)),
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=core),
        ):
            return await AccountDevelopmentService().create_workspace(_principal(), request)

    async def test_matching_receipt_replay_completes_interrupted_claim_without_redispatch(self) -> None:
        claim = _claim("interrupted")
        ledger = _Ledger([claim])
        database = _Database(ledger)
        service = AccountDevelopmentService()
        workspace = SimpleNamespace(id="workspace-1", name="Recovered")
        with (
            mock.patch("ragtime.userspace.development_accounts.read_completion_receipt", return_value={"version": 1}),
            mock.patch.object(service, "_authorized_workspace", new=mock.AsyncMock(return_value=workspace)),
            mock.patch("ragtime.userspace.development_accounts.userspace_service._workspace_files_dir", return_value=Path("/tmp/workspace/files")),
        ):
            result = await service._replay_claim(database, _principal(), claim)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["replayed"])
        self.assertEqual(claim.state, "completed")

    async def test_row_without_missing_or_malformed_receipt_never_completes_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for malformed in (False, True):
                with self.subTest(malformed=malformed):
                    claim = _claim()
                    workspace_dir = Path(temporary) / ("malformed" if malformed else "missing")
                    workspace_dir.mkdir()
                    if malformed:
                        receipt_path(workspace_dir).write_text("not json")
                    service = AccountDevelopmentService()
                    with (
                        mock.patch.object(service, "_authorized_workspace", new=mock.AsyncMock(return_value=SimpleNamespace(id="workspace-1", name="row only"))),
                        mock.patch("ragtime.userspace.development_accounts.userspace_service._workspace_files_dir", return_value=workspace_dir / "files"),
                    ):
                        result = await service._replay_claim(_Database(_Ledger([claim])), _principal(), claim)
                    self.assertEqual(result["outcome"], "unconfirmed")
                    self.assertEqual(claim.state, "accepted")

    async def test_cancellation_finalizes_interrupted_claim_then_reraises(self) -> None:
        ledger = _Ledger()
        core = mock.AsyncMock(side_effect=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await self._create(ledger, _request(), core=core)
        self.assertEqual(ledger.rows[0].state, "interrupted")
        self.assertEqual(ledger.rows[0].errorCode, "create_interrupted")

    async def test_completion_unconfirmed_finalizes_interrupted_claim_without_claiming_completion(self) -> None:
        ledger = _Ledger()
        result = await self._create(ledger, _request(), core=mock.AsyncMock(side_effect=CompletionUnconfirmed("receipt unavailable")))
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["outcome"], "unconfirmed")
        self.assertEqual(ledger.rows[0].state, "interrupted")
        self.assertEqual(ledger.rows[0].errorCode, "create_interrupted")
        self.assertIn("Retry this exact idempotency key", result["recovery"])
        self.assertIn("will not provision another workspace", result["recovery"])
        self.assertIn("full workspace UUID", result["recovery"])
        self.assertIn("user explicitly asks", result["recovery"])

    async def test_cancellation_during_failure_finalization_reraises(self) -> None:
        ledger = _Ledger()
        database = _Database(ledger)
        service = AccountDevelopmentService()
        finalization_started = asyncio.Event()
        allow_finalization = asyncio.Event()

        async def finalize(*_: Any, **__: Any) -> None:
            finalization_started.set()
            await allow_finalization.wait()

        with (
            mock.patch("ragtime.userspace.development_accounts.get_db", new=mock.AsyncMock(return_value=database)),
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=mock.AsyncMock(side_effect=RuntimeError("storage failed"))),
            mock.patch.object(service, "_finalize_claim", new=finalize),
        ):
            create_task = asyncio.create_task(service.create_workspace(_principal(), _request()))
            await finalization_started.wait()
            create_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await create_task
            allow_finalization.set()
            await asyncio.sleep(0)

    async def test_finalize_failure_after_core_success_keeps_reserved_uuid_recoverable(self) -> None:
        ledger = _Ledger()
        core = mock.AsyncMock(return_value=SimpleNamespace(id="workspace-1", name="A workspace"))
        service = AccountDevelopmentService()
        database = _Database(ledger)
        with (
            mock.patch("ragtime.userspace.development_accounts.get_db", new=mock.AsyncMock(return_value=database)),
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=core),
            mock.patch.object(service, "_finalize_claim", new=mock.AsyncMock(side_effect=RuntimeError("ledger offline"))),
        ):
            result = await service.create_workspace(_principal(), _request())
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["outcome"], "unconfirmed")
        self.assertEqual(result["workspace_id"], ledger.rows[0].workspaceId)
        self.assertEqual(core.await_count, 1)
        recovered = AccountDevelopmentService()
        with (
            mock.patch("ragtime.userspace.development_accounts.get_db", new=mock.AsyncMock(return_value=database)),
            mock.patch("ragtime.userspace.development_accounts.read_completion_receipt", return_value={"version": 1}),
            mock.patch.object(recovered, "_authorized_workspace", new=mock.AsyncMock(return_value=SimpleNamespace(id="workspace-1", name="A workspace"))),
            mock.patch("ragtime.userspace.development_accounts.userspace_service._workspace_files_dir", return_value=Path("/tmp/workspace/files")),
            mock.patch("ragtime.userspace.development_accounts.userspace_service.create_workspace", new=mock.AsyncMock()),
        ):
            replay = await recovered.create_workspace(_principal(), _request())
        self.assertEqual(replay["status"], "completed")
        self.assertTrue(replay["replayed"])

    async def test_definite_failure_tombstone_replays_same_error_without_redispatch(self) -> None:
        ledger = _Ledger()
        core = mock.AsyncMock(side_effect=RuntimeError("storage failed"))
        with self.assertRaises(HTTPException) as first:
            await self._create(ledger, _request(), core=core)
        self.assertEqual(first.exception.detail, "workspace_create_failed")
        replay_core = mock.AsyncMock()
        with self.assertRaises(HTTPException) as replay:
            await self._create(ledger, _request(), core=replay_core)
        self.assertEqual(replay.exception.detail, "workspace_create_failed")
        self.assertEqual(first.exception.status_code, replay.exception.status_code)
        self.assertEqual(replay_core.await_count, 0)

    async def test_name_conflict_replays_same_safe_conflict(self) -> None:
        ledger = _Ledger()
        core = mock.AsyncMock(side_effect=HTTPException(status_code=409, detail="owner_user_id_name_normalized"))
        with self.assertRaises(HTTPException) as first:
            await self._create(ledger, _request(), core=core)
        with self.assertRaises(HTTPException) as replay:
            await self._create(ledger, _request(), core=mock.AsyncMock())
        self.assertEqual(first.exception.status_code, 409)
        self.assertEqual(first.exception.status_code, replay.exception.status_code)
        self.assertEqual(first.exception.detail, "workspace_name_conflict")
        self.assertEqual(first.exception.detail, replay.exception.detail)

    async def test_completed_but_deleted_workspace_is_unavailable(self) -> None:
        claim = _claim("completed")
        service = AccountDevelopmentService()
        with mock.patch.object(service, "_authorized_workspace", new=mock.AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as error:
                await service._replay_claim(_Database(_Ledger([claim])), _principal(), claim)
        self.assertEqual(error.exception.detail, "workspace_unavailable")

    async def test_lost_membership_denies_user_and_admin_replay_without_bootstrap_writes(self) -> None:
        for principal in (_principal(), DevelopmentPrincipal(user_id="user-1", is_admin=True, auth_source="user")):
            with self.subTest(is_admin=principal.is_admin):
                claim = _claim("completed")
                workspace = mock.AsyncMock()
                workspace.find_first.return_value = None
                service = AccountDevelopmentService()
                with self.assertRaises(HTTPException):
                    await service._replay_claim(_Database(_Ledger([claim]), workspace), principal, claim)
                workspace.find_first.assert_awaited_once()
                self.assertEqual(
                    workspace.find_first.await_args.kwargs["where"]["OR"],
                    [{"ownerUserId": "user-1"}, {"members": {"some": {"userId": "user-1"}}}],
                )
                self.assertEqual(workspace.method_calls, [mock.call.find_first(where=workspace.find_first.await_args.kwargs["where"])])

    async def test_same_key_with_changed_hash_returns_conflict(self) -> None:
        claim = _claim()
        with self.assertRaises(HTTPException) as error:
            await self._create(_Ledger([claim]), _request(name="Changed"), core=mock.AsyncMock())
        self.assertEqual(error.exception.status_code, 409)

    async def test_ineligible_principal_has_stable_account_tools_error(self) -> None:
        with self.assertRaises(HTTPException) as error:
            await AccountDevelopmentService().list_workspaces(
                DevelopmentPrincipal(user_id="user-1", is_admin=False),
                AccountWorkspaceListRequest(),
            )
        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(error.exception.detail, "account_tools_unavailable")


class AccountRequestAndDiscoveryTests(TestCase):
    def test_create_dto_rejects_blank_name_and_internal_ids(self) -> None:
        with self.assertRaises(ValidationError):
            _request(name="   ")
        with self.assertRaises(ValidationError):
            _request(creation_workspace_id="forbidden")

    def test_list_paging_rejects_boolean_string_and_float_values(self) -> None:
        for value in (True, "1", 1.0):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    AccountWorkspaceListRequest(offset=value)

    def test_default_selection_hash_matches_explicit_default_all_with_absent_lists(self) -> None:
        implicit = _request()
        explicit = _request(tool_selection_mode="default_all")
        self.assertEqual(implicit.payload_hash(), explicit.payload_hash())

    def test_list_forwards_unprivileged_search_and_advances_cursor_by_emitted_items(self) -> None:
        updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
        huge_name = "é" * (ACCOUNT_LIST_MAX_BYTES // 2)
        workspaces = [
            SimpleNamespace(id="workspace-1", name="Duplicate", owner_user_id="user-1", updated_at=updated, members=[]),
            SimpleNamespace(id="workspace-2", name="Duplicate", owner_user_id="other", updated_at=updated, members=[SimpleNamespace(user_id="user-1", role="editor")]),
            SimpleNamespace(id="workspace-3", name=huge_name, owner_user_id="user-1", updated_at=updated, members=[]),
        ]
        page = SimpleNamespace(items=workspaces, total=3, offset=7)
        with mock.patch("ragtime.userspace.development_accounts.userspace_service.list_workspaces", new=mock.AsyncMock(return_value=page)) as listed:
            result = asyncio.run(AccountDevelopmentService().list_workspaces(_principal(), AccountWorkspaceListRequest(query="  Duplicate  ", offset=7)))
        listed.assert_awaited_once_with("user-1", offset=7, limit=20, is_admin=False, query="Duplicate")
        self.assertEqual([item["name"] for item in result["items"]], ["Duplicate", "Duplicate"])
        self.assertEqual(result["next_offset"], 9)

    def test_list_rejects_an_oversized_first_item_instead_of_repeating_the_offset(self) -> None:
        updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
        page = SimpleNamespace(
            items=[SimpleNamespace(id="workspace-1", name="x" * ACCOUNT_LIST_MAX_BYTES, owner_user_id="user-1", updated_at=updated, members=[])],
            total=1,
            offset=0,
        )
        with mock.patch("ragtime.userspace.development_accounts.userspace_service.list_workspaces", new=mock.AsyncMock(return_value=page)):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(AccountDevelopmentService().list_workspaces(_principal(), AccountWorkspaceListRequest()))
        self.assertEqual(error.exception.status_code, 413)


class CanonicalWorkspaceDiscoveryTests(IsolatedAsyncioTestCase):
    async def test_full_uuid_query_normalizes_and_preserves_owner_member_acl(self) -> None:
        workspace_model = mock.AsyncMock()
        workspace_model.find_many.return_value = []
        workspace_model.count.return_value = 0
        database = SimpleNamespace(workspace=workspace_model, query_raw=mock.AsyncMock())
        user_id = "user-1"
        workspace_id = UUID("A0B1C2D3-E4F5-4A6B-8C9D-0E1F2A3B4C5D")

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=database)):
            await userspace_service.list_workspaces(user_id, query=str(workspace_id).upper())

        expected_where = {
            "AND": [
                {"OR": [{"ownerUserId": user_id}, {"members": {"some": {"userId": user_id}}}]},
                {"id": str(workspace_id)},
            ]
        }
        self.assertEqual(workspace_model.find_many.await_args.kwargs["where"], expected_where)
        self.assertEqual(workspace_model.count.await_args.kwargs["where"], expected_where)

    async def test_uuid_prefix_is_not_resolved_as_an_id(self) -> None:
        workspace_model = mock.AsyncMock()
        workspace_model.find_many.return_value = []
        workspace_model.count.return_value = 0
        database = SimpleNamespace(workspace=workspace_model, query_raw=mock.AsyncMock())

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=database)):
            await userspace_service.list_workspaces("user-1", query="A0B1C2D3")

        matching_clause = workspace_model.find_many.await_args.kwargs["where"]["AND"][1]
        self.assertNotIn("id", matching_clause)
        self.assertEqual(matching_clause["OR"][0]["name"]["contains"], "A0B1C2D3")
