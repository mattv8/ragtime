"""Account-scoped discovery and durable workspace creation for development MCP."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import timedelta
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragtime.core.database import get_db
from ragtime.core.datetimes import utc_now
from ragtime.core.logging import get_logger
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_creation_receipts import CompletionUnconfirmed, read_completion_receipt
from ragtime.userspace.models import CreateWorkspaceRequest
from ragtime.userspace.service import userspace_service

MAX_ACCOUNT_CREATE_CLAIMS = 5
ACCOUNT_CREATE_WINDOW_SECONDS = 60
ACCOUNT_LIST_MAX_BYTES = 24 * 1024
logger = get_logger(__name__)


def _account_operation_log(
    user_id: str,
    operation: str,
    state: str,
    ledger_id: str | None = None,
    exception: BaseException | None = None,
) -> None:
    logger.info(
        "account_operation user_id=%s operation=%s state=%s ledger_id=%s exception_type=%s",
        user_id,
        operation,
        state,
        ledger_id,
        type(exception).__name__ if exception is not None else None,
    )


class AccountWorkspaceListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str | None = Field(default=None, max_length=120)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None


class AccountWorkspaceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    idempotency_key: str = Field(min_length=8, max_length=128)
    sqlite_persistence_mode: Literal["include", "exclude"] = "include"
    tool_selection_mode: Literal["default_all", "custom"] | None = None
    selected_tool_ids: list[str] | None = None
    selected_tool_group_ids: list[str] | None = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("selected_tool_ids", "selected_tool_group_ids")
    @classmethod
    def normalize_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return sorted(set(value))

    def create_request(self) -> CreateWorkspaceRequest:
        return CreateWorkspaceRequest(
            name=self.name,
            description=self.description,
            sqlite_persistence_mode=self.sqlite_persistence_mode,
            tool_selection_mode=self.tool_selection_mode,
            selected_tool_ids=self.selected_tool_ids,
            selected_tool_group_ids=self.selected_tool_group_ids,
        )

    def payload_hash(self) -> str:
        effective_tool_selection_mode = self.tool_selection_mode or (
            "default_all" if self.selected_tool_ids is None and self.selected_tool_group_ids is None else "custom"
        )
        payload = {
            "name": self.name,
            "description": self.description,
            "sqlite_persistence_mode": self.sqlite_persistence_mode,
            "tool_selection_mode": effective_tool_selection_mode,
            "selected_tool_ids": self.selected_tool_ids,
            "selected_tool_group_ids": self.selected_tool_group_ids,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class AccountDevelopmentService:
    """Own account eligibility, ledger transitions, and safe replay semantics."""

    @staticmethod
    def can_use_account_tools(principal: DevelopmentPrincipal) -> bool:
        return principal.auth_source == "user" and principal.credential_id is None and principal.workspace_id is None

    @staticmethod
    def _require_account_principal(principal: DevelopmentPrincipal) -> None:
        if not AccountDevelopmentService.can_use_account_tools(principal):
            raise HTTPException(status_code=403, detail="account_tools_unavailable")

    @staticmethod
    def _ledger_model(db: Any) -> Any:
        model = getattr(db, "workspacedevelopmentcreaterequest", None)
        if model is None:
            raise HTTPException(status_code=503, detail="account_tools_unavailable")
        return model

    async def list_workspaces(self, principal: DevelopmentPrincipal, request: AccountWorkspaceListRequest) -> dict[str, Any]:
        self._require_account_principal(principal)
        page = await userspace_service.list_workspaces(
            principal.user_id,
            offset=request.offset,
            limit=request.limit,
            is_admin=False,
            query=request.query,
        )
        items: list[dict[str, Any]] = []
        truncated = False
        for workspace in page.items:
            role = next((member.role for member in workspace.members if member.user_id == principal.user_id), None)
            item = {
                "workspace_id": workspace.id,
                "name": workspace.name,
                "role": role or ("owner" if workspace.owner_user_id == principal.user_id else None),
                "updated_at": workspace.updated_at.isoformat(),
                "next_call": {"name": "workspace_development_context", "arguments": {"workspace_id": workspace.id}},
            }
            candidate_items = [*items, item]
            candidate_next_offset = page.offset + len(candidate_items)
            candidate = {
                "items": candidate_items,
                "total": page.total,
                "offset": page.offset,
                "next_offset": candidate_next_offset if candidate_next_offset < page.total else None,
            }
            if len(json.dumps(candidate, separators=(",", ":")).encode("utf-8")) > ACCOUNT_LIST_MAX_BYTES:
                truncated = True
                break
            items.append(item)
        next_offset = page.offset + len(items)
        if truncated and not items and page.offset < page.total:
            raise HTTPException(status_code=413, detail="An account workspace item exceeds bounded delivery limit")
        response = {
            "items": items,
            "total": page.total,
            "offset": page.offset,
            "next_offset": next_offset if truncated or next_offset < page.total else None,
        }
        _account_operation_log(principal.user_id, "list", "completed")
        return response

    async def create_workspace(self, principal: DevelopmentPrincipal, request: AccountWorkspaceCreateRequest) -> dict[str, Any]:
        self._require_account_principal(principal)
        payload_hash = request.payload_hash()
        db = await get_db()
        existing_claim = None
        async with db.tx() as tx:
            await tx.query_raw(
                "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"userspace-development-create:{principal.user_id}",
            )
            model = self._ledger_model(tx)
            claim = await model.find_first(where={"userId": principal.user_id, "idempotencyKey": request.idempotency_key})
            if claim is not None:
                if str(getattr(claim, "payloadHash", "")) != payload_hash:
                    raise HTTPException(status_code=409, detail="idempotency_key_conflict")
                existing_claim = claim
            else:
                window_start = utc_now() - timedelta(seconds=ACCOUNT_CREATE_WINDOW_SECONDS)
                claims = await model.count(where={"userId": principal.user_id, "createdAt": {"gte": window_start}})
                if claims >= MAX_ACCOUNT_CREATE_CLAIMS:
                    raise HTTPException(status_code=429, detail={"code": "workspace_create_throttled", "retry_after": ACCOUNT_CREATE_WINDOW_SECONDS})
                claim = await model.create(
                    data={
                        "id": str(uuid4()),
                        "userId": principal.user_id,
                        "idempotencyKey": request.idempotency_key,
                        "payloadHash": payload_hash,
                        "workspaceId": str(uuid4()),
                        "state": "accepted",
                    }
                )
        if existing_claim is not None:
            return await self._replay_claim(db, principal, existing_claim)
        try:
            workspace = await userspace_service.create_workspace(
                request.create_request(),
                principal.user_id,
                creation_workspace_id=str(claim.workspaceId),
                creation_request_id=str(claim.id),
            )
        except BaseException as exc:
            state = "interrupted" if isinstance(exc, CompletionUnconfirmed) or not isinstance(exc, Exception) else "failed"
            error_code = "create_interrupted" if state == "interrupted" else self._failure_code(exc)
            finalized = await self._finalize_safely(db, claim, state=state, error_code=error_code)
            if state == "interrupted":
                if isinstance(exc, (asyncio.CancelledError, SystemExit, KeyboardInterrupt)):
                    raise
                if not finalized:
                    return self._unconfirmed_claim(claim, str(getattr(claim, "state", "accepted")))
                return self._unconfirmed_claim(claim, state)
            if not finalized:
                return self._unconfirmed_claim(claim, str(getattr(claim, "state", "accepted")))
            raise HTTPException(status_code=self._failure_status(error_code), detail=error_code) from None
        try:
            await self._finalize_claim(db, claim, state="completed", completed=True)
        except BaseException as exc:
            _account_operation_log(principal.user_id, "create", "finalize_failed", str(claim.id), exc)
            if isinstance(exc, (asyncio.CancelledError, SystemExit, KeyboardInterrupt)):
                raise
            return self._unconfirmed_claim(claim, str(getattr(claim, "state", "accepted")))
        _account_operation_log(principal.user_id, "create", "completed", str(claim.id))
        return self._completed(workspace.id, workspace.name, replayed=False)

    @staticmethod
    def _failure_code(exc: BaseException) -> str:
        if isinstance(exc, HTTPException) and exc.status_code == 409:
            return "workspace_name_conflict"
        return "workspace_create_failed"

    @staticmethod
    def _failure_status(error_code: str) -> int:
        return 409 if error_code == "workspace_name_conflict" else 500

    async def _finalize_safely(
        self,
        db: Any,
        claim: Any,
        *,
        state: str,
        error_code: str,
    ) -> bool:
        finalize_task = asyncio.create_task(self._finalize_claim(db, claim, state=state, error_code=error_code))
        try:
            await asyncio.shield(finalize_task)
        except BaseException as finalize_error:
            _account_operation_log(str(claim.userId), "create", "finalize_failed", str(claim.id), finalize_error)
            if not finalize_task.done():
                finalize_task.add_done_callback(lambda task: self._log_finalize_task_failure(task, str(claim.userId), str(claim.id)))
            if isinstance(finalize_error, (asyncio.CancelledError, SystemExit, KeyboardInterrupt)):
                raise
            return False
        _account_operation_log(str(claim.userId), "create", state, str(claim.id))
        return True

    @staticmethod
    def _log_finalize_task_failure(task: asyncio.Task[Any], user_id: str, ledger_id: str) -> None:
        if task.cancelled():
            _account_operation_log(user_id, "create", "finalize_failed", ledger_id, asyncio.CancelledError())
            return
        exception = task.exception()
        if exception is not None:
            _account_operation_log(user_id, "create", "finalize_failed", ledger_id, exception)

    async def _finalize_claim(self, db: Any, claim: Any, *, state: str, error_code: str | None = None, completed: bool = False) -> None:
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", f"userspace-development-create:{claim.userId}")
            data: dict[str, Any] = {"state": state, "errorCode": error_code}
            if completed:
                data["completedAt"] = utc_now()
            states = ["accepted", "interrupted"] if completed else ["accepted"]
            await self._ledger_model(tx).update_many(where={"id": claim.id, "state": {"in": states}}, data=data)

    async def _replay_claim(self, db: Any, principal: DevelopmentPrincipal, claim: Any) -> dict[str, Any]:
        async with db.tx() as tx:
            await tx.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", f"userspace-development-create:{principal.user_id}")
            current = await self._ledger_model(tx).find_unique(where={"id": claim.id})
            if current is None:
                _account_operation_log(principal.user_id, "create_replay", "workspace_unavailable", str(claim.id))
                raise HTTPException(status_code=409, detail="workspace_unavailable")
            state = str(getattr(current, "state", "accepted"))
            workspace = await self._authorized_workspace(tx, str(current.workspaceId), principal.user_id)
            if state == "completed":
                if workspace is None:
                    _account_operation_log(principal.user_id, "create_replay", "workspace_unavailable", str(current.id))
                    raise HTTPException(status_code=409, detail="workspace_unavailable")
                _account_operation_log(principal.user_id, "create_replay", "completed", str(current.id))
                return self._completed(str(workspace.id), str(workspace.name), replayed=True)
            if state == "failed":
                error_code = str(getattr(current, "errorCode", None) or "workspace_create_failed")
                _account_operation_log(principal.user_id, "create_replay", "failed", str(current.id))
                raise HTTPException(status_code=self._failure_status(error_code), detail=error_code)
            workspace_dir = userspace_service._workspace_files_dir(str(current.workspaceId)).parent
            receipt = await asyncio.to_thread(
                read_completion_receipt,
                workspace_dir,
                workspace_id=str(current.workspaceId),
                request_id=str(current.id),
                user_id=principal.user_id,
            )
            if receipt is None or workspace is None:
                _account_operation_log(principal.user_id, "create_replay", state, str(current.id))
                return self._unconfirmed_claim(current, state)
            updated = await self._ledger_model(tx).update_many(
                where={"id": current.id, "state": {"in": ["accepted", "interrupted"]}},
                data={"state": "completed", "completedAt": utc_now(), "errorCode": None},
            )
            if not updated:
                current = await self._ledger_model(tx).find_unique(where={"id": current.id})
                if current is None or str(getattr(current, "state", "")) != "completed":
                    _account_operation_log(principal.user_id, "create_replay", state, str(claim.id))
                    return self._unconfirmed_claim(current or claim, state)
            _account_operation_log(principal.user_id, "create_replay", "completed", str(current.id))
            return self._completed(str(workspace.id), str(workspace.name), replayed=True)

    @staticmethod
    async def _authorized_workspace(db: Any, workspace_id: str, user_id: str) -> Any | None:
        return await db.workspace.find_first(
            where={
                "id": workspace_id,
                "OR": [{"ownerUserId": user_id}, {"members": {"some": {"userId": user_id}}}],
            }
        )

    @staticmethod
    def _completed(workspace_id: str, name: str, *, replayed: bool) -> dict[str, Any]:
        return {
            "status": "completed",
            "workspace_id": workspace_id,
            "name": name,
            "replayed": replayed,
            "next_call": {"name": "workspace_development_context", "arguments": {"workspace_id": workspace_id}},
        }

    @staticmethod
    def _unconfirmed_claim(claim: Any, state: str) -> dict[str, Any]:
        return {
            "status": "interrupted" if state == "interrupted" else "pending",
            "workspace_id": str(claim.workspaceId),
            "outcome": "unconfirmed",
            "recovery": (
                "Retry this exact idempotency key to replay the outcome; it will not provision another workspace. "
                "Use the reserved full workspace UUID with workspace_development_workspaces to inspect the outcome. "
                "Do not use a new idempotency key unless the user explicitly asks to create another workspace."
            ),
        }


account_development_service = AccountDevelopmentService()
