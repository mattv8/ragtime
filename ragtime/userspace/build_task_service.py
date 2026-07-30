"""Start, follow, and reply to externally submitted builder tasks.

Transport-agnostic: the agent HTTP routes (and any future MCP wrapper) call
into this service. Reuses the existing background chat task pipeline; it
never duplicates builder execution logic.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.indexer.repository import repository
from ragtime.indexer.tool_selection import resolve_effective_tool_ids
from ragtime.userspace.agent_access import (
    AGENT_ACCESS_SOURCE,
    AGENT_REPLY_SOURCE,
    bind_external_build_request_conversation,
    create_external_build_request,
    delete_external_build_request,
    finalize_external_build_request,
    find_external_build_request,
    find_external_build_request_by_conversation,
)
from ragtime.userspace.agent_briefs import (
    BuildBriefInput,
    compute_brief_payload_hash,
    render_build_brief,
)

logger = get_logger(__name__)

_STALL_SECONDS = 180
_DEFAULT_MAX_RESULT_CHARS = 20000
_MIN_RESULT_CHARS = 1000
_MAX_RESULT_CHARS = 100_000


def _compute_reply_payload_hash(workspace_id: str, task_id: str, message: str) -> str:
    payload = {
        "message": (message or "").strip(),
        "task_id": task_id,
        "workspace_id": workspace_id,
    }
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _is_unique_violation(exc: Exception) -> bool:
    try:
        unique_violation_error = getattr(importlib.import_module("prisma.errors"), "UniqueViolationError", None)
    except Exception:
        return False
    return isinstance(unique_violation_error, type) and isinstance(exc, unique_violation_error)


class WorkspaceBuildTaskService:
    """External build task lifecycle acting as the token's minting user."""

    async def _load_prisma_user(self, user_id: str) -> Any:
        db = await get_db()
        user = await db.user.find_unique(where={"id": user_id})
        if user is None:
            raise HTTPException(status_code=404, detail="Acting user no longer exists")
        return user

    async def _enforce_editor(self, workspace_id: str, user: Any) -> Any:
        from ragtime.userspace.service import userspace_service

        is_admin = getattr(user, "role", "") == "admin"
        return await userspace_service.enforce_workspace_role(workspace_id, user.id, "editor", is_admin=is_admin)

    async def _validate_brief(self, workspace: Any, brief: BuildBriefInput, user: Any) -> None:
        from ragtime.userspace.service import userspace_service

        if brief.preserve_paths:
            files = await userspace_service.list_workspace_files(
                workspace.id,
                user.id,
                is_admin=getattr(user, "role", "") == "admin",
            )
            existing = {str(f.path) for f in files}
            missing = [path for path in brief.preserve_paths if str(path).strip().lstrip("/") not in existing]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"preserve_paths not found in workspace: {', '.join(sorted(missing))}",
                )
        if brief.data_component_ids:
            selected = await resolve_effective_tool_ids(
                tool_selection_mode=getattr(workspace, "tool_selection_mode", ""),
                selected_tool_ids=getattr(workspace, "selected_tool_ids", []),
                selected_tool_group_ids=getattr(workspace, "selected_tool_group_ids", []),
                list_healthy_enabled_tool_ids=repository.list_healthy_enabled_tool_ids,
                list_enabled_tool_ids=repository.list_enabled_tool_ids,
                get_tool_ids_for_groups=repository.get_tool_ids_for_groups,
            )
            selected = await userspace_service.filter_tool_ids_for_workspace_owner(workspace, selected)
            invalid = [cid for cid in brief.data_component_ids if cid not in set(selected)]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"data_component_ids not selected for this workspace: {', '.join(sorted(invalid))}",
                )

    async def _recover_latest_task_for_conversation(self, conversation_id: str) -> Any | None:
        db = await get_db()
        return await db.chattask.find_first(
            where={"conversationId": conversation_id},
            order={"createdAt": "desc"},
        )

    async def _deduplicated_response(
        self,
        existing: Any,
        payload_hash: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        if str(existing.workspaceId or "") != workspace_id:
            raise HTTPException(
                status_code=409,
                detail="idempotency_key was already used for another workspace",
            )
        if str(existing.payloadHash) != payload_hash:
            raise HTTPException(
                status_code=409,
                detail="idempotency_key was already used with a different brief payload",
            )
        conversation_id = str(existing.conversationId or "")
        if not conversation_id:
            raise HTTPException(
                status_code=409,
                detail="A submission with this idempotency_key is still being created; retry shortly",
            )
        task_id = str(existing.taskId or "")
        if not task_id:
            latest_task = await self._recover_latest_task_for_conversation(conversation_id)
            if latest_task is None:
                raise HTTPException(
                    status_code=409,
                    detail="A submission with this idempotency_key is still being created; retry shortly",
                )
            task_id = str(latest_task.id)
            await finalize_external_build_request(
                str(existing.id),
                conversation_id=conversation_id,
                task_id=task_id,
            )
        status = "unknown"
        if task_id:
            task = await repository.get_chat_task(task_id)
            if task is not None:
                status = _status_value(task.status)
        return {
            "deduplicated": True,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "status": status,
        }

    async def _deduplicated_reply_response(
        self,
        existing: Any,
        payload_hash: str,
        workspace_id: str,
    ) -> dict[str, Any]:
        if str(existing.workspaceId or "") != workspace_id:
            raise HTTPException(
                status_code=409,
                detail="idempotency_key was already used for another workspace",
            )
        if str(existing.payloadHash) != payload_hash:
            raise HTTPException(
                status_code=409,
                detail="idempotency_key was already used with a different reply payload",
            )
        task_id = str(existing.taskId or "")
        if not task_id:
            raise HTTPException(
                status_code=409,
                detail="A reply with this idempotency_key is still being created; retry shortly",
            )
        task = await repository.get_chat_task(task_id)
        return {
            "deduplicated": True,
            "conversation_id": str(existing.conversationId or ""),
            "task_id": task_id,
            "status": _status_value(task.status) if task is not None else "unknown",
        }

    async def start_build_task(
        self,
        workspace_id: str,
        acting_user_id: str,
        brief: BuildBriefInput,
    ) -> dict[str, Any]:
        user = await self._load_prisma_user(acting_user_id)
        workspace = await self._enforce_editor(workspace_id, user)
        await self._validate_brief(workspace, brief, user)

        payload_hash = compute_brief_payload_hash(brief)
        existing = await find_external_build_request(acting_user_id, AGENT_ACCESS_SOURCE, brief.idempotency_key)
        if existing is not None:
            return await self._deduplicated_response(existing, payload_hash, workspace_id)

        try:
            ledger_row = await create_external_build_request(
                user_id=acting_user_id,
                source=AGENT_ACCESS_SOURCE,
                request_id=brief.idempotency_key,
                payload_hash=payload_hash,
                workspace_id=workspace_id,
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raced = await find_external_build_request(acting_user_id, AGENT_ACCESS_SOURCE, brief.idempotency_key)
                if raced is not None:
                    return await self._deduplicated_response(raced, payload_hash, workspace_id)
            raise

        conversation = None
        try:
            conversation = await repository.create_conversation(
                title=brief.title.strip()[:120],
                user_id=acting_user_id,
                workspace_id=workspace_id,
            )
            await bind_external_build_request_conversation(
                ledger_row.id,
                conversation_id=conversation.id,
            )

            from ragtime.indexer.models import SendMessageRequest
            from ragtime.indexer.routes import _send_background_message_to_loaded_conversation

            rendered = render_build_brief(brief, workspace_name=str(getattr(workspace, "name", "")))
            result = await _send_background_message_to_loaded_conversation(
                conversation,
                SendMessageRequest(message=rendered),
                user,
                workspace_id=workspace_id,
            )
        except Exception:
            if conversation is not None:
                try:
                    await repository.delete_conversation(conversation.id)
                except Exception:
                    logger.warning(
                        "Failed to remove incomplete external-build conversation %s",
                        conversation.id,
                        exc_info=True,
                    )
            try:
                await delete_external_build_request(ledger_row.id)
            except Exception:
                logger.warning(
                    "Failed to release external-build idempotency claim %s",
                    ledger_row.id,
                    exc_info=True,
                )
            raise
        task = result["task"]
        await finalize_external_build_request(ledger_row.id, conversation_id=conversation.id, task_id=task.id)
        logger.info(
            "External build task started: workspace=%s conversation=%s task=%s user=%s",
            workspace_id,
            conversation.id,
            task.id,
            acting_user_id,
        )
        return {
            "deduplicated": False,
            "conversation_id": conversation.id,
            "task_id": task.id,
            "status": _status_value(task.status),
        }

    async def get_build_task(
        self,
        workspace_id: str,
        acting_user_id: str,
        task_id: str,
        max_result_chars: int = _DEFAULT_MAX_RESULT_CHARS,
    ) -> dict[str, Any]:
        max_result_chars = max(_MIN_RESULT_CHARS, min(int(max_result_chars), _MAX_RESULT_CHARS))
        user = await self._load_prisma_user(acting_user_id)
        task = await repository.get_chat_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        conversation = await repository.get_conversation(task.conversation_id)
        if conversation is None or str(getattr(conversation, "workspace_id", "") or "") != workspace_id:
            raise HTTPException(status_code=404, detail="Task not found")
        ledger_row = await find_external_build_request_by_conversation(
            acting_user_id,
            AGENT_ACCESS_SOURCE,
            workspace_id,
            task.conversation_id,
        )
        if ledger_row is None:
            raise HTTPException(status_code=404, detail="Task not found")
        has_access = await repository.check_conversation_access(
            task.conversation_id,
            acting_user_id,
            is_admin=getattr(user, "role", "") == "admin",
            workspace_id=workspace_id,
        )
        if not has_access:
            raise HTTPException(status_code=404, detail="Task not found")

        status = _status_value(task.status)
        result_text = task.response_content or ""
        truncated = len(result_text) > max_result_chars
        if truncated:
            result_text = result_text[:max_result_chars]
        possibly_stalled = False
        if status == "running" and task.last_update_at is not None:
            last_update = task.last_update_at
            if last_update.tzinfo is None:
                last_update = last_update.replace(tzinfo=timezone.utc)
            possibly_stalled = (datetime.now(timezone.utc) - last_update).total_seconds() > _STALL_SECONDS
        return {
            "task_id": task.id,
            "conversation_id": task.conversation_id,
            "status": status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "last_update_at": task.last_update_at,
            "possibly_stalled": possibly_stalled,
            "result": result_text if status == "completed" else None,
            "result_truncated": truncated,
            "error": task.error_message,
        }

    async def reply_to_build_task(
        self,
        workspace_id: str,
        acting_user_id: str,
        task_id: str,
        message: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        cleaned = (message or "").strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="message is required")
        request_id = (idempotency_key or "").strip()
        if not request_id:
            raise HTTPException(status_code=400, detail="idempotency_key is required")
        user = await self._load_prisma_user(acting_user_id)
        await self._enforce_editor(workspace_id, user)
        task = await repository.get_chat_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        conversation = await repository.get_conversation(task.conversation_id)
        if conversation is None or str(getattr(conversation, "workspace_id", "") or "") != workspace_id:
            raise HTTPException(status_code=404, detail="Task not found")
        ledger_row = await find_external_build_request_by_conversation(
            acting_user_id,
            AGENT_ACCESS_SOURCE,
            workspace_id,
            task.conversation_id,
        )
        if ledger_row is None:
            raise HTTPException(status_code=404, detail="Task not found")
        has_access = await repository.check_conversation_access(
            task.conversation_id,
            acting_user_id,
            is_admin=getattr(user, "role", "") == "admin",
            workspace_id=workspace_id,
        )
        if not has_access:
            raise HTTPException(status_code=404, detail="Task not found")
        payload_hash = _compute_reply_payload_hash(workspace_id, task_id, cleaned)
        existing_reply = await find_external_build_request(acting_user_id, AGENT_REPLY_SOURCE, request_id)
        if existing_reply is not None:
            return await self._deduplicated_reply_response(existing_reply, payload_hash, workspace_id)
        current_task_id = str(getattr(ledger_row, "taskId", "") or "")
        if current_task_id and current_task_id != task_id:
            current_task = await repository.get_chat_task(current_task_id)
            if current_task is None:
                raise HTTPException(status_code=409, detail="Current task state is unavailable; retry shortly")
            if str(getattr(current_task, "conversation_id", "") or "") != str(task.conversation_id):
                raise HTTPException(status_code=404, detail="Task not found")
            return {
                "deduplicated": False,
                "conversation_id": task.conversation_id,
                "task_id": current_task.id,
                "status": _status_value(current_task.status),
            }
        if _status_value(task.status) in {"pending", "running"}:
            raise HTTPException(status_code=409, detail="Task is still running; wait before replying")
        if getattr(conversation, "parent_conversation_id", None):
            raise HTTPException(status_code=403, detail="Subagent conversations are read-only")

        try:
            reply_ledger = await create_external_build_request(
                user_id=acting_user_id,
                source=AGENT_REPLY_SOURCE,
                request_id=request_id,
                payload_hash=payload_hash,
                workspace_id=workspace_id,
            )
        except Exception as exc:
            if _is_unique_violation(exc):
                raced = await find_external_build_request(acting_user_id, AGENT_REPLY_SOURCE, request_id)
                if raced is not None:
                    return await self._deduplicated_reply_response(raced, payload_hash, workspace_id)
            raise

        from ragtime.indexer.models import SendMessageRequest
        from ragtime.indexer.routes import _send_background_message_to_loaded_conversation

        try:
            result = await _send_background_message_to_loaded_conversation(
                conversation,
                SendMessageRequest(message=cleaned),
                user,
                workspace_id=workspace_id,
            )
        except Exception:
            await delete_external_build_request(reply_ledger.id)
            raise
        new_task = result["task"]
        await finalize_external_build_request(
            reply_ledger.id,
            conversation_id=task.conversation_id,
            task_id=new_task.id,
        )
        await finalize_external_build_request(
            ledger_row.id,
            conversation_id=task.conversation_id,
            task_id=new_task.id,
        )
        return {
            "deduplicated": False,
            "conversation_id": task.conversation_id,
            "task_id": new_task.id,
            "status": _status_value(new_task.status),
        }


build_task_service = WorkspaceBuildTaskService()
