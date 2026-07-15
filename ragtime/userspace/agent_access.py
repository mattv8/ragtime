"""Workspace external-agent access tokens and external build request ledger.

A workspace owner mints a token that external chat agents use against
``/agent/w/{token}``. Every agent-surface action executes as the minting
user; workspace ACLs are enforced by the userspace services the callers
invoke. The token alone never grants more than that user's current role.
"""

from __future__ import annotations

import importlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

AGENT_ACCESS_SOURCE = "workspace_agent"
AGENT_REPLY_SOURCE = "workspace_agent_reply"
_TOKEN_MAX_LENGTH = 128


def _is_unique_violation(exc: Exception) -> bool:
    try:
        unique_violation_error = getattr(importlib.import_module("prisma.errors"), "UniqueViolationError", None)
    except Exception:
        return False
    return isinstance(unique_violation_error, type) and isinstance(exc, unique_violation_error)


@dataclass(frozen=True)
class AgentAccessContext:
    """Resolved identity and scope for one agent-surface request."""

    access_id: str
    workspace_id: str
    acting_user_id: str
    acting_user_is_admin: bool
    allow_task_submission: bool


def _mint_token() -> str:
    return secrets.token_urlsafe(32)


def _status_payload(
    workspace_id: str,
    record: Any | None,
) -> dict[str, Any]:
    if record is None:
        return {
            "workspace_id": workspace_id,
            "enabled": False,
            "allow_task_submission": True,
            "token": None,
            "created_at": None,
            "last_used_at": None,
            "hit_count": 0,
        }
    enabled = bool(record.enabled)
    return {
        "workspace_id": workspace_id,
        "enabled": enabled,
        "allow_task_submission": bool(record.allowTaskSubmission),
        "token": record.token if enabled else None,
        "created_at": record.createdAt,
        "last_used_at": record.lastUsedAt,
        "hit_count": int(record.hitCount or 0),
    }


async def _require_owner(workspace_id: str, user_id: str) -> None:
    """Owner (or admin) gate for managing agent access."""
    db = await get_db()
    user = await db.user.find_unique(where={"id": user_id})
    is_admin = bool(user and getattr(user, "role", "") == "admin")
    await userspace_service.enforce_workspace_role(workspace_id, user_id, "owner", is_admin=is_admin)


async def get_agent_access_status(workspace_id: str, user_id: str) -> dict[str, Any]:
    await _require_owner(workspace_id, user_id)
    db = await get_db()
    record = await db.workspaceagentaccess.find_unique(where={"workspaceId": workspace_id})
    return _status_payload(workspace_id, record)


async def _update_existing_agent_access(record_id: str, *, allow_task_submission: bool) -> Any:
    db = await get_db()
    return await db.workspaceagentaccess.update(
        where={"id": record_id},
        data={
            "enabled": True,
            "allowTaskSubmission": allow_task_submission,
            "updatedAt": datetime.now(timezone.utc),
        },
    )


async def enable_agent_access(
    workspace_id: str,
    user_id: str,
    *,
    allow_task_submission: bool = True,
) -> dict[str, Any]:
    await _require_owner(workspace_id, user_id)
    db = await get_db()
    record = await db.workspaceagentaccess.find_unique(where={"workspaceId": workspace_id})
    if record is None:
        try:
            record = await db.workspaceagentaccess.create(
                data={
                    "workspaceId": workspace_id,
                    "createdByUserId": user_id,
                    "token": _mint_token(),
                    "enabled": True,
                    "allowTaskSubmission": allow_task_submission,
                }
            )
        except Exception as exc:
            if not _is_unique_violation(exc):
                raise
            record = await db.workspaceagentaccess.find_unique(where={"workspaceId": workspace_id})
            if record is None:
                raise
            record = await _update_existing_agent_access(
                str(record.id),
                allow_task_submission=allow_task_submission,
            )
    else:
        record = await _update_existing_agent_access(
            str(record.id),
            allow_task_submission=allow_task_submission,
        )
    logger.info("Agent access enabled for workspace %s by user %s", workspace_id, user_id)
    return _status_payload(workspace_id, record)


async def disable_agent_access(workspace_id: str, user_id: str) -> dict[str, Any]:
    await _require_owner(workspace_id, user_id)
    db = await get_db()
    record = await db.workspaceagentaccess.find_unique(where={"workspaceId": workspace_id})
    if record is None:
        return _status_payload(workspace_id, None)
    record = await db.workspaceagentaccess.update(
        where={"id": record.id},
        data={"enabled": False, "updatedAt": datetime.now(timezone.utc)},
    )
    logger.info("Agent access disabled for workspace %s by user %s", workspace_id, user_id)
    return _status_payload(workspace_id, record)


async def rotate_agent_access_token(workspace_id: str, user_id: str) -> dict[str, Any]:
    await _require_owner(workspace_id, user_id)
    db = await get_db()
    record = await db.workspaceagentaccess.find_unique(where={"workspaceId": workspace_id})
    if record is None:
        raise HTTPException(status_code=404, detail="Agent access is not configured for this workspace")
    record = await db.workspaceagentaccess.update(
        where={"id": record.id},
        data={
            "createdByUserId": user_id,
            "token": _mint_token(),
            "enabled": True,
            "updatedAt": datetime.now(timezone.utc),
        },
    )
    logger.info("Agent access token rotated for workspace %s by user %s", workspace_id, user_id)
    return _status_payload(workspace_id, record)


async def resolve_agent_access_token(token: str) -> AgentAccessContext:
    """Resolve a public agent token to its acting identity.

    Raises HTTPException(404) for unknown, malformed, or disabled tokens so
    callers cannot distinguish "never existed" from "revoked".
    """
    cleaned = (token or "").strip()
    if not cleaned or len(cleaned) > _TOKEN_MAX_LENGTH:
        raise HTTPException(status_code=404, detail="Unknown agent access token")
    db = await get_db()
    record = await db.workspaceagentaccess.find_unique(where={"token": cleaned})
    if record is None or not record.enabled:
        raise HTTPException(status_code=404, detail="Unknown agent access token")
    user = await db.user.find_unique(where={"id": record.createdByUserId})
    if user is None:
        raise HTTPException(status_code=404, detail="Unknown agent access token")
    try:
        await db.workspaceagentaccess.update(
            where={"id": record.id},
            data={
                "lastUsedAt": datetime.now(timezone.utc),
                "hitCount": {"increment": 1},
            },
        )
    except Exception:
        logger.debug("Agent access usage bookkeeping failed", exc_info=True)
    return AgentAccessContext(
        access_id=str(record.id),
        workspace_id=str(record.workspaceId),
        acting_user_id=str(record.createdByUserId),
        acting_user_is_admin=getattr(user, "role", "") == "admin",
        allow_task_submission=bool(record.allowTaskSubmission),
    )


async def find_external_build_request(user_id: str, source: str, request_id: str) -> Any | None:
    db = await get_db()
    return await db.externalbuildrequest.find_unique(
        where={
            "userId_source_requestId": {
                "userId": user_id,
                "source": source,
                "requestId": request_id,
            }
        }
    )


async def find_external_build_request_by_task(
    user_id: str,
    source: str,
    workspace_id: str,
    task_id: str,
) -> Any | None:
    """Return the external-build ledger row that owns a task."""
    db = await get_db()
    return await db.externalbuildrequest.find_first(
        where={
            "userId": user_id,
            "source": source,
            "workspaceId": workspace_id,
            "taskId": task_id,
        }
    )


async def find_external_build_request_by_conversation(
    user_id: str,
    source: str,
    workspace_id: str,
    conversation_id: str,
) -> Any | None:
    db = await get_db()
    return await db.externalbuildrequest.find_first(
        where={
            "userId": user_id,
            "source": source,
            "workspaceId": workspace_id,
            "conversationId": conversation_id,
        }
    )


async def create_external_build_request(
    *,
    user_id: str,
    source: str,
    request_id: str,
    payload_hash: str,
    workspace_id: str,
) -> Any:
    """Insert a ledger row. Prisma UniqueViolationError propagates to the caller."""
    db = await get_db()
    return await db.externalbuildrequest.create(
        data={
            "userId": user_id,
            "source": source,
            "requestId": request_id,
            "payloadHash": payload_hash,
            "workspaceId": workspace_id,
        }
    )


async def finalize_external_build_request(row_id: str, *, conversation_id: str, task_id: str) -> None:
    db = await get_db()
    await db.externalbuildrequest.update(
        where={"id": row_id},
        data={"conversationId": conversation_id, "taskId": task_id},
    )


async def bind_external_build_request_conversation(row_id: str, *, conversation_id: str) -> None:
    db = await get_db()
    await db.externalbuildrequest.update(
        where={"id": row_id},
        data={"conversationId": conversation_id},
    )


async def delete_external_build_request(row_id: str) -> None:
    """Remove an unfinished idempotency claim after builder startup fails."""
    db = await get_db()
    await db.externalbuildrequest.delete(where={"id": row_id})
