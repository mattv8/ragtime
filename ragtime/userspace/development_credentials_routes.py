"""Session-only management endpoints for workspace development credentials."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ragtime.core.database import get_db
from ragtime.core.security import get_current_user
from ragtime.userspace.development_access import (
    DEVELOPMENT_SCOPES,
    create_workspace_development_credential,
    development_credential_response,
    revoke_workspace_development_credential,
    rotate_workspace_development_credential,
)
from ragtime.userspace.service import userspace_service

router = APIRouter(prefix="/indexes/userspace/development/workspaces/{workspace_id}/credentials", tags=["User Space Development"])


class CreateWorkspaceDevelopmentCredentialRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=lambda: sorted(DEVELOPMENT_SCOPES))
    expires_at: datetime | None = None


async def _require_workspace_owner_or_admin(workspace_id: str, user: Any) -> None:
    await userspace_service.enforce_workspace_role(workspace_id, user.id, "owner", is_admin=getattr(user, "role", None) == "admin")


@router.get("")
async def list_workspace_development_credentials(workspace_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    db = await get_db()
    rows = await db.workspacedevelopmentcredential.find_many(where={"workspaceId": workspace_id}, order={"createdAt": "asc"})
    return {"items": [development_credential_response(row) for row in rows]}


@router.post("")
async def create_workspace_development_credential_route(
    workspace_id: str, payload: CreateWorkspaceDevelopmentCredentialRequest, user: Any = Depends(get_current_user)
) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await create_workspace_development_credential(
        workspace_id=workspace_id, user_id=user.id, name=payload.name, scopes=payload.scopes, expires_at=payload.expires_at
    )


@router.post("/{credential_id}/rotate")
async def rotate_workspace_development_credential_route(workspace_id: str, credential_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await rotate_workspace_development_credential(workspace_id=workspace_id, credential_id=credential_id)


@router.delete("/{credential_id}")
async def revoke_workspace_development_credential_route(workspace_id: str, credential_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await revoke_workspace_development_credential(workspace_id=workspace_id, credential_id=credential_id)
