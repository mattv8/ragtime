from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel, Field

from ragtime.core.auth import get_external_origin
from ragtime.core.security import get_current_user
from ragtime.userspace.external_api import (
    create_workspace_service_credential,
    delete_revoked_workspace_service_credential,
    get_external_api_manifest_payload,
    list_workspace_api_requests_payload,
    list_workspace_published_endpoints_payload,
    list_workspace_service_credentials_payload,
    publish_workspace_endpoint,
    revoke_workspace_service_credential,
    rotate_workspace_service_credential,
    unpublish_workspace_endpoint,
)
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service

router = APIRouter(prefix="/indexes/userspace/workspaces/{workspace_id}/external-api", tags=["User Space"])


class CreateWorkspaceServiceCredentialRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    endpoint_keys: list[str] = Field(min_length=1)
    expires_at: datetime | None = None


async def _require_workspace_owner_or_admin(workspace_id: str, user: Any) -> None:
    await userspace_service.enforce_workspace_role(
        workspace_id,
        user.id,
        "owner",
        is_admin=getattr(user, "role", None) == "admin",
    )


@router.get("/manifest")
async def get_workspace_external_api_manifest(workspace_id: str, request: Request, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await get_external_api_manifest_payload(
        workspace_id=workspace_id,
        preview_origin=userspace_runtime_service.get_preview_origin(workspace_id, control_plane_origin=get_external_origin(request)),
    )


@router.get("/endpoints")
async def list_workspace_external_api_endpoints(workspace_id: str, request: Request, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await list_workspace_published_endpoints_payload(
        workspace_id=workspace_id,
        preview_origin=userspace_runtime_service.get_preview_origin(workspace_id, control_plane_origin=get_external_origin(request)),
    )


@router.post("/endpoints/{key}/publish")
async def publish_workspace_external_api_endpoint(workspace_id: str, key: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await publish_workspace_endpoint(workspace_id=workspace_id, key=key, user_id=user.id)


@router.delete("/endpoints/{endpoint_id}")
async def unpublish_workspace_external_api_endpoint(workspace_id: str, endpoint_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await unpublish_workspace_endpoint(workspace_id=workspace_id, endpoint_id=endpoint_id, user_id=user.id)


@router.get("/credentials")
async def list_workspace_external_api_credentials(workspace_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await list_workspace_service_credentials_payload(workspace_id=workspace_id)


@router.post("/credentials")
async def create_workspace_external_api_credential(
    workspace_id: str,
    payload: CreateWorkspaceServiceCredentialRequest,
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await create_workspace_service_credential(
        workspace_id=workspace_id,
        user_id=user.id,
        label=payload.label,
        endpoint_keys=payload.endpoint_keys,
        expires_at=payload.expires_at,
    )


@router.post("/credentials/{credential_id}/rotate")
async def rotate_workspace_external_api_credential(workspace_id: str, credential_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await rotate_workspace_service_credential(workspace_id=workspace_id, credential_id=credential_id, user_id=user.id)


@router.delete("/credentials/{credential_id}")
async def revoke_workspace_external_api_credential(workspace_id: str, credential_id: str, user: Any = Depends(get_current_user)) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await revoke_workspace_service_credential(workspace_id=workspace_id, credential_id=credential_id, user_id=user.id)


@router.delete("/credentials/{credential_id}/record", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace_external_api_credential_record(
    workspace_id: str,
    credential_id: str,
    user: Any = Depends(get_current_user),
) -> Response:
    await _require_workspace_owner_or_admin(workspace_id, user)
    await delete_revoked_workspace_service_credential(workspace_id=workspace_id, credential_id=credential_id, user_id=user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/requests")
async def list_workspace_external_api_requests(
    workspace_id: str,
    cursor: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    user: Any = Depends(get_current_user),
) -> dict[str, Any]:
    await _require_workspace_owner_or_admin(workspace_id, user)
    return await list_workspace_api_requests_payload(workspace_id=workspace_id, cursor=cursor, limit=limit)
