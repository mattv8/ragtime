from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Query, Request

from ragtime.core.security import get_current_user, get_current_user_optional
from ragtime.indexer.repository import repository
from ragtime.userspace.models import (
    CreateSnapshotRequest,
    CreateWorkspaceRequest,
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    PaginatedWorkspacesResponse,
    RestoreSnapshotResponse,
    UpdateWorkspaceMembersRequest,
    UpdateWorkspaceRequest,
    UpdateWorkspaceShareAccessRequest,
    UpdateWorkspaceShareSlugRequest,
    UpsertWorkspaceFileRequest,
    UserSpaceAvailableTool,
    UserSpaceFileInfo,
    UserSpaceFileResponse,
    UserSpaceSharedPreviewResponse,
    UserSpaceSnapshot,
    UserSpaceWorkspace,
    UserSpaceWorkspaceShareLink,
    UserSpaceWorkspaceShareLinkStatus,
    WorkspaceShareSlugAvailabilityResponse,
)
from ragtime.userspace.service import userspace_service

router = APIRouter(prefix="/indexes/userspace", tags=["User Space"])


async def _normalize_selected_tool_ids(
    selected_tool_ids: list[str] | None,
) -> list[str] | None:
    if selected_tool_ids is None:
        return None

    tool_configs = await repository.list_tool_configs(enabled_only=True)
    enabled_tool_ids = {tool.id for tool in tool_configs if tool.id}

    normalized: list[str] = []
    seen: set[str] = set()
    for tool_id in selected_tool_ids:
        if tool_id not in enabled_tool_ids or tool_id in seen:
            continue
        normalized.append(tool_id)
        seen.add(tool_id)
    return normalized


@router.get("/tools", response_model=list[UserSpaceAvailableTool])
async def list_userspace_tools(user: Any = Depends(get_current_user)):
    del user
    tool_configs = await repository.list_tool_configs(enabled_only=True)
    results: list[UserSpaceAvailableTool] = []
    for tool in tool_configs:
        tool_id = tool.id
        if not tool_id:
            continue
        results.append(
            UserSpaceAvailableTool(
                id=tool_id,
                name=tool.name,
                tool_type=tool.tool_type.value,
                description=tool.description,
            )
        )
    return results


@router.get("/workspaces", response_model=PaginatedWorkspacesResponse)
async def list_workspaces(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    user: Any = Depends(get_current_user),
):
    return await userspace_service.list_workspaces(user.id, offset=offset, limit=limit)


@router.post("/workspaces", response_model=UserSpaceWorkspace)
async def create_workspace(
    request: CreateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    request.selected_tool_ids = (
        await _normalize_selected_tool_ids(request.selected_tool_ids) or []
    )
    return await userspace_service.create_workspace(request, user.id)


@router.get("/workspaces/{workspace_id}", response_model=UserSpaceWorkspace)
async def get_workspace(workspace_id: str, user: Any = Depends(get_current_user)):
    return await userspace_service.get_workspace(workspace_id, user.id)


@router.put("/workspaces/{workspace_id}", response_model=UserSpaceWorkspace)
async def update_workspace(
    workspace_id: str,
    request: UpdateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    request.selected_tool_ids = await _normalize_selected_tool_ids(
        request.selected_tool_ids
    )
    return await userspace_service.update_workspace(workspace_id, request, user.id)


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    await userspace_service.enforce_workspace_role(workspace_id, user.id, "owner")
    await repository.delete_workspace_conversations(workspace_id)
    await userspace_service.delete_workspace(workspace_id, user.id)
    return {"success": True}


@router.put("/workspaces/{workspace_id}/members", response_model=UserSpaceWorkspace)
async def update_workspace_members(
    workspace_id: str,
    request: UpdateWorkspaceMembersRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.update_workspace_members(
        workspace_id, request, user.id
    )


@router.get("/workspaces/{workspace_id}/files", response_model=list[UserSpaceFileInfo])
async def list_workspace_files(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.list_workspace_files(workspace_id, user.id)


@router.put(
    "/workspaces/{workspace_id}/files/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def upsert_workspace_file(
    workspace_id: str,
    file_path: str,
    request: UpsertWorkspaceFileRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.upsert_workspace_file(
        workspace_id, file_path, request, user.id
    )


@router.get(
    "/workspaces/{workspace_id}/files/{file_path:path}",
    response_model=UserSpaceFileResponse,
)
async def get_workspace_file(
    workspace_id: str,
    file_path: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_workspace_file(workspace_id, file_path, user.id)


@router.delete("/workspaces/{workspace_id}/files/{file_path:path}")
async def delete_workspace_file(
    workspace_id: str,
    file_path: str,
    user: Any = Depends(get_current_user),
):
    await userspace_service.delete_workspace_file(workspace_id, file_path, user.id)
    return {"success": True}


@router.post(
    "/workspaces/{workspace_id}/execute-component",
    response_model=ExecuteComponentResponse,
)
async def execute_component(
    workspace_id: str,
    request: ExecuteComponentRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.execute_component(workspace_id, request, user.id)


@router.post(
    "/workspaces/{workspace_id}/share-link",
    response_model=UserSpaceWorkspaceShareLink,
)
async def create_workspace_share_link(
    workspace_id: str,
    request: Request,
    rotate_token: bool = Query(default=False),
    user: Any = Depends(get_current_user),
):
    base_url = str(request.base_url).rstrip("/")
    return await userspace_service.create_workspace_share_link(
        workspace_id,
        user.id,
        base_url=base_url,
        rotate_token=rotate_token,
    )


@router.get(
    "/workspaces/{workspace_id}/share-link",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def get_workspace_share_link(
    workspace_id: str,
    request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = str(request.base_url).rstrip("/")
    return await userspace_service.get_workspace_share_link_status(
        workspace_id,
        user.id,
        base_url=base_url,
    )


@router.delete(
    "/workspaces/{workspace_id}/share-link",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def revoke_workspace_share_link(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.revoke_workspace_share_link(
        workspace_id,
        user.id,
    )


@router.put(
    "/workspaces/{workspace_id}/share-link/access",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def update_workspace_share_access(
    workspace_id: str,
    request: UpdateWorkspaceShareAccessRequest,
    base_request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = str(base_request.base_url).rstrip("/")
    return await userspace_service.update_workspace_share_access(
        workspace_id,
        request,
        user.id,
        base_url=base_url,
    )


@router.put(
    "/workspaces/{workspace_id}/share-link/slug",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def update_workspace_share_slug(
    workspace_id: str,
    request: UpdateWorkspaceShareSlugRequest,
    base_request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = str(base_request.base_url).rstrip("/")
    return await userspace_service.update_workspace_share_slug(
        workspace_id,
        request.slug,
        user.id,
        base_url=base_url,
    )


@router.get(
    "/workspaces/{workspace_id}/share-link/availability",
    response_model=WorkspaceShareSlugAvailabilityResponse,
)
async def check_workspace_share_slug_availability(
    workspace_id: str,
    slug: str = Query(min_length=1, max_length=120),
    user: Any = Depends(get_current_user),
):
    return await userspace_service.check_workspace_share_slug_availability(
        workspace_id,
        slug,
        user.id,
    )


@router.get(
    "/shared/{share_token}",
    response_model=UserSpaceSharedPreviewResponse,
)
async def get_shared_preview(
    share_token: str,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.get_shared_preview(
        share_token,
        current_user=user,
        password=share_password,
    )


@router.get(
    "/shared/{owner_username}/{share_slug}",
    response_model=UserSpaceSharedPreviewResponse,
)
async def get_shared_preview_by_slug(
    owner_username: str,
    share_slug: str,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.get_shared_preview_by_slug(
        owner_username,
        share_slug,
        current_user=user,
        password=share_password,
    )


@router.post(
    "/shared/{share_token}/execute-component",
    response_model=ExecuteComponentResponse,
)
async def execute_shared_component(
    share_token: str,
    request: ExecuteComponentRequest,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.execute_shared_component(
        share_token,
        request,
        current_user=user,
        password=share_password,
    )


@router.post(
    "/shared/{owner_username}/{share_slug}/execute-component",
    response_model=ExecuteComponentResponse,
)
async def execute_shared_component_by_slug(
    owner_username: str,
    share_slug: str,
    request: ExecuteComponentRequest,
    share_password: str | None = Header(
        default=None,
        alias="X-UserSpace-Share-Password",
    ),
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.execute_shared_component_by_slug(
        owner_username,
        share_slug,
        request,
        current_user=user,
        password=share_password,
    )


@router.get(
    "/workspaces/{workspace_id}/snapshots", response_model=list[UserSpaceSnapshot]
)
async def list_snapshots(workspace_id: str, user: Any = Depends(get_current_user)):
    return await userspace_service.list_snapshots(workspace_id, user.id)


@router.post("/workspaces/{workspace_id}/snapshots", response_model=UserSpaceSnapshot)
async def create_snapshot(
    workspace_id: str,
    request: CreateSnapshotRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.create_snapshot(
        workspace_id, user.id, request.message
    )


@router.post(
    "/workspaces/{workspace_id}/snapshots/{snapshot_id}/restore",
    response_model=RestoreSnapshotResponse,
)
async def restore_snapshot(
    workspace_id: str,
    snapshot_id: str,
    user: Any = Depends(get_current_user),
):
    snapshot = await userspace_service.restore_snapshot(
        workspace_id,
        snapshot_id,
        user.id,
    )
    return RestoreSnapshotResponse(
        restored_snapshot_id=snapshot.id,
        file_count=snapshot.file_count,
    )
