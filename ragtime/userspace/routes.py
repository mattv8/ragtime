from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response

from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret
from ragtime.core.git import check_repo_visibility as git_check_visibility
from ragtime.core.git import fetch_branches as git_fetch_branches
from ragtime.core.logging import get_logger
from ragtime.core.security import (
    get_current_user,
    get_current_user_optional,
    require_admin,
)
from ragtime.indexer.models import (
    CheckRepoVisibilityRequest,
    FetchBranchesRequest,
    FetchBranchesResponse,
    RepoVisibilityResponse,
)
from ragtime.indexer.repository import repository
from ragtime.userspace.models import (
    BrowseUserspaceMountSourceRequest,
    CreateSnapshotRequest,
    CreateUserspaceMountSourceRequest,
    CreateWorkspaceMountRequest,
    CreateWorkspaceRequest,
    DeleteUserspaceMountSourceResponse,
    DeleteWorkspaceEnvVarResponse,
    DeleteWorkspaceMountResponse,
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    MountableSource,
    MountSourceAffectedWorkspacesResponse,
    PaginatedWorkspacesResponse,
    RestoreSnapshotResponse,
    SwitchSnapshotBranchRequest,
    UpdateSnapshotRequest,
    UpdateUserspaceMountSourceRequest,
    UpdateWorkspaceMembersRequest,
    UpdateWorkspaceMountRequest,
    UpdateWorkspaceRequest,
    UpdateWorkspaceShareAccessRequest,
    UpdateWorkspaceShareSlugRequest,
    UpsertWorkspaceEnvVarRequest,
    UpsertWorkspaceFileRequest,
    UserSpaceAcknowledgeChangedFilePathRequest,
    UserSpaceAvailableTool,
    UserSpaceChangedFileStateResponse,
    UserSpaceFileInfo,
    UserSpaceFileResponse,
    UserspaceMountSource,
    UserSpaceSharedPreviewResponse,
    UserSpaceSnapshot,
    UserSpaceSnapshotDiffSummaryResponse,
    UserSpaceSnapshotFileDiffResponse,
    UserSpaceSnapshotTimelineResponse,
    UserSpaceWorkspace,
    UserSpaceWorkspaceEnvVar,
    UserSpaceWorkspaceScmConnectionRequest,
    UserSpaceWorkspaceScmConnectionResponse,
    UserSpaceWorkspaceScmExportRequest,
    UserSpaceWorkspaceScmImportRequest,
    UserSpaceWorkspaceScmPreviewRequest,
    UserSpaceWorkspaceScmPreviewResponse,
    UserSpaceWorkspaceScmSyncResponse,
    UserSpaceWorkspaceShareLink,
    UserSpaceWorkspaceShareLinkStatus,
    WorkspaceMount,
    WorkspaceMountBrowseRequest,
    WorkspaceMountBrowseResponse,
    WorkspaceMountSyncPreviewRequest,
    WorkspaceMountSyncPreviewResponse,
    WorkspaceMountSyncRequest,
    WorkspaceMountSyncResponse,
    WorkspaceShareSlugAvailabilityResponse,
)
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service

logger = get_logger(__name__)

router = APIRouter(prefix="/indexes/userspace", tags=["User Space"])

_USERSPACE_SURFACE_HEADER = "X-Ragtime-Userspace-Surface"
_USERSPACE_TREE_MOUNTS_HEADER = "X-Ragtime-Userspace-Tree-Includes-Mounts"


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
                group_id=tool.group_id,
                group_name=tool.group_name,
            )
        )
    return results


@router.get("/tool-groups")
async def list_userspace_tool_groups(user: Any = Depends(get_current_user)):
    del user
    groups = await repository.list_tool_groups()
    return [g.model_dump() for g in groups]


@router.get("/workspaces", response_model=PaginatedWorkspacesResponse)
async def list_workspaces(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    include_all: bool = Query(default=False),
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin" and include_all
    return await userspace_service.list_workspaces(
        user.id, offset=offset, limit=limit, is_admin=is_admin
    )


@router.post("/workspaces", response_model=UserSpaceWorkspace)
async def create_workspace(
    request: CreateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    request.selected_tool_ids = await _normalize_selected_tool_ids(
        request.selected_tool_ids
    )
    # selected_tool_group_ids are validated in service layer
    return await userspace_service.create_workspace(request, user.id)


@router.get("/workspaces/{workspace_id}", response_model=UserSpaceWorkspace)
async def get_workspace(workspace_id: str, user: Any = Depends(get_current_user)):
    return await userspace_service.get_workspace(workspace_id, user.id)


@router.get(
    "/workspaces/{workspace_id}/scm",
    response_model=UserSpaceWorkspaceScmConnectionResponse,
)
async def get_workspace_scm_connection(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_workspace_scm_connection(workspace_id, user.id)


@router.put(
    "/workspaces/{workspace_id}/scm",
    response_model=UserSpaceWorkspaceScmConnectionResponse,
)
async def update_workspace_scm_connection(
    workspace_id: str,
    request: UserSpaceWorkspaceScmConnectionRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.update_workspace_scm_connection(
        workspace_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/scm/preview-sync",
    response_model=UserSpaceWorkspaceScmPreviewResponse,
)
async def preview_workspace_scm_sync(
    workspace_id: str,
    request: UserSpaceWorkspaceScmPreviewRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.preview_workspace_scm_sync(
        workspace_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/scm/preview-import",
    response_model=UserSpaceWorkspaceScmPreviewResponse,
)
async def preview_workspace_scm_import(
    workspace_id: str,
    request: UserSpaceWorkspaceScmPreviewRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.preview_workspace_scm_import(
        workspace_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/scm/import",
    response_model=UserSpaceWorkspaceScmSyncResponse,
)
async def import_workspace_from_scm(
    workspace_id: str,
    request: UserSpaceWorkspaceScmImportRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.import_workspace_from_scm(
        workspace_id,
        user.id,
        request,
    )
    await userspace_runtime_service.invalidate_workspace_runtime_state(workspace_id)
    return result


@router.post(
    "/workspaces/{workspace_id}/scm/preview-export",
    response_model=UserSpaceWorkspaceScmPreviewResponse,
)
async def preview_workspace_scm_export(
    workspace_id: str,
    request: UserSpaceWorkspaceScmPreviewRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.preview_workspace_scm_export(
        workspace_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/scm/export",
    response_model=UserSpaceWorkspaceScmSyncResponse,
)
async def export_workspace_to_scm(
    workspace_id: str,
    request: UserSpaceWorkspaceScmExportRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.export_workspace_to_scm(
        workspace_id,
        user.id,
        request,
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id, "scm_export"
    )
    return result


@router.put("/workspaces/{workspace_id}", response_model=UserSpaceWorkspace)
async def update_workspace(
    workspace_id: str,
    request: UpdateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    request.selected_tool_ids = await _normalize_selected_tool_ids(
        request.selected_tool_ids
    )
    is_admin = user.role == "admin"
    result = await userspace_service.update_workspace(
        workspace_id, request, user.id, is_admin=is_admin
    )
    if is_admin and result.owner_user_id != user.id:
        logger.info(
            "Admin '%s' updated workspace '%s' (owner: %s)",
            user.username,
            workspace_id,
            result.owner_user_id,
        )
    return result


@router.post(
    "/workspaces/{workspace_id}/scm/check-visibility",
    response_model=RepoVisibilityResponse,
)
async def check_workspace_scm_repo_visibility(
    workspace_id: str,
    request: CheckRepoVisibilityRequest,
    user: Any = Depends(get_current_user),
):
    await userspace_service.enforce_workspace_role(workspace_id, user.id, "editor")
    db = await get_db()
    workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
    stored_token = None
    encrypted = (
        getattr(workspace_record, "scmToken", None) if workspace_record else None
    )
    if encrypted:
        stored_token = decrypt_secret(encrypted)

    result = await git_check_visibility(
        url=request.git_url,
        stored_token=stored_token,
    )
    return RepoVisibilityResponse(
        visibility=result.visibility.value,
        has_stored_token=result.has_stored_token,
        needs_token=result.needs_token,
        message=result.message,
    )


@router.post(
    "/workspaces/{workspace_id}/scm/fetch-branches",
    response_model=FetchBranchesResponse,
)
async def fetch_workspace_scm_branches(
    workspace_id: str,
    request: FetchBranchesRequest,
    user: Any = Depends(get_current_user),
):
    await userspace_service.enforce_workspace_role(workspace_id, user.id, "editor")
    db = await get_db()
    workspace_record = await db.workspace.find_unique(where={"id": workspace_id})
    token = request.git_token
    if not token and workspace_record and getattr(workspace_record, "scmToken", None):
        token = decrypt_secret(getattr(workspace_record, "scmToken"))

    branches, error = await git_fetch_branches(
        url=request.git_url,
        token=token,
    )
    if error:
        needs_token = "private" in error.lower() or "token" in error.lower()
        return FetchBranchesResponse(
            branches=[],
            error=error,
            needs_token=needs_token,
        )

    return FetchBranchesResponse(
        branches=branches,
        error=None,
        needs_token=False,
    )


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    ws = await userspace_service.enforce_workspace_role(
        workspace_id, user.id, "owner", is_admin=is_admin
    )
    if is_admin and ws.owner_user_id != user.id:
        logger.info(
            "Admin '%s' deleted workspace '%s' (owner: %s)",
            user.username,
            workspace_id,
            ws.owner_user_id,
        )
    try:
        await userspace_runtime_service.stop_runtime_session(workspace_id, user.id)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
    await repository.delete_workspace_conversations(workspace_id)
    await userspace_service.delete_workspace(workspace_id, user.id, is_admin=is_admin)
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


@router.get(
    "/workspaces/{workspace_id}/env-vars",
    response_model=list[UserSpaceWorkspaceEnvVar],
)
async def list_workspace_env_vars(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.list_workspace_env_vars(workspace_id, user.id)


@router.put(
    "/workspaces/{workspace_id}/env-vars",
    response_model=UserSpaceWorkspaceEnvVar,
)
async def upsert_workspace_env_var(
    workspace_id: str,
    request: UpsertWorkspaceEnvVarRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.upsert_workspace_env_var(
        workspace_id,
        user.id,
        request,
    )
    await userspace_runtime_service.refresh_runtime_env_vars(workspace_id)
    return result


@router.delete(
    "/workspaces/{workspace_id}/env-vars/{env_key}",
    response_model=DeleteWorkspaceEnvVarResponse,
)
async def delete_workspace_env_var(
    workspace_id: str,
    env_key: str,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.delete_workspace_env_var(
        workspace_id,
        user.id,
        env_key,
    )
    await userspace_runtime_service.refresh_runtime_env_vars(workspace_id)
    return result


# ── Workspace Mounts ─────────────────────────────────────────────────


@router.get("/mount-sources", response_model=list[UserspaceMountSource])
async def list_userspace_mount_sources(_user: Any = Depends(require_admin)):
    return await userspace_service.list_userspace_mount_sources()


@router.post("/mount-sources", response_model=UserspaceMountSource)
async def create_userspace_mount_source(
    request: CreateUserspaceMountSourceRequest,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.create_userspace_mount_source(request)


@router.put("/mount-sources/{mount_source_id}", response_model=UserspaceMountSource)
async def update_userspace_mount_source(
    mount_source_id: str,
    request: UpdateUserspaceMountSourceRequest,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.update_userspace_mount_source(
        mount_source_id,
        request,
    )


@router.get(
    "/mount-sources/{mount_source_id}/affected-workspaces",
    response_model=MountSourceAffectedWorkspacesResponse,
)
async def get_mount_source_affected_workspaces(
    mount_source_id: str,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.get_mount_source_affected_workspaces(
        mount_source_id,
    )


@router.delete(
    "/mount-sources/{mount_source_id}",
    response_model=DeleteUserspaceMountSourceResponse,
)
async def delete_userspace_mount_source(
    mount_source_id: str,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.delete_userspace_mount_source(mount_source_id)


@router.post(
    "/mount-sources/{mount_source_id}/browse",
    response_model=WorkspaceMountBrowseResponse,
)
async def browse_userspace_mount_source(
    mount_source_id: str,
    request: BrowseUserspaceMountSourceRequest,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.browse_userspace_mount_source(
        mount_source_id,
        request,
    )


@router.post(
    "/tool-configs/{tool_config_id}/browse",
    response_model=WorkspaceMountBrowseResponse,
)
async def browse_tool_config(
    tool_config_id: str,
    request: BrowseUserspaceMountSourceRequest,
    _user: Any = Depends(require_admin),
):
    """Browse the filesystem of a tool config directly (before a mount source is saved)."""
    return await userspace_service.browse_tool_config(
        tool_config_id,
        request,
    )


@router.get(
    "/workspaces/{workspace_id}/mounts",
    response_model=list[WorkspaceMount],
)
async def list_workspace_mounts(
    workspace_id: str,
    response: Response,
    user: Any = Depends(get_current_user),
):
    response.headers[_USERSPACE_SURFACE_HEADER] = "mount-config"
    return await userspace_service.list_workspace_mounts(workspace_id, user.id)


@router.get(
    "/workspaces/{workspace_id}/mountable-sources",
    response_model=list[MountableSource],
)
async def list_mountable_sources(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.list_mountable_sources(workspace_id, user.id)


@router.post(
    "/workspaces/{workspace_id}/mounts/browse",
    response_model=WorkspaceMountBrowseResponse,
)
async def browse_workspace_mount_source(
    workspace_id: str,
    request: WorkspaceMountBrowseRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.browse_workspace_mount_source(
        workspace_id,
        user.id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/mounts",
    response_model=WorkspaceMount,
)
async def create_workspace_mount(
    workspace_id: str,
    request: CreateWorkspaceMountRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.create_workspace_mount(
        workspace_id, user.id, request
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id,
        "mount_create",
    )
    return result


@router.put(
    "/workspaces/{workspace_id}/mounts/{mount_id}",
    response_model=WorkspaceMount,
)
async def update_workspace_mount(
    workspace_id: str,
    mount_id: str,
    request: UpdateWorkspaceMountRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.update_workspace_mount(
        workspace_id,
        user.id,
        mount_id,
        request,
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id,
        "mount_update",
    )
    return result


@router.delete(
    "/workspaces/{workspace_id}/mounts/{mount_id}",
    response_model=DeleteWorkspaceMountResponse,
)
async def delete_workspace_mount(
    workspace_id: str,
    mount_id: str,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.delete_workspace_mount(
        workspace_id, user.id, mount_id
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id,
        "mount_delete",
    )
    return result


@router.post(
    "/workspaces/{workspace_id}/mounts/{mount_id}/sync-preview",
    response_model=WorkspaceMountSyncPreviewResponse,
)
async def preview_workspace_mount_sync(
    workspace_id: str,
    mount_id: str,
    request: WorkspaceMountSyncPreviewRequest | None = None,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.preview_workspace_mount_sync(
        workspace_id,
        user.id,
        mount_id,
        request,
    )


@router.post(
    "/workspaces/{workspace_id}/mounts/{mount_id}/sync",
    response_model=WorkspaceMountSyncResponse,
)
async def sync_workspace_mount(
    workspace_id: str,
    mount_id: str,
    request: WorkspaceMountSyncRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.sync_workspace_mount(
        workspace_id,
        user.id,
        mount_id,
        request,
    )
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id,
        "mount_sync",
    )
    return result


@router.get("/workspaces/{workspace_id}/files", response_model=list[UserSpaceFileInfo])
async def list_workspace_files(
    workspace_id: str,
    response: Response,
    include_dirs: bool = False,
    user: Any = Depends(get_current_user),
):
    response.headers[_USERSPACE_SURFACE_HEADER] = "workspace-tree"
    response.headers[_USERSPACE_TREE_MOUNTS_HEADER] = "true"
    result = await userspace_service.list_workspace_files(
        workspace_id,
        user.id,
        include_dirs=include_dirs,
    )
    return result


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
    result = await userspace_service.upsert_workspace_file(
        workspace_id, file_path, request, user.id
    )
    userspace_service.invalidate_file_list_cache(workspace_id)
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id, "file_upsert"
    )
    return result


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
    userspace_service.invalidate_file_list_cache(workspace_id)
    await userspace_runtime_service.bump_workspace_generation(
        workspace_id, "file_delete"
    )
    return {"success": True}


@router.get(
    "/workspaces/{workspace_id}/changed-file-state",
    response_model=UserSpaceChangedFileStateResponse,
)
async def get_workspace_changed_file_state(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    changed_file_paths = await userspace_service.list_workspace_changed_file_paths(
        workspace_id,
        user.id,
    )
    acknowledged_changed_file_paths = (
        await userspace_service.list_workspace_changed_file_acknowledgements(
            workspace_id,
            user.id,
        )
    )
    generation = await userspace_runtime_service.get_workspace_generation(workspace_id)
    return UserSpaceChangedFileStateResponse(
        workspace_id=workspace_id,
        generation=generation,
        changed_file_paths=changed_file_paths,
        acknowledged_changed_file_paths=acknowledged_changed_file_paths,
    )


@router.post(
    "/workspaces/{workspace_id}/changed-file-acknowledgements",
    response_model=UserSpaceChangedFileStateResponse,
)
async def acknowledge_workspace_changed_file_path(
    workspace_id: str,
    request: UserSpaceAcknowledgeChangedFilePathRequest,
    user: Any = Depends(get_current_user),
):
    acknowledged_changed_file_paths = (
        await userspace_service.acknowledge_workspace_changed_file_path(
            workspace_id,
            user.id,
            request.path,
        )
    )
    changed_file_paths = await userspace_service.list_workspace_changed_file_paths(
        workspace_id,
        user.id,
    )
    generation = await userspace_runtime_service.get_workspace_generation(workspace_id)
    return UserSpaceChangedFileStateResponse(
        workspace_id=workspace_id,
        generation=generation,
        changed_file_paths=changed_file_paths,
        acknowledged_changed_file_paths=acknowledged_changed_file_paths,
    )


@router.delete(
    "/workspaces/{workspace_id}/changed-file-acknowledgements",
    response_model=UserSpaceChangedFileStateResponse,
)
async def clear_workspace_changed_file_acknowledgements(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    acknowledged_changed_file_paths = (
        await userspace_service.clear_workspace_changed_file_acknowledgements(
            workspace_id,
            user.id,
        )
    )
    changed_file_paths = await userspace_service.list_workspace_changed_file_paths(
        workspace_id,
        user.id,
    )
    generation = await userspace_runtime_service.get_workspace_generation(workspace_id)
    return UserSpaceChangedFileStateResponse(
        workspace_id=workspace_id,
        generation=generation,
        changed_file_paths=changed_file_paths,
        acknowledged_changed_file_paths=acknowledged_changed_file_paths,
    )


@router.delete(
    "/workspaces/{workspace_id}/changed-file-acknowledgements/{file_path:path}",
    response_model=UserSpaceChangedFileStateResponse,
)
async def clear_workspace_changed_file_acknowledgement(
    workspace_id: str,
    file_path: str,
    user: Any = Depends(get_current_user),
):
    acknowledged_changed_file_paths = (
        await userspace_service.clear_workspace_changed_file_acknowledgements(
            workspace_id,
            user.id,
            relative_path=file_path,
        )
    )
    changed_file_paths = await userspace_service.list_workspace_changed_file_paths(
        workspace_id,
        user.id,
    )
    generation = await userspace_runtime_service.get_workspace_generation(workspace_id)
    return UserSpaceChangedFileStateResponse(
        workspace_id=workspace_id,
        generation=generation,
        changed_file_paths=changed_file_paths,
        acknowledged_changed_file_paths=acknowledged_changed_file_paths,
    )


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


@router.get(
    "/workspaces/{workspace_id}/snapshots/timeline",
    response_model=UserSpaceSnapshotTimelineResponse,
)
async def get_snapshot_timeline(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_snapshot_timeline(workspace_id, user.id)


@router.get(
    "/workspaces/{workspace_id}/snapshots/{snapshot_id}/diff-summary",
    response_model=UserSpaceSnapshotDiffSummaryResponse,
)
async def get_snapshot_diff_summary(
    workspace_id: str,
    snapshot_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_snapshot_diff_summary(
        workspace_id,
        snapshot_id,
        user.id,
    )


@router.get(
    "/workspaces/{workspace_id}/snapshots/{snapshot_id}/file-diff",
    response_model=UserSpaceSnapshotFileDiffResponse,
)
async def get_snapshot_file_diff(
    workspace_id: str,
    snapshot_id: str,
    file_path: str = Query(min_length=1),
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_snapshot_file_diff(
        workspace_id,
        snapshot_id,
        file_path,
        user.id,
    )


@router.post("/workspaces/{workspace_id}/snapshots", response_model=UserSpaceSnapshot)
async def create_snapshot(
    workspace_id: str,
    request: CreateSnapshotRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.create_snapshot(
        workspace_id, user.id, request.message
    )
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result


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
    await userspace_runtime_service.invalidate_workspace_runtime_state(workspace_id)
    timeline = await userspace_service.get_snapshot_timeline(workspace_id, user.id)
    return RestoreSnapshotResponse(
        restored_snapshot_id=snapshot.id,
        file_count=snapshot.file_count,
        current_branch_id=snapshot.branch_id,
        has_previous=timeline.has_previous,
        has_next=timeline.has_next,
    )


@router.patch(
    "/workspaces/{workspace_id}/snapshots/{snapshot_id}",
    response_model=UserSpaceSnapshot,
)
async def update_snapshot(
    workspace_id: str,
    snapshot_id: str,
    request: UpdateSnapshotRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.update_snapshot(
        workspace_id,
        snapshot_id,
        request,
        user.id,
    )
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result


@router.post(
    "/workspaces/{workspace_id}/snapshots/previous",
    response_model=RestoreSnapshotResponse,
)
async def restore_previous_snapshot(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    snapshot = await userspace_service.navigate_snapshot_previous(workspace_id, user.id)
    await userspace_runtime_service.invalidate_workspace_runtime_state(workspace_id)
    timeline = await userspace_service.get_snapshot_timeline(workspace_id, user.id)
    return RestoreSnapshotResponse(
        restored_snapshot_id=snapshot.id,
        file_count=snapshot.file_count,
        current_branch_id=snapshot.branch_id,
        has_previous=timeline.has_previous,
        has_next=timeline.has_next,
    )


@router.post(
    "/workspaces/{workspace_id}/snapshots/next",
    response_model=RestoreSnapshotResponse,
)
async def restore_next_snapshot(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    snapshot = await userspace_service.navigate_snapshot_next(workspace_id, user.id)
    await userspace_runtime_service.invalidate_workspace_runtime_state(workspace_id)
    timeline = await userspace_service.get_snapshot_timeline(workspace_id, user.id)
    return RestoreSnapshotResponse(
        restored_snapshot_id=snapshot.id,
        file_count=snapshot.file_count,
        current_branch_id=snapshot.branch_id,
        has_previous=timeline.has_previous,
        has_next=timeline.has_next,
    )


@router.post(
    "/workspaces/{workspace_id}/snapshots/switch-branch",
    response_model=UserSpaceSnapshotTimelineResponse,
)
async def switch_snapshot_branch(
    workspace_id: str,
    request: SwitchSnapshotBranchRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.switch_snapshot_branch(
        workspace_id, request, user.id
    )
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result
