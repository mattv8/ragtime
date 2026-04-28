from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse

from ragtime.core.auth import get_browser_matched_origin
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
    CreateSnapshotBranchRequest,
    CreateSnapshotRequest,
    CreateUserspaceMountSourceRequest,
    CreateUserSpaceObjectStorageBucketRequest,
    CreateWorkspaceMountRequest,
    CreateWorkspaceRequest,
    CreateWorkspaceShareLinkRequest,
    DeleteGlobalEnvVarResponse,
    DeleteUserspaceMountSourceResponse,
    DeleteUserSpaceObjectStorageBucketResponse,
    DeleteUserSpaceWorkspaceArchiveExportResponse,
    DeleteWorkspaceEnvVarResponse,
    DeleteWorkspaceMountResponse,
    DuplicateWorkspaceRequest,
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    MountableSource,
    MountSourceAffectedWorkspacesResponse,
    PaginatedWorkspacesResponse,
    PromoteBranchToMainRequest,
    RestoreSnapshotResponse,
    SqliteImportResponse,
    SwitchSnapshotBranchRequest,
    UpdateSnapshotRequest,
    UpdateUserspaceMountSourceRequest,
    UpdateUserSpaceObjectStorageBucketRequest,
    UpdateWorkspaceMembersRequest,
    UpdateWorkspaceMountRequest,
    UpdateWorkspaceRequest,
    UpdateWorkspaceShareAccessRequest,
    UpdateWorkspaceShareLinkRequest,
    UpdateWorkspaceShareSlugRequest,
    UpsertGlobalEnvVarRequest,
    UpsertWorkspaceAgentGrantRequest,
    UpsertWorkspaceEnvVarRequest,
    UpsertWorkspaceFileRequest,
    UserSpaceAcknowledgeChangedFilePathRequest,
    UserSpaceAvailableTool,
    UserSpaceChangedFileStateResponse,
    UserSpaceCollabPresenceResponse,
    UserSpaceCollabPresenceUser,
    UserSpaceFileInfo,
    UserSpaceFileResponse,
    UserspaceMountSource,
    UserSpaceObjectStorageConfig,
    UserSpaceRuntimeRestartBatchTask,
    UserSpaceSharedPreviewResponse,
    UserSpaceSnapshot,
    UserSpaceSnapshotDiffSummaryResponse,
    UserSpaceSnapshotFileDiffResponse,
    UserSpaceSnapshotTimelineResponse,
    UserSpaceWorkspace,
    UserSpaceWorkspaceArchiveExportListResponse,
    UserSpaceWorkspaceArchiveExportRequest,
    UserSpaceWorkspaceArchiveExportTask,
    UserSpaceWorkspaceArchiveImportTask,
    UserSpaceWorkspaceCreateTask,
    UserSpaceWorkspaceDeleteTask,
    UserSpaceWorkspaceDuplicateTask,
    UserSpaceWorkspaceEnvVar,
    UserSpaceWorkspaceScmConnectionRequest,
    UserSpaceWorkspaceScmConnectionResponse,
    UserSpaceWorkspaceScmExportRequest,
    UserSpaceWorkspaceScmImportRequest,
    UserSpaceWorkspaceScmPreviewRequest,
    UserSpaceWorkspaceScmPreviewResponse,
    UserSpaceWorkspaceScmSettingsRequest,
    UserSpaceWorkspaceScmSyncResponse,
    UserSpaceWorkspaceShareLink,
    UserSpaceWorkspaceShareLinkListResponse,
    UserSpaceWorkspaceShareLinkStatus,
    WorkspaceAgentGrant,
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
from ragtime.userspace.share_auth import share_auth_token_from_request

logger = get_logger(__name__)

router = APIRouter(prefix="/indexes/userspace", tags=["User Space"])

_USERSPACE_SURFACE_HEADER = "X-Ragtime-Userspace-Surface"
_USERSPACE_TREE_MOUNTS_HEADER = "X-Ragtime-Userspace-Tree-Includes-Mounts"


def _subdomain_share_disabled_reason(
    *,
    has_share_link: bool,
) -> str | None:
    if not has_share_link:
        return "Create a share link first"
    return None


def _apply_share_link_variants(
    payload: Any,
    *,
    base_url: str,
    access_mode: str | None = None,
    has_share_link: bool | None = None,
) -> Any:
    share_token = str(getattr(payload, "share_token", "") or "").strip() or None
    share_url = str(getattr(payload, "share_url", "") or "").strip() or None
    workspace_id = str(getattr(payload, "workspace_id", "") or "").strip()
    resolved_has_password = bool(getattr(payload, "has_password", False))
    resolved_access_mode = (
        str(
            access_mode or getattr(payload, "share_access_mode", "token") or "token"
        ).strip()
        or "token"
    )
    resolved_has_share_link = (
        bool(getattr(payload, "has_share_link", True))
        if has_share_link is None
        else has_share_link
    ) and bool(share_token)
    subdomain_share_enabled = resolved_has_share_link
    subdomain_share_url = None
    if subdomain_share_enabled and workspace_id:
        subdomain_share_url = (
            userspace_runtime_service.get_preview_origin(
                workspace_id,
                control_plane_origin=base_url,
            ).rstrip("/")
            + "/"
        )
    anonymous_share_url = None
    if share_token:
        anonymous_share_url = userspace_service.build_workspace_anonymous_share_url(
            share_token,
            base_url=base_url,
        )
    return payload.model_copy(
        update={
            "share_url": share_url,
            "anonymous_share_url": anonymous_share_url,
            "subdomain_share_url": subdomain_share_url,
            "subdomain_share_enabled": subdomain_share_enabled,
            "subdomain_share_disabled_reason": _subdomain_share_disabled_reason(
                has_share_link=resolved_has_share_link,
            ),
        }
    )


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


@router.post(
    "/workspaces/create-task",
    response_model=UserSpaceWorkspaceCreateTask,
    status_code=202,
)
async def queue_workspace_create_task(
    request: CreateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    request.selected_tool_ids = await _normalize_selected_tool_ids(
        request.selected_tool_ids
    )
    return await userspace_service.enqueue_workspace_create_task(request, user.id)


@router.get(
    "/workspace-create-tasks/{task_id}",
    response_model=UserSpaceWorkspaceCreateTask,
)
async def get_workspace_create_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.get_workspace_create_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


@router.post(
    "/workspaces/{workspace_id}/duplicate-task",
    response_model=UserSpaceWorkspaceDuplicateTask,
    status_code=202,
)
async def queue_workspace_duplicate_task(
    workspace_id: str,
    request: DuplicateWorkspaceRequest,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.enqueue_workspace_duplicate_task(
        workspace_id,
        request,
        user.id,
    )


@router.get(
    "/workspace-duplicate-tasks/{task_id}",
    response_model=UserSpaceWorkspaceDuplicateTask,
)
async def get_workspace_duplicate_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.get_workspace_duplicate_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


@router.post(
    "/workspaces/{workspace_id}/archive/export-task",
    response_model=UserSpaceWorkspaceArchiveExportTask,
    status_code=202,
)
async def queue_workspace_archive_export_task(
    workspace_id: str,
    request: UserSpaceWorkspaceArchiveExportRequest,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.enqueue_workspace_archive_export_task(
        workspace_id,
        request,
        user.id,
        is_admin=is_admin,
    )


@router.get(
    "/workspace-archive-export-tasks/{task_id}",
    response_model=UserSpaceWorkspaceArchiveExportTask,
)
async def get_workspace_archive_export_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.get_workspace_archive_export_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


@router.get("/workspace-archive-export-tasks/{task_id}/download")
async def download_workspace_archive_export_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    archive_path, archive_file_name = (
        await userspace_service.get_workspace_archive_export_download(
            task_id,
            user.id,
            is_admin=is_admin,
        )
    )
    return FileResponse(path=archive_path, filename=archive_file_name)


@router.get(
    "/workspaces/{workspace_id}/archive-exports",
    response_model=UserSpaceWorkspaceArchiveExportListResponse,
)
async def list_workspace_archive_exports(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.list_workspace_archive_exports(
        workspace_id,
        user.id,
        is_admin=is_admin,
    )


@router.delete(
    "/workspace-archive-export-tasks/{task_id}",
    response_model=DeleteUserSpaceWorkspaceArchiveExportResponse,
)
async def delete_workspace_archive_export_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.delete_workspace_archive_export_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


@router.post(
    "/workspaces/{workspace_id}/archive/import-task",
    response_model=UserSpaceWorkspaceArchiveImportTask,
    status_code=202,
)
async def queue_workspace_archive_import_task(
    workspace_id: str,
    archive_file: UploadFile = File(...),
    include_snapshots: bool = Form(default=False),
    include_chat_history: bool = Form(default=False),
    user: Any = Depends(get_current_user),
):
    suffix = ".zip"
    normalized_file_name = str(archive_file.filename or "workspace-export.zip")
    lowered_file_name = normalized_file_name.lower()
    if lowered_file_name.endswith(".tar.gz") or lowered_file_name.endswith(".tgz"):
        suffix = ".tar.gz"
    temp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:

        def _copy_upload_sync() -> None:
            archive_file.file.seek(0)
            with temp_path.open("wb") as output_handle:
                shutil.copyfileobj(archive_file.file, output_handle, 1024 * 1024)

        await asyncio.to_thread(_copy_upload_sync)
    finally:
        await archive_file.close()

    try:
        is_admin = user.role == "admin"
        return await userspace_service.enqueue_workspace_archive_import_task(
            workspace_id,
            user.id,
            temp_path,
            normalized_file_name,
            include_snapshots=include_snapshots,
            include_chat_history=include_chat_history,
            is_admin=is_admin,
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


@router.get(
    "/workspace-archive-import-tasks/{task_id}",
    response_model=UserSpaceWorkspaceArchiveImportTask,
)
async def get_workspace_archive_import_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.get_workspace_archive_import_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


@router.get("/workspaces/{workspace_id}", response_model=UserSpaceWorkspace)
async def get_workspace(workspace_id: str, user: Any = Depends(get_current_user)):
    return await userspace_service.get_workspace(workspace_id, user.id)


@router.get(
    "/workspaces/{workspace_id}/collab/presence",
    response_model=UserSpaceCollabPresenceResponse,
)
async def get_workspace_collab_presence(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    # Enforce access to the workspace (any viewer role is sufficient).
    await userspace_service.get_workspace(workspace_id, user.id)

    presence_entries = await userspace_runtime_service.get_workspace_collab_presence(
        workspace_id
    )

    user_ids = [str(entry.get("user_id") or "") for entry in presence_entries]
    user_ids = [uid for uid in user_ids if uid]

    user_lookup: dict[str, Any] = {}
    if user_ids:
        db = await get_db()
        try:
            rows = await db.user.find_many(where={"id": {"in": user_ids}})
        except Exception:
            rows = []
        for row in rows:
            user_lookup[str(getattr(row, "id", "") or "")] = row

    users: list[UserSpaceCollabPresenceUser] = []
    for entry in presence_entries:
        uid = str(entry.get("user_id") or "")
        if not uid:
            continue
        row = user_lookup.get(uid)
        username = str(getattr(row, "username", "") or "") if row else None
        display_name = str(getattr(row, "displayName", "") or "") if row else None
        users.append(
            UserSpaceCollabPresenceUser(
                user_id=uid,
                username=username or None,
                display_name=display_name or None,
                updated_at=str(entry.get("updated_at") or "") or None,
            )
        )

    return UserSpaceCollabPresenceResponse(workspace_id=workspace_id, users=users)


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


@router.patch(
    "/workspaces/{workspace_id}/scm/settings",
    response_model=UserSpaceWorkspaceScmConnectionResponse,
)
async def update_workspace_scm_settings(
    workspace_id: str,
    request: UserSpaceWorkspaceScmSettingsRequest,
    user: Any = Depends(get_current_user),
):
    scm = await userspace_service.update_workspace_scm_settings(
        workspace_id,
        user.id,
        request,
    )
    return UserSpaceWorkspaceScmConnectionResponse(
        workspace_id=workspace_id,
        scm=scm,
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


@router.post(
    "/workspaces/{workspace_id}/delete-task",
    response_model=UserSpaceWorkspaceDeleteTask,
    status_code=202,
)
async def queue_workspace_delete_task(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.enqueue_workspace_delete_task(
        workspace_id,
        user.id,
        is_admin=is_admin,
    )


@router.get(
    "/workspace-delete-tasks/{task_id}",
    response_model=UserSpaceWorkspaceDeleteTask,
)
async def get_workspace_delete_task(
    task_id: str,
    user: Any = Depends(get_current_user),
):
    is_admin = user.role == "admin"
    return await userspace_service.get_workspace_delete_task(
        task_id,
        user.id,
        is_admin=is_admin,
    )


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
    "/workspaces/{workspace_id}/agent-grants",
    response_model=list[WorkspaceAgentGrant],
)
async def list_workspace_agent_grants(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    """List active cross-workspace agent grants originating from this workspace."""
    return await userspace_service.list_workspace_agent_grants(
        workspace_id,
        user.id,
        is_admin=user.role == "admin",
    )


@router.put(
    "/workspaces/{workspace_id}/agent-grants",
    response_model=WorkspaceAgentGrant,
)
async def upsert_workspace_agent_grant(
    workspace_id: str,
    request: UpsertWorkspaceAgentGrantRequest,
    user: Any = Depends(get_current_user),
):
    """Grant the agent in this (source) workspace access to a target workspace."""
    return await userspace_service.upsert_workspace_agent_grant(
        workspace_id,
        request,
        user.id,
        is_admin=user.role == "admin",
    )


@router.delete(
    "/workspaces/{workspace_id}/agent-grants/{target_workspace_id}",
    response_model=dict,
)
async def revoke_workspace_agent_grant(
    workspace_id: str,
    target_workspace_id: str,
    user: Any = Depends(get_current_user),
):
    revoked = await userspace_service.revoke_workspace_agent_grant(
        workspace_id,
        target_workspace_id,
        user.id,
        is_admin=user.role == "admin",
    )
    return {
        "source_workspace_id": workspace_id,
        "target_workspace_id": target_workspace_id,
        "revoked": revoked,
    }


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


@router.get(
    "/admin/env-vars",
    response_model=list[UserSpaceWorkspaceEnvVar],
)
async def list_global_env_vars(
    _user: Any = Depends(require_admin),
):
    return await userspace_service.list_global_env_vars()


@router.put(
    "/admin/env-vars",
    response_model=UserSpaceWorkspaceEnvVar,
)
async def upsert_global_env_var(
    request: UpsertGlobalEnvVarRequest,
    user: Any = Depends(require_admin),
):
    return await userspace_service.upsert_global_env_var(user.id, request)


@router.delete(
    "/admin/env-vars/{env_key}",
    response_model=DeleteGlobalEnvVarResponse,
)
async def delete_global_env_var(
    env_key: str,
    user: Any = Depends(require_admin),
):
    return await userspace_service.delete_global_env_var(user.id, env_key)


@router.get(
    "/admin/runtime-restart-task",
    response_model=UserSpaceRuntimeRestartBatchTask,
)
async def get_latest_runtime_restart_task(
    _user: Any = Depends(require_admin),
):
    return await userspace_service.get_latest_runtime_restart_batch_task()


@router.post(
    "/admin/runtime-restart-task",
    response_model=UserSpaceRuntimeRestartBatchTask,
    status_code=202,
)
async def enqueue_runtime_restart_task(
    user: Any = Depends(require_admin),
):
    return await userspace_service.enqueue_runtime_restart_batch_task(user.id)


@router.get(
    "/admin/runtime-restart-task/{task_id}",
    response_model=UserSpaceRuntimeRestartBatchTask,
)
async def get_runtime_restart_task(
    task_id: str,
    _user: Any = Depends(require_admin),
):
    return await userspace_service.get_runtime_restart_batch_task(task_id)


@router.get(
    "/workspaces/{workspace_id}/object-storage",
    response_model=UserSpaceObjectStorageConfig,
)
async def get_workspace_object_storage_config(
    workspace_id: str,
    user: Any = Depends(get_current_user),
):
    return await userspace_service.get_workspace_object_storage_config(
        workspace_id,
        user.id,
    )


@router.post(
    "/workspaces/{workspace_id}/object-storage/buckets",
    response_model=UserSpaceObjectStorageConfig,
)
async def create_workspace_object_storage_bucket(
    workspace_id: str,
    request: CreateUserSpaceObjectStorageBucketRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.create_workspace_object_storage_bucket(
        workspace_id,
        user.id,
        request,
    )
    await userspace_runtime_service.refresh_runtime_env_vars(workspace_id)
    return result


@router.put(
    "/workspaces/{workspace_id}/object-storage/buckets/{bucket_name}",
    response_model=UserSpaceObjectStorageConfig,
)
async def update_workspace_object_storage_bucket(
    workspace_id: str,
    bucket_name: str,
    request: UpdateUserSpaceObjectStorageBucketRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.update_workspace_object_storage_bucket(
        workspace_id,
        user.id,
        bucket_name,
        request,
    )
    await userspace_runtime_service.refresh_runtime_env_vars(workspace_id)
    return result


@router.delete(
    "/workspaces/{workspace_id}/object-storage/buckets/{bucket_name}",
    response_model=DeleteUserSpaceObjectStorageBucketResponse,
)
async def delete_workspace_object_storage_bucket(
    workspace_id: str,
    bucket_name: str,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.delete_workspace_object_storage_bucket(
        workspace_id,
        user.id,
        bucket_name,
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
    "/workspaces/{workspace_id}/exec",
)
async def exec_workspace_command(
    workspace_id: str,
    request: Request,
    user: Any = Depends(get_current_user),
):
    body = await request.json()
    command = (body.get("command") or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="command is required")
    timeout_seconds = min(int(body.get("timeout_seconds", 30)), 120)
    cwd = (body.get("cwd") or "").strip() or None
    result = await userspace_runtime_service.exec_workspace_command(
        workspace_id=workspace_id,
        user_id=user.id,
        command=command,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
    )
    return result


@router.get(
    "/workspaces/{workspace_id}/share-links",
    response_model=UserSpaceWorkspaceShareLinkListResponse,
)
async def list_workspace_share_links(
    workspace_id: str,
    request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = get_browser_matched_origin(request)
    response = await userspace_service.list_workspace_share_links(
        workspace_id,
        user.id,
        base_url=base_url,
    )
    return response.model_copy(
        update={
            "links": [
                _apply_share_link_variants(link, base_url=base_url)
                for link in response.links
            ],
        }
    )


@router.post(
    "/workspaces/{workspace_id}/share-links",
    response_model=UserSpaceWorkspaceShareLink,
)
async def create_workspace_share_link(
    workspace_id: str,
    request: Request,
    body: CreateWorkspaceShareLinkRequest = Body(
        default_factory=CreateWorkspaceShareLinkRequest
    ),
    user: Any = Depends(get_current_user),
):
    base_url = get_browser_matched_origin(request)
    link = await userspace_service.create_workspace_share_link(
        workspace_id,
        user.id,
        base_url=base_url,
        label=body.label,
    )
    return _apply_share_link_variants(
        link,
        base_url=base_url,
        access_mode="token",
        has_share_link=True,
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
    base_url = get_browser_matched_origin(request)
    status = await userspace_service.get_workspace_share_link_status(
        workspace_id,
        user.id,
        base_url=base_url,
    )
    return _apply_share_link_variants(status, base_url=base_url)


@router.delete(
    "/workspaces/{workspace_id}/share-links/{share_id}",
    status_code=204,
)
async def delete_workspace_share_link(
    workspace_id: str,
    share_id: str,
    user: Any = Depends(get_current_user),
):
    await userspace_service.delete_workspace_share_link(
        workspace_id,
        share_id,
        user.id,
    )
    return Response(status_code=204)


@router.put(
    "/workspaces/{workspace_id}/share-links/{share_id}",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def update_workspace_share_link_label(
    workspace_id: str,
    share_id: str,
    body: UpdateWorkspaceShareLinkRequest,
    base_request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = get_browser_matched_origin(base_request)
    status = await userspace_service.update_workspace_share_link_label(
        workspace_id,
        share_id,
        body.label,
        user.id,
        base_url=base_url,
    )
    return _apply_share_link_variants(status, base_url=base_url)


@router.put(
    "/workspaces/{workspace_id}/share-links/{share_id}/access",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def update_workspace_share_access(
    workspace_id: str,
    share_id: str,
    request: UpdateWorkspaceShareAccessRequest,
    base_request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = get_browser_matched_origin(base_request)
    status = await userspace_service.update_workspace_share_access(
        workspace_id,
        share_id,
        request,
        user.id,
        base_url=base_url,
    )
    return _apply_share_link_variants(status, base_url=base_url)


@router.put(
    "/workspaces/{workspace_id}/share-links/{share_id}/slug",
    response_model=UserSpaceWorkspaceShareLinkStatus,
)
async def update_workspace_share_slug(
    workspace_id: str,
    share_id: str,
    request: UpdateWorkspaceShareSlugRequest,
    base_request: Request,
    user: Any = Depends(get_current_user),
):
    base_url = get_browser_matched_origin(base_request)
    status = await userspace_service.update_workspace_share_slug(
        workspace_id,
        share_id,
        request.slug,
        user.id,
        base_url=base_url,
    )
    return _apply_share_link_variants(status, base_url=base_url)


@router.get(
    "/workspaces/{workspace_id}/share-links/availability",
    response_model=WorkspaceShareSlugAvailabilityResponse,
)
async def check_workspace_share_slug_availability(
    workspace_id: str,
    slug: str = Query(min_length=1, max_length=120),
    share_id: str | None = Query(default=None),
    user: Any = Depends(get_current_user),
):
    return await userspace_service.check_workspace_share_slug_availability(
        workspace_id,
        share_id,
        slug,
        user.id,
    )


@router.get(
    "/shared/{share_token}",
    response_model=UserSpaceSharedPreviewResponse,
)
async def get_shared_preview(
    share_token: str,
    request: Request,
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.get_shared_preview(
        share_token,
        current_user=user,
        share_auth_token=share_auth_token_from_request(
            request.headers,
            request.cookies,
            share_token=share_token,
        ),
    )


@router.get(
    "/shared/{owner_username}/{share_slug}",
    response_model=UserSpaceSharedPreviewResponse,
)
async def get_shared_preview_by_slug(
    owner_username: str,
    share_slug: str,
    request: Request,
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.get_shared_preview_by_slug(
        owner_username,
        share_slug,
        current_user=user,
        share_auth_token=share_auth_token_from_request(
            request.headers,
            request.cookies,
            owner_username=owner_username,
            share_slug=share_slug,
        ),
    )


@router.post(
    "/shared/{share_token}/execute-component",
    response_model=ExecuteComponentResponse,
)
async def execute_shared_component(
    share_token: str,
    request: ExecuteComponentRequest,
    http_request: Request,
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.execute_shared_component(
        share_token,
        request,
        current_user=user,
        share_auth_token=share_auth_token_from_request(
            http_request.headers,
            http_request.cookies,
            share_token=share_token,
        ),
    )


@router.post(
    "/shared/{owner_username}/{share_slug}/execute-component",
    response_model=ExecuteComponentResponse,
)
async def execute_shared_component_by_slug(
    owner_username: str,
    share_slug: str,
    request: ExecuteComponentRequest,
    http_request: Request,
    user: Any | None = Depends(get_current_user_optional),
):
    return await userspace_service.execute_shared_component_by_slug(
        owner_username,
        share_slug,
        request,
        current_user=user,
        share_auth_token=share_auth_token_from_request(
            http_request.headers,
            http_request.cookies,
            owner_username=owner_username,
            share_slug=share_slug,
        ),
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

    # Per-message snapshot link: anchor the new snapshot to the latest message
    # in the active conversation when one was supplied. "Latest only" semantics
    # are enforced by upsert on (conversation_id, message_id).
    if request.conversation_id:
        try:
            conv = await repository.get_conversation(request.conversation_id)
            if conv and (conv.workspace_id or "") == workspace_id and conv.messages:
                last_msg = conv.messages[-1]
                if last_msg.message_id:
                    # User-side anchor includes the message; assistant-side anchor
                    # excludes the assistant reply ("before assistant reply").
                    keep_count = (
                        len(conv.messages)
                        if last_msg.role == "user"
                        else len(conv.messages) - 1
                    )
                    if keep_count >= 0:
                        await repository.upsert_message_snapshot_link(
                            conversation_id=conv.id,
                            workspace_id=workspace_id,
                            message_id=last_msg.message_id,
                            snapshot_id=result.id,
                            restore_message_count=keep_count,
                        )
        except Exception:
            # Link creation is best-effort; never block snapshot creation on it.
            pass

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


@router.post(
    "/workspaces/{workspace_id}/snapshots/create-branch",
    response_model=UserSpaceSnapshotTimelineResponse,
)
async def create_snapshot_branch(
    workspace_id: str,
    request: CreateSnapshotBranchRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.create_snapshot_branch(
        workspace_id, user.id, name=request.name
    )
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result


@router.post(
    "/workspaces/{workspace_id}/snapshots/promote-branch",
    response_model=UserSpaceSnapshotTimelineResponse,
)
async def promote_branch_to_main(
    workspace_id: str,
    request: PromoteBranchToMainRequest,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.promote_branch_to_main(
        workspace_id, request.branch_id, user.id
    )
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result


@router.delete(
    "/workspaces/{workspace_id}/snapshots/{snapshot_id}",
    response_model=UserSpaceSnapshotTimelineResponse,
)
async def delete_snapshot(
    workspace_id: str,
    snapshot_id: str,
    user: Any = Depends(get_current_user),
):
    result = await userspace_service.delete_snapshot(workspace_id, snapshot_id, user.id)
    await userspace_runtime_service.bump_workspace_generation(workspace_id, "snapshot")
    return result


@router.post(
    "/workspaces/{workspace_id}/sqlite-import",
    response_model=SqliteImportResponse,
)
async def import_sql_to_workspace_sqlite(
    workspace_id: str,
    file: UploadFile = File(
        ...,
        description="SQL dump file (.sql, .dump, .pg, .pgsql)",
    ),
    user: Any = Depends(get_current_user),
):
    filename = file.filename or "upload.sql"
    allowed_extensions = {".sql", ".dump", ".pg", ".pgsql", ".mysql"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Accepted: {', '.join(sorted(allowed_extensions))}"
            ),
        )
    file_bytes = await file.read()
    return await userspace_service.import_sql_to_workspace_sqlite(
        workspace_id, user.id, file_bytes, filename
    )
