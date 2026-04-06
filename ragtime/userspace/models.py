from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ragtime.indexer.models import WorkspaceChatStateResponse

WorkspaceRole = Literal["owner", "editor", "viewer"]
ArtifactType = Literal["module_ts"]
SqlitePersistenceMode = Literal["include", "exclude"]
ShareAccessMode = Literal[
    "token",
    "password",
    "authenticated_users",
    "selected_users",
    "ldap_groups",
]
WorkspaceScmProvider = Literal["github", "gitlab", "generic"]
WorkspaceScmDirection = Literal["import", "export"]
WorkspaceScmPreviewState = Literal[
    "missing_remote",
    "missing_branch",
    "up_to_date",
    "safe",
    "destructive",
]
RuntimeSessionState = Literal[
    "starting",
    "running",
    "stopping",
    "stopped",
    "error",
]
RuntimeOperationPhase = Literal[
    "queued",
    "provisioning",
    "bootstrapping",
    "deps_install",
    "launching",
    "probing",
    "ready",
    "failed",
    "stopped",
]
UserspaceMountSourceType = Literal["ssh", "filesystem"]
UserspaceMountBackend = Literal["ssh", "docker_volume", "smb", "nfs", "local"]
WorkspaceMountSyncMode = Literal[
    "merge",
    "source_authoritative",
    "target_authoritative",
]


class WorkspaceMember(BaseModel):
    user_id: str = Field(description="User ID")
    role: WorkspaceRole = Field(description="Permission role")


class UserSpaceAvailableTool(BaseModel):
    id: str
    name: str
    tool_type: str
    description: str | None = None
    group_id: str | None = None
    group_name: str | None = None


class UserSpaceWorkspace(BaseModel):
    id: str
    name: str
    description: str | None = None
    sqlite_persistence_mode: SqlitePersistenceMode = "exclude"
    owner_user_id: str
    owner_username: str | None = None
    owner_display_name: str | None = None
    selected_tool_ids: list[str] = Field(default_factory=list)
    selected_tool_group_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    members: list[WorkspaceMember] = Field(default_factory=list)
    scm: "UserSpaceWorkspaceScmStatus | None" = None
    created_at: datetime
    updated_at: datetime


class CreateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    sqlite_persistence_mode: SqlitePersistenceMode = "exclude"
    selected_tool_ids: list[str] | None = Field(
        default=None,
        description=(
            "Workspace-selected tool config IDs. When omitted, all enabled tools "
            "are selected by default."
        ),
    )
    selected_tool_group_ids: list[str] | None = Field(
        default=None,
        description="Workspace-selected tool group IDs.",
    )


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    sqlite_persistence_mode: SqlitePersistenceMode | None = None
    selected_tool_ids: list[str] | None = None
    selected_tool_group_ids: list[str] | None = None
    owner_user_id: str | None = Field(
        default=None, description="Transfer ownership to this user (admin only)"
    )


class UserSpaceWorkspaceScmStatus(BaseModel):
    connected: bool = False
    git_url: str | None = None
    git_branch: str | None = None
    provider: WorkspaceScmProvider | None = None
    repo_visibility: str | None = None
    has_stored_token: bool = False
    connected_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_sync_direction: WorkspaceScmDirection | None = None
    last_sync_status: str | None = None
    last_sync_message: str | None = None
    last_remote_commit_hash: str | None = None
    last_synced_snapshot_id: str | None = None


class UserSpaceWorkspaceScmConnectionRequest(BaseModel):
    git_url: str = Field(min_length=1, max_length=2000)
    git_branch: str = Field(default="main", min_length=1, max_length=255)
    git_token: str | None = Field(default=None, max_length=10000)
    repo_visibility: str | None = Field(default=None, max_length=32)


class UserSpaceWorkspaceScmConnectionResponse(BaseModel):
    workspace_id: str
    scm: UserSpaceWorkspaceScmStatus


class UserSpaceWorkspaceScmPreviewRequest(BaseModel):
    git_url: str | None = Field(default=None, min_length=1, max_length=2000)
    git_branch: str | None = Field(default=None, min_length=1, max_length=255)
    git_token: str | None = Field(default=None, max_length=10000)
    create_repo_if_missing: bool = False
    create_repo_private: bool = True
    create_repo_description: str | None = Field(default=None, max_length=2000)


class UserSpaceWorkspaceScmPreviewResponse(BaseModel):
    workspace_id: str
    direction: WorkspaceScmDirection
    state: WorkspaceScmPreviewState
    summary: str
    git_url: str
    git_branch: str
    provider: WorkspaceScmProvider
    repo_visibility: str | None = None
    local_changed: bool = False
    remote_changed: bool = False
    local_has_uncommitted_changes: bool = False
    will_overwrite_local: bool = False
    will_overwrite_remote: bool = False
    can_proceed_without_force: bool = False
    local_commit_hash: str | None = None
    remote_commit_hash: str | None = None
    current_snapshot_id: str | None = None
    changed_files_sample: list[str] = Field(default_factory=list)
    preview_token: str | None = None
    preview_expires_at: datetime | None = None


class UserSpaceWorkspaceScmImportRequest(UserSpaceWorkspaceScmPreviewRequest):
    overwrite_preview_token: str | None = Field(default=None, max_length=255)


class UserSpaceWorkspaceScmExportRequest(UserSpaceWorkspaceScmPreviewRequest):
    overwrite_preview_token: str | None = Field(default=None, max_length=255)


class UserSpaceWorkspaceScmSyncResponse(BaseModel):
    workspace_id: str
    direction: WorkspaceScmDirection
    state: str
    summary: str
    scm: UserSpaceWorkspaceScmStatus
    snapshot: "UserSpaceSnapshot | None" = None
    remote_commit_hash: str | None = None
    suggested_setup_prompt: str | None = None


class UserSpaceWorkspaceEnvVar(BaseModel):
    key: str = Field(min_length=1, max_length=128)
    has_value: bool = Field(
        default=True,
        description="True when an encrypted value exists for this key",
    )
    description: str | None = Field(default=None, max_length=1000)
    created_at: datetime
    updated_at: datetime


class UpsertWorkspaceEnvVarRequest(BaseModel):
    key: str = Field(
        min_length=1,
        max_length=128,
        description="Current environment variable key",
    )
    value: str | None = Field(
        default=None,
        max_length=10000,
        description=(
            "Replacement value. Required for create, optional for rename-only updates."
        ),
    )
    new_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional new key name for rename",
    )
    description: str | None = Field(default=None, max_length=1000)


class DeleteWorkspaceEnvVarResponse(BaseModel):
    success: bool = True
    key: str


class UpdateWorkspaceMembersRequest(BaseModel):
    members: list[WorkspaceMember] = Field(default_factory=list)


class UserSpaceFileInfo(BaseModel):
    path: str
    size_bytes: int
    updated_at: datetime
    entry_type: Literal["file", "directory"] = "file"


class UserSpaceAcknowledgeChangedFilePathRequest(BaseModel):
    path: str = Field(min_length=1, description="Workspace-relative file path")


class UserSpaceChangedFileStateResponse(BaseModel):
    workspace_id: str
    generation: int
    changed_file_paths: list[str] = Field(default_factory=list)
    acknowledged_changed_file_paths: list[str] = Field(default_factory=list)


class UpsertWorkspaceFileRequest(BaseModel):
    content: str = Field(default="", description="UTF-8 text content")
    artifact_type: ArtifactType | None = None
    live_data_requested: bool = Field(
        default=False,
        description=(
            "Set true only when the user explicitly requested live/refreshable data. "
            "When true for eligible module-source writes, live_data_connections are required."
        ),
    )
    live_data_connections: list["UserSpaceLiveDataConnection"] | None = None
    live_data_checks: list["UserSpaceLiveDataCheck"] | None = None


class UserSpaceLiveDataConnection(BaseModel):
    component_kind: Literal["tool_config"] = "tool_config"
    component_id: str = Field(min_length=1)
    request: dict[str, Any] | str
    component_name: str | None = None
    component_type: str | None = None
    refresh_interval_seconds: int | None = Field(default=None, ge=1)


class UserSpaceLiveDataCheck(BaseModel):
    component_id: str = Field(min_length=1)
    connection_check_passed: bool
    transformation_check_passed: bool
    input_row_count: int | None = Field(default=None, ge=0)
    output_row_count: int | None = Field(default=None, ge=0)
    note: str | None = None


class UserSpaceFileResponse(BaseModel):
    path: str
    content: str
    artifact_type: ArtifactType | None = None
    live_data_connections: list[UserSpaceLiveDataConnection] | None = None
    live_data_checks: list[UserSpaceLiveDataCheck] | None = None
    updated_at: datetime


class ExecuteComponentRequest(BaseModel):
    component_id: str = Field(min_length=1)
    request: dict[str, Any] | str


class ExecuteComponentResponse(BaseModel):
    component_id: str
    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    error: str | None = None


class UserSpaceWorkspaceShareLink(BaseModel):
    workspace_id: str
    share_token: str
    owner_username: str
    share_slug: str
    share_url: str


class UserSpaceWorkspaceShareLinkStatus(BaseModel):
    workspace_id: str
    has_share_link: bool
    owner_username: str
    share_slug: str | None = None
    share_token: str | None = None
    share_url: str | None = None
    created_at: datetime | None = None
    share_access_mode: ShareAccessMode = "token"
    selected_user_ids: list[str] = Field(default_factory=list)
    selected_ldap_groups: list[str] = Field(default_factory=list)
    has_password: bool = False


class UpdateWorkspaceShareAccessRequest(BaseModel):
    share_access_mode: ShareAccessMode
    password: str | None = Field(default=None, max_length=512)
    selected_user_ids: list[str] = Field(default_factory=list)
    selected_ldap_groups: list[str] = Field(default_factory=list)


class UpdateWorkspaceShareSlugRequest(BaseModel):
    slug: str = Field(min_length=1, max_length=120)


class WorkspaceShareSlugAvailabilityResponse(BaseModel):
    slug: str
    available: bool


class UserSpaceSharedPreviewResponse(BaseModel):
    workspace_id: str
    workspace_name: str
    entry_path: str
    workspace_files: dict[str, str]
    live_data_connections: list["UserSpaceLiveDataConnection"] | None = None


class UserSpaceSnapshot(BaseModel):
    id: str
    workspace_id: str
    branch_id: str
    branch_name: str
    parent_snapshot_id: str | None = None
    is_current: bool = False
    can_rename: bool = True
    git_commit_hash: str | None = None
    remote_commit_hash: str | None = (
        None  # When set, snapshot is backed by remote commit (authoritative)
    )
    message: str | None = None
    created_at: datetime
    file_count: int


class UserSpaceSnapshotDiffFileSummary(BaseModel):
    path: str
    status: Literal["A", "D", "M", "R"]
    old_path: str | None = None
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False


class UserSpaceSnapshotDiffSummaryResponse(BaseModel):
    workspace_id: str
    snapshot_id: str
    snapshot_commit_hash: str | None = None
    files: list[UserSpaceSnapshotDiffFileSummary] = Field(default_factory=list)
    is_snapshot_own_diff: bool = False


class UserSpaceSnapshotFileDiffResponse(BaseModel):
    workspace_id: str
    snapshot_id: str
    path: str
    status: Literal["A", "D", "M", "R"]
    old_path: str | None = None
    before_path: str | None = None
    after_path: str | None = None
    before_content: str = ""
    after_content: str = ""
    additions: int = 0
    deletions: int = 0
    is_binary: bool = False
    is_deleted_in_current: bool = False
    is_untracked_in_current: bool = False
    is_snapshot_own_diff: bool = False
    is_truncated: bool = False
    message: str | None = None


class UserSpaceSnapshotBranch(BaseModel):
    id: str
    workspace_id: str
    name: str
    git_ref_name: str
    base_snapshot_id: str | None = None
    branched_from_snapshot_id: str | None = None
    is_active: bool = False
    created_at: datetime


class UserSpaceSnapshotTimelineResponse(BaseModel):
    workspace_id: str
    current_snapshot_id: str | None = None
    current_branch_id: str | None = None
    has_previous: bool = False
    has_next: bool = False
    snapshots: list[UserSpaceSnapshot] = Field(default_factory=list)
    branches: list[UserSpaceSnapshotBranch] = Field(default_factory=list)


class CreateSnapshotRequest(BaseModel):
    message: str | None = Field(default=None, max_length=5000)


class UpdateSnapshotRequest(BaseModel):
    message: str = Field(min_length=1, max_length=200)


class SwitchSnapshotBranchRequest(BaseModel):
    branch_id: str = Field(min_length=1)


class RestoreSnapshotResponse(BaseModel):
    restored_snapshot_id: str
    file_count: int
    current_branch_id: str | None = None
    has_previous: bool = False
    has_next: bool = False


class UserSpaceRuntimeSession(BaseModel):
    id: str
    workspace_id: str
    leased_by_user_id: str
    state: RuntimeSessionState
    runtime_provider: str = "microvm_pool_v1"
    provider_session_id: str | None = None
    preview_internal_url: str | None = None
    launch_framework: str | None = None
    launch_command: str | None = None
    launch_cwd: str | None = None
    launch_port: int | None = None
    created_at: datetime
    updated_at: datetime
    last_heartbeat_at: datetime | None = None
    # Vestigial — never populated by the current provider. Kept for API compat.
    idle_expires_at: datetime | None = None
    ttl_expires_at: datetime | None = None
    last_error: str | None = None


class UserSpaceRuntimeSessionResponse(BaseModel):
    workspace_id: str
    session: UserSpaceRuntimeSession | None = None


class UserSpaceRuntimeStatusResponse(BaseModel):
    workspace_id: str
    session_state: RuntimeSessionState
    session_id: str | None = None
    devserver_running: bool = False
    devserver_port: int = 5173
    launch_framework: str | None = None
    launch_command: str | None = None
    launch_cwd: str | None = None
    runtime_capabilities: dict[str, Any] | None = None
    runtime_has_cap_sys_admin: bool | None = None
    preview_url: str | None = None
    last_error: str | None = None
    runtime_operation_id: str | None = None
    runtime_operation_phase: RuntimeOperationPhase | None = None
    runtime_operation_started_at: datetime | None = None
    runtime_operation_updated_at: datetime | None = None


class UserSpaceWorkspaceTabStateResponse(BaseModel):
    workspace_id: str
    runtime_status: UserSpaceRuntimeStatusResponse
    chat_state: WorkspaceChatStateResponse


class UserSpaceRuntimeActionResponse(BaseModel):
    workspace_id: str
    session_id: str
    state: RuntimeSessionState
    success: bool = True
    runtime_operation_id: str | None = None
    runtime_operation_phase: RuntimeOperationPhase | None = None
    runtime_operation_started_at: datetime | None = None
    runtime_operation_updated_at: datetime | None = None


class UserSpaceCapabilityTokenResponse(BaseModel):
    token: str
    expires_at: datetime
    workspace_id: str
    session_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class UserSpaceCollabSnapshotResponse(BaseModel):
    workspace_id: str
    file_path: str
    version: int
    content: str
    read_only: bool


class PaginatedWorkspacesResponse(BaseModel):
    items: list[UserSpaceWorkspace]
    total: int
    offset: int
    limit: int


# -------------------------------------------------------------------------
# Workspace Mount Models
# -------------------------------------------------------------------------

MountSyncStatus = Literal["pending", "synced", "error"]


class UserspaceMountSource(BaseModel):
    id: str
    name: str
    description: str | None = None
    enabled: bool = True
    source_type: UserspaceMountSourceType
    mount_backend: UserspaceMountBackend
    tool_config_id: str | None = None
    tool_name: str | None = None
    connection_config: dict[str, Any] = Field(default_factory=dict)
    approved_paths: list[str] = Field(default_factory=list)
    sync_interval_seconds: int | None = Field(
        default=30,
        description="Polling interval in seconds for auto-sync (1 to 2592000)",
    )
    usage_count: int = 0
    created_at: datetime
    updated_at: datetime


class CreateUserspaceMountSourceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    enabled: bool = True
    tool_config_id: str | None = Field(
        default=None,
        description="Backing tool config; when set, source_type and connection_config are derived from the tool",
    )
    source_type: UserspaceMountSourceType | None = Field(
        default=None, description="Required only when tool_config_id is not provided"
    )
    connection_config: dict[str, Any] = Field(default_factory=dict)
    approved_paths: list[str] = Field(default_factory=list)
    sync_interval_seconds: int | None = Field(
        default=30,
        ge=1,
        le=2592000,
        description="Polling interval in seconds for auto-sync (1s to ~1 month)",
    )


class UpdateUserspaceMountSourceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    enabled: bool | None = None
    connection_config: dict[str, Any] | None = None
    approved_paths: list[str] | None = None
    sync_interval_seconds: int | None = Field(
        default=None,
        ge=1,
        le=2592000,
        description="Polling interval in seconds for auto-sync (1s to ~1 month)",
    )


class BrowseUserspaceMountSourceRequest(BaseModel):
    path: str = Field(
        default="/", description="Synthetic absolute path relative to the source root"
    )


class DeleteUserspaceMountSourceResponse(BaseModel):
    success: bool
    mount_source_id: str


class MountSourceAffectedWorkspace(BaseModel):
    workspace_id: str
    workspace_name: str
    owner_user_id: str
    mount_count: int


class MountSourceAffectedWorkspacesResponse(BaseModel):
    mount_source_id: str
    mount_source_name: str
    source_type: UserspaceMountSourceType
    total_mounts: int
    workspaces: list[MountSourceAffectedWorkspace]


class WorkspaceMount(BaseModel):
    id: str
    workspace_id: str
    mount_source_id: str
    source_path: str
    target_path: str
    description: str | None = None
    enabled: bool = True
    sync_mode: WorkspaceMountSyncMode = "merge"
    sync_status: MountSyncStatus = "pending"
    sync_backend: str | None = None
    sync_notice: str | None = None
    last_sync_at: datetime | None = None
    last_sync_error: str | None = None
    auto_sync_enabled: bool = False
    source_name: str | None = None
    source_type: UserspaceMountSourceType | None = None
    mount_backend: UserspaceMountBackend | None = None
    source_available: bool = True
    created_at: datetime
    updated_at: datetime


class MountableSource(BaseModel):
    mount_source_id: str
    source_name: str
    source_type: UserspaceMountSourceType
    mount_backend: UserspaceMountBackend
    source_path: str


class WorkspaceMountBrowseRequest(BaseModel):
    mount_source_id: str = Field(min_length=1)
    root_source_path: str = Field(
        min_length=1,
        description="Admin-approved source relpath that defines the browse root",
    )
    path: str = Field(
        default="/",
        description="Synthetic absolute path relative to the tool root",
    )


class WorkspaceMountDirectoryEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class WorkspaceMountBrowseResponse(BaseModel):
    path: str
    entries: list[WorkspaceMountDirectoryEntry] = Field(default_factory=list)
    error: str | None = None


class CreateWorkspaceMountRequest(BaseModel):
    mount_source_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1, description="Admin-approved source relpath")
    target_path: str = Field(
        min_length=1,
        max_length=200,
        description="User-chosen sandbox target path (e.g. /mnt/data)",
    )
    source_directory_to_create: str | None = Field(
        default=None,
        min_length=1,
        description="Optional source directory to create before saving the mount",
    )
    target_directory_to_create: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional workspace target directory to create before saving the mount",
    )
    auto_sync_enabled: bool = Field(
        default=False,
        description="Enable background watch mode that periodically auto-syncs this mount",
    )
    sync_mode: WorkspaceMountSyncMode = Field(
        default="merge",
        description=(
            "Sync policy for SSH-backed mounts: merge keeps both sides, "
            "source_authoritative makes the remote source authoritative, and "
            "target_authoritative makes the workspace target authoritative."
        ),
    )
    description: str | None = Field(default=None, max_length=1000)


class UpdateWorkspaceMountRequest(BaseModel):
    target_path: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    enabled: bool | None = Field(default=None)
    auto_sync_enabled: bool | None = Field(default=None)
    sync_mode: WorkspaceMountSyncMode | None = Field(
        default=None,
        description="Updated sync policy for SSH-backed mounts.",
    )
    destructive_auto_sync_preview_token: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Fresh destructive preview token required when enabling auto-sync "
            "or switching an auto-sync mount into a destructive mode."
        ),
    )


class WorkspaceMountSyncPreviewRequest(BaseModel):
    sync_mode: WorkspaceMountSyncMode | None = Field(
        default=None,
        description="Optional sync mode override for previewing a pending auto-sync configuration change.",
    )


class WorkspaceMountSyncRequest(BaseModel):
    preview_token: str | None = Field(
        default=None,
        min_length=1,
        description="Required for destructive sync modes after a successful preview.",
    )


class WorkspaceMountSyncPreviewResponse(BaseModel):
    mount_id: str
    sync_mode: WorkspaceMountSyncMode
    sync_backend: str | None = None
    sync_notice: str | None = None
    requires_confirmation: bool = False
    preview_token: str
    preview_expires_at: datetime
    delete_from_source_count: int = 0
    delete_from_target_count: int = 0
    delete_from_source_paths: list[str] = Field(default_factory=list)
    delete_from_target_paths: list[str] = Field(default_factory=list)
    sample_limit: int = 0
    last_sync_error: str | None = None


class DeleteWorkspaceMountResponse(BaseModel):
    success: bool = True
    mount_id: str


class WorkspaceMountSyncResponse(BaseModel):
    mount_id: str
    sync_mode: WorkspaceMountSyncMode
    sync_status: MountSyncStatus
    files_synced: int = 0
    sync_backend: str | None = None
    sync_notice: str | None = None
    last_sync_error: str | None = None
