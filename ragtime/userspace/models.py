from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

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
RuntimeSessionState = Literal[
    "starting",
    "running",
    "stopping",
    "stopped",
    "error",
]
RuntimeOperationPhase = Literal[
    "queued",
    "bootstrapping",
    "deps_install",
    "launching",
    "probing",
    "ready",
    "failed",
    "stopped",
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
    selected_tool_ids: list[str] = Field(default_factory=list)
    selected_tool_group_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    members: list[WorkspaceMember] = Field(default_factory=list)
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
    message: str | None = None
    created_at: datetime
    file_count: int


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
