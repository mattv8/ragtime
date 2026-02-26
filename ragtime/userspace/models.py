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


class WorkspaceMember(BaseModel):
    user_id: str = Field(description="User ID")
    role: WorkspaceRole = Field(description="Permission role")


class UserSpaceAvailableTool(BaseModel):
    id: str
    name: str
    tool_type: str
    description: str | None = None


class UserSpaceWorkspace(BaseModel):
    id: str
    name: str
    description: str | None = None
    sqlite_persistence_mode: SqlitePersistenceMode = "include"
    owner_user_id: str
    selected_tool_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    members: list[WorkspaceMember] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CreateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    sqlite_persistence_mode: SqlitePersistenceMode = "include"
    selected_tool_ids: list[str] = Field(default_factory=list)


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    sqlite_persistence_mode: SqlitePersistenceMode | None = None
    selected_tool_ids: list[str] | None = None


class UpdateWorkspaceMembersRequest(BaseModel):
    members: list[WorkspaceMember] = Field(default_factory=list)


class UserSpaceFileInfo(BaseModel):
    path: str
    size_bytes: int
    updated_at: datetime


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
    message: str | None = None
    created_at: datetime
    file_count: int


class CreateSnapshotRequest(BaseModel):
    message: str | None = Field(default=None, max_length=5000)


class RestoreSnapshotResponse(BaseModel):
    restored_snapshot_id: str
    file_count: int


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
    preview_url: str | None = None
    last_error: str | None = None


class UserSpaceRuntimeActionResponse(BaseModel):
    workspace_id: str
    session_id: str
    state: RuntimeSessionState
    success: bool = True


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
