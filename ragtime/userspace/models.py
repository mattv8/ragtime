from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

WorkspaceRole = Literal["owner", "editor", "viewer"]
ArtifactType = Literal["module_ts"]


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
    owner_user_id: str
    selected_tool_ids: list[str] = Field(default_factory=list)
    conversation_ids: list[str] = Field(default_factory=list)
    members: list[WorkspaceMember] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    selected_tool_ids: list[str] = Field(default_factory=list)


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
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
    live_data_connections: list["UserSpaceLiveDataConnection"] | None = None


class UserSpaceLiveDataConnection(BaseModel):
    component_kind: Literal["tool_config"] = "tool_config"
    component_id: str = Field(min_length=1)
    request: dict[str, Any] | str
    component_name: str | None = None
    component_type: str | None = None
    refresh_interval_seconds: int | None = Field(default=None, ge=1)


class UserSpaceFileResponse(BaseModel):
    path: str
    content: str
    artifact_type: ArtifactType | None = None
    live_data_connections: list[UserSpaceLiveDataConnection] | None = None
    updated_at: datetime


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


class PaginatedWorkspacesResponse(BaseModel):
    items: list[UserSpaceWorkspace]
    total: int
    offset: int
    limit: int
