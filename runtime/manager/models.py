from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

RuntimeSessionState = Literal["starting", "running", "stopping", "stopped", "error"]


@dataclass
class ManagerSession:
    provider_session_id: str
    workspace_id: str
    leased_by_user_id: str
    worker_id: str
    worker_base_url: str
    worker_session_id: str
    pty_access_token: str
    preview_internal_url: str
    launch_framework: str | None
    launch_command: str | None
    launch_cwd: str | None
    launch_port: int | None
    state: RuntimeSessionState
    devserver_running: bool
    last_error: str | None
    updated_at: datetime
    lease_expires_at: datetime


@dataclass
class PoolSlot:
    slot_id: str
    vm_id: str
    lease_provider_session_id: str | None
    workspace_id: str | None
    state: Literal["warm", "leased"]
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime


class StartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    leased_by_user_id: str = Field(description="User ID leasing the runtime")
    provider_session_id: str | None = Field(
        default=None,
        description="Optional provider session id for idempotent resume",
    )


class RuntimeSessionResponse(BaseModel):
    provider_session_id: str = Field(description="Runtime provider session identifier")
    workspace_id: str = Field(description="Workspace ID")
    state: RuntimeSessionState = Field(description="Runtime state")
    preview_internal_url: str = Field(description="Internal preview URL")
    launch_framework: str | None = Field(
        default=None,
        description="Detected runtime framework",
    )
    launch_command: str | None = Field(
        default=None,
        description="Launch command for the runtime devserver",
    )
    launch_cwd: str | None = Field(
        default=None,
        description="Workspace-relative directory used to launch runtime",
    )
    launch_port: int | None = Field(
        default=None,
        description="Worker-local devserver port",
    )
    devserver_running: bool = Field(description="Whether devserver is running")
    last_error: str | None = Field(default=None, description="Last runtime error")
    updated_at: datetime = Field(description="Session update time")


class WorkerStartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    provider_session_id: str = Field(description="Manager provider session identifier")
    pty_access_token: str = Field(description="Manager-issued PTY access token")


class WorkerSessionResponse(BaseModel):
    worker_session_id: str = Field(description="Worker-local session id")
    workspace_id: str = Field(description="Workspace ID")
    state: RuntimeSessionState = Field(description="Worker runtime state")
    preview_internal_url: str = Field(description="Worker preview internal URL")
    launch_framework: str | None = Field(
        default=None,
        description="Detected runtime framework",
    )
    launch_command: str | None = Field(
        default=None,
        description="Launch command for runtime devserver",
    )
    launch_cwd: str | None = Field(
        default=None,
        description="Workspace-relative launch cwd",
    )
    launch_port: int | None = Field(
        default=None,
        description="Worker-local devserver port",
    )
    devserver_running: bool = Field(description="Whether worker devserver is running")
    last_error: str | None = Field(
        default=None, description="Last worker runtime error"
    )
    updated_at: datetime = Field(description="Worker session update time")


class RuntimePtyUrlResponse(BaseModel):
    ws_url: str = Field(description="Websocket URL for PTY stream")


class RuntimeFileReadResponse(BaseModel):
    path: str = Field(description="Workspace-relative path")
    content: str = Field(description="File content")
    exists: bool = Field(description="Whether the file exists")
    updated_at: datetime = Field(description="Updated timestamp")


class RuntimeFileWriteRequest(BaseModel):
    content: str = Field(description="File content to persist")


class RuntimeScreenshotRequest(BaseModel):
    path: str = Field(default="", description="Workspace-relative preview path")
    width: int = Field(default=1440, description="Requested viewport width")
    height: int = Field(default=900, description="Requested viewport height")
    full_page: bool = Field(default=True, description="Capture full page screenshot")
    timeout_ms: int = Field(default=25000, description="Navigation timeout in ms")
    wait_for_selector: str = Field(
        default="body",
        description="Optional selector to wait for before capture",
    )
    wait_after_load_ms: int = Field(
        default=900,
        description="Extra wait after page load before capture",
    )
    refresh_before_capture: bool = Field(
        default=True,
        description="Refresh the page once before screenshot capture",
    )
    filename: str | None = Field(
        default=None,
        description="Optional filename for saved screenshot",
    )


class RuntimeScreenshotResponse(BaseModel):
    ok: bool = Field(description="Whether screenshot capture succeeded")
    workspace_id: str = Field(description="Workspace ID")
    preview_path: str = Field(description="Workspace-relative preview path")
    screenshot_path: str = Field(description="Absolute screenshot file path")
    screenshot_size_bytes: int = Field(description="Screenshot file size in bytes")
    render: dict[str, Any] = Field(description="Capture/render settings metadata")
    probe: dict[str, Any] = Field(description="Captured browser probe metadata")


class RuntimeManagerHealthResponse(BaseModel):
    status: str = Field(description="Service status")
    workers_total: int = Field(description="Configured worker count")
    workers_leased: int = Field(description="Currently leased worker count")
    active_sessions: int = Field(description="Active runtime session count")


class WorkerHealthResponse(BaseModel):
    status: str = Field(description="Worker status")
    service_mode: str = Field(description="Runtime service mode")
    active_sessions: int = Field(description="Active worker sessions")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Worker metadata"
    )
