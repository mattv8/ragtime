from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from runtime.shared import RuntimeSessionState


@dataclass
class ManagerSession:
    provider_session_id: str
    workspace_id: str
    leased_by_user_id: str
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


class StartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    leased_by_user_id: str = Field(description="User ID leasing the runtime")
    provider_session_id: str | None = Field(
        default=None,
        description="Optional provider session id for idempotent resume",
    )


class _BaseSessionFields(BaseModel):
    """Shared fields across manager and worker session responses."""

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


class RuntimeSessionResponse(_BaseSessionFields):
    provider_session_id: str = Field(description="Runtime provider session identifier")


class WorkerStartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    provider_session_id: str = Field(description="Manager provider session identifier")
    pty_access_token: str = Field(description="Manager-issued PTY access token")


class WorkerSessionResponse(_BaseSessionFields):
    worker_session_id: str = Field(description="Worker-local session id")


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
        default=1800,
        description="Extra post-load settle wait before capture (helps absorb HMR reloads)",
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


class ManagerSessionSummary(BaseModel):
    provider_session_id: str = Field(description="Provider session ID")
    workspace_id: str = Field(description="Workspace ID")
    state: str = Field(description="Session state")
    devserver_running: bool = Field(description="Whether devserver is running")


class RuntimeManagerHealthResponse(BaseModel):
    status: str = Field(description="Service status")
    workers_total: int = Field(description="Configured worker count")
    workers_leased: int = Field(description="Currently leased worker count")
    active_sessions: int = Field(description="Active runtime session count")
    max_sessions: int = Field(description="Maximum concurrent sessions")
    sessions: list[ManagerSessionSummary] = Field(
        default_factory=list, description="Summary of active sessions"
    )


class WorkerHealthResponse(BaseModel):
    status: str = Field(description="Worker status")
    service_mode: str = Field(description="Runtime service mode")
    active_sessions: int = Field(description="Active worker sessions")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Worker metadata"
    )
