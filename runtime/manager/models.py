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
    runtime_capabilities: dict[str, Any] | None
    state: RuntimeSessionState
    devserver_running: bool
    last_error: str | None
    runtime_operation_id: str | None
    runtime_operation_phase: str | None
    runtime_operation_started_at: datetime | None
    runtime_operation_updated_at: datetime | None
    updated_at: datetime
    lease_expires_at: datetime


class StartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    leased_by_user_id: str = Field(description="User ID leasing the runtime")
    provider_session_id: str | None = Field(
        default=None,
        description="Optional provider session id for idempotent resume",
    )
    workspace_env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Workspace environment variables resolved by control plane. "
            "Values are injected server-side into devserver runtime only."
        ),
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
    runtime_capabilities: dict[str, Any] | None = Field(
        default=None,
        description="Runtime worker capability metadata",
    )
    devserver_running: bool = Field(description="Whether devserver is running")
    last_error: str | None = Field(default=None, description="Last runtime error")
    runtime_operation_id: str | None = Field(
        default=None,
        description="Current async runtime operation id",
    )
    runtime_operation_phase: str | None = Field(
        default=None,
        description="Current async runtime operation phase",
    )
    runtime_operation_started_at: datetime | None = Field(
        default=None,
        description="Timestamp when current runtime operation started",
    )
    runtime_operation_updated_at: datetime | None = Field(
        default=None,
        description="Timestamp when current runtime operation phase was last updated",
    )
    updated_at: datetime = Field(description="Session update time")


class RuntimeSessionResponse(_BaseSessionFields):
    provider_session_id: str = Field(description="Runtime provider session identifier")


class WorkerStartSessionRequest(BaseModel):
    workspace_id: str = Field(description="Workspace ID")
    provider_session_id: str = Field(description="Manager provider session identifier")
    pty_access_token: str = Field(description="Manager-issued PTY access token")
    workspace_env: dict[str, str] = Field(
        default_factory=dict,
        description="Workspace env map for devserver injection",
    )


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


class RuntimeContentProbeRequest(BaseModel):
    path: str = Field(default="", description="Workspace-relative preview path")
    timeout_ms: int = Field(default=15000, description="Navigation timeout in ms")
    wait_after_load_ms: int = Field(
        default=2000,
        description="Post-load settle wait for JS rendering",
    )


class RuntimeContentProbeResponse(BaseModel):
    ok: bool = Field(description="Whether the content probe succeeded")
    workspace_id: str = Field(description="Workspace ID")
    preview_path: str = Field(description="Workspace-relative preview path")
    status_code: int | None = Field(
        default=None, description="HTTP status code from the page"
    )
    body_text_length: int = Field(
        default=0, description="Length of visible text in the rendered page"
    )
    body_text_preview: str = Field(
        default="", description="First 200 chars of visible text"
    )
    body_html_length: int = Field(
        default=0, description="Length of innerHTML in the rendered page"
    )
    title: str = Field(default="", description="Document title after render")
    has_error_indicator: bool = Field(
        default=False,
        description="Whether common error strings were found in page text",
    )
    console_errors: list[str] = Field(
        default_factory=list,
        description="First few console errors captured during page load",
    )


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
    capture_element: bool = Field(
        default=False,
        description=(
            "Capture the unique visible element matched by wait_for_selector. "
            "Fails when selector is missing or ambiguous."
        ),
    )
    clip_padding_px: int = Field(
        default=16,
        ge=0,
        le=256,
        description="Optional padding around element clip in pixels",
    )
    wait_after_load_ms: int = Field(
        default=1800,
        description="Extra post-load settle wait before capture (helps absorb HMR reloads)",
    )
    refresh_before_capture: bool = Field(
        default=True,
        description="Refresh the page once before screenshot capture",
    )


class RuntimeScreenshotResponse(BaseModel):
    ok: bool = Field(description="Whether screenshot capture succeeded")
    workspace_id: str = Field(description="Workspace ID")
    preview_path: str = Field(description="Workspace-relative preview path")
    screenshot_path: str = Field(description="Absolute screenshot file path")
    screenshot_size_bytes: int = Field(description="Screenshot file size in bytes")
    render: dict[str, Any] = Field(description="Capture/render settings metadata")
    probe: dict[str, Any] = Field(
        description=(
            "Captured browser probe metadata including optional element clipping "
            "diagnostics"
        )
    )


class RuntimeRestartRequest(BaseModel):
    workspace_env: dict[str, str] | None = Field(
        default=None,
        description="If provided, update workspace environment variables before restart",
    )


class RuntimeExecRequest(BaseModel):
    command: str = Field(description="Shell command to execute")
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=120,
        description="Maximum execution time in seconds",
    )
    cwd: str | None = Field(
        default=None,
        description="Optional workspace-relative working directory",
    )


class RuntimeExecResponse(BaseModel):
    exit_code: int = Field(description="Process exit code")
    stdout: str = Field(description="Standard output (truncated if large)")
    stderr: str = Field(description="Standard error (truncated if large)")
    timed_out: bool = Field(default=False, description="Whether command timed out")
    truncated: bool = Field(
        default=False,
        description="Whether output was truncated due to size limits",
    )


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
