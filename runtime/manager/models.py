from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RuntimeSessionState = Literal["starting", "running", "stopping", "stopped", "error"]


@dataclass
class ManagerSession:
    provider_session_id: str
    workspace_id: str
    leased_by_user_id: str
    preview_internal_url: str
    state: RuntimeSessionState
    devserver_running: bool
    last_error: str | None
    updated_at: datetime
    lease_slot_id: str | None
    lease_vm_id: str | None
    lease_started_at: datetime
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
    devserver_running: bool = Field(description="Whether devserver is running")
    last_error: str | None = Field(default=None, description="Last runtime error")
    updated_at: datetime = Field(description="Session update time")
