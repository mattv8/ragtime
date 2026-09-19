"""Request and response models for protected workspace SQLite history."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SqliteHistoryCaptureRequest(BaseModel):
    database_name: str = Field(min_length=1, max_length=255)
    request_id: str | None = Field(default=None, min_length=8, max_length=128)


class SqliteHistoryCaptureJobRequest(BaseModel):
    database_name: str = Field(min_length=1, max_length=255)
    request_id: str | None = Field(default=None, min_length=8, max_length=128)


class SqliteHistoryPreviewRequest(BaseModel):
    mode: Literal["merge", "overwrite"]
    conflict_policy: Literal["keep_current", "use_backup"] = "keep_current"
    table_policies: dict[str, Literal["keep_current", "use_backup"]] | None = None


class SqliteHistoryRestoreRequest(BaseModel):
    preview_id: str = Field(min_length=1, max_length=128)


class SqliteHistoryRecoveryRequest(BaseModel):
    action: Literal["complete", "abort"]


class SqliteHistoryBackup(BaseModel):
    id: str
    workspace_id: str
    database_name: str
    created_at: str
    trigger: Literal["manual", "snapshot", "scheduled", "pre_restore"]
    snapshot_id: str | None = None
    snapshot_git_commit_hash: str | None = None
    status: Literal["ready", "failed"]
    size_bytes: int = 0
    sha256: str | None = None
    error: str | None = None
    can_restore: bool = False
    can_delete: bool = False
    capture_job_id: str | None = None


SqliteHistoryCaptureJobStatus = Literal["pending", "running", "completed", "failed", "cancelled", "interrupted"]


class SqliteHistoryCaptureJob(BaseModel):
    """Safe public representation of a durable SQLite capture request."""

    id: str
    workspace_id: str
    trigger: Literal["manual", "snapshot", "scheduled"]
    database_names: list[str] = Field(default_factory=list)
    snapshot_id: str | None = None
    snapshot_git_commit_hash: str | None = None
    status: SqliteHistoryCaptureJobStatus
    created_at: datetime
    available_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime
    completed_databases: int = 0
    total_databases: int = 0
    backup_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None
    cancel_requested: bool = False


class SqliteHistoryCaptureJobResponse(BaseModel):
    job: SqliteHistoryCaptureJob


class SqliteHistoryCaptureJobListResponse(BaseModel):
    jobs: list[SqliteHistoryCaptureJob]


class SqliteHistoryInterruptedMaintenance(BaseModel):
    """Maintenance status; absent legacy state is interpreted as interrupted."""

    state: Literal["active", "interrupted", "invalid", "release_pending"] = Field(
        default="interrupted",
        description="Active work, recoverable interruption, invalid fence, or terminal receipt awaiting release.",
    )
    operation_id: str | None = None
    detail: str
    can_complete: bool = False
    can_abort: bool = False


class SqliteHistoryListResponse(BaseModel):
    workspace_id: str
    backups: list[SqliteHistoryBackup]
    can_manage: bool
    interrupted_maintenance: SqliteHistoryInterruptedMaintenance | None = None


class SqliteHistoryCaptureResponse(BaseModel):
    backup: SqliteHistoryBackup


class SqliteHistoryPreview(BaseModel):
    preview_id: str | None
    backup_id: str
    database_name: str
    mode: Literal["merge", "overwrite"]
    conflict_policy: Literal["keep_current", "use_backup"]
    tables: list[dict[str, Any]] = Field(default_factory=list)
    migrations_applied: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    can_apply: bool
    expires_at: str | None = None


class SqliteHistoryRestoreResponse(BaseModel):
    operation_id: str
    restored_backup_id: str
    safety_backup_id: str | None = None
    runtime_stopped: bool = True
    status: Literal["completed"] = "completed"
