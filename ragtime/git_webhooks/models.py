from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GitWebhookTargetType(str, Enum):
    GIT_INDEX = "git_index"
    WORKSPACE_SCM = "workspace_scm"


def format_git_webhook_target_key(target_type: GitWebhookTargetType, target_id: str) -> str:
    return f"{target_type.value}:{target_id}"


class GitWebhookDeliveryStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    IGNORED = "ignored"


class GitPushEvent(BaseModel):
    kind: str = Field(default="push", description="Normalized webhook event kind.")
    message: str | None = Field(default=None, description="Operator-facing explanation for ignored events.")
    provider_delivery_id: str | None = Field(default=None, description="Provider delivery/event identifier when supplied.")
    event_name: str = Field(description="Webhook event name from the SCM provider.")
    branch: str | None = Field(default=None, description="Branch associated with the push event.")
    head_commit: str | None = Field(default=None, description="Head commit SHA for the push event.")


class GitWebhookTarget(BaseModel):
    target_type: GitWebhookTargetType = Field(description="Resolved webhook target type.")
    target_id: str = Field(description="Internal index/workspace identifier.")
    key: str | None = Field(default=None, description="Stable per-target dispatcher key.")
    webhook_id: str = Field(description="Opaque public webhook identifier.")
    secret: str = Field(description="Decrypted webhook signing secret.")
    provider: str = Field(default="generic", description="Resolved webhook provider.")
    branch: str | None = Field(default=None, description="Configured branch for this webhook target.")
    paused: bool = Field(default=False, description="Whether webhook processing is temporarily paused for this target.")
    created_at: datetime | None = Field(default=None, description="When the webhook configuration was created.")
    name: str | None = Field(default=None, description="Index name when the target is a git index.")
    description: str | None = Field(default=None, description="Stored index description for git targets.")
    source: str | None = Field(default=None, description="Stored git URL or SCM source for this target.")
    git_token: str | None = Field(default=None, description="Current decrypted git token snapshot for git targets.")
    config_snapshot: dict[str, Any] | None = Field(default=None, description="Stored git index configuration snapshot.")


class GitWebhookDelivery(BaseModel):
    id: str = Field(description="Webhook delivery row identifier.")
    target_type: GitWebhookTargetType = Field(description="Target type for this delivery.")
    target_id: str = Field(description="Internal index/workspace identifier for this delivery.")
    provider_delivery_id: str | None = Field(default=None, description="Provider delivery identifier when supplied.")
    event_name: str = Field(description="Webhook event name.")
    branch: str | None = Field(default=None, description="Branch associated with the delivery.")
    head_commit: str | None = Field(default=None, description="Head commit SHA associated with the delivery.")
    status: GitWebhookDeliveryStatus = Field(description="Persisted delivery lifecycle status.")
    index_job_id: str | None = Field(default=None, description="Linked index job when one was started.")
    message: str | None = Field(default=None, description="Operator-facing status or error message.")
    received_at: datetime = Field(description="When the webhook delivery was received.")
    started_at: datetime | None = Field(default=None, description="When processing started.")
    completed_at: datetime | None = Field(default=None, description="When processing completed.")


class GitWebhookDeliveryResponse(GitWebhookDelivery):
    pass


class GitWebhookConfigResponse(BaseModel):
    enabled: bool = Field(description="Whether webhook delivery is enabled for this target.")
    paused: bool = Field(default=False, description="Whether webhook delivery is temporarily paused for this target.")
    webhook_id: str | None = Field(default=None, description="Opaque public webhook identifier.")
    webhook_url: str | None = Field(default=None, description="Full Ragtime webhook URL for manual provider registration.")
    provider: str | None = Field(default=None, description="Resolved SCM provider for this webhook target.")
    branch: str | None = Field(default=None, description="Configured branch for this webhook target.")
    created_at: datetime | None = Field(default=None, description="When the current webhook configuration was created.")


class GitWebhookEnableResponse(GitWebhookConfigResponse):
    secret: str | None = Field(default=None, description="Plaintext webhook secret on first enable or rotation only.")
