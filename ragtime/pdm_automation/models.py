from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PdmWebhookConfigResponse(BaseModel):
    enabled: bool
    paused: bool
    webhook_id: str | None = None
    webhook_url: str | None = None
    created_at: datetime | None = None
    last_received_at: datetime | None = None
    pending: bool
    active_job_id: str | None = None
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None


class PdmWebhookEnableResponse(PdmWebhookConfigResponse):
    secret: str | None = None


class PdmWebhookPayload(BaseModel):
    event_id: str | None = Field(default=None, max_length=200)
    reason: str | None = Field(default=None, max_length=500)
