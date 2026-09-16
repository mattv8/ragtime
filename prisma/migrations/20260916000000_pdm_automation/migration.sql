CREATE TABLE IF NOT EXISTS pdm_automation_state (
    tool_config_id TEXT PRIMARY KEY REFERENCES tool_configs(id) ON DELETE CASCADE,
    webhook_id TEXT UNIQUE,
    webhook_secret TEXT,
    webhook_paused BOOLEAN NOT NULL DEFAULT FALSE,
    webhook_created_at TIMESTAMP,
    pending_webhook BOOLEAN NOT NULL DEFAULT FALSE,
    pending_schedule BOOLEAN NOT NULL DEFAULT FALSE,
    first_pending_at TIMESTAMP,
    last_received_at TIMESTAMP,
    last_event_id TEXT,
    pending_generation INTEGER NOT NULL DEFAULT 0,
    claimed_generation INTEGER,
    active_job_id TEXT,
    last_attempt_at TIMESTAMP,
    last_success_at TIMESTAMP,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS pdm_automation_state_pending_idx
    ON pdm_automation_state (pending_webhook, pending_schedule);
