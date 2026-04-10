-- Add low-risk indexes and vacuum tuning for userspace/runtime-heavy paths.

CREATE INDEX IF NOT EXISTS "chat_tasks_status_last_update_at_idx"
ON "chat_tasks"("status", "last_update_at");

CREATE INDEX IF NOT EXISTS "chat_tasks_conversation_id_status_created_at_idx"
ON "chat_tasks"("conversation_id", "status", "created_at" DESC);

CREATE INDEX IF NOT EXISTS "workspace_mounts_auto_sync_enabled_enabled_idx"
ON "workspace_mounts"("auto_sync_enabled", "enabled");

CREATE INDEX IF NOT EXISTS "workspace_mounts_sync_status_idx"
ON "workspace_mounts"("sync_status");

ALTER TABLE "chat_tasks"
SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE "workspace_mounts"
SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_analyze_scale_factor = 0.02
);