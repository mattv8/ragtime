-- External agent reliability persistence. Additive/idempotent for pgvector deployments.
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "userspace_build_model" TEXT;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "openrouter_credit_monitor_enabled" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "openrouter_low_credit_threshold_usd" DOUBLE PRECISION NOT NULL DEFAULT 5;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "openrouter_management_api_key" TEXT;

ALTER TABLE "workspaces" ADD COLUMN IF NOT EXISTS "bridge_credential_mode" TEXT NOT NULL DEFAULT 'env';
ALTER TABLE "workspace_agent_access" ADD COLUMN IF NOT EXISTS "allow_runtime_restart" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "chat_tasks" ADD COLUMN IF NOT EXISTS "execution_policy" JSONB;
ALTER TABLE "chat_tasks" ADD COLUMN IF NOT EXISTS "termination_reason" TEXT;
ALTER TABLE "chat_tasks" ADD COLUMN IF NOT EXISTS "outcome_summary" JSONB;

CREATE TABLE IF NOT EXISTS "workspace_runtime_operations" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "request_hash" TEXT NOT NULL,
    "state" TEXT NOT NULL DEFAULT 'accepted',
    "provider_session_id" TEXT,
    "operation_id" TEXT,
    "error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_runtime_operations_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_runtime_operations_workspace_id_user_id_idempotency_key_key"
    ON "workspace_runtime_operations"("workspace_id", "user_id", "idempotency_key");
CREATE INDEX IF NOT EXISTS "workspace_runtime_operations_workspace_id_created_at_idx"
    ON "workspace_runtime_operations"("workspace_id", "created_at");
