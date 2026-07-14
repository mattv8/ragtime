-- =============================================================================
-- Add per-workspace tool options table
-- =============================================================================

CREATE TABLE IF NOT EXISTS "workspace_tool_options" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "options" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_tool_options_workspace_id_tool_config_id_key"
    ON "workspace_tool_options" ("workspace_id", "tool_config_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_options_workspace_id_idx"
    ON "workspace_tool_options" ("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_options_tool_config_id_idx"
    ON "workspace_tool_options" ("tool_config_id");

ALTER TABLE "workspace_tool_options"
    DROP CONSTRAINT IF EXISTS "workspace_tool_options_workspace_id_fkey",
    ADD CONSTRAINT "workspace_tool_options_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "workspace_tool_options"
    DROP CONSTRAINT IF EXISTS "workspace_tool_options_tool_config_id_fkey",
    ADD CONSTRAINT "workspace_tool_options_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs" ("id") ON DELETE CASCADE ON UPDATE CASCADE;
