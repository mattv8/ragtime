-- =============================================================================
-- Add per-conversation tool options table
-- =============================================================================

CREATE TABLE IF NOT EXISTS "conversation_tool_options" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "conversation_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "options" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "conversation_tool_options_conversation_id_tool_config_id_key"
    ON "conversation_tool_options" ("conversation_id", "tool_config_id");

CREATE INDEX IF NOT EXISTS "conversation_tool_options_conversation_id_idx"
    ON "conversation_tool_options" ("conversation_id");

CREATE INDEX IF NOT EXISTS "conversation_tool_options_tool_config_id_idx"
    ON "conversation_tool_options" ("tool_config_id");

ALTER TABLE "conversation_tool_options"
    DROP CONSTRAINT IF EXISTS "conversation_tool_options_conversation_id_fkey",
    ADD CONSTRAINT "conversation_tool_options_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "conversation_tool_options"
    DROP CONSTRAINT IF EXISTS "conversation_tool_options_tool_config_id_fkey",
    ADD CONSTRAINT "conversation_tool_options_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs" ("id") ON DELETE CASCADE ON UPDATE CASCADE;
