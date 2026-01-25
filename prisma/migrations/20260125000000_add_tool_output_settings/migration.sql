-- Add tool call output settings to app_settings and conversations
-- These settings control whether tool call output is sent over the API

-- Global setting: suppress tool call output in API responses (default: false - show tool output)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "suppress_tool_output" BOOLEAN NOT NULL DEFAULT false;

-- Per-conversation setting: tool output mode
-- 'default' = use global setting
-- 'show' = always show tool output for this conversation
-- 'hide' = always hide tool output for this conversation
-- 'auto' = let AI decide based on relevance to query
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'ConversationToolOutputMode') THEN
        CREATE TYPE "ConversationToolOutputMode" AS ENUM ('default', 'show', 'hide', 'auto');
    END IF;
END $$;

ALTER TABLE "conversations"
ADD COLUMN IF NOT EXISTS "tool_output_mode" "ConversationToolOutputMode" NOT NULL DEFAULT 'default';

-- Add comments for documentation
COMMENT ON COLUMN "app_settings"."suppress_tool_output" IS 'Global setting: if true, suppress tool call output in API responses (not MCP). Users can override per-conversation.';
COMMENT ON COLUMN "conversations"."tool_output_mode" IS 'Per-conversation tool output mode: default (use global), show (always show), hide (always hide), auto (AI decides)';
