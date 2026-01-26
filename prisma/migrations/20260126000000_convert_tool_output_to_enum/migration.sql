-- Convert app_settings.suppress_tool_output (boolean) to tool_output_mode (enum)
-- This migration handles the transition from the original boolean setting to the enum-based setting

-- Add the new enum column to app_settings (reuses ConversationToolOutputMode from previous migration)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "tool_output_mode" "ConversationToolOutputMode" NOT NULL DEFAULT 'default';

-- Migrate data: convert boolean to enum
-- false (show output) -> 'default' (or 'show')
-- true (suppress output) -> 'hide'
UPDATE "app_settings"
SET "tool_output_mode" = CASE
    WHEN "suppress_tool_output" = true THEN 'hide'::"ConversationToolOutputMode"
    ELSE 'default'::"ConversationToolOutputMode"
END
WHERE "tool_output_mode" = 'default';

-- Drop the old boolean column
ALTER TABLE "app_settings"
DROP COLUMN IF EXISTS "suppress_tool_output";

-- Update comment
COMMENT ON COLUMN "app_settings"."tool_output_mode" IS 'Global setting: tool output mode (default=show, show, hide, auto). Users can override per-conversation.';
