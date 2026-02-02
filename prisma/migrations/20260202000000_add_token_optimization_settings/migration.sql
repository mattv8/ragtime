-- Add token optimization settings to app_settings
-- These settings help reduce token consumption during multi-tool agent loops

-- max_tool_output_chars: Maximum characters per tool output before truncation (0=unlimited)
-- Default 8000 chars (~2000 tokens) balances detail vs. token budget
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "max_tool_output_chars" INTEGER NOT NULL DEFAULT 8000;

-- scratchpad_window_size: Keep last N tool calls in full detail; older ones are summarized (0=keep all)
-- Default 10 keeps recent context while compressing older steps
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "scratchpad_window_size" INTEGER NOT NULL DEFAULT 10;
