-- Add MCP enabled toggle to app_settings
-- Default to false (disabled) for security

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "mcp_enabled" BOOLEAN NOT NULL DEFAULT false;
