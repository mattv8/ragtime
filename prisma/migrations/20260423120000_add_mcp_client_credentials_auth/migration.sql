-- Add 'client_credentials' variant to McpAuthMethod enum (OAuth2 Client Credentials grant
-- with pre-shared client_id / client_secret for clients like Claude custom connectors).
DO $$
BEGIN
  ALTER TYPE "McpAuthMethod" ADD VALUE IF NOT EXISTS 'client_credentials';
EXCEPTION WHEN duplicate_object THEN NULL;
END$$;

-- Add client_id column to custom MCP route configs (plaintext public identifier).
ALTER TABLE "mcp_route_configs"
  ADD COLUMN IF NOT EXISTS "auth_client_id" TEXT;

-- Add client_id column for the default /mcp route on app_settings.
ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "mcp_default_route_client_id" TEXT;
