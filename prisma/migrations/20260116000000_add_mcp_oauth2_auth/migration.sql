-- Add OAuth2 authentication support for MCP routes
-- This allows MCP clients to authenticate via LDAP using OAuth2 flow

-- Create enum for MCP auth methods
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'McpAuthMethod') THEN
        CREATE TYPE "McpAuthMethod" AS ENUM ('password', 'oauth2');
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Add auth_method and allowed_ldap_group to mcp_route_configs
ALTER TABLE "mcp_route_configs"
ADD COLUMN IF NOT EXISTS "auth_method" "McpAuthMethod" NOT NULL DEFAULT 'password',
ADD COLUMN IF NOT EXISTS "allowed_ldap_group" TEXT;

-- Add auth_method and allowed_group to app_settings for default route
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "mcp_default_route_auth_method" "McpAuthMethod" NOT NULL DEFAULT 'password',
ADD COLUMN IF NOT EXISTS "mcp_default_route_allowed_group" TEXT;
