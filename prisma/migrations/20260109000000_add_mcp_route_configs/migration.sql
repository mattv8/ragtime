-- Add MCP configuration to app_settings
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "mcp_default_route_auth" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "mcp_default_route_password" TEXT;

-- Create MCP route configurations table
CREATE TABLE IF NOT EXISTS "mcp_route_configs" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "route_path" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "require_auth" BOOLEAN NOT NULL DEFAULT false,
    "auth_password" TEXT,
    "include_knowledge_search" BOOLEAN NOT NULL DEFAULT true,
    "include_git_history" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "mcp_route_configs_pkey" PRIMARY KEY ("id")
);

-- Create unique index on route_path
CREATE UNIQUE INDEX IF NOT EXISTS "mcp_route_configs_route_path_key" ON "mcp_route_configs"("route_path");

-- Create junction table for MCP route to tool config many-to-many relationship
CREATE TABLE IF NOT EXISTS "mcp_route_tool_selections" (
    "id" TEXT NOT NULL,
    "mcp_route_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,

    CONSTRAINT "mcp_route_tool_selections_pkey" PRIMARY KEY ("id")
);

-- Create unique constraint on route + tool combination
CREATE UNIQUE INDEX IF NOT EXISTS "mcp_route_tool_selections_mcp_route_id_tool_config_id_key"
ON "mcp_route_tool_selections"("mcp_route_id", "tool_config_id");

-- Add foreign key constraints
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'mcp_route_tool_selections_mcp_route_id_fkey'
    ) THEN
        ALTER TABLE "mcp_route_tool_selections"
        ADD CONSTRAINT "mcp_route_tool_selections_mcp_route_id_fkey"
        FOREIGN KEY ("mcp_route_id") REFERENCES "mcp_route_configs"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'mcp_route_tool_selections_tool_config_id_fkey'
    ) THEN
        ALTER TABLE "mcp_route_tool_selections"
        ADD CONSTRAINT "mcp_route_tool_selections_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- Add index selection columns for per-index control
ALTER TABLE "mcp_route_configs"
ADD COLUMN IF NOT EXISTS "selected_document_indexes" TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE "mcp_route_configs"
ADD COLUMN IF NOT EXISTS "selected_filesystem_indexes" TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE "mcp_route_configs"
ADD COLUMN IF NOT EXISTS "selected_schema_indexes" TEXT[] NOT NULL DEFAULT '{}';
