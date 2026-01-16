-- Add MCP Default Route Filters
-- Allows configuring LDAP group-based tool filtering on the default /mcp route

-- Create the default route filters table
CREATE TABLE IF NOT EXISTS "mcp_default_route_filters" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "ldap_group_dn" TEXT NOT NULL,
    "include_knowledge_search" BOOLEAN NOT NULL DEFAULT true,
    "include_git_history" BOOLEAN NOT NULL DEFAULT true,
    "selected_document_indexes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "selected_filesystem_indexes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "selected_schema_indexes" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "mcp_default_route_filters_pkey" PRIMARY KEY ("id")
);

-- Unique constraint on LDAP group DN (one filter per group)
CREATE UNIQUE INDEX IF NOT EXISTS "mcp_default_route_filters_ldap_group_dn_key" ON "mcp_default_route_filters"("ldap_group_dn");

-- Create the junction table for filter to tool config mapping
CREATE TABLE IF NOT EXISTS "mcp_default_route_filter_tool_selections" (
    "id" TEXT NOT NULL,
    "filter_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,

    CONSTRAINT "mcp_default_route_filter_tool_selections_pkey" PRIMARY KEY ("id")
);

-- Unique constraint on filter + tool combination
CREATE UNIQUE INDEX IF NOT EXISTS "mcp_default_route_filter_tool_selections_filter_id_tool_config_key" ON "mcp_default_route_filter_tool_selections"("filter_id", "tool_config_id");

-- Foreign key to mcp_default_route_filters
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'mcp_default_route_filter_tool_selections_filter_id_fkey'
    ) THEN
        ALTER TABLE "mcp_default_route_filter_tool_selections"
        ADD CONSTRAINT "mcp_default_route_filter_tool_selections_filter_id_fkey"
        FOREIGN KEY ("filter_id") REFERENCES "mcp_default_route_filters"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- Foreign key to tool_configs
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'mcp_default_route_filter_tool_selections_tool_config_id_fkey'
    ) THEN
        ALTER TABLE "mcp_default_route_filter_tool_selections"
        ADD CONSTRAINT "mcp_default_route_filter_tool_selections_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
