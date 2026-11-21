CREATE TABLE IF NOT EXISTS "instruction_skills" (
    "id" TEXT NOT NULL,
    "tool_config_id" TEXT,
    "mcp_route_id" TEXT,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "mode" TEXT NOT NULL DEFAULT 'on_demand',
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "sort_order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "instruction_skills_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "instruction_skills_owner_check" CHECK (num_nonnulls("tool_config_id", "mcp_route_id") = 1),
    CONSTRAINT "instruction_skills_mode_check" CHECK ("mode" IN ('always', 'on_demand')),
    CONSTRAINT "instruction_skills_tool_config_id_fkey" FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "instruction_skills_mcp_route_id_fkey" FOREIGN KEY ("mcp_route_id") REFERENCES "mcp_route_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "instruction_skills_tool_config_id_name_key" ON "instruction_skills"("tool_config_id", "name");
CREATE UNIQUE INDEX IF NOT EXISTS "instruction_skills_mcp_route_id_name_key" ON "instruction_skills"("mcp_route_id", "name");
CREATE INDEX IF NOT EXISTS "instruction_skills_tool_config_id_idx" ON "instruction_skills"("tool_config_id");
CREATE INDEX IF NOT EXISTS "instruction_skills_mcp_route_id_idx" ON "instruction_skills"("mcp_route_id");
