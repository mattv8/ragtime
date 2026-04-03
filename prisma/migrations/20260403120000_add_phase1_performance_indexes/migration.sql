-- Add performance-oriented indexes for auth, conversations, and tool selection tables.

CREATE INDEX IF NOT EXISTS "sessions_user_id_idx"
ON "sessions"("user_id");

CREATE INDEX IF NOT EXISTS "sessions_user_id_expires_at_idx"
ON "sessions"("user_id", "expires_at");

CREATE INDEX IF NOT EXISTS "sessions_expires_at_idx"
ON "sessions"("expires_at");

CREATE INDEX IF NOT EXISTS "conversations_user_id_idx"
ON "conversations"("user_id");

CREATE INDEX IF NOT EXISTS "conversations_active_task_id_idx"
ON "conversations"("active_task_id");

CREATE INDEX IF NOT EXISTS "mcp_route_tool_selections_mcp_route_id_idx"
ON "mcp_route_tool_selections"("mcp_route_id");

CREATE INDEX IF NOT EXISTS "mcp_route_tool_selections_tool_config_id_idx"
ON "mcp_route_tool_selections"("tool_config_id");

CREATE INDEX IF NOT EXISTS "mcp_default_route_filter_tool_selections_filter_id_idx"
ON "mcp_default_route_filter_tool_selections"("filter_id");

CREATE INDEX IF NOT EXISTS "mcp_default_route_filter_tool_selections_tool_config_id_idx"
ON "mcp_default_route_filter_tool_selections"("tool_config_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_selections_workspace_id_idx"
ON "workspace_tool_selections"("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_group_selections_workspace_id_idx"
ON "workspace_tool_group_selections"("workspace_id");

CREATE INDEX IF NOT EXISTS "conversation_tool_selections_conversation_id_idx"
ON "conversation_tool_selections"("conversation_id");

CREATE INDEX IF NOT EXISTS "conversation_tool_group_selections_conversation_id_idx"
ON "conversation_tool_group_selections"("conversation_id");