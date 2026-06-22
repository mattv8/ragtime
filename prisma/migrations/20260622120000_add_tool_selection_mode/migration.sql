ALTER TABLE "conversations" ADD COLUMN "tool_selection_mode" TEXT NOT NULL DEFAULT 'custom';
ALTER TABLE "workspaces" ADD COLUMN "tool_selection_mode" TEXT NOT NULL DEFAULT 'custom';

UPDATE "conversations" c
SET "tool_selection_mode" = 'default_all'
WHERE NOT EXISTS (
  SELECT 1 FROM "conversation_tool_selections" s WHERE s."conversation_id" = c."id"
)
AND NOT EXISTS (
  SELECT 1 FROM "conversation_tool_group_selections" gs WHERE gs."conversation_id" = c."id"
);

UPDATE "workspaces" w
SET "tool_selection_mode" = 'default_all'
WHERE NOT EXISTS (
  SELECT 1 FROM "workspace_tool_selections" s WHERE s."workspace_id" = w."id"
)
AND NOT EXISTS (
  SELECT 1 FROM "workspace_tool_group_selections" gs WHERE gs."workspace_id" = w."id"
);
