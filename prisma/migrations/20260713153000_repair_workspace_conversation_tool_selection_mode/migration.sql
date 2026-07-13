UPDATE "conversations" c
SET "tool_selection_mode" = 'default_all'
WHERE c."workspace_id" IS NOT NULL
  AND c."tool_selection_mode" = 'custom'
  AND NOT EXISTS (
    SELECT 1
    FROM "conversation_tool_selections" s
    WHERE s."conversation_id" = c."id"
  )
  AND NOT EXISTS (
    SELECT 1
    FROM "conversation_tool_group_selections" gs
    WHERE gs."conversation_id" = c."id"
  );
