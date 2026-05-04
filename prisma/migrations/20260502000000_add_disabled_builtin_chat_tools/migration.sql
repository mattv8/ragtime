ALTER TABLE "conversations"
ADD COLUMN IF NOT EXISTS "disabled_builtin_tool_ids" JSONB NOT NULL DEFAULT '[]'::jsonb;