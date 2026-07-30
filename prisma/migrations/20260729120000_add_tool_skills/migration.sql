ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "tool_skills_enabled" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "conversations" ADD COLUMN IF NOT EXISTS "loaded_tool_skill_ids" JSONB NOT NULL DEFAULT '[]'::jsonb;
