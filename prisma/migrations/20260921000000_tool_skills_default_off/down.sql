ALTER TABLE "app_settings" ALTER COLUMN "tool_skills_enabled" SET DEFAULT true;
ALTER TABLE "users" DROP COLUMN IF EXISTS "chat_enabled";
ALTER TABLE "app_settings" DROP COLUMN IF EXISTS "chat_enabled";
ALTER TABLE "users" RENAME COLUMN "userspace_generation_enabled" TO "hosted_chat_enabled";
ALTER TABLE "app_settings" RENAME COLUMN "userspace_generation_enabled" TO "hosted_chat_enabled";
ALTER TABLE "workspaces" ADD COLUMN IF NOT EXISTS "bridge_credential_mode" TEXT NOT NULL DEFAULT 'env';
