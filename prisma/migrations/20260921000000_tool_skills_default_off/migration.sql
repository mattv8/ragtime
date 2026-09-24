ALTER TABLE "app_settings" ALTER COLUMN "tool_skills_enabled" SET DEFAULT false;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'app_settings' AND column_name = 'hosted_chat_enabled') THEN
    ALTER TABLE "app_settings" RENAME COLUMN "hosted_chat_enabled" TO "userspace_generation_enabled";
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'app_settings' AND column_name = 'chat_enabled') THEN
    ALTER TABLE "app_settings" ADD COLUMN "chat_enabled" BOOLEAN NOT NULL DEFAULT true;
    UPDATE "app_settings" SET "chat_enabled" = "userspace_generation_enabled";
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'hosted_chat_enabled') THEN
    ALTER TABLE "users" RENAME COLUMN "hosted_chat_enabled" TO "userspace_generation_enabled";
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'chat_enabled') THEN
    ALTER TABLE "users" ADD COLUMN "chat_enabled" BOOLEAN;
    UPDATE "users" SET "chat_enabled" = "userspace_generation_enabled";
  END IF;
END $$;

ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "bridge_credential_mode";
