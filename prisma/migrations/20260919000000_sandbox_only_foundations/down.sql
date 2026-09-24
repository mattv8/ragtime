DROP TABLE IF EXISTS "workspace_index_grants";
DROP TABLE IF EXISTS "workspace_development_credentials";
ALTER TABLE "users" DROP COLUMN IF EXISTS "hosted_chat_enabled";
ALTER TABLE "app_settings" DROP COLUMN IF EXISTS "hosted_chat_enabled";
