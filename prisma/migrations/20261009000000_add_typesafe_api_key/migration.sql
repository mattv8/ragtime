ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "typesafe_api_key" TEXT NOT NULL DEFAULT '';
