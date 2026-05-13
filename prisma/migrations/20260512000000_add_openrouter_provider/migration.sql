ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "openrouter_api_key" TEXT NOT NULL DEFAULT '';
