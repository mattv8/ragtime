-- Add dedicated encrypted PAT field for GitHub Models provider
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "github_models_api_token" TEXT NOT NULL DEFAULT '';
