-- Add GitHub Copilot chat provider settings to app_settings.
ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "github_copilot_access_token" TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS "github_copilot_refresh_token" TEXT NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS "github_copilot_token_expires_at" TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS "github_copilot_enterprise_url" TEXT,
  ADD COLUMN IF NOT EXISTS "github_copilot_base_url" TEXT NOT NULL DEFAULT 'https://api.githubcopilot.com';
