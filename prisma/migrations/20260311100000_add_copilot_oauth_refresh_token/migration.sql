-- AlterTable
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "github_copilot_oauth_refresh_token" TEXT NOT NULL DEFAULT '';
