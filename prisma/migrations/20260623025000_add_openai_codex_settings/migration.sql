ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "openai_codex_access_token" TEXT NOT NULL DEFAULT '',
ADD COLUMN IF NOT EXISTS "openai_codex_refresh_token" TEXT NOT NULL DEFAULT '',
ADD COLUMN IF NOT EXISTS "openai_codex_token_expires_at" TIMESTAMP(3),
ADD COLUMN IF NOT EXISTS "openai_codex_account_id" TEXT NOT NULL DEFAULT '',
ADD COLUMN IF NOT EXISTS "openai_codex_base_url" TEXT NOT NULL DEFAULT 'https://chatgpt.com/backend-api/codex';
