ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "chat_attachment_token_budget" INTEGER NOT NULL DEFAULT 0;
