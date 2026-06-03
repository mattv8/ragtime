ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "chat_auto_compaction_threshold_percent" INTEGER NOT NULL DEFAULT 99;