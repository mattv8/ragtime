-- Add snapshot retention days setting for User Space
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "snapshot_retention_days" INTEGER NOT NULL DEFAULT 0;
