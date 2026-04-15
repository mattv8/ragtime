-- Add stale branch threshold setting (hide branches whose head is N+ commits behind active head)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "snapshot_stale_branch_threshold" INTEGER NOT NULL DEFAULT 20;
