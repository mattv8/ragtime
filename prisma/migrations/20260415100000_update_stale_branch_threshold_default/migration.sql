-- Update stale branch threshold default from 20 to 50; allow 0 (show all)
ALTER TABLE "app_settings"
ALTER COLUMN "snapshot_stale_branch_threshold" SET DEFAULT 50;
