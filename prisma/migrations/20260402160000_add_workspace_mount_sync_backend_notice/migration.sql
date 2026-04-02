-- AlterTable: persist SSH mount sync backend and user-facing notice
ALTER TABLE "workspace_mounts"
ADD COLUMN IF NOT EXISTS "sync_backend" TEXT,
ADD COLUMN IF NOT EXISTS "sync_notice" TEXT;
