-- AlterTable: add configurable sync interval to mount sources
ALTER TABLE "userspace_mount_sources"
ADD COLUMN IF NOT EXISTS "sync_interval_seconds" INTEGER DEFAULT 30;
