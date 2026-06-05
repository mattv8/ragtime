ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "userspace_mount_sync_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "userspace_mount_sync_timezone" TEXT;

ALTER TABLE "workspaces"
    ADD COLUMN IF NOT EXISTS "scm_auto_push_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "scm_auto_push_timezone" TEXT,
    ADD COLUMN IF NOT EXISTS "scm_auto_pull_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "scm_auto_pull_timezone" TEXT;

ALTER TABLE "userspace_mount_sources"
    ADD COLUMN IF NOT EXISTS "sync_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "sync_timezone" TEXT;

ALTER TABLE "user_userspace_mount_sources"
    ADD COLUMN IF NOT EXISTS "sync_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "sync_timezone" TEXT;

ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "sync_start_minute" INTEGER,
    ADD COLUMN IF NOT EXISTS "sync_timezone" TEXT;
