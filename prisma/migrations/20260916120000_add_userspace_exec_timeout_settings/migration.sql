ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "userspace_exec_timeout_default_seconds" INTEGER NOT NULL DEFAULT 120;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "userspace_exec_timeout_max_seconds" INTEGER NOT NULL DEFAULT 600;
