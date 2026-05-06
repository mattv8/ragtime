ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_sqlite_import_max_bytes" BIGINT NOT NULL DEFAULT 104857600;
