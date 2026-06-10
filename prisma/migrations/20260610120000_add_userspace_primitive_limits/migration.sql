ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "userspace_primitive_upload_max_bytes" INTEGER NOT NULL DEFAULT 104857600,
  ADD COLUMN IF NOT EXISTS "userspace_primitive_archive_max_entries" INTEGER NOT NULL DEFAULT 500;
