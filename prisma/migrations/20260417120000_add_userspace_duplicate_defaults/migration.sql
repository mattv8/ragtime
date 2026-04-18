ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_duplicate_copy_files_default" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_duplicate_copy_metadata_default" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_duplicate_copy_chats_default" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_duplicate_copy_mounts_default" BOOLEAN NOT NULL DEFAULT false;
