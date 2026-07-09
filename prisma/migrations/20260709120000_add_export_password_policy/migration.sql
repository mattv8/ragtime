-- Add shared export password policy settings to app_settings

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "export_password_min_length" INTEGER NOT NULL DEFAULT 12;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "export_password_require_uppercase" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "export_password_require_lowercase" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "export_password_require_number" BOOLEAN NOT NULL DEFAULT true;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "export_password_require_special" BOOLEAN NOT NULL DEFAULT true;
