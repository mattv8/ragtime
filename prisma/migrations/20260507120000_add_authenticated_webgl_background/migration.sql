ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "authenticated_webgl_background_enabled" BOOLEAN NOT NULL DEFAULT true;
