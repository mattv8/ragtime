ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "openapi_model_prefix_enabled" BOOLEAN NOT NULL DEFAULT true;