ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "available_models_cache_enabled" BOOLEAN NOT NULL DEFAULT true;
