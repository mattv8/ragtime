ALTER TABLE "app_settings"
ADD COLUMN "openai_compatible_base_url" TEXT NOT NULL DEFAULT '',
ADD COLUMN "openai_compatible_api_key" TEXT NOT NULL DEFAULT '',
ADD COLUMN "openai_compatible_catalog_provider" TEXT NOT NULL DEFAULT '',
ADD COLUMN "openai_compatible_model_limits" JSONB NOT NULL DEFAULT '{}';
