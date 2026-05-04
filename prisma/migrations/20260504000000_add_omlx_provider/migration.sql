ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "omlx_protocol" TEXT NOT NULL DEFAULT 'http';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "omlx_host" TEXT NOT NULL DEFAULT 'host.docker.internal';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "omlx_port" INTEGER NOT NULL DEFAULT 8000;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "omlx_base_url" TEXT NOT NULL DEFAULT 'http://host.docker.internal:8000';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "omlx_api_key" TEXT;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "llm_omlx_protocol" TEXT NOT NULL DEFAULT 'http';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "llm_omlx_host" TEXT NOT NULL DEFAULT 'host.docker.internal';
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "llm_omlx_port" INTEGER NOT NULL DEFAULT 8000;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "llm_omlx_base_url" TEXT NOT NULL DEFAULT 'http://host.docker.internal:8000';