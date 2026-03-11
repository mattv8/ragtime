-- Add OpenAPI model list and sync toggle to app_settings
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "allowed_openapi_models" TEXT[] DEFAULT ARRAY[]::TEXT[];
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "openapi_sync_chat_models" BOOLEAN NOT NULL DEFAULT true;
