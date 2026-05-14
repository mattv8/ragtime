-- Remove deprecated model provider precedence settings.
-- Model selections are now host-scoped, so explicit provider ordering and
-- per-model/family override JSON blobs are no longer used.

ALTER TABLE "app_settings"
DROP COLUMN IF EXISTS "model_provider_precedence",
DROP COLUMN IF EXISTS "openapi_model_provider_precedence";