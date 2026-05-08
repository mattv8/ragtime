-- Add model provider precedence configuration to AppSettings.
--
-- Stores admin-defined ordering of LLM providers plus optional per-model and
-- per-family overrides. Used to pick which provider to surface a model from
-- when the same model id is offered by multiple providers, and to gracefully
-- fall back when the originally-selected provider is offline.
--
-- Shape: {"providers": string[], "model_overrides": {modelId: provider}, "family_overrides": {family: provider}}
--
-- Two columns are added: one governs in-app chat resolution, the other governs
-- the same logic for OpenAI-compatible /v1/models clients.

ALTER TABLE "AppSettings"
ADD COLUMN "model_provider_precedence" JSONB NOT NULL
DEFAULT '{"providers": [], "model_overrides": {}, "family_overrides": {}}'::jsonb;

ALTER TABLE "AppSettings"
ADD COLUMN "openapi_model_provider_precedence" JSONB NOT NULL
DEFAULT '{"providers": [], "model_overrides": {}, "family_overrides": {}}'::jsonb;
