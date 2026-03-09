-- Persist whether GitHub Copilot model discovery should include 3rd-party families
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "include_copilot_third_party_models" BOOLEAN NOT NULL DEFAULT false;
