ALTER TABLE "provider_prompt_debug_records"
    ADD COLUMN IF NOT EXISTS "debug_metadata" JSONB;