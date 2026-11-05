ALTER TABLE "index_configs"
    ADD COLUMN IF NOT EXISTS "config_snapshot" JSONB;
