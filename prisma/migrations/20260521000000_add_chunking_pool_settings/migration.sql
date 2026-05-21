-- Add tunable chunking pool sizing knobs to app_settings.
-- Defaults preserve historical behavior (4 workers, 100-doc batches).
ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "chunking_max_workers" INTEGER NOT NULL DEFAULT 4;

ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "chunking_max_batch_size" INTEGER NOT NULL DEFAULT 100;
