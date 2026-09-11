-- New installations use adaptive resource scheduling. Existing positive caps
-- are deliberately preserved by changing only column defaults.
ALTER TABLE "app_settings"
    ADD COLUMN IF NOT EXISTS "indexing_memory_budget_mb" INTEGER NOT NULL DEFAULT 0;

ALTER TABLE "app_settings"
    ALTER COLUMN "chunking_max_workers" SET DEFAULT 0,
    ALTER COLUMN "chunking_max_batch_size" SET DEFAULT 0;
