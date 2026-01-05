-- Add search configuration to app_settings
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "search_results_k" INTEGER NOT NULL DEFAULT 5,
ADD COLUMN IF NOT EXISTS "aggregate_search" BOOLEAN NOT NULL DEFAULT true;

-- Add search_weight to index_metadata
ALTER TABLE "index_metadata"
ADD COLUMN IF NOT EXISTS "search_weight" DOUBLE PRECISION NOT NULL DEFAULT 1.0;

-- Add comment for clarity
COMMENT ON COLUMN "app_settings"."search_results_k" IS 'Number of results per vector search query (k)';
COMMENT ON COLUMN "app_settings"."aggregate_search" IS 'If true, use single search_knowledge tool; if false, create per-index tools';
COMMENT ON COLUMN "index_metadata"."search_weight" IS 'Search weight for result prioritization (0.0-10.0, default 1.0)';
