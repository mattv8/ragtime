-- Add retrieval optimization settings to app_settings
-- These settings control chunking, context budget, MMR diversification, and pgvector tuning

-- MMR (Max Marginal Relevance) settings for result diversification
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "search_use_mmr" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS "search_mmr_lambda" DOUBLE PRECISION NOT NULL DEFAULT 0.5;

-- Token-based context budget (limits tokens sent to LLM from retrieved chunks)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "context_token_budget" INTEGER NOT NULL DEFAULT 4000;

-- Token-based chunking (more accurate than character-based)
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "chunking_use_tokens" BOOLEAN NOT NULL DEFAULT true;

-- pgvector IVFFlat index tuning
ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "ivfflat_lists" INTEGER NOT NULL DEFAULT 100;

-- Add comments for documentation
COMMENT ON COLUMN "app_settings"."search_use_mmr" IS 'Use Max Marginal Relevance for result diversification (reduces near-duplicate results)';
COMMENT ON COLUMN "app_settings"."search_mmr_lambda" IS 'MMR lambda parameter: 0=max diversity, 1=max relevance (default 0.5)';
COMMENT ON COLUMN "app_settings"."context_token_budget" IS 'Maximum tokens for retrieved context sent to LLM (0=unlimited)';
COMMENT ON COLUMN "app_settings"."chunking_use_tokens" IS 'Use token-based chunking instead of character-based for more consistent chunk sizes';
COMMENT ON COLUMN "app_settings"."ivfflat_lists" IS 'IVFFlat index lists parameter for pgvector (higher=slower build, faster query for large datasets)';
