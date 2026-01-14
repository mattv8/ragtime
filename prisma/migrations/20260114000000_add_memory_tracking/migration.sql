-- Add memory tracking fields to index_metadata
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "embedding_dimension" INTEGER;
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "steady_memory_bytes" BIGINT;
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "peak_memory_bytes" BIGINT;
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "load_time_seconds" DOUBLE PRECISION;

-- Add sequential index loading setting to app_settings
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "sequential_index_loading" BOOLEAN NOT NULL DEFAULT false;
