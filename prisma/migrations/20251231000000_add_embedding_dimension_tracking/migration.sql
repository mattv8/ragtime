-- Add embedding dimension tracking to app_settings
-- This tracks the vector dimension used in filesystem_embeddings table
-- and the embedding provider+model configuration hash for mismatch detection

-- Add columns for embedding dimension tracking
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "embedding_dimension" INTEGER;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "embedding_config_hash" TEXT;

-- Add user-configurable embedding dimensions for OpenAI text-embedding-3-* models
-- These models support Matryoshka Representation Learning (MRL), allowing output
-- dimensions to be reduced at API call time without retraining.
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "embedding_dimensions" INTEGER;

-- Note: The filesystem_embeddings table uses pgvector's vector type.
-- Default dimensions by model:
-- - OpenAI text-embedding-3-small: 1536 dimensions (configurable 256-1536)
-- - OpenAI text-embedding-3-large: 3072 dimensions (configurable 256-3072)
-- - Ollama nomic-embed-text: 768 dimensions (fixed)
-- - Ollama mxbai-embed-large: 1024 dimensions (fixed)
--
-- IMPORTANT: pgvector IVFFlat/HNSW indexes have a 2000-dimension limit.
-- For text-embedding-3-large, set embedding_dimensions <= 2000 (e.g., 1536)
-- to enable fast indexed vector search. Without this, exact search is used.
--
-- When embedding_config_hash changes (provider, model, or dimensions), existing
-- embeddings become incompatible and a full reindex is required.
