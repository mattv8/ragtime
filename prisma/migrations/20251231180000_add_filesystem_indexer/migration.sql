-- Add filesystem indexer tables for pgvector-based document indexing
-- Requires pgvector extension (should already be enabled)

-- Ensure pgvector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- Add new ToolType enum value for filesystem_indexer
ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'filesystem_indexer';

-- Filesystem mount type enum
CREATE TYPE "FilesystemMountType" AS ENUM ('docker_volume', 'smb', 'nfs', 'local');

-- Filesystem index status enum
CREATE TYPE "FilesystemIndexStatus" AS ENUM ('pending', 'indexing', 'completed', 'failed', 'cancelled');

-- Filesystem indexing jobs table
CREATE TABLE "filesystem_index_jobs" (
    "id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "status" "FilesystemIndexStatus" NOT NULL DEFAULT 'pending',
    "index_name" TEXT NOT NULL,
    "total_files" INTEGER NOT NULL DEFAULT 0,
    "processed_files" INTEGER NOT NULL DEFAULT 0,
    "skipped_files" INTEGER NOT NULL DEFAULT 0,
    "total_chunks" INTEGER NOT NULL DEFAULT 0,
    "processed_chunks" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "filesystem_index_jobs_pkey" PRIMARY KEY ("id")
);

-- Filesystem file metadata for incremental indexing
CREATE TABLE "filesystem_file_metadata" (
    "id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "file_path" TEXT NOT NULL,
    "file_hash" TEXT NOT NULL,
    "file_size" BIGINT NOT NULL,
    "mime_type" TEXT,
    "chunk_count" INTEGER NOT NULL DEFAULT 0,
    "last_indexed" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "filesystem_file_metadata_pkey" PRIMARY KEY ("id")
);

-- Filesystem embeddings with pgvector
-- Note: embedding column dimension set to 1536 (OpenAI text-embedding-3-small default)
-- Can be altered if using different embedding model
CREATE TABLE "filesystem_embeddings" (
    "id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "file_path" TEXT NOT NULL,
    "chunk_index" INTEGER NOT NULL,
    "content" TEXT NOT NULL,
    "embedding" vector(1536),
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "filesystem_embeddings_pkey" PRIMARY KEY ("id")
);

-- Foreign key constraint for filesystem_index_jobs -> tool_configs
ALTER TABLE "filesystem_index_jobs" ADD CONSTRAINT "filesystem_index_jobs_tool_config_id_fkey"
    FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Unique constraint for file metadata (one entry per file per index)
CREATE UNIQUE INDEX "filesystem_file_metadata_index_name_file_path_key"
    ON "filesystem_file_metadata"("index_name", "file_path");

-- Index for fast file metadata lookups by index name
CREATE INDEX "filesystem_file_metadata_index_name_idx"
    ON "filesystem_file_metadata"("index_name");

-- Index for fast embedding lookups by index name
CREATE INDEX "filesystem_embeddings_index_name_idx"
    ON "filesystem_embeddings"("index_name");

-- HNSW index for fast approximate nearest neighbor search
-- Using cosine distance (suitable for normalized embeddings)
-- Note: HNSW indexes have a 2000-dimension limit in pgvector
CREATE INDEX "filesystem_embeddings_embedding_idx"
    ON "filesystem_embeddings"
    USING hnsw ("embedding" vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
