-- SolidWorks PDM Indexer Migration
-- Adds support for indexing SolidWorks PDM document metadata

-- Add solidworks_pdm to ToolType enum
ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'solidworks_pdm';

-- Create PDM index status enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'PdmIndexStatus') THEN
        CREATE TYPE "PdmIndexStatus" AS ENUM ('pending', 'indexing', 'completed', 'failed', 'cancelled');
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Create PDM index jobs table
CREATE TABLE IF NOT EXISTS "pdm_index_jobs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "tool_config_id" TEXT NOT NULL,
    "status" "PdmIndexStatus" NOT NULL DEFAULT 'pending',
    "index_name" TEXT NOT NULL,
    "total_documents" INTEGER NOT NULL DEFAULT 0,
    "processed_documents" INTEGER NOT NULL DEFAULT 0,
    "skipped_documents" INTEGER NOT NULL DEFAULT 0,
    "total_chunks" INTEGER NOT NULL DEFAULT 0,
    "processed_chunks" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "started_at" TIMESTAMPTZ,
    "completed_at" TIMESTAMPTZ
);

-- Create index on tool_config_id for job lookup
CREATE INDEX IF NOT EXISTS "pdm_index_jobs_tool_config_id_idx" ON "pdm_index_jobs" ("tool_config_id");

-- Create PDM document metadata table for incremental indexing
CREATE TABLE IF NOT EXISTS "pdm_document_metadata" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "index_name" TEXT NOT NULL,
    "document_id" INTEGER NOT NULL,
    "filename" TEXT NOT NULL,
    "revision_no" INTEGER NOT NULL,
    "metadata_hash" TEXT NOT NULL,
    "last_indexed" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create unique constraint on index_name + document_id
CREATE UNIQUE INDEX IF NOT EXISTS "pdm_document_metadata_index_doc_unique"
    ON "pdm_document_metadata" ("index_name", "document_id");

-- Create index on index_name for efficient filtering
CREATE INDEX IF NOT EXISTS "pdm_document_metadata_index_name_idx" ON "pdm_document_metadata" ("index_name");

-- Create PDM embeddings table
CREATE TABLE IF NOT EXISTS "pdm_embeddings" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "index_name" TEXT NOT NULL,
    "document_id" INTEGER NOT NULL,
    "document_type" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "part_number" TEXT,
    "filename" TEXT NOT NULL,
    "folder_path" TEXT,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create unique constraint on index_name + document_id
CREATE UNIQUE INDEX IF NOT EXISTS "pdm_embeddings_index_doc_unique"
    ON "pdm_embeddings" ("index_name", "document_id");

-- Create index on index_name for efficient filtering
CREATE INDEX IF NOT EXISTS "pdm_embeddings_index_name_idx" ON "pdm_embeddings" ("index_name");

-- Create index on part_number for filtering
CREATE INDEX IF NOT EXISTS "pdm_embeddings_part_number_idx" ON "pdm_embeddings" ("part_number");

-- Add embedding column with vector type (dimension set dynamically, starting with 1536)
-- This matches the filesystem_embeddings and schema_embeddings pattern
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'pdm_embeddings' AND column_name = 'embedding'
    ) THEN
        EXECUTE 'ALTER TABLE pdm_embeddings ADD COLUMN embedding vector(1536)';
    END IF;
END $$;

-- Create vector index for similarity search (only if dimension <= 2000)
-- Using IVFFlat index for approximate nearest neighbor search
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'pdm_embeddings_embedding_idx'
    ) THEN
        -- Check dimension before creating index (pgvector limit is 2000)
        -- Default 1536 is safe for index
        CREATE INDEX "pdm_embeddings_embedding_idx"
            ON "pdm_embeddings"
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    END IF;
EXCEPTION
    WHEN others THEN
        -- If index creation fails (e.g., not enough data for IVFFlat), skip
        RAISE NOTICE 'Could not create vector index: %', SQLERRM;
END $$;
