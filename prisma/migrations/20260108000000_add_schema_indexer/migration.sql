-- Schema Indexer Migration
-- Adds support for indexing SQL database schemas (PostgreSQL, MSSQL, MySQL/MariaDB)

-- Create schema index status enum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'SchemaIndexStatus') THEN
        CREATE TYPE "SchemaIndexStatus" AS ENUM ('pending', 'indexing', 'completed', 'failed', 'cancelled');
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Create schema index jobs table
CREATE TABLE IF NOT EXISTS "schema_index_jobs" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "tool_config_id" TEXT NOT NULL,
    "status" "SchemaIndexStatus" NOT NULL DEFAULT 'pending',
    "index_name" TEXT NOT NULL,
    "total_tables" INTEGER NOT NULL DEFAULT 0,
    "processed_tables" INTEGER NOT NULL DEFAULT 0,
    "total_chunks" INTEGER NOT NULL DEFAULT 0,
    "processed_chunks" INTEGER NOT NULL DEFAULT 0,
    "error_message" TEXT,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    "started_at" TIMESTAMPTZ,
    "completed_at" TIMESTAMPTZ
);

-- Create index on tool_config_id for job lookup
CREATE INDEX IF NOT EXISTS "schema_index_jobs_tool_config_id_idx" ON "schema_index_jobs" ("tool_config_id");

-- Create schema embeddings table
CREATE TABLE IF NOT EXISTS "schema_embeddings" (
    "id" UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "index_name" TEXT NOT NULL,
    "table_name" TEXT NOT NULL,
    "table_schema" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create unique constraint on index_name + table_name
CREATE UNIQUE INDEX IF NOT EXISTS "schema_embeddings_index_table_unique"
    ON "schema_embeddings" ("index_name", "table_name");

-- Create index on index_name for efficient filtering
CREATE INDEX IF NOT EXISTS "schema_embeddings_index_name_idx" ON "schema_embeddings" ("index_name");

-- Add embedding column with vector type (dimension set dynamically, starting with 1536)
-- This matches the filesystem_embeddings pattern
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'schema_embeddings' AND column_name = 'embedding'
    ) THEN
        EXECUTE 'ALTER TABLE schema_embeddings ADD COLUMN embedding vector(1536)';
    END IF;
END $$;

-- Create vector index for similarity search (only if dimension <= 2000)
-- Using IVFFlat index for approximate nearest neighbor search
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'schema_embeddings_embedding_idx'
    ) THEN
        -- Check dimension before creating index (pgvector limit is 2000)
        -- Default 1536 is safe for index
        CREATE INDEX "schema_embeddings_embedding_idx"
            ON "schema_embeddings"
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    END IF;
EXCEPTION
    WHEN others THEN
        -- If index creation fails (e.g., not enough data for IVFFlat), skip
        RAISE NOTICE 'Could not create vector index: %', SQLERRM;
END $$;
