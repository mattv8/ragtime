-- Add vector_store_type to index_metadata table for document indexes
-- This allows document indexes (upload + git) to use either FAISS or pgvector
-- Default is 'faiss' to match existing behavior for document indexes

-- Standardize on VectorStoreType (deprecate FilesystemVectorStoreType)
DO $$
BEGIN
    -- Rename FilesystemVectorStoreType to VectorStoreType if it exists
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'FilesystemVectorStoreType') THEN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'VectorStoreType') THEN
            ALTER TYPE "FilesystemVectorStoreType" RENAME TO "VectorStoreType";
        END IF;
    -- Create VectorStoreType if neither exists
    ELSIF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'VectorStoreType') THEN
        CREATE TYPE "VectorStoreType" AS ENUM ('pgvector', 'faiss');
    END IF;
END
$$;

ALTER TABLE index_metadata
ADD COLUMN IF NOT EXISTS vector_store_type "VectorStoreType" NOT NULL DEFAULT 'faiss';

-- Add comment for documentation
COMMENT ON COLUMN index_metadata.vector_store_type IS 'Vector store backend: pgvector (PostgreSQL, persistent) or faiss (in-memory, loaded at startup). Default faiss for backward compatibility.';
