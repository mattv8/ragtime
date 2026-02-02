-- Fix filesystem table names - handles various database states idempotently
-- This migration ensures the correct table names exist regardless of prior state

-- Case 1: index_embeddings exists but filesystem_embeddings doesn't -> rename
-- Case 2: Both exist -> drop index_embeddings (it's orphaned)
-- Case 3: Only filesystem_embeddings exists -> no action needed
-- Case 4: Neither exists -> let original migration handle it

-- Fix index_embeddings -> filesystem_embeddings
DO $$
DECLARE
    index_emb_exists BOOLEAN;
    fs_emb_exists BOOLEAN;
BEGIN
    -- Check what tables exist
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'index_embeddings' AND table_schema = 'public'
    ) INTO index_emb_exists;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'filesystem_embeddings' AND table_schema = 'public'
    ) INTO fs_emb_exists;

    IF index_emb_exists AND NOT fs_emb_exists THEN
        -- Rename index_embeddings to filesystem_embeddings
        RAISE NOTICE 'Renaming index_embeddings to filesystem_embeddings';
        ALTER TABLE index_embeddings RENAME TO filesystem_embeddings;

        -- Fix primary key constraint name if needed
        IF EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'index_embeddings_pkey'
        ) THEN
            ALTER TABLE filesystem_embeddings
            RENAME CONSTRAINT index_embeddings_pkey TO filesystem_embeddings_pkey;
        END IF;

        -- Fix index names
        IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'index_embeddings_index_name_idx') THEN
            ALTER INDEX index_embeddings_index_name_idx RENAME TO filesystem_embeddings_index_name_idx;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'index_embeddings_embedding_idx') THEN
            ALTER INDEX index_embeddings_embedding_idx RENAME TO filesystem_embeddings_embedding_idx;
        END IF;

    ELSIF index_emb_exists AND fs_emb_exists THEN
        -- Both exist - drop the orphan index_embeddings
        RAISE NOTICE 'Dropping orphaned index_embeddings table';
        DROP TABLE index_embeddings CASCADE;
    END IF;
END $$;

-- Fix index_file_metadata -> filesystem_file_metadata
DO $$
DECLARE
    index_meta_exists BOOLEAN;
    fs_meta_exists BOOLEAN;
BEGIN
    -- Check what tables exist
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'index_file_metadata' AND table_schema = 'public'
    ) INTO index_meta_exists;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'filesystem_file_metadata' AND table_schema = 'public'
    ) INTO fs_meta_exists;

    IF index_meta_exists AND NOT fs_meta_exists THEN
        -- Rename index_file_metadata to filesystem_file_metadata
        RAISE NOTICE 'Renaming index_file_metadata to filesystem_file_metadata';
        ALTER TABLE index_file_metadata RENAME TO filesystem_file_metadata;

        -- Fix primary key constraint name if needed
        IF EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'index_file_metadata_pkey'
        ) THEN
            ALTER TABLE filesystem_file_metadata
            RENAME CONSTRAINT index_file_metadata_pkey TO filesystem_file_metadata_pkey;
        END IF;

        -- Fix index names
        IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'index_file_metadata_index_name_idx') THEN
            ALTER INDEX index_file_metadata_index_name_idx RENAME TO filesystem_file_metadata_index_name_idx;
        END IF;
        IF EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'index_file_metadata_index_name_file_path_key') THEN
            ALTER INDEX index_file_metadata_index_name_file_path_key RENAME TO filesystem_file_metadata_index_name_file_path_key;
        END IF;

    ELSIF index_meta_exists AND fs_meta_exists THEN
        -- Both exist - drop the orphan index_file_metadata
        RAISE NOTICE 'Dropping orphaned index_file_metadata table';
        DROP TABLE index_file_metadata CASCADE;
    END IF;
END $$;

-- Fix any constraint name mismatches on existing tables
-- (e.g., filesystem_embeddings table with index_embeddings_pkey constraint)
DO $$
BEGIN
    -- Fix pkey constraint on filesystem_embeddings if misnamed
    IF EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        WHERE t.relname = 'filesystem_embeddings'
        AND c.conname != 'filesystem_embeddings_pkey'
        AND c.contype = 'p'
    ) THEN
        -- Get the actual constraint name and rename it
        EXECUTE (
            SELECT format('ALTER TABLE filesystem_embeddings RENAME CONSTRAINT %I TO filesystem_embeddings_pkey', c.conname)
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'filesystem_embeddings' AND c.contype = 'p'
            LIMIT 1
        );
    END IF;

    -- Fix pkey constraint on filesystem_file_metadata if misnamed
    IF EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        WHERE t.relname = 'filesystem_file_metadata'
        AND c.conname != 'filesystem_file_metadata_pkey'
        AND c.contype = 'p'
    ) THEN
        EXECUTE (
            SELECT format('ALTER TABLE filesystem_file_metadata RENAME CONSTRAINT %I TO filesystem_file_metadata_pkey', c.conname)
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'filesystem_file_metadata' AND c.contype = 'p'
            LIMIT 1
        );
    END IF;
END $$;

-- Ensure filesystem_embeddings table exists with correct structure
-- (in case neither old nor new table existed)
CREATE TABLE IF NOT EXISTS "filesystem_embeddings" (
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

-- Ensure filesystem_file_metadata table exists with correct structure
CREATE TABLE IF NOT EXISTS "filesystem_file_metadata" (
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

-- Ensure indexes exist
CREATE INDEX IF NOT EXISTS "filesystem_embeddings_index_name_idx"
    ON "filesystem_embeddings"("index_name");

CREATE UNIQUE INDEX IF NOT EXISTS "filesystem_file_metadata_index_name_file_path_key"
    ON "filesystem_file_metadata"("index_name", "file_path");

CREATE INDEX IF NOT EXISTS "filesystem_file_metadata_index_name_idx"
    ON "filesystem_file_metadata"("index_name");
