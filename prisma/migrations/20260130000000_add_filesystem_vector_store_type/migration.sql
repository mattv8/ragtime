-- Add FilesystemVectorStoreType enum
-- Allows filesystem indexes to choose between pgvector (default) and FAISS storage

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'FilesystemVectorStoreType') THEN
        CREATE TYPE "FilesystemVectorStoreType" AS ENUM ('pgvector', 'faiss');
    END IF;
END
$$;

-- Add vector_store_type enum value if it doesn't exist
-- The actual storage of vector_store_type is in the JSON connection_config column of tool_configs table
-- This enum is for documentation and future potential use in dedicated columns

-- Add OCR concurrency limit setting (max concurrent Ollama vision OCR requests)
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "ocr_concurrency_limit" INTEGER NOT NULL DEFAULT 1;
