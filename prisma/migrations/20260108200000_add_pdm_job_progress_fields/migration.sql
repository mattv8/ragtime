-- Add progress tracking fields to pdm_index_jobs table
-- These fields provide more granular progress visibility during PDM indexing

-- Add current_step to show which phase the job is in
ALTER TABLE "pdm_index_jobs" ADD COLUMN IF NOT EXISTS "current_step" TEXT;

-- Add extracted_documents to track extraction progress separately from embedding progress
ALTER TABLE "pdm_index_jobs" ADD COLUMN IF NOT EXISTS "extracted_documents" INTEGER NOT NULL DEFAULT 0;
