-- Add display_name column to index_metadata for human-readable names
-- The name column contains the safe tool name, display_name stores the original user input

ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "display_name" TEXT;
