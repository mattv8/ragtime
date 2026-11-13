ALTER TABLE "index_jobs"
    ADD COLUMN IF NOT EXISTS "warnings" JSONB;

ALTER TABLE "filesystem_index_jobs"
    ADD COLUMN IF NOT EXISTS "warnings" JSONB;
