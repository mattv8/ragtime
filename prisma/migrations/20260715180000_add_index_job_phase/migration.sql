DO $$ BEGIN
    CREATE TYPE "IndexJobPhase" AS ENUM (
        'preparing',
        'cloning',
        'scanning',
        'loading',
        'chunking',
        'embedding',
        'finalizing',
        'completed',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

ALTER TABLE "index_jobs"
    ADD COLUMN IF NOT EXISTS "phase" "IndexJobPhase" NOT NULL DEFAULT 'preparing';

UPDATE "index_jobs"
SET "phase" = CASE
    WHEN "status" = 'completed' THEN 'completed'::"IndexJobPhase"
    WHEN COALESCE("error_message", '') ILIKE '%cancel%' THEN 'cancelled'::"IndexJobPhase"
    ELSE 'failed'::"IndexJobPhase"
END
WHERE "status" IN ('completed', 'failed');
