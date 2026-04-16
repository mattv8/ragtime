-- CreateEnum
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'usage_attempt_status') THEN
        CREATE TYPE "usage_attempt_status" AS ENUM ('started', 'completed', 'failed', 'cancelled', 'interrupted');
    END IF;
END $$;

-- CreateTable
CREATE TABLE IF NOT EXISTS "user_usage_attempts" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "conversation_id" TEXT,
    "chat_task_id" TEXT,
    "request_source" TEXT NOT NULL,
    "provider" TEXT NOT NULL DEFAULT '',
    "model" TEXT NOT NULL DEFAULT '',
    "request_count" INTEGER NOT NULL DEFAULT 1,
    "input_tokens" INTEGER NOT NULL DEFAULT 0,
    "output_tokens" INTEGER NOT NULL DEFAULT 0,
    "total_tokens" INTEGER NOT NULL DEFAULT 0,
    "tokens_estimated" BOOLEAN NOT NULL DEFAULT true,
    "status" "usage_attempt_status" NOT NULL DEFAULT 'started',
    "failure_reason" TEXT,
    "started_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finalized_at" TIMESTAMP(3),

    CONSTRAINT "user_usage_attempts_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_usage_attempts_user_id_fkey'
          AND table_name = 'user_usage_attempts'
    ) THEN
        ALTER TABLE "user_usage_attempts"
        ADD CONSTRAINT "user_usage_attempts_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- CreateIndexes
CREATE INDEX IF NOT EXISTS "user_usage_attempts_user_id_started_at_idx"
    ON "user_usage_attempts"("user_id", "started_at");
CREATE INDEX IF NOT EXISTS "user_usage_attempts_status_started_at_idx"
    ON "user_usage_attempts"("status", "started_at");
CREATE INDEX IF NOT EXISTS "user_usage_attempts_provider_started_at_idx"
    ON "user_usage_attempts"("provider", "started_at");
CREATE INDEX IF NOT EXISTS "user_usage_attempts_started_at_idx"
    ON "user_usage_attempts"("started_at");
CREATE INDEX IF NOT EXISTS "user_usage_attempts_chat_task_id_idx"
    ON "user_usage_attempts"("chat_task_id");
