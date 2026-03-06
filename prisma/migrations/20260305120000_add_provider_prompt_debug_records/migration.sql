-- Create table for DEBUG-mode prompt input capture at provider call boundaries.
CREATE TABLE IF NOT EXISTS "provider_prompt_debug_records" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "chat_task_id" TEXT,
    "user_id" TEXT,
    "provider" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "mode" TEXT NOT NULL,
    "request_kind" TEXT NOT NULL,
    "rendered_system_prompt" TEXT NOT NULL,
    "rendered_user_input" TEXT NOT NULL,
    "rendered_provider_messages" JSONB NOT NULL,
    "rendered_chat_history" JSONB NOT NULL,
    "tool_scope_prompt" TEXT NOT NULL DEFAULT '',
    "prompt_additions" TEXT NOT NULL DEFAULT '',
    "turn_reminders" TEXT NOT NULL DEFAULT '',
    "message_index" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "provider_prompt_debug_records_pkey" PRIMARY KEY ("id")
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'provider_prompt_debug_records_conversation_id_fkey'
          AND table_name = 'provider_prompt_debug_records'
    ) THEN
        ALTER TABLE "provider_prompt_debug_records"
        ADD CONSTRAINT "provider_prompt_debug_records_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'provider_prompt_debug_records_chat_task_id_fkey'
          AND table_name = 'provider_prompt_debug_records'
    ) THEN
        ALTER TABLE "provider_prompt_debug_records"
        ADD CONSTRAINT "provider_prompt_debug_records_chat_task_id_fkey"
        FOREIGN KEY ("chat_task_id") REFERENCES "chat_tasks"("id")
        ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'provider_prompt_debug_records_user_id_fkey'
          AND table_name = 'provider_prompt_debug_records'
    ) THEN
        ALTER TABLE "provider_prompt_debug_records"
        ADD CONSTRAINT "provider_prompt_debug_records_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS "provider_prompt_debug_records_conversation_id_created_at_idx"
    ON "provider_prompt_debug_records"("conversation_id", "created_at");

CREATE INDEX IF NOT EXISTS "provider_prompt_debug_records_chat_task_id_created_at_idx"
    ON "provider_prompt_debug_records"("chat_task_id", "created_at");

CREATE INDEX IF NOT EXISTS "provider_prompt_debug_records_user_id_created_at_idx"
    ON "provider_prompt_debug_records"("user_id", "created_at");

CREATE INDEX IF NOT EXISTS "provider_prompt_debug_records_created_at_idx"
    ON "provider_prompt_debug_records"("created_at");
