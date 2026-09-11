ALTER TABLE "conversation_branches"
    ADD COLUMN IF NOT EXISTS "base_messages" JSONB;
