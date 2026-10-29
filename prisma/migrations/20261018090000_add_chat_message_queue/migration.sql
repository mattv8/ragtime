CREATE TABLE "chat_queued_messages" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "user_id" TEXT,
    "workspace_id" TEXT,
    "message" TEXT NOT NULL,
    "segments" JSONB NOT NULL DEFAULT '[]',
    "request_context" JSONB NOT NULL DEFAULT '{}',
    "seq" BIGSERIAL NOT NULL,
    "dispatch_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "chat_queued_messages_pkey" PRIMARY KEY ("id")
);

ALTER TABLE "chat_queued_messages"
    ADD CONSTRAINT "chat_queued_messages_conversation_id_fkey"
    FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "chat_queued_messages"
    ADD CONSTRAINT "chat_queued_messages_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
CREATE INDEX IF NOT EXISTS "chat_queued_messages_conversation_id_seq_idx"
    ON "chat_queued_messages"("conversation_id", "seq");
