ALTER TABLE "conversations"
ADD COLUMN IF NOT EXISTS "workspace_id" TEXT;

CREATE INDEX IF NOT EXISTS "conversations_workspace_id_updated_at_idx"
ON "conversations"("workspace_id", "updated_at");

CREATE INDEX IF NOT EXISTS "conversations_user_id_workspace_id_updated_at_idx"
ON "conversations"("user_id", "workspace_id", "updated_at");
