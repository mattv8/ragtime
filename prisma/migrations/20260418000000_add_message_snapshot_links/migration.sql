-- CreateTable
CREATE TABLE IF NOT EXISTS "conversation_message_snapshot_links" (
  "id" TEXT NOT NULL,
  "conversation_id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "message_id" TEXT NOT NULL,
  "snapshot_id" TEXT NOT NULL,
  "restore_message_count" INTEGER NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "conversation_message_snapshot_links_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "conversation_message_snapshot_links_conversation_id_message_id_key"
ON "conversation_message_snapshot_links"("conversation_id", "message_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "conversation_message_snapshot_links_workspace_id_idx"
ON "conversation_message_snapshot_links"("workspace_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "conversation_message_snapshot_links_snapshot_id_idx"
ON "conversation_message_snapshot_links"("snapshot_id");

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "conversation_message_snapshot_links"
  ADD CONSTRAINT "conversation_message_snapshot_links_conversation_id_fkey"
  FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "conversation_message_snapshot_links"
  ADD CONSTRAINT "conversation_message_snapshot_links_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "conversation_message_snapshot_links"
  ADD CONSTRAINT "conversation_message_snapshot_links_snapshot_id_fkey"
  FOREIGN KEY ("snapshot_id") REFERENCES "userspace_snapshots"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
