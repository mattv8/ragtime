-- CreateTable
CREATE TABLE IF NOT EXISTS "conversation_branches" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "parent_branch_id" TEXT,
    "branch_point_index" INTEGER NOT NULL,
    "preserved_messages" JSONB NOT NULL DEFAULT '[]',
    "associated_snapshot_id" TEXT,
    "created_by_user_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversation_branches_pkey" PRIMARY KEY ("id")
);

-- AddColumn
ALTER TABLE "conversations" ADD COLUMN IF NOT EXISTS "active_branch_id" TEXT;

-- CreateIndex
CREATE INDEX IF NOT EXISTS "conversation_branches_conversation_id_branch_point_index_cre_idx"
    ON "conversation_branches"("conversation_id", "branch_point_index", "created_at");
CREATE INDEX IF NOT EXISTS "conversation_branches_conversation_id_created_at_idx"
    ON "conversation_branches"("conversation_id", "created_at");

-- AddForeignKey
DO $$ BEGIN
    ALTER TABLE "conversation_branches"
        ADD CONSTRAINT "conversation_branches_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE "conversation_branches"
        ADD CONSTRAINT "conversation_branches_created_by_user_id_fkey"
        FOREIGN KEY ("created_by_user_id") REFERENCES "users"("id")
        ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
