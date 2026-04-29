DO $$ BEGIN
    CREATE TYPE "ConversationBranchKind" AS ENUM ('edit', 'delete', 'replay');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE "conversation_branches"
    ADD COLUMN IF NOT EXISTS "branch_kind" "ConversationBranchKind";