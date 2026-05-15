CREATE TABLE IF NOT EXISTS "conversation_visualization_branches" (
	"id" TEXT NOT NULL,
	"conversation_id" TEXT NOT NULL,
	"message_id" TEXT,
	"message_index" INTEGER NOT NULL,
	"event_index" INTEGER NOT NULL,
	"tool_name" TEXT NOT NULL,
	"sequence" INTEGER NOT NULL,
	"output" TEXT NOT NULL,
	"active" BOOLEAN NOT NULL DEFAULT false,
	"created_by_user_id" TEXT,
	"created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
	"updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

	CONSTRAINT "conversation_visualization_branches_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "conv_viz_branch_msg_id_idx"
	ON "conversation_visualization_branches"("conversation_id", "message_id", "event_index", "sequence");

CREATE INDEX IF NOT EXISTS "conv_viz_branch_msg_index_idx"
	ON "conversation_visualization_branches"("conversation_id", "message_index", "event_index", "sequence");

CREATE INDEX IF NOT EXISTS "conv_viz_branch_active_idx"
	ON "conversation_visualization_branches"("conversation_id", "active");

DO $$
BEGIN
	ALTER TABLE "conversation_visualization_branches"
		ADD CONSTRAINT "conversation_visualization_branches_conversation_id_fkey"
		FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
	ALTER TABLE "conversation_visualization_branches"
		ADD CONSTRAINT "conversation_visualization_branches_created_by_user_id_fkey"
		FOREIGN KEY ("created_by_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;