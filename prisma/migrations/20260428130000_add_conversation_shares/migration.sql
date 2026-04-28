DO $$ BEGIN
    CREATE TYPE "ConversationShareRole" AS ENUM ('viewer', 'editor');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS "conversation_shares" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "owner_user_id" TEXT NOT NULL,
    "label" TEXT,
    "share_token" TEXT,
    "share_token_created_at" TIMESTAMP(3),
    "share_slug" TEXT,
    "share_access_mode" TEXT NOT NULL DEFAULT 'token',
    "share_password" TEXT,
    "share_selected_user_ids" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "share_selected_ldap_groups" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "granted_role" "ConversationShareRole" NOT NULL DEFAULT 'viewer',
    "scope_anchor_message_idx" INTEGER,
    "scope_direction" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "conversation_shares_pkey" PRIMARY KEY ("id")
);

-- Multiple share links per conversation are allowed; the conversation_id column
-- is intentionally non-unique. shareToken remains globally unique.
CREATE INDEX IF NOT EXISTS "conversation_shares_conversation_id_idx"
    ON "conversation_shares"("conversation_id");

CREATE UNIQUE INDEX IF NOT EXISTS "conversation_shares_share_token_key"
    ON "conversation_shares"("share_token");

CREATE INDEX IF NOT EXISTS "conversation_shares_owner_user_id_share_slug_idx"
    ON "conversation_shares"("owner_user_id", "share_slug");

CREATE INDEX IF NOT EXISTS "conversation_shares_conversation_id_share_token_idx"
    ON "conversation_shares"("conversation_id", "share_token");

-- If an older variant of this migration ran first with a unique constraint on
-- conversation_id, drop it so additional links can be created.
DROP INDEX IF EXISTS "conversation_shares_conversation_id_key";

-- Backfill the new columns when re-running this migration on a database that
-- already has the original schema.
ALTER TABLE "conversation_shares" ADD COLUMN IF NOT EXISTS "label" TEXT;
ALTER TABLE "conversation_shares" ADD COLUMN IF NOT EXISTS "scope_anchor_message_idx" INTEGER;
ALTER TABLE "conversation_shares" ADD COLUMN IF NOT EXISTS "scope_direction" TEXT;

DO $$ BEGIN
    ALTER TABLE "conversation_shares"
        ADD CONSTRAINT "conversation_shares_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE "conversation_shares"
        ADD CONSTRAINT "conversation_shares_owner_user_id_fkey"
        FOREIGN KEY ("owner_user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
