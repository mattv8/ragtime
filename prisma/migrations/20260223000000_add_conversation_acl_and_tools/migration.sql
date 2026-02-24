-- Add conversation-level ACL and tool selections

-- Create conversation_members table (conversation ACL)
CREATE TABLE IF NOT EXISTS "conversation_members" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "role" "WorkspaceRole" NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "conversation_members_pkey" PRIMARY KEY ("id")
);

-- Create conversation_tool_selections table
CREATE TABLE IF NOT EXISTS "conversation_tool_selections" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "conversation_tool_selections_pkey" PRIMARY KEY ("id")
);

-- Create unique constraint on (conversation_id, user_id)
CREATE UNIQUE INDEX IF NOT EXISTS "conversation_members_conversation_id_user_id_key"
ON "conversation_members"("conversation_id", "user_id");

-- Create unique constraint on (conversation_id, tool_config_id)
CREATE UNIQUE INDEX IF NOT EXISTS "conversation_tool_selections_conversation_id_tool_config_id_key"
ON "conversation_tool_selections"("conversation_id", "tool_config_id");

-- Create index on user_id for conversation_members
CREATE INDEX IF NOT EXISTS "conversation_members_user_id_idx"
ON "conversation_members"("user_id");

-- Create index on tool_config_id for conversation_tool_selections
CREATE INDEX IF NOT EXISTS "conversation_tool_selections_tool_config_id_idx"
ON "conversation_tool_selections"("tool_config_id");

-- Add foreign key constraints

-- conversation_members -> conversations (CASCADE delete)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversation_members_conversation_id_fkey'
    ) THEN
        ALTER TABLE "conversation_members"
        ADD CONSTRAINT "conversation_members_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- conversation_members -> users (CASCADE delete)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversation_members_user_id_fkey'
    ) THEN
        ALTER TABLE "conversation_members"
        ADD CONSTRAINT "conversation_members_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- conversation_tool_selections -> conversations (CASCADE delete)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversation_tool_selections_conversation_id_fkey'
    ) THEN
        ALTER TABLE "conversation_tool_selections"
        ADD CONSTRAINT "conversation_tool_selections_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- conversation_tool_selections -> tool_configs (CASCADE delete)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversation_tool_selections_tool_config_id_fkey'
    ) THEN
        ALTER TABLE "conversation_tool_selections"
        ADD CONSTRAINT "conversation_tool_selections_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- Backfill existing conversations with owner entries
-- Add conversation_member for each conversation that has a user_id set
DO $$
BEGIN
    -- Only insert if the conversation_id/user_id pair doesn't already exist
    INSERT INTO "conversation_members" ("id", "conversation_id", "user_id", "role", "created_at", "updated_at")
    SELECT
        gen_random_uuid()::text,
        c.id,
        c.user_id,
        'owner'::"WorkspaceRole",
        CURRENT_TIMESTAMP,
        CURRENT_TIMESTAMP
    FROM "conversations" c
    WHERE c.user_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM "conversation_members" cm
          WHERE cm.conversation_id = c.id
            AND cm.user_id = c.user_id
      );
EXCEPTION
    WHEN OTHERS THEN
        -- Safely handle any edge cases (duplicate keys, invalid UUIDs, etc.)
        -- Log the error but don't fail the migration
        RAISE NOTICE 'Backfill encountered error (non-fatal): %', SQLERRM;
END $$;
