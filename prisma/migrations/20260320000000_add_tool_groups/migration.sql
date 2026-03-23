-- Tool Groups: named folders for organising tool configs
CREATE TABLE IF NOT EXISTS "tool_groups" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "sort_order" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "tool_groups_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "tool_groups_name_key" ON "tool_groups"("name");

-- Add optional group membership to tool_configs
ALTER TABLE "tool_configs" ADD COLUMN IF NOT EXISTS "group_id" TEXT;

-- FK from tool_configs -> tool_groups (SET NULL on delete)
DO $$ BEGIN
    ALTER TABLE "tool_configs"
        ADD CONSTRAINT "tool_configs_group_id_fkey"
        FOREIGN KEY ("group_id") REFERENCES "tool_groups"("id")
        ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS "tool_configs_group_id_idx" ON "tool_configs"("group_id");

-- Workspace-selected tool groups junction
CREATE TABLE IF NOT EXISTS "workspace_tool_group_selections" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "tool_group_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_tool_group_selections_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_tool_group_selections_workspace_id_tool_group_id_key"
    ON "workspace_tool_group_selections"("workspace_id", "tool_group_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_group_selections_tool_group_id_idx"
    ON "workspace_tool_group_selections"("tool_group_id");

DO $$ BEGIN
    ALTER TABLE "workspace_tool_group_selections"
        ADD CONSTRAINT "workspace_tool_group_selections_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE "workspace_tool_group_selections"
        ADD CONSTRAINT "workspace_tool_group_selections_tool_group_id_fkey"
        FOREIGN KEY ("tool_group_id") REFERENCES "tool_groups"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Conversation-selected tool groups junction
CREATE TABLE IF NOT EXISTS "conversation_tool_group_selections" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "tool_group_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "conversation_tool_group_selections_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "conversation_tool_group_selections_conversation_id_tool_group_key"
    ON "conversation_tool_group_selections"("conversation_id", "tool_group_id");

CREATE INDEX IF NOT EXISTS "conversation_tool_group_selections_tool_group_id_idx"
    ON "conversation_tool_group_selections"("tool_group_id");

DO $$ BEGIN
    ALTER TABLE "conversation_tool_group_selections"
        ADD CONSTRAINT "conversation_tool_group_selections_conversation_id_fkey"
        FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE "conversation_tool_group_selections"
        ADD CONSTRAINT "conversation_tool_group_selections_tool_group_id_fkey"
        FOREIGN KEY ("tool_group_id") REFERENCES "tool_groups"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
