DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'WorkspaceRole') THEN
        CREATE TYPE "WorkspaceRole" AS ENUM ('owner', 'editor', 'viewer');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS "workspaces" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT DEFAULT '',
    "owner_user_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspaces_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "workspace_members" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "role" "WorkspaceRole" NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_members_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "workspace_tool_selections" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_tool_selections_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_members_workspace_id_user_id_key"
ON "workspace_members"("workspace_id", "user_id");

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_tool_selections_workspace_id_tool_config_id_key"
ON "workspace_tool_selections"("workspace_id", "tool_config_id");

CREATE INDEX IF NOT EXISTS "workspaces_owner_user_id_idx"
ON "workspaces"("owner_user_id");

CREATE INDEX IF NOT EXISTS "workspaces_updated_at_idx"
ON "workspaces"("updated_at");

CREATE INDEX IF NOT EXISTS "workspace_members_user_id_idx"
ON "workspace_members"("user_id");

CREATE INDEX IF NOT EXISTS "workspace_tool_selections_tool_config_id_idx"
ON "workspace_tool_selections"("tool_config_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'workspaces_owner_user_id_fkey'
    ) THEN
        ALTER TABLE "workspaces"
        ADD CONSTRAINT "workspaces_owner_user_id_fkey"
        FOREIGN KEY ("owner_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'workspace_members_workspace_id_fkey'
    ) THEN
        ALTER TABLE "workspace_members"
        ADD CONSTRAINT "workspace_members_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'workspace_members_user_id_fkey'
    ) THEN
        ALTER TABLE "workspace_members"
        ADD CONSTRAINT "workspace_members_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'workspace_tool_selections_workspace_id_fkey'
    ) THEN
        ALTER TABLE "workspace_tool_selections"
        ADD CONSTRAINT "workspace_tool_selections_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'workspace_tool_selections_tool_config_id_fkey'
    ) THEN
        ALTER TABLE "workspace_tool_selections"
        ADD CONSTRAINT "workspace_tool_selections_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    UPDATE "conversations"
    SET "workspace_id" = NULL
    WHERE "workspace_id" IS NOT NULL
      AND "workspace_id" NOT IN (SELECT "id" FROM "workspaces");

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints
        WHERE constraint_name = 'conversations_workspace_id_fkey'
    ) THEN
        ALTER TABLE "conversations"
        ADD CONSTRAINT "conversations_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;
