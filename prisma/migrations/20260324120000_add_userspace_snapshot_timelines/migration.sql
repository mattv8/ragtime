ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "current_snapshot_id" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "current_snapshot_branch_id" TEXT;

CREATE TABLE IF NOT EXISTS "userspace_snapshot_branches" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "git_ref_name" TEXT NOT NULL,
    "base_snapshot_id" TEXT,
    "branched_from_snapshot_id" TEXT,
    "is_active" BOOLEAN NOT NULL DEFAULT FALSE,
    "archived_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "userspace_snapshot_branches_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "userspace_snapshots" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "branch_id" TEXT NOT NULL,
    "git_commit_hash" TEXT NOT NULL,
    "message" TEXT,
    "file_count" INTEGER NOT NULL DEFAULT 0,
    "parent_snapshot_id" TEXT,
    "created_by_user_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "userspace_snapshots_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "userspace_snapshot_branches_workspace_id_git_ref_name_key"
ON "userspace_snapshot_branches"("workspace_id", "git_ref_name");

CREATE INDEX IF NOT EXISTS "userspace_snapshot_branches_workspace_id_created_at_idx"
ON "userspace_snapshot_branches"("workspace_id", "created_at");

CREATE UNIQUE INDEX IF NOT EXISTS "userspace_snapshots_workspace_id_branch_id_git_commit_hash_key"
ON "userspace_snapshots"("workspace_id", "branch_id", "git_commit_hash");

CREATE INDEX IF NOT EXISTS "userspace_snapshots_workspace_id_created_at_idx"
ON "userspace_snapshots"("workspace_id", "created_at");

CREATE INDEX IF NOT EXISTS "userspace_snapshots_branch_id_created_at_idx"
ON "userspace_snapshots"("branch_id", "created_at");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'userspace_snapshot_branches_workspace_id_fkey'
    ) THEN
        ALTER TABLE "userspace_snapshot_branches"
        ADD CONSTRAINT "userspace_snapshot_branches_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'userspace_snapshots_workspace_id_fkey'
    ) THEN
        ALTER TABLE "userspace_snapshots"
        ADD CONSTRAINT "userspace_snapshots_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'userspace_snapshots_branch_id_fkey'
    ) THEN
        ALTER TABLE "userspace_snapshots"
        ADD CONSTRAINT "userspace_snapshots_branch_id_fkey"
        FOREIGN KEY ("branch_id") REFERENCES "userspace_snapshot_branches"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'userspace_snapshots_parent_snapshot_id_fkey'
    ) THEN
        ALTER TABLE "userspace_snapshots"
        ADD CONSTRAINT "userspace_snapshots_parent_snapshot_id_fkey"
        FOREIGN KEY ("parent_snapshot_id") REFERENCES "userspace_snapshots"("id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'workspaces_current_snapshot_id_fkey'
    ) THEN
        ALTER TABLE "workspaces"
        ADD CONSTRAINT "workspaces_current_snapshot_id_fkey"
        FOREIGN KEY ("current_snapshot_id") REFERENCES "userspace_snapshots"("id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'workspaces_current_snapshot_branch_id_fkey'
    ) THEN
        ALTER TABLE "workspaces"
        ADD CONSTRAINT "workspaces_current_snapshot_branch_id_fkey"
        FOREIGN KEY ("current_snapshot_branch_id") REFERENCES "userspace_snapshot_branches"("id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;
