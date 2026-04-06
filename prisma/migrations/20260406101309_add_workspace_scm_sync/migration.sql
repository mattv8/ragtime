ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_git_url" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_git_branch" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_provider" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_repo_visibility" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_token" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_connected_at" TIMESTAMP(3);

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_sync_at" TIMESTAMP(3);

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_sync_direction" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_sync_status" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_sync_message" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_remote_commit_hash" TEXT;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_last_synced_snapshot_id" TEXT;

-- Add remote_commit_hash to snapshots for remote-backed snapshots
-- When set, this indicates the snapshot represents a specific commit on the remote
-- Snapshots created from imports or auto-pushed will have this set to the remote commit hash
ALTER TABLE "userspace_snapshots" ADD COLUMN IF NOT EXISTS "remote_commit_hash" VARCHAR(255);

-- Create index for filtering remote-backed snapshots
CREATE INDEX IF NOT EXISTS "idx_userspace_snapshots_remote_commit" ON "userspace_snapshots"("workspace_id", "remote_commit_hash") WHERE "remote_commit_hash" IS NOT NULL;
