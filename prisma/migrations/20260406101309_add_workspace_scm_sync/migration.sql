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
