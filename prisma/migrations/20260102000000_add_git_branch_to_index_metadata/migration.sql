-- Add git_branch and git_token columns for git-based indexes
-- git_branch: tracks which branch was indexed
-- git_token: stores private repo token for re-indexing without re-entering

-- For index_metadata (used after successful indexing)
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "git_branch" TEXT;
ALTER TABLE "index_metadata" ADD COLUMN IF NOT EXISTS "git_token" TEXT;

-- For index_jobs (used during active job processing)
ALTER TABLE "index_jobs" ADD COLUMN IF NOT EXISTS "git_token" TEXT;
