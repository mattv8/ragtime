-- Add persisted auto-pull toggle for upstream SCM workspaces.
ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_auto_pull_enabled" BOOLEAN NOT NULL DEFAULT false;

-- Add configurable SCM auto-sync intervals for push and pull jobs.
ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_auto_push_interval_seconds" INTEGER NOT NULL DEFAULT 3600,
ADD COLUMN IF NOT EXISTS "scm_auto_pull_interval_seconds" INTEGER NOT NULL DEFAULT 3600;

-- Clamp any legacy/invalid values into the supported slider range (5 minutes to 30 days).
UPDATE "workspaces"
SET "scm_auto_push_interval_seconds" = GREATEST(300, LEAST(2592000, "scm_auto_push_interval_seconds")),
	"scm_auto_pull_interval_seconds" = GREATEST(300, LEAST(2592000, "scm_auto_pull_interval_seconds"));
