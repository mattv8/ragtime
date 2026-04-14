-- CreateEnum: workspace_scm_remote_role
DO $$ BEGIN
  CREATE TYPE "workspace_scm_remote_role" AS ENUM ('upstream', 'publish');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- CreateEnum: workspace_scm_auto_sync_policy
DO $$ BEGIN
  CREATE TYPE "workspace_scm_auto_sync_policy" AS ENUM ('manual', 'auto_push');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Add new SCM policy columns to workspaces
ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_remote_role" "workspace_scm_remote_role";

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_auto_sync_policy" "workspace_scm_auto_sync_policy";

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_sync_paused" BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_sync_paused_reason" TEXT;

-- Backfill: Workspaces that were imported from Git are treated as upstream/manual.
-- Detection: last_sync_direction='import', OR snapshot history contains an "Import from"
-- message (covers cases where a failed auto-push overwrote the direction to 'export').
-- Remaining workspaces with a connected remote keep publish/auto_push.
UPDATE "workspaces"
SET "scm_remote_role" = 'upstream',
    "scm_auto_sync_policy" = 'manual',
    "scm_sync_paused" = CASE
      WHEN "scm_last_sync_status" = 'error' THEN TRUE
      ELSE FALSE
    END,
    "scm_sync_paused_reason" = CASE
      WHEN "scm_last_sync_status" = 'error' THEN 'Sync paused after failed push following import. Review and push manually when ready.'
      ELSE NULL
    END
WHERE "scm_git_url" IS NOT NULL
  AND (
    "scm_last_sync_direction" = 'import'
    OR EXISTS (
      SELECT 1 FROM "userspace_snapshots" s
      WHERE s."workspace_id" = "workspaces"."id"
        AND s."message" LIKE 'Import from %'
    )
  );

UPDATE "workspaces"
SET "scm_remote_role" = 'publish',
    "scm_auto_sync_policy" = 'auto_push'
WHERE "scm_git_url" IS NOT NULL
  AND "scm_remote_role" IS NULL;
