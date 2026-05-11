-- Add restricted-by-default ACLs for admin-managed userspace mount sources.
ALTER TABLE "userspace_mount_sources"
  ADD COLUMN IF NOT EXISTS "access_user_ids" JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS "access_group_identifiers" JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Preserve access for mount sources that were already attached to workspaces by
-- granting the workspace owner and current editor members access to those sources.
WITH existing_access AS (
  SELECT wm."mount_source_id", w."owner_user_id" AS user_id
  FROM "workspace_mounts" wm
  JOIN "workspaces" w ON w."id" = wm."workspace_id"
  WHERE wm."mount_source_id" IS NOT NULL

  UNION

  SELECT wm."mount_source_id", m."user_id" AS user_id
  FROM "workspace_mounts" wm
  JOIN "workspace_members" m ON m."workspace_id" = wm."workspace_id"
  WHERE wm."mount_source_id" IS NOT NULL
    AND m."role" IN ('owner', 'editor')
), aggregated AS (
  SELECT
    "mount_source_id",
    jsonb_agg(DISTINCT user_id ORDER BY user_id) AS access_user_ids
  FROM existing_access
  WHERE user_id IS NOT NULL AND user_id <> ''
  GROUP BY "mount_source_id"
)
UPDATE "userspace_mount_sources" s
SET "access_user_ids" = aggregated.access_user_ids,
    "updated_at" = CURRENT_TIMESTAMP
FROM aggregated
WHERE s."id" = aggregated."mount_source_id";

-- Add persisted workspace mount sync progress fields.
ALTER TABLE "workspace_mounts"
  ADD COLUMN IF NOT EXISTS "sync_progress_files_done" INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS "sync_progress_files_total" INTEGER,
  ADD COLUMN IF NOT EXISTS "sync_progress_message" TEXT,
  ADD COLUMN IF NOT EXISTS "sync_started_at" TIMESTAMP(3);

UPDATE "workspace_mounts"
SET "sync_progress_files_done" = 0
WHERE "sync_progress_files_done" IS NULL;
