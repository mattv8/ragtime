-- Add sync_mode enum to workspace mounts and backfill legacy sync_deletes.
-- Idempotent for existing development environments.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type WHERE typname = 'WorkspaceMountSyncMode'
    ) THEN
        CREATE TYPE "WorkspaceMountSyncMode" AS ENUM (
            'merge',
            'source_authoritative',
            'target_authoritative'
        );
    END IF;
END $$;

ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "sync_mode" "WorkspaceMountSyncMode" NOT NULL DEFAULT 'merge';

ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "destructive_auto_sync_confirmed_at" TIMESTAMP(3);

ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "destructive_auto_sync_confirmed_mode" "WorkspaceMountSyncMode";

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'workspace_mounts' AND column_name = 'sync_deletes'
    ) THEN
        UPDATE "workspace_mounts"
        SET "sync_mode" = CASE
            WHEN COALESCE("sync_deletes", false) THEN 'source_authoritative'::"WorkspaceMountSyncMode"
            ELSE 'merge'::"WorkspaceMountSyncMode"
        END;

        UPDATE "workspace_mounts"
        SET "auto_sync_enabled" = false
        WHERE COALESCE("sync_deletes", false);

        ALTER TABLE "workspace_mounts"
            DROP COLUMN IF EXISTS "sync_deletes";
    END IF;
END $$;
