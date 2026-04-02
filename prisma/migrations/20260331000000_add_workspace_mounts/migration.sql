-- ===========================================================================
-- Combined workspace mounts migration
-- Creates userspace_mount_sources + workspace_mounts tables, migrates
-- legacy tool_configs mount fields, and adds all per-mount flags.
-- Fully idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS throughout).
-- ===========================================================================

-- 1. Enum for mount source types
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'UserspaceMountSourceType') THEN
        CREATE TYPE "UserspaceMountSourceType" AS ENUM ('ssh', 'filesystem');
    END IF;
END $$;

-- 2. Mount sources table (admin-managed)
CREATE TABLE IF NOT EXISTS "userspace_mount_sources" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT DEFAULT '',
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "source_type" "UserspaceMountSourceType" NOT NULL,
    "tool_config_id" TEXT,
    "connection_config" JSONB NOT NULL,
    "approved_paths" JSONB NOT NULL DEFAULT '[]',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "userspace_mount_sources_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "userspace_mount_sources_enabled_source_type_idx"
    ON "userspace_mount_sources"("enabled", "source_type");

CREATE INDEX IF NOT EXISTS "userspace_mount_sources_tool_config_id_idx"
    ON "userspace_mount_sources"("tool_config_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'userspace_mount_sources_tool_config_id_fkey'
          AND table_name = 'userspace_mount_sources'
    ) THEN
        ALTER TABLE "userspace_mount_sources"
            ADD CONSTRAINT "userspace_mount_sources_tool_config_id_fkey"
            FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id")
            ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

-- 3. Workspace mounts table (per-workspace attachment)
CREATE TABLE IF NOT EXISTS "workspace_mounts" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "mount_source_id" TEXT NOT NULL,
    "source_path" TEXT NOT NULL,
    "target_path" TEXT NOT NULL,
    "description" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "auto_sync_enabled" BOOLEAN NOT NULL DEFAULT false,
    "sync_deletes" BOOLEAN NOT NULL DEFAULT false,
    "sync_status" TEXT NOT NULL DEFAULT 'pending',
    "last_sync_at" TIMESTAMP(3),
    "last_sync_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "workspace_mounts_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_mounts_workspace_id_target_path_key"
    ON "workspace_mounts"("workspace_id", "target_path");

CREATE INDEX IF NOT EXISTS "workspace_mounts_workspace_id_idx"
    ON "workspace_mounts"("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_mounts_mount_source_id_idx"
    ON "workspace_mounts"("mount_source_id");

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'workspace_mounts_workspace_id_fkey') THEN
        ALTER TABLE "workspace_mounts"
            ADD CONSTRAINT "workspace_mounts_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id")
            ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'workspace_mounts_mount_source_id_fkey') THEN
        ALTER TABLE "workspace_mounts"
            ADD CONSTRAINT "workspace_mounts_mount_source_id_fkey"
            FOREIGN KEY ("mount_source_id") REFERENCES "userspace_mount_sources"("id")
            ON DELETE RESTRICT ON UPDATE CASCADE;
    END IF;
END $$;

-- 4. Ensure all columns exist for pre-existing tables (idempotent adds)
ALTER TABLE "workspace_mounts" ADD COLUMN IF NOT EXISTS "description" TEXT;
ALTER TABLE "workspace_mounts" ADD COLUMN IF NOT EXISTS "enabled" BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE "workspace_mounts" ADD COLUMN IF NOT EXISTS "auto_sync_enabled" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "workspace_mounts" ADD COLUMN IF NOT EXISTS "sync_deletes" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "userspace_mount_sources" ADD COLUMN IF NOT EXISTS "tool_config_id" TEXT;

-- 5. Legacy migration: copy eligible tool_configs into userspace_mount_sources
--    (only runs if the legacy columns still exist on tool_configs)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tool_configs' AND column_name = 'userspace_mounts_enabled'
    ) THEN
        INSERT INTO "userspace_mount_sources" (
            "id", "name", "description", "enabled", "source_type",
            "connection_config", "approved_paths", "created_at", "updated_at"
        )
        SELECT
            tc."id", tc."name", '',
            CASE
                WHEN tc."userspace_mounts_enabled" THEN true
                WHEN EXISTS (
                    SELECT 1 FROM "workspace_mounts" wm
                    WHERE wm."tool_config_id" = tc."id"
                ) THEN true
                ELSE false
            END,
            CASE
                WHEN tc."tool_type" = 'ssh_shell' THEN 'ssh'::"UserspaceMountSourceType"
                ELSE 'filesystem'::"UserspaceMountSourceType"
            END,
            tc."connection_config",
            CASE
                WHEN jsonb_typeof(tc."userspace_mount_paths") = 'array'
                    AND jsonb_array_length(tc."userspace_mount_paths") > 0
                THEN tc."userspace_mount_paths"
                ELSE '["."]'::jsonb
            END,
            COALESCE(tc."created_at", CURRENT_TIMESTAMP),
            COALESCE(tc."updated_at", CURRENT_TIMESTAMP)
        FROM "tool_configs" tc
        WHERE (
                tc."userspace_mounts_enabled" = true
                OR (
                    jsonb_typeof(tc."userspace_mount_paths") = 'array'
                    AND jsonb_array_length(tc."userspace_mount_paths") > 0
                )
                OR EXISTS (
                    SELECT 1 FROM "workspace_mounts" wm
                    WHERE wm."tool_config_id" = tc."id"
                )
            )
            AND NOT EXISTS (
                SELECT 1 FROM "userspace_mount_sources" ums
                WHERE ums."id" = tc."id"
            );

        -- Backfill mount_source_id from legacy tool_config_id
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'workspace_mounts' AND column_name = 'tool_config_id'
        ) THEN
            UPDATE "workspace_mounts"
            SET "mount_source_id" = "tool_config_id"
            WHERE "mount_source_id" IS NULL;

            ALTER TABLE "workspace_mounts"
                DROP CONSTRAINT IF EXISTS "workspace_mounts_tool_config_id_fkey";
            DROP INDEX IF EXISTS "workspace_mounts_tool_config_id_idx";
            ALTER TABLE "workspace_mounts"
                DROP COLUMN IF EXISTS "tool_config_id";
        END IF;

        -- Drop legacy columns from tool_configs
        ALTER TABLE "tool_configs"
            DROP COLUMN IF EXISTS "userspace_mounts_enabled";
        ALTER TABLE "tool_configs"
            DROP COLUMN IF EXISTS "userspace_mount_paths";
    END IF;
END $$;
