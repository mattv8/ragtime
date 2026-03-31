-- Add userspace mount policy fields to tool_configs
ALTER TABLE "tool_configs" ADD COLUMN IF NOT EXISTS "userspace_mounts_enabled" BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE "tool_configs" ADD COLUMN IF NOT EXISTS "userspace_mount_paths" JSONB NOT NULL DEFAULT '[]';

-- Workspace mount attachment table
CREATE TABLE IF NOT EXISTS "workspace_mounts" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "tool_config_id" TEXT NOT NULL,
    "source_path" TEXT NOT NULL,
    "target_path" TEXT NOT NULL,
    "description" TEXT,
    "auto_sync_enabled" BOOLEAN NOT NULL DEFAULT false,
    "sync_status" TEXT NOT NULL DEFAULT 'pending',
    "last_sync_at" TIMESTAMP(3),
    "last_sync_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "workspace_mounts_pkey" PRIMARY KEY ("id")
);

-- Unique constraint: one target path per workspace
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_mounts_workspace_id_target_path_key" ON "workspace_mounts"("workspace_id", "target_path");

-- Indexes
CREATE INDEX IF NOT EXISTS "workspace_mounts_workspace_id_idx" ON "workspace_mounts"("workspace_id");
CREATE INDEX IF NOT EXISTS "workspace_mounts_tool_config_id_idx" ON "workspace_mounts"("tool_config_id");

-- Foreign keys
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'workspace_mounts_workspace_id_fkey') THEN
        ALTER TABLE "workspace_mounts" ADD CONSTRAINT "workspace_mounts_workspace_id_fkey"
            FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'workspace_mounts_tool_config_id_fkey') THEN
        ALTER TABLE "workspace_mounts" ADD CONSTRAINT "workspace_mounts_tool_config_id_fkey"
            FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

-- Add watch mode flag for pre-existing workspace_mounts tables
ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "auto_sync_enabled" BOOLEAN NOT NULL DEFAULT false;
