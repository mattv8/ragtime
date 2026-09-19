CREATE TABLE IF NOT EXISTS "workspace_sqlite_backup_jobs" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "requested_by_id" TEXT,
    "trigger" TEXT NOT NULL,
    "database_names" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "snapshot_id" TEXT,
    "snapshot_git_commit_hash" TEXT,
    "request_key" TEXT NOT NULL,
    "request_hash" TEXT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "available_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "finished_at" TIMESTAMP(3),
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "heartbeat_at" TIMESTAMP(3),
    "owner_token" TEXT,
    "cancel_requested" BOOLEAN NOT NULL DEFAULT FALSE,
    "completed_databases" INTEGER NOT NULL DEFAULT 0,
    "total_databases" INTEGER NOT NULL DEFAULT 0,
    "backup_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "error_message" TEXT,
    CONSTRAINT "workspace_sqlite_backup_jobs_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "workspace_sqlite_backup_jobs_trigger_check" CHECK ("trigger" IN ('manual', 'snapshot', 'scheduled')),
    CONSTRAINT "workspace_sqlite_backup_jobs_status_check" CHECK ("status" IN ('pending', 'running', 'completed', 'failed', 'cancelled', 'interrupted'))
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_workspace_id_request_key_key"
    ON "workspace_sqlite_backup_jobs"("workspace_id", "request_key");
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_one_running_workspace_key"
    ON "workspace_sqlite_backup_jobs"("workspace_id") WHERE "status" = 'running';
CREATE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_status_created_at_idx"
    ON "workspace_sqlite_backup_jobs"("status", "created_at");
CREATE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_workspace_id_created_at_idx"
    ON "workspace_sqlite_backup_jobs"("workspace_id", "created_at");
CREATE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_workspace_id_status_idx"
    ON "workspace_sqlite_backup_jobs"("workspace_id", "status");
CREATE INDEX IF NOT EXISTS "workspace_sqlite_backup_jobs_heartbeat_at_idx"
    ON "workspace_sqlite_backup_jobs"("heartbeat_at");

DO $$ BEGIN
    ALTER TABLE "workspace_sqlite_backup_jobs"
    ADD CONSTRAINT "workspace_sqlite_backup_jobs_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;
