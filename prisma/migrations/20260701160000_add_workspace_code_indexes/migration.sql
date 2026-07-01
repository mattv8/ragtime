-- User Space code index global settings
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "userspace_code_index_enabled" BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "userspace_code_index_debounce_seconds" INTEGER NOT NULL DEFAULT 2;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "userspace_code_index_reconcile_interval_seconds" INTEGER NOT NULL DEFAULT 300;
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "userspace_code_index_max_attempts" INTEGER NOT NULL DEFAULT 3;

DO $$ BEGIN
    CREATE TYPE "WorkspaceCodeIndexStatus" AS ENUM ('pending', 'indexing', 'ready', 'stale', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS "workspace_code_index_states" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "status" "WorkspaceCodeIndexStatus" NOT NULL DEFAULT 'pending',
    "last_indexed_at" TIMESTAMP(3),
    "last_reconciled_at" TIMESTAMP(3),
    "embedding_config_hash" TEXT,
    "embedding_dimension" INTEGER,
    "file_count" INTEGER NOT NULL DEFAULT 0,
    "chunk_count" INTEGER NOT NULL DEFAULT 0,
    "symbol_count" INTEGER NOT NULL DEFAULT 0,
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_code_index_states_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_code_index_states_workspace_id_key"
    ON "workspace_code_index_states"("workspace_id");
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_code_index_states_index_name_key"
    ON "workspace_code_index_states"("index_name");
CREATE INDEX IF NOT EXISTS "workspace_code_index_states_status_idx"
    ON "workspace_code_index_states"("status");

-- Remove durable progress columns that now live on canonical job rows.
ALTER TABLE "workspace_code_index_states"
    DROP COLUMN IF EXISTS "progress_percent",
    DROP COLUMN IF EXISTS "total_files",
    DROP COLUMN IF EXISTS "processed_files",
    DROP COLUMN IF EXISTS "current_file";

CREATE TABLE IF NOT EXISTS "workspace_code_index_dirty_paths" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "path" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "dirty_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "attempt_count" INTEGER NOT NULL DEFAULT 0,
    "last_error" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_code_index_dirty_paths_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_code_index_dirty_paths_workspace_id_path_key"
    ON "workspace_code_index_dirty_paths"("workspace_id", "path");
CREATE INDEX IF NOT EXISTS "workspace_code_index_dirty_paths_workspace_id_dirty_at_idx"
    ON "workspace_code_index_dirty_paths"("workspace_id", "dirty_at");

CREATE TABLE IF NOT EXISTS "workspace_code_symbols" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "path" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "signature" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_code_symbols_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "workspace_code_symbols_workspace_id_name_idx"
    ON "workspace_code_symbols"("workspace_id", "name");
CREATE INDEX IF NOT EXISTS "workspace_code_symbols_workspace_id_path_idx"
    ON "workspace_code_symbols"("workspace_id", "path");

DO $$ BEGIN
    CREATE TYPE "WorkspaceCodeIndexJobStatus" AS ENUM ('pending', 'indexing', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE "WorkspaceCodeIndexJobPhase" AS ENUM (
        'collecting',
        'loading_files',
        'chunking',
        'embedding',
        'indexing_symbols',
        'finalizing'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

CREATE TABLE IF NOT EXISTS "workspace_code_index_jobs" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "index_name" TEXT NOT NULL,
    "status" "WorkspaceCodeIndexJobStatus" NOT NULL DEFAULT 'pending',
    "phase" "WorkspaceCodeIndexJobPhase" NOT NULL DEFAULT 'collecting',
    "total_files" INTEGER NOT NULL DEFAULT 0,
    "processed_files" INTEGER NOT NULL DEFAULT 0,
    "total_chunks" INTEGER NOT NULL DEFAULT 0,
    "processed_chunks" INTEGER NOT NULL DEFAULT 0,
    "current_file" TEXT,
    "error_message" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    CONSTRAINT "workspace_code_index_jobs_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "workspace_code_index_jobs_workspace_id_idx"
    ON "workspace_code_index_jobs"("workspace_id");
CREATE INDEX IF NOT EXISTS "workspace_code_index_jobs_status_idx"
    ON "workspace_code_index_jobs"("status");
CREATE INDEX IF NOT EXISTS "workspace_code_index_jobs_created_at_idx"
    ON "workspace_code_index_jobs"("created_at" DESC);

DO $$ BEGIN
    ALTER TABLE "workspace_code_index_states"
    ADD CONSTRAINT "workspace_code_index_states_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "workspace_code_index_dirty_paths"
    ADD CONSTRAINT "workspace_code_index_dirty_paths_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "workspace_code_symbols"
    ADD CONSTRAINT "workspace_code_symbols_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    ALTER TABLE "workspace_code_index_jobs"
    ADD CONSTRAINT "workspace_code_index_jobs_workspace_id_fkey"
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;
