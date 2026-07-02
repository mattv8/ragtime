DO $$ BEGIN
    EXECUTE format(
        'ALTER TYPE %I ADD VALUE IF NOT EXISTS ''cancelled''',
        (SELECT typname FROM pg_type WHERE typtype = 'e' AND typname ILIKE 'workspacecodeindexjobstatus')
    );
EXCEPTION
    WHEN undefined_object THEN null;
END $$;

ALTER TABLE "app_settings"
  ADD COLUMN IF NOT EXISTS "userspace_code_index_max_concurrency" INTEGER NOT NULL DEFAULT 1;

ALTER TABLE "workspace_code_index_jobs"
  ADD COLUMN IF NOT EXISTS "waiting_for_job_id" TEXT,
  ADD COLUMN IF NOT EXISTS "cancel_requested" BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS "workspace_code_index_jobs_waiting_for_job_id_idx"
  ON "workspace_code_index_jobs"("waiting_for_job_id");
