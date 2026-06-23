ALTER TABLE "workspaces"
ADD COLUMN IF NOT EXISTS "scm_import_task_id" TEXT,
ADD COLUMN IF NOT EXISTS "scm_import_task_phase" TEXT,
ADD COLUMN IF NOT EXISTS "scm_last_setup_prompt" TEXT;
