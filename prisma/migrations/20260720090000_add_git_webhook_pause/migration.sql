ALTER TABLE "index_metadata"
  ADD COLUMN IF NOT EXISTS "webhook_paused" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "workspaces"
  ADD COLUMN IF NOT EXISTS "scm_webhook_paused" BOOLEAN NOT NULL DEFAULT false;
