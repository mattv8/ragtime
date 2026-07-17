DO $$
BEGIN
    CREATE TYPE "GitWebhookTargetType" AS ENUM ('git_index', 'workspace_scm');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    CREATE TYPE "GitWebhookDeliveryStatus" AS ENUM ('pending', 'processing', 'completed', 'failed', 'skipped', 'ignored');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE "index_metadata"
    ADD COLUMN IF NOT EXISTS "webhook_id" TEXT,
    ADD COLUMN IF NOT EXISTS "webhook_secret" TEXT,
    ADD COLUMN IF NOT EXISTS "webhook_created_at" TIMESTAMP(3);

ALTER TABLE "workspaces"
    ADD COLUMN IF NOT EXISTS "scm_webhook_id" TEXT,
    ADD COLUMN IF NOT EXISTS "scm_webhook_secret" TEXT,
    ADD COLUMN IF NOT EXISTS "scm_webhook_created_at" TIMESTAMP(3);

CREATE TABLE IF NOT EXISTS "git_webhook_deliveries" (
    "id" TEXT NOT NULL,
    "target_type" "GitWebhookTargetType" NOT NULL,
    "index_metadata_id" TEXT,
    "workspace_id" TEXT,
    "provider_delivery_id" TEXT,
    "event_name" TEXT NOT NULL,
    "branch" TEXT,
    "head_commit" TEXT,
    "status" "GitWebhookDeliveryStatus" NOT NULL DEFAULT 'pending',
    "index_job_id" TEXT,
    "message" TEXT,
    "received_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "started_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    CONSTRAINT "git_webhook_deliveries_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "index_metadata_webhook_id_key" ON "index_metadata"("webhook_id");
CREATE UNIQUE INDEX IF NOT EXISTS "workspaces_scm_webhook_id_key" ON "workspaces"("scm_webhook_id");
CREATE INDEX IF NOT EXISTS "git_webhook_deliveries_index_metadata_id_received_at_idx" ON "git_webhook_deliveries"("index_metadata_id", "received_at");
CREATE INDEX IF NOT EXISTS "git_webhook_deliveries_workspace_id_received_at_idx" ON "git_webhook_deliveries"("workspace_id", "received_at");
CREATE INDEX IF NOT EXISTS "git_webhook_deliveries_status_idx" ON "git_webhook_deliveries"("status");

DO $$
BEGIN
  ALTER TABLE "git_webhook_deliveries"
  ADD CONSTRAINT "git_webhook_delivery_one_target"
  CHECK (
    ("target_type" = 'git_index' AND "index_metadata_id" IS NOT NULL AND "workspace_id" IS NULL)
    OR
    ("target_type" = 'workspace_scm' AND "workspace_id" IS NOT NULL AND "index_metadata_id" IS NULL)
  );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_one_pending_index"
ON "git_webhook_deliveries" ("index_metadata_id")
WHERE "status" = 'pending' AND "index_metadata_id" IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_one_processing_index"
ON "git_webhook_deliveries" ("index_metadata_id")
WHERE "status" = 'processing' AND "index_metadata_id" IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_one_pending_workspace"
ON "git_webhook_deliveries" ("workspace_id")
WHERE "status" = 'pending' AND "workspace_id" IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_one_processing_workspace"
ON "git_webhook_deliveries" ("workspace_id")
WHERE "status" = 'processing' AND "workspace_id" IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_provider_delivery_index"
ON "git_webhook_deliveries" ("index_metadata_id", "provider_delivery_id")
WHERE "provider_delivery_id" IS NOT NULL AND "index_metadata_id" IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS "git_webhook_provider_delivery_workspace"
ON "git_webhook_deliveries" ("workspace_id", "provider_delivery_id")
WHERE "provider_delivery_id" IS NOT NULL AND "workspace_id" IS NOT NULL;

DO $$
BEGIN
    ALTER TABLE "git_webhook_deliveries"
        ADD CONSTRAINT "git_webhook_deliveries_index_metadata_id_fkey"
        FOREIGN KEY ("index_metadata_id") REFERENCES "index_metadata"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "git_webhook_deliveries"
        ADD CONSTRAINT "git_webhook_deliveries_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
