-- Workspace external-agent access tokens
CREATE TABLE IF NOT EXISTS "workspace_agent_access" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "created_by_user_id" TEXT NOT NULL,
    "token" TEXT NOT NULL,
    "label" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "allow_task_submission" BOOLEAN NOT NULL DEFAULT true,
    "last_used_at" TIMESTAMP(3),
    "hit_count" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_agent_access_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_agent_access_token_key"
    ON "workspace_agent_access"("token");
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_agent_access_workspace_id_key"
    ON "workspace_agent_access"("workspace_id");

DO $$ BEGIN
    ALTER TABLE "workspace_agent_access"
        ADD CONSTRAINT "workspace_agent_access_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Idempotency ledger for externally submitted build tasks
CREATE TABLE IF NOT EXISTS "external_build_requests" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "source" TEXT NOT NULL DEFAULT 'workspace_agent',
    "request_id" TEXT NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL DEFAULT '',
    "task_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "external_build_requests_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "external_build_requests_user_id_source_request_id_key"
    ON "external_build_requests"("user_id", "source", "request_id");
CREATE INDEX IF NOT EXISTS "external_build_requests_workspace_id_created_at_idx"
    ON "external_build_requests"("workspace_id", "created_at");
CREATE INDEX IF NOT EXISTS "external_build_requests_user_source_workspace_task_idx"
    ON "external_build_requests"("user_id", "source", "workspace_id", "task_id");
