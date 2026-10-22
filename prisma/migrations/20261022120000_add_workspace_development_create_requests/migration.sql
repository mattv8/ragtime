CREATE TABLE IF NOT EXISTS "workspace_development_create_requests" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "state" TEXT NOT NULL DEFAULT 'accepted',
    "error_code" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),
    CONSTRAINT "workspace_development_create_requests_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_development_create_requests_workspace_id_key" ON "workspace_development_create_requests"("workspace_id");
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_development_create_requests_user_id_idempotency_k_key" ON "workspace_development_create_requests"("user_id", "idempotency_key");
CREATE INDEX IF NOT EXISTS "workspace_development_create_requests_user_id_created_at_idx" ON "workspace_development_create_requests"("user_id", "created_at");
