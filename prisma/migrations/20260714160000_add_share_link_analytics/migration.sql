-- Add generic public share analytics events plus denormalized hit counters.

ALTER TABLE "workspace_shares"
    ADD COLUMN IF NOT EXISTS "public_hit_count" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "last_public_hit_at" TIMESTAMP(3);

ALTER TABLE "conversation_shares"
    ADD COLUMN IF NOT EXISTS "public_hit_count" INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "last_public_hit_at" TIMESTAMP(3);

CREATE TABLE IF NOT EXISTS "share_link_request_logs" (
    "id" TEXT NOT NULL,
    "share_target_type" TEXT NOT NULL,
    "share_id" TEXT NOT NULL,
    "event_name" TEXT NOT NULL,
    "outcome" TEXT NOT NULL,
    "request_path" TEXT NOT NULL,
    "request_method" TEXT NOT NULL,
    "referrer" TEXT,
    "user_agent" TEXT,
    "authenticated_user_id" TEXT,
    "client_fingerprint" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "share_link_request_logs_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "share_link_req_logs_target_share_created_idx"
    ON "share_link_request_logs" ("share_target_type", "share_id", "created_at");

CREATE INDEX IF NOT EXISTS "share_link_req_logs_event_created_idx"
    ON "share_link_request_logs" ("event_name", "created_at");
