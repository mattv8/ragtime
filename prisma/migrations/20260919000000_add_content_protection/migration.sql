CREATE TABLE IF NOT EXISTS "content_protection_config" (
  "id" TEXT NOT NULL DEFAULT 'default',
  "revision" INTEGER NOT NULL DEFAULT 0,
  "config" JSONB NOT NULL DEFAULT '{}',
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_by" TEXT,
  CONSTRAINT "content_protection_config_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "content_protection_decisions" (
  "id" TEXT NOT NULL,
  "request_id" TEXT NOT NULL,
  "metadata" JSONB NOT NULL DEFAULT '{}',
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "content_protection_decisions_pkey" PRIMARY KEY ("id")
);
CREATE INDEX IF NOT EXISTS "content_protection_decisions_created_at_idx" ON "content_protection_decisions"("created_at");
CREATE INDEX IF NOT EXISTS "content_protection_decisions_request_id_idx" ON "content_protection_decisions"("request_id");
