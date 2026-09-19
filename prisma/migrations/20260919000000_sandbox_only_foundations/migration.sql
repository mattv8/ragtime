-- Hosted-execution policy and external workspace development foundations.
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "hosted_chat_enabled" BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "hosted_chat_enabled" BOOLEAN;

CREATE TABLE IF NOT EXISTS "workspace_development_credentials" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "selector" TEXT NOT NULL,
  "token_hash" TEXT NOT NULL,
  "name" TEXT NOT NULL,
  "scopes" JSONB NOT NULL DEFAULT '["read", "write", "exec"]',
  "expires_at" TIMESTAMP(3),
  "revoked_at" TIMESTAMP(3),
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "workspace_development_credentials_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_development_credentials_selector_key" ON "workspace_development_credentials"("selector");
CREATE INDEX IF NOT EXISTS "workspace_development_credentials_workspace_id_idx" ON "workspace_development_credentials"("workspace_id");
CREATE INDEX IF NOT EXISTS "workspace_development_credentials_user_id_idx" ON "workspace_development_credentials"("user_id");
CREATE INDEX IF NOT EXISTS "workspace_development_credentials_expires_at_idx" ON "workspace_development_credentials"("expires_at");
DO $$ BEGIN
  ALTER TABLE "workspace_development_credentials" ADD CONSTRAINT "workspace_development_credentials_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  ALTER TABLE "workspace_development_credentials" ADD CONSTRAINT "workspace_development_credentials_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS "workspace_index_grants" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "index_name" TEXT NOT NULL,
  "created_by_id" TEXT NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "workspace_index_grants_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_index_grants_workspace_id_index_name_key" ON "workspace_index_grants"("workspace_id", "index_name");
CREATE INDEX IF NOT EXISTS "workspace_index_grants_created_by_id_idx" ON "workspace_index_grants"("created_by_id");
DO $$ BEGIN
  ALTER TABLE "workspace_index_grants" ADD CONSTRAINT "workspace_index_grants_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  ALTER TABLE "workspace_index_grants" ADD CONSTRAINT "workspace_index_grants_created_by_id_fkey" FOREIGN KEY ("created_by_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
