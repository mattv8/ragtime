CREATE TABLE IF NOT EXISTS "workspace_published_endpoints" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "key" TEXT NOT NULL,
  "label" TEXT NOT NULL,
  "description" TEXT,
  "method" TEXT NOT NULL,
  "path" TEXT NOT NULL,
  "definition_hash" TEXT NOT NULL,
  "enabled" BOOLEAN NOT NULL DEFAULT TRUE,
  "approved_by_user_id" TEXT,
  "approved_at" TIMESTAMP(3) NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "workspace_published_endpoints_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_published_endpoints_workspace_id_key_key"
ON "workspace_published_endpoints"("workspace_id", "key");

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_published_endpoints_workspace_id_method_path_key"
ON "workspace_published_endpoints"("workspace_id", "method", "path");

CREATE INDEX IF NOT EXISTS "workspace_published_endpoints_workspace_id_idx"
ON "workspace_published_endpoints"("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_published_endpoints_approved_by_user_id_idx"
ON "workspace_published_endpoints"("approved_by_user_id");

DO $$
BEGIN
  ALTER TABLE "workspace_published_endpoints"
  ADD CONSTRAINT "workspace_published_endpoints_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  ALTER TABLE "workspace_published_endpoints"
  ADD CONSTRAINT "workspace_published_endpoints_approved_by_user_id_fkey"
  FOREIGN KEY ("approved_by_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

CREATE TABLE IF NOT EXISTS "workspace_service_credentials" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "label" TEXT NOT NULL,
  "token_prefix" TEXT NOT NULL,
  "token_hash" TEXT NOT NULL,
  "enabled" BOOLEAN NOT NULL DEFAULT TRUE,
  "expires_at" TIMESTAMP(3),
  "last_used_at" TIMESTAMP(3),
  "request_count" INTEGER NOT NULL DEFAULT 0,
  "created_by_user_id" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "revoked_at" TIMESTAMP(3),
  CONSTRAINT "workspace_service_credentials_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_service_credentials_token_prefix_key"
ON "workspace_service_credentials"("token_prefix");

CREATE INDEX IF NOT EXISTS "workspace_service_credentials_workspace_id_idx"
ON "workspace_service_credentials"("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_service_credentials_expires_at_idx"
ON "workspace_service_credentials"("expires_at");

CREATE INDEX IF NOT EXISTS "workspace_service_credentials_created_by_user_id_idx"
ON "workspace_service_credentials"("created_by_user_id");

DO $$
BEGIN
  ALTER TABLE "workspace_service_credentials"
  ADD CONSTRAINT "workspace_service_credentials_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  ALTER TABLE "workspace_service_credentials"
  ADD CONSTRAINT "workspace_service_credentials_created_by_user_id_fkey"
  FOREIGN KEY ("created_by_user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

CREATE TABLE IF NOT EXISTS "workspace_service_credential_endpoints" (
  "id" TEXT NOT NULL,
  "credential_id" TEXT NOT NULL,
  "endpoint_id" TEXT NOT NULL,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "workspace_service_credential_endpoints_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_service_credential_endpoints_credential_id_endpoint_id_key"
ON "workspace_service_credential_endpoints"("credential_id", "endpoint_id");

CREATE INDEX IF NOT EXISTS "workspace_service_credential_endpoints_credential_id_idx"
ON "workspace_service_credential_endpoints"("credential_id");

CREATE INDEX IF NOT EXISTS "workspace_service_credential_endpoints_endpoint_id_idx"
ON "workspace_service_credential_endpoints"("endpoint_id");

DO $$
BEGIN
  ALTER TABLE "workspace_service_credential_endpoints"
  ADD CONSTRAINT "workspace_service_credential_endpoints_credential_id_fkey"
  FOREIGN KEY ("credential_id") REFERENCES "workspace_service_credentials"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  ALTER TABLE "workspace_service_credential_endpoints"
  ADD CONSTRAINT "workspace_service_credential_endpoints_endpoint_id_fkey"
  FOREIGN KEY ("endpoint_id") REFERENCES "workspace_published_endpoints"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

CREATE TABLE IF NOT EXISTS "workspace_api_request_logs" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "credential_id" TEXT,
  "credential_label" TEXT NOT NULL DEFAULT '',
  "endpoint_key" TEXT NOT NULL DEFAULT '',
  "endpoint_label" TEXT NOT NULL DEFAULT '',
  "method" TEXT NOT NULL,
  "path_template" TEXT NOT NULL,
  "status_code" INTEGER NOT NULL,
  "duration_ms" INTEGER NOT NULL,
  "client_fingerprint" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "workspace_api_request_logs_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "workspace_api_request_logs_workspace_id_created_at_idx"
ON "workspace_api_request_logs"("workspace_id", "created_at");

CREATE INDEX IF NOT EXISTS "workspace_api_request_logs_credential_id_created_at_idx"
ON "workspace_api_request_logs"("credential_id", "created_at");

CREATE INDEX IF NOT EXISTS "workspace_api_request_logs_endpoint_key_created_at_idx"
ON "workspace_api_request_logs"("endpoint_key", "created_at");

CREATE INDEX IF NOT EXISTS "workspace_api_request_logs_status_code_created_at_idx"
ON "workspace_api_request_logs"("status_code", "created_at");

DO $$
BEGIN
  ALTER TABLE "workspace_api_request_logs"
  ADD CONSTRAINT "workspace_api_request_logs_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  ALTER TABLE "workspace_api_request_logs"
  ADD CONSTRAINT "workspace_api_request_logs_credential_id_fkey"
  FOREIGN KEY ("credential_id") REFERENCES "workspace_service_credentials"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
