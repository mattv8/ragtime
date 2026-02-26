-- CreateEnum
DO $$
BEGIN
  CREATE TYPE "RuntimeSessionState" AS ENUM ('starting', 'running', 'stopping', 'stopped', 'error');
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- CreateTable
CREATE TABLE IF NOT EXISTS "userspace_runtime_sessions" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "leased_by_user_id" TEXT NOT NULL,
  "state" "RuntimeSessionState" NOT NULL DEFAULT 'starting',
  "runtime_provider" TEXT NOT NULL DEFAULT 'microvm_pool_v1',
  "provider_session_id" TEXT,
  "preview_internal_url" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  "last_heartbeat_at" TIMESTAMP(3),
  "idle_expires_at" TIMESTAMP(3),
  "ttl_expires_at" TIMESTAMP(3),
  "last_error" TEXT,
  CONSTRAINT "userspace_runtime_sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "userspace_runtime_audit_events" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "user_id" TEXT,
  "session_id" TEXT,
  "event_type" TEXT NOT NULL,
  "event_payload" JSONB NOT NULL DEFAULT '{}',
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "userspace_runtime_audit_events_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "userspace_collab_docs" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "file_path" TEXT NOT NULL,
  "checkpoint_version" INTEGER NOT NULL DEFAULT 0,
  "doc_state_base64" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "userspace_collab_docs_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "userspace_runtime_sessions_workspace_id_updated_at_idx"
ON "userspace_runtime_sessions"("workspace_id", "updated_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "userspace_runtime_sessions_leased_by_user_id_updated_at_idx"
ON "userspace_runtime_sessions"("leased_by_user_id", "updated_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "userspace_runtime_audit_events_workspace_id_created_at_idx"
ON "userspace_runtime_audit_events"("workspace_id", "created_at");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "userspace_runtime_audit_events_user_id_created_at_idx"
ON "userspace_runtime_audit_events"("user_id", "created_at");

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "userspace_collab_docs_workspace_id_file_path_key"
ON "userspace_collab_docs"("workspace_id", "file_path");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "userspace_collab_docs_workspace_id_updated_at_idx"
ON "userspace_collab_docs"("workspace_id", "updated_at");

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "userspace_runtime_sessions"
  ADD CONSTRAINT "userspace_runtime_sessions_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "userspace_runtime_sessions"
  ADD CONSTRAINT "userspace_runtime_sessions_leased_by_user_id_fkey"
  FOREIGN KEY ("leased_by_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "userspace_runtime_audit_events"
  ADD CONSTRAINT "userspace_runtime_audit_events_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "userspace_runtime_audit_events"
  ADD CONSTRAINT "userspace_runtime_audit_events_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "userspace_collab_docs"
  ADD CONSTRAINT "userspace_collab_docs_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
