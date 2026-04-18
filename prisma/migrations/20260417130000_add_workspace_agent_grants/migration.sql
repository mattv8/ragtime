-- CreateEnum
DO $$
BEGIN
  CREATE TYPE "workspace_agent_grant_mode" AS ENUM ('read', 'read_write');
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- CreateTable
CREATE TABLE IF NOT EXISTS "workspace_agent_grants" (
  "id" TEXT NOT NULL,
  "source_workspace_id" TEXT NOT NULL,
  "target_workspace_id" TEXT NOT NULL,
  "access_mode" "workspace_agent_grant_mode" NOT NULL DEFAULT 'read',
  "granted_by_user_id" TEXT NOT NULL,
  "expires_at" TIMESTAMP(3),
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "workspace_agent_grants_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_agent_grants_source_workspace_id_target_workspace_id_key"
ON "workspace_agent_grants"("source_workspace_id", "target_workspace_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "workspace_agent_grants_source_workspace_id_idx"
ON "workspace_agent_grants"("source_workspace_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "workspace_agent_grants_target_workspace_id_idx"
ON "workspace_agent_grants"("target_workspace_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "workspace_agent_grants_expires_at_idx"
ON "workspace_agent_grants"("expires_at");

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "workspace_agent_grants"
  ADD CONSTRAINT "workspace_agent_grants_source_workspace_id_fkey"
  FOREIGN KEY ("source_workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "workspace_agent_grants"
  ADD CONSTRAINT "workspace_agent_grants_target_workspace_id_fkey"
  FOREIGN KEY ("target_workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "workspace_agent_grants"
  ADD CONSTRAINT "workspace_agent_grants_granted_by_user_id_fkey"
  FOREIGN KEY ("granted_by_user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
