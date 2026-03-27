-- CreateTable
CREATE TABLE IF NOT EXISTS "workspace_environment_variables" (
  "id" TEXT NOT NULL,
  "workspace_id" TEXT NOT NULL,
  "key" TEXT NOT NULL,
  "value" TEXT NOT NULL,
  "description" TEXT DEFAULT '',
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "workspace_environment_variables_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX IF NOT EXISTS "workspace_environment_variables_workspace_id_key_key"
ON "workspace_environment_variables"("workspace_id", "key");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "workspace_environment_variables_workspace_id_updated_at_idx"
ON "workspace_environment_variables"("workspace_id", "updated_at");

-- AddForeignKey
DO $$
BEGIN
  ALTER TABLE "workspace_environment_variables"
  ADD CONSTRAINT "workspace_environment_variables_workspace_id_fkey"
  FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;
