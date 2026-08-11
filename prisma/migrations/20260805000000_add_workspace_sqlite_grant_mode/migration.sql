-- CreateEnum
DO $$
BEGIN
  CREATE TYPE "workspace_sqlite_grant_mode" AS ENUM ('none', 'read', 'read_write');
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

-- AddColumn
ALTER TABLE "workspace_agent_grants"
ADD COLUMN IF NOT EXISTS "sqlite_access_mode" "workspace_sqlite_grant_mode";

-- Backfill legacy rows
UPDATE "workspace_agent_grants"
SET "sqlite_access_mode" = 'none'
WHERE "sqlite_access_mode" IS NULL;

-- Enforce default + non-null
ALTER TABLE "workspace_agent_grants"
ALTER COLUMN "sqlite_access_mode" SET DEFAULT 'none';

ALTER TABLE "workspace_agent_grants"
ALTER COLUMN "sqlite_access_mode" SET NOT NULL;
