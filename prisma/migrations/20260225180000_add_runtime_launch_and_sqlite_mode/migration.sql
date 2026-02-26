DO $$
BEGIN
    CREATE TYPE "SqlitePersistenceMode" AS ENUM ('include', 'exclude');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE "workspaces"
    ADD COLUMN IF NOT EXISTS "sqlite_persistence_mode" "SqlitePersistenceMode" NOT NULL DEFAULT 'include';

ALTER TABLE "userspace_runtime_sessions"
    ADD COLUMN IF NOT EXISTS "launch_framework" TEXT,
    ADD COLUMN IF NOT EXISTS "launch_command" TEXT,
    ADD COLUMN IF NOT EXISTS "launch_cwd" TEXT,
    ADD COLUMN IF NOT EXISTS "launch_port" INTEGER;

UPDATE "workspaces"
SET "sqlite_persistence_mode" = 'include'
WHERE "sqlite_persistence_mode" IS NULL;
