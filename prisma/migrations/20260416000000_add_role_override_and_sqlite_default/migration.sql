-- AlterTable
ALTER TABLE "users" ADD COLUMN IF NOT EXISTS "role_manually_set" BOOLEAN NOT NULL DEFAULT false;

-- Change User Space workspace SQLite snapshot policy default to include.
-- Existing rows are intentionally unchanged; only new workspaces use this default.
ALTER TABLE "workspaces"
ALTER COLUMN "sqlite_persistence_mode"
SET DEFAULT 'include';
