-- Set User Space workspace SQLite snapshot policy default to exclude.
-- Existing rows are intentionally unchanged; only new workspaces use this default.
ALTER TABLE "workspaces"
ALTER COLUMN "sqlite_persistence_mode"
SET DEFAULT 'exclude';
