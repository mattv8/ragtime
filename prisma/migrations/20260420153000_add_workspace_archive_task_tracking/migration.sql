ALTER TABLE workspaces
ADD COLUMN IF NOT EXISTS archive_export_task_id TEXT,
ADD COLUMN IF NOT EXISTS archive_export_task_phase TEXT,
ADD COLUMN IF NOT EXISTS archive_import_task_id TEXT,
ADD COLUMN IF NOT EXISTS archive_import_task_phase TEXT;
