DO $$
BEGIN
    ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'http_api';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
