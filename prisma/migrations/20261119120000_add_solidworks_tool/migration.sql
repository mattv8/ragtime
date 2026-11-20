DO $$
BEGIN
    ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'solidworks';
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
