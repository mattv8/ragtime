-- Add server_name column to app_settings table
ALTER TABLE "app_settings" ADD COLUMN IF NOT EXISTS "server_name" TEXT NOT NULL DEFAULT 'Ragtime';
