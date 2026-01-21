-- Add mysql to ToolType enum
-- MySQL/MariaDB database tool support

ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'mysql';
