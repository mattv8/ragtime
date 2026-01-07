-- Add mssql to ToolType enum
-- PostgreSQL enum modification

ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'mssql';
