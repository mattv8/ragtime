-- Add sort_order column to tool_configs for drag-and-drop reordering
ALTER TABLE "tool_configs" ADD COLUMN IF NOT EXISTS "sort_order" INTEGER NOT NULL DEFAULT 0;
