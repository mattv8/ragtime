-- Align DB default conversation title with application defaults.
ALTER TABLE "conversations"
ALTER COLUMN "title" SET DEFAULT 'Untitled Chat';
