ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS subagents_enabled BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS parent_conversation_id TEXT;

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS subagent_role TEXT;

ALTER TABLE conversations
  ADD COLUMN IF NOT EXISTS subagent_index INTEGER;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'conversations_parent_conversation_id_fkey'
  ) THEN
    ALTER TABLE conversations
      ADD CONSTRAINT conversations_parent_conversation_id_fkey
      FOREIGN KEY (parent_conversation_id)
      REFERENCES conversations(id)
      ON DELETE CASCADE
      ON UPDATE CASCADE;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS conversations_parent_conversation_id_idx
  ON conversations(parent_conversation_id);

CREATE INDEX IF NOT EXISTS conversations_workspace_parent_updated_idx
  ON conversations(workspace_id, parent_conversation_id, updated_at);
