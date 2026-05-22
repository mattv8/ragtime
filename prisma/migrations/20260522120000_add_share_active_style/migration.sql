-- Add `active_share_style` to workspace_shares and conversation_shares so the
-- UI's named/anonymous/subdomain toggle can persist the share owner's choice.
-- Idempotent for repeated deploys against environments that may already have
-- the column.

ALTER TABLE "workspace_shares"
    ADD COLUMN IF NOT EXISTS "active_share_style" TEXT NOT NULL DEFAULT 'anonymous';

ALTER TABLE "conversation_shares"
    ADD COLUMN IF NOT EXISTS "active_share_style" TEXT NOT NULL DEFAULT 'anonymous';
