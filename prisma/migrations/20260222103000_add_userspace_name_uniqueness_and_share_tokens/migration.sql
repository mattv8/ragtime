ALTER TABLE "workspaces"
    ADD COLUMN IF NOT EXISTS "name_normalized" TEXT,
    ADD COLUMN IF NOT EXISTS "share_token" TEXT,
    ADD COLUMN IF NOT EXISTS "share_token_created_at" TIMESTAMP(3);

CREATE UNIQUE INDEX IF NOT EXISTS "workspaces_owner_user_id_name_normalized_key"
    ON "workspaces"("owner_user_id", "name_normalized");

CREATE UNIQUE INDEX IF NOT EXISTS "workspaces_share_token_key"
    ON "workspaces"("share_token");
