-- Extract workspace share fields into a dedicated workspace_shares table so
-- multiple share links can be created per workspace, mirroring conversation_shares.

CREATE TABLE IF NOT EXISTS "workspace_shares" (
    "id" TEXT NOT NULL,
    "workspace_id" TEXT NOT NULL,
    "owner_user_id" TEXT NOT NULL,
    "label" TEXT,
    "share_token" TEXT,
    "share_token_created_at" TIMESTAMP(3),
    "share_slug" TEXT,
    "share_access_mode" TEXT NOT NULL DEFAULT 'token',
    "share_password" TEXT,
    "share_selected_user_ids" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "share_selected_ldap_groups" JSONB NOT NULL DEFAULT '[]'::jsonb,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "workspace_shares_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "workspace_shares_workspace_id_idx"
    ON "workspace_shares"("workspace_id");

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_shares_share_token_key"
    ON "workspace_shares"("share_token");

CREATE INDEX IF NOT EXISTS "workspace_shares_owner_user_id_share_slug_idx"
    ON "workspace_shares"("owner_user_id", "share_slug");

CREATE INDEX IF NOT EXISTS "workspace_shares_workspace_id_share_token_idx"
    ON "workspace_shares"("workspace_id", "share_token");

DO $$ BEGIN
    ALTER TABLE "workspace_shares"
        ADD CONSTRAINT "workspace_shares_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    ALTER TABLE "workspace_shares"
        ADD CONSTRAINT "workspace_shares_owner_user_id_fkey"
        FOREIGN KEY ("owner_user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Backfill any existing share state from workspaces into the new table.
-- Only insert when there is meaningful share state (token, slug, or non-default
-- access mode/password/selected lists).
INSERT INTO "workspace_shares" (
    "id", "workspace_id", "owner_user_id", "label",
    "share_token", "share_token_created_at", "share_slug",
    "share_access_mode", "share_password",
    "share_selected_user_ids", "share_selected_ldap_groups",
    "created_at", "updated_at"
)
SELECT
    gen_random_uuid()::text,
    w."id",
    w."owner_user_id",
    NULL,
    w."share_token",
    w."share_token_created_at",
    w."share_slug",
    COALESCE(w."share_access_mode", 'token'),
    w."share_password",
    COALESCE(w."share_selected_user_ids", '[]'::jsonb),
    COALESCE(w."share_selected_ldap_groups", '[]'::jsonb),
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
FROM "workspaces" w
WHERE
    w."share_token" IS NOT NULL
    OR w."share_slug" IS NOT NULL
    OR (w."share_access_mode" IS NOT NULL AND w."share_access_mode" <> 'token')
    OR w."share_password" IS NOT NULL
    OR (w."share_selected_user_ids" IS NOT NULL AND w."share_selected_user_ids" <> '[]'::jsonb)
    OR (w."share_selected_ldap_groups" IS NOT NULL AND w."share_selected_ldap_groups" <> '[]'::jsonb)
ON CONFLICT DO NOTHING;

-- Drop the old denormalized columns. The unique constraint on (owner_user_id,
-- share_slug) is also removed because uniqueness now lives in workspace_shares.
ALTER TABLE "workspaces" DROP CONSTRAINT IF EXISTS "workspaces_owner_user_id_share_slug_key";
DROP INDEX IF EXISTS "workspaces_owner_user_id_share_slug_key";
DROP INDEX IF EXISTS "workspaces_share_token_key";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_token";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_token_created_at";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_slug";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_access_mode";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_password";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_selected_user_ids";
ALTER TABLE "workspaces" DROP COLUMN IF EXISTS "share_selected_ldap_groups";
