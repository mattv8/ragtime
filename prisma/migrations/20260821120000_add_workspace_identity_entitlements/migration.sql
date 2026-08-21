-- =============================================================================
-- Add workspace identity entitlement rules
-- =============================================================================

CREATE TABLE IF NOT EXISTS "workspace_identity_entitlement_rules" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "workspace_id" TEXT NOT NULL,
    "auth_group_id" TEXT NOT NULL,
    "entitlements" JSONB NOT NULL DEFAULT '[]',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_identity_entitlement_rules_workspace_id_auth_group_id_key"
    ON "workspace_identity_entitlement_rules" ("workspace_id", "auth_group_id");

CREATE INDEX IF NOT EXISTS "workspace_identity_entitlement_rules_workspace_id_idx"
    ON "workspace_identity_entitlement_rules" ("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_identity_entitlement_rules_auth_group_id_idx"
    ON "workspace_identity_entitlement_rules" ("auth_group_id");

ALTER TABLE "workspace_identity_entitlement_rules"
    DROP CONSTRAINT IF EXISTS "workspace_identity_entitlement_rules_workspace_id_fkey",
    ADD CONSTRAINT "workspace_identity_entitlement_rules_workspace_id_fkey"
        FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "workspace_identity_entitlement_rules"
    DROP CONSTRAINT IF EXISTS "workspace_identity_entitlement_rules_auth_group_id_fkey",
    ADD CONSTRAINT "workspace_identity_entitlement_rules_auth_group_id_fkey"
        FOREIGN KEY ("auth_group_id") REFERENCES "auth_groups" ("id") ON DELETE CASCADE ON UPDATE CASCADE;
