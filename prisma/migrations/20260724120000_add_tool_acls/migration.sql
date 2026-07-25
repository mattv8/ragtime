-- CreateEnum: tool_access_level
DO $$ BEGIN
  CREATE TYPE "tool_access_level" AS ENUM ('deny', 'read', 'read_write');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS "tool_access_policies" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "tool_config_id" TEXT NOT NULL,
    "default_chat_access" "tool_access_level" NOT NULL DEFAULT 'deny',
    "default_workspace_access" "tool_access_level" NOT NULL DEFAULT 'deny',
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "tool_user_access" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "policy_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "chat_access" "tool_access_level",
    "workspace_access" "tool_access_level",
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "tool_auth_group_access" (
    "id" TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "policy_id" TEXT NOT NULL,
    "auth_group_id" TEXT NOT NULL,
    "chat_access" "tool_access_level",
    "workspace_access" "tool_access_level",
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "tool_access_policies_tool_config_id_key"
    ON "tool_access_policies" ("tool_config_id");

CREATE UNIQUE INDEX IF NOT EXISTS "tool_user_access_policy_id_user_id_key"
    ON "tool_user_access" ("policy_id", "user_id");

CREATE INDEX IF NOT EXISTS "tool_user_access_user_id_idx"
    ON "tool_user_access" ("user_id");

CREATE UNIQUE INDEX IF NOT EXISTS "tool_auth_group_access_policy_id_auth_group_id_key"
    ON "tool_auth_group_access" ("policy_id", "auth_group_id");

CREATE INDEX IF NOT EXISTS "tool_auth_group_access_auth_group_id_idx"
    ON "tool_auth_group_access" ("auth_group_id");

ALTER TABLE "tool_access_policies"
    DROP CONSTRAINT IF EXISTS "tool_access_policies_tool_config_id_fkey",
    ADD CONSTRAINT "tool_access_policies_tool_config_id_fkey"
        FOREIGN KEY ("tool_config_id") REFERENCES "tool_configs" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "tool_user_access"
    DROP CONSTRAINT IF EXISTS "tool_user_access_policy_id_fkey",
    ADD CONSTRAINT "tool_user_access_policy_id_fkey"
        FOREIGN KEY ("policy_id") REFERENCES "tool_access_policies" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "tool_user_access"
    DROP CONSTRAINT IF EXISTS "tool_user_access_user_id_fkey",
    ADD CONSTRAINT "tool_user_access_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "tool_auth_group_access"
    DROP CONSTRAINT IF EXISTS "tool_auth_group_access_policy_id_fkey",
    ADD CONSTRAINT "tool_auth_group_access_policy_id_fkey"
        FOREIGN KEY ("policy_id") REFERENCES "tool_access_policies" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "tool_auth_group_access"
    DROP CONSTRAINT IF EXISTS "tool_auth_group_access_auth_group_id_fkey",
    ADD CONSTRAINT "tool_auth_group_access_auth_group_id_fkey"
        FOREIGN KEY ("auth_group_id") REFERENCES "auth_groups" ("id") ON DELETE CASCADE ON UPDATE CASCADE;

INSERT INTO "tool_access_policies" (id, tool_config_id, default_chat_access, default_workspace_access)
SELECT gen_random_uuid()::text, tc.id, 'read_write', 'read_write'
FROM "tool_configs" tc
WHERE NOT EXISTS (SELECT 1 FROM "tool_access_policies" p WHERE p.tool_config_id = tc.id);
