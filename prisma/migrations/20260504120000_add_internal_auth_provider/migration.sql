-- Add local managed auth provider and provider-neutral auth cache tables.
--
-- PostgreSQL does not allow using an enum value added with ALTER TYPE as a
-- column default until the transaction commits. Recreate the enum up front
-- when needed so the new value is safe to use throughout this migration.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_enum e
        JOIN pg_type t ON t.oid = e.enumtypid
        WHERE t.typname = 'AuthProvider'
          AND e.enumlabel = 'local_managed'
    ) THEN
        ALTER TYPE "AuthProvider" RENAME TO "AuthProvider_old";
        CREATE TYPE "AuthProvider" AS ENUM ('ldap', 'local', 'local_managed');

        ALTER TABLE "users" ALTER COLUMN "auth_provider" DROP DEFAULT;
        ALTER TABLE "users"
            ALTER COLUMN "auth_provider"
            TYPE "AuthProvider"
            USING "auth_provider"::text::"AuthProvider";
        ALTER TABLE "users" ALTER COLUMN "auth_provider" SET DEFAULT 'ldap';

        DROP TYPE "AuthProvider_old";
    END IF;
END $$;

ALTER TABLE "users"
    ADD COLUMN IF NOT EXISTS "source_provider" "AuthProvider",
    ADD COLUMN IF NOT EXISTS "source_id" TEXT,
    ADD COLUMN IF NOT EXISTS "password_hash" TEXT,
    ADD COLUMN IF NOT EXISTS "cached_groups" JSONB NOT NULL DEFAULT '[]',
    ADD COLUMN IF NOT EXISTS "source_synced_at" TIMESTAMP(3),
    ADD COLUMN IF NOT EXISTS "source_expires_at" TIMESTAMP(3);

CREATE TABLE IF NOT EXISTS "auth_provider_config" (
    "id" TEXT NOT NULL DEFAULT 'default',
    "local_users_enabled" BOOLEAN NOT NULL DEFAULT true,
    "ldap_lazy_sync_enabled" BOOLEAN NOT NULL DEFAULT true,
    "manual_role_override_wins" BOOLEAN NOT NULL DEFAULT true,
    "cache_ttl_minutes" INTEGER NOT NULL DEFAULT 240,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "auth_provider_config_pkey" PRIMARY KEY ("id")
);

INSERT INTO "auth_provider_config" ("id")
VALUES ('default')
ON CONFLICT ("id") DO NOTHING;

CREATE TABLE IF NOT EXISTS "auth_groups" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "display_name" TEXT NOT NULL,
    "description" TEXT NOT NULL DEFAULT '',
    "provider" "AuthProvider" NOT NULL DEFAULT 'local_managed',
    "source_id" TEXT,
    "source_dn" TEXT,
    "role" "UserRole",
    "is_logon_group" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "auth_groups_pkey" PRIMARY KEY ("id")
);

ALTER TABLE "auth_groups"
    ADD COLUMN IF NOT EXISTS "is_logon_group" BOOLEAN NOT NULL DEFAULT false;

ALTER TABLE "ldap_config"
    ADD COLUMN IF NOT EXISTS "admin_group_dns" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS "user_group_dns" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'ldap_config'
          AND column_name = 'admin_group_dn'
    ) THEN
        UPDATE "ldap_config"
        SET "admin_group_dns" = CASE
            WHEN COALESCE(NULLIF(BTRIM("admin_group_dn"), ''), '') = '' THEN ARRAY[]::TEXT[]
            ELSE ARRAY["admin_group_dn"]
        END
        WHERE "admin_group_dns" = ARRAY[]::TEXT[];
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'ldap_config'
          AND column_name = 'user_group_dn'
    ) THEN
        UPDATE "ldap_config"
        SET "user_group_dns" = CASE
            WHEN COALESCE(NULLIF(BTRIM("user_group_dn"), ''), '') = '' THEN ARRAY[]::TEXT[]
            ELSE ARRAY["user_group_dn"]
        END
        WHERE "user_group_dns" = ARRAY[]::TEXT[];
    END IF;
END $$;

UPDATE "auth_groups" AS ag
SET "role" = 'admin'
FROM "ldap_config" AS lc
WHERE ag."provider" = 'ldap'
  AND ag."source_dn" IS NOT NULL
    AND LOWER(ag."source_dn") = ANY(ARRAY(SELECT LOWER(dn.value) FROM unnest(lc."admin_group_dns") AS dn(value)));

UPDATE "auth_groups" AS ag
SET "is_logon_group" = true
FROM "ldap_config" AS lc
WHERE ag."provider" = 'ldap'
  AND ag."source_dn" IS NOT NULL
    AND LOWER(ag."source_dn") = ANY(ARRAY(SELECT LOWER(dn.value) FROM unnest(lc."user_group_dns") AS dn(value)));

ALTER TABLE "ldap_config"
    DROP COLUMN IF EXISTS "admin_group_dn",
    DROP COLUMN IF EXISTS "user_group_dn";

CREATE TABLE IF NOT EXISTS "auth_group_memberships" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "group_id" TEXT NOT NULL,
    "source_provider" "AuthProvider" NOT NULL DEFAULT 'local_managed',
    "source_synced_at" TIMESTAMP(3),
    "expires_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "auth_group_memberships_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "auth_sync_events" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "username" TEXT NOT NULL,
    "source_provider" "AuthProvider" NOT NULL,
    "action" TEXT NOT NULL,
    "status" TEXT NOT NULL,
    "detail" TEXT NOT NULL DEFAULT '',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "auth_sync_events_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "auth_groups_key_key" ON "auth_groups"("key");
CREATE INDEX IF NOT EXISTS "users_auth_provider_idx" ON "users"("auth_provider");
CREATE INDEX IF NOT EXISTS "users_source_provider_source_id_idx" ON "users"("source_provider", "source_id");
CREATE INDEX IF NOT EXISTS "users_source_expires_at_idx" ON "users"("source_expires_at");
CREATE INDEX IF NOT EXISTS "auth_groups_provider_idx" ON "auth_groups"("provider");
CREATE INDEX IF NOT EXISTS "auth_groups_source_dn_idx" ON "auth_groups"("source_dn");
CREATE UNIQUE INDEX IF NOT EXISTS "auth_group_memberships_user_id_group_id_key" ON "auth_group_memberships"("user_id", "group_id");
CREATE INDEX IF NOT EXISTS "auth_group_memberships_group_id_idx" ON "auth_group_memberships"("group_id");
CREATE INDEX IF NOT EXISTS "auth_group_memberships_source_provider_expires_at_idx" ON "auth_group_memberships"("source_provider", "expires_at");
CREATE INDEX IF NOT EXISTS "auth_sync_events_user_id_created_at_idx" ON "auth_sync_events"("user_id", "created_at");
CREATE INDEX IF NOT EXISTS "auth_sync_events_source_provider_created_at_idx" ON "auth_sync_events"("source_provider", "created_at");

DO $$
BEGIN
    ALTER TABLE "auth_group_memberships"
        ADD CONSTRAINT "auth_group_memberships_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "auth_group_memberships"
        ADD CONSTRAINT "auth_group_memberships_group_id_fkey"
        FOREIGN KEY ("group_id") REFERENCES "auth_groups"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "auth_sync_events"
        ADD CONSTRAINT "auth_sync_events_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
