-- Add cloud-backed userspace mount support with separate user-scoped sources.
-- Add configurable User Space mount auto-sync intervals.

DO $$
BEGIN
    ALTER TYPE "UserspaceMountSourceType" ADD VALUE IF NOT EXISTS 'microsoft_drive';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TYPE "UserspaceMountSourceType" ADD VALUE IF NOT EXISTS 'google_drive';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'UserspaceMountSourceScope') THEN
        CREATE TYPE "UserspaceMountSourceScope" AS ENUM ('global', 'user');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'UserCloudOAuthProvider') THEN
        CREATE TYPE "UserCloudOAuthProvider" AS ENUM ('microsoft_drive', 'google_drive');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS "user_cloud_oauth_accounts" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "provider" "UserCloudOAuthProvider" NOT NULL,
    "account_email" TEXT,
    "account_name" TEXT,
    "access_token" TEXT NOT NULL DEFAULT '',
    "refresh_token" TEXT NOT NULL DEFAULT '',
    "expires_at" TIMESTAMP(3),
    "scopes" JSONB NOT NULL DEFAULT '[]',
    "metadata" JSONB NOT NULL DEFAULT '{}',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_cloud_oauth_accounts_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "user_cloud_oauth_accounts_user_id_provider_idx"
    ON "user_cloud_oauth_accounts"("user_id", "provider");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_cloud_oauth_accounts_user_id_fkey'
          AND table_name = 'user_cloud_oauth_accounts'
    ) THEN
        ALTER TABLE "user_cloud_oauth_accounts"
            ADD CONSTRAINT "user_cloud_oauth_accounts_user_id_fkey"
            FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS "user_userspace_mount_sources" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT DEFAULT '',
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "source_type" "UserspaceMountSourceType" NOT NULL,
    "oauth_account_id" TEXT,
    "connection_config" JSONB NOT NULL,
    "approved_paths" JSONB NOT NULL DEFAULT '[]',
    "sync_interval_seconds" INTEGER DEFAULT 30,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_userspace_mount_sources_pkey" PRIMARY KEY ("id")
);

CREATE INDEX IF NOT EXISTS "user_userspace_mount_sources_user_id_enabled_source_type_idx"
    ON "user_userspace_mount_sources"("user_id", "enabled", "source_type");

CREATE INDEX IF NOT EXISTS "user_userspace_mount_sources_oauth_account_id_idx"
    ON "user_userspace_mount_sources"("oauth_account_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_userspace_mount_sources_user_id_fkey'
          AND table_name = 'user_userspace_mount_sources'
    ) THEN
        ALTER TABLE "user_userspace_mount_sources"
            ADD CONSTRAINT "user_userspace_mount_sources_user_id_fkey"
            FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'user_userspace_mount_sources_oauth_account_id_fkey'
          AND table_name = 'user_userspace_mount_sources'
    ) THEN
        ALTER TABLE "user_userspace_mount_sources"
            ADD CONSTRAINT "user_userspace_mount_sources_oauth_account_id_fkey"
            FOREIGN KEY ("oauth_account_id") REFERENCES "user_cloud_oauth_accounts"("id") ON DELETE SET NULL ON UPDATE CASCADE;
    END IF;
END $$;

ALTER TABLE "workspace_mounts"
    ADD COLUMN IF NOT EXISTS "user_mount_source_id" TEXT;

ALTER TABLE "workspace_mounts"
    ALTER COLUMN "mount_source_id" DROP NOT NULL;

CREATE INDEX IF NOT EXISTS "workspace_mounts_user_mount_source_id_idx"
    ON "workspace_mounts"("user_mount_source_id");

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'workspace_mounts_user_mount_source_id_fkey'
          AND table_name = 'workspace_mounts'
    ) THEN
        ALTER TABLE "workspace_mounts"
            ADD CONSTRAINT "workspace_mounts_user_mount_source_id_fkey"
            FOREIGN KEY ("user_mount_source_id") REFERENCES "user_userspace_mount_sources"("id") ON DELETE CASCADE ON UPDATE CASCADE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'workspace_mounts_exactly_one_source_chk'
    ) THEN
        ALTER TABLE "workspace_mounts"
            ADD CONSTRAINT "workspace_mounts_exactly_one_source_chk"
            CHECK (
                ("mount_source_id" IS NOT NULL AND "user_mount_source_id" IS NULL)
                OR ("mount_source_id" IS NULL AND "user_mount_source_id" IS NOT NULL)
            );
    END IF;
END $$;

ALTER TABLE "app_settings"
ADD COLUMN IF NOT EXISTS "userspace_mount_sync_interval_seconds" INTEGER NOT NULL DEFAULT 30;

ALTER TABLE "workspace_mounts"
ADD COLUMN IF NOT EXISTS "sync_interval_seconds" INTEGER;
