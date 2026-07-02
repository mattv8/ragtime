-- Add first-class TOTP MFA policy, factors, recovery codes, and trusted devices.

DO $$
BEGIN
    CREATE TYPE "TotpPolicy" AS ENUM ('optional', 'required_all', 'required_admins_groups');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE "auth_provider_config"
    ADD COLUMN IF NOT EXISTS "totp_policy" "TotpPolicy" NOT NULL DEFAULT 'optional',
    ADD COLUMN IF NOT EXISTS "totp_required_group_ids" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    ADD COLUMN IF NOT EXISTS "totp_remember_device_days" INTEGER NOT NULL DEFAULT 30;

ALTER TABLE "sessions"
    ADD COLUMN IF NOT EXISTS "mfa_verified_at" TIMESTAMP(3),
    ADD COLUMN IF NOT EXISTS "auth_methods" JSONB NOT NULL DEFAULT '[]';

CREATE TABLE IF NOT EXISTS "user_mfa_factors" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "factor_type" TEXT NOT NULL DEFAULT 'totp',
    "label" TEXT NOT NULL DEFAULT 'Authenticator app',
    "secret_encrypted" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT false,
    "confirmed_at" TIMESTAMP(3),
    "last_used_step" BIGINT,
    "last_used_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_mfa_factors_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "user_mfa_recovery_codes" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "code_hash" TEXT NOT NULL,
    "used_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_mfa_recovery_codes_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "user_mfa_trusted_devices" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "token_hash" TEXT NOT NULL,
    "user_agent" TEXT,
    "ip_address" TEXT,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "last_used_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_mfa_trusted_devices_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "user_mfa_factors_user_id_factor_type_key" ON "user_mfa_factors"("user_id", "factor_type");
CREATE INDEX IF NOT EXISTS "user_mfa_factors_enabled_idx" ON "user_mfa_factors"("enabled");
CREATE INDEX IF NOT EXISTS "user_mfa_recovery_codes_user_id_used_at_idx" ON "user_mfa_recovery_codes"("user_id", "used_at");
CREATE UNIQUE INDEX IF NOT EXISTS "user_mfa_trusted_devices_token_hash_key" ON "user_mfa_trusted_devices"("token_hash");
CREATE INDEX IF NOT EXISTS "user_mfa_trusted_devices_user_id_expires_at_idx" ON "user_mfa_trusted_devices"("user_id", "expires_at");
CREATE INDEX IF NOT EXISTS "user_mfa_trusted_devices_expires_at_idx" ON "user_mfa_trusted_devices"("expires_at");

DO $$
BEGIN
    ALTER TABLE "user_mfa_factors"
        ADD CONSTRAINT "user_mfa_factors_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "user_mfa_recovery_codes"
        ADD CONSTRAINT "user_mfa_recovery_codes_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "user_mfa_trusted_devices"
        ADD CONSTRAINT "user_mfa_trusted_devices_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
