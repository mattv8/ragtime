-- Add WebAuthn passkey 2FA support: allowed methods policy and credential storage.

ALTER TABLE "auth_provider_config"
    ADD COLUMN IF NOT EXISTS "mfa_allowed_methods" TEXT[] NOT NULL DEFAULT ARRAY['totp']::TEXT[];

CREATE TABLE IF NOT EXISTS "user_webauthn_credentials" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "credential_id" TEXT NOT NULL,
    "public_key" TEXT NOT NULL,
    "sign_count" INTEGER NOT NULL DEFAULT 0,
    "transports" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "aaguid" TEXT,
    "name" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_used_at" TIMESTAMP(3),
    CONSTRAINT "user_webauthn_credentials_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "user_webauthn_credentials_credential_id_key" ON "user_webauthn_credentials"("credential_id");
CREATE INDEX IF NOT EXISTS "user_webauthn_credentials_user_id_idx" ON "user_webauthn_credentials"("user_id");

DO $$
BEGIN
    ALTER TABLE "user_webauthn_credentials"
        ADD CONSTRAINT "user_webauthn_credentials_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Preferred/default 2FA method selection.
ALTER TABLE "auth_provider_config"
    ADD COLUMN IF NOT EXISTS "mfa_default_method" TEXT;

ALTER TABLE "users"
    ADD COLUMN IF NOT EXISTS "mfa_preferred_method" TEXT;
