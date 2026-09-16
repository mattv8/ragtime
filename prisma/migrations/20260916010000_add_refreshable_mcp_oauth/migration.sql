-- Additive storage for configurable web-session policy and refreshable MCP OAuth grants.
ALTER TABLE "auth_provider_config"
    ADD COLUMN IF NOT EXISTS "web_session_hours" INTEGER,
    ADD COLUMN IF NOT EXISTS "mcp_access_token_minutes" INTEGER NOT NULL DEFAULT 60,
    ADD COLUMN IF NOT EXISTS "mcp_authorization_days" INTEGER NOT NULL DEFAULT 30;

ALTER TABLE "users"
    ADD COLUMN IF NOT EXISTS "security_generation" INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS "oauth_grants" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "client_id" TEXT NOT NULL,
    "audience" TEXT NOT NULL,
    "scope" TEXT NOT NULL DEFAULT '',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "revoked_at" TIMESTAMP(3),
    "security_generation" INTEGER NOT NULL,
    "mfa_verified_at" TIMESTAMP(3),
    "auth_methods" JSONB NOT NULL DEFAULT '[]',
    CONSTRAINT "oauth_grants_pkey" PRIMARY KEY ("id")
);

CREATE TABLE IF NOT EXISTS "oauth_refresh_tokens" (
    "id" TEXT NOT NULL,
    "grant_id" TEXT NOT NULL,
    "token_hash" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "consumed_at" TIMESTAMP(3),
    CONSTRAINT "oauth_refresh_tokens_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "oauth_refresh_tokens_token_hash_key" ON "oauth_refresh_tokens"("token_hash");
CREATE INDEX IF NOT EXISTS "oauth_grants_user_id_expires_at_idx" ON "oauth_grants"("user_id", "expires_at");
CREATE INDEX IF NOT EXISTS "oauth_grants_expires_at_idx" ON "oauth_grants"("expires_at");
CREATE INDEX IF NOT EXISTS "oauth_refresh_tokens_grant_id_idx" ON "oauth_refresh_tokens"("grant_id");

DO $$
BEGIN
    ALTER TABLE "oauth_grants"
        ADD CONSTRAINT "oauth_grants_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
BEGIN
    ALTER TABLE "oauth_refresh_tokens"
        ADD CONSTRAINT "oauth_refresh_tokens_grant_id_fkey"
        FOREIGN KEY ("grant_id") REFERENCES "oauth_grants"("id") ON DELETE CASCADE ON UPDATE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
