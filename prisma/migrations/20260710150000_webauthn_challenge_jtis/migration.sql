-- Persist consumed WebAuthn challenge JTIs so replay detection survives process restarts.

CREATE TABLE IF NOT EXISTS "user_webauthn_challenges" (
    "id" TEXT NOT NULL,
    "jti" TEXT NOT NULL,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "user_webauthn_challenges_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "user_webauthn_challenges_jti_key" ON "user_webauthn_challenges"("jti");
