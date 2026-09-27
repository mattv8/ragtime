CREATE TABLE "user_mfa_recovery_passes" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pass_hash" TEXT NOT NULL,
    "security_generation" INTEGER NOT NULL,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" TIMESTAMP(3) NOT NULL,
    "redeemed_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "revoked_at" TIMESTAMP(3),
    CONSTRAINT "user_mfa_recovery_passes_pkey" PRIMARY KEY ("id")
);
CREATE UNIQUE INDEX "user_mfa_recovery_passes_user_id_key" ON "user_mfa_recovery_passes"("user_id");
CREATE INDEX "user_mfa_recovery_passes_expires_at_idx" ON "user_mfa_recovery_passes"("expires_at");
ALTER TABLE "user_mfa_recovery_passes" ADD CONSTRAINT "user_mfa_recovery_passes_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
