-- Remote enrollment records deliberately keep only hashes of bearer credentials.
ALTER TYPE "ToolType" ADD VALUE IF NOT EXISTS 'windows_shell';

CREATE TYPE "RemotePairingStatus" AS ENUM ('pending', 'consumed', 'paired', 'failed', 'expired', 'revoked');

CREATE TABLE "remote_pairings" (
    "id" TEXT NOT NULL,
    "tool_id" TEXT NOT NULL,
    "status" "RemotePairingStatus" NOT NULL DEFAULT 'pending',
    "enrollment_token_hash" TEXT NOT NULL,
    "enrollment_expires_at" TIMESTAMP(3) NOT NULL,
    "enrollment_nonce" TEXT,
    "encrypted_private_key" TEXT NOT NULL,
    "public_key" TEXT NOT NULL,
    "tunnel_credential_hash" TEXT NOT NULL,
    "encrypted_enrollment_response" TEXT,
    "recovery_expires_at" TIMESTAMP(3),
    "is_draft" BOOLEAN NOT NULL DEFAULT true,
    "facts" JSONB,
    "transport" TEXT,
    "error" TEXT,
    "last_activity_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "remote_pairings_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "remote_pairings_enrollment_token_hash_key" ON "remote_pairings"("enrollment_token_hash");
CREATE UNIQUE INDEX "remote_pairings_tunnel_credential_hash_key" ON "remote_pairings"("tunnel_credential_hash");
CREATE INDEX "remote_pairings_tool_id_idx" ON "remote_pairings"("tool_id");
CREATE INDEX "remote_pairings_status_last_activity_at_idx" ON "remote_pairings"("status", "last_activity_at");
ALTER TABLE "remote_pairings" ADD CONSTRAINT "remote_pairings_tool_id_fkey" FOREIGN KEY ("tool_id") REFERENCES "tool_configs"("id") ON DELETE CASCADE ON UPDATE CASCADE;
