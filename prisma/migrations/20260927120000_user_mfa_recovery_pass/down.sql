ALTER TABLE "user_mfa_recovery_passes" DROP CONSTRAINT IF EXISTS "user_mfa_recovery_passes_user_id_fkey";
DROP TABLE IF EXISTS "user_mfa_recovery_passes";
