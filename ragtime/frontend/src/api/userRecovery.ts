/** Credential-recovery endpoints.  Keep continuation tokens in component state only. */
import { api } from './client';

export const userRecoveryApi = {
  verifyAdminSecurity: api.verifyAdminSecurity,
  getRecoveryPass: api.getRecoveryPass,
  issueRecoveryPass: api.issueRecoveryPass,
  revokeRecoveryPass: api.revokeRecoveryPass,
  redeem: api.redeemRecoveryPass,
  startTotp: api.startRecoveryTotp,
  completeTotp: api.completeRecoveryTotp,
  startWebauthn: api.startRecoveryWebauthn,
  completeWebauthn: api.completeRecoveryWebauthn,
};
