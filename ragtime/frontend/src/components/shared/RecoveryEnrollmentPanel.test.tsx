import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RecoveryEnrollmentPanel } from './RecoveryEnrollmentPanel';

const recoveryMock = vi.hoisted(() => ({
  startTotp: vi.fn(),
  completeTotp: vi.fn(),
  startWebauthn: vi.fn(),
  completeWebauthn: vi.fn(),
}));
const passkeyMock = vi.hoisted(() => ({
  createPasskeyCredential: vi.fn(),
  isWebAuthnSupported: vi.fn(() => true),
  WebAuthnCancelledError: class extends Error {},
}));
vi.mock('@/api/userRecovery', () => ({ userRecoveryApi: recoveryMock }));
vi.mock('@/utils/webauthn', () => passkeyMock);
vi.mock('../TotpEnrollmentInstructions', () => ({
  TotpEnrollmentInstructions: () => <div />,
  TotpManualSetup: () => <div />,
}));

describe('RecoveryEnrollmentPanel', () => {
  const restart = vi.fn();
  beforeEach(() => {
    vi.clearAllMocks();
    passkeyMock.isWebAuthnSupported.mockReturnValue(true);
  });
  afterEach(cleanup);
  it('finishes restricted TOTP enrollment, shows codes, then only restarts login', async () => {
    recoveryMock.startTotp.mockResolvedValue({
      secret: 'secret',
      otpauth_uri: 'otpauth://x',
      enrollment_token: 'enroll',
    });
    recoveryMock.completeTotp.mockResolvedValue({ success: true, recovery_codes: ['fresh-code'] });
    render(
      <RecoveryEnrollmentPanel
        recoveryToken="restricted"
        expiresAt="2026-01-01T00:00:00Z"
        allowedMethods={['totp']}
        onRestartLogin={restart}
      />,
    );
    await screen.findByLabelText('Verification code');
    fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } });
    fireEvent.click(screen.getByRole('button', { name: 'Finish setup' }));
    await screen.findByText('fresh-code');
    fireEvent.click(screen.getByRole('button', { name: 'Return to sign in' }));
    expect(recoveryMock.completeTotp).toHaveBeenCalledWith({
      recovery_token: 'restricted',
      enrollment_token: 'enroll',
      code: '123456',
    });
    expect(restart).toHaveBeenCalledOnce();
  });
  it('completes a WebAuthn recovery replacement without a session callback', async () => {
    recoveryMock.startWebauthn.mockResolvedValue({ options: {}, registration_token: 'register' });
    passkeyMock.createPasskeyCredential.mockResolvedValue({ id: 'credential' });
    recoveryMock.completeWebauthn.mockResolvedValue({
      success: true,
      recovery_codes: ['passkey-code'],
    });
    render(
      <RecoveryEnrollmentPanel
        recoveryToken="restricted"
        expiresAt="2026-01-01T00:00:00Z"
        allowedMethods={['webauthn']}
        onRestartLogin={restart}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Create passkey' }));
    await screen.findByText('passkey-code');
    expect(restart).not.toHaveBeenCalled();
  });
  it('offers normal login when a continuation is expired', async () => {
    recoveryMock.startTotp.mockRejectedValue(new Error('recovery continuation expired'));
    render(
      <RecoveryEnrollmentPanel
        recoveryToken="restricted"
        expiresAt="2026-01-01T00:00:00Z"
        allowedMethods={['totp']}
        onRestartLogin={restart}
      />,
    );
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: 'Return to sign in' }));
    expect(restart).toHaveBeenCalledOnce();
  });
});
