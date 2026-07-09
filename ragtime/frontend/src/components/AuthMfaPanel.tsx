import type { FormEvent } from 'react';

import { TotpInstructions, TotpManualSetup, TotpQrCard } from './TotpEnrollmentInstructions';

export type AuthMfaMode = 'verify' | 'enroll' | 'recovery';

interface AuthMfaPanelProps {
  mode: AuthMfaMode;
  error: string | null;
  isLoading: boolean;
  code: string;
  rememberDevice: boolean;
  totpSecret: string;
  otpauthUri: string;
  recoveryCodes: string[];
  recoveryContinueLabel?: string;
  onCodeChange: (value: string) => void;
  onRememberDeviceChange: (value: boolean) => void;
  onVerify: (event: FormEvent) => void;
  onEnrollComplete: (event: FormEvent) => void;
  onRecoveryContinue: () => void;
}

export function AuthMfaPanel({
  mode,
  error,
  isLoading,
  code,
  rememberDevice,
  totpSecret,
  otpauthUri,
  recoveryCodes,
  recoveryContinueLabel = 'Continue',
  onCodeChange,
  onRememberDeviceChange,
  onVerify,
  onEnrollComplete,
  onRecoveryContinue,
}: AuthMfaPanelProps) {
  if (mode === 'recovery') {
    return (
      <div className="login-form">
        <p className="login-info">Save these recovery codes now. They will not be shown again.</p>
        <div className="cloud-oauth-callback-code">
          {recoveryCodes.map((recoveryCode) => (
            <div key={recoveryCode}>{recoveryCode}</div>
          ))}
        </div>
        <button type="button" className="btn btn-primary login-submit" onClick={onRecoveryContinue}>
          {recoveryContinueLabel}
        </button>
      </div>
    );
  }

  const isEnrollment = mode === 'enroll';
  const formClassName = isEnrollment
    ? 'login-form totp-enrollment-form'
    : 'login-form mfa-challenge-form';
  return (
    <form onSubmit={isEnrollment ? onEnrollComplete : onVerify} className={formClassName}>
      {error && <div className="login-error">{error}</div>}
      {isEnrollment ? (
        <p className="login-info">Set up an authenticator app before continuing.</p>
      ) : (
        <div className="mfa-challenge-intro">
          <div className="mfa-challenge-mark" aria-hidden="true">
            6
          </div>
          <div>
            <h2 className="mfa-challenge-title">Two-step verification</h2>
            <p className="mfa-challenge-help">
              Use a 6-digit code from your authenticator app, or enter one of your recovery codes.
            </p>
          </div>
        </div>
      )}
      {isEnrollment && (
        <div className="totp-qr">
          <TotpQrCard otpauthUri={otpauthUri} />
        </div>
      )}
      <div className={isEnrollment ? 'form-group' : 'form-group mfa-code-group'}>
        <label htmlFor="mfa-code" className="form-label">
          {isEnrollment ? 'Verification code' : 'Authenticator or recovery code'}
        </label>
        <input
          type="text"
          id="mfa-code"
          className={isEnrollment ? 'form-input' : 'form-input mfa-code-input'}
          value={code}
          onChange={(event) => onCodeChange(event.target.value)}
          autoComplete="one-time-code"
          inputMode="numeric"
          autoFocus
          required
        />
      </div>
      <label className={isEnrollment ? 'checkbox-label' : 'checkbox-label mfa-remember-device'}>
        <input
          type="checkbox"
          checked={rememberDevice}
          onChange={(event) => onRememberDeviceChange(event.target.checked)}
        />
        Remember this device for 30 days
      </label>
      <button type="submit" className="btn btn-primary login-submit" disabled={isLoading || !code}>
        {isLoading
          ? isEnrollment
            ? 'Enrolling...'
            : 'Verifying...'
          : isEnrollment
            ? 'Finish setup'
            : 'Verify'}
      </button>
      {isEnrollment && (
        <>
          <TotpInstructions />
          <TotpManualSetup secret={totpSecret} otpauthUri={otpauthUri} />
        </>
      )}
    </form>
  );
}
