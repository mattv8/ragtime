import type { FormEvent } from 'react';

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
  return (
    <form onSubmit={isEnrollment ? onEnrollComplete : onVerify} className="login-form">
      {error && <div className="login-error">{error}</div>}
      <p className="login-info">
        {isEnrollment
          ? 'Set up an authenticator app before continuing.'
          : 'Enter your authenticator code or a recovery code.'}
      </p>
      {isEnrollment && (
        <>
          <div className="form-group">
            <label className="form-label">Manual setup key</label>
            <code className="cloud-oauth-callback-code">{totpSecret}</code>
          </div>
          <div className="form-group">
            <label className="form-label">Authenticator URI</label>
            <code className="cloud-oauth-callback-code">{otpauthUri}</code>
          </div>
        </>
      )}
      <div className="form-group">
        <label htmlFor="mfa-code" className="form-label">
          {isEnrollment ? 'Verification code' : 'MFA code'}
        </label>
        <input
          id="mfa-code"
          className="form-input"
          value={code}
          onChange={(event) => onCodeChange(event.target.value)}
          autoComplete="one-time-code"
          autoFocus
          required
        />
      </div>
      <label className="checkbox-label">
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
    </form>
  );
}
