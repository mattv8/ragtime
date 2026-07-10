import { useState } from 'react';

import { api } from '@/api';
import type { MfaMethod } from '@/types';
import { getPasskeyAssertion, isWebAuthnSupported, WebAuthnCancelledError } from '@/utils/webauthn';
import { MfaEnrollmentWizard } from './MfaEnrollmentWizard';
import { RecoveryCodesDisplay } from './shared/RecoveryCodesDisplay';

export type AuthMfaMode = 'verify' | 'enroll' | 'recovery';

interface AuthMfaPanelProps {
  mode: AuthMfaMode;
  error: string | null;
  isLoading: boolean;
  code: string;
  rememberDevice: boolean;
  recoveryCodes: string[];
  recoveryContinueLabel?: string;
  methods?: MfaMethod[];
  preferredMethod?: MfaMethod | null;
  mfaChallengeToken?: string;
  serverName?: string;
  onCodeChange: (value: string) => void;
  onRememberDeviceChange: (value: boolean) => void;
  onVerify: () => void | Promise<void>;
  /**
   * Called after a passkey ceremony has fully verified the challenge and the
   * session cookie is already set (no further code verification is needed).
   */
  onVerified: () => void | Promise<void>;
  onEnrollComplete: () => void | Promise<void>;
  onRecoveryContinue: () => void;
}

export function AuthMfaPanel({
  mode,
  error: parentError,
  isLoading,
  code,
  rememberDevice,
  recoveryCodes,
  recoveryContinueLabel = 'Continue',
  methods = ['totp'],
  preferredMethod = null,
  mfaChallengeToken,
  serverName = 'Ragtime',
  onCodeChange,
  onRememberDeviceChange,
  onVerify,
  onVerified,
  onEnrollComplete,
  onRecoveryContinue,
}: AuthMfaPanelProps) {
  const [verifyView, setVerifyView] = useState<'passkey' | 'totp'>(() => {
    if (preferredMethod === 'webauthn' && methods.includes('webauthn') && isWebAuthnSupported()) {
      return 'passkey';
    }
    if (preferredMethod === 'totp' && methods.includes('totp')) {
      return 'totp';
    }
    return methods.includes('webauthn') && isWebAuthnSupported() ? 'passkey' : 'totp';
  });
  const [passkeyError, setPasskeyError] = useState<string | null>(null);
  const [passkeyLoading, setPasskeyLoading] = useState(false);

  const error = parentError || passkeyError;

  const handlePasskeyVerify = async () => {
    if (!mfaChallengeToken) return;
    setPasskeyError(null);
    setPasskeyLoading(true);
    try {
      const { options, authentication_token: authenticationToken } =
        await api.startWebauthnAuthentication(mfaChallengeToken);
      const credential = await getPasskeyAssertion(options);
      const response = await api.completeWebauthnAuthentication({
        mfa_challenge_token: mfaChallengeToken,
        authentication_token: authenticationToken,
        credential,
        remember_device: rememberDevice,
      });
      if (!response.success) {
        setPasskeyError(response.error || 'Passkey verification failed');
        return;
      }
      await onVerified();
    } catch (err) {
      if (err instanceof WebAuthnCancelledError) {
        setPasskeyError('Passkey verification was cancelled.');
      } else if (err instanceof Error && typeof err.message === 'string') {
        setPasskeyError(err.message);
      } else {
        setPasskeyError('Passkey verification failed');
      }
    } finally {
      setPasskeyLoading(false);
    }
  };

  const handleWizardComplete = () => {
    void onEnrollComplete();
  };

  const isWebauthnOnlyUnsupported =
    methods.length === 1 && methods[0] === 'webauthn' && !isWebAuthnSupported();

  const showingRecovery = mode === 'recovery';
  const displayedRecoveryCodes = recoveryCodes;

  if (showingRecovery) {
    return (
      <div className="login-form">
        <p className="login-info">Save these recovery codes now. They will not be shown again.</p>
        <RecoveryCodesDisplay codes={displayedRecoveryCodes} />
        <button type="button" className="btn btn-primary login-submit" onClick={onRecoveryContinue}>
          {recoveryContinueLabel}
        </button>
      </div>
    );
  }

  if (mode === 'enroll') {
    return (
      <div className="login-form">
        {parentError && <div className="login-error">{parentError}</div>}
        <MfaEnrollmentWizard
          allowedMethods={methods}
          mfaChallengeToken={mfaChallengeToken}
          onComplete={handleWizardComplete}
          serverName={serverName}
        />
      </div>
    );
  }

  const formClassName = 'login-form mfa-challenge-form';

  const showPasskeyOption = methods.includes('webauthn') && isWebAuthnSupported();

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (verifyView === 'totp') {
          void onVerify();
        }
      }}
      className={formClassName}
    >
      {error && <div className="login-error">{error}</div>}
      {isWebauthnOnlyUnsupported ? (
        <div className="login-error">
          Your browser or device does not support passkeys. Sign in from a device that supports
          passkeys, or contact your administrator.
        </div>
      ) : (
        <h2 className="mfa-challenge-title mfa-challenge-title-borderless">
          Two-step verification
        </h2>
      )}

      {verifyView === 'totp' && (
        <>
          <div className="form-group mfa-code-group">
            <label htmlFor="mfa-code" className="form-label">
              {methods.includes('totp') ? 'Authenticator or recovery code' : 'Recovery code'}
            </label>
            <input
              type="text"
              id="mfa-code"
              className="form-input mfa-code-input"
              value={code}
              onChange={(event) => onCodeChange(event.target.value)}
              autoComplete="one-time-code"
              inputMode="numeric"
              autoFocus
              required
            />
          </div>
          <label className="checkbox-label mfa-remember-device">
            <input
              type="checkbox"
              checked={rememberDevice}
              onChange={(event) => onRememberDeviceChange(event.target.checked)}
            />
            Remember this device for 30 days
          </label>
          <button
            type="submit"
            className="btn btn-primary login-submit"
            disabled={isLoading || !code}
          >
            {isLoading ? 'Verifying...' : 'Verify'}
          </button>
          {showPasskeyOption && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              onClick={() => {
                setVerifyView('passkey');
                setPasskeyError(null);
              }}
            >
              Use a passkey instead
            </button>
          )}
        </>
      )}

      {verifyView === 'passkey' && showPasskeyOption && (
        <>
          <label className="checkbox-label mfa-remember-device">
            <input
              type="checkbox"
              checked={rememberDevice}
              onChange={(event) => onRememberDeviceChange(event.target.checked)}
            />
            Remember this device for 30 days
          </label>
          <button
            type="button"
            className="btn btn-primary login-submit"
            disabled={passkeyLoading}
            onClick={() => void handlePasskeyVerify()}
          >
            {passkeyLoading ? 'Verifying...' : 'Use passkey'}
          </button>
          {methods.includes('totp') && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              onClick={() => {
                setVerifyView('totp');
                setPasskeyError(null);
              }}
            >
              Use authenticator code instead
            </button>
          )}
        </>
      )}

      {verifyView === 'passkey' && !showPasskeyOption && methods.includes('totp') && (
        <div className="login-error">
          Your browser or device does not support passkeys. Use your authenticator code instead.
        </div>
      )}
    </form>
  );
}
