import type { MfaMethod } from '@/types';
import { AuthMfaPanel } from '../AuthMfaPanel';
import { RecoveryEnrollmentPanel } from './RecoveryEnrollmentPanel';
import { userRecoveryApi } from '@/api/userRecovery';
import { useEffect, useState } from 'react';

interface LoginMfaPanelProps {
  mode: 'verify' | 'enroll' | 'recovery';
  error: string | null;
  isLoading: boolean;
  code: string;
  rememberDevice: boolean;
  methods: MfaMethod[];
  preferredMethod: MfaMethod | null;
  mfaChallengeToken?: string;
  serverName: string;
  onCodeChange: (code: string) => void;
  onRememberDeviceChange: (remember: boolean) => void;
  onVerify: () => void;
  onSessionEstablished: () => void;
  onRecoveryContinue: () => void;
  recoveryContinueLabel?: string;
  onRestartLogin?: () => void;
}

export function LoginMfaPanel({
  mode,
  error,
  isLoading,
  code,
  rememberDevice,
  methods,
  preferredMethod,
  mfaChallengeToken,
  serverName,
  onCodeChange,
  onRememberDeviceChange,
  onVerify,
  onSessionEstablished,
  onRecoveryContinue,
  recoveryContinueLabel,
  onRestartLogin,
}: LoginMfaPanelProps) {
  const [showRecoveryPass, setShowRecoveryPass] = useState(false);
  const [recoveryPass, setRecoveryPass] = useState('');
  const [recoveryToken, setRecoveryToken] = useState<string | null>(null);
  const [recoveryMethods, setRecoveryMethods] = useState<MfaMethod[]>([]);
  const [recoveryExpiresAt, setRecoveryExpiresAt] = useState('');
  const [recoveryError, setRecoveryError] = useState<string | null>(null);
  const [recoveryBusy, setRecoveryBusy] = useState(false);

  useEffect(() => {
    setShowRecoveryPass(false);
    setRecoveryPass('');
    setRecoveryToken(null);
    setRecoveryMethods([]);
    setRecoveryExpiresAt('');
    setRecoveryError(null);
  }, [mfaChallengeToken]);

  if (recoveryToken && onRestartLogin) {
    return (
      <RecoveryEnrollmentPanel
        recoveryToken={recoveryToken}
        expiresAt={recoveryExpiresAt}
        allowedMethods={recoveryMethods}
        onRestartLogin={onRestartLogin}
      />
    );
  }

  const redeemRecoveryPass = async () => {
    if (!mfaChallengeToken || !recoveryPass) return;
    setRecoveryBusy(true);
    setRecoveryError(null);
    try {
      const result = await userRecoveryApi.redeem({
        mfa_challenge_token: mfaChallengeToken,
        pass: recoveryPass,
      });
      setRecoveryPass('');
      setRecoveryMethods(result.allowed_methods);
      setRecoveryExpiresAt(result.expires_at);
      setRecoveryToken(result.recovery_token);
    } catch (err) {
      setRecoveryError(err instanceof Error ? err.message : 'Recovery pass could not be verified');
    } finally {
      setRecoveryBusy(false);
    }
  };

  if (mode === 'verify' && showRecoveryPass) {
    return (
      <section id="login-admin-recovery-pass" className="login-form" aria-busy={recoveryBusy}>
        <p className="login-info">
          Enter the one-time recovery pass provided by your administrator. You must then set up a
          replacement factor.
        </p>
        {recoveryError && (
          <div className="login-error" role="alert">
            {recoveryError}
          </div>
        )}
        <label className="form-label" htmlFor="admin-recovery-pass">
          Administrator recovery pass
        </label>
        <input
          id="admin-recovery-pass"
          className="form-input"
          type="password"
          value={recoveryPass}
          onChange={(event) => setRecoveryPass(event.target.value)}
          autoComplete="off"
        />
        <button
          type="button"
          className="btn btn-primary login-submit"
          disabled={recoveryBusy || !recoveryPass}
          onClick={() => void redeemRecoveryPass()}
        >
          {recoveryBusy ? 'Verifying...' : 'Continue'}
        </button>
        <button
          type="button"
          className="btn btn-secondary login-submit"
          disabled={recoveryBusy}
          onClick={() => {
            setRecoveryPass('');
            setRecoveryError(null);
            setShowRecoveryPass(false);
          }}
        >
          Use another method
        </button>
      </section>
    );
  }
  return (
    <>
      <AuthMfaPanel
        mode={mode}
        error={error}
        isLoading={isLoading}
        code={code}
        rememberDevice={rememberDevice}
        recoveryCodes={[]}
        {...(recoveryContinueLabel ? { recoveryContinueLabel } : {})}
        methods={methods}
        preferredMethod={preferredMethod}
        mfaChallengeToken={mfaChallengeToken}
        serverName={serverName}
        onCodeChange={onCodeChange}
        onRememberDeviceChange={onRememberDeviceChange}
        onVerify={onVerify}
        onVerified={onSessionEstablished}
        onEnrollComplete={onSessionEstablished}
        onRecoveryContinue={onRecoveryContinue}
      />
      {mode === 'verify' && onRestartLogin && (
        <button type="button" className="btn btn-link" onClick={() => setShowRecoveryPass(true)}>
          Use administrator recovery pass
        </button>
      )}
    </>
  );
}
