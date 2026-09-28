import { useEffect, useRef, useState } from 'react';
import { userRecoveryApi } from '@/api/userRecovery';
import type { MfaMethod } from '@/types';
import {
  createPasskeyCredential,
  isWebAuthnSupported,
  WebAuthnCancelledError,
} from '@/utils/webauthn';
import { TotpEnrollmentInstructions, TotpManualSetup } from '../TotpEnrollmentInstructions';
import { RecoveryCodesDisplay } from './RecoveryCodesDisplay';

interface RecoveryEnrollmentPanelProps {
  recoveryToken: string;
  expiresAt: string;
  allowedMethods: MfaMethod[];
  onRestartLogin: () => void;
}

const errorMessage = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback;
const continuationInvalid = (error: unknown) =>
  /expired|revoked|invalid.*continuation|recovery.*token/i.test(errorMessage(error, ''));

/** Restricted MFA replacement flow; it never establishes an application session. */
export function RecoveryEnrollmentPanel({
  recoveryToken,
  expiresAt,
  allowedMethods,
  onRestartLogin,
}: RecoveryEnrollmentPanelProps) {
  const [method, setMethod] = useState<MfaMethod | null>(
    allowedMethods.length === 1 ? allowedMethods[0] : null,
  );
  const [setup, setSetup] = useState<{
    secret: string;
    otpauth_uri: string;
    enrollment_token: string;
  } | null>(null);
  const [code, setCode] = useState('');
  const [name, setName] = useState('Passkey');
  const [codes, setCodes] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);
  const tokenRef = useRef(recoveryToken);

  useEffect(() => {
    tokenRef.current = recoveryToken;
    setMethod(allowedMethods.length === 1 ? allowedMethods[0] : null);
    setSetup(null);
    setCode('');
    setCodes(null);
    setError(null);
    setInvalid(false);
    setBusy(false);
  }, [allowedMethods, recoveryToken]);

  const startTotp = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await userRecoveryApi.startTotp(recoveryToken);
      if (tokenRef.current === recoveryToken) setSetup(next);
    } catch (err) {
      if (tokenRef.current !== recoveryToken) return;
      setInvalid(continuationInvalid(err));
      setError(errorMessage(err, 'Could not start authenticator setup'));
    } finally {
      if (tokenRef.current === recoveryToken) setBusy(false);
    }
  };
  // Only method/token changes start setup; retry is an explicit user action.
  useEffect(() => {
    if (method === 'totp' && !setup && !invalid) void startTotp();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method, recoveryToken]); // setup/retry intentionally invoke explicitly

  const finishTotp = async () => {
    if (!setup || !code) return;
    setBusy(true);
    setError(null);
    try {
      const result = await userRecoveryApi.completeTotp({
        recovery_token: recoveryToken,
        enrollment_token: setup.enrollment_token,
        code,
      });
      if (tokenRef.current === recoveryToken) setCodes(result.recovery_codes);
    } catch (err) {
      if (tokenRef.current !== recoveryToken) return;
      setInvalid(continuationInvalid(err));
      setError(errorMessage(err, 'Could not finish authenticator setup'));
    } finally {
      if (tokenRef.current === recoveryToken) setBusy(false);
    }
  };
  const finishPasskey = async () => {
    setBusy(true);
    setError(null);
    try {
      const start = await userRecoveryApi.startWebauthn(recoveryToken);
      const credential = await createPasskeyCredential(start.options);
      const result = await userRecoveryApi.completeWebauthn({
        recovery_token: recoveryToken,
        registration_token: start.registration_token,
        credential,
        name: name.trim() || 'Passkey',
      });
      if (tokenRef.current === recoveryToken) setCodes(result.recovery_codes);
    } catch (err) {
      if (tokenRef.current !== recoveryToken) return;
      setInvalid(continuationInvalid(err));
      setError(
        err instanceof WebAuthnCancelledError
          ? 'Passkey setup was cancelled.'
          : errorMessage(err, 'Could not create passkey'),
      );
    } finally {
      if (tokenRef.current === recoveryToken) setBusy(false);
    }
  };
  const choose = (next: MfaMethod | null) => {
    setError(null);
    setInvalid(false);
    setSetup(null);
    setMethod(next);
  };
  const canChooseAnother = allowedMethods.length > 1;

  if (codes)
    return (
      <section id="recovery-enrollment-codes" className="login-form">
        <p className="login-info">Save these new recovery codes. They will not be shown again.</p>
        <RecoveryCodesDisplay codes={codes} />
        <button type="button" className="btn btn-primary login-submit" onClick={onRestartLogin}>
          Return to sign in
        </button>
      </section>
    );
  if (invalid)
    return (
      <section id="recovery-enrollment-invalid" className="login-form">
        <div className="login-error" role="alert">
          {error}
        </div>
        <button type="button" className="btn btn-primary login-submit" onClick={onRestartLogin}>
          Return to sign in
        </button>
      </section>
    );
  return (
    <section id="recovery-enrollment" className="login-form" aria-busy={busy}>
      <p className="login-info">
        Replacement enrollment expires {new Date(expiresAt).toLocaleString()}.
      </p>
      {error && (
        <div className="login-error" role="alert">
          {error}
        </div>
      )}
      {!method && (
        <>
          <p className="login-info">Choose a replacement verification method.</p>
          {allowedMethods.includes('totp') && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              onClick={() => choose('totp')}
            >
              Use an authenticator app
            </button>
          )}
          {allowedMethods.includes('webauthn') && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              disabled={!isWebAuthnSupported()}
              onClick={() => choose('webauthn')}
            >
              Use a passkey
            </button>
          )}
          {allowedMethods.length === 1 &&
            allowedMethods[0] === 'webauthn' &&
            !isWebAuthnSupported() && (
              <div className="login-error" role="alert">
                This browser does not support the required passkey enrollment.
              </div>
            )}
        </>
      )}
      {method === 'totp' && (
        <>
          <p className="login-info">Set up your authenticator app, then enter its code.</p>
          {setup ? (
            <>
              <TotpEnrollmentInstructions otpauthUri={setup.otpauth_uri} />
              <TotpManualSetup secret={setup.secret} otpauthUri={setup.otpauth_uri} />
            </>
          ) : (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              disabled={busy}
              onClick={() => void startTotp()}
            >
              Retry setup
            </button>
          )}
          <label className="form-label" htmlFor="recovery-totp-code">
            Verification code
          </label>
          <input
            id="recovery-totp-code"
            className="form-input"
            value={code}
            onChange={(event) => setCode(event.target.value)}
            autoComplete="one-time-code"
          />
          <button
            type="button"
            className="btn btn-primary login-submit"
            disabled={busy || !setup || !code}
            onClick={() => void finishTotp()}
          >
            Finish setup
          </button>
          {canChooseAnother && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              disabled={busy}
              onClick={() => choose(null)}
            >
              Choose another method
            </button>
          )}
        </>
      )}
      {method === 'webauthn' && (
        <>
          <label className="form-label" htmlFor="recovery-passkey-name">
            Passkey name
          </label>
          <input
            id="recovery-passkey-name"
            className="form-input"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <button
            type="button"
            className="btn btn-primary login-submit"
            disabled={busy || !name.trim()}
            onClick={() => void finishPasskey()}
          >
            Create passkey
          </button>
          {canChooseAnother && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              disabled={busy}
              onClick={() => choose(null)}
            >
              Choose another method
            </button>
          )}
        </>
      )}
    </section>
  );
}
