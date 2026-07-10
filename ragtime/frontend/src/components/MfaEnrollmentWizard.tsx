import { useCallback, useEffect, useRef, useState } from 'react';
import { KeyRound, Smartphone } from 'lucide-react';

import { api } from '@/api';
import type { MfaMethod } from '@/types';
import {
  createPasskeyCredential,
  isWebAuthnSupported,
  WebAuthnCancelledError,
} from '@/utils/webauthn';
import { TotpEnrollmentInstructions, TotpManualSetup } from './TotpEnrollmentInstructions';
import { RecoveryCodesDisplay } from './shared/RecoveryCodesDisplay';

interface MfaEnrollmentWizardProps {
  allowedMethods: MfaMethod[];
  /**
   * Methods the user has already enrolled. TOTP can only be enrolled once, so
   * an already-enrolled TOTP factor is disabled on the method-selection page.
   * Passkeys (webauthn) support multiple credentials and are never disabled.
   */
  enrolledMethods?: MfaMethod[];
  mfaChallengeToken?: string;
  onComplete(result: { recoveryCodes?: string[] }): void;
  serverName?: string;
}

const STEP_TITLES = ['Choose method', 'Set up', 'Recovery codes'];
const STEP_ORDER = ['method', 'enroll', 'recovery'] as const;

export function MfaEnrollmentWizard({
  allowedMethods,
  enrolledMethods = [],
  mfaChallengeToken,
  onComplete,
  serverName = 'Ragtime',
}: MfaEnrollmentWizardProps) {
  const webauthnSupported = isWebAuthnSupported();
  const totpAlreadyEnrolled = enrolledMethods.includes('totp');
  // TOTP is a singleton factor, so once enrolled it is no longer selectable.
  const isSelectable = (method: MfaMethod) => !(method === 'totp' && totpAlreadyEnrolled);
  // Only auto-skip the selection page when there is genuinely a single allowed
  // method AND it is still enrollable. When multiple methods are allowed we keep
  // the selection page visible so already-enrolled methods show as disabled.
  const singleAllowedMethod = allowedMethods.length === 1 ? allowedMethods[0] : null;
  const singleMethod =
    singleAllowedMethod && isSelectable(singleAllowedMethod) ? singleAllowedMethod : null;

  const [currentStep, setCurrentStep] = useState<'method' | 'enroll' | 'recovery'>(
    singleMethod ? 'enroll' : 'method',
  );
  const [selectedMethod, setSelectedMethod] = useState<MfaMethod | null>(singleMethod);

  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [rememberDevice, setRememberDevice] = useState(true);

  // TOTP enrollment state
  const [totpSecret, setTotpSecret] = useState('');
  const [otpauthUri, setOtpauthUri] = useState('');
  const [enrollmentToken, setEnrollmentToken] = useState('');
  const [totpCode, setTotpCode] = useState('');

  // Passkey enrollment state
  const [passkeyName, setPasskeyName] = useState('Passkey');
  const [passkeyError, setPasskeyError] = useState<string | null>(null);

  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);

  const startTotpEnrollment = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const setup = await api.startMfaEnrollment(mfaChallengeToken);
      setTotpSecret(setup.secret);
      setOtpauthUri(setup.otpauth_uri);
      setEnrollmentToken(setup.enrollment_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start authenticator setup');
    } finally {
      setIsLoading(false);
    }
  }, [mfaChallengeToken]);

  useEffect(() => {
    if (singleMethod === 'totp') {
      void startTotpEnrollment();
    }
  }, [singleMethod, startTotpEnrollment]);

  const stepIndex = currentStep === 'method' ? 0 : currentStep === 'enroll' ? 1 : 2;
  const activeStepRef = useRef<HTMLButtonElement | null>(null);

  // Keep the current step (and the one before it) visible when the progress bar
  // overflows horizontally by scrolling the active step to the trailing edge.
  useEffect(() => {
    // scrollIntoView is unavailable in some environments (e.g. jsdom); guard it.
    if (typeof activeStepRef.current?.scrollIntoView === 'function') {
      activeStepRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'end',
      });
    }
  }, [stepIndex]);

  if (singleMethod === 'webauthn' && !webauthnSupported) {
    return (
      <div className="login-form">
        <div className="login-error">
          Your browser or device does not support passkeys. Sign in from a device that supports
          passkeys, or contact your administrator.
        </div>
      </div>
    );
  }

  const completeTotpEnrollment = async () => {
    if (!enrollmentToken || !totpCode) return;
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.completeMfaEnrollment({
        mfa_challenge_token: mfaChallengeToken,
        enrollment_token: enrollmentToken,
        code: totpCode,
        remember_device: rememberDevice,
      });
      if (result.recovery_codes && result.recovery_codes.length > 0) {
        setRecoveryCodes(result.recovery_codes);
        setCurrentStep('recovery');
      } else {
        onComplete({});
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to finish authenticator setup');
    } finally {
      setIsLoading(false);
    }
  };

  const createPasskey = async () => {
    setPasskeyError(null);
    setIsLoading(true);
    try {
      const { options, registration_token: registrationToken } =
        await api.startWebauthnRegistration(mfaChallengeToken);
      const credential = await createPasskeyCredential(options);
      const result = await api.completeWebauthnRegistration({
        registration_token: registrationToken,
        credential,
        name: passkeyName.trim() || 'Passkey',
        mfa_challenge_token: mfaChallengeToken,
        remember_device: rememberDevice,
      });
      if (result.recovery_codes && result.recovery_codes.length > 0) {
        setRecoveryCodes(result.recovery_codes);
        setCurrentStep('recovery');
      } else {
        onComplete({});
      }
    } catch (err) {
      if (err instanceof WebAuthnCancelledError) {
        setPasskeyError('Passkey setup was cancelled.');
      } else {
        const msg = err instanceof Error ? err.message : 'Failed to create passkey';
        setPasskeyError(msg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectMethod = (method: MfaMethod) => {
    setSelectedMethod(method);
    if (method === 'totp') {
      void startTotpEnrollment();
    }
    setCurrentStep('enroll');
  };

  // Allow stepping backward through the wizard, but never out of the recovery
  // step (enrollment is already committed once recovery codes are shown).
  const goToStep = (index: number) => {
    if (index >= stepIndex || currentStep === 'recovery') return;
    setError(null);
    setCurrentStep(STEP_ORDER[index]);
  };

  const renderMethodSelection = () => (
    <div className="tool-type-selection" style={{ marginTop: 'var(--space-md)' }}>
      <p className="login-info">Choose how you want to secure your {serverName} account.</p>
      {allowedMethods.includes('totp') &&
        renderMethodCard(
          'totp',
          <Smartphone size={20} />,
          'Authenticator app',
          'Use an app like Google Authenticator, 1Password, or Duo Mobile.',
          totpAlreadyEnrolled,
          totpAlreadyEnrolled
            ? 'Already enabled. Ask an administrator to reset MFA before re-enrolling.'
            : undefined,
        )}
      {allowedMethods.includes('webauthn') &&
        renderMethodCard(
          'webauthn',
          <KeyRound size={20} />,
          'Passkey',
          'Use your device biometric, a security key, or your phone or tablet.',
          !webauthnSupported,
          !webauthnSupported ? 'Your browser or device does not support passkeys.' : undefined,
        )}
    </div>
  );

  const renderMethodCard = (
    method: MfaMethod,
    icon: React.ReactNode,
    title: string,
    description: string,
    disabled = false,
    hint?: string,
  ) => (
    <button
      key={method}
      type="button"
      className={`tool-type-option ${selectedMethod === method ? 'selected' : ''} ${disabled ? 'disabled' : ''}`}
      onClick={() => {
        if (!disabled) {
          handleSelectMethod(method);
        }
      }}
      disabled={disabled || isLoading}
    >
      <div className="tool-type-option-icon">{icon}</div>
      <div style={{ minWidth: 0 }}>
        <span className="tool-type-option-name">{title}</span>
        <span className="tool-type-option-desc">{description}</span>
        {hint && (
          <span className="field-help" style={{ display: 'block', marginTop: '4px' }}>
            {hint}
          </span>
        )}
      </div>
    </button>
  );

  const renderEnrollment = () => {
    if (selectedMethod === 'totp') {
      return (
        <div className="totp-enrollment-form">
          {error && <div className="login-error">{error}</div>}
          <h1 className="login-info">Set up an authenticator app before continuing.</h1>
          <div className="totp-qr">
            <TotpEnrollmentInstructions otpauthUri={otpauthUri} />
          </div>
          <div className="totp-enrollment-fields">
            <div className="form-group">
              <label htmlFor="mfa-code" className="form-label">
                Verification code
              </label>
              <input
                type="text"
                id="mfa-code"
                className="form-input"
                value={totpCode}
                onChange={(event) => setTotpCode(event.target.value)}
                autoComplete="one-time-code"
                inputMode="numeric"
                autoFocus
                required
              />
            </div>
            {mfaChallengeToken && (
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={rememberDevice}
                  onChange={(event) => setRememberDevice(event.target.checked)}
                />
                Remember this device for 30 days
              </label>
            )}
            <button
              type="button"
              className="btn btn-primary login-submit"
              disabled={isLoading || !totpCode}
              onClick={() => void completeTotpEnrollment()}
            >
              {isLoading ? 'Enrolling...' : 'Finish setup'}
            </button>
            {!singleMethod && (
              <button
                type="button"
                className="btn btn-secondary login-submit"
                disabled={isLoading}
                onClick={() => setCurrentStep('method')}
              >
                Choose a different method
              </button>
            )}
          </div>
          <TotpManualSetup secret={totpSecret} otpauthUri={otpauthUri} />
        </div>
      );
    }

    if (selectedMethod === 'webauthn') {
      return (
        <div className="login-form webauthn-enrollment-form">
          {error && <div className="login-error">{error}</div>}
          {passkeyError && <div className="login-error">{passkeyError}</div>}
          <p className="login-info">
            Create a passkey for this device, or use your phone or tablet. Your browser will show a
            QR code to scan when you choose that option.
          </p>
          <div className="form-group">
            <label htmlFor="passkey-name" className="form-label">
              Passkey name
            </label>
            <input
              type="text"
              id="passkey-name"
              className="form-input"
              value={passkeyName}
              onChange={(event) => setPasskeyName(event.target.value)}
              placeholder="e.g. MacBook Touch ID"
            />
          </div>
          {mfaChallengeToken && (
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={rememberDevice}
                onChange={(event) => setRememberDevice(event.target.checked)}
              />
              Remember this device for 30 days
            </label>
          )}
          <button
            type="button"
            className="btn btn-primary login-submit"
            disabled={isLoading || !passkeyName.trim()}
            onClick={() => void createPasskey()}
          >
            {isLoading ? 'Creating passkey...' : 'Create passkey'}
          </button>
          {!singleMethod && (
            <button
              type="button"
              className="btn btn-secondary login-submit"
              disabled={isLoading}
              onClick={() => setCurrentStep('method')}
            >
              Choose a different method
            </button>
          )}
        </div>
      );
    }

    return null;
  };

  const renderRecoveryCodes = () => (
    <div className="login-form">
      <p className="login-info">Save these recovery codes now. They will not be shown again.</p>
      <RecoveryCodesDisplay codes={recoveryCodes} />
      <div className="wizard-actions wizard-actions-borderless">
        <button
          type="button"
          className="btn btn-primary login-submit"
          onClick={() => onComplete({ recoveryCodes })}
        >
          Done
        </button>
      </div>
    </div>
  );

  return (
    <div>
      {!singleMethod && (
        <div className="wizard-progress">
          {STEP_TITLES.map((title, index) => {
            const navigable = index < stepIndex && currentStep !== 'recovery';
            return (
              <button
                key={title}
                ref={index === stepIndex ? activeStepRef : undefined}
                type="button"
                className={`wizard-step ${index === stepIndex ? 'active' : ''} ${index < stepIndex ? 'completed' : ''} ${navigable ? 'navigable' : ''}`}
                onClick={() => goToStep(index)}
                disabled={!navigable}
                aria-current={index === stepIndex ? 'step' : undefined}
              >
                <span className="step-number">{index + 1}</span>
                <span className="step-title">{title}</span>
              </button>
            );
          })}
        </div>
      )}
      {currentStep === 'method' && renderMethodSelection()}
      {currentStep === 'enroll' && renderEnrollment()}
      {currentStep === 'recovery' && renderRecoveryCodes()}
    </div>
  );
}
