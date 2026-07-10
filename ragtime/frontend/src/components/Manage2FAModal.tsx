import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Pencil, Shield, X } from 'lucide-react';

import type { MfaMethod, MfaStatusResponse, WebauthnCredentialSummary } from '@/types';
import { api } from '@/api';
import { createPasskeyCredential, isWebAuthnSupported } from '@/utils/webauthn';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { MfaEnrollmentWizard } from './MfaEnrollmentWizard';
import { TotpEnrollmentInstructions, TotpManualSetup } from './TotpEnrollmentInstructions';
import { RecoveryCodesDisplay } from './shared/RecoveryCodesDisplay';

interface Manage2FAModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type Manage2FATab = 'authenticator' | 'passkeys' | 'recovery';

export function Manage2FAModal({ isOpen, onClose }: Manage2FAModalProps) {
  const [status, setStatus] = useState<MfaStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [displayedRecoveryCodes, setDisplayedRecoveryCodes] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<Manage2FATab>('authenticator');

  // Replace authenticator flow
  const [rotationStep, setRotationStep] = useState<'start' | 'verify'>('start');
  const [rotationStepUpCode, setRotationStepUpCode] = useState('');
  const [rotationSecret, setRotationSecret] = useState('');
  const [rotationOtpauthUri, setRotationOtpauthUri] = useState('');
  const [rotationEnrollmentToken, setRotationEnrollmentToken] = useState('');
  const [rotationVerifyCode, setRotationVerifyCode] = useState('');
  const [rotationLoading, setRotationLoading] = useState(false);

  // Regenerate recovery codes
  const [regenerateVisible, setRegenerateVisible] = useState(false);
  const [regenerateStepUpCode, setRegenerateStepUpCode] = useState('');
  const [regenerateLoading, setRegenerateLoading] = useState(false);

  // Passkeys
  const [passkeys, setPasskeys] = useState<WebauthnCredentialSummary[]>([]);
  const [passkeysLoading, setPasskeysLoading] = useState(false);
  const [passkeyName, setPasskeyName] = useState('Passkey');
  const [editingPasskeyId, setEditingPasskeyId] = useState<string | null>(null);
  const [editingPasskeyName, setEditingPasskeyName] = useState('');
  const [deletingPasskeyId, setDeletingPasskeyId] = useState<string | null>(null);
  const cancelledPasskeyEditRef = useRef(false);

  const resetTransientState = useCallback(() => {
    setError(null);
    setSuccess(null);
    setDisplayedRecoveryCodes([]);
    setActiveTab('authenticator');

    setRotationStep('start');
    setRotationStepUpCode('');
    setRotationSecret('');
    setRotationOtpauthUri('');
    setRotationEnrollmentToken('');
    setRotationVerifyCode('');
    setRotationLoading(false);

    setRegenerateVisible(false);
    setRegenerateStepUpCode('');
    setRegenerateLoading(false);

    setPasskeys([]);
    setPasskeyName('Passkey');
    setEditingPasskeyId(null);
    setEditingPasskeyName('');
    setDeletingPasskeyId(null);
  }, []);

  const loadStatus = useCallback(async () => {
    setStatusLoading(true);
    setError(null);
    try {
      const nextStatus = await api.getMfaStatus();
      setStatus(nextStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load MFA status');
    } finally {
      setStatusLoading(false);
    }
  }, []);

  const refreshPasskeys = useCallback(async (): Promise<WebauthnCredentialSummary[] | null> => {
    setPasskeysLoading(true);
    try {
      const result = await api.listWebauthnCredentials();
      setPasskeys(result.credentials);
      return result.credentials;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load passkeys');
      return null;
    } finally {
      setPasskeysLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) {
      resetTransientState();
      setStatus(null);
      return;
    }
    resetTransientState();
    void loadStatus();
    void refreshPasskeys();
  }, [isOpen, loadStatus, refreshPasskeys, resetTransientState]);

  const handleWizardComplete = (result: { recoveryCodes?: string[] }) => {
    if (result.recoveryCodes && result.recoveryCodes.length > 0) {
      setDisplayedRecoveryCodes(result.recoveryCodes);
    } else {
      void loadStatus();
      void refreshPasskeys();
    }
  };

  const startRotation = async () => {
    if (!rotationStepUpCode.trim()) return;
    setRotationLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const setup = await api.startTotpRotation(rotationStepUpCode.trim());
      setRotationSecret(setup.secret);
      setRotationOtpauthUri(setup.otpauth_uri);
      setRotationEnrollmentToken(setup.enrollment_token);
      setRotationStep('verify');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start authenticator replacement');
    } finally {
      setRotationLoading(false);
    }
  };

  const completeRotation = async () => {
    if (!rotationEnrollmentToken || !rotationVerifyCode.trim()) return;
    setRotationLoading(true);
    setError(null);
    setSuccess(null);
    try {
      await api.completeTotpRotation({
        enrollment_token: rotationEnrollmentToken,
        code: rotationVerifyCode.trim(),
      });
      setRotationStep('start');
      setRotationStepUpCode('');
      setRotationSecret('');
      setRotationOtpauthUri('');
      setRotationEnrollmentToken('');
      setRotationVerifyCode('');
      setSuccess('Authenticator app replaced successfully.');
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to confirm authenticator replacement');
    } finally {
      setRotationLoading(false);
    }
  };

  const cancelRotation = () => {
    setRotationStep('start');
    setRotationStepUpCode('');
    setRotationSecret('');
    setRotationOtpauthUri('');
    setRotationEnrollmentToken('');
    setRotationVerifyCode('');
    setError(null);
  };

  const handleRegenerate = async () => {
    if (!regenerateStepUpCode.trim()) return;
    setRegenerateLoading(true);
    setError(null);
    try {
      const result = await api.regenerateRecoveryCodes(regenerateStepUpCode.trim());
      setDisplayedRecoveryCodes(result.recovery_codes);
      setRegenerateVisible(false);
      setRegenerateStepUpCode('');
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to regenerate recovery codes');
    } finally {
      setRegenerateLoading(false);
    }
  };

  const handleAddPasskey = async () => {
    setError(null);
    setDisplayedRecoveryCodes([]);
    setPasskeysLoading(true);
    try {
      const { options, registration_token: registrationToken } =
        await api.startWebauthnRegistration();
      const credential = await createPasskeyCredential(options);
      const result = await api.completeWebauthnRegistration({
        registration_token: registrationToken,
        credential,
        name: passkeyName.trim() || 'Passkey',
      });
      setPasskeyName('Passkey');
      if (result.recovery_codes && result.recovery_codes.length > 0) {
        setDisplayedRecoveryCodes(result.recovery_codes);
      }
      await refreshPasskeys();
      await loadStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add passkey');
    } finally {
      setPasskeysLoading(false);
    }
  };

  const handleRenamePasskey = async (id: string, originalName: string) => {
    const name = editingPasskeyName.trim();
    if (!name || name === originalName) {
      setEditingPasskeyId(null);
      setEditingPasskeyName('');
      return;
    }
    setError(null);
    try {
      await api.renameWebauthnCredential(id, name);
      setEditingPasskeyId(null);
      setEditingPasskeyName('');
      await refreshPasskeys();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename passkey');
    }
  };

  const handleCancelPasskeyEdit = () => {
    cancelledPasskeyEditRef.current = true;
    setEditingPasskeyId(null);
    setEditingPasskeyName('');
  };

  const handleDeletePasskey = async (id: string) => {
    setError(null);
    setDeletingPasskeyId(id);
    const previousPasskeys = passkeys;
    setPasskeys((current) => current.filter((credential) => credential.id !== id));
    try {
      await api.deleteWebauthnCredential(id);
      await refreshPasskeys();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete passkey';
      const isNotFoundError =
        typeof err === 'object' && err !== null && 'status' in err && err.status === 404;
      const refreshedPasskeys = await refreshPasskeys();
      const credentialStillExists = refreshedPasskeys?.some((credential) => credential.id === id);
      if (isNotFoundError && refreshedPasskeys && !credentialStillExists) {
        return;
      }
      if (!refreshedPasskeys) {
        setPasskeys(previousPasskeys);
      }
      setError(message);
    } finally {
      setDeletingPasskeyId(null);
    }
  };

  const formatCredentialDate = (value: string | null) => {
    if (!value) return 'Never';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Invalid date' : date.toLocaleString();
  };

  const methodLabel = (method: MfaMethod) => {
    if (method === 'totp') return 'Authenticator app';
    if (method === 'webauthn') return 'Passkey';
    return method;
  };

  if (!isOpen) return null;

  const allowedMethods = status?.allowed_methods ?? ['totp'];
  const enrolledMethods = status?.methods_enrolled ?? [];
  const totpEnrolled = enrolledMethods.includes('totp');
  const totpAllowed = allowedMethods.includes('totp');
  const webauthnAllowed = allowedMethods.includes('webauthn');
  const passkeyCount = status?.webauthn_credential_count ?? 0;
  const recoveryRemaining = status?.recovery_codes_remaining ?? 0;

  const tabs: { id: Manage2FATab; label: string }[] = [
    { id: 'authenticator', label: 'Authenticator app' },
    ...(webauthnAllowed ? [{ id: 'passkeys' as const, label: 'Passkeys' }] : []),
    { id: 'recovery' as const, label: 'Recovery codes' },
  ];
  // Guard against an active tab that is no longer available (e.g. passkeys
  // disabled by policy after the modal opened).
  const currentTab: Manage2FATab = tabs.some((tab) => tab.id === activeTab)
    ? activeTab
    : 'authenticator';

  const renderAuthenticatorPanel = () => {
    if (!totpEnrolled && totpAllowed) {
      return (
        <MfaEnrollmentWizard
          allowedMethods={allowedMethods}
          enrolledMethods={enrolledMethods}
          onComplete={handleWizardComplete}
        />
      );
    }
    if (!totpEnrolled) {
      return <p className="field-help">Authenticator app enrollment is not available.</p>;
    }
    if (rotationStep === 'verify') {
      return (
        <div className="mfa-tab-grid">
          <div>
            <p className="field-help" style={{ marginTop: 0 }}>
              Scan the new QR code with your authenticator app, then enter the 6-digit code to
              replace your current authenticator.
            </p>
            <TotpEnrollmentInstructions otpauthUri={rotationOtpauthUri} />
            <TotpManualSetup secret={rotationSecret} otpauthUri={rotationOtpauthUri} />
          </div>
          <div>
            <div className="form-group">
              <label htmlFor="rotation-verify-code" className="form-label">
                New verification code
              </label>
              <input
                id="rotation-verify-code"
                type="text"
                className="form-input"
                value={rotationVerifyCode}
                onChange={(event) => setRotationVerifyCode(event.target.value)}
                inputMode="numeric"
                autoComplete="one-time-code"
                autoFocus
                placeholder="Enter 6-digit code"
              />
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
              <button
                type="button"
                className="btn btn-primary"
                disabled={rotationLoading || !rotationVerifyCode.trim()}
                onClick={() => void completeRotation()}
              >
                {rotationLoading ? 'Confirming...' : 'Confirm replacement'}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={rotationLoading}
                onClick={cancelRotation}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div>
        <p className="field-help" style={{ marginTop: 0 }}>
          Your account is protected by an authenticator app. To switch apps or devices, verify your
          identity and set up a new one.
        </p>
        <div className="form-group">
          <label htmlFor="rotation-step-up-code" className="form-label">
            Current authenticator code or recovery code
          </label>
          <input
            id="rotation-step-up-code"
            type="text"
            className="form-input"
            value={rotationStepUpCode}
            onChange={(event) => setRotationStepUpCode(event.target.value)}
            inputMode="text"
            autoComplete="one-time-code"
            placeholder="Enter current code"
          />
        </div>
        <button
          type="button"
          className="btn btn-primary"
          disabled={rotationLoading || !rotationStepUpCode.trim()}
          onClick={() => void startRotation()}
        >
          {rotationLoading ? 'Verifying...' : 'Replace authenticator'}
        </button>
      </div>
    );
  };

  const renderRecoveryPanel = () => (
    <div>
      <p className="field-help" style={{ marginTop: 0 }}>
        {recoveryRemaining} recovery code{recoveryRemaining === 1 ? '' : 's'} remaining. Recovery
        codes let you sign in if you lose access to your other methods.
      </p>
      {regenerateVisible ? (
        <div className="form-group">
          <label htmlFor="regenerate-step-up-code" className="form-label">
            Current authenticator code or recovery code
          </label>
          <input
            id="regenerate-step-up-code"
            type="text"
            className="form-input"
            value={regenerateStepUpCode}
            onChange={(event) => setRegenerateStepUpCode(event.target.value)}
            inputMode="text"
            autoComplete="one-time-code"
            placeholder="Enter current code"
          />
          <div style={{ display: 'flex', gap: 'var(--space-sm)', marginTop: 'var(--space-sm)' }}>
            <button
              type="button"
              className="btn btn-primary"
              disabled={regenerateLoading || !regenerateStepUpCode.trim()}
              onClick={() => void handleRegenerate()}
            >
              {regenerateLoading ? 'Regenerating...' : 'Regenerate codes'}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={regenerateLoading}
              onClick={() => {
                setRegenerateVisible(false);
                setRegenerateStepUpCode('');
                setError(null);
              }}
            >
              Cancel
            </button>
          </div>
          <p className="field-help">Regenerating invalidates your existing recovery codes.</p>
        </div>
      ) : (
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => setRegenerateVisible(true)}
        >
          Regenerate recovery codes
        </button>
      )}
    </div>
  );

  const renderPasskeysPanel = () => (
    <div>
      <div className="form-group">
        <label className="form-label">Add a passkey</label>
        <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
          <input
            type="text"
            className="form-input"
            value={passkeyName}
            onChange={(event) => setPasskeyName(event.target.value)}
            placeholder="e.g. MacBook Touch ID"
            disabled={!isWebAuthnSupported()}
          />
          <button
            type="button"
            className="btn btn-primary"
            disabled={passkeysLoading || !passkeyName.trim() || !isWebAuthnSupported()}
            onClick={() => void handleAddPasskey()}
          >
            {passkeysLoading ? 'Adding...' : 'Add passkey'}
          </button>
        </div>
        {isWebAuthnSupported() ? (
          <p className="field-help" style={{ marginTop: 'var(--space-xs)' }}>
            You can create the passkey on this device, or pick the phone/tablet option in your
            browser prompt to scan a QR code.
          </p>
        ) : (
          <p className="field-help" style={{ marginTop: 'var(--space-xs)' }}>
            Your browser or device does not support passkeys.
          </p>
        )}
      </div>

      {passkeysLoading && passkeys.length === 0 ? (
        <p className="field-help">Loading passkeys...</p>
      ) : passkeys.length === 0 ? (
        <p className="field-help">No passkeys registered yet.</p>
      ) : (
        <div className="users-detail-list-shell" style={{ padding: 0 }}>
          {passkeys.map((credential) => (
            <div
              key={credential.id}
              className="admin-ws-item"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 'var(--space-sm)',
                padding: 'var(--space-sm) var(--space-md)',
              }}
            >
              <div style={{ flex: 1, minWidth: 0 }}>
                {editingPasskeyId === credential.id ? (
                  <input
                    type="text"
                    className="form-input"
                    value={editingPasskeyName}
                    onChange={(event) => setEditingPasskeyName(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter') {
                        event.currentTarget.blur();
                      } else if (event.key === 'Escape') {
                        handleCancelPasskeyEdit();
                      }
                    }}
                    onBlur={() => {
                      if (cancelledPasskeyEditRef.current) {
                        cancelledPasskeyEditRef.current = false;
                        return;
                      }
                      void handleRenamePasskey(credential.id, credential.name);
                    }}
                    autoFocus
                  />
                ) : (
                  <>
                    <button
                      type="button"
                      className="passkey-name-edit-trigger"
                      onClick={() => {
                        setEditingPasskeyId(credential.id);
                        setEditingPasskeyName(credential.name);
                      }}
                      title="Rename passkey"
                    >
                      <span className="passkey-name-text">{credential.name}</span>
                      <Pencil size={12} className="passkey-name-edit-icon" aria-hidden="true" />
                    </button>
                    <div className="muted" style={{ fontSize: '0.78rem', marginTop: '2px' }}>
                      Created {formatCredentialDate(credential.created_at)} · Last used{' '}
                      {formatCredentialDate(credential.last_used_at)}
                    </div>
                  </>
                )}
              </div>
              <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                <DeleteConfirmButton
                  onDelete={() => void handleDeletePasskey(credential.id)}
                  disabled={deletingPasskeyId !== null && deletingPasskeyId !== credential.id}
                  deleting={deletingPasskeyId === credential.id}
                  className="btn btn-sm btn-danger"
                  title="Delete passkey"
                  buttonText="Delete"
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content"
        style={{ maxWidth: 640 }}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3>Manage 2FA</h3>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          {error && <div className="login-error">{error}</div>}
          {success && <div className="status-message success">{success}</div>}

          {displayedRecoveryCodes.length > 0 ? (
            <>
              <p className="field-help" style={{ marginTop: 0 }}>
                Save these recovery codes now. They will not be shown again.
              </p>
              <RecoveryCodesDisplay codes={displayedRecoveryCodes} />
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setDisplayedRecoveryCodes([])}
                style={{ marginTop: 'var(--space-md)' }}
              >
                Done
              </button>
            </>
          ) : statusLoading && !status ? (
            <p className="field-help">Loading status...</p>
          ) : (
            <>
              {/* Compact status summary + preferred method — replaces the tall status block */}
              <div className="mfa-status-bar">
                <div className="mfa-status-facts">
                  <Shield size={15} aria-hidden="true" />
                  <span>
                    Authenticator: <strong>{totpEnrolled ? 'Enabled' : 'Not set up'}</strong>
                  </span>
                  {webauthnAllowed && (
                    <span>
                      Passkeys: <strong>{passkeyCount}</strong>
                    </span>
                  )}
                  <span>
                    Recovery codes: <strong>{recoveryRemaining}</strong>
                  </span>
                </div>
                {enrolledMethods.length > 1 && (
                  <label className="mfa-preferred-inline">
                    <span>Sign in with</span>
                    <select
                      className="form-input"
                      aria-label="Preferred sign-in method"
                      value={status?.preferred_method ?? ''}
                      onChange={(event) => {
                        const value = event.target.value as MfaMethod | '';
                        void (async () => {
                          try {
                            const next = await api.setPreferredMfaMethod(value || null);
                            setStatus(next);
                          } catch (err) {
                            setError(
                              err instanceof Error
                                ? err.message
                                : 'Failed to update preferred method',
                            );
                          }
                        })();
                      }}
                    >
                      <option value="">
                        {status?.default_method
                          ? `Default (${methodLabel(status.default_method)})`
                          : 'No preference'}
                      </option>
                      {enrolledMethods.map((method) => (
                        <option key={method} value={method}>
                          {methodLabel(method)}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              <div
                className="wizard-tabs"
                role="tablist"
                aria-label="Manage 2FA"
                style={{ display: 'flex', marginBottom: 'var(--space-lg)' }}
              >
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={currentTab === tab.id}
                    className={`wizard-tab ${currentTab === tab.id ? 'active' : ''}`}
                    style={{ flex: 1 }}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div role="tabpanel">
                {currentTab === 'authenticator' && renderAuthenticatorPanel()}
                {currentTab === 'passkeys' && webauthnAllowed && renderPasskeysPanel()}
                {currentTab === 'recovery' && renderRecoveryPanel()}
              </div>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
