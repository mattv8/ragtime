import { useEffect, useRef } from 'react';

export type AuthRecoveryAction = 'retry-bootstrap' | 'retry-logout' | 'check-session';

export interface AuthRecoveryStateProps {
  action: AuthRecoveryAction;
  busy: boolean;
  message: string;
  onRetry: () => void;
}

const ACTION_LABELS: Record<AuthRecoveryAction, string> = {
  'retry-bootstrap': 'Try connecting again',
  'retry-logout': 'Retry sign out',
  'check-session': 'Check session again',
};

const ACTION_TITLES: Record<AuthRecoveryAction, string> = {
  'retry-bootstrap': 'Connection Issue',
  'retry-logout': 'Sign out could not be confirmed',
  'check-session': 'Check your session',
};

export function AuthRecoveryState({ action, busy, message, onRetry }: AuthRecoveryStateProps) {
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Focus the action button on entry and after a failed retry (busy true→false)
  useEffect(() => {
    if (!busy && buttonRef.current) {
      buttonRef.current.focus();
    }
  }, [busy]);

  const actionLabel = ACTION_LABELS[action];

  return (
    <div
      id="auth-recovery-state"
      className="login-card"
      role="region"
      aria-busy={busy}
      aria-label={`Authentication recovery: ${actionLabel}`}
    >
      <div className="login-header">
        <h1 className="login-title">{ACTION_TITLES[action]}</h1>
      </div>

      {/* Error text as alert */}
      {message && (
        <div role="alert" className="login-error" style={{ marginBottom: 'var(--space-lg)' }}>
          {message}
        </div>
      )}

      {/* Keep one live region mounted so retry status changes are announced once. */}
      <div
        className="login-status"
        aria-label="Attempting recovery"
        aria-live="polite"
        aria-atomic="true"
        style={{ marginBottom: busy ? 'var(--space-lg)' : 0 }}
      >
        {busy ? (
          <>
            <span className="loading-spinner" aria-hidden="true" />
            Attempting recovery...
          </>
        ) : null}
      </div>

      {/* Action button */}
      <button
        ref={buttonRef}
        type="button"
        onClick={onRetry}
        disabled={busy}
        className="btn btn-primary login-submit"
        aria-busy={busy}
      >
        {actionLabel}
      </button>
    </div>
  );
}
