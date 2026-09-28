import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import { userRecoveryApi } from '@/api/userRecovery';
import type { RecoveryPassStatus, User } from '@/types';
import './UserSecurityTab.css';

export interface UserSecurityTabProps {
  user: User;
  isSelf: boolean;
  onUserUpdated: (user: User) => void;
  onDirtyChange: (dirty: boolean) => void;
  onBusyChange?: (busy: boolean) => void;
}

const messageFor = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback;

/** Admin-only security operations; every credential mutation is freshly verified. */
export function UserSecurityTab({
  user,
  isSelf,
  onUserUpdated,
  onDirtyChange,
  onBusyChange,
}: UserSecurityTabProps) {
  const [grant, setGrant] = useState<RecoveryPassStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [adminPassword, setAdminPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [revealedPass, setRevealedPass] = useState<string | null>(null);
  const [confirmMfaReset, setConfirmMfaReset] = useState(false);
  const [confirmPasswordReset, setConfirmPasswordReset] = useState(false);
  const currentUserId = useRef(user.id);
  const mounted = useRef(true);
  const working = useRef(false);
  const revealedGrantId = useRef<string | null>(null);
  const refreshEpoch = useRef(0);
  const onUserUpdatedRef = useRef(onUserUpdated);
  const onDirtyChangeRef = useRef(onDirtyChange);
  const onBusyChangeRef = useRef(onBusyChange);
  currentUserId.current = user.id;
  onUserUpdatedRef.current = onUserUpdated;
  onDirtyChangeRef.current = onDirtyChange;
  onBusyChangeRef.current = onBusyChange;

  const setWorking = useCallback((value: boolean) => {
    if (!mounted.current) return;
    working.current = value;
    setBusy(value);
    onBusyChangeRef.current?.(value);
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      working.current = false;
      onBusyChangeRef.current?.(false);
    };
  }, []);
  useEffect(() => {
    onDirtyChangeRef.current(Boolean(adminPassword || newPassword || confirmPassword));
  }, [adminPassword, confirmPassword, newPassword]);
  useEffect(() => {
    if (working.current) setWorking(false);
    setGrant(null);
    revealedGrantId.current = null;
    setRevealedPass(null);
    setError(null);
    setNotice(null);
    setAdminPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setConfirmMfaReset(false);
    setConfirmPasswordReset(false);
  }, [setWorking, user.id]);

  const refresh = useCallback(
    async (showRefreshError = true): Promise<boolean> => {
      const requestedId = user.id;
      const epoch = refreshEpoch.current + 1;
      refreshEpoch.current = epoch;
      setLoading(true);
      try {
        const [nextGrant, refreshed] = await Promise.all([
          userRecoveryApi.getRecoveryPass(requestedId),
          api.getUser(requestedId),
        ]);
        if (
          !mounted.current ||
          currentUserId.current !== requestedId ||
          refreshEpoch.current !== epoch
        )
          return false;
        setGrant(nextGrant);
        // A secret is only useful for the exact currently-issued grant it came from.
        if (
          nextGrant?.status !== 'issued' ||
          (revealedGrantId.current !== null && nextGrant.id !== revealedGrantId.current)
        ) {
          revealedGrantId.current = null;
          setRevealedPass(null);
        }
        onUserUpdatedRef.current(refreshed);
        return true;
      } catch (err) {
        if (
          mounted.current &&
          currentUserId.current === requestedId &&
          refreshEpoch.current === epoch &&
          showRefreshError
        )
          setError(messageFor(err, 'Could not load security status'));
        return false;
      } finally {
        if (
          mounted.current &&
          currentUserId.current === requestedId &&
          refreshEpoch.current === epoch
        )
          setLoading(false);
      }
    },
    [user.id],
  );
  useEffect(() => {
    void refresh();
  }, [refresh]);

  const verify = async () => {
    if (!adminPassword) throw new Error('Enter your current administrator password.');
    return (await userRecoveryApi.verifyAdminSecurity(adminPassword)).verification_token;
  };
  const sensitive = async (
    action: (token: string, requestedId: string) => Promise<void>,
    clearPassword = false,
  ) => {
    const requestedId = user.id;
    setWorking(true);
    setError(null);
    setNotice(null);
    try {
      const token = await verify();
      if (!mounted.current || currentUserId.current !== requestedId) return;
      await action(token, requestedId);
      if (!mounted.current || currentUserId.current !== requestedId) return;
      if (clearPassword) {
        setNewPassword('');
        setConfirmPassword('');
        setConfirmPasswordReset(false);
      }
      setAdminPassword('');
      const refreshed = await refresh(false);
      if (!mounted.current || currentUserId.current !== requestedId) return;
      if (!refreshed)
        setNotice(
          'Security change was saved, but the current status is unavailable. Retry status loading.',
        );
    } catch (err) {
      if (mounted.current && currentUserId.current === requestedId)
        setError(messageFor(err, 'Security operation failed'));
    } finally {
      if (mounted.current && currentUserId.current === requestedId) setWorking(false);
    }
  };
  const issue = () =>
    void sensitive(async (token, requestedId) => {
      const result = await userRecoveryApi.issueRecoveryPass(requestedId, token);
      if (!mounted.current || currentUserId.current !== requestedId) return;
      revealedGrantId.current = result.grant.id;
      setGrant(result.grant);
      setRevealedPass(result.pass);
    });
  const revoke = () =>
    void sensitive(async (token, requestedId) => {
      await userRecoveryApi.revokeRecoveryPass(requestedId, token);
      if (!mounted.current || currentUserId.current !== requestedId) return;
      revealedGrantId.current = null;
      setRevealedPass(null);
    });
  const resetMfa = () =>
    void sensitive(async (token, requestedId) => {
      await api.resetUserMfa(requestedId, token);
      if (!mounted.current || currentUserId.current !== requestedId) return;
      revealedGrantId.current = null;
      setRevealedPass(null);
      setConfirmMfaReset(false);
    });
  const resetPassword = () =>
    void sensitive(async (token, requestedId) => {
      if (!newPassword || newPassword !== confirmPassword)
        throw new Error('New passwords must match.');
      await api.updateLocalUser(requestedId, { password: newPassword, verification_token: token });
      if (!mounted.current || currentUserId.current !== requestedId) return;
      revealedGrantId.current = null;
      setRevealedPass(null);
    }, true);
  const copyPass = async () => {
    if (!revealedPass || !navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(revealedPass);
    } catch {
      setError('Could not copy the recovery pass. Select it and copy manually.');
    }
  };
  const selfReason =
    'You cannot perform administrator recovery or reset actions on your own account.';
  const localManaged = user.auth_provider === 'local_managed';
  const mfaMethods =
    user.mfa_methods
      ?.map((method) => (method === 'totp' ? 'Authenticator app' : 'Passkey'))
      .join(', ') || (user.mfa_enabled ? 'enrolled' : 'not enrolled');
  const locked = loading || busy || isSelf;

  return (
    <section
      id={`user-security-${user.id}`}
      className="user-security-tab"
      aria-busy={loading || busy}
    >
      <header className="user-security-header">
        <h3>Security</h3>
        <p>
          MFA: {mfaMethods}. Recovery codes remaining: {user.recovery_codes_remaining ?? 'Unknown'}.
        </p>
      </header>
      {error && (
        <div className="user-security-error" role="alert">
          {error}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading || busy}
            onClick={() => void refresh()}
          >
            Retry
          </button>
        </div>
      )}
      {notice && (
        <div className="user-security-notice" role="status">
          {notice}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={loading || busy}
            onClick={() => void refresh()}
          >
            Retry status
          </button>
        </div>
      )}
      {loading ? (
        <p>Loading security status…</p>
      ) : (
        <>
          {isSelf && <p className="user-security-notice">{selfReason}</p>}
          <section data-security-section="verification" className="user-security-section">
            <h4>Confirm administrator password</h4>
            <label htmlFor={`admin-password-${user.id}`}>Current password</label>
            <input
              id={`admin-password-${user.id}`}
              type="password"
              value={adminPassword}
              onChange={(event) => setAdminPassword(event.target.value)}
              autoComplete="current-password"
              disabled={busy}
            />
            <p>Required immediately before each security change.</p>
          </section>
          <section data-security-section="recovery-pass" className="user-security-section">
            <h4>Administrator recovery pass</h4>
            <p>
              Status: {grant?.status ?? 'none'}.{' '}
              {(grant?.status === 'issued' || grant?.status === 'redeemed') && (
                <>Expires {new Date(grant.expires_at).toLocaleString()}.</>
              )}
            </p>
            {revealedPass && grant?.status === 'issued' && (
              <div className="user-security-secret" role="status">
                <label htmlFor={`recovery-pass-${user.id}`}>One-time pass — save it now</label>
                <input id={`recovery-pass-${user.id}`} readOnly value={revealedPass} />
                <button type="button" className="btn btn-secondary" onClick={() => void copyPass()}>
                  Copy pass
                </button>
                <p>It cannot be shown again after you leave this screen.</p>
              </div>
            )}
            <button
              type="button"
              className="btn btn-primary"
              disabled={locked}
              title={isSelf ? selfReason : undefined}
              onClick={issue}
            >
              {grant?.status === 'issued' || grant?.status === 'redeemed'
                ? 'Reissue recovery pass'
                : 'Issue recovery pass'}
            </button>
            {(grant?.status === 'issued' || grant?.status === 'redeemed') && (
              <button
                type="button"
                className="btn btn-secondary"
                disabled={locked}
                title={isSelf ? selfReason : undefined}
                onClick={revoke}
              >
                Revoke pass
              </button>
            )}
          </section>
          {localManaged && (
            <section data-security-section="password-reset" className="user-security-section">
              <h4>Reset local password</h4>
              <p>
                Changing this password ends the user’s existing sessions and recovery continuation.
              </p>
              <label htmlFor={`new-password-${user.id}`}>New password</label>
              <input
                id={`new-password-${user.id}`}
                type="password"
                value={newPassword}
                onChange={(event) => setNewPassword(event.target.value)}
                autoComplete="new-password"
                disabled={busy}
              />
              {confirmPasswordReset && (
                <>
                  <label htmlFor={`confirm-password-${user.id}`}>Confirm new password</label>
                  <input
                    id={`confirm-password-${user.id}`}
                    type="password"
                    value={confirmPassword}
                    onChange={(event) => setConfirmPassword(event.target.value)}
                    autoComplete="new-password"
                    disabled={busy}
                  />
                </>
              )}{' '}
              {confirmPasswordReset ? (
                <>
                  <button
                    type="button"
                    className="btn btn-danger"
                    disabled={locked || !newPassword || !confirmPassword}
                    onClick={resetPassword}
                  >
                    Confirm password reset
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={busy}
                    onClick={() => setConfirmPasswordReset(false)}
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={locked || !newPassword}
                  title={isSelf ? selfReason : undefined}
                  onClick={() => setConfirmPasswordReset(true)}
                >
                  Reset password…
                </button>
              )}
            </section>
          )}
          <section
            data-security-section="mfa-reset"
            className="user-security-section user-security-danger"
          >
            <h4>Reset MFA</h4>
            <p>
              This removes authenticator factors, passkeys, trusted devices, recovery codes, active
              sessions, and any recovery continuation. The user enrolls again only when MFA policy
              requires it.
            </p>
            {confirmMfaReset ? (
              <>
                <p role="alert">Confirm this destructive action.</p>
                <button
                  type="button"
                  className="btn btn-danger"
                  disabled={locked}
                  onClick={resetMfa}
                >
                  Confirm MFA reset
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={busy}
                  onClick={() => setConfirmMfaReset(false)}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn btn-danger"
                disabled={locked}
                title={isSelf ? selfReason : undefined}
                onClick={() => setConfirmMfaReset(true)}
              >
                Reset MFA…
              </button>
            )}
          </section>
        </>
      )}
    </section>
  );
}
