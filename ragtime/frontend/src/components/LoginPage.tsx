import { useEffect, useRef, useState, type FormEvent } from 'react';
import { api } from '@/api';
import type { User, AuthStatus, AuthMethodStatus, MfaMethod } from '@/types';
import { BrandName } from '@/utils/buildEnvironment';
import { AuthCredentialsForm } from './AuthCredentialsForm';
import { LoginMfaPanel } from './shared/LoginMfaPanel';
import WebGLGradient from './WebGLGradient';

interface LoginPageProps {
  authStatus: AuthStatus;
  onLoginSuccess: (user: User) => void;
  serverName?: string;
}

// TOTP window is 30s and verification accepts a +/-1 step grace period, so a
// 15s refresh keeps the pre-filled code inside its valid window at click time.
const DEBUG_TOTP_REFRESH_MS = 15_000;

function resolveAuthMethods(authStatus: AuthStatus): AuthMethodStatus[] {
  if (authStatus.auth_methods && authStatus.auth_methods.length > 0) {
    return authStatus.auth_methods.filter((method) => method.configured && method.key !== 'local');
  }

  const methods: AuthMethodStatus[] = [];
  if (authStatus.ldap_configured) {
    methods.push({
      key: 'ldap',
      label: 'LDAP',
      configured: true,
      available: true,
      status: 'available',
      detail: 'Configured',
    });
  }

  return methods;
}

export function LoginCard({ authStatus, onLoginSuccess, serverName = 'Ragtime' }: LoginPageProps) {
  const authMethods = resolveAuthMethods(authStatus);
  const hasLdapMethod = authMethods.some((method) => method.key === 'ldap' && method.configured);
  const [username, setUsername] = useState(authStatus.debug_username || '');
  const [password, setPassword] = useState(authStatus.debug_password || '');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mfaChallengeToken, setMfaChallengeToken] = useState<string | null>(null);
  const [mfaMode, setMfaMode] = useState<'none' | 'verify' | 'enroll' | 'recovery'>('none');
  const [mfaMethods, setMfaMethods] = useState<MfaMethod[]>(['totp']);
  const [mfaPreferredMethod, setMfaPreferredMethod] = useState<MfaMethod | null>(null);
  const [mfaCode, setMfaCode] = useState(authStatus.debug_totp_code || '');
  const [rememberDevice, setRememberDevice] = useState(true);
  const lastAutoFilledCodeRef = useRef(authStatus.debug_totp_code || '');

  // TOTP codes rotate every 30s, so the code computed when the page loaded
  // goes stale while the user is on the MFA step. Poll a fresh code from the
  // debug endpoint (DEBUG_MODE only) and keep the pre-filled field rotating.
  // If the user typed their own code, auto-fill stops so we never clobber it.
  useEffect(() => {
    if (mfaMode !== 'verify' || !authStatus.debug_totp_code) {
      return;
    }
    let disposed = false;

    const refreshDebugCode = async () => {
      try {
        const { code } = await api.getDebugTotpCode();
        if (disposed || !code) return;
        setMfaCode((prev) => {
          if (prev !== lastAutoFilledCodeRef.current) return prev;
          lastAutoFilledCodeRef.current = code;
          return code;
        });
      } catch {
        // Best effort: transient failures keep the previous code in place.
      }
    };

    void refreshDebugCode();
    const timer = window.setInterval(refreshDebugCode, DEBUG_TOTP_REFRESH_MS);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [mfaMode, authStatus.debug_totp_code]);

  const finishLogin = async (userOverride?: User | null) => {
    const user = userOverride || (await api.getCurrentUser());
    onLoginSuccess(user);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const response = await api.login({ username, password });

      if (response.mfa_required && response.mfa_challenge_token) {
        setMfaChallengeToken(response.mfa_challenge_token);
        setMfaMethods(response.mfa_methods ?? ['totp']);
        setMfaPreferredMethod(response.mfa_preferred_method ?? null);
        setMfaMode('verify');
        setPassword('');
        return;
      }

      if (response.mfa_enrollment_required && response.mfa_challenge_token) {
        setMfaChallengeToken(response.mfa_challenge_token);
        setMfaMethods(response.mfa_enroll_methods ?? ['totp']);
        setMfaPreferredMethod(response.mfa_preferred_method ?? null);
        setMfaMode('enroll');
        setPassword('');
        return;
      }

      if (response.success && response.user_id) {
        // Fetch full user info
        await finishLogin();
      } else {
        setError(response.error || 'Login failed');
      }
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleMfaVerify = async () => {
    if (!mfaChallengeToken) return;
    setError(null);
    setIsLoading(true);
    try {
      const response = await api.verifyMfaChallenge({
        mfa_challenge_token: mfaChallengeToken,
        code: mfaCode,
        remember_device: rememberDevice,
      });
      if (response.success) {
        await finishLogin();
      } else {
        setError(response.error || 'MFA verification failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'MFA verification failed');
    } finally {
      setIsLoading(false);
    }
  };

  // Shared by passkey verification and MFA enrollment: in both cases the
  // backend has already issued the session cookie; we only need to load the
  // user and finish the login.
  const handleMfaSessionEstablished = async () => {
    setError(null);
    try {
      await finishLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in could not be completed');
    }
  };

  return (
    <div className="login-card">
      <div className="login-header">
        <h1 className="login-title">
          <BrandName name={serverName} />
        </h1>
      </div>

      {authStatus.cookie_warning && (
        <div className="status-message warning">
          <strong>Warning:</strong> {authStatus.cookie_warning}
        </div>
      )}

      {mfaMode === 'none' && (
        <AuthCredentialsForm
          username={username}
          password={password}
          usernamePlaceholder={hasLdapMethod ? 'Username' : 'Local admin'}
          error={error}
          isLoading={isLoading}
          onUsernameChange={setUsername}
          onPasswordChange={setPassword}
          onSubmit={handleSubmit}
        />
      )}

      {mfaMode !== 'none' && (
        <LoginMfaPanel
          mode={mfaMode}
          error={error}
          isLoading={isLoading}
          code={mfaCode}
          rememberDevice={rememberDevice}
          methods={mfaMethods}
          preferredMethod={mfaPreferredMethod}
          mfaChallengeToken={mfaChallengeToken ?? undefined}
          serverName={serverName}
          onCodeChange={setMfaCode}
          onRememberDeviceChange={setRememberDevice}
          onVerify={handleMfaVerify}
          onSessionEstablished={handleMfaSessionEstablished}
          onRecoveryContinue={() => void finishLogin()}
        />
      )}

      <div className="login-footer">
        <ul className="login-auth-method-list" aria-live="polite">
          {authMethods.map((method) => (
            <li className="login-auth-method-item" key={method.key}>
              <span className={`login-auth-dot status-${method.status}`} aria-hidden="true" />
              <span className="login-auth-method-label">{method.label}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export function LoginPage(props: LoginPageProps) {
  return (
    <div className="login-container login-gradient-container">
      <WebGLGradient className="login-background-gradient" fullscreen />
      <LoginCard {...props} />
    </div>
  );
}
