import { useState, type FormEvent } from 'react';
import { api } from '@/api';
import type { User, AuthStatus, AuthMethodStatus } from '@/types';
import { BrandName } from '@/utils/buildEnvironment';
import { AuthCredentialsForm } from './AuthCredentialsForm';
import { AuthMfaPanel } from './AuthMfaPanel';
import WebGLGradient from './WebGLGradient';

interface LoginPageProps {
  authStatus: AuthStatus;
  onLoginSuccess: (user: User) => void;
  serverName?: string;
}

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
  const [mfaCode, setMfaCode] = useState('');
  const [rememberDevice, setRememberDevice] = useState(true);
  const [totpSecret, setTotpSecret] = useState('');
  const [otpauthUri, setOtpauthUri] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);
  const [enrolledUser, setEnrolledUser] = useState<User | null>(null);

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
        setMfaMode('verify');
        setPassword('');
        return;
      }

      if (response.mfa_enrollment_required && response.mfa_challenge_token) {
        const setup = await api.startMfaEnrollment(response.mfa_challenge_token);
        setMfaChallengeToken(response.mfa_challenge_token);
        setTotpSecret(setup.secret);
        setOtpauthUri(setup.otpauth_uri);
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

  const handleMfaVerify = async (event: FormEvent) => {
    event.preventDefault();
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

  const handleMfaEnrollComplete = async (event: FormEvent) => {
    event.preventDefault();
    if (!mfaChallengeToken) return;
    setError(null);
    setIsLoading(true);
    try {
      const response = await api.completeMfaEnrollment({
        mfa_challenge_token: mfaChallengeToken,
        code: mfaCode,
        remember_device: rememberDevice,
      });
      setRecoveryCodes(response.recovery_codes);
      setEnrolledUser(response.user ?? null);
      setMfaMode('recovery');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'MFA enrollment failed');
    } finally {
      setIsLoading(false);
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
        <AuthMfaPanel
          mode={mfaMode}
          error={error}
          isLoading={isLoading}
          code={mfaCode}
          rememberDevice={rememberDevice}
          totpSecret={totpSecret}
          otpauthUri={otpauthUri}
          recoveryCodes={recoveryCodes}
          onCodeChange={setMfaCode}
          onRememberDeviceChange={setRememberDevice}
          onVerify={handleMfaVerify}
          onEnrollComplete={handleMfaEnrollComplete}
          onRecoveryContinue={() => void finishLogin(enrolledUser)}
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
