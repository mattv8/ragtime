import { useRef, useState, type FormEvent } from 'react';
import {
  api,
  apiFetch,
  beginResponseSessionEstablishment,
  isResponseAuthContextCurrent,
} from '@/api';
import type { MfaMethod } from '@/types';
import {
  buildAuthorizeForm,
  parseAuthorizeError,
  type OAuthParams,
} from '@/auth/oauthAuthorization';
import { BrandName } from '@/utils/buildEnvironment';
import { AuthCredentialsForm } from './AuthCredentialsForm';
import { LoginGradientShell } from './LoginGradientShell';
import { LoginMfaPanel } from './shared/LoginMfaPanel';

export type { OAuthParams } from '@/auth/oauthAuthorization';

interface OAuthLoginPageProps {
  params: OAuthParams;
  serverName?: string;
}

export function OAuthLoginPage({ params, serverName = 'Ragtime' }: OAuthLoginPageProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [mfaChallengeToken, setMfaChallengeToken] = useState<string | null>(null);
  const [mfaMode, setMfaMode] = useState<'none' | 'verify' | 'enroll' | 'recovery'>('none');
  const [mfaMethods, setMfaMethods] = useState<MfaMethod[]>(['totp']);
  const [mfaPreferredMethod, setMfaPreferredMethod] = useState<MfaMethod | null>(null);
  const [mfaCode, setMfaCode] = useState('');
  const [rememberDevice, setRememberDevice] = useState(true);
  const [authorizationRetryAvailable, setAuthorizationRetryAvailable] = useState(false);
  const authorizationAttemptRef = useRef(0);

  const beginAuthorizationAttempt = () => {
    authorizationAttemptRef.current += 1;
    return authorizationAttemptRef.current;
  };

  const isCurrentAttempt = (attempt: number) => authorizationAttemptRef.current === attempt;
  const isObsoleteResult = (error: unknown) =>
    error instanceof DOMException && error.name === 'AbortError';

  // Extract display name from client_id (often contains URL info)
  const getClientDisplay = () => {
    const clientId = params.client_id;
    const display = clientId.includes(' ') ? clientId.split(' ')[0] : clientId;
    return display.length > 50 ? display.substring(0, 47) + '...' : display;
  };

  const completeOAuthFromSession = async (attempt: number) => {
    const response = await apiFetch(
      '/authorize/session',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: buildAuthorizeForm(params).toString(),
      },
      'session',
    );
    const data = await response.json();
    if (!isCurrentAttempt(attempt) || !isResponseAuthContextCurrent(response)) {
      throw new DOMException('OAuth authorization result is no longer current', 'AbortError');
    }
    if (response.ok && data.redirect_url) {
      window.location.assign(data.redirect_url);
      return;
    }
    throw new Error(parseAuthorizeError(data).summary);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const attempt = beginAuthorizationAttempt();
    setError(null);
    setIsLoading(true);

    try {
      const formData = buildAuthorizeForm(params);
      formData.append('username', username);
      formData.append('password', password);

      const response = await apiFetch(
        '/authorize',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData.toString(),
        },
        'challenge',
      );

      // If we get redirected (302), the browser should follow it
      // But with fetch, we need to check if we ended up at a different URL
      if (response.redirected) {
        const establishedContext = beginResponseSessionEstablishment(response);
        if (!establishedContext || !isCurrentAttempt(attempt)) {
          throw new DOMException('OAuth authorization result is no longer current', 'AbortError');
        }
        // The redirect was followed - navigate to final URL
        window.location.assign(response.url);
        return;
      }

      // Try to parse JSON response
      let data;
      try {
        data = await response.json();
      } catch {
        // Response wasn't JSON
      }

      if (!isCurrentAttempt(attempt) || !isResponseAuthContextCurrent(response)) {
        throw new DOMException('OAuth authorization result is no longer current', 'AbortError');
      }

      if (response.ok) {
        if (data && data.mfa_required && data.mfa_challenge_token) {
          setMfaChallengeToken(data.mfa_challenge_token);
          setMfaMethods(data.mfa_methods ?? ['totp']);
          setMfaPreferredMethod(data.mfa_preferred_method ?? null);
          setMfaMode('verify');
          setPassword('');
          return;
        }
        if (data && data.mfa_enrollment_required && data.mfa_challenge_token) {
          setMfaChallengeToken(data.mfa_challenge_token);
          setMfaMethods(data.mfa_enroll_methods ?? ['totp']);
          setMfaPreferredMethod(data.mfa_preferred_method ?? null);
          setMfaMode('enroll');
          setPassword('');
          return;
        }
        if (data && data.redirect_url) {
          const establishedContext = beginResponseSessionEstablishment(response);
          if (!establishedContext || !isCurrentAttempt(attempt)) {
            throw new DOMException('OAuth authorization result is no longer current', 'AbortError');
          }
          // Navigate to the redirect URL
          window.location.assign(data.redirect_url);
          return;
        }
        // Fallback or unexpected success without redirect info
        return;
      }

      // Handle error response
      if (data && data.error) {
        setError(data.error);
      } else {
        setError('Authentication failed');
      }
    } catch (err) {
      if (!isCurrentAttempt(attempt) || isObsoleteResult(err)) return;
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      if (isCurrentAttempt(attempt)) setIsLoading(false);
    }
  };

  const handleMfaVerify = async () => {
    if (!mfaChallengeToken) return;
    const attempt = beginAuthorizationAttempt();
    setError(null);
    setIsLoading(true);
    let sessionEstablished = false;
    try {
      await api.verifyMfaChallenge({
        mfa_challenge_token: mfaChallengeToken,
        code: mfaCode,
        remember_device: rememberDevice,
      });
      if (!isCurrentAttempt(attempt)) return;
      sessionEstablished = true;
      await completeOAuthFromSession(attempt);
    } catch (err) {
      if (!isCurrentAttempt(attempt) || isObsoleteResult(err)) return;
      setError(err instanceof Error ? err.message : 'MFA verification failed');
      if (sessionEstablished) setAuthorizationRetryAvailable(true);
    } finally {
      if (isCurrentAttempt(attempt)) setIsLoading(false);
    }
  };

  // Shared by passkey verification and MFA enrollment: the session cookie is
  // already set, so we only need to complete the OAuth authorization.
  const handleMfaSessionEstablished = async () => {
    const attempt = beginAuthorizationAttempt();
    setError(null);
    setIsLoading(true);
    try {
      await completeOAuthFromSession(attempt);
    } catch (err) {
      if (!isCurrentAttempt(attempt) || isObsoleteResult(err)) return;
      setError(err instanceof Error ? err.message : 'OAuth authorization failed');
      setAuthorizationRetryAvailable(true);
    } finally {
      if (isCurrentAttempt(attempt)) setIsLoading(false);
    }
  };

  const retryAuthorization = async () => {
    const attempt = beginAuthorizationAttempt();
    setError(null);
    setIsLoading(true);
    try {
      await completeOAuthFromSession(attempt);
      setAuthorizationRetryAvailable(false);
    } catch (err) {
      if (!isCurrentAttempt(attempt) || isObsoleteResult(err)) return;
      setError(err instanceof Error ? err.message : 'OAuth authorization failed');
    } finally {
      if (isCurrentAttempt(attempt)) setIsLoading(false);
    }
  };

  const handleRecoveryContinue = async () => {
    const attempt = beginAuthorizationAttempt();
    setError(null);
    setIsLoading(true);
    try {
      await completeOAuthFromSession(attempt);
    } catch (err) {
      if (!isCurrentAttempt(attempt) || isObsoleteResult(err)) return;
      setError(err instanceof Error ? err.message : 'OAuth authorization failed');
    } finally {
      if (isCurrentAttempt(attempt)) setIsLoading(false);
    }
  };

  return (
    <LoginGradientShell>
      <div id="oauth-login-card" className="login-card">
        <div className="login-header">
          <h1 className="login-title">
            <BrandName name={serverName} />
          </h1>
          <p className="login-subtitle">Sign in to authorize MCP access</p>
        </div>

        <div className="oauth-client-info">
          Authorizing: <strong>{getClientDisplay()}</strong>
        </div>

        {mfaMode === 'none' && (
          <AuthCredentialsForm
            username={username}
            password={password}
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
            recoveryContinueLabel="Continue authorization"
            methods={mfaMethods}
            preferredMethod={mfaPreferredMethod}
            mfaChallengeToken={mfaChallengeToken ?? undefined}
            serverName={serverName}
            onCodeChange={setMfaCode}
            onRememberDeviceChange={setRememberDevice}
            onVerify={handleMfaVerify}
            onSessionEstablished={handleMfaSessionEstablished}
            onRecoveryContinue={() => void handleRecoveryContinue()}
          />
        )}

        {authorizationRetryAvailable && (
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => void retryAuthorization()}
            disabled={isLoading}
          >
            Retry authorization
          </button>
        )}

        <div className="login-footer">
          <p className="login-info">Sign in with your LDAP credentials</p>
        </div>
      </div>
    </LoginGradientShell>
  );
}
