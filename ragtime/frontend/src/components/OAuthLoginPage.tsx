import { useState, type FormEvent } from 'react';
import { api } from '@/api';
import { BrandName } from '@/utils/buildEnvironment';
import { AuthCredentialsForm } from './AuthCredentialsForm';
import { AuthMfaPanel } from './AuthMfaPanel';

export interface OAuthParams {
  client_id: string;
  redirect_uri: string;
  response_type: string;
  code_challenge: string;
  code_challenge_method: string;
  state: string;
}

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
  const [mfaCode, setMfaCode] = useState('');
  const [rememberDevice, setRememberDevice] = useState(true);
  const [totpSecret, setTotpSecret] = useState('');
  const [otpauthUri, setOtpauthUri] = useState('');
  const [totpEnrollmentToken, setTotpEnrollmentToken] = useState('');
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);

  // Extract display name from client_id (often contains URL info)
  const getClientDisplay = () => {
    const clientId = params.client_id;
    const display = clientId.includes(' ') ? clientId.split(' ')[0] : clientId;
    return display.length > 50 ? display.substring(0, 47) + '...' : display;
  };

  const completeOAuthFromSession = async () => {
    const formData = new URLSearchParams();
    formData.append('client_id', params.client_id);
    formData.append('redirect_uri', params.redirect_uri);
    formData.append('response_type', params.response_type);
    formData.append('code_challenge', params.code_challenge);
    formData.append('code_challenge_method', params.code_challenge_method);
    formData.append('state', params.state);

    const response = await fetch('/authorize/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString(),
      credentials: 'include',
    });
    const data = await response.json();
    if (response.ok && data.redirect_url) {
      window.location.href = data.redirect_url;
      return;
    }
    throw new Error(data.error || 'OAuth authorization failed');
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // POST to /authorize endpoint
      const formData = new URLSearchParams();
      formData.append('client_id', params.client_id);
      formData.append('redirect_uri', params.redirect_uri);
      formData.append('response_type', params.response_type);
      formData.append('code_challenge', params.code_challenge);
      formData.append('code_challenge_method', params.code_challenge_method);
      formData.append('state', params.state);
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch('/authorize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData.toString(),
        credentials: 'include', // Include session cookie
      });

      // If we get redirected (302), the browser should follow it
      // But with fetch, we need to check if we ended up at a different URL
      if (response.redirected) {
        // The redirect was followed - navigate to final URL
        window.location.href = response.url;
        return;
      }

      // Try to parse JSON response
      let data;
      try {
        data = await response.json();
      } catch {
        // Response wasn't JSON
      }

      if (response.ok) {
        if (data && data.mfa_required && data.mfa_challenge_token) {
          setMfaChallengeToken(data.mfa_challenge_token);
          setMfaMode('verify');
          setPassword('');
          return;
        }
        if (data && data.mfa_enrollment_required && data.mfa_challenge_token) {
          const setup = await api.startMfaEnrollment(data.mfa_challenge_token);
          setMfaChallengeToken(data.mfa_challenge_token);
          setTotpSecret(setup.secret);
          setOtpauthUri(setup.otpauth_uri);
          setTotpEnrollmentToken(setup.enrollment_token);
          setMfaMode('enroll');
          setPassword('');
          return;
        }
        if (data && data.redirect_url) {
          // Navigate to the redirect URL
          window.location.href = data.redirect_url;
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
      await api.verifyMfaChallenge({
        mfa_challenge_token: mfaChallengeToken,
        code: mfaCode,
        remember_device: rememberDevice,
      });
      await completeOAuthFromSession();
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
        enrollment_token: totpEnrollmentToken,
        code: mfaCode,
        remember_device: rememberDevice,
      });
      setRecoveryCodes(response.recovery_codes);
      setMfaMode('recovery');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'MFA enrollment failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
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
          <AuthMfaPanel
            mode={mfaMode}
            error={error}
            isLoading={isLoading}
            code={mfaCode}
            rememberDevice={rememberDevice}
            totpSecret={totpSecret}
            otpauthUri={otpauthUri}
            recoveryCodes={recoveryCodes}
            recoveryContinueLabel="Continue authorization"
            onCodeChange={setMfaCode}
            onRememberDeviceChange={setRememberDevice}
            onVerify={handleMfaVerify}
            onEnrollComplete={handleMfaEnrollComplete}
            onRecoveryContinue={() => void completeOAuthFromSession()}
          />
        )}

        <div className="login-footer">
          <p className="login-info">Sign in with your LDAP credentials</p>
        </div>
      </div>
    </div>
  );
}
