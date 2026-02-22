import { useState, type FormEvent } from 'react';
import { AuthCredentialsForm } from './AuthCredentialsForm';

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

  // Extract display name from client_id (often contains URL info)
  const getClientDisplay = () => {
    const clientId = params.client_id;
    const display = clientId.includes(' ') ? clientId.split(' ')[0] : clientId;
    return display.length > 50 ? display.substring(0, 47) + '...' : display;
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

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1 className="login-title">{serverName}</h1>
          <p className="login-subtitle">Sign in to authorize MCP access</p>
        </div>

        <div className="oauth-client-info">
          Authorizing: <strong>{getClientDisplay()}</strong>
        </div>

        <AuthCredentialsForm
          username={username}
          password={password}
          error={error}
          isLoading={isLoading}
          onUsernameChange={setUsername}
          onPasswordChange={setPassword}
          onSubmit={handleSubmit}
        />

        <div className="login-footer">
          <p className="login-info">
            Sign in with your LDAP credentials
          </p>
        </div>
      </div>
    </div>
  );
}
