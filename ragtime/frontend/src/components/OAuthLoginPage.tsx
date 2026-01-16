import { useState, type FormEvent } from 'react';

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
        redirect: 'follow', // Follow redirects automatically
      });

      // If we get redirected (302), the browser should follow it
      // But with fetch, we need to check if we ended up at a different URL
      if (response.redirected) {
        // The redirect was followed - navigate to final URL
        window.location.href = response.url;
        return;
      }

      if (response.ok) {
        // Success without redirect - shouldn't happen but handle it
        return;
      }

      // Handle error response (JSON)
      if (response.status === 401) {
        try {
          const data = await response.json();
          setError(data.error || 'Invalid username or password');
        } catch {
          setError('Invalid username or password');
        }
      } else {
        try {
          const data = await response.json();
          setError(data.error || 'Authentication failed');
        } catch {
          const text = await response.text();
          setError(text || 'Authentication failed');
        }
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

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error">
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="username" className="form-label">
              Username
            </label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="form-input"
              placeholder="Username"
              required
              autoFocus
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="form-input"
              placeholder="Password"
              required
              autoComplete="current-password"
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary login-submit"
            disabled={isLoading || !username || !password}
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="login-footer">
          <p className="login-info">
            Sign in with your LDAP credentials
          </p>
        </div>
      </div>
    </div>
  );
}
