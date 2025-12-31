import { useState, type FormEvent } from 'react';
import { api } from '@/api';
import type { User, AuthStatus } from '@/types';

interface LoginPageProps {
  authStatus: AuthStatus;
  onLoginSuccess: (user: User) => void;
}

export function LoginPage({ authStatus, onLoginSuccess }: LoginPageProps) {
  const [username, setUsername] = useState(authStatus.debug_username || '');
  const [password, setPassword] = useState(authStatus.debug_password || '');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      const response = await api.login({ username, password });

      if (response.success && response.user_id) {
        // Fetch full user info
        const user = await api.getCurrentUser();
        onLoginSuccess(user);
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

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <h1 className="login-title">Ragtime</h1>
          <p className="login-subtitle">Sign in to continue</p>
        </div>

        {authStatus.cookie_warning && (
          <div className="login-warning">
            <strong>Warning:</strong> {authStatus.cookie_warning}
          </div>
        )}

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
              placeholder={authStatus.ldap_configured ? 'Username' : 'Local admin'}
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
          {authStatus.ldap_configured && (
            <p className="login-info">
              Sign in with your LDAP credentials
            </p>
          )}
          {authStatus.local_admin_enabled && !authStatus.ldap_configured && (
            <p className="login-info">
              Sign in with the local admin account
            </p>
          )}
          {authStatus.ldap_configured && authStatus.local_admin_enabled && (
            <p className="login-info-small">
              Local admin account is also available
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
