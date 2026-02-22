import { useState, type FormEvent } from 'react';
import { api } from '@/api';
import type { User, AuthStatus } from '@/types';
import { AuthCredentialsForm } from './AuthCredentialsForm';

interface LoginPageProps {
  authStatus: AuthStatus;
  onLoginSuccess: (user: User) => void;
  serverName?: string;
}

export function LoginPage({ authStatus, onLoginSuccess, serverName = 'Ragtime' }: LoginPageProps) {
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
          <h1 className="login-title">{serverName}</h1>
          <p className="login-subtitle">Sign in to continue</p>
        </div>

        {authStatus.cookie_warning && (
          <div className="status-message warning">
            <strong>Warning:</strong> {authStatus.cookie_warning}
          </div>
        )}

        <AuthCredentialsForm
          username={username}
          password={password}
          usernamePlaceholder={authStatus.ldap_configured ? 'Username' : 'Local admin'}
          error={error}
          isLoading={isLoading}
          onUsernameChange={setUsername}
          onPasswordChange={setPassword}
          onSubmit={handleSubmit}
        />

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
            <p className="login-info">
              Local admin account is also available
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
