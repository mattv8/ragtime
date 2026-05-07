import { useState, type FormEvent } from 'react';
import { api } from '@/api';
import type { User, AuthStatus, AuthMethodStatus } from '@/types';
import { AuthCredentialsForm } from './AuthCredentialsForm';
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
    <div className="login-card">
      <div className="login-header">
        <h1 className="login-title">{serverName}</h1>
      </div>

      {authStatus.cookie_warning && (
        <div className="status-message warning">
          <strong>Warning:</strong> {authStatus.cookie_warning}
        </div>
      )}

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

      <div className="login-footer">
        <ul className="login-auth-method-list" aria-live="polite">
          {authMethods.map((method) => (
            <li className="login-auth-method-item" key={method.key}>
              <span
                className={`login-auth-dot status-${method.status}`}
                aria-hidden="true"
              />
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
