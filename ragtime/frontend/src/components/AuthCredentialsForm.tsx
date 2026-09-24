import { useId, type FormEvent } from 'react';

interface AuthCredentialsFormProps {
  username: string;
  password: string;
  usernamePlaceholder?: string;
  error: string | null;
  isLoading: boolean;
  onUsernameChange: (value: string) => void;
  onPasswordChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  submitLabel?: string;
  loadingLabel?: string;
}

export function AuthCredentialsForm({
  username,
  password,
  usernamePlaceholder = 'Username',
  error,
  isLoading,
  onUsernameChange,
  onPasswordChange,
  onSubmit,
  submitLabel = 'Sign In',
  loadingLabel = 'Signing in...',
}: AuthCredentialsFormProps) {
  const formId = `auth-credentials-${useId().replace(/:/g, '')}`;
  const errorId = `${formId}-error`;
  const usernameId = `${formId}-username`;
  const passwordId = `${formId}-password`;

  return (
    <form id={formId} data-auth-credentials-form="true" onSubmit={onSubmit} className="login-form">
      {error && (
        <div id={errorId} data-auth-credentials-error="true" className="login-error" role="alert">
          {error}
        </div>
      )}

      <div className="form-group">
        <label htmlFor={usernameId} className="form-label">
          Username
        </label>
        <input
          type="text"
          id={usernameId}
          value={username}
          onChange={(event) => onUsernameChange(event.target.value)}
          className="form-input"
          placeholder={usernamePlaceholder}
          required
          autoFocus
          autoComplete="username"
          aria-describedby={error ? errorId : undefined}
        />
      </div>

      <div className="form-group">
        <label htmlFor={passwordId} className="form-label">
          Password
        </label>
        <input
          type="password"
          id={passwordId}
          value={password}
          onChange={(event) => onPasswordChange(event.target.value)}
          className="form-input"
          placeholder="Password"
          required
          autoComplete="current-password"
          aria-describedby={error ? errorId : undefined}
        />
      </div>

      <button
        type="submit"
        className="btn btn-primary login-submit"
        disabled={isLoading || !username || !password}
      >
        {isLoading ? loadingLabel : submitLabel}
      </button>
    </form>
  );
}
