import type { FormEvent } from 'react';

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
  return (
    <form onSubmit={onSubmit} className="login-form">
      {error && <div className="login-error">{error}</div>}

      <div className="form-group">
        <label htmlFor="username" className="form-label">
          Username
        </label>
        <input
          type="text"
          id="username"
          value={username}
          onChange={(event) => onUsernameChange(event.target.value)}
          className="form-input"
          placeholder={usernamePlaceholder}
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
          onChange={(event) => onPasswordChange(event.target.value)}
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
        {isLoading ? loadingLabel : submitLabel}
      </button>
    </form>
  );
}
