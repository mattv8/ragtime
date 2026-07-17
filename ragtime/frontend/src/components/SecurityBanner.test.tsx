import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { AuthStatus } from '@/types';
import { SecurityBanner } from './SecurityBanner';

const baseStatus: AuthStatus = {
  authenticated: true,
  ldap_configured: false,
  local_admin_enabled: true,
  debug_mode: false,
  api_key_configured: true,
  session_cookie_secure: true,
  allowed_origins_open: false,
  runtime_auth_token_warning: false,
};

describe('SecurityBanner', () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  afterEach(() => {
    cleanup();
    window.sessionStorage.clear();
  });

  it('does not render posture warnings from a stale unauthenticated status', () => {
    render(
      <SecurityBanner
        authStatus={{ ...baseStatus, authenticated: false, api_key_configured: false }}
        isAdmin
      />,
    );

    expect(screen.queryByText(/The API endpoint accepts an API Key/i)).toBeNull();
  });

  it('renders the API-key warning for an authenticated insecure status', () => {
    render(
      <SecurityBanner
        authStatus={{ ...baseStatus, authenticated: true, api_key_configured: false }}
        isAdmin
      />,
    );

    expect(screen.getByText(/The API endpoint accepts an API Key/i)).toBeTruthy();
  });
});
