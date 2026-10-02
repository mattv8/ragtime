import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
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
  chat_enabled: true,
  userspace_generation_enabled: true,
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

  it('shows, dismisses, and clears the untrusted model endpoint notice independently', () => {
    const onNavigateToSettings = vi.fn();
    const { rerender } = render(
      <SecurityBanner
        authStatus={baseStatus}
        isAdmin
        isUntrustedModelEndpointConfigured
        onNavigateToSettings={onNavigateToSettings}
      />,
    );

    expect(
      screen.getByText(
        'Public generic OpenAI API model endpoints receive prompts and tool results and can influence tool calls; use only providers you trust.',
      ),
    ).toBeTruthy();
    expect(
      document.querySelector('[data-security-notice="generic-provider"] strong')?.textContent,
    ).toBe('Security:');
    fireEvent.click(
      document.querySelector('[data-security-notice="generic-provider"] .security-banner-link')!,
    );
    expect(onNavigateToSettings).toHaveBeenCalledWith('llm_provider');
    fireEvent.click(
      document.querySelector('[data-security-notice="generic-provider"] .security-banner-dismiss')!,
    );
    expect(screen.queryByText(/model endpoints receive prompts/i)).toBeNull();

    rerender(
      <SecurityBanner authStatus={baseStatus} isAdmin isUntrustedModelEndpointConfigured={false} />,
    );
    expect(screen.queryByText(/model endpoints receive prompts/i)).toBeNull();
  });

  it('does not show the untrusted model endpoint notice by default', () => {
    render(<SecurityBanner authStatus={baseStatus} isAdmin />);

    expect(screen.queryByText(/model endpoints receive prompts/i)).toBeNull();
  });
});
