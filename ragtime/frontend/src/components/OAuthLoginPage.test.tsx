import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { OAuthLoginPage, type OAuthParams } from './OAuthLoginPage';

const apiMock = vi.hoisted(() => ({
  verifyMfaChallenge: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: apiMock,
}));

vi.mock('./WebGLGradient', () => ({
  default: () => <div data-testid="webgl-gradient" />,
}));

const oauthParams: OAuthParams = {
  client_id: 'Claude Desktop',
  redirect_uri: 'https://example.com/callback',
  response_type: 'code',
  code_challenge: 'challenge',
  code_challenge_method: 'S256',
  state: 'state-1',
};

describe('OAuthLoginPage gradient shell', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders the credential step inside the shared auth gradient surface', () => {
    render(<OAuthLoginPage params={oauthParams} />);

    const surface = document.querySelector('[data-auth-surface="gradient"]');
    expect(surface).toBeTruthy();
    expect(screen.getByTestId('webgl-gradient')).toBeTruthy();
    expect(screen.getByText('Sign in to authorize MCP access')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Sign In' })).toBeTruthy();
  });

  it('keeps the MFA step inside the shared auth gradient surface', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      redirected: false,
      url: 'http://localhost/authorize',
      json: async () => ({
        mfa_required: true,
        mfa_challenge_token: 'challenge-token',
        mfa_methods: ['totp'],
        mfa_preferred_method: null,
      }),
    } as Response);

    render(<OAuthLoginPage params={oauthParams} />);

    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => {
      expect(screen.getByLabelText('Authenticator or recovery code')).toBeTruthy();
    });

    const surface = document.querySelector('[data-auth-surface="gradient"]');
    expect(surface).toBeTruthy();
    expect(screen.getByTestId('webgl-gradient')).toBeTruthy();
  });
});
