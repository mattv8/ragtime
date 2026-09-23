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
  resource: 'https://ragtime.example/mcp/engineering',
  scope: 'tools.read tools.search',
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

  it('forwards resource and scope through password authorization', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      redirected: false,
      json: async () => ({
        mfa_required: true,
        mfa_challenge_token: 'challenge-token',
        mfa_methods: ['totp'],
      }),
    } as Response);

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => expect(fetch).toHaveBeenCalled());
    const request = vi.mocked(fetch).mock.calls[0]?.[1] as RequestInit;
    const body = new URLSearchParams(request.body as string);
    expect(body.get('resource')).toBe(oauthParams.resource);
    expect(body.get('scope')).toBe(oauthParams.scope);
  });

  it('announces a server failure, stays on credentials, and permits an MFA retry', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce({
        ok: false,
        redirected: false,
        json: async () => ({ error: 'sentinel OAuth failure from server' }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        redirected: false,
        json: async () => ({
          mfa_required: true,
          mfa_challenge_token: 'challenge-token',
          mfa_methods: ['totp'],
          mfa_preferred_method: null,
        }),
      } as Response);

    render(<OAuthLoginPage params={oauthParams} />);

    const username = screen.getByLabelText('Username');
    const password = screen.getByLabelText('Password');
    expect(username.getAttribute('aria-describedby')).toBeNull();
    expect(password.getAttribute('aria-describedby')).toBeNull();
    fireEvent.change(username, { target: { value: 'ldap-user' } });
    fireEvent.change(password, { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('sentinel OAuth failure from server');
    expect(username.getAttribute('aria-describedby')).toBe(alert.id);
    expect(password.getAttribute('aria-describedby')).toBe(alert.id);
    expect(screen.queryByLabelText('Authenticator or recovery code')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));
    await waitFor(() => {
      expect(screen.getByLabelText('Authenticator or recovery code')).toBeTruthy();
    });
  });
});
