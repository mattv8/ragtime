import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { OAuthLoginPage, type OAuthParams } from './OAuthLoginPage';

const apiMock = vi.hoisted(() => ({
  verifyMfaChallenge: vi.fn(),
  apiFetch: vi.fn(),
  beginResponseSessionEstablishment: vi.fn(),
  isResponseAuthContextCurrent: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: { verifyMfaChallenge: apiMock.verifyMfaChallenge },
  apiFetch: apiMock.apiFetch,
  beginResponseSessionEstablishment: apiMock.beginResponseSessionEstablishment,
  isResponseAuthContextCurrent: apiMock.isResponseAuthContextCurrent,
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
    apiMock.isResponseAuthContextCurrent.mockReturnValue(true);
    apiMock.beginResponseSessionEstablishment.mockReturnValue({ generation: 1 });
  });

  afterEach(() => {
    cleanup();
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
    apiMock.apiFetch.mockResolvedValue({
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
    apiMock.apiFetch.mockResolvedValue({
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

    await waitFor(() => expect(apiMock.apiFetch).toHaveBeenCalled());
    const request = apiMock.apiFetch.mock.calls[0]?.[1] as RequestInit;
    const body = new URLSearchParams(request.body as string);
    expect(body.get('resource')).toBe(oauthParams.resource);
    expect(body.get('scope')).toBe(oauthParams.scope);
    expect(apiMock.apiFetch).toHaveBeenCalledWith('/authorize', expect.any(Object), 'challenge');
  });

  it('preserves all OAuth and PKCE fields while waiting for MFA without establishing a session', async () => {
    apiMock.apiFetch.mockResolvedValue({
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

    await waitFor(() =>
      expect(screen.getByLabelText('Authenticator or recovery code')).toBeTruthy(),
    );
    const request = apiMock.apiFetch.mock.calls[0]?.[1] as RequestInit;
    const body = new URLSearchParams(request.body as string);
    expect(body.get('client_id')).toBe(oauthParams.client_id);
    expect(body.get('redirect_uri')).toBe(oauthParams.redirect_uri);
    expect(body.get('response_type')).toBe(oauthParams.response_type);
    expect(body.get('code_challenge')).toBe(oauthParams.code_challenge);
    expect(body.get('code_challenge_method')).toBe(oauthParams.code_challenge_method);
    expect(body.get('state')).toBe(oauthParams.state);
    expect(body.get('resource')).toBe(oauthParams.resource);
    expect(body.get('scope')).toBe(oauthParams.scope);
    expect(apiMock.beginResponseSessionEstablishment).not.toHaveBeenCalled();
  });

  it('keeps an invalid MFA factor local and does not issue session authorization', async () => {
    apiMock.apiFetch.mockResolvedValueOnce({
      ok: true,
      redirected: false,
      json: async () => ({
        mfa_required: true,
        mfa_challenge_token: 'challenge-token',
        mfa_methods: ['totp'],
      }),
    } as Response);
    apiMock.verifyMfaChallenge.mockRejectedValue(new Error('Invalid authentication code'));

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));
    await screen.findByLabelText('Authenticator or recovery code');
    fireEvent.change(screen.getByLabelText('Authenticator or recovery code'), {
      target: { value: '123456' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));

    await screen.findByText('Invalid authentication code');
    expect(apiMock.apiFetch).toHaveBeenCalledTimes(1);
    expect(apiMock.beginResponseSessionEstablishment).not.toHaveBeenCalled();
  });

  it('silently discards obsolete credential results and stale authorization errors', async () => {
    apiMock.apiFetch.mockResolvedValue({
      ok: false,
      redirected: false,
      json: async () => ({ error: 'old authorization error' }),
    } as Response);
    apiMock.isResponseAuthContextCurrent.mockReturnValue(false);

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => expect(apiMock.isResponseAuthContextCurrent).toHaveBeenCalled());
    expect(screen.queryByText('old authorization error')).toBeNull();
    expect(screen.queryByText('OAuth authorization result is no longer current')).toBeNull();
  });

  it('silently discards an obsolete fetch rejection', async () => {
    apiMock.apiFetch.mockRejectedValue(
      new DOMException('OAuth authorization result is no longer current', 'AbortError'),
    );

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => expect(apiMock.apiFetch).toHaveBeenCalled());
    expect(screen.queryByText('OAuth authorization result is no longer current')).toBeNull();
  });

  it('does not redirect or display an error from a stale session authorization after MFA', async () => {
    apiMock.apiFetch
      .mockResolvedValueOnce({
        ok: true,
        redirected: false,
        json: async () => ({
          mfa_required: true,
          mfa_challenge_token: 'challenge-token',
          mfa_methods: ['totp'],
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ redirect_url: 'https://example.com/callback?code=old' }),
      } as Response);
    apiMock.verifyMfaChallenge.mockResolvedValue({ success: true });
    apiMock.isResponseAuthContextCurrent.mockReturnValueOnce(true).mockReturnValueOnce(false);

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));
    await screen.findByLabelText('Authenticator or recovery code');
    fireEvent.change(screen.getByLabelText('Authenticator or recovery code'), {
      target: { value: '123456' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));

    await waitFor(() => expect(apiMock.apiFetch).toHaveBeenCalledTimes(2));
    expect(apiMock.isResponseAuthContextCurrent).toHaveBeenCalledTimes(2);
    expect(screen.queryByText('OAuth authorization result is no longer current')).toBeNull();
  });

  it('begins cookie establishment before a terminal credential redirect', async () => {
    apiMock.apiFetch.mockResolvedValue({
      ok: true,
      redirected: false,
      json: async () => ({ redirect_url: '#oauth-callback' }),
    } as Response);

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => expect(apiMock.beginResponseSessionEstablishment).toHaveBeenCalledTimes(1));
    expect(apiMock.beginResponseSessionEstablishment).toHaveBeenCalledTimes(1);
  });

  it('preserves a redirected terminal authorization response', async () => {
    apiMock.apiFetch.mockResolvedValue({
      ok: true,
      redirected: true,
      url: '#oauth-redirected',
    } as Response);

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));

    await waitFor(() => expect(apiMock.beginResponseSessionEstablishment).toHaveBeenCalledTimes(1));
  });

  it('offers an explicit authorization retry after TOTP verification succeeds but authorization fails', async () => {
    apiMock.apiFetch
      .mockResolvedValueOnce({
        ok: true,
        redirected: false,
        json: async () => ({
          mfa_required: true,
          mfa_challenge_token: 'challenge-token',
          mfa_methods: ['totp'],
        }),
      } as Response)
      .mockRejectedValueOnce(new Error('authorization offline'))
      .mockRejectedValueOnce(new Error('authorization still offline'));
    apiMock.verifyMfaChallenge.mockResolvedValue({ success: true });

    render(<OAuthLoginPage params={oauthParams} />);
    fireEvent.change(screen.getByLabelText('Username'), { target: { value: 'local:admin' } });
    fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));
    await screen.findByLabelText('Authenticator or recovery code');
    fireEvent.change(screen.getByLabelText('Authenticator or recovery code'), {
      target: { value: '123456' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Verify' }));

    const retry = await screen.findByRole('button', { name: 'Retry authorization' });
    expect(apiMock.verifyMfaChallenge).toHaveBeenCalledTimes(1);
    expect(apiMock.apiFetch).toHaveBeenCalledTimes(2);

    fireEvent.click(retry);
    await waitFor(() => expect(apiMock.apiFetch).toHaveBeenCalledTimes(3));
    expect(apiMock.verifyMfaChallenge).toHaveBeenCalledTimes(1);
  });

  it('announces a server failure, stays on credentials, and permits an MFA retry', async () => {
    apiMock.apiFetch
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
