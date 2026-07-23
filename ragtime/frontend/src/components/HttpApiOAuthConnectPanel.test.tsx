import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '@/api/client';
import type { HttpApiConnectionConfig } from '@/types';

import { HttpApiOAuthConnectPanel } from './HttpApiOAuthConnectPanel';

const baseValue: HttpApiConnectionConfig = {
  auth_mode: 'oauth2',
  oauth_flow: 'device_code',
  oauth_issuer_url: 'https://issuer.example.test',
  oauth_client_id: 'client-id',
  oauth_scopes: ['openid'],
};

function renderPanel(overrides: Partial<HttpApiConnectionConfig> = {}) {
  const onChange = vi.fn();
  const onConnected = vi.fn();
  render(
    <HttpApiOAuthConnectPanel
      value={{ ...baseValue, ...overrides }}
      configuredSecretFields={[]}
      onChange={onChange}
      onConnected={onConnected}
    />,
  );
  return { onChange, onConnected };
}

describe('HttpApiOAuthConnectPanel', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('fills editable endpoint and capability fields from discovery', async () => {
    vi.spyOn(api, 'discoverHttpApiOAuth').mockResolvedValue({
      issuer: 'https://issuer.example.test',
      authorization_endpoint: 'https://issuer.example.test/authorize',
      device_authorization_endpoint: 'https://issuer.example.test/device',
      token_endpoint: 'https://issuer.example.test/token',
      grant_types_supported: ['device_code', 'authorization_code'],
      code_challenge_methods_supported: ['S256'],
      scopes_supported: ['openid', 'profile'],
      token_endpoint_auth_methods_supported: ['none'],
    });
    const { onChange } = renderPanel();

    fireEvent.click(screen.getByRole('button', { name: 'Discover' }));

    await waitFor(() =>
      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          oauth_authorization_url: 'https://issuer.example.test/authorize',
          oauth_device_authorization_url: 'https://issuer.example.test/device',
          oauth_token_url: 'https://issuer.example.test/token',
          oauth_scopes: ['openid', 'profile'],
        }),
      ),
    );
    expect(screen.getByText('Advanced endpoints')).toBeTruthy();
  });

  it('copies the device code, opens the preferred URL, and polls to connected', async () => {
    vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'session-1',
      verification_uri: 'https://issuer.example.test/device',
      verification_uri_complete: 'https://issuer.example.test/device?code=complete',
      user_code: 'ABCD-EFGH',
      interval: 1,
    });
    vi.spyOn(api, 'pollHttpApiOAuth').mockResolvedValue({
      status: 'connected',
      session_id: 'session-1',
    });
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    const { onConnected } = renderPanel();

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() => expect(screen.getByText('Code: ABCD-EFGH')).toBeTruthy());
    expect(open).toHaveBeenCalledWith(
      'https://issuer.example.test/device?code=complete',
      '_blank',
      'noopener,noreferrer',
    );
    await waitFor(() => expect(onConnected).toHaveBeenCalledWith('session-1'), { timeout: 2_000 });
  });

  it('opens a synchronous noopener PKCE popup before awaiting start and navigates it', async () => {
    let resolveStart:
      | ((value: Awaited<ReturnType<typeof api.startHttpApiOAuth>>) => void)
      | undefined;
    vi.spyOn(api, 'startHttpApiOAuth').mockReturnValue(
      new Promise((resolve) => {
        resolveStart = resolve;
      }),
    );
    const popup = { location: { href: '' }, close: vi.fn() } as unknown as Window;
    const open = vi.spyOn(window, 'open').mockReturnValue(popup);
    renderPanel({
      oauth_flow: 'authorization_code_pkce',
      oauth_authorization_url: 'https://issuer.example.test/authorize',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));
    expect(open).toHaveBeenCalledWith('', '_blank');
    expect(popup.opener).toBeNull();
    resolveStart?.({
      status: 'pending',
      session_id: 'pkce-session',
      authorization_url: 'https://issuer.example.test/authorize?state=opaque',
      retry_after_seconds: 30,
    });
    await waitFor(() => expect(popup.location.href).toContain('state=opaque'));
  });

  it('masks a loaded client secret, reveals it locally, and keeps saved-token status separate', () => {
    const onChange = vi.fn();

    render(
      <HttpApiOAuthConnectPanel
        value={{ ...baseValue, oauth_client_secret: 'client-secret-value' }}
        configuredSecretFields={['oauth_client_secret', 'oauth_access_token']}
        onChange={onChange}
        onConnected={vi.fn()}
      />,
    );

    const input = screen.getByLabelText('Client secret (optional)') as HTMLInputElement;
    expect(input.value).toBe('client-secret-value');
    expect(input.type).toBe('password');
    expect(screen.queryByText('Saved secret')).toBeNull();
    expect(screen.queryByText(/\(saved\)/i)).toBeNull();
    expect(screen.getByRole('status').textContent).toContain('Connected');

    fireEvent.click(screen.getByRole('button', { name: 'Show client secret' }));

    expect(input.type).toBe('text');
    expect(screen.getByRole('button', { name: 'Hide client secret' })).toBeTruthy();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('uses semantic URL/text input types and the OAuth fill-width class', () => {
    render(
      <HttpApiOAuthConnectPanel
        value={baseValue}
        configuredSecretFields={[]}
        onChange={vi.fn()}
        onConnected={vi.fn()}
      />,
    );

    expect(screen.getByLabelText('Issuer URL').getAttribute('type')).toBe('url');
    expect(screen.getByLabelText('Client ID').getAttribute('type')).toBe('text');
    expect(screen.getByLabelText('Scopes').getAttribute('type')).toBe('text');
    expect(screen.getByLabelText('Authorization endpoint').getAttribute('type')).toBe('url');
    expect(screen.getByLabelText('Device authorization endpoint').getAttribute('type')).toBe('url');
    expect(screen.getByLabelText('Token endpoint').getAttribute('type')).toBe('url');
    expect(screen.getByLabelText('Issuer URL').classList.contains('form-input')).toBe(true);
  });
});
