import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '@/api/client';
import type { HttpApiConnectionConfig } from '@/types';

import { HttpApiOAuthConnectPanel } from './HttpApiOAuthConnectPanel';

const toastError = vi.fn();

vi.mock('./shared/Toast', () => ({
  useToast: () => [
    [],
    { success: vi.fn(), error: toastError, info: vi.fn(), dismiss: vi.fn(), clear: vi.fn() },
  ],
  ToastContainer: () => null,
}));

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
    toastError.mockReset();
    vi.restoreAllMocks();
    vi.useRealTimers();
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

  it('shows the blocked-popup recovery message and toast when PKCE popup creation is blocked', async () => {
    vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'pkce-session',
      authorization_url: 'https://issuer.example.test/authorize?state=opaque',
      retry_after_seconds: 30,
    });
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    renderPanel({
      oauth_flow: 'authorization_code_pkce',
      oauth_authorization_url: 'https://issuer.example.test/authorize',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe(
        'Your browser blocked the OAuth popup. Allow popups and try again.',
      ),
    );
    expect(open).toHaveBeenCalledWith('', '_blank');
    expect(toastError).toHaveBeenCalledWith(
      'Your browser blocked the OAuth popup. Allow popups and try again.',
    );
    expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it('surfaces a rejected start request in the panel and toast and closes the placeholder popup', async () => {
    vi.spyOn(api, 'startHttpApiOAuth').mockRejectedValue(new Error('connection reset'));
    const popup = { location: { href: '' }, close: vi.fn() } as unknown as Window;
    vi.spyOn(window, 'open').mockReturnValue(popup);
    renderPanel({
      oauth_flow: 'authorization_code_pkce',
      oauth_authorization_url: 'https://issuer.example.test/authorize',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(screen.getByRole('alert').textContent).toBe(
        'OAuth connection failed. Check the configuration and try again.',
      ),
    );
    expect(popup.close).toHaveBeenCalledTimes(1);
    expect(toastError).toHaveBeenCalledWith(
      'OAuth connection failed. Check the configuration and try again.',
    );
    expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it('treats an incomplete HTTP 200 PKCE start response as an actionable error', async () => {
    vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'pkce-session',
      retry_after_seconds: 30,
    });
    const popup = { location: { href: '' }, close: vi.fn() } as unknown as Window;
    vi.spyOn(window, 'open').mockReturnValue(popup);
    renderPanel({
      oauth_flow: 'authorization_code_pkce',
      oauth_authorization_url: 'https://issuer.example.test/authorize',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() => {
      const status = screen.getByRole('alert').textContent ?? '';
      expect(status).toContain('incomplete');
      expect(status.toLowerCase()).toContain('try again');
    });
    expect(popup.close).toHaveBeenCalledTimes(1);
    expect(toastError).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Waiting for authorization…')).toBeNull();
    expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it('continues PKCE polling when later pending responses omit the start-only authorization URL', async () => {
    vi.useFakeTimers();
    vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'pkce-session',
      authorization_url: 'https://issuer.example.test/authorize?state=opaque',
      retry_after_seconds: 1,
    });
    vi.spyOn(api, 'pollHttpApiOAuth')
      .mockResolvedValueOnce({
        status: 'pending',
        session_id: 'pkce-session',
        retry_after_seconds: 1,
      })
      .mockResolvedValueOnce({
        status: 'connected',
        session_id: 'pkce-session',
      });
    const popup = { opener: window, location: { href: '' }, close: vi.fn() } as unknown as Window;
    vi.spyOn(window, 'open').mockReturnValue(popup);
    const { onConnected } = renderPanel({
      oauth_flow: 'authorization_code_pkce',
      oauth_authorization_url: 'https://issuer.example.test/authorize',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await Promise.resolve();
    await Promise.resolve();
    expect(popup.location.href).toContain('state=opaque');
    await vi.runAllTimersAsync();

    expect(onConnected).toHaveBeenCalledWith('pkce-session');
    expect(screen.getByRole('status').textContent).toBe(
      'Connected. Save the tool to keep this credential.',
    );
    expect(screen.queryByRole('alert')).toBeNull();
    expect(toastError).not.toHaveBeenCalled();
  });

  it('surfaces a rejected poll request in the panel and toast and clears the pending device step', async () => {
    vi.useFakeTimers();
    vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'session-1',
      verification_uri_complete: 'https://issuer.example.test/device?code=complete',
      user_code: 'ABCD-EFGH',
      interval: 0,
    });
    vi.spyOn(api, 'pollHttpApiOAuth').mockRejectedValue(new Error('timed out'));
    vi.spyOn(window, 'open').mockReturnValue(null);
    renderPanel();

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await Promise.resolve();
    await Promise.resolve();
    expect(screen.getByText('Code: ABCD-EFGH')).toBeTruthy();
    await vi.advanceTimersByTimeAsync(250);
    expect(screen.getByRole('alert').textContent).toBe(
      'OAuth connection failed. Check the configuration and try again.',
    );
    expect(screen.queryByLabelText('Device authorization')).toBeNull();
    expect(toastError).toHaveBeenCalledWith(
      'OAuth connection failed. Check the configuration and try again.',
    );
    expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it.each([
    ['failed', 'Authorization was denied at the provider.'],
    ['expired', 'This OAuth session expired. Reconnect to try again.'],
  ] as const)(
    'surfaces terminal %s status in the panel and toast',
    async (terminalStatus, expectedMessage) => {
      vi.spyOn(api, 'startHttpApiOAuth').mockResolvedValue({
        status: terminalStatus,
        session_id: 'session-1',
        message: 'Authorization was denied at the provider.',
      });
      renderPanel();

      fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

      await waitFor(() => expect(screen.getByRole('alert').textContent).toBe(expectedMessage));
      expect(toastError).toHaveBeenCalledWith(expectedMessage);
      expect((screen.getByRole('button', { name: 'Connect' }) as HTMLButtonElement).disabled).toBe(
        false,
      );
    },
  );

  it('announces OAuth errors with alert semantics while keeping non-error messages as status', async () => {
    vi.spyOn(api, 'startHttpApiOAuth')
      .mockResolvedValueOnce({
        status: 'failed',
        session_id: 'session-1',
        message: 'Authorization was denied at the provider.',
      })
      .mockResolvedValueOnce({
        status: 'pending',
        session_id: 'session-2',
        verification_uri_complete: 'https://issuer.example.test/device?code=complete',
        user_code: 'WXYZ-1234',
        interval: 60,
      });
    vi.spyOn(window, 'open').mockReturnValue(null);
    renderPanel();

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toBe(
        'Authorization was denied at the provider.',
      );
    });

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() => {
      expect(screen.getByRole('status').textContent).toBe('Waiting for authorization…');
    });
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
