import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { App } from './App';
import { api } from './api';
import { sessionLifecycle } from './auth/sessionLifecycle';

vi.mock('./components/WebGLGradient', () => ({ default: () => null }));

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('App anonymous HTTP bootstrap', () => {
  beforeEach(() => {
    if (sessionLifecycle.phase === 'logging-out') {
      sessionLifecycle.finishLogout(sessionLifecycle.capture('public'));
    }
    if (sessionLifecycle.phase === 'authenticated' || sessionLifecycle.phase === 'establishing') {
      sessionLifecycle.markAnonymous(sessionLifecycle.capture('public'));
    }
    if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    window.history.replaceState({}, '', '/');
  });

  it('keeps a delayed real status response after an unrelated anonymous 401 and mounts the real login defaults', async () => {
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => undefined,
      removeItem: () => undefined,
    });
    let releaseStatus!: () => void;
    const delayedStatus = new Promise<Response>((resolve) => {
      releaseStatus = () =>
        resolve(
          json({
            authenticated: false,
            ldap_configured: false,
            local_admin_enabled: true,
            debug_mode: false,
            api_key_configured: false,
            session_cookie_secure: false,
            allowed_origins_open: false,
            debug_username: 'dev-user',
            debug_password: 'dev-password',
          }),
        );
    });
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/auth/status')) return delayedStatus;
        if (url.includes('/indexes/settings'))
          return Promise.resolve(json({ detail: 'unauthorized' }, 401));
        if (url.includes('/auth/debug/totp')) return Promise.resolve(json({ code: '123456' }));
        return Promise.resolve(json({ detail: 'not found' }, 404));
      }),
    );

    render(<App />);
    expect(screen.getByText('Loading...')).toBeTruthy();
    // This is deliberately an unrelated session-purpose 401 while bootstrap is pending.
    void api.getSettings().catch(() => undefined);
    await act(async () => releaseStatus());

    await waitFor(() =>
      expect((screen.getByLabelText(/username/i) as HTMLInputElement).value).toBe('dev-user'),
    );
    expect((screen.getByLabelText(/password/i) as HTMLInputElement).value).toBe('dev-password');
    expect(
      vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('/indexes/settings')),
    ).toHaveLength(1);
  });

  it('keeps the terminal-login verification error visible when the issued cookie is immediately rejected', async () => {
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => undefined,
      removeItem: () => undefined,
    });
    const anonymousStatus = {
      authenticated: false,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: false,
      session_cookie_secure: false,
      allowed_origins_open: false,
      cookie_warning: 'Cookies must be enabled.',
    };
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/auth/status')) return Promise.resolve(json(anonymousStatus));
        if (url.includes('/auth/login'))
          return Promise.resolve(json({ success: true, user_id: 'u1' }));
        if (url.includes('/auth/me')) return Promise.resolve(json({ detail: 'expired' }, 401));
        return Promise.resolve(json({ detail: 'not found' }, 404));
      }),
    );

    render(<App />);
    await screen.findByLabelText(/username/i);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'u1' } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await screen.findByText('Your sign-in session could not be verified. Please sign in again.');
    expect(screen.getByText('Cookies must be enabled.')).toBeTruthy();
    expect(screen.getByLabelText(/username/i)).toBeTruthy();
  });

  it('recovers a real StrictMode 503 bootstrap and restores debug login defaults with explicit focus', async () => {
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => undefined,
      removeItem: () => undefined,
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(json({ detail: 'temporarily unavailable' }, 503))
      .mockResolvedValueOnce(
        json({
          authenticated: false,
          ldap_configured: false,
          local_admin_enabled: true,
          debug_mode: true,
          api_key_configured: false,
          session_cookie_secure: false,
          allowed_origins_open: false,
          debug_username: 'dev-user',
          debug_password: 'dev-password',
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );

    const retry = await screen.findByRole('button', { name: 'Try connecting again' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fireEvent.click(retry);

    const username = (await screen.findByLabelText(/username/i)) as HTMLInputElement;
    expect(username.value).toBe('dev-user');
    expect((screen.getByLabelText(/password/i) as HTMLInputElement).value).toBe('dev-password');
    expect(document.activeElement).toBe(username);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
