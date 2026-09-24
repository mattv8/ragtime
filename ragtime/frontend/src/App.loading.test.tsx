import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState, type ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  getSettings: vi.fn(),
  logout: vi.fn(),
}));
const localStorageMock = vi.hoisted(() => ({
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
}));
const sharedViewModuleGate = vi.hoisted(() => {
  let resolve: (() => void) | null = null;
  const promise = new Promise<void>((resolver) => {
    resolve = resolver;
  });
  return {
    promise,
    requested: vi.fn(),
    resolve: () => resolve?.(),
  };
});

vi.mock('@/api', () => ({
  api: apiMock,
  onAuthExpired: vi.fn(() => vi.fn()),
}));
vi.stubGlobal('localStorage', localStorageMock);

vi.mock('@/components/WebGLGradient', () => ({ default: () => null }));
vi.mock('@/theme', () => ({ resolveThemePackId: vi.fn(), setThemePack: vi.fn() }));
vi.mock('./components/ConfigurationBanner', () => ({ ConfigurationBanner: () => null }));
vi.mock('./components/LoginGradientShell', () => ({
  LoginGradientShell: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));
vi.mock('./components/LoginPage', () => ({ LoginPage: () => null }));
vi.mock('./components/MemoryStatus', () => ({ MemoryStatus: () => null }));
vi.mock('./components/OAuthCallbackError', () => ({ OAuthCallbackError: () => null }));
vi.mock('./components/OAuthLoginPage', () => ({ OAuthLoginPage: () => null }));
vi.mock('./components/SecurityBanner', () => ({ SecurityBanner: () => null }));
vi.mock('./components/UserMenu', () => ({ UserMenu: () => null }));
vi.mock('./components/WarningsBanner', () => ({ WarningsBanner: () => null }));
vi.mock('./components/shared/Toast', () => ({
  ToastContainer: () => null,
  useToast: () => [[], { dismiss: vi.fn(), error: vi.fn(), success: vi.fn() }] as const,
}));
vi.mock('./components/PublicSharedChatView', async () => {
  sharedViewModuleGate.requested();
  await sharedViewModuleGate.promise;
  return {
    PublicSharedChatView: ({
      shareToken,
      ownerUsername,
      shareSlug,
      onLogout,
    }: {
      shareToken?: string;
      ownerUsername?: string;
      shareSlug?: string;
      onLogout: () => Promise<void>;
    }) => {
      const [password, setPassword] = useState('');
      return (
        <div data-testid="shared-view-instance">
          <span>{shareToken ?? `${ownerUsername}/${shareSlug}`}</span>
          <button type="button" onClick={() => void onLogout()}>
            Shared sign out
          </button>
          <label>
            Share password
            <input value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
        </div>
      );
    },
  };
});

import { App } from './App';
import { sessionLifecycle } from './auth/sessionLifecycle';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolver) => {
    resolve = resolver;
  });
  return { promise, resolve };
}

describe('App shared chat loading', () => {
  beforeEach(() => {
    if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });
    apiMock.getCurrentUser.mockResolvedValue(null);
    apiMock.logout.mockResolvedValue(undefined);
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    window.history.replaceState({}, '', '/');
  });

  it('waits for auth to settle before requesting the shared view, then shows its suspense fallback', async () => {
    const authStatus = deferred<{ authenticated: boolean }>();
    apiMock.getAuthStatus.mockReturnValue(authStatus.promise);
    window.history.replaceState({}, '', '/shared/token-123');

    render(<App />);

    expect(screen.getByText('Loading...')).toBeTruthy();
    expect(sharedViewModuleGate.requested).not.toHaveBeenCalled();

    await act(async () => {
      authStatus.resolve({ authenticated: false });
    });

    await waitFor(() => {
      expect(sharedViewModuleGate.requested).toHaveBeenCalledTimes(1);
      expect(screen.getByText('Loading view...')).toBeTruthy();
    });

    await act(async () => {
      sharedViewModuleGate.resolve();
    });

    expect(await screen.findByText('token-123')).toBeTruthy();
  });

  it('renders the lazy shared view for canonical owner/slug routes', async () => {
    apiMock.getAuthStatus.mockResolvedValue({ authenticated: false });
    window.history.replaceState({}, '', '/owner-a/shared-chat');

    render(<App />);

    // This test may run alone, before any other test has released the lazy module.
    await act(async () => {
      sharedViewModuleGate.resolve();
    });

    expect(await screen.findByText('owner-a/shared-chat')).toBeTruthy();
  });

  it('keeps an already-mounted shared view and its local input mounted through expiry recovery failure', async () => {
    apiMock.getAuthStatus.mockResolvedValueOnce({
      authenticated: true,
      chat_enabled: true,
      userspace_generation_enabled: true,
    });
    apiMock.getCurrentUser.mockResolvedValueOnce({
      id: 'shared-user',
      username: 'shared-user',
      display_name: 'Shared User',
      email: null,
      role: 'user',
      auth_provider: 'local',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    window.history.replaceState({}, '', '/shared/persistent-token');

    render(<App />);
    await act(async () => sharedViewModuleGate.resolve());
    const password = (await screen.findByLabelText('Share password')) as HTMLInputElement;
    fireEvent.change(password, { target: { value: 'kept-locally' } });

    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('offline'));
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    await waitFor(() => expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId('shared-view-instance')).toBeTruthy();
    expect((screen.getByLabelText('Share password') as HTMLInputElement).value).toBe(
      'kept-locally',
    );
    expect(document.querySelector('#auth-recovery-state')).toBeNull();
    expect(screen.getByRole('button', { name: 'Check session again' })).toBeTruthy();
  });

  it('retries a failed shared-route logout without remounting the share or dispatching twice', async () => {
    const retryLogout = deferred<void>();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockResolvedValueOnce({ authenticated: false });
    apiMock.getCurrentUser.mockResolvedValueOnce({
      id: 'shared-user',
      username: 'shared-user',
      display_name: 'Shared User',
      email: null,
      role: 'user',
      auth_provider: 'local',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    apiMock.logout
      .mockRejectedValueOnce(new Error('offline'))
      .mockReturnValueOnce(retryLogout.promise);
    window.history.replaceState({}, '', '/shared/logout-token');

    render(<App />);
    await act(async () => sharedViewModuleGate.resolve());
    const sharedView = await screen.findByTestId('shared-view-instance');
    const password = screen.getByLabelText('Share password') as HTMLInputElement;
    fireEvent.change(password, { target: { value: 'preserve-me' } });
    fireEvent.click(screen.getByRole('button', { name: 'Shared sign out' }));

    const retry = await screen.findByRole('button', { name: 'Retry sign out' });
    const banner = document.querySelector('#public-shared-auth-recovery-banner') as HTMLElement;
    const route = document.querySelector('#public-shared-route');
    expect(banner).toBeTruthy();
    expect(route).toBeTruthy();
    expect(banner.parentElement).toBe(route?.parentElement);
    expect(banner.parentElement).not.toBe(route);
    expect(banner.style.position).toBe('fixed');
    expect(banner.style.maxWidth).toBeTruthy();
    expect(banner.style.background).toBe('var(--color-surface)');
    expect(banner.style.zIndex).toBe('1000');
    expect(sharedView.isConnected).toBe(true);

    fireEvent.click(retry);
    fireEvent.click(retry);
    expect(apiMock.logout).toHaveBeenCalledTimes(2);
    expect(retry).toHaveProperty('disabled', true);

    await act(async () => retryLogout.resolve());
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Retry sign out' })).toBeNull(),
    );
    expect(screen.getByTestId('shared-view-instance')).toBe(sharedView);
    expect((screen.getByLabelText('Share password') as HTMLInputElement).value).toBe('preserve-me');
  });
});
