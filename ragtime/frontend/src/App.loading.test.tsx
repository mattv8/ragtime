import { act, cleanup, render, screen, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  getSettings: vi.fn(),
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
    }: {
      shareToken?: string;
      ownerUsername?: string;
      shareSlug?: string;
    }) => <div>{shareToken ?? `${ownerUsername}/${shareSlug}`}</div>,
  };
});

import { App } from './App';

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolver) => {
    resolve = resolver;
  });
  return { promise, resolve };
}

describe('App shared chat loading', () => {
  beforeEach(() => {
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });
    apiMock.getCurrentUser.mockResolvedValue(null);
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
});
