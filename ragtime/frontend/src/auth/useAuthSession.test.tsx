import { StrictMode, type ReactNode } from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AuthStatus, User } from '@/types';
import { createSessionLifecycle } from './sessionLifecycle';
import { useAuthSession } from './useAuthSession';

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  logout: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

afterEach(() => vi.resetAllMocks());

function status(overrides: Partial<AuthStatus> = {}): AuthStatus {
  return {
    authenticated: false,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: false,
    api_key_configured: false,
    session_cookie_secure: false,
    allowed_origins_open: false,
    chat_enabled: false,
    userspace_generation_enabled: false,
    ...overrides,
  };
}

function user(overrides: Partial<User> = {}): User {
  return {
    id: 'user-1',
    username: 'local:admin',
    display_name: 'Admin',
    email: null,
    auth_provider: 'local_managed',
    role: 'admin',
    chat_enabled: null,
    userspace_generation_enabled: null,
    chat_enabled_effective: true,
    userspace_generation_enabled_effective: true,
    ...overrides,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function unauthorized(message = 'Session expired') {
  return Object.assign(new Error(message), { status: 401 });
}

async function renderAuthenticated(
  overrides: {
    authStatus?: Partial<AuthStatus>;
    currentUser?: Partial<User>;
  } = {},
) {
  const lifecycle = createSessionLifecycle();
  apiMock.getAuthStatus.mockResolvedValueOnce(
    status({
      authenticated: true,
      chat_enabled: true,
      userspace_generation_enabled: true,
      ...overrides.authStatus,
    }),
  );
  apiMock.getCurrentUser.mockResolvedValueOnce(user(overrides.currentUser));
  const hook = renderHook(() => useAuthSession(lifecycle));
  await waitFor(() => expect(hook.result.current.phase).toBe('authenticated'));
  return { lifecycle, ...hook };
}

describe('useAuthSession', () => {
  it('uses an isolated lifecycle supplied by its owner', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus.mockResolvedValue(status());

    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    expect(lifecycle.phase).toBe('anonymous');
    unmount();
  });

  it('publishes authoritative anonymous metadata without a protected current-user read', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus.mockResolvedValue(
      status({ debug_username: 'development-user', debug_password: 'development-password' }),
    );

    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    expect(result.current.authStatus?.debug_username).toBe('development-user');
    expect(apiMock.getCurrentUser).not.toHaveBeenCalled();
    unmount();
  });

  it.each([
    [true, true, true],
    [true, false, false],
    [false, true, false],
    [false, false, false],
    [true, undefined, false],
    [undefined, true, false],
  ] as const)(
    'requires true from status=%s and user=%s before publishing a capability',
    async (statusFlag, userFlag, expected) => {
      const lifecycle = createSessionLifecycle();
      apiMock.getAuthStatus.mockResolvedValueOnce(
        status({ authenticated: true, chat_enabled: statusFlag as boolean }),
      );
      apiMock.getCurrentUser.mockResolvedValueOnce(
        user({ chat_enabled_effective: userFlag as boolean }),
      );
      const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

      await waitFor(() => expect(result.current.phase).toBe('authenticated'));
      expect(result.current.authStatus?.chat_enabled).toBe(expected);
      expect(result.current.currentUser?.chat_enabled_effective).toBe(expected);
      unmount();
    },
  );

  it('joins bootstrap, return, and step-up callers to the actual in-flight ticket', async () => {
    const lifecycle = createSessionLifecycle();
    const pending = deferred<AuthStatus>();
    apiMock.getAuthStatus.mockReturnValue(pending.promise);
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    const returned = result.current.refresh('return');
    const stepUp = result.current.refresh('step-up');
    expect(returned).toBe(stepUp);

    await act(async () => {
      pending.resolve(status());
      await returned;
    });
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('lets an explicit save supersede an older read without stale success or finally publication', async () => {
    const lifecycle = createSessionLifecycle();
    const oldStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus
      .mockReturnValueOnce(oldStatus.promise)
      .mockResolvedValueOnce(status({ debug_username: 'new-status' }));
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await act(async () => {
      await result.current.refresh('policy-save');
    });
    expect(result.current.authStatus?.debug_username).toBe('new-status');

    await act(async () => {
      oldStatus.resolve(status({ authenticated: true, chat_enabled: true }));
      await oldStatus.promise;
    });
    expect(result.current.authStatus?.debug_username).toBe('new-status');
    expect(apiMock.getCurrentUser).not.toHaveBeenCalled();
    unmount();
  });

  it('performs one bounded public recovery when bootstrap status is true but current user is unauthorized', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus
      .mockResolvedValueOnce(status({ authenticated: true, chat_enabled: true }))
      .mockResolvedValueOnce(status({ debug_username: 'recovered-user' }));
    apiMock.getCurrentUser.mockRejectedValueOnce(unauthorized());
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    expect(result.current.authStatus?.debug_username).toBe('recovered-user');
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2);
    expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('bounds contradictory bootstrap recovery when the fresh status still reports authenticated', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus.mockResolvedValue(status({ authenticated: true, chat_enabled: true }));
    apiMock.getCurrentUser.mockRejectedValueOnce(unauthorized());
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await waitFor(() => expect(result.current.phase).toBe('unavailable'));
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2);
    expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('recovers a post-login current-user 401 through one fresh public read', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus
      .mockResolvedValueOnce(status())
      .mockResolvedValueOnce(status({ authenticated: true }))
      .mockResolvedValueOnce(status({ debug_username: 'login-again' }));
    apiMock.getCurrentUser.mockRejectedValueOnce(unauthorized());
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    lifecycle.beginEstablishment(lifecycle.capture('challenge'));

    await act(async () => {
      await result.current.completeLogin(user()).catch(() => undefined);
    });

    expect(result.current.phase).toBe('anonymous');
    expect(result.current.authStatus?.debug_username).toBe('login-again');
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(3);
    unmount();
  });

  it('settles StrictMode bootstrap after its delayed authoritative response', async () => {
    const lifecycle = createSessionLifecycle();
    const pending = deferred<AuthStatus>();
    apiMock.getAuthStatus.mockReturnValue(pending.promise);
    const wrapper = ({ children }: { children: ReactNode }) => <StrictMode>{children}</StrictMode>;
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle), { wrapper });

    await act(async () => pending.resolve(status()));

    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('prevents passive refresh and retry from bypassing an in-flight logout and sends one logout request', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    const pendingLogout = deferred<void>();
    apiMock.logout.mockReturnValue(pendingLogout.promise);
    apiMock.getAuthStatus.mockResolvedValueOnce(status());
    const first = result.current.logout();
    const second = result.current.logout();

    await act(async () => {
      await result.current.refresh('return');
      await result.current.retryBootstrap();
    });
    expect(apiMock.logout).toHaveBeenCalledTimes(1);
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(1);
    expect(lifecycle.phase).toBe('logging-out');

    await act(async () => {
      pendingLogout.resolve();
      await Promise.all([first, second]);
    });
    expect(result.current.phase).toBe('anonymous');
    unmount();
  });

  it('keeps failed logout locally signed out and blocks focus adoption until logout is retried', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    apiMock.logout.mockRejectedValueOnce(new Error('offline'));

    await act(async () => result.current.logout());

    expect(result.current.phase).toBe('unavailable');
    expect(result.current.recoveryAction).toBe('retry-logout');
    await act(async () => {
      await result.current.refresh('return');
    });
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(1);
    expect(lifecycle.phase).toBe('logging-out');
    unmount();
  });

  it('keeps recovery visible when successful logout is followed by a still-authenticated status', async () => {
    const { result, unmount } = await renderAuthenticated();
    apiMock.logout.mockResolvedValueOnce(undefined);
    apiMock.getAuthStatus.mockResolvedValueOnce(status({ authenticated: true }));

    await act(async () => result.current.logout());

    expect(result.current.phase).toBe('unavailable');
    expect(result.current.recoveryAction).toBe('check-session');
    expect(result.current.currentUser).toBeNull();
    unmount();
  });

  it('fails closed in both published snapshots after a policy-save refresh failure without changing raw overrides', async () => {
    const { result, unmount } = await renderAuthenticated({
      currentUser: { chat_enabled: true, userspace_generation_enabled: false },
    });
    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('read failed'));

    await act(async () => {
      await result.current.refresh('policy-save').catch(() => undefined);
    });

    expect(result.current.phase).toBe('authenticated');
    expect(result.current.authStatus?.chat_enabled).toBe(false);
    expect(result.current.authStatus?.userspace_generation_enabled).toBe(false);
    expect(result.current.currentUser?.chat_enabled_effective).toBe(false);
    expect(result.current.currentUser?.userspace_generation_enabled_effective).toBe(false);
    expect(result.current.currentUser?.chat_enabled).toBe(true);
    expect(result.current.currentUser?.userspace_generation_enabled).toBe(false);
    unmount();
  });

  it('suspends both generation capabilities during same-user renewal and keeps them closed after failure', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    const pendingStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus.mockReturnValueOnce(pendingStatus.promise);
    const renewal = lifecycle.beginRenewal(lifecycle.capture('session'))!;

    act(() => {
      lifecycle.renewSession(renewal, 'user-1');
    });

    expect(result.current.phase).toBe('authenticated');
    expect(result.current.authStatus?.chat_enabled).toBe(false);
    expect(result.current.currentUser?.chat_enabled_effective).toBe(false);

    await act(async () => pendingStatus.reject(new Error('renewal read failed')));
    await waitFor(() => expect(result.current.refreshError).toBe('renewal read failed'));
    expect(result.current.authStatus?.chat_enabled).toBe(false);
    expect(result.current.currentUser?.chat_enabled_effective).toBe(false);
    unmount();
  });

  it('keeps the authenticated snapshot when an old cookie reports anonymous during renewal', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    lifecycle.beginRenewal(lifecycle.capture('session'));
    apiMock.getAuthStatus.mockResolvedValueOnce(status());

    await act(async () => {
      await result.current.refresh('return');
    });

    expect(lifecycle.phase).toBe('authenticated');
    expect(result.current.phase).toBe('authenticated');
    expect(result.current.currentUser?.id).toBe('user-1');
    expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('keeps the authenticated snapshot when an old-cookie current-user 401 arrives during renewal', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    lifecycle.beginRenewal(lifecycle.capture('session'));
    apiMock.getAuthStatus.mockResolvedValueOnce(status({ authenticated: true }));
    apiMock.getCurrentUser.mockImplementationOnce(() => {
      lifecycle.expire(lifecycle.capture('session'));
      return Promise.reject(unauthorized());
    });

    await act(async () => {
      await result.current.refresh('return').catch(() => undefined);
    });

    expect(lifecycle.phase).toBe('authenticated');
    expect(result.current.phase).toBe('authenticated');
    expect(result.current.currentUser?.id).toBe('user-1');
    unmount();
  });

  it('reports a rejected step-up revalidation without an unhandled promise', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('revalidation failed'));

    act(() => {
      lifecycle.revalidate(lifecycle.capture('session'));
    });

    await waitFor(() => expect(result.current.refreshError).toBe('revalidation failed'));
    expect(result.current.phase).toBe('authenticated');
    unmount();
  });

  it('does not let focus refresh unmount an authentication challenge before explicit completion', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus.mockResolvedValue(status({ debug_username: 'typed-user' }));
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    lifecycle.beginEstablishment(lifecycle.capture('challenge'));

    await act(async () => {
      expect(await result.current.refresh('return')).toBe('superseded');
    });

    expect(result.current.phase).toBe('anonymous');
    expect(result.current.authStatus?.debug_username).toBe('typed-user');
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('clears a current protected session immediately and retains the signed-out check-session latch', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    apiMock.getAuthStatus.mockResolvedValueOnce(status({ debug_username: 'expired-user' }));

    act(() => {
      lifecycle.expire(lifecycle.capture('session'));
    });

    await waitFor(() => expect(result.current.phase).toBe('anonymous'));
    expect(result.current.currentUser).toBeNull();
    expect(result.current.authStatus?.debug_username).toBe('expired-user');
    expect(result.current.recoveryAction).toBe('check-session');
    unmount();
  });

  it('distinguishes successful logout followed by a failed metadata read from server logout failure', async () => {
    const { result, unmount } = await renderAuthenticated();
    apiMock.logout.mockResolvedValueOnce(undefined);
    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('status unavailable'));

    await act(async () => result.current.logout());

    expect(result.current.phase).toBe('unavailable');
    expect(result.current.recoveryAction).toBe('check-session');
    expect(result.current.refreshError).toBe('status unavailable');
    expect(result.current.refreshError).not.toContain('sign-out could not be confirmed');
    unmount();
  });

  it('exposes busy state and safely joins repeated bootstrap retries', async () => {
    const lifecycle = createSessionLifecycle();
    const retryStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus
      .mockRejectedValueOnce(new Error('offline'))
      .mockReturnValueOnce(retryStatus.promise);
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('unavailable'));

    let first!: Promise<void>;
    let second!: Promise<void>;
    act(() => {
      first = result.current.retryBootstrap();
      second = result.current.retryBootstrap();
    });
    expect(result.current.recoveryBusy).toBe(true);
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2);

    await act(async () => {
      retryStatus.resolve(status());
      await Promise.all([first, second]);
    });
    expect(result.current.phase).toBe('anonymous');
    expect(result.current.recoveryBusy).toBe(false);
    unmount();
  });

  it('restores both effective generation capabilities only after renewal verification succeeds', async () => {
    const { lifecycle, result, unmount } = await renderAuthenticated();
    apiMock.getAuthStatus.mockResolvedValueOnce(
      status({ authenticated: true, chat_enabled: true, userspace_generation_enabled: true }),
    );
    apiMock.getCurrentUser.mockResolvedValueOnce(user());
    const renewal = lifecycle.beginRenewal(lifecycle.capture('session'))!;

    act(() => {
      lifecycle.renewSession(renewal, 'user-1');
    });
    expect(result.current.authStatus?.chat_enabled).toBe(false);
    expect(result.current.currentUser?.userspace_generation_enabled_effective).toBe(false);

    await waitFor(() => expect(result.current.authStatus?.chat_enabled).toBe(true));
    expect(result.current.authStatus?.userspace_generation_enabled).toBe(true);
    expect(result.current.currentUser?.chat_enabled_effective).toBe(true);
    expect(result.current.currentUser?.userspace_generation_enabled_effective).toBe(true);
    unmount();
  });

  it('fails closed for a missing User Space effective field while preserving the raw overrides', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus.mockResolvedValueOnce(
      status({ authenticated: true, chat_enabled: true, userspace_generation_enabled: true }),
    );
    apiMock.getCurrentUser.mockResolvedValueOnce(
      user({
        chat_enabled: true,
        userspace_generation_enabled: false,
        chat_enabled_effective: true,
        userspace_generation_enabled_effective: undefined,
      }),
    );
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));

    await waitFor(() => expect(result.current.phase).toBe('authenticated'));
    expect(result.current.authStatus?.chat_enabled).toBe(true);
    expect(result.current.authStatus?.userspace_generation_enabled).toBe(false);
    expect(result.current.currentUser?.userspace_generation_enabled_effective).toBe(false);
    expect(result.current.currentUser?.chat_enabled).toBe(true);
    expect(result.current.currentUser?.userspace_generation_enabled).toBe(false);
    unmount();
  });

  it('clears a terminal-establishment failure latch after authenticated metadata recovery', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus
      .mockResolvedValueOnce(status())
      .mockResolvedValueOnce(status({ authenticated: true }))
      .mockResolvedValueOnce(status());
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('anonymous'));

    act(() => {
      lifecycle.beginEstablishment(lifecycle.capture('challenge'));
      lifecycle.expire(lifecycle.capture('session'));
    });
    await waitFor(() => expect(result.current.recoveryAction).toBe('check-session'));

    await act(async () => result.current.retryBootstrap());
    expect(result.current.phase).toBe('anonymous');
    expect(result.current.refreshError).toBeNull();
    unmount();
  });

  it('clears a prior terminal-establishment failure when the next sign-in starts', async () => {
    const lifecycle = createSessionLifecycle();
    apiMock.getAuthStatus
      .mockResolvedValueOnce(status())
      .mockRejectedValueOnce(new Error('metadata offline'))
      .mockResolvedValueOnce(status({ authenticated: true }))
      .mockResolvedValueOnce(status());
    apiMock.getCurrentUser.mockResolvedValueOnce(user());
    apiMock.logout.mockResolvedValueOnce(undefined);
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('anonymous'));

    act(() => {
      lifecycle.beginEstablishment(lifecycle.capture('challenge'));
      lifecycle.expire(lifecycle.capture('session'));
    });
    await waitFor(() => expect(result.current.phase).toBe('unavailable'));

    act(() => {
      lifecycle.beginEstablishment(lifecycle.capture('challenge'));
    });
    await act(async () => {
      await result.current.completeLogin(user());
    });
    expect(result.current.phase).toBe('authenticated');

    await act(async () => result.current.logout());
    expect(result.current.phase).toBe('anonymous');
    expect(result.current.refreshError).toBeNull();
    unmount();
  });

  it('clears establishment error when refreshing to authenticated after prior 401 during login', async () => {
    const lifecycle = createSessionLifecycle();
    // Initial bootstrap returns anonymous
    apiMock.getAuthStatus.mockResolvedValueOnce(status());
    const { result, unmount } = renderHook(() => useAuthSession(lifecycle));
    await waitFor(() => expect(result.current.phase).toBe('anonymous'));

    // Begin establishment and simulate login with 401 followed by recovery to anonymous
    act(() => {
      lifecycle.beginEstablishment(lifecycle.capture('challenge'));
    });
    apiMock.getAuthStatus.mockResolvedValueOnce(status({ authenticated: true }));
    apiMock.getCurrentUser.mockRejectedValueOnce(unauthorized());
    // After 401, recovery gets anonymous status
    apiMock.getAuthStatus.mockResolvedValueOnce(status());

    await act(async () => {
      await result.current.completeLogin(user()).catch(() => undefined);
    });

    // Verify we recovered to anonymous without an error (error was cleared by prior tests)
    expect(result.current.phase).toBe('anonymous');
    expect(result.current.refreshError).toBeNull();

    // Now test that if we get authenticated during refresh, the error is cleared
    act(() => {
      lifecycle.beginEstablishment(lifecycle.capture('challenge'));
    });
    apiMock.getAuthStatus.mockResolvedValueOnce(status({ authenticated: true }));
    apiMock.getCurrentUser.mockResolvedValueOnce(user());

    await act(async () => {
      await result.current.completeLogin(user());
    });

    // After adopting authenticated session, error must be null
    await waitFor(() => expect(result.current.phase).toBe('authenticated'));
    expect(result.current.refreshError).toBeNull();
    unmount();
  });
});
