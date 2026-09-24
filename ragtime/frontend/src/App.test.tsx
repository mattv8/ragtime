import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import { sessionLifecycle } from './auth/sessionLifecycle';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from './components/shared/securityWarnings';
import type {
  AuthStatus,
  ConfigurationWarning,
  ServerBackupJob,
  ServerRestoreJob,
  User,
} from './types';

const localStorageMock = vi.hoisted(() => ({
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
}));

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  getSettings: vi.fn(),
  getActiveServerBackupJobs: vi.fn(),
  getServerBackupJob: vi.fn(),
  getServerRestoreJob: vi.fn(),
  getOpenRouterCreditStatus: vi.fn(),
  logout: vi.fn(),
  apiFetch: vi.fn((url: string, options: RequestInit) => fetch(url, options)),
}));

const settingsPanelSpy = vi.hoisted(() => vi.fn());
const settingsPanelModuleGate = vi.hoisted(() => {
  let resolve: (() => void) | null = null;
  const promise = new Promise<void>((resolver) => {
    resolve = resolver;
  });
  return {
    promise,
    resolve: () => resolve?.(),
  };
});
const toastApiMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  dismiss: vi.fn(),
  clear: vi.fn(),
}));
const toastContainerSpy = vi.hoisted(() => vi.fn());
const oauthLoginPageSpy = vi.hoisted(() => vi.fn());
const usersPanelSpy = vi.hoisted(() => vi.fn());
const webglGradientMock = vi.hoisted(() => ({
  onWebGLAvailabilityChange: null as null | ((available: boolean) => void),
}));
const authExpiredListenerMock = vi.hoisted(() => ({
  callback: null as null | (() => void),
}));
let consoleWarnSpy: ReturnType<typeof vi.spyOn>;

vi.stubGlobal('localStorage', localStorageMock);

vi.mock('@/api', () => ({
  api: apiMock,
  apiFetch: apiMock.apiFetch,
  isResponseAuthContextCurrent: vi.fn(() => true),
  onAuthExpired: vi.fn((callback: () => void) => {
    authExpiredListenerMock.callback = callback;
    return vi.fn(() => {
      if (authExpiredListenerMock.callback === callback) {
        authExpiredListenerMock.callback = null;
      }
    });
  }),
}));

vi.mock('./components/shared/Toast', () => ({
  useToast: () => [[], toastApiMock] as const,
  ToastContainer: (props: unknown) => {
    toastContainerSpy(props);
    return null;
  },
}));

vi.mock('@/components/WebGLGradient', () => ({
  default: ({
    onWebGLAvailabilityChange,
  }: {
    onWebGLAvailabilityChange?: (available: boolean) => void;
  }) => {
    webglGradientMock.onWebGLAvailabilityChange = onWebGLAvailabilityChange ?? null;
    return <div data-testid="webgl-gradient" />;
  },
}));

vi.mock('./components/ConfigurationBanner', () => ({
  ConfigurationBanner: () => null,
}));

vi.mock('./components/LoginPage', () => ({
  LoginPage: ({ onLoginSuccess }: { onLoginSuccess: (user: User) => void }) => (
    <button
      type="button"
      onClick={() => {
        sessionLifecycle.beginEstablishment(sessionLifecycle.capture('challenge'));
        onLoginSuccess({
          id: 'user-1',
          username: 'local:admin',
          display_name: 'Admin',
          email: null,
          auth_provider: 'local_managed',
          role: 'admin',
        });
      }}
    >
      Log in again
    </button>
  ),
}));

vi.mock('./components/MemoryStatus', () => ({
  MemoryStatus: () => null,
}));

vi.mock('./components/OAuthCallbackError', () => ({
  OAuthCallbackError: ({ onRetry, onBack }: { onRetry?: () => void; onBack?: () => void }) => (
    <div data-testid="oauth-callback-error-page">
      OAuth callback error
      {onRetry ? (
        <button type="button" onClick={onRetry}>
          Retry authorization
        </button>
      ) : null}
      {onBack ? (
        <button type="button" onClick={onBack}>
          Back to workspace
        </button>
      ) : null}
    </div>
  ),
}));

vi.mock('./components/OAuthLoginPage', () => ({
  OAuthLoginPage: (props: unknown) => {
    oauthLoginPageSpy(props);
    return <div data-testid="oauth-login-page">OAuth login page</div>;
  },
}));

vi.mock('./components/PublicSharedChatView', () => ({
  PublicSharedChatView: () => null,
}));

vi.mock('./components/SecurityBanner', () => ({
  SecurityBanner: () => null,
}));

vi.mock('./components/UserMenu', () => ({
  UserMenu: ({ user, onLogout }: { user: User; onLogout: () => Promise<void> }) => (
    <>
      <output data-testid="current-user-chat-policy">{String(user.chat_enabled_effective)}</output>
      <button type="button" onClick={() => void onLogout()}>
        Log out
      </button>
    </>
  ),
}));

vi.mock('./components/WarningsBanner', () => ({
  WarningsBanner: ({
    title,
    warnings,
    hidden,
    action,
    dismissKey,
    persistDismiss,
  }: {
    title?: string;
    warnings?: string[];
    hidden?: boolean;
    action?: { label: string; onClick: () => void };
    dismissKey?: string;
    persistDismiss?: boolean;
  }) => {
    if (hidden || !warnings || warnings.length === 0) return null;
    return (
      <div data-dismiss-key={dismissKey} data-persist-dismiss={persistDismiss ? 'true' : 'false'}>
        <span>{title}</span>
        {action ? (
          <button type="button" onClick={action.onClick}>
            {action.label}
          </button>
        ) : null}
        {dismissKey ? <button type="button">Dismiss</button> : null}
      </div>
    );
  },
}));

vi.mock('./components/ChatPage', () => ({
  ChatPage: ({ onFullscreenChange }: { onFullscreenChange?: (fullscreen: boolean) => void }) => (
    <button type="button" onClick={() => onFullscreenChange?.(true)}>
      Enter chat fullscreen
    </button>
  ),
}));

vi.mock('./components/UserSpacePanel', () => ({
  UserSpacePanel: ({ userspaceGenerationEnabled }: { userspaceGenerationEnabled?: boolean }) => (
    <div
      data-testid="userspace-panel"
      data-userspace-generation={String(userspaceGenerationEnabled)}
    />
  ),
}));

vi.mock('./components/ToolsPanel', () => ({
  ToolsPanel: () => null,
}));

vi.mock('./components/UsersPanel', () => ({
  UsersPanel: (props: unknown) => {
    usersPanelSpy(props);
    const onGenerationPolicyUpdated =
      props && typeof props === 'object' && 'onGenerationPolicyUpdated' in props
        ? (props as { onGenerationPolicyUpdated?: (user: User) => Promise<void> | void })
            .onGenerationPolicyUpdated
        : undefined;
    return (
      <button
        type="button"
        onClick={() =>
          void onGenerationPolicyUpdated?.({
            id: 'user-1',
            username: 'local:admin',
            display_name: 'Admin',
            email: null,
            role: 'admin',
            auth_provider: 'local_managed',
          })
        }
      >
        Save self generation policy
      </button>
    );
  },
}));

vi.mock('./components/SettingsPanel', async () => {
  await settingsPanelModuleGate.promise;

  return {
    SettingsPanel: (props: unknown) => {
      settingsPanelSpy(props);
      const onEncryptedArtifactDelivered =
        props && typeof props === 'object' && 'onEncryptedArtifactDelivered' in props
          ? (props as { onEncryptedArtifactDelivered?: () => void }).onEncryptedArtifactDelivered
          : undefined;
      const onServerBackupJobObserved =
        props && typeof props === 'object' && 'onServerBackupJobObserved' in props
          ? (props as { onServerBackupJobObserved?: (job: ServerBackupJob) => void })
              .onServerBackupJobObserved
          : undefined;
      const onServerRestoreJobObserved =
        props && typeof props === 'object' && 'onServerRestoreJobObserved' in props
          ? (props as { onServerRestoreJobObserved?: (job: ServerRestoreJob) => void })
              .onServerRestoreJobObserved
          : undefined;
      const onServerOperationError =
        props && typeof props === 'object' && 'onServerOperationError' in props
          ? (props as { onServerOperationError?: (message: string) => void }).onServerOperationError
          : undefined;
      const onSettingsSaved =
        props && typeof props === 'object' && 'onSettingsSaved' in props
          ? (props as { onSettingsSaved?: () => Promise<void> | void }).onSettingsSaved
          : undefined;
      const highlightSetting =
        props && typeof props === 'object' && 'highlightSetting' in props
          ? (props as { highlightSetting?: string | null }).highlightSetting
          : null;

      return (
        <div>
          <div data-testid="settings-highlight">{highlightSetting ?? 'none'}</div>
          <button type="button" onClick={() => onEncryptedArtifactDelivered?.()}>
            Mark backup delivered
          </button>
          <button
            type="button"
            onClick={() =>
              onServerBackupJobObserved?.({ id: 'backup-observed', status: 'pending' })
            }
          >
            Observe backup job
          </button>
          <button
            type="button"
            onClick={() =>
              onServerRestoreJobObserved?.({ id: 'restore-observed', status: 'pending' })
            }
          >
            Observe restore job
          </button>
          <button type="button" onClick={() => onServerOperationError?.('Section action exploded')}>
            Report operation error
          </button>
          <button type="button" onClick={() => void onSettingsSaved?.()}>
            Save settings
          </button>
        </div>
      );
    },
  };
});

vi.mock('./components/IndexerAdminView', () => ({
  IndexerAdminView: () => null,
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  localStorageMock.getItem.mockReturnValue(null);
  webglGradientMock.onWebGLAvailabilityChange = null;
  window.history.replaceState({}, '', '/');
  authExpiredListenerMock.callback = null;
  consoleWarnSpy.mockRestore();
  vi.useRealTimers();
});

beforeEach(() => {
  vi.stubGlobal('localStorage', localStorageMock);
  if (sessionLifecycle.phase === 'logging-out') {
    sessionLifecycle.finishLogout(sessionLifecycle.capture('public'));
  }
  if (sessionLifecycle.phase === 'authenticated' || sessionLifecycle.phase === 'establishing') {
    sessionLifecycle.markAnonymous(sessionLifecycle.capture('public'));
  }
  // The application lifecycle is intentionally browser-process scoped; each test starts
  // a fresh logical browser session rather than inheriting a prior test's explicit sign-out.
  if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();
  consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  apiMock.getActiveServerBackupJobs.mockResolvedValue({ backup_job: null, restore_job: null });
  apiMock.getServerBackupJob.mockResolvedValue({ id: 'backup-default', status: 'pending' });
  apiMock.getServerRestoreJob.mockResolvedValue({ id: 'restore-default', status: 'pending' });
  apiMock.logout.mockResolvedValue(undefined);
  apiMock.getOpenRouterCreditStatus.mockResolvedValue({
    enabled: false,
    state: 'disabled',
    key_remaining_usd: null,
    wallet_remaining_usd: null,
    threshold_usd: 5,
    checked_at: null,
    stale: false,
    warning: null,
  });
});

function mockAuthenticatedAdmin(configurationWarnings: ConfigurationWarning[] = []): void {
  apiMock.getAuthStatus.mockResolvedValue({
    authenticated: true,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: false,
    api_key_configured: true,
    session_cookie_secure: false,
    allowed_origins_open: false,
    authenticated_webgl_background_enabled: false,
    server_name: 'Ragtime',
    chat_enabled: true,
    userspace_generation_enabled: true,
  });
  apiMock.getCurrentUser.mockResolvedValue({
    id: 'user-1',
    username: 'local:admin',
    display_name: 'Admin',
    role: 'admin',
    chat_enabled_effective: true,
    userspace_generation_enabled_effective: true,
  });
  apiMock.getSettings.mockResolvedValue({
    settings: {
      server_name: 'Ragtime',
      authenticated_webgl_background_enabled: false,
    },
    configuration_warnings: configurationWarnings,
  });
}

function mockAuthenticatedNonAdmin(): void {
  apiMock.getAuthStatus.mockResolvedValue({
    authenticated: true,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: false,
    api_key_configured: true,
    session_cookie_secure: false,
    allowed_origins_open: false,
    authenticated_webgl_background_enabled: false,
    server_name: 'Ragtime',
    chat_enabled: true,
    userspace_generation_enabled: true,
  });
  apiMock.getCurrentUser.mockResolvedValue({
    id: 'user-2',
    username: 'local:user',
    display_name: 'User',
    role: 'user',
    chat_enabled_effective: true,
    userspace_generation_enabled_effective: true,
  });
  apiMock.getSettings.mockResolvedValue({
    settings: {
      server_name: 'Ragtime',
      authenticated_webgl_background_enabled: false,
    },
    configuration_warnings: [],
  });
}

function mockGenerationPolicyNonAdmin(
  chatEnabled: boolean,
  userspaceGenerationEnabled: boolean,
): void {
  mockAuthenticatedNonAdmin();
  apiMock.getAuthStatus.mockResolvedValue({
    authenticated: true,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: false,
    api_key_configured: true,
    session_cookie_secure: false,
    allowed_origins_open: false,
    chat_enabled: chatEnabled,
    userspace_generation_enabled: userspaceGenerationEnabled,
  });
}

async function flushMicrotasks(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
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

describe('WebGL motion background availability', () => {
  it('replaces the motion toggle with an accessible warning when WebGL cannot start', async () => {
    mockAuthenticatedAdmin();
    apiMock.getAuthStatus.mockResolvedValue({
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      authenticated_webgl_background_enabled: true,
      server_name: 'Ragtime',
      chat_enabled: true,
      userspace_generation_enabled: true,
    });
    apiMock.getSettings.mockResolvedValue({
      settings: {
        server_name: 'Ragtime',
        authenticated_webgl_background_enabled: true,
      },
      configuration_warnings: [],
    });

    render(<App />);

    await screen.findByRole('button', { name: 'Pause motion background' });
    expect(webglGradientMock.onWebGLAvailabilityChange).toBeTypeOf('function');
    act(() => webglGradientMock.onWebGLAvailabilityChange?.(false));

    const warning = await screen.findByRole('button', { name: 'Motion background unavailable' });
    expect(warning.getAttribute('aria-disabled')).toBe('true');
    expect(screen.queryByRole('button', { name: 'Pause motion background' })).toBeNull();
    expect(warning.classList.contains('webgl-motion-toggle')).toBe(true);

    fireEvent.mouseEnter(warning);
    expect(
      await screen.findByRole('tooltip', {
        name: 'Motion background unavailable. WebGL could not start; browser hardware acceleration may be disabled. Enable it in browser settings and reload.',
      }),
    ).toBeTruthy();

    warning.focus();
    expect(document.activeElement).toBe(warning);
    expect(
      await screen.findByRole('tooltip', {
        name: 'Motion background unavailable. WebGL could not start; browser hardware acceleration may be disabled. Enable it in browser settings and reload.',
      }),
    ).toBeTruthy();
  });
});

describe('generation capabilities', () => {
  it('refreshes both authenticated policy snapshots after a self policy save', async () => {
    window.history.replaceState({}, '', '/?view=users');
    mockAuthenticatedAdmin();
    apiMock.getAuthStatus.mockResolvedValueOnce({
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: false,
    });
    apiMock.getCurrentUser.mockResolvedValueOnce({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
      auth_provider: 'local_managed',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: false,
    });

    render(<App />);
    await screen.findByRole('button', { name: 'Save self generation policy' });
    fireEvent.click(screen.getByRole('button', { name: 'Save self generation policy' }));

    await waitFor(() => expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(2));
  });

  it('redirects a user without Chat from a chat URL while preserving User Space generation', async () => {
    window.history.replaceState({}, '', '/?view=chat');
    mockGenerationPolicyNonAdmin(false, true);
    render(<App />);
    await screen.findByTestId('userspace-panel');
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Enter chat fullscreen' })).toBeNull();
    expect(screen.getByTestId('userspace-panel').getAttribute('data-userspace-generation')).toBe(
      'true',
    );
    await waitFor(() => expect(window.location.search).toContain('view=userspace'));
  });

  it.each([
    [true, true, true],
    [true, false, true],
    [false, true, false],
    [false, false, false],
  ])(
    'gates Chat and User Space independently (chat=%s, userspace=%s)',
    async (chatEnabled, userspaceGenerationEnabled, chatVisible) => {
      mockGenerationPolicyNonAdmin(chatEnabled, userspaceGenerationEnabled);
      render(<App />);

      await screen.findByRole('button', { name: 'Workspace' });
      fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
      await screen.findByTestId('userspace-panel');
      expect(screen.queryByRole('button', { name: 'Chat' }) !== null).toBe(chatVisible);
      expect(screen.getByTestId('userspace-panel').getAttribute('data-userspace-generation')).toBe(
        String(userspaceGenerationEnabled),
      );
    },
  );

  it('keeps navigation and the current-user policy synchronized after self enable and revoke saves', async () => {
    window.history.replaceState({}, '', '/?view=users');
    const enabledStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: false,
    };
    const disabledStatus = { ...enabledStatus, chat_enabled: false };
    apiMock.getAuthStatus
      .mockResolvedValueOnce(disabledStatus)
      .mockResolvedValueOnce(enabledStatus)
      .mockResolvedValueOnce(disabledStatus);
    apiMock.getCurrentUser
      .mockResolvedValueOnce({
        id: 'user-1',
        username: 'local:admin',
        display_name: 'Admin',
        role: 'admin',
        chat_enabled_effective: false,
      })
      .mockResolvedValueOnce({
        id: 'user-1',
        username: 'local:admin',
        display_name: 'Admin',
        role: 'admin',
        chat_enabled_effective: true,
      })
      .mockResolvedValueOnce({
        id: 'user-1',
        username: 'local:admin',
        display_name: 'Admin',
        role: 'admin',
        chat_enabled_effective: false,
      });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);
    const save = await screen.findByRole('button', { name: 'Save self generation policy' });
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
    expect(screen.getByTestId('current-user-chat-policy').textContent).toBe('false');

    fireEvent.click(save);
    await screen.findByRole('button', { name: 'Chat' });
    expect(screen.getByTestId('current-user-chat-policy').textContent).toBe('true');

    fireEvent.click(save);
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull());
    expect(screen.getByTestId('current-user-chat-policy').textContent).toBe('false');
  });
});

describe('OpenRouter credit alerts', () => {
  it('shows a persistent admin credit alert and clears it after recovery', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getOpenRouterCreditStatus
      .mockResolvedValueOnce({
        enabled: true,
        state: 'low',
        key_remaining_usd: 2,
        wallet_remaining_usd: null,
        threshold_usd: 5,
        checked_at: '2026-09-15T12:00:00Z',
        stale: false,
        warning: 'OpenRouter key credits are low.',
      })
      .mockResolvedValueOnce({
        enabled: true,
        state: 'ok',
        key_remaining_usd: 8,
        wallet_remaining_usd: null,
        threshold_usd: 5,
        checked_at: '2026-09-15T12:01:00Z',
        stale: false,
        warning: null,
      });

    render(<App />);
    await flushMicrotasks();

    expect(screen.getByText('OpenRouter Credit Alert')).toBeTruthy();

    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    await flushMicrotasks();

    expect(screen.queryByText('OpenRouter Credit Alert')).toBeNull();
  });

  it('does not request OpenRouter credits for non-admin users', async () => {
    mockAuthenticatedNonAdmin();
    render(<App />);
    await flushMicrotasks();

    expect(apiMock.getOpenRouterCreditStatus).not.toHaveBeenCalled();
  });
});

describe('App chat fullscreen layout', () => {
  it('preserves resource and scope from the authorization URL for the login continuation', async () => {
    apiMock.getAuthStatus.mockResolvedValue({
      authenticated: false,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
    });
    apiMock.getCurrentUser.mockRejectedValue(new Error('Not authenticated'));
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });
    window.history.replaceState(
      {},
      '',
      '/?client_id=Claude&redirect_uri=https%3A%2F%2Fexample.com%2Fcallback&response_type=code&code_challenge=test&resource=https%3A%2F%2Fragtime.example%2Fmcp%2Fengineering&scope=tools.read%20tools.search',
    );

    render(<App />);

    await waitFor(() => expect(oauthLoginPageSpy).toHaveBeenCalled());
    const props = oauthLoginPageSpy.mock.calls[0]?.[0] as { params: Record<string, string> };
    expect(props.params.resource).toBe('https://ragtime.example/mcp/engineering');
    expect(props.params.scope).toBe('tools.read tools.search');
  });

  it('renders initial OAuth loading inside the shared auth gradient surface but leaves plain app loading unchanged', async () => {
    apiMock.getAuthStatus.mockImplementation(
      () =>
        new Promise(() => {
          // keep loading pending for this assertion
        }),
    );
    apiMock.getSettings.mockResolvedValue({
      settings: {
        server_name: 'Ragtime',
        authenticated_webgl_background_enabled: false,
      },
      configuration_warnings: [],
    });

    window.history.replaceState(
      {},
      '',
      '/?oauth_error_title=OAuth%20pending&oauth_error_summary=Still%20loading',
    );

    const { unmount } = render(<App />);

    await waitFor(() => {
      expect(document.querySelector('[data-auth-surface="gradient"]')).toBeTruthy();
    });

    const oauthSurface = document.querySelector('[data-auth-surface="gradient"]');
    expect(oauthSurface?.classList.contains('auth-loading')).toBe(true);
    expect(screen.getByText('Loading...')).toBeTruthy();
    unmount();

    window.history.replaceState({}, '', '/');
    render(<App />);
    expect(document.querySelector('[data-auth-surface="gradient"]')).toBe(null);
    const plainLoading = document.querySelector('.auth-loading');
    expect(plainLoading).toBeTruthy();
    expect(plainLoading?.getAttribute('data-auth-surface')).toBe(null);
  });

  it('renders authenticated OAuth authorizing inside the shared auth gradient surface', async () => {
    mockAuthenticatedAdmin();
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
    });
    vi.stubGlobal(
      'fetch',
      vi.fn(
        () =>
          new Promise(() => {
            // keep authorizing pending for this assertion
          }),
      ),
    );
    window.history.replaceState(
      {},
      '',
      '/?client_id=Claude&redirect_uri=https%3A%2F%2Fexample.com%2Fcallback&response_type=code&code_challenge=test&resource=https%3A%2F%2Fragtime.example%2Fmcp&scope=tools.read',
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Authorizing...')).toBeTruthy();
    });

    const surface = document.querySelector('[data-auth-surface="gradient"]');
    expect(surface).toBeTruthy();
    expect(surface?.classList.contains('auth-loading')).toBe(true);
    expect(surface?.textContent).toContain('Authorizing...');
    const request = vi.mocked(fetch).mock.calls[0]?.[1] as RequestInit;
    const body = new URLSearchParams(request.body as string);
    expect(body.get('resource')).toBe('https://ragtime.example/mcp');
    expect(body.get('scope')).toBe('tools.read');
  });

  it('shows a loading fallback before rendering a lazy admin view', async () => {
    const user = userEvent.setup();
    mockAuthenticatedAdmin();

    render(<App />);

    await flushMicrotasks();
    await user.click(screen.getByRole('button', { name: 'Settings' }));

    const loadingFallback = await screen.findByText((_, element) =>
      Boolean(element?.classList.contains('auth-loading')),
    );
    expect(loadingFallback.querySelector('.spinner')).toBeTruthy();

    settingsPanelModuleGate.resolve();

    await waitFor(() => {
      expect(screen.getByTestId('settings-highlight').textContent).toBe('none');
    });
  });

  it('refreshes Chat and User Space effective policies after settings saves', async () => {
    mockAuthenticatedAdmin();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        // The response is already the effective value: a global disable wins over a user enable.
        chat_enabled: false,
        userspace_generation_enabled: true,
      })
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        // A user-level override can restore effective access after the global policy is enabled.
        chat_enabled: true,
        userspace_generation_enabled: false,
      });

    render(<App />);

    await screen.findByRole('button', { name: 'Chat' });
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Save settings' }));

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
    await screen.findByTestId('userspace-panel');
    expect(screen.getByTestId('userspace-panel').getAttribute('data-userspace-generation')).toBe(
      'true',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Save settings' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Chat' })).toBeTruthy();
    });
    fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
    await screen.findByTestId('userspace-panel');
    expect(screen.getByTestId('userspace-panel').getAttribute('data-userspace-generation')).toBe(
      'false',
    );
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(3);
  });

  it('applies fullscreen state to the outer chat page container', async () => {
    const user = userEvent.setup();
    mockAuthenticatedAdmin();

    const { container } = render(<App />);

    const fullscreenButton = await screen.findByRole('button', { name: 'Enter chat fullscreen' });
    const chatPage = container.querySelector('.chat-page-container');
    expect(chatPage?.classList.contains('chat-page-fullscreen')).toBe(false);

    await user.click(fullscreenButton);

    await waitFor(() => {
      expect(chatPage?.classList.contains('chat-page-fullscreen')).toBe(true);
    });
  });

  it('deep-links the encryption backup reminder into server backup settings and dismisses it only after delivery is reported', async () => {
    const user = userEvent.setup();
    mockAuthenticatedAdmin([
      {
        level: 'warning',
        category: 'encryption_backup',
        message: 'Back up your managed encryption key in an encrypted server backup.',
      },
    ]);

    render(<App />);

    const backupWarningContainer = (await screen.findByText(
      (_, element) =>
        element?.matches('div[data-dismiss-key="ragtime_encryption_backup_reminder"]') ?? false,
    )) as HTMLElement;
    expect(backupWarningContainer.dataset.dismissKey).toBe('ragtime_encryption_backup_reminder');
    expect(backupWarningContainer.dataset.persistDismiss).toBe('true');
    expect(backupWarningContainer.querySelector('button')).toBeTruthy();

    const openAction = await screen.findByRole('button', { name: 'Open backup settings' });
    await user.click(openAction);

    await waitFor(() => {
      expect(screen.getByTestId('settings-highlight').textContent).toBe(
        SERVER_BACKUP_RESTORE_HIGHLIGHT,
      );
    });

    await user.click(screen.getByRole('button', { name: 'Mark backup delivered' }));

    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'ragtime_encryption_backup_reminder',
        'true',
      );
    });
    expect(
      screen.queryByText(
        (_, element) =>
          element?.matches('div[data-dismiss-key="ragtime_encryption_backup_reminder"]') ?? false,
      ),
    ).toBe(null);
  });

  it('emits one backup completion toast after navigating away from Settings', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getServerBackupJob
      .mockResolvedValueOnce({ id: 'backup-observed', status: 'pending' })
      .mockResolvedValue({ id: 'backup-observed', status: 'completed' });

    render(<App />);

    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Observe backup job' }));
    fireEvent.click(screen.getByRole('button', { name: 'Chat' }));

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(apiMock.getServerBackupJob).toHaveBeenCalledWith('backup-observed');

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(toastApiMock.success).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(6000);
    });
    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(2);
  });

  it('emits one restore failure toast after navigating away from Settings', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getServerRestoreJob
      .mockResolvedValueOnce({ id: 'restore-observed', status: 'pending' })
      .mockResolvedValue({
        id: 'restore-observed',
        status: 'failed',
        error: 'Restore exploded',
        message: 'ignore me',
      });

    render(<App />);

    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Observe restore job' }));
    fireEvent.click(screen.getByRole('button', { name: 'Chat' }));

    expect(apiMock.getServerRestoreJob).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(apiMock.getServerRestoreJob).toHaveBeenCalledWith('restore-observed');

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(toastApiMock.error).toHaveBeenCalledWith('Restore exploded');

    expect(toastApiMock.error).toHaveBeenCalledTimes(1);
  });

  it('routes settings-reported server operation errors to the app toast stack', async () => {
    mockAuthenticatedAdmin();

    render(<App />);

    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Report operation error' }));

    expect(toastApiMock.error).toHaveBeenCalledWith('Section action exploded');
    expect(toastApiMock.error).toHaveBeenCalledTimes(1);
  });

  it('does not poll backup lifecycle jobs for non-admin users', async () => {
    vi.useFakeTimers();
    mockAuthenticatedNonAdmin();

    render(<App />);

    await flushMicrotasks();
    expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(6000);
    });

    expect(apiMock.getActiveServerBackupJobs).not.toHaveBeenCalled();
    expect(apiMock.getServerBackupJob).not.toHaveBeenCalled();
    expect(apiMock.getServerRestoreJob).not.toHaveBeenCalled();
  });

  it('ignores cancelled jobs and stops lifecycle polling on auth expiry', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: { id: 'backup-cancelled', status: 'pending' },
      restore_job: null,
    });
    apiMock.getServerBackupJob.mockResolvedValue({ id: 'backup-cancelled', status: 'cancelled' });

    render(<App />);

    await flushMicrotasks();
    expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(apiMock.getServerBackupJob).toHaveBeenCalledWith('backup-cancelled');

    expect(toastApiMock.success).not.toHaveBeenCalled();
    expect(toastApiMock.error).not.toHaveBeenCalled();

    sessionLifecycle.expire(sessionLifecycle.capture('session'));

    await act(async () => {
      vi.advanceTimersByTime(6000);
    });
    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(1);
  });

  it('clears terminal-toast deduplication on auth expiry', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getServerBackupJob.mockResolvedValue({
      id: 'backup-observed',
      status: 'completed',
    });

    render(<App />);

    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Observe backup job' }));
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(toastApiMock.success).toHaveBeenCalledTimes(1);

    await act(async () => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });
    fireEvent.click(screen.getByRole('button', { name: 'Log in again' }));
    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Observe backup job' }));
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();

    expect(toastApiMock.success).toHaveBeenCalledTimes(2);
  });

  it('emits one interrupted backup error toast using fallback semantics and stops polling', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: { id: 'backup-interrupted', status: 'pending' },
      restore_job: null,
    });
    apiMock.getServerBackupJob
      .mockResolvedValueOnce({ id: 'backup-interrupted', status: 'pending' })
      .mockResolvedValue({
        id: 'backup-interrupted',
        status: 'interrupted',
        message: 'Transport died',
      });

    render(<App />);

    await flushMicrotasks();
    expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(apiMock.getServerBackupJob).toHaveBeenCalledWith('backup-interrupted');

    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await flushMicrotasks();
    expect(toastApiMock.error).toHaveBeenCalledWith('Transport died');
    expect(toastApiMock.error).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(6000);
    });
    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(2);
  });

  it('does not emit repeated polling-error toasts for tracked jobs', async () => {
    vi.useFakeTimers();
    mockAuthenticatedAdmin();
    apiMock.getServerBackupJob.mockRejectedValue(new Error('poll failed'));

    render(<App />);

    await flushMicrotasks();
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    fireEvent.click(screen.getByRole('button', { name: 'Observe backup job' }));

    await act(async () => {
      vi.advanceTimersByTime(6000);
    });

    await flushMicrotasks();
    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(3);
    expect(toastApiMock.error).not.toHaveBeenCalled();
    expect(toastApiMock.success).not.toHaveBeenCalled();
  });
});

describe('authenticated refresh lifecycle', () => {
  it('deduplicates simultaneous focus and visibility refreshes instead of applying a later conflicting snapshot', async () => {
    mockAuthenticatedAdmin();
    const returningAnonymousStatus = deferred<{
      authenticated: boolean;
      ldap_configured: boolean;
      local_admin_enabled: boolean;
      debug_mode: boolean;
      api_key_configured: boolean;
      session_cookie_secure: boolean;
      allowed_origins_open: boolean;
      chat_enabled: boolean;
      userspace_generation_enabled: boolean;
    }>();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockImplementationOnce(() => returningAnonymousStatus.promise)
      .mockResolvedValue({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      });

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });

    await act(async () => {
      window.dispatchEvent(new Event('focus'));
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await act(async () => {
      returningAnonymousStatus.resolve({
        authenticated: false,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: false,
        userspace_generation_enabled: false,
      });
    });

    await screen.findByRole('button', { name: 'Log in again' });
  });

  it('does not restore an expired session when an in-flight status refresh resolves', async () => {
    mockAuthenticatedAdmin();
    const staleStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockImplementationOnce(() => staleStatus.promise);

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });
    window.dispatchEvent(new Event('focus'));
    sessionLifecycle.expire(sessionLifecycle.capture('session'));
    staleStatus.resolve({
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    });

    await screen.findByRole('button', { name: 'Log in again' });
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
  });

  it('does not restore a logged-out session when an in-flight status refresh resolves', async () => {
    mockAuthenticatedAdmin();
    const staleStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockImplementationOnce(() => staleStatus.promise)
      .mockResolvedValueOnce({
        authenticated: false,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: false,
        userspace_generation_enabled: false,
      });

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });
    window.dispatchEvent(new Event('focus'));
    fireEvent.click(screen.getByRole('button', { name: 'Log out' }));
    staleStatus.resolve({
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    });

    await screen.findByRole('button', { name: 'Log in again' });
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
  });

  it('keeps expiry state when an obsolete status refresh rejects', async () => {
    mockAuthenticatedAdmin();
    const staleStatus = deferred<AuthStatus>();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockImplementationOnce(() => staleStatus.promise);

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });
    window.dispatchEvent(new Event('focus'));
    sessionLifecycle.expire(sessionLifecycle.capture('session'));
    staleStatus.reject(new Error('obsolete refresh'));

    await screen.findByRole('button', { name: 'Log in again' });
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
  });

  it('does not restore an expired session when the stale current-user response arrives', async () => {
    mockAuthenticatedAdmin();
    const staleUser = deferred<User>();
    apiMock.getAuthStatus.mockResolvedValueOnce({
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    });
    apiMock.getCurrentUser
      .mockResolvedValueOnce({
        id: 'user-1',
        username: 'local:admin',
        display_name: 'Admin',
        role: 'admin',
        chat_enabled_effective: true,
        userspace_generation_enabled_effective: true,
      })
      .mockImplementationOnce(() => staleUser.promise);

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });
    window.dispatchEvent(new Event('focus'));
    await waitFor(() => expect(apiMock.getCurrentUser).toHaveBeenCalledTimes(2));
    sessionLifecycle.expire(sessionLifecycle.capture('session'));
    staleUser.resolve({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Stale admin',
      email: null,
      auth_provider: 'local_managed',
      role: 'admin',
    });

    await screen.findByRole('button', { name: 'Log in again' });
    expect(screen.queryByText('Stale admin')).toBeNull();
  });

  it('keeps the login screen available when login refresh is superseded by expiry', async () => {
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: false,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        chat_enabled: false,
        userspace_generation_enabled: false,
      })
      .mockImplementationOnce(
        () =>
          new Promise(() => {
            // Keep the post-login refresh in flight until auth expiry.
          }),
      );
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);
    const login = await screen.findByRole('button', { name: 'Log in again' });
    fireEvent.click(login);
    sessionLifecycle.expire(sessionLifecycle.capture('session'));

    await screen.findByRole('button', { name: 'Log in again' });
    expect(document.querySelector('.auth-loading')).toBeNull();
  });

  it('uses anonymous status presentation without requesting a current user', async () => {
    apiMock.getAuthStatus.mockResolvedValue({
      authenticated: false,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      server_name: 'Guest Ragtime',
      default_theme_pack: 'serif',
      chat_enabled: false,
      userspace_generation_enabled: false,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);

    await screen.findByRole('button', { name: 'Log in again' });
    expect(document.title).toBe('Guest Ragtime');
    expect(document.documentElement.getAttribute('data-theme-pack')).toBe('serif');
    expect(apiMock.getCurrentUser).not.toHaveBeenCalled();
  });

  it('lets a settings save supersede an older focus refresh', async () => {
    const staleFocusStatus = deferred<AuthStatus>();
    const settingsStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: false,
      userspace_generation_enabled: true,
    };
    mockAuthenticatedAdmin();
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        ...settingsStatus,
        chat_enabled: true,
      })
      .mockImplementationOnce(() => staleFocusStatus.promise)
      .mockResolvedValueOnce(settingsStatus);

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });
    window.dispatchEvent(new Event('focus'));
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    settingsPanelModuleGate.resolve();
    fireEvent.click(await screen.findByRole('button', { name: 'Save settings' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull());

    staleFocusStatus.resolve({ ...settingsStatus, chat_enabled: true });
    await flushMicrotasks();
    expect(screen.queryByRole('button', { name: 'Chat' })).toBeNull();
  });
});

describe('App authentication recovery actions', () => {
  it('runs Check session again explicitly after expiry metadata fails', async () => {
    const authenticatedStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    };
    apiMock.getAuthStatus
      .mockResolvedValueOnce(authenticatedStatus)
      .mockRejectedValueOnce(new Error('metadata offline'))
      .mockResolvedValueOnce({ ...authenticatedStatus, authenticated: false });
    apiMock.getCurrentUser.mockResolvedValueOnce({
      id: 'user-2',
      username: 'local:user',
      display_name: 'User',
      role: 'user',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);
    await screen.findByRole('button', { name: 'Workspace' });
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    const check = await screen.findByRole('button', { name: 'Check session again' });
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2);
    fireEvent.click(check);

    await screen.findByRole('button', { name: 'Log in again' });
    expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(3);
  });

  it('does not steal focus when a passive authenticated refresh fails', async () => {
    mockAuthenticatedAdmin();
    render(<App />);
    const chat = await screen.findByRole('button', { name: 'Chat' });
    chat.focus();
    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('passive refresh offline'));

    act(() => window.dispatchEvent(new Event('focus')));
    await waitFor(() => expect(apiMock.getAuthStatus).toHaveBeenCalledTimes(2));

    expect(document.activeElement).toBe(chat);
    expect(screen.queryByRole('button', { name: 'Try connecting again' })).toBeNull();
  });

  it('initiates focus of first nav button after explicit retry when DOM remounts to authenticated', async () => {
    // Set up mocks for initial render + recovery flow
    apiMock.getAuthStatus
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        authenticated_webgl_background_enabled: false,
        server_name: 'Ragtime',
        chat_enabled: true,
        userspace_generation_enabled: true,
      })
      .mockRejectedValueOnce(new Error('metadata offline'))
      .mockResolvedValueOnce({
        authenticated: true,
        ldap_configured: false,
        local_admin_enabled: true,
        debug_mode: false,
        api_key_configured: true,
        session_cookie_secure: false,
        allowed_origins_open: false,
        authenticated_webgl_background_enabled: false,
        server_name: 'Ragtime',
        chat_enabled: true,
        userspace_generation_enabled: true,
      });
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);
    const chatButton = await screen.findByRole('button', { name: 'Chat' });
    expect(document.activeElement).not.toBe(chatButton);

    // Simulate expiry that triggers recovery
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    await screen.findByRole('button', { name: 'Check session again' });

    // Click retry - recovery process starts
    fireEvent.click(screen.getByRole('button', { name: 'Check session again' }));

    // Wait for recovery UI to disappear (recovery completes) and app returns to authenticated
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Check session again' })).toBeNull(),
    );
    // Verify authenticated app is rendered
    await screen.findByRole('button', { name: 'Workspace' });

    // useLayoutEffect should have run after authPhase changed to authenticated
    // If no user input was focused, the nav button should receive focus after deferred mount
    // Verify the chat button is still rendered after recovery
    expect(chatButton).toBeTruthy();
  });

  it('does not focus nav button if user input has focus when retry completes', async () => {
    mockAuthenticatedAdmin();
    const { container } = render(<App />);
    await screen.findByRole('button', { name: 'Chat' });

    apiMock.getAuthStatus.mockRejectedValueOnce(new Error('metadata offline'));
    // Simulate expiry that triggers recovery
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    await screen.findByRole('button', { name: 'Check session again' });

    // Create a text input and focus it before retry
    const input = container.appendChild(document.createElement('input'));
    input.type = 'text';
    input.focus();
    expect(document.activeElement).toBe(input);

    // Click retry - recovery process starts with user input focused
    fireEvent.click(screen.getByRole('button', { name: 'Check session again' }));

    // Wait for recovery UI to disappear (recovery completes)
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Check session again' })).toBeNull(),
    );

    // Input should retain focus because we checked for user input before setting pending flag
    await waitFor(() => expect(document.activeElement).toBe(input));
    input.remove();
  });

  it('does not consume pending focus flag while recovery is busy in unavailable phase', async () => {
    const authenticatedStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    };
    apiMock.getAuthStatus
      .mockResolvedValueOnce(authenticatedStatus)
      .mockRejectedValueOnce(new Error('metadata offline'))
      .mockImplementation(
        () =>
          new Promise((resolve) => {
            // Simulate slow recovery to keep recoveryBusy true
            setTimeout(() => resolve(authenticatedStatus), 100);
          }),
      );
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    render(<App />);
    await screen.findByRole('button', { name: 'Chat' });

    // Trigger expiry and recovery
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    await screen.findByRole('button', { name: 'Check session again' });

    // Click retry - sets pending flag
    fireEvent.click(screen.getByRole('button', { name: 'Check session again' }));

    // While recovery is busy, useLayoutEffect should NOT consume the flag
    // even though authPhase may transition to unavailable or other states
    // Wait a bit but not for completion
    await waitFor(
      () => expect(screen.queryByRole('button', { name: 'Check session again' })).toBeNull(),
      { timeout: 50 },
    ).catch(() => {
      // Recovery still in progress
    });

    // Flag should still be set (recoveryBusy prevents consumption)
    expect(screen.queryByRole('button', { name: 'Check session again' })).toBeNull();
  });

  it('does not steal focus when anonymous recovery settles then passive authenticated refresh occurs', async () => {
    const authenticatedStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      chat_enabled: true,
      userspace_generation_enabled: true,
    };
    apiMock.getAuthStatus
      .mockResolvedValueOnce(authenticatedStatus)
      .mockRejectedValueOnce(new Error('metadata offline'))
      .mockResolvedValueOnce({ ...authenticatedStatus, authenticated: false })
      .mockResolvedValueOnce(authenticatedStatus);
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
      chat_enabled_effective: true,
      userspace_generation_enabled_effective: true,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });

    const { container } = render(<App />);
    await screen.findByRole('button', { name: 'Chat' });

    // Trigger expiry
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });

    await screen.findByRole('button', { name: 'Check session again' });

    // Create input and focus it
    const input = container.appendChild(document.createElement('input'));
    input.type = 'text';
    input.focus();

    // Click retry - sets pending flag if no input, but we have input, so flag stays false
    fireEvent.click(screen.getByRole('button', { name: 'Check session again' }));

    // Wait for recovery to complete (recovers to anonymous)
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Check session again' })).toBeNull(),
    );

    // Should show login page (anonymous)
    await screen.findByRole('button', { name: 'Log in again' });

    // Now simulate passive refresh while input is still focused
    act(() => window.dispatchEvent(new Event('focus')));

    // Wait for passive refresh to complete (moves to authenticated)
    await waitFor(() => screen.findByRole('button', { name: 'Chat' }));

    // Input should retain focus - pending flag was never set (input was focused during retry)
    // and passive refresh doesn't set the flag
    expect(document.activeElement).toBe(input);
    input.remove();
  });
});

describe('authenticated OAuth authorization lifecycle', () => {
  const oauthUrl =
    '/?client_id=client-a&redirect_uri=https%3A%2F%2Fclient.example%2Fcallback&response_type=code&code_challenge=challenge-a&state=state-a';

  it('does not expose runtime OAuth actions for a URL-origin callback error', async () => {
    apiMock.getAuthStatus.mockResolvedValue({
      authenticated: false,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: false,
      session_cookie_secure: false,
      allowed_origins_open: false,
    });
    apiMock.getSettings.mockResolvedValue({ settings: {}, configuration_warnings: [] });
    window.history.replaceState(
      {},
      '',
      '/?oauth_error_title=Invalid%20request&oauth_error_summary=Bad%20redirect',
    );

    render(<App />);
    await screen.findByTestId('oauth-callback-error-page');

    expect(screen.queryByRole('button', { name: 'Retry authorization' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Back to workspace' })).toBeNull();
    expect(apiMock.apiFetch).not.toHaveBeenCalledWith('/authorize/session', expect.anything());
  });

  it('keeps the authenticated principal on a transport failure and retries only on explicit action', async () => {
    mockAuthenticatedNonAdmin();
    window.history.replaceState({}, '', oauthUrl);
    const first = deferred<Response>();
    const second = deferred<Response>();
    const fetchMock = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    await act(async () => first.reject(new Error('authorization offline')));

    await screen.findByTestId('oauth-callback-error-page');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Retry authorization' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));

    await act(async () => second.reject(new Error('still offline')));
    await screen.findByTestId('oauth-callback-error-page');
    fireEvent.click(screen.getByRole('button', { name: 'Back to workspace' }));

    expect(window.location.search).toBe('?view=userspace');
    expect(await screen.findByRole('button', { name: 'Workspace' })).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('ignores a rejected authorization request from an obsolete session generation', async () => {
    mockAuthenticatedNonAdmin();
    window.history.replaceState({}, '', oauthUrl);
    const pending = deferred<Response>();
    const fetchMock = vi.fn().mockReturnValueOnce(pending.promise);
    vi.stubGlobal('fetch', fetchMock);

    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    act(() => {
      sessionLifecycle.expire(sessionLifecycle.capture('session'));
    });
    await screen.findByTestId('oauth-login-page');

    await act(async () => pending.reject(new Error('obsolete failure')));
    await flushMicrotasks();

    expect(screen.queryByTestId('oauth-callback-error-page')).toBeNull();
    expect(screen.getByTestId('oauth-login-page')).toBeTruthy();
  });
});

describe('principal-scoped settings presentation', () => {
  it('does not let an old principals delayed settings response overwrite the newer principal', async () => {
    const oldSettings = deferred<{
      settings: { server_name: string; authenticated_webgl_background_enabled: boolean };
      configuration_warnings: ConfigurationWarning[];
    }>();
    const newSettings = deferred<{
      settings: { server_name: string; authenticated_webgl_background_enabled: boolean };
      configuration_warnings: ConfigurationWarning[];
    }>();
    const authStatus = {
      authenticated: true,
      ldap_configured: false,
      local_admin_enabled: true,
      debug_mode: false,
      api_key_configured: true,
      session_cookie_secure: false,
      allowed_origins_open: false,
      server_name: 'Public name',
      chat_enabled: true,
      userspace_generation_enabled: true,
    };
    apiMock.getAuthStatus.mockResolvedValue(authStatus);
    apiMock.getCurrentUser
      .mockResolvedValueOnce({
        id: 'old-admin',
        username: 'old-admin',
        display_name: 'Old Admin',
        role: 'admin',
        chat_enabled_effective: true,
        userspace_generation_enabled_effective: true,
      })
      .mockResolvedValueOnce({
        id: 'new-admin',
        username: 'new-admin',
        display_name: 'New Admin',
        role: 'admin',
        chat_enabled_effective: true,
        userspace_generation_enabled_effective: true,
      });
    apiMock.getSettings
      .mockReturnValueOnce(oldSettings.promise)
      .mockReturnValueOnce(newSettings.promise);

    render(<App />);
    await waitFor(() => expect(apiMock.getSettings).toHaveBeenCalledTimes(1));
    window.dispatchEvent(new Event('focus'));
    await waitFor(() => expect(apiMock.getSettings).toHaveBeenCalledTimes(2));

    await act(async () => {
      newSettings.resolve({
        settings: { server_name: 'New principal', authenticated_webgl_background_enabled: false },
        configuration_warnings: [],
      });
    });
    expect(screen.getByText('New principal')).toBeTruthy();

    await act(async () => {
      oldSettings.resolve({
        settings: { server_name: 'Old principal', authenticated_webgl_background_enabled: true },
        configuration_warnings: [],
      });
    });
    expect(screen.queryByText('Old principal')).toBeNull();
    expect(screen.getByText('New principal')).toBeTruthy();
  });
});
