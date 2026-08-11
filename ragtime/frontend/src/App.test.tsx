import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from './components/shared/securityWarnings';
import type { ServerBackupJob, ServerRestoreJob, User } from './types';

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
  logout: vi.fn(),
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
const authExpiredListenerMock = vi.hoisted(() => ({
  callback: null as null | (() => void),
}));
let consoleWarnSpy: ReturnType<typeof vi.spyOn>;

vi.stubGlobal('localStorage', localStorageMock);

vi.mock('@/api', () => ({
  api: apiMock,
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
  default: () => <div data-testid="webgl-gradient" />,
}));

vi.mock('./components/ConfigurationBanner', () => ({
  ConfigurationBanner: () => null,
}));

vi.mock('./components/LoginPage', () => ({
  LoginPage: ({ onLoginSuccess }: { onLoginSuccess: (user: User) => void }) => (
    <button
      type="button"
      onClick={() =>
        onLoginSuccess({
          id: 'user-1',
          username: 'local:admin',
          display_name: 'Admin',
          email: null,
          auth_provider: 'local_managed',
          role: 'admin',
        })
      }
    >
      Log in again
    </button>
  ),
}));

vi.mock('./components/MemoryStatus', () => ({
  MemoryStatus: () => null,
}));

vi.mock('./components/OAuthCallbackError', () => ({
  OAuthCallbackError: () => null,
}));

vi.mock('./components/OAuthLoginPage', () => ({
  OAuthLoginPage: () => null,
}));

vi.mock('./components/PublicSharedChatView', () => ({
  PublicSharedChatView: () => null,
}));

vi.mock('./components/SecurityBanner', () => ({
  SecurityBanner: () => null,
}));

vi.mock('./components/UserMenu', () => ({
  UserMenu: () => null,
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
  UserSpacePanel: () => null,
}));

vi.mock('./components/ToolsPanel', () => ({
  ToolsPanel: () => null,
}));

vi.mock('./components/UsersPanel', () => ({
  UsersPanel: () => null,
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
  localStorageMock.getItem.mockReturnValue(null);
  window.history.replaceState({}, '', '/');
  authExpiredListenerMock.callback = null;
  consoleWarnSpy.mockRestore();
  vi.useRealTimers();
});

beforeEach(() => {
  consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  apiMock.getActiveServerBackupJobs.mockResolvedValue({ backup_job: null, restore_job: null });
  apiMock.getServerBackupJob.mockResolvedValue({ id: 'backup-default', status: 'pending' });
  apiMock.getServerRestoreJob.mockResolvedValue({ id: 'restore-default', status: 'pending' });
  apiMock.logout.mockResolvedValue(undefined);
});

function mockAuthenticatedAdmin(): void {
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
  });
  apiMock.getCurrentUser.mockResolvedValue({
    id: 'user-1',
    username: 'local:admin',
    display_name: 'Admin',
    role: 'admin',
  });
  apiMock.getSettings.mockResolvedValue({
    settings: {
      server_name: 'Ragtime',
      authenticated_webgl_background_enabled: false,
    },
    configuration_warnings: [],
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
  });
  apiMock.getCurrentUser.mockResolvedValue({
    id: 'user-2',
    username: 'local:user',
    display_name: 'User',
    role: 'user',
  });
  apiMock.getSettings.mockResolvedValue({
    settings: {
      server_name: 'Ragtime',
      authenticated_webgl_background_enabled: false,
    },
    configuration_warnings: [],
  });
}

async function flushMicrotasks(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe('App chat fullscreen layout', () => {
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

  it('applies fullscreen state to the outer chat page container', async () => {
    const user = userEvent.setup();
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
    });
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
    });
    apiMock.getSettings.mockResolvedValue({
      settings: {
        server_name: 'Ragtime',
        authenticated_webgl_background_enabled: false,
      },
      configuration_warnings: [],
    });

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
    });
    apiMock.getCurrentUser.mockResolvedValue({
      id: 'user-1',
      username: 'local:admin',
      display_name: 'Admin',
      role: 'admin',
    });
    apiMock.getSettings.mockResolvedValue({
      settings: {
        server_name: 'Ragtime',
        authenticated_webgl_background_enabled: false,
      },
      configuration_warnings: [
        {
          level: 'warning',
          category: 'encryption_backup',
          message: 'Back up your managed encryption key in an encrypted server backup.',
        },
      ],
    });

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

    authExpiredListenerMock.callback?.();

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
      authExpiredListenerMock.callback?.();
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
