import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from './components/shared/securityWarnings';

const localStorageMock = vi.hoisted(() => ({
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
}));

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  getSettings: vi.fn(),
}));

const settingsPanelSpy = vi.hoisted(() => vi.fn());

vi.stubGlobal('localStorage', localStorageMock);

vi.mock('@/api', () => ({
  api: apiMock,
  onAuthExpired: vi.fn(() => vi.fn()),
}));

vi.mock('@/components/WebGLGradient', () => ({
  default: () => <div data-testid="webgl-gradient" />,
}));

vi.mock('@/components', () => ({
  ChatPage: ({ onFullscreenChange }: { onFullscreenChange?: (fullscreen: boolean) => void }) => (
    <button type="button" onClick={() => onFullscreenChange?.(true)}>
      Enter chat fullscreen
    </button>
  ),
  ConfigurationBanner: () => null,
  FilesystemIndexPanel: () => null,
  IndexesList: () => null,
  JobsTable: () => null,
  LoginPage: () => null,
  MemoryStatus: () => null,
  OAuthCallbackError: () => null,
  OAuthLoginPage: () => null,
  PublicSharedChatView: () => null,
  SecurityBanner: () => null,
  SettingsPanel: (props: unknown) => {
    settingsPanelSpy(props);
    const onEncryptedArtifactDelivered =
      props && typeof props === 'object' && 'onEncryptedArtifactDelivered' in props
        ? (props as { onEncryptedArtifactDelivered?: () => void }).onEncryptedArtifactDelivered
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
      </div>
    );
  },
  ToolsPanel: () => null,
  UserMenu: () => null,
  UserSpacePanel: () => null,
  UsersPanel: () => null,
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

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  localStorageMock.getItem.mockReturnValue(null);
  window.history.replaceState({}, '', '/');
});

describe('App chat fullscreen layout', () => {
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

    const backupWarning = await screen.findByText('Back Up Your Encryption Key');
    const backupWarningContainer = backupWarning.closest('div[data-dismiss-key]') as HTMLElement;
    expect(backupWarningContainer.dataset.dismissKey).toBe('ragtime_encryption_backup_reminder');
    expect(backupWarningContainer.dataset.persistDismiss).toBe('true');
    expect(screen.getByRole('button', { name: 'Dismiss' })).toBeTruthy();

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
    expect(screen.queryByText('Back Up Your Encryption Key')).toBe(null);
  });
});
