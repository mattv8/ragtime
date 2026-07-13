import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';

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
  SettingsPanel: () => null,
  ToolsPanel: () => null,
  UserMenu: () => null,
  UserSpacePanel: () => null,
  UsersPanel: () => null,
  WarningsBanner: () => null,
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
});
