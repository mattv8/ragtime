import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { StrictMode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import { sessionLifecycle } from './auth/sessionLifecycle';
import type { AuthStatus, SharedConversationResponse, User } from './types';

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
  getOpenRouterCreditStatus: vi.fn(),
  getMountHealth: vi.fn(),
  recheckMountHealth: vi.fn(),
  logout: vi.fn(),
  resolvePublicShareTarget: vi.fn(),
  getSharedConversation: vi.fn(),
  getSharedConversationEventsUrl: vi.fn(),
  getModelPreferences: vi.fn(),
  listUserSpaceWorkspaces: vi.fn(),
  apiFetch: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: apiMock,
  apiFetch: apiMock.apiFetch,
  isResponseAuthContextCurrent: vi.fn(() => true),
  onAuthExpired: vi.fn(() => vi.fn()),
}));

vi.mock('@/components/WebGLGradient', () => ({
  default: () => null,
}));

const SHARED_TITLE = 'Quarterly plan';
const SHARED_MESSAGE = 'Summarize the Q3 roadmap risks';
const AVAILABLE_MODELS_PATH = '/indexes/chat/available-models';

const anonymousAuthStatus = {
  authenticated: false,
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
} satisfies AuthStatus;

const viewer = {
  id: 'user-viewer',
  username: 'local:viewer',
  display_name: 'Viewer',
  email: null,
  role: 'user',
  auth_provider: 'local_managed',
  chat_enabled_effective: true,
  userspace_generation_enabled_effective: false,
} satisfies User;

/** Token-mode, read-only shared conversation. `can_edit` false means no composer renders. */
function sharedConversationFor(isAuthenticated: boolean) {
  return {
    conversation: {
      id: 'conv-shared',
      title: SHARED_TITLE,
      model: 'test-model',
      messages: [{ role: 'user', content: SHARED_MESSAGE, timestamp: '2026-01-01T00:00:00Z' }],
      total_tokens: 0,
      active_task_id: null,
      tool_output_mode: 'default',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
      read_only: true,
    },
    owner_username: 'local:owner',
    owner_display_name: 'Owner',
    share_access_mode: 'token',
    granted_role: 'viewer',
    can_edit: false,
    is_authenticated: isAuthenticated,
  } satisfies SharedConversationResponse;
}

function mockSharedConversationApi(isAuthenticated: boolean): void {
  apiMock.resolvePublicShareTarget.mockResolvedValue({ target_type: 'conversation' });
  apiMock.getSharedConversation.mockResolvedValue(sharedConversationFor(isAuthenticated));
  apiMock.getSharedConversationEventsUrl.mockReturnValue('/events-url-for-test');
}

function mockAuthenticatedViewer(): void {
  apiMock.getAuthStatus.mockResolvedValue({ ...anonymousAuthStatus, authenticated: true });
  apiMock.getCurrentUser.mockResolvedValue(viewer);
  apiMock.getSettings.mockResolvedValue({
    settings: { server_name: 'Ragtime', authenticated_webgl_background_enabled: false },
    configuration_warnings: [],
  });
  mockSharedConversationApi(true);
}

async function renderAppWithSharedViewReady(): Promise<void> {
  await import('./components/PublicSharedChatView');
  render(<App />);
}

function mockAnonymousVisitor(): void {
  apiMock.getAuthStatus.mockResolvedValue(anonymousAuthStatus);
  apiMock.getCurrentUser.mockRejectedValue(new Error('Not signed in'));
  apiMock.getSettings.mockResolvedValue({
    settings: { server_name: 'Ragtime', authenticated_webgl_background_enabled: false },
    configuration_warnings: [],
  });
  mockSharedConversationApi(false);
}

// jsdom does not implement scrollIntoView; the shared chat auto-scrolls after rendering messages.
const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;

class FakeEventSource {
  addEventListener(): void {}
  removeEventListener(): void {}
  close(): void {}
}

describe('App public share route provider boundary', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    HTMLElement.prototype.scrollIntoView = vi.fn();
    if (sessionLifecycle.phase === 'logging-out') {
      sessionLifecycle.finishLogout(sessionLifecycle.capture('public'));
    }
    if (sessionLifecycle.phase === 'authenticated' || sessionLifecycle.phase === 'establishing') {
      sessionLifecycle.markAnonymous(sessionLifecycle.capture('public'));
    }
    if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();

    window.history.replaceState({}, '', '/shared/test-token');
    vi.stubGlobal('localStorage', localStorageMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    fetchMock = vi.fn(async () => new Response(JSON.stringify({ models: [] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    apiMock.getActiveServerBackupJobs.mockResolvedValue({ backup_job: null, restore_job: null });
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
    apiMock.getMountHealth.mockResolvedValue({
      status: 'ok',
      checked_at: null,
      runtime_checked: true,
      problems: [],
    });
    apiMock.logout.mockResolvedValue(undefined);
    apiMock.getModelPreferences.mockResolvedValue({
      user_default_chat_model: null,
      global_default_chat_model: 'global-test-model',
      effective_default_chat_model: 'global-test-model',
    });
    apiMock.listUserSpaceWorkspaces.mockResolvedValue({
      items: [],
      total: 0,
      offset: 0,
      limit: 50,
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
    window.history.replaceState({}, '', '/');
  });

  it('renders an authenticated shared conversation with its user menu and preferences modal', async () => {
    mockAuthenticatedViewer();
    const user = userEvent.setup();

    await renderAppWithSharedViewReady();

    await screen.findByRole('heading', { name: SHARED_TITLE });
    expect(screen.getByText(SHARED_MESSAGE)).toBeTruthy();
    expect(screen.queryByRole('textbox')).toBeNull();

    await user.click(await screen.findByRole('button', { name: /Viewer/ }));
    await user.click(screen.getByRole('button', { name: /Preferences/ }));

    expect(await screen.findByRole('heading', { name: 'Preferences' })).toBeTruthy();
    expect(apiMock.getModelPreferences).toHaveBeenCalled();
  });

  it('keeps an anonymous shared conversation without a composer and offers sign-in', async () => {
    mockAnonymousVisitor();

    await renderAppWithSharedViewReady();

    const title = await screen.findByRole('heading', { name: SHARED_TITLE });
    const shell = title.closest('#public-shared-route') as HTMLElement;
    expect(shell).not.toBeNull();
    expect(within(shell).getByText(SHARED_MESSAGE)).toBeTruthy();
    expect(
      document
        .querySelector('#public-shared-chat-messages')
        ?.getAttribute('data-shared-chat-state'),
    ).toBe('conversation');
    expect(document.querySelector('#public-shared-topnav')).not.toBeNull();
    expect(document.querySelector('#public-shared-chat-header')).not.toBeNull();
    expect(within(shell).getByRole('button', { name: 'Sign In' })).toBeTruthy();
    expect(screen.queryByRole('textbox')).toBeNull();
    expect(screen.queryByRole('button', { name: /Viewer/ })).toBeNull();
    await waitFor(() => expect(apiMock.getSharedConversation).toHaveBeenCalled());
    const modelRequests = fetchMock.mock.calls.filter(
      ([url]) => String(url) === AVAILABLE_MODELS_PATH,
    );
    expect(modelRequests).toHaveLength(0);
  });

  it('traps and restores sign-in focus under StrictMode and supports dismissal', async () => {
    mockAnonymousVisitor();
    const user = userEvent.setup();
    await import('./components/PublicSharedChatView');
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    const signIn = await screen.findByRole('button', { name: 'Sign In' });
    await user.click(signIn);
    const dialog = await screen.findByRole('dialog', { name: 'Sign in' });
    expect(dialog.id).toBe('public-shared-login-dialog');
    const username = within(dialog).getByLabelText(/username/i);
    await waitFor(() => expect(document.activeElement).toBe(username));
    const cancel = within(dialog).getByRole('button', { name: 'Cancel' });
    cancel.focus();
    await user.tab();
    expect(document.activeElement).toBe(username);
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog', { name: 'Sign in' })).toBeNull();
    expect(document.activeElement).toBe(signIn);
    await user.click(signIn);
    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(screen.queryByRole('dialog', { name: 'Sign in' })).toBeNull();
    await user.click(signIn);
    await user.click(document.querySelector('#public-shared-login-overlay') as HTMLElement);
    expect(screen.queryByRole('dialog', { name: 'Sign in' })).toBeNull();
  });

  it('submits a share password on Enter and displays an invalid-password error', async () => {
    mockAnonymousVisitor();
    apiMock.getSharedConversation
      .mockReset()
      .mockRejectedValueOnce(new Error('Password required'))
      .mockRejectedValue(new Error('Invalid password'));
    const user = userEvent.setup();
    await renderAppWithSharedViewReady();
    const password = await screen.findByLabelText('Share password');
    expect(screen.queryByRole('alert')).toBeNull();
    await user.type(password, 'secret{Enter}');
    await waitFor(() =>
      expect(apiMock.getSharedConversation).toHaveBeenCalledWith('test-token', 'secret'),
    );
    expect((await screen.findByRole('alert')).textContent).toContain('Invalid password');
    expect(screen.getByLabelText('Share password').getAttribute('aria-invalid')).toBe('true');
    expect(
      apiMock.getSharedConversation.mock.calls.filter(([, password]) => password === 'secret'),
    ).toHaveLength(1);
  });

  it('retries the same share password with one new request', async () => {
    mockAnonymousVisitor();
    apiMock.getSharedConversation
      .mockReset()
      .mockRejectedValueOnce(new Error('Password required'))
      .mockRejectedValue(new Error('Invalid password'));
    const user = userEvent.setup();
    await renderAppWithSharedViewReady();
    await user.type(await screen.findByLabelText('Share password'), 'wrong{Enter}');
    expect((await screen.findByRole('alert')).textContent).toContain('Invalid password');
    expect(
      apiMock.getSharedConversation.mock.calls.filter(([, password]) => password === 'wrong'),
    ).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: 'Unlock' }));
    await waitFor(() =>
      expect(
        apiMock.getSharedConversation.mock.calls.filter(([, password]) => password === 'wrong'),
      ).toHaveLength(2),
    );
    expect((await screen.findByRole('alert')).textContent).toContain('Invalid password');
  });

  it('renders an unavailable message for a failed shared conversation', async () => {
    mockAnonymousVisitor();
    apiMock.getSharedConversation.mockRejectedValue(new Error('Unexpected error'));
    await renderAppWithSharedViewReady();
    expect(await screen.findByText('This shared chat could not be loaded.')).toBeTruthy();
    expect(screen.getAllByText('Unexpected error')).toHaveLength(1);
    expect(screen.getByRole('alert').textContent).toBe('Unexpected error');
    expect(document.querySelector('.chat-shared-owner-col')).toBeNull();
    expect(
      document
        .querySelector('#public-shared-chat-messages')
        ?.getAttribute('data-shared-chat-state'),
    ).toBe('unavailable');
  });
});
