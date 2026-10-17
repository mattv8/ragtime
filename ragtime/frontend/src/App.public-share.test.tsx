import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

    render(<App />);

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

    render(<App />);

    const title = await screen.findByRole('heading', { name: SHARED_TITLE });
    const shell = title.closest('#public-shared-route') as HTMLElement;
    expect(shell).not.toBeNull();
    expect(within(shell).getByText(SHARED_MESSAGE)).toBeTruthy();
    expect(within(shell).getByRole('button', { name: 'Sign In' })).toBeTruthy();
    expect(screen.queryByRole('textbox')).toBeNull();
    expect(screen.queryByRole('button', { name: /Viewer/ })).toBeNull();
    await waitFor(() => expect(apiMock.getSharedConversation).toHaveBeenCalled());
    const modelRequests = fetchMock.mock.calls.filter(
      ([url]) => String(url) === AVAILABLE_MODELS_PATH,
    );
    expect(modelRequests).toHaveLength(0);
  });
});
