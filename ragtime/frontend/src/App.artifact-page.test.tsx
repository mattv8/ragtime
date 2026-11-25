import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { App } from './App';
import { sessionLifecycle } from './auth/sessionLifecycle';
import type { AuthStatus, User } from './types';

const ARTIFACT_ID = 'a1b2c3d4'.repeat(4);

const apiMock = vi.hoisted(() => ({
  getAuthStatus: vi.fn(),
  getCurrentUser: vi.fn(),
  getSettings: vi.fn(),
  getActiveServerBackupJobs: vi.fn(),
  getOpenRouterCreditStatus: vi.fn(),
  getMountHealth: vi.fn(),
  logout: vi.fn(),
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

// The page has its own tests; here only the routing decision and the id it receives matter.
vi.mock('./components/ModelArtifactPage', () => ({
  ModelArtifactPage: ({ artifactId }: { artifactId: string }) => (
    <div data-testid="artifact-page">{artifactId}</div>
  ),
}));

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

function mockAnonymous(): void {
  apiMock.getAuthStatus.mockResolvedValue(anonymousAuthStatus);
  apiMock.getCurrentUser.mockRejectedValue(new Error('Not signed in'));
}

function mockSignedIn(): void {
  apiMock.getAuthStatus.mockResolvedValue({ ...anonymousAuthStatus, authenticated: true });
  apiMock.getCurrentUser.mockResolvedValue(viewer);
}

beforeEach(() => {
  if (sessionLifecycle.phase === 'logging-out') {
    sessionLifecycle.finishLogout(sessionLifecycle.capture('public'));
  }
  if (sessionLifecycle.phase === 'authenticated' || sessionLifecycle.phase === 'establishing') {
    sessionLifecycle.markAnonymous(sessionLifecycle.capture('public'));
  }
  if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();

  vi.stubGlobal('localStorage', { getItem: () => null, setItem: vi.fn(), removeItem: vi.fn() });
  apiMock.getSettings.mockResolvedValue({
    settings: { server_name: 'Ragtime', authenticated_webgl_background_enabled: false },
    configuration_warnings: [],
  });
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
  apiMock.logout.mockResolvedValue(undefined);
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('{}', { status: 200 })),
  );
  window.history.replaceState({}, '', `/?artifact=${ARTIFACT_ID}`);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState({}, '', '/');
});

describe('App standalone model artifact route', () => {
  it('keeps ?artifact= in the URL for a signed-out user and shows LoginPage there', async () => {
    mockAnonymous();
    render(<App />);

    expect(await screen.findByLabelText(/username/i)).toBeTruthy();
    expect(screen.queryByTestId('artifact-page')).toBeNull();
    expect(window.location.pathname).toBe('/');
    expect(window.location.search).toBe(`?artifact=${ARTIFACT_ID}`);
  });

  it('renders the artifact page for a signed-in user without rewriting the URL', async () => {
    mockSignedIn();
    render(<App />);

    expect(await screen.findByTestId('artifact-page')).toBeTruthy();
    expect(screen.getByTestId('artifact-page').textContent).toBe(ARTIFACT_ID);
    await waitFor(() => expect(window.location.search).toBe(`?artifact=${ARTIFACT_ID}`));
  });

  it('passes an invalid artifact value to the page so it can show not-found', async () => {
    mockSignedIn();
    window.history.replaceState({}, '', '/?artifact=not-a-real-id');
    render(<App />);

    expect(await screen.findByTestId('artifact-page')).toBeTruthy();
    expect(screen.getByTestId('artifact-page').textContent).toBe('not-a-real-id');
    expect(window.location.search).toBe('?artifact=not-a-real-id');
  });
});
