import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@xterm/xterm', () => ({ Terminal: class {} }));
vi.mock('@xterm/addon-fit', () => ({ FitAddon: class {} }));
vi.mock('@uiw/react-codemirror', () => ({
  default: () => <div data-testid="code-editor" />,
}));

const {
  previewApiMock,
  panelToastMock,
  availableModelsMock,
  diffHoverTimersMock,
  workspaceChatSearchMock,
  workspaceScmActivityMock,
} = vi.hoisted(() => ({
  previewApiMock: {
    listUserSpaceWorkspaces: vi.fn(),
    getUserSpaceWorkspace: vi.fn(),
    getWorkspacesConversationStateSummaryLite: vi.fn(),
    getUserSpaceWorkspaceTabState: vi.fn(),
    listUserSpaceFiles: vi.fn(),
    getUserSpaceFile: vi.fn(),
    getUserSpaceChangedFileState: vi.fn(),
    getUserSpaceSnapshotTimeline: vi.fn(),
    getUserSpaceWorkspaceCollabPresence: vi.fn(),
    listUserSpaceSqliteDatabases: vi.fn(),
    listUserSpaceAvailableTools: vi.fn(),
    listUserSpaceToolGroups: vi.fn(),
    listWorkspaceMounts: vi.fn(),
    getUserSpaceCollabWebSocketUrl: vi.fn(),
    authorizeUserSpaceBrowserSurfaces: vi.fn(),
    launchUserSpacePreview: vi.fn(),
    subscribeWorkspaceEvents: vi.fn(),
    refreshUserSpaceBridgeCredentials: vi.fn(),
    listUsersDirectory: vi.fn(),
    getLdapConfig: vi.fn(),
    discoverLdapWithStoredCredentials: vi.fn(),
    listUserSpaceWorkspaceShareLinks: vi.fn(),
    deleteUserSpaceWorkspaceShareLink: vi.fn(),
    subscribeUserSpaceWorkspaceShareLinkAnalytics: vi.fn(),
  },
  panelToastMock: [[], { error: vi.fn(), success: vi.fn(), info: vi.fn() }],
  availableModelsMock: {
    refresh: vi.fn(),
    awaitReady: vi.fn().mockResolvedValue(undefined),
  },
  diffHoverTimersMock: {
    registerHoverTarget: vi.fn(),
    clearHoverTarget: vi.fn(),
    dismiss: vi.fn(),
  },
  workspaceChatSearchMock: {
    query: '',
    setQuery: vi.fn(),
    results: [],
    activeIndex: -1,
    setActiveIndex: vi.fn(),
    clear: vi.fn(),
  },
  workspaceScmActivityMock: { hasActivity: false, syncState: null },
}));

let latestSqliteInspectorModalProps: unknown = null;
let sqliteInspectorModalRender: (props: unknown) => unknown = () => null;

vi.mock('@/api', () => ({ api: previewApiMock, ApiError: class ApiError extends Error {} }));
vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => availableModelsMock,
}));
vi.mock('@/utils/codemirrorLanguage', () => ({
  useCodeMirrorLanguageExtension: () => null,
}));
vi.mock('@/utils/useUserSpaceToolHealthEvents', () => ({
  useUserSpaceToolHealthEvents: () => undefined,
}));
vi.mock('@/utils/useDiffHoverTimers', () => ({
  useDiffHoverTimers: () => diffHoverTimersMock,
}));
vi.mock('@/utils/useWorkspaceChatSearch', () => ({
  useWorkspaceChatSearch: () => workspaceChatSearchMock,
}));
vi.mock('./ChatPanel', () => ({
  ChatPanel: () => <div data-testid="chat-panel" />,
}));
vi.mock('./ConstrainedPathBrowser', () => ({
  ConstrainedPathBrowser: () => <div data-testid="path-browser" />,
}));
vi.mock('./shared/FileDiffOverlay', () => ({
  FileDiffOverlay: () => null,
}));
vi.mock('./shared/Toast', () => ({
  useToast: () => panelToastMock,
  ToastContainer: () => null,
}));
vi.mock('./shared/UserSpaceEnvVarsModal', () => ({
  UserSpaceEnvVarsModal: () => null,
}));
vi.mock('./shared/WorkspaceSqliteInspectorModal', () => ({
  WorkspaceSqliteInspectorModal: (props: unknown) => {
    latestSqliteInspectorModalProps = props;
    return sqliteInspectorModalRender(props);
  },
}));
vi.mock('./shared/WorkspaceObjectStorageExplorer', () => ({
  WorkspaceObjectStorageExplorer: () => null,
}));
vi.mock('./shared/ShareLinkModal', () => ({
  ShareLinkModal: ({
    isOpen,
    shareLinks,
    onDeleteSelectedShareLink,
  }: {
    isOpen: boolean;
    shareLinks: Array<{ id: string; label: string | null }>;
    onDeleteSelectedShareLink: (shareId: string) => void;
  }) =>
    isOpen ? (
      <div data-testid="share-link-modal">
        {shareLinks.map((link) => (
          <div key={link.id}>{link.label}</div>
        ))}
        <button
          type="button"
          onClick={() => {
            if (shareLinks[0]) {
              onDeleteSelectedShareLink(shareLinks[0].id);
            }
          }}
        >
          Delete first share
        </button>
      </div>
    ) : null,
}));
vi.mock('./WorkspaceScmWizard', () => ({
  useWorkspaceScmWizardActivity: () => workspaceScmActivityMock,
  WorkspaceScmWizard: ({ workspace }: { workspace?: { sqlite_persistence_mode?: string } }) => (
    <div data-testid="workspace-scm-mode">{workspace?.sqlite_persistence_mode ?? 'missing'}</div>
  ),
}));
vi.mock('./shared/AdminWorkspaceModal', () => ({ default: () => null }));
vi.mock('./shared/AgentAccessButton', () => ({ AgentAccessButton: () => null }));
vi.mock('./shared/AgentAccessModal', () => ({ AgentAccessModal: () => null }));
vi.mock('./shared/MemberManagementButton', () => ({ MemberManagementButton: () => null }));
vi.mock('./shared/MemberManagementModal', () => ({ MemberManagementModal: () => null }));
vi.mock('./shared/MiniLoadingSpinner', () => ({
  MiniLoadingSpinner: ({ title }: { title?: string }) => <span>{title ?? 'loading'}</span>,
}));
vi.mock('./shared/ToolSelectorDropdown', () => ({
  ToolSelectorDropdown: () => <div data-testid="tool-selector" />,
}));
vi.mock('./Popover', () => ({
  Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  DisabledPopover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock('./UserSpaceArtifactPreview', () => ({
  UserSpaceArtifactPreview: () => <div data-testid="preview-frame" />,
}));

import {
  UserSpacePanel,
  getWorkspaceToolReadOnlyDescription,
  getWorkspaceToolStatusBadgeForState,
} from './UserSpacePanel';

const CURRENT_USER = {
  id: 'user-1',
  username: 'ada',
  display_name: 'Ada',
  email: 'ada@example.com',
  role: 'admin',
  auth_provider: 'local',
} as const;

const WORKSPACE = {
  id: 'ws-1',
  name: 'Workspace One',
  sqlite_persistence_mode: 'exclude',
  owner_user_id: 'user-1',
  selected_tool_ids: [],
  selected_tool_group_ids: [],
  conversation_ids: [],
  members: [],
  created_at: '2026-07-14T00:00:00Z',
  updated_at: '2026-07-14T00:00:00Z',
} as const;

const STARTING_RUNTIME_STATUS = {
  workspace_id: 'ws-1',
  session_state: 'running',
  devserver_running: false,
  devserver_port: 4173,
  last_error: null,
  live_data_warning: null,
  bridge_status: {
    state: 'healthy',
    bridge_url: 'https://bridge.example',
    token_session_id: 'session-1',
    current_session_id: 'session-1',
    issued_at: '2099-08-05T18:00:00Z',
    expires_at: '2099-08-05T19:00:00Z',
    last_success_at: '2099-08-05T18:30:00Z',
    detail: null,
  },
} as const;

const DEFAULT_CHAT_STATE = {
  conversation_id: null,
  has_live: false,
  has_interrupted: false,
};

const SHARE_LINKS_RESPONSE = {
  owner_username: 'ada',
  links: [
    {
      id: 'share-1',
      workspace_id: 'ws-1',
      has_share_link: true,
      owner_username: 'ada',
      label: 'Share one',
      share_slug: 'share-one',
      share_token: 'token-1',
      share_url: 'https://example.test/share-one',
      anonymous_share_url: 'https://example.test/a/share-one',
      subdomain_share_url: null,
      subdomain_share_enabled: false,
      subdomain_share_disabled_reason: null,
      created_at: '2026-07-14T00:00:00Z',
      public_hit_count: 0,
      last_public_hit_at: null,
      share_access_mode: 'token',
      selected_user_ids: [],
      selected_ldap_groups: [],
      has_password: false,
      active_share_style: 'anonymous',
    },
    {
      id: 'share-2',
      workspace_id: 'ws-1',
      has_share_link: true,
      owner_username: 'ada',
      label: 'Share two',
      share_slug: 'share-two',
      share_token: 'token-2',
      share_url: 'https://example.test/share-two',
      anonymous_share_url: 'https://example.test/a/share-two',
      subdomain_share_url: null,
      subdomain_share_enabled: false,
      subdomain_share_disabled_reason: null,
      created_at: '2026-07-14T00:00:00Z',
      public_hit_count: 0,
      last_public_hit_at: null,
      share_access_mode: 'token',
      selected_user_ids: [],
      selected_ldap_groups: [],
      has_password: false,
      active_share_style: 'anonymous',
    },
  ],
} as const;

const SQLITE_LIST_RESPONSE = {
  workspace_id: 'ws-1',
  databases: [
    {
      name: 'app.sqlite3',
      relative_path: '.ragtime/db/app.sqlite3',
      size_bytes: 1024,
      table_count: 0,
      last_modified_ms: 1_786_032_000_000,
      owner_workspace_id: 'ws-1',
      owner_workspace_name: 'Workspace One',
      ownership: 'owned',
      access_mode: 'read_write',
      persistence_mode: 'exclude',
      initialized: true,
    },
  ],
  total_bytes: 1024,
  default_database_name: 'app.sqlite3',
  persistence_mode: 'exclude',
} as const;

function isoAtOffsetFromNow(offsetMs: number): string {
  return new Date(Date.now() + offsetMs).toISOString();
}

function setLayoutCookie(rightPaneCollapsed: boolean): void {
  document.cookie = `userspace_layout_${encodeURIComponent(CURRENT_USER.id)}=${encodeURIComponent(
    JSON.stringify({
      sidebarWidth: 180,
      sidebarCollapsed: false,
      leftPaneFraction: 0.5,
      rightPaneCollapsed,
      editorFraction: 0.6,
      editorChatCollapsedSide: null,
    }),
  )}; path=/`;
}

async function renderPanelWithRuntimeOverlay(rightPaneCollapsed: boolean) {
  setLayoutCookie(rightPaneCollapsed);

  render(<UserSpacePanel currentUser={{ ...CURRENT_USER }} />);

  await waitFor(() => {
    expect(screen.getByText('Starting runtime...')).toBeTruthy();
  });

  const status = screen.getByRole('status');
  return status;
}

beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
      unobserve() {}
    },
  );
  vi.stubGlobal(
    'WebSocket',
    class {
      close() {}
      send() {}
      addEventListener() {}
      removeEventListener() {}
    },
  );
  vi.stubGlobal('localStorage', {
    getItem: vi.fn(() => null),
    setItem: vi.fn(),
    removeItem: vi.fn(),
  });
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    }),
  });
});

beforeEach(() => {
  document.cookie = 'userspace_layout_user-1=; path=/; max-age=0';
  previewApiMock.listUserSpaceWorkspaces.mockResolvedValue({ items: [{ ...WORKSPACE }], total: 1 });
  previewApiMock.getUserSpaceWorkspace.mockResolvedValue({ ...WORKSPACE });
  previewApiMock.getWorkspacesConversationStateSummaryLite.mockResolvedValue([]);
  previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
    workspace_id: WORKSPACE.id,
    runtime_status: { ...STARTING_RUNTIME_STATUS },
    chat_state: { ...DEFAULT_CHAT_STATE },
  });
  previewApiMock.listUserSpaceFiles.mockResolvedValue([
    { path: 'dashboard/main.ts', updated_at: '2026-07-14T00:00:00Z', artifact_type: 'code' },
  ]);
  previewApiMock.getUserSpaceFile.mockResolvedValue({
    path: 'dashboard/main.ts',
    content: 'export const ready = true;\n',
    updated_at: '2026-07-14T00:00:00Z',
    artifact_type: 'code',
  });
  previewApiMock.getUserSpaceChangedFileState.mockResolvedValue({
    changed_paths: [],
    acknowledged_paths: [],
  });
  previewApiMock.getUserSpaceSnapshotTimeline.mockResolvedValue({
    snapshots: [],
    branches: [],
    current_snapshot_id: null,
    current_branch_id: null,
  });
  previewApiMock.getUserSpaceWorkspaceCollabPresence.mockResolvedValue({
    version: 0,
    users: [],
    read_only: false,
  });
  previewApiMock.listUserSpaceSqliteDatabases.mockResolvedValue({
    ...SQLITE_LIST_RESPONSE,
    databases: SQLITE_LIST_RESPONSE.databases.map((database) => ({ ...database })),
  });
  previewApiMock.listUserSpaceAvailableTools.mockResolvedValue([]);
  previewApiMock.listUserSpaceToolGroups.mockResolvedValue([]);
  previewApiMock.listWorkspaceMounts.mockResolvedValue([]);
  previewApiMock.getUserSpaceCollabWebSocketUrl.mockReturnValue('ws://example.test/collab');
  previewApiMock.authorizeUserSpaceBrowserSurfaces.mockResolvedValue({
    authorizations: [
      {
        surface: 'collab',
        expires_at: '2026-07-15T00:00:00Z',
      },
      {
        surface: 'runtime_pty',
        expires_at: '2026-07-15T00:00:00Z',
      },
    ],
  });
  previewApiMock.launchUserSpacePreview.mockResolvedValue({
    workspace_id: WORKSPACE.id,
    preview_url: 'http://preview.test/session',
    preview_origin: 'http://preview.test',
    expires_at: '2026-07-15T00:00:00Z',
    preview_warning: null,
  });
  previewApiMock.subscribeWorkspaceEvents.mockReturnValue({
    close: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    onmessage: null,
    onerror: null,
  });
  previewApiMock.refreshUserSpaceBridgeCredentials.mockResolvedValue({
    ...STARTING_RUNTIME_STATUS.bridge_status,
  });
  previewApiMock.listUsersDirectory.mockResolvedValue([]);
  previewApiMock.getLdapConfig.mockResolvedValue({ discovered_groups: [] });
  previewApiMock.discoverLdapWithStoredCredentials.mockResolvedValue({
    success: true,
    groups: [],
  });
  previewApiMock.listUserSpaceWorkspaceShareLinks.mockResolvedValue({
    owner_username: SHARE_LINKS_RESPONSE.owner_username,
    links: SHARE_LINKS_RESPONSE.links.map((link) => ({ ...link })),
  });
  previewApiMock.deleteUserSpaceWorkspaceShareLink.mockResolvedValue(undefined);
  previewApiMock.subscribeUserSpaceWorkspaceShareLinkAnalytics.mockReturnValue({
    close: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
  latestSqliteInspectorModalProps = null;
  sqliteInspectorModalRender = () => null;
  cleanup();
});

describe('UserSpacePanel workspace tool descriptions', () => {
  it('gives admins an inline Settings > Tools link for read-only tools', async () => {
    const user = userEvent.setup();
    const onNavigateToTools = vi.fn();

    render(
      <div>{getWorkspaceToolReadOnlyDescription(true, 'tool-read-only', onNavigateToTools)}</div>,
    );

    await user.click(screen.getByRole('link', { name: 'Settings > Tools' }));

    expect(onNavigateToTools).toHaveBeenCalledWith('tool:tool-read-only');
  });

  it('does not expose a settings link to non-admins', () => {
    render(<div>{getWorkspaceToolReadOnlyDescription(false, 'tool-read-only')}</div>);

    expect(screen.queryByRole('link', { name: 'Settings > Tools' })).toBeNull();
    expect(screen.getByText(/Ask an admin to enable/)).toBeTruthy();
  });

  it('explains that only workspace owners may manage workspace write access for a globally read-only tool', () => {
    render(
      <div>{getWorkspaceToolReadOnlyDescription(false, 'tool-read-only', undefined, true)}</div>,
    );

    expect(screen.queryByRole('link', { name: 'Settings > Tools' })).toBeNull();
    expect(
      screen.getByText(/This workspace only has Read access to this tool via its access policy/i),
    ).toBeTruthy();
  });

  it('does not render a badge for globally read-only tools', () => {
    expect(getWorkspaceToolStatusBadgeForState('ineligible')).toBeNull();
  });

  it('labels selected workspace tool access explicitly', () => {
    expect(getWorkspaceToolStatusBadgeForState('enabled')).toMatchObject({
      label: 'WORKSPACE WRITE',
      tone: 'warning',
      title: 'Workspace write enabled for this workspace',
    });
    expect(getWorkspaceToolStatusBadgeForState('eligible')).toMatchObject({
      label: 'WORKSPACE READ',
      tone: 'read',
      title: 'Selected for this workspace with read access.',
    });
  });

  it('renders the runtime overlay exactly once inside the preview section when expanded', async () => {
    await renderPanelWithRuntimeOverlay(false);

    const overlays = document.querySelectorAll('.userspace-status-overlay');
    expect(overlays).toHaveLength(1);

    const previewSection = document.querySelector('.userspace-preview-section');
    expect(previewSection).not.toBeNull();
    expect(previewSection?.querySelector('.userspace-status-overlay')).toBe(overlays[0]);
    expect(document.querySelector('.userspace-status-overlay-root')).toBeNull();
  });

  it('moves the runtime overlay to the collapsed-pane root when the preview pane starts collapsed', async () => {
    await renderPanelWithRuntimeOverlay(true);

    const overlays = document.querySelectorAll('.userspace-status-overlay');
    expect(overlays).toHaveLength(1);

    const collapsedRoot = document.querySelector('.userspace-status-overlay-root');
    expect(collapsedRoot).toBe(overlays[0]);

    const previewSection = document.querySelector('.userspace-preview-section');
    expect(previewSection?.querySelector('.userspace-status-overlay')).toBeNull();
  });

  it('removes the deleted share locally without reloading the share list', async () => {
    await renderPanelWithRuntimeOverlay(false);

    const manageShareButton = document.querySelector('[title="Manage share link"]');
    expect(manageShareButton).not.toBeNull();

    (manageShareButton as HTMLButtonElement).click();

    await waitFor(() => {
      const modal = document.querySelector('[data-testid="share-link-modal"]');
      expect(modal?.textContent?.includes('Share one')).toBe(true);
      expect(modal?.textContent?.includes('Share two')).toBe(true);
    });

    const modal = document.querySelector('[data-testid="share-link-modal"]');
    expect(modal).not.toBeNull();

    const deleteFirstShareButton = Array.from(
      (modal as HTMLElement).querySelectorAll('button'),
    ).find((button) => button.textContent === 'Delete first share');
    expect(deleteFirstShareButton).not.toBeNull();

    (deleteFirstShareButton as HTMLButtonElement).click();

    await waitFor(() => {
      const currentModal = document.querySelector('[data-testid="share-link-modal"]');
      expect(currentModal?.textContent?.includes('Share one')).toBe(false);
    });

    expect(
      document
        .querySelector('[data-testid="share-link-modal"]')
        ?.textContent?.includes('Share two'),
    ).toBe(true);
    expect(previewApiMock.deleteUserSpaceWorkspaceShareLink).toHaveBeenCalledWith(
      'ws-1',
      'share-1',
    );
    expect(previewApiMock.listUserSpaceWorkspaceShareLinks).toHaveBeenCalledTimes(1);
  });

  it('restores share links when delete fails without reloading the share list', async () => {
    previewApiMock.deleteUserSpaceWorkspaceShareLink.mockRejectedValueOnce(
      new Error('Delete share failed'),
    );

    await renderPanelWithRuntimeOverlay(false);

    const manageShareButton = document.querySelector('[title="Manage share link"]');
    expect(manageShareButton).not.toBeNull();

    (manageShareButton as HTMLButtonElement).click();

    await waitFor(() => {
      const modal = document.querySelector('[data-testid="share-link-modal"]');
      expect(modal?.textContent?.includes('Share one')).toBe(true);
      expect(modal?.textContent?.includes('Share two')).toBe(true);
    });

    const modal = document.querySelector('[data-testid="share-link-modal"]');
    expect(modal).not.toBeNull();

    const deleteFirstShareButton = Array.from(
      (modal as HTMLElement).querySelectorAll('button'),
    ).find((button) => button.textContent === 'Delete first share');
    expect(deleteFirstShareButton).not.toBeNull();

    (deleteFirstShareButton as HTMLButtonElement).click();

    await waitFor(() => {
      const currentModal = document.querySelector('[data-testid="share-link-modal"]');
      expect(currentModal?.textContent?.includes('Share one')).toBe(true);
      expect(currentModal?.textContent?.includes('Share two')).toBe(true);
    });

    expect(previewApiMock.deleteUserSpaceWorkspaceShareLink).toHaveBeenCalledWith(
      'ws-1',
      'share-1',
    );
    expect(previewApiMock.listUserSpaceWorkspaceShareLinks).toHaveBeenCalledTimes(1);
  });

  it('treats linked databases with tables as activating the SQLite inspector toolbar state', async () => {
    previewApiMock.listUserSpaceSqliteDatabases.mockResolvedValueOnce({
      ...SQLITE_LIST_RESPONSE,
      databases: [
        { ...SQLITE_LIST_RESPONSE.databases[0], table_count: 0 },
        {
          ...SQLITE_LIST_RESPONSE.databases[0],
          owner_workspace_id: 'linked-ws',
          owner_workspace_name: 'Reporting',
          ownership: 'linked',
          access_mode: 'read',
          table_count: 3,
        },
      ],
    });

    await renderPanelWithRuntimeOverlay(false);

    await waitFor(() => {
      const button = screen.getByLabelText('Open SQLite inspector');
      expect(button.getAttribute('title')).toBe('Open SQLite Inspector');
      expect(button.className.includes('btn-primary')).toBe(true);
    });
  });

  it('does not promote the active source workspace when a linked target promotion is reported', async () => {
    const user = userEvent.setup();
    sqliteInspectorModalRender = (props: unknown) => {
      const { isOpen, onPersistencePromoted } = props as {
        isOpen: boolean;
        onPersistencePromoted?: (workspaceId: string) => void;
      };
      return isOpen ? (
        <button type="button" onClick={() => onPersistencePromoted?.('linked-ws')}>
          Trigger linked promotion
        </button>
      ) : null;
    };

    await renderPanelWithRuntimeOverlay(false);

    const openInspectorButton = screen.getByLabelText('Open SQLite inspector');
    await user.click(openInspectorButton);
    await user.click(screen.getByRole('button', { name: 'Trigger linked promotion' }));

    expect(latestSqliteInspectorModalProps).toEqual(
      expect.objectContaining({
        workspaceId: 'ws-1',
        onPersistencePromoted: expect.any(Function),
      }),
    );
    const openInspectorButtonAfter = screen.getByLabelText('Open SQLite inspector');
    expect(openInspectorButtonAfter.className.includes('userspace-sqlite-mode-excluded')).toBe(
      true,
    );
  });

  it('shows unhealthy bridge status details to viewers without a refresh action', async () => {
    const viewerWorkspace = {
      ...WORKSPACE,
      owner_user_id: 'owner-1',
    };
    previewApiMock.listUserSpaceWorkspaces.mockResolvedValue({
      items: [{ ...viewerWorkspace }],
      total: 1,
    });
    previewApiMock.getUserSpaceWorkspace.mockResolvedValue({ ...viewerWorkspace });
    previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        bridge_status: {
          state: 'expired',
          bridge_url: 'https://bridge.example',
          token_session_id: 'session-old',
          current_session_id: 'session-new',
          issued_at: '2026-08-05T18:00:00Z',
          expires_at: '2026-08-05T19:00:00Z',
          last_success_at: '2026-08-05T18:30:00Z',
          detail: 'Bridge credentials expired.',
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    });

    render(
      <UserSpacePanel
        currentUser={{
          ...CURRENT_USER,
          id: 'viewer-1',
          username: 'viewer',
          display_name: 'Viewer',
          role: 'user',
        }}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText('Bridge expired')).toBeTruthy();
    });

    expect(previewApiMock.refreshUserSpaceBridgeCredentials).not.toHaveBeenCalled();
    expect(screen.getByText(/Bridge credentials expired\./)).toBeTruthy();
    expect(screen.getByText(/Expired /)).toBeTruthy();
    expect(screen.getByText(/Last success /)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Refresh bridge credentials' })).toBeNull();
  });

  it('automatically refreshes near-expiry bridge credentials for editable workspaces and reloads workspace state', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2099-08-05T18:00:00Z'));

    const nearExpiry = isoAtOffsetFromNow(4 * 60 * 1000);
    const refreshedExpiry = isoAtOffsetFromNow(65 * 60 * 1000);

    previewApiMock.getUserSpaceWorkspaceTabState.mockImplementation(async () => ({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        devserver_running: true,
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          expires_at: nearExpiry,
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    }));
    previewApiMock.refreshUserSpaceBridgeCredentials.mockResolvedValueOnce({
      ...STARTING_RUNTIME_STATUS.bridge_status,
      expires_at: refreshedExpiry,
      last_success_at: refreshedExpiry,
    });

    render(<UserSpacePanel currentUser={{ ...CURRENT_USER }} />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(1);
    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledWith('ws-1');

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });

    expect(previewApiMock.getUserSpaceWorkspaceTabState.mock.calls.length).toBeGreaterThanOrEqual(
      2,
    );
    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Refresh bridge credentials' })).toBeTruthy();
  });

  it('automatically refreshes an expired bridge credential for an editable workspace', async () => {
    previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        devserver_running: true,
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          state: 'expired',
          expires_at: '2000-08-05T19:00:00Z',
          detail: 'Runtime bridge credential has expired',
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    });

    render(<UserSpacePanel currentUser={{ ...CURRENT_USER }} />);

    await waitFor(() => {
      expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(1);
    });
    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledWith('ws-1');
  });

  it('retries a failed automatic refresh only after the 60-second cooldown while the credential stays unchanged', async () => {
    vi.useFakeTimers();
    const initialNow = new Date('2099-08-05T18:00:00Z');
    vi.setSystemTime(initialNow);

    const nearExpiry = isoAtOffsetFromNow(4 * 60 * 1000);
    const refreshedExpiry = isoAtOffsetFromNow(65 * 60 * 1000);

    previewApiMock.getUserSpaceWorkspaceTabState.mockImplementation(async () => ({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        devserver_running: true,
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          expires_at: nearExpiry,
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    }));
    previewApiMock.refreshUserSpaceBridgeCredentials
      .mockRejectedValueOnce(new Error('Refresh bridge failed'))
      .mockResolvedValueOnce({
        ...STARTING_RUNTIME_STATUS.bridge_status,
        expires_at: refreshedExpiry,
        last_success_at: refreshedExpiry,
      });

    render(<UserSpacePanel currentUser={{ ...CURRENT_USER }} />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(1);
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(50_000);
    });
    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(1);

    vi.setSystemTime(new Date(initialNow.getTime() + 60_000));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledTimes(2);
  });

  it('labels a future invalid bridge expiry as Expires instead of Expired', async () => {
    previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          state: 'invalid',
          expires_at: '2099-08-05T19:00:00Z',
          detail: 'Bridge credentials are invalid.',
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    });

    await renderPanelWithRuntimeOverlay(false);

    expect(screen.getByText('Bridge invalid')).toBeTruthy();
    expect(screen.getByText(/Expires /)).toBeTruthy();
    expect(screen.queryByText(/Expired /)).toBeNull();
  });

  it('shows a low-noise healthy bridge label without status details', async () => {
    await renderPanelWithRuntimeOverlay(false);

    expect(screen.getByText('Bridge healthy')).toBeTruthy();
    expect(screen.queryByText(/Last success /)).toBeNull();
    expect(screen.queryByText(/Expired /)).toBeNull();
  });

  it('shows editors a bridge refresh action in the runtime controls', async () => {
    await renderPanelWithRuntimeOverlay(false);

    expect(screen.getByRole('button', { name: 'Refresh bridge credentials' })).toBeTruthy();
  });

  it('hides the bridge refresh action when the runtime is not running', async () => {
    previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        session_state: 'stopped',
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          state: 'not_running',
          detail: 'Runtime session is not running.',
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    });

    render(<UserSpacePanel currentUser={{ ...CURRENT_USER }} />);

    await waitFor(() => {
      expect(screen.getByText('Bridge not running')).toBeTruthy();
    });

    expect(screen.queryByRole('button', { name: 'Refresh bridge credentials' })).toBeNull();
  });

  it('refreshes bridge credentials, disables runtime actions while busy, and reloads workspace state', async () => {
    let releaseRefresh!: () => void;
    const refreshGate = new Promise<void>((resolve) => {
      releaseRefresh = resolve;
    });
    previewApiMock.getUserSpaceWorkspaceTabState
      .mockResolvedValueOnce({
        workspace_id: WORKSPACE.id,
        runtime_status: {
          ...STARTING_RUNTIME_STATUS,
          bridge_status: {
            ...STARTING_RUNTIME_STATUS.bridge_status,
            state: 'session_mismatch',
            token_session_id: 'session-old',
            current_session_id: 'session-1',
            expires_at: '2099-08-05T19:00:00Z',
            detail: 'Bridge session mismatch.',
          },
        },
        chat_state: { ...DEFAULT_CHAT_STATE },
      })
      .mockResolvedValueOnce({
        workspace_id: WORKSPACE.id,
        runtime_status: {
          ...STARTING_RUNTIME_STATUS,
          devserver_running: true,
          bridge_status: {
            ...STARTING_RUNTIME_STATUS.bridge_status,
            state: 'healthy',
            token_session_id: 'session-1',
            current_session_id: 'session-1',
            detail: null,
          },
        },
        chat_state: { ...DEFAULT_CHAT_STATE },
      });
    previewApiMock.refreshUserSpaceBridgeCredentials.mockImplementationOnce(async () => {
      await refreshGate;
      return {
        state: 'healthy' as const,
        bridge_url: 'https://bridge.example',
        token_session_id: 'session-1',
        current_session_id: 'session-1',
        issued_at: '2026-08-05T18:00:00Z',
        expires_at: '2026-08-05T19:00:00Z',
        last_success_at: '2026-08-05T18:45:00Z',
        detail: null,
      };
    });

    await renderPanelWithRuntimeOverlay(false);
    const tabStateCallsBeforeRefresh =
      previewApiMock.getUserSpaceWorkspaceTabState.mock.calls.length;

    const refreshButton = screen.getByRole('button', { name: 'Refresh bridge credentials' });
    const stopButton = screen.getByTitle('Stop runtime');

    fireEvent.click(refreshButton);

    expect(previewApiMock.refreshUserSpaceBridgeCredentials).toHaveBeenCalledWith('ws-1');
    await waitFor(() => {
      expect(
        (
          screen.getByRole('button', {
            name: 'Refreshing bridge credentials…',
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(true);
    });
    expect((stopButton as HTMLButtonElement).disabled).toBe(true);

    releaseRefresh();

    await waitFor(() => {
      expect(screen.getByText('Bridge healthy')).toBeTruthy();
    });
    expect(previewApiMock.getUserSpaceWorkspaceTabState.mock.calls.length).toBeGreaterThan(
      tabStateCallsBeforeRefresh,
    );
    expect(
      (screen.getByRole('button', { name: 'Refresh bridge credentials' }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it('shows a refresh error and re-enables bridge controls when the refresh fails', async () => {
    previewApiMock.getUserSpaceWorkspaceTabState.mockResolvedValue({
      workspace_id: WORKSPACE.id,
      runtime_status: {
        ...STARTING_RUNTIME_STATUS,
        bridge_status: {
          ...STARTING_RUNTIME_STATUS.bridge_status,
          state: 'invalid',
          detail: 'Bridge credentials are invalid.',
        },
      },
      chat_state: { ...DEFAULT_CHAT_STATE },
    });
    previewApiMock.refreshUserSpaceBridgeCredentials.mockRejectedValueOnce(
      new Error('Refresh bridge failed'),
    );

    await renderPanelWithRuntimeOverlay(false);
    const tabStateCallsBeforeRefresh =
      previewApiMock.getUserSpaceWorkspaceTabState.mock.calls.length;

    fireEvent.click(screen.getByRole('button', { name: 'Refresh bridge credentials' }));

    await waitFor(() => {
      expect(screen.getByText('Refresh bridge failed')).toBeTruthy();
    });
    expect(
      (screen.getByRole('button', { name: 'Refresh bridge credentials' }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    expect(previewApiMock.getUserSpaceWorkspaceTabState.mock.calls.length).toBe(
      tabStateCallsBeforeRefresh,
    );
  });
});
