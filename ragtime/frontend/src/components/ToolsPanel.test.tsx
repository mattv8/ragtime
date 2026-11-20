import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { HTMLAttributes, ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolsPanel } from './ToolsPanel';
import type { ToolConfig, ToolGroup } from '@/types';

const apiMock = vi.hoisted(() => ({
  listToolConfigs: vi.fn(),
  listToolGroups: vi.fn(),
  listUserspaceMountSources: vi.fn(),
  getSettings: vi.fn(),
  getToolHeartbeats: vi.fn(),
  subscribeToolHealthEvents: vi.fn(),
  updateToolConfig: vi.fn(),
  toggleToolConfig: vi.fn(),
  deleteToolConfig: vi.fn(),
  createToolConfig: vi.fn(),
  reorderTools: vi.fn(),
  clearToolUndecryptableCredentials: vi.fn(),
  getPdmIndexStats: vi.fn(),
  getToolAccessPolicy: vi.fn(),
  updateToolAccessPolicy: vi.fn(),
  listUsers: vi.fn(),
  listUsersDirectory: vi.fn(),
  listAuthGroups: vi.fn(),
}));

const toastMock = {
  success: vi.fn(),
  info: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
};

const toolFilterState = {
  queries: [] as string[],
  tags: [] as string[],
  input: '',
  debouncedInput: '',
  hasActiveFilters: false,
  setInput: vi.fn(),
  setTags: vi.fn(),
  clear: vi.fn(),
};

const resetToolFilterState = (): void => {
  toolFilterState.queries = [];
  toolFilterState.tags = [];
  toolFilterState.input = '';
  toolFilterState.debouncedInput = '';
  toolFilterState.hasActiveFilters = false;
};

vi.mock('@/api', () => ({ api: apiMock }));
const contentProtectionMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
}));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig: contentProtectionMock.getConfig },
  updateContentProtectionConfigSlice: contentProtectionMock.updateContentProtectionConfigSlice,
}));
let lastToolWizardExistingTool: ToolConfig | null | undefined = undefined;

vi.mock('./ToolWizard', () => ({
  ToolWizard: (props: { existingTool: ToolConfig | null }) => {
    lastToolWizardExistingTool = props.existingTool;
    return null;
  },
}));
vi.mock('./MountSourceWizard', () => ({ MountSourceWizard: () => null }));
vi.mock('./ToolAccessModal', () => ({
  ToolAccessModal: ({
    open,
    toolName,
    onSave,
  }: {
    open: boolean;
    toolName: string;
    onSave: (policy: unknown) => void;
  }) =>
    open ? (
      <div>
        <h3>Tool Access {toolName}</h3>
        <button
          type="button"
          onClick={() =>
            onSave({
              tool_id: 'tool-grouped',
              default_chat_access: 'read',
              default_workspace_access: 'deny',
              users: [],
              groups: [],
            })
          }
        >
          Save Access
        </button>
      </div>
    ) : null,
}));

type MockPopoverProps = {
  children: ReactNode;
  content?: ReactNode;
  position?: string;
  show?: boolean;
  trigger?: string;
  disabled?: boolean;
  openDelayMs?: number;
  followCursor?: boolean;
  requireHoverIdleMs?: number;
  focusTrigger?: boolean;
  ignoreSelector?: string;
} & HTMLAttributes<HTMLDivElement>;

vi.mock('./Popover', () => ({
  Popover: ({
    children,
    content: _content,
    position,
    show: _show,
    trigger: _trigger,
    disabled: _disabled,
    openDelayMs: _openDelayMs,
    followCursor,
    requireHoverIdleMs,
    focusTrigger,
    ignoreSelector,
    ...rest
  }: MockPopoverProps) => (
    <div
      {...rest}
      data-follow-cursor={followCursor === undefined ? undefined : String(followCursor)}
      data-require-hover-idle-ms={requireHoverIdleMs ?? undefined}
      data-focus-trigger={focusTrigger === undefined ? undefined : String(focusTrigger)}
      data-position={position}
      data-popover-content={typeof _content === 'string' ? _content : undefined}
      data-ignore-selector={ignoreSelector}
    >
      {children}
    </div>
  ),
}));
vi.mock('./AnimatedCreateButton', () => ({
  AnimatedCreateButton: ({ onClick, label }: { onClick: () => void; label: string }) => (
    <button type="button" onClick={onClick}>
      {label}
    </button>
  ),
}));
vi.mock('./DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({
    onDelete,
    className,
    title,
  }: {
    onDelete: () => void;
    className?: string;
    title?: string;
  }) => (
    <button
      type="button"
      className={className}
      title={title}
      data-delete-confirm-button
      onClick={onDelete}
    >
      Delete
    </button>
  ),
}));
vi.mock('./shared/Toast', () => ({
  useToast: () => [[], toastMock],
  ToastContainer: () => null,
}));
vi.mock('./shared/SearchFilterBar', () => ({
  SearchFilterBar: () => null,
  searchFilterTextMatchesQuery: () => true,
  useUrlSearchFilterState: () => toolFilterState,
}));

const groupedTool: ToolConfig = {
  id: 'tool-grouped',
  name: 'Grouped Tool',
  tool_type: 'ssh_shell',
  enabled: true,
  description: 'Grouped tool',
  connection_config: { host: 'example.com', user: 'deploy', port: 22 },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 100,
  group_id: 'group-1',
  group_name: 'Alpha Group',
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const ungroupedTool: ToolConfig = {
  id: 'tool-ungrouped',
  name: 'Ungrouped Tool',
  tool_type: 'ssh_shell',
  enabled: true,
  description: 'Ungrouped tool',
  connection_config: { host: 'example.org', user: 'deploy', port: 22 },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 200,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const pdmTool: ToolConfig = {
  id: 'tool-pdm',
  name: 'PDM Tool',
  tool_type: 'solidworks_pdm',
  enabled: true,
  description: 'SolidWorks PDM database',
  connection_config: {
    host: '192.168.10.12',
    port: 1433,
    user: 'pdm-user',
    password: 'secret',
    database: 'PDM',
  },
  max_results: 10,
  timeout_max_seconds: 30,
  allow_write: false,
  sort_order: 300,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const httpApiTool: ToolConfig = {
  id: 'tool-http-api',
  name: 'Orders API',
  tool_type: 'http_api',
  enabled: true,
  description: 'HTTP API connection',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'bearer',
  },
  max_results: 50,
  timeout_max_seconds: 300,
  allow_write: false,
  sort_order: 400,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: ['bearer_token'],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const pairedWindowsTool: ToolConfig = {
  ...ungroupedTool,
  id: 'tool-windows-paired',
  name: 'Paired Windows Tool',
  tool_type: 'windows_shell',
  connection_config: {
    host: 'windows.example.org',
    user: 'administrator',
    port: 22,
    pairing_id: 'a0a0a0a0-0000-4000-8000-000000000000',
  },
};

const linkedSolidworksTool: ToolConfig = {
  ...ungroupedTool,
  id: 'tool-solidworks',
  name: 'CAD workstation',
  tool_type: 'solidworks',
  connection_config: {
    host: 'cad.example.org',
    user: 'cad',
    port: 22,
    pdm_tool_id: pdmTool.id,
    pdm_approved_folders: ['\\Designs', '\\Parts'],
  },
};

const toolGroup: ToolGroup = {
  id: 'group-1',
  name: 'Alpha Group',
  description: '',
  sort_order: 100,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

class ResizeObserverMock {
  observe(): void {}
  disconnect(): void {}
  unobserve(): void {}
}

describe('ToolsPanel', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverMock);
    resetToolFilterState();
    toolFilterState.clear.mockReset();
    toolFilterState.clear.mockImplementation(() => {
      resetToolFilterState();
    });
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool]);
    apiMock.listToolGroups.mockResolvedValue([toolGroup]);
    apiMock.listUserspaceMountSources.mockResolvedValue([]);
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: false },
    });
    apiMock.getToolHeartbeats.mockResolvedValue({ statuses: {} });
    apiMock.subscribeToolHealthEvents.mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
      onmessage: null,
    });
    apiMock.updateToolConfig.mockResolvedValue({ ...groupedTool, allow_write: true });
    apiMock.deleteToolConfig.mockResolvedValue(undefined);
    apiMock.createToolConfig.mockResolvedValue({
      ...ungroupedTool,
      id: 'tool-ungrouped-copy',
      name: 'Ungrouped Tool Copy',
      sort_order: 300,
    });
    apiMock.reorderTools.mockResolvedValue(undefined);
    apiMock.getPdmIndexStats.mockResolvedValue({
      document_count: 0,
      embedding_count: 0,
      last_indexed_at: null,
    });
    apiMock.getToolAccessPolicy.mockResolvedValue({
      tool_id: groupedTool.id,
      default_chat_access: 'deny',
      default_workspace_access: 'deny',
      users: [],
      groups: [],
    });
    apiMock.updateToolAccessPolicy.mockResolvedValue({
      tool_id: groupedTool.id,
      default_chat_access: 'read',
      default_workspace_access: 'deny',
      users: [],
      groups: [],
    });
    apiMock.listUsers.mockResolvedValue([]);
    apiMock.listAuthGroups.mockResolvedValue([]);
    contentProtectionMock.getConfig.mockResolvedValue(null);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each([
    ['healthy', true, true, 'PDM: PDM Tool', 'Linked to SolidWorks PDM tool PDM Tool, vault PDM'],
    ['missing', false, false, 'PDM unavailable', 'The linked PDM tool no longer exists.'],
    [
      'disabled',
      true,
      false,
      'PDM unavailable',
      '“PDM Tool” is disabled, so the AI cannot search it. Enable it or choose another tool.',
    ],
  ])('shows the %s PDM link badge', async (_state, exists, enabled, label, description) => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...linkedSolidworksTool,
        pdm_link: { tool_id: pdmTool.id, tool_name: pdmTool.name, exists, enabled },
      },
      pdmTool,
    ]);
    render(<ToolsPanel />);
    await screen.findByRole('heading', { name: 'CAD workstation' });
    const badge = document.querySelector('[data-solidworks-pdm-badge]')!;
    expect(badge.textContent).toBe(label);
    expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(description);
    expect(badge.classList.contains('badge-warning')).toBe(!exists || !enabled);
  });

  it.each([
    ['healthy', [pdmTool], 'PDM: PDM Tool'],
    ['missing', [], 'PDM unavailable'],
    ['disabled', [{ ...pdmTool, enabled: false }], 'PDM unavailable'],
  ])('derives a %s link from loaded tools without API metadata', async (_state, targets, label) => {
    apiMock.listToolConfigs.mockResolvedValue([linkedSolidworksTool, ...targets]);
    render(<ToolsPanel />);
    await screen.findByRole('heading', { name: 'CAD workstation' });
    expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(label);
  });

  it.each([
    [
      true,
      ['\\Designs', '\\Parts'],
      'Can add files, check out and check in in 2 approved vault folders',
    ],
    [true, [], 'No approved folders: changes are refused'],
    [false, ['\\Designs'], null],
  ])(
    'shows write badge for allow_write=%s and folders=%j',
    async (allowWrite, folders, content) => {
      apiMock.listToolConfigs.mockResolvedValue([
        {
          ...linkedSolidworksTool,
          allow_write: allowWrite,
          connection_config: {
            ...linkedSolidworksTool.connection_config,
            pdm_approved_folders: folders,
          },
        },
        pdmTool,
      ]);
      render(<ToolsPanel />);
      await screen.findByRole('heading', { name: 'CAD workstation' });
      const badge = document.querySelector('[data-solidworks-write-badge]');
      if (content === null) expect(badge).toBeNull();
      else {
        expect(badge?.textContent).toBe('Write');
        expect(badge?.parentElement?.getAttribute('data-popover-content')).toBe(content);
      }
    },
  );

  it.each([true, false])('offers the changes kill switch only when on=%s', async (allowWrite) => {
    apiMock.listToolConfigs.mockResolvedValue([
      { ...linkedSolidworksTool, allow_write: allowWrite },
      pdmTool,
    ]);
    apiMock.updateToolConfig.mockResolvedValue({ ...linkedSolidworksTool, allow_write: false });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByRole('heading', { name: 'CAD workstation' }));
    const toggle = screen.queryByRole('button', { name: 'Allow changes' });
    if (!allowWrite) {
      expect(toggle).toBeNull();
      return;
    }
    expect(toggle?.getAttribute('aria-checked')).toBe('true');
    fireEvent.click(toggle!);
    await waitFor(() =>
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(linkedSolidworksTool.id, {
        allow_write: false,
      }),
    );
    expect(screen.queryByRole('heading', { name: 'Enable Write Access' })).toBeNull();
  });

  it.each([true, false])('shows Used by only for a linked PDM tool: %s', async (linked) => {
    apiMock.listToolConfigs.mockResolvedValue(linked ? [linkedSolidworksTool, pdmTool] : [pdmTool]);
    render(<ToolsPanel />);
    await screen.findByText('PDM Tool');
    const row = screen.queryByTestId(`pdm-linked-from-${pdmTool.id}`);
    if (linked) expect(row?.textContent).toBe('Used by:CAD workstation');
    else expect(row).toBeNull();
  });

  it.each([true, false])(
    'confirms linked PDM card deletion with changes on=%s',
    async (allowWrite) => {
      apiMock.getSettings.mockResolvedValue({ settings: { show_tool_card_footer_actions: true } });
      apiMock.listToolConfigs.mockResolvedValue([
        { ...linkedSolidworksTool, allow_write: allowWrite },
        pdmTool,
      ]);
      render(<ToolsPanel />);
      await screen.findByText('PDM Tool');
      const card = document.getElementById(`tools-tool-${pdmTool.id}`)!;
      const deleteButton = card.querySelector('[title="Delete tool"]')!;
      expect(deleteButton.hasAttribute('data-delete-confirm-button')).toBe(false);
      fireEvent.click(deleteButton);
      expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
      expect(screen.getByRole('heading', { name: 'Delete Tool' })).toBeTruthy();
      const note = screen.getByRole('note');
      expect(note.id).toBe('tool-delete-linked-note');
      expect(note.textContent).toContain(
        '“PDM Tool” is linked to the SolidWorks tool “CAD workstation”. If you delete it, that tool can no longer search the vault or open, add, check out or check in vault files until you link another PDM tool. The SolidWorks tool keeps its other settings and shows a warning.',
      );
      expect(note.textContent?.includes('Changes are currently on in 2 approved folders.')).toBe(
        allowWrite,
      );
      fireEvent.click(screen.getByRole('button', { name: 'Delete Tool' }));
      await waitFor(() => expect(apiMock.deleteToolConfig).toHaveBeenCalledWith(pdmTool.id));
      expect(apiMock.updateToolConfig).not.toHaveBeenCalled();
      expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(
        'PDM unavailable',
      );
    },
  );

  it('shows every linked SolidWorks tool in the PDM row and context-menu delete note', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      linkedSolidworksTool,
      { ...linkedSolidworksTool, id: 'cad-second', name: 'Second workstation', allow_write: true },
      pdmTool,
    ]);
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByRole('heading', { name: 'PDM Tool' }));
    expect(screen.getByTestId(`pdm-linked-from-${pdmTool.id}`).textContent).toBe(
      'Used by:CAD workstation, Second workstation',
    );
    fireEvent.click(
      within(document.querySelector('.userspace-tool-context-menu') as HTMLElement).getByRole(
        'button',
        { name: 'Delete' },
      ),
    );
    const note = screen.getByRole('note');
    expect(note.textContent).toContain('SolidWorks tool “CAD workstation”');
    expect(note.textContent).toContain('SolidWorks tool “Second workstation”');
    expect(note.textContent).toContain('Changes are currently on in 2 approved folders.');
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
  });

  it('keeps the unlinked PDM card deletion path unchanged', async () => {
    apiMock.getSettings.mockResolvedValue({ settings: { show_tool_card_footer_actions: true } });
    apiMock.listToolConfigs.mockResolvedValue([pdmTool]);
    render(<ToolsPanel />);
    await screen.findByText('PDM Tool');
    const deleteButton = document.querySelector('[title="Delete tool"]')!;
    expect(deleteButton.hasAttribute('data-delete-confirm-button')).toBe(true);
    fireEvent.click(deleteButton);
    await waitFor(() => expect(apiMock.deleteToolConfig).toHaveBeenCalledWith(pdmTool.id));
    expect(screen.queryByRole('heading', { name: 'Delete Tool' })).toBeNull();
  });

  it('shows linked-tool consequences after disabling PDM succeeds', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...linkedSolidworksTool,
        pdm_link: { tool_id: pdmTool.id, tool_name: pdmTool.name, enabled: true, exists: true },
      },
      pdmTool,
    ]);
    apiMock.toggleToolConfig.mockResolvedValue({ enabled: false });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByText('PDM Tool'));
    fireEvent.click(screen.getByRole('button', { name: 'Enabled' }));
    await waitFor(() =>
      expect(toastMock.info).toHaveBeenCalledWith(
        '“PDM Tool” is disabled. The SolidWorks tool “CAD workstation” cannot search the vault until you enable it again.',
      ),
    );
    expect(apiMock.toggleToolConfig).toHaveBeenCalledWith(pdmTool.id, false);
    expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(
      'PDM unavailable',
    );
  });

  it('shows SolidWorks connection and relevant badges without write or working directory', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...pairedWindowsTool,
        id: 'solidworks-pinned',
        name: 'SolidWorks pinned',
        tool_type: 'solidworks',
        allow_write: true,
        connection_config: {
          host: 'cad.example.org',
          user: 'cad',
          port: 2222,
          solidworks_version: 2024,
          instance_mode: 'attach',
          working_directory: 'C:\\work',
        },
      },
      {
        ...pairedWindowsTool,
        id: 'solidworks-default',
        name: 'SolidWorks default',
        tool_type: 'solidworks',
        connection_config: { host: 'other.example.org', user: 'cad', port: 22 },
      },
    ]);
    render(<ToolsPanel />);
    await screen.findByText('SolidWorks pinned');
    const cards = document.querySelectorAll('[data-tool-type="solidworks"]');
    expect(cards).toHaveLength(2);
    const pinned = Array.from(cards).find((card) =>
      card.textContent?.includes('SolidWorks pinned'),
    )!;
    const defaults = Array.from(cards).find((card) =>
      card.textContent?.includes('SolidWorks default'),
    )!;
    expect(pinned.querySelector('[data-tool-summary]')?.textContent).toBe(
      'cad@cad.example.org:2222',
    );
    expect(pinned.querySelector('[data-solidworks-version-badge]')?.textContent).toBe(
      'SolidWorks 2024',
    );
    expect(pinned.querySelector('[data-solidworks-mode-badge]')?.textContent).toBe('Attach');
    expect(pinned.textContent).not.toContain('Write');
    expect(pinned.textContent).not.toContain('C:\\work');
    expect(defaults.querySelector('[data-solidworks-version-badge]')).toBeNull();
    expect(defaults.querySelector('[data-solidworks-mode-badge]')).toBeNull();
    fireEvent.contextMenu(pinned);
    expect(screen.queryByRole('button', { name: 'Write Access' })).toBeNull();
  });

  it('keeps the selected group open when the write confirmation modal is dismissed', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    await user.click(screen.getByText('Alpha Group'));

    await waitFor(() => {
      expect(screen.getByText('Grouped Tool')).toBeTruthy();
      expect(screen.queryByText('Ungrouped Tool')).toBeNull();
    });

    fireEvent.contextMenu(screen.getByText('Grouped Tool'));
    await user.click(await screen.findByRole('button', { name: 'Write Access' }));

    await screen.findByRole('heading', { name: 'Enable Write Access' });
    expect(screen.queryByRole('button', { name: 'Write Access' })).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      expect(screen.queryByText('Enable Write Access')).toBeNull();
      expect(screen.getByText('Grouped Tool')).toBeTruthy();
      expect(screen.queryByText('Ungrouped Tool')).toBeNull();
    });
  });

  it('still closes the selected group on outside mousedown when no modal is open', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    await user.click(screen.getByText('Alpha Group'));

    await waitFor(() => {
      expect(screen.getByText('Grouped Tool')).toBeTruthy();
    });

    fireEvent.mouseDown(document.body);

    await waitFor(() => {
      expect(screen.getByText('Ungrouped Tool')).toBeTruthy();
      expect(screen.queryByText('Grouped Tool')).toBeNull();
    });
  });

  it('clears typed search filters when selecting an inactive group tab but not when closing it', async () => {
    const user = userEvent.setup();
    toolFilterState.queries = ['ssh', 'database', 'staging'];
    toolFilterState.tags = ['database'];
    toolFilterState.input = 'staging';
    toolFilterState.debouncedInput = 'staging';
    toolFilterState.hasActiveFilters = true;

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    await user.click(screen.getByText('Alpha Group'));

    expect(toolFilterState.clear).toHaveBeenCalledTimes(1);
    expect(toolFilterState.queries).toEqual([]);
    expect(toolFilterState.tags).toEqual([]);
    expect(toolFilterState.input).toBe('');
    expect(toolFilterState.hasActiveFilters).toBe(false);

    await user.click(screen.getByText('Alpha Group'));

    expect(toolFilterState.clear).toHaveBeenCalledTimes(1);
  });

  it('adds soft wrap opportunities only to heartbeat error tokens longer than 30 characters', async () => {
    apiMock.getToolHeartbeats.mockResolvedValue({
      statuses: {
        [ungroupedTool.id]: {
          tool_id: ungroupedTool.id,
          alive: false,
          latency_ms: null,
          error: `Error: ${'a'.repeat(30)} ${'b'.repeat(31)}`,
          checked_at: '2026-01-01T00:00:00Z',
        },
      },
    });

    const { container } = render(<ToolsPanel />);

    await screen.findByText(`Error: ${'a'.repeat(30)} ${'b'.repeat(31)}`);

    expect(container.querySelectorAll('.tool-card-heartbeat-error wbr')).toHaveLength(1);
  });

  it('does not add soft wrap opportunities to a 30-character heartbeat error token', async () => {
    apiMock.getToolHeartbeats.mockResolvedValue({
      statuses: {
        [ungroupedTool.id]: {
          tool_id: ungroupedTool.id,
          alive: false,
          latency_ms: null,
          error: `Error: ${'a'.repeat(30)}`,
          checked_at: '2026-01-01T00:00:00Z',
        },
      },
    });

    const { container } = render(<ToolsPanel />);

    await screen.findByText(`Error: ${'a'.repeat(30)}`);

    expect(container.querySelectorAll('.tool-card-heartbeat-error wbr')).toHaveLength(0);
  });

  it('duplicates a tool from the right-click card menu and keeps the copy after the original', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith({
        name: 'Ungrouped Tool Copy',
        tool_type: 'ssh_shell',
        description: 'Ungrouped tool',
        connection_config: { host: 'example.org', user: 'deploy', port: 22 },
        max_results: 10,
        timeout_max_seconds: 30,
        allow_write: false,
        group_id: null,
      });
    });

    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-ungrouped', 'tool-ungrouped-copy'],
    });
    expect(await screen.findByText('Ungrouped Tool Copy')).toBeTruthy();
    expect(toastMock.success).toHaveBeenCalledWith('Tool duplicated');
  });

  it('opens the access modal from the card menu and saves ACL changes', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    await user.click(screen.getByText('Alpha Group'));
    await screen.findByText('Grouped Tool');

    fireEvent.contextMenu(screen.getByText('Grouped Tool'));
    await user.click(await screen.findByRole('button', { name: 'Edit Users' }));

    await waitFor(() => {
      expect(apiMock.getToolAccessPolicy).toHaveBeenCalledWith('tool-grouped');
    });
    expect(apiMock.listUsers).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('button', { name: 'Save Access' }));

    await waitFor(() => {
      expect(apiMock.updateToolAccessPolicy).toHaveBeenCalledWith(
        'tool-grouped',
        expect.objectContaining({
          default_chat_access: 'read',
          default_workspace_access: 'deny',
        }),
      );
    });
  });

  it('hides card footer actions by default and exposes them in the right-click menu', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    expect(screen.queryByRole('button', { name: /^Test$/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /^Edit$/ })).toBeNull();
    expect(screen.queryAllByRole('button', { name: /^Delete$/ })).toHaveLength(1);

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));

    expect(await screen.findByRole('button', { name: /^Test$/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Edit$/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Duplicate$/ })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: /^Delete$/ })).toHaveLength(2);

    await user.click(screen.getByRole('button', { name: /^Edit$/ }));

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /^Test$/ })).toBeNull();
    });
  });

  it('shows card footer actions when the global legacy setting is enabled', async () => {
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: true },
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    expect(screen.getByRole('button', { name: /^Test$/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Edit$/ })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: /^Delete$/ })).toHaveLength(2);
  });

  it('shows delete confirmation modal before deleting a tool', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await screen.findByRole('button', { name: /Duplicate/ });
    const deleteButtons = screen.getAllByRole('button', { name: /Delete/ });
    await user.click(deleteButtons[deleteButtons.length - 1]);

    await screen.findByRole('heading', { name: 'Delete Tool' });
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Delete Tool' }));

    await waitFor(() => {
      expect(apiMock.deleteToolConfig).toHaveBeenCalledWith('tool-ungrouped');
    });
  });

  it('opens Windows cleanup confirmation from both tool deletion entrypoints', async () => {
    const user = userEvent.setup();
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: true },
    });
    apiMock.listToolConfigs.mockResolvedValue([pairedWindowsTool]);

    render(<ToolsPanel />);

    await screen.findByText('Paired Windows Tool');
    await user.click(screen.getByTitle('Delete tool'));
    expect(await screen.findByRole('heading', { name: 'Delete paired Windows tool' })).toBeTruthy();
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
    expect(document.querySelector('[title="Delete tool"][data-delete-confirm-button]')).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Close' }));
    fireEvent.contextMenu(screen.getByText('Paired Windows Tool'));
    const contextDeleteButtons = await screen.findAllByRole('button', { name: /^Delete$/ });
    await user.click(contextDeleteButtons[contextDeleteButtons.length - 1]);
    expect(await screen.findByRole('heading', { name: 'Delete paired Windows tool' })).toBeTruthy();
  });

  it('shows write badge on the card when write access is enabled', async () => {
    const writeTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-write',
      name: 'Write Tool',
      allow_write: true,
      undecryptable_fields: [],
      connection_config: { host: 'example.org', user: 'deploy', port: 22 },
    };
    apiMock.listToolConfigs.mockResolvedValue([writeTool]);

    render(<ToolsPanel />);

    await screen.findByText('Write Tool');
    expect(screen.getByText('Write')).toBeTruthy();
  });

  it('shows a Classified badge only for required tools while protection is enabled', async () => {
    contentProtectionMock.getConfig.mockResolvedValue({
      revision: 1,
      enabled: true,
      classifier_model: null,
      coverage_mode: 'selected_scopes',
      profiles: [],
      group_profiles: [],
      requirements: [{ scope_kind: 'tool', scope_key: 'tool-ungrouped', mode: 'require' }],
      user_overrides: [],
    });

    render(<ToolsPanel />);

    expect(await screen.findByText('Classified')).toBeTruthy();
  });

  it('hides the Classified badge while content protection is disabled', async () => {
    contentProtectionMock.getConfig.mockResolvedValue({
      revision: 1,
      enabled: false,
      classifier_model: null,
      coverage_mode: 'selected_scopes',
      profiles: [],
      group_profiles: [],
      requirements: [{ scope_kind: 'tool', scope_key: 'tool-ungrouped', mode: 'require' }],
      user_overrides: [],
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    expect(screen.queryByText('Classified')).toBeNull();
  });

  it('shows PDM index document stats on SolidWorks PDM cards', async () => {
    apiMock.listToolConfigs.mockResolvedValue([pdmTool]);
    apiMock.getPdmIndexStats.mockResolvedValue({
      document_count: 1160,
      embedding_count: 3480,
      last_indexed_at: '2026-01-02T00:00:00Z',
    });

    render(<ToolsPanel />);

    await screen.findByText('PDM Tool');

    expect(apiMock.getPdmIndexStats).toHaveBeenCalledWith('tool-pdm');
    expect(await screen.findByText('PDM Index:')).toBeTruthy();
    expect(screen.getByText('1160 documents')).toBeTruthy();
  });

  it('shows / in the working directory badge when unset', async () => {
    const noWdTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-nowd',
      name: 'No Work Dir Tool',
      undecryptable_fields: [],
      connection_config: { host: 'example.org', user: 'deploy', port: 22 },
    };
    apiMock.listToolConfigs.mockResolvedValue([noWdTool]);

    render(<ToolsPanel />);

    await screen.findByText('No Work Dir Tool');
    expect(screen.getByText('/')).toBeTruthy();
  });

  it('highlights the requested tool card when navigated to a tool-specific section target', async () => {
    render(<ToolsPanel highlightSection="tool:tool-ungrouped" />);

    const toolName = await screen.findByText('Ungrouped Tool');
    const toolCard = toolName.closest('.tool-card');

    await waitFor(() => expect(toolCard?.classList.contains('highlight-setting')).toBe(true));
  });

  it('highlights tool connections when opened with its hash', async () => {
    const originalHash = window.location.hash;
    window.location.hash = '#tools-connections';
    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    await waitFor(() =>
      expect(
        document.getElementById('tools-connections')?.classList.contains('highlight-setting'),
      ).toBe(true),
    );
    window.location.hash = originalHash;
  });

  it('uses the next copy suffix when duplicating a tool with an existing copy', async () => {
    const user = userEvent.setup();
    const existingCopy: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-existing-copy',
      name: 'Ungrouped Tool Copy',
      sort_order: 300,
      undecryptable_fields: [],
    };
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool, existingCopy]);
    apiMock.createToolConfig.mockResolvedValue({
      ...ungroupedTool,
      id: 'tool-ungrouped-copy-2',
      name: 'Ungrouped Tool Copy 2',
      sort_order: 400,
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'Ungrouped Tool Copy 2' }),
      );
    });
    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-ungrouped', 'tool-ungrouped-copy-2', 'tool-existing-copy'],
    });
  });

  it('preserves disabled state after duplicating a disabled tool', async () => {
    const user = userEvent.setup();
    const disabledTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-disabled',
      name: 'Disabled Tool',
      enabled: false,
      sort_order: 300,
      undecryptable_fields: [],
    };
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, disabledTool]);
    apiMock.createToolConfig.mockResolvedValue({
      ...disabledTool,
      id: 'tool-disabled-copy',
      name: 'Disabled Tool Copy',
      enabled: true,
      sort_order: 400,
    });
    apiMock.updateToolConfig.mockResolvedValue({
      ...disabledTool,
      id: 'tool-disabled-copy',
      name: 'Disabled Tool Copy',
      enabled: false,
      sort_order: 400,
    });

    render(<ToolsPanel />);

    await screen.findByText('Disabled Tool');

    fireEvent.contextMenu(screen.getByText('Disabled Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith('tool-disabled-copy', {
        enabled: false,
      });
    });
    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-disabled', 'tool-disabled-copy'],
    });
  });

  it('shows a warning badge when a tool has undecryptable credential fields', async () => {
    const brokenTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-broken',
      name: 'Broken Tool',
      undecryptable_fields: ['password'],
    };
    apiMock.listToolConfigs.mockResolvedValue([ungroupedTool, brokenTool]);

    render(<ToolsPanel />);

    await screen.findByText('Broken Tool');
    expect(screen.getByText('Key Mismatch')).toBeTruthy();
    expect(
      screen.getByTitle(/credentials cannot be decrypted with the current server key/i),
    ).toBeTruthy();
  });

  it('clears broken credentials and opens the edit wizard after confirmation', async () => {
    const user = userEvent.setup();
    const brokenTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-broken',
      name: 'Broken Tool',
      undecryptable_fields: ['password'],
    };
    const clearedTool: ToolConfig = {
      ...brokenTool,
      undecryptable_fields: [],
      updated_at: '2026-01-02T00:00:00Z',
    };
    apiMock.listToolConfigs.mockResolvedValue([ungroupedTool, brokenTool]);
    apiMock.clearToolUndecryptableCredentials.mockResolvedValue(clearedTool);

    render(<ToolsPanel />);

    await screen.findByText('Broken Tool');

    fireEvent.contextMenu(screen.getByText('Broken Tool'));
    await user.click(await screen.findByRole('button', { name: /Clear broken credentials/ }));

    await screen.findByRole('heading', { name: 'Clear Broken Credentials' });
    expect(screen.getByText(/cannot be decrypted with the current server key/i)).toBeTruthy();
    expect(screen.getByText('password')).toBeTruthy();
    expect(apiMock.clearToolUndecryptableCredentials).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Clear Credentials' }));

    await waitFor(() => {
      expect(apiMock.clearToolUndecryptableCredentials).toHaveBeenCalledWith('tool-broken');
    });
    expect(toastMock.success).toHaveBeenCalledWith(
      'Broken tool credentials cleared. Re-enter them to restore the connection.',
    );
    expect(lastToolWizardExistingTool).toEqual(clearedTool);
  });

  it('shows live password requirements in export modal and enables export only for a strong matching password', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /^Export$/ }));

    await screen.findByRole('heading', { name: 'Export Tool Config' });
    // With an empty password every requirement is still outstanding.
    expect(screen.getByText('12+ characters')).toBeTruthy();
    expect(screen.getByText('Uppercase')).toBeTruthy();
    expect(screen.getByText('Lowercase')).toBeTruthy();
    expect(screen.getByText('Number')).toBeTruthy();
    expect(screen.getByText('Special character')).toBeTruthy();

    const passwordInput = screen.getByLabelText(/^Password$/i) as HTMLInputElement;
    const confirmInput = screen.getByLabelText(/^Confirm Password$/i) as HTMLInputElement;
    const exportButton = screen.getByRole('button', { name: /^Export$/ }) as HTMLButtonElement;

    await user.type(passwordInput, 'short1!');
    expect(exportButton.disabled).toBe(true);
    // Met requirements drop out of the "still needed" hint.
    expect(screen.queryByText('Lowercase')).toBeNull();
    expect(screen.getByText('12+ characters')).toBeTruthy();

    await user.clear(passwordInput);
    await user.type(passwordInput, 'StrongPass123!');
    expect(exportButton.disabled).toBe(true);
    // All requirements satisfied collapses to the success line.
    expect(screen.getByText('Password meets all requirements')).toBeTruthy();

    await user.type(confirmInput, 'StrongPass123!');
    await waitFor(() => expect(exportButton.disabled).toBe(false));
  });

  it('uses idle-hover popovers for tool-card right-click hints', async () => {
    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    const toolCardPopover = document.querySelector('.tool-card-drag-wrap') as HTMLElement;
    expect(toolCardPopover.dataset.followCursor).toBe('true');
    expect(toolCardPopover.dataset.requireHoverIdleMs).toBe('1000');
    expect(toolCardPopover.dataset.focusTrigger).toBe('false');
    expect(toolCardPopover.dataset.position).toBe('top');
    expect(toolCardPopover.dataset.ignoreSelector).not.toContain('.editable-field-wrapper');
    expect(toolCardPopover.dataset.ignoreSelector).toContain('button');
    expect(toolCardPopover.dataset.ignoreSelector).toContain('input');
    expect(toolCardPopover.dataset.ignoreSelector).toContain('textarea');
  });

  it('shows an HTTP API summary with base URL and auth mode without secrets', async () => {
    apiMock.listToolConfigs.mockResolvedValue([httpApiTool]);

    render(<ToolsPanel />);

    expect(await screen.findByText('Orders API')).toBeTruthy();
    expect(screen.getByText(/https:\/\/api\.example\.com/i)).toBeTruthy();
    expect(screen.getByText(/bearer/i)).toBeTruthy();
    expect(screen.queryByText(/token/i)).toBeNull();
  });
});
