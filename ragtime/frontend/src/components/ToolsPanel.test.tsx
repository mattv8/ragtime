import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
    <button type="button" className={className} title={title} onClick={onDelete}>
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
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
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
    await user.click(await screen.findByRole('button', { name: /Read Only/ }));

    await screen.findByRole('heading', { name: 'Enable Write Access' });
    expect(screen.queryByRole('button', { name: /Read Only/ })).toBeNull();

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
