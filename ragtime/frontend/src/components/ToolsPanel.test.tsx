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
}));

const toastMock = {
  success: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
};

const toolFilterState = {
  queries: [],
  tags: [],
  input: '',
  hasActiveFilters: false,
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

type MockPopoverProps = {
  children: ReactNode;
  content?: ReactNode;
  position?: string;
  show?: boolean;
  trigger?: string;
  disabled?: boolean;
  openDelayMs?: number;
  followCursor?: boolean;
  ignoreSelector?: string;
} & HTMLAttributes<HTMLDivElement>;

vi.mock('./Popover', () => ({
  Popover: ({
    children,
    content: _content,
    position: _position,
    show: _show,
    trigger: _trigger,
    disabled: _disabled,
    openDelayMs: _openDelayMs,
    followCursor: _followCursor,
    ignoreSelector: _ignoreSelector,
    ...rest
  }: MockPopoverProps) => <div {...rest}>{children}</div>,
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
});
