import type { ReactNode } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolsPanel } from './ToolsPanel';
import type { ToolConfig, ToolGroup } from '@/types';

const apiMock = vi.hoisted(() => ({
  listToolConfigs: vi.fn(),
  listToolGroups: vi.fn(),
  listUserspaceMountSources: vi.fn(),
  getToolHeartbeats: vi.fn(),
  subscribeToolHealthEvents: vi.fn(),
  updateToolConfig: vi.fn(),
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
vi.mock('./ToolWizard', () => ({ ToolWizard: () => null }));
vi.mock('./MountSourceWizard', () => ({ MountSourceWizard: () => null }));
vi.mock('./Popover', () => ({ Popover: ({ children }: { children: ReactNode }) => children }));
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
    apiMock.getToolHeartbeats.mockResolvedValue({ statuses: {} });
    apiMock.subscribeToolHealthEvents.mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
      onmessage: null,
    });
    apiMock.updateToolConfig.mockResolvedValue({ ...groupedTool, allow_write: true });
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

    await user.click(screen.getByLabelText('Read Only'));

    await screen.findByRole('heading', { name: 'Enable Write Access' });

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
});
