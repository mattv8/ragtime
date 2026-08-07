import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const {
  apiMock,
  eventSourceInstances,
  toastMock,
}: {
  apiMock: Record<string, ReturnType<typeof vi.fn>>;
  eventSourceInstances: Array<{
    listeners: Map<string, (event: MessageEvent) => void>;
    close: ReturnType<typeof vi.fn>;
  }>;
  toastMock: [
    never[],
    {
      error: ReturnType<typeof vi.fn>;
      success: ReturnType<typeof vi.fn>;
      info: ReturnType<typeof vi.fn>;
      dismiss: ReturnType<typeof vi.fn>;
    },
  ];
} = vi.hoisted(() => ({
  apiMock: {
    listUserSpaceSqliteDatabases: vi.fn(),
    listUserSpaceSqliteTables: vi.fn(),
    listUserSpaceSqliteRows: vi.fn(),
    getUserSpaceSqliteTableSchema: vi.fn(),
    initializeUserSpaceSqliteDatabase: vi.fn(),
    importUserSpaceSqliteDatabase: vi.fn(),
    deleteUserSpaceSqliteDatabase: vi.fn(),
    exportUserSpaceSqliteDatabase: vi.fn(),
    exportUserSpaceSqliteTable: vi.fn(),
    importUserSpaceSqliteTable: vi.fn(),
    createUserSpaceSqliteTable: vi.fn(),
    dropUserSpaceSqliteTable: vi.fn(),
    insertUserSpaceSqliteRow: vi.fn(),
    updateUserSpaceSqliteRow: vi.fn(),
    deleteUserSpaceSqliteRow: vi.fn(),
    alterUserSpaceSqliteTable: vi.fn(),
    queryUserSpaceSqliteDatabase: vi.fn(),
  },
  eventSourceInstances: [],
  toastMock: [[], { error: vi.fn(), success: vi.fn(), info: vi.fn(), dismiss: vi.fn() }],
}));

vi.mock('@/api/client', () => ({
  api: apiMock,
}));

vi.mock('./Toast', () => ({
  useToast: () => toastMock,
  ToastContainer: () => null,
}));

import { WorkspaceSqliteInspectorModal } from './WorkspaceSqliteInspectorModal';

type InspectorDatabase = {
  name: string;
  relative_path: string;
  size_bytes: number;
  table_count: number;
  last_modified_ms: number | null;
  owner_workspace_id: string;
  owner_workspace_name: string;
  ownership: 'owned' | 'linked';
  access_mode: 'read' | 'read_write';
  persistence_mode: 'include' | 'exclude';
  initialized: boolean;
};

type LegacyInspectorDatabase = Pick<
  InspectorDatabase,
  'name' | 'relative_path' | 'size_bytes' | 'table_count' | 'last_modified_ms'
>;

const owned: InspectorDatabase = {
  name: 'app.sqlite3',
  relative_path: '.ragtime/db/app.sqlite3',
  size_bytes: 1024,
  table_count: 2,
  last_modified_ms: 1_786_032_000_000,
  owner_workspace_id: 'source-ws',
  owner_workspace_name: 'Source Workspace',
  ownership: 'owned',
  access_mode: 'read_write',
  persistence_mode: 'include',
  initialized: true,
};

const linkedRead: InspectorDatabase = {
  ...owned,
  owner_workspace_id: 'target-read',
  owner_workspace_name: 'Reporting',
  ownership: 'linked',
  access_mode: 'read',
};

const linkedWriteMissing: InspectorDatabase = {
  ...owned,
  size_bytes: 0,
  table_count: 0,
  last_modified_ms: null,
  owner_workspace_id: 'target-write',
  owner_workspace_name: 'Operations',
  ownership: 'linked',
  access_mode: 'read_write',
  persistence_mode: 'exclude',
  initialized: false,
};

const legacyOwned: LegacyInspectorDatabase = {
  name: 'app.sqlite3',
  relative_path: '.ragtime/db/app.sqlite3',
  size_bytes: 0,
  table_count: 0,
  last_modified_ms: 1_786_032_000_000,
};

const listResponse = {
  workspace_id: 'source-ws',
  databases: [owned, linkedRead, linkedWriteMissing],
  total_bytes: 1024,
  default_database_name: 'app.sqlite3',
  persistence_mode: 'include' as const,
};

const tableListResponse = {
  workspace_id: 'source-ws',
  database: linkedRead,
  tables: [{ name: 'events', type: 'table', row_count: 1 }],
  persistence_mode: 'include' as const,
  mode_promoted: false,
};

const rowPageResponse = {
  database: linkedRead,
  table: { name: 'events', type: 'table', row_count: 1 },
  columns: [{ name: 'id', type: 'INTEGER', primary_key: true, not_null: true }],
  rows: [{ id: 1 }],
  total: 1,
  offset: 0,
  limit: 50,
  elapsed_ms: 3,
};

const schemaResponse = {
  database: linkedRead,
  table: { name: 'events', type: 'table', row_count: 1 },
  schema: {
    columns: [{ name: 'id', type: 'INTEGER', primary_key: true, not_null: true }],
    indexes: [],
    foreign_keys: [],
    create_sql: 'CREATE TABLE events (id INTEGER PRIMARY KEY);',
  },
  mode_promoted: false,
};

class MockEventSource {
  listeners = new Map<string, (event: MessageEvent) => void>();
  close = vi.fn();

  constructor(_url: string) {
    eventSourceInstances.push(this);
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    this.listeners.set(type, listener);
  }
}

function renderModal() {
  return render(
    <WorkspaceSqliteInspectorModal
      isOpen
      workspaceId="source-ws"
      workspaceName="Source Workspace"
      canEdit
      onClose={vi.fn()}
      onPersistencePromoted={vi.fn()}
    />,
  );
}

describe('WorkspaceSqliteInspectorModal', () => {
  beforeEach(() => {
    vi.stubGlobal('EventSource', MockEventSource as unknown as typeof EventSource);
    eventSourceInstances.length = 0;
    Object.values(apiMock).forEach((mockFn) => mockFn.mockReset());
    apiMock.listUserSpaceSqliteDatabases.mockResolvedValue(listResponse);
    apiMock.listUserSpaceSqliteTables.mockResolvedValue(tableListResponse);
    apiMock.listUserSpaceSqliteRows.mockResolvedValue(rowPageResponse);
    apiMock.getUserSpaceSqliteTableSchema.mockResolvedValue(schemaResponse);
    apiMock.queryUserSpaceSqliteDatabase.mockResolvedValue({
      sql: 'SELECT 1;',
      columns: ['value'],
      rows: [{ value: 1 }],
      row_count: 1,
      truncated: false,
      elapsed_ms: 1,
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders one unified database grid with owned and linked cards', async () => {
    apiMock.listUserSpaceSqliteDatabases.mockResolvedValue({
      ...listResponse,
      databases: [legacyOwned, linkedRead, linkedWriteMissing],
    });
    renderModal();

    await screen.findAllByRole('button', { name: /^app\.sqlite3/i });

    expect(screen.queryByRole('heading', { name: 'Workspace databases' })).toBeNull();
    expect(screen.queryByRole('heading', { name: 'Linked databases' })).toBeNull();
    expect(screen.getByText('0 tables')).toBeTruthy();
    expect(screen.getByText('0 B')).toBeTruthy();
    expect(screen.queryByText('Owned')).toBeNull();
    expect(screen.queryByText('Read only')).toBeNull();
    expect(screen.queryByText('Read / Write')).toBeNull();
    expect(screen.getByText('Linked from Reporting')).toBeTruthy();
    expect(screen.getByText('Linked from Operations')).toBeTruthy();
    expect(screen.getByText('Not initialized')).toBeTruthy();
    expect(screen.getAllByRole('button', { name: /^app\.sqlite3/i })).toHaveLength(3);

    const cards = screen.getAllByRole('button', { name: /^app\.sqlite3/i });
    const ownedCard = cards[0]?.closest('.userspace-sqlite-card');
    expect(ownedCard).toBeTruthy();
    expect(ownedCard?.querySelector('.userspace-sqlite-card-badge')).toBeNull();

    const reportingCard = screen
      .getByText('Linked from Reporting')
      .closest('.userspace-sqlite-card');
    expect(reportingCard).toBeTruthy();
    expect(reportingCard?.querySelector('.lucide-database')).toBeTruthy();
    expect(reportingCard?.querySelector('.lucide-link-2')).toBeNull();
    const modifiedSpans = screen.getAllByText(/^Modified /i);
    expect(modifiedSpans.length).toBeGreaterThan(0);
    modifiedSpans.forEach((modifiedSpan) => {
      expect(modifiedSpan.classList.contains('userspace-sqlite-card-modified')).toBe(true);
    });
    const reportingSource = screen.getByText('Linked from Reporting');
    const reportingMeta = reportingSource.closest('.userspace-sqlite-card-meta');
    expect(reportingMeta).toBeTruthy();
    const reportingModified = reportingMeta?.querySelector('.userspace-sqlite-card-modified');
    expect(reportingModified).toBeTruthy();
    const reportingBadge = reportingCard?.querySelector('.userspace-sqlite-card-badge');
    expect(reportingBadge?.textContent).toBe('Reporting');
    expect(reportingBadge?.getAttribute('title')).toBe('Reporting');
    expect(reportingSource.classList.contains('userspace-sqlite-card-source')).toBe(true);
    expect(reportingModified?.nextElementSibling).toBe(reportingSource);
    expect(reportingMeta?.lastElementChild).toBe(reportingSource);

    const operationsCard = screen
      .getByText('Linked from Operations')
      .closest('.userspace-sqlite-card');
    expect(operationsCard).toBeTruthy();
    const operationsBadge = operationsCard?.querySelector('.userspace-sqlite-card-badge');
    expect(operationsBadge?.textContent).toBe('Operations');
    expect(operationsBadge?.getAttribute('title')).toBe('Operations');
  });

  it('routes linked reads through the owner workspace, keeps read-only access visible, and reconciles SSE by owner-aware key', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click((await screen.findAllByRole('button', { name: /^app\.sqlite3/i }))[1]);

    await waitFor(() => {
      expect(apiMock.listUserSpaceSqliteTables).toHaveBeenCalledWith(
        'source-ws',
        'app.sqlite3',
        'target-read',
      );
    });

    expect(screen.getByText(/Linked from Reporting/i)).toBeTruthy();
    expect(screen.getByText(/Read only/i)).toBeTruthy();
    expect(screen.queryByText(/Changes affect the Reporting workspace\./i)).toBeNull();
    expect(screen.queryByRole('button', { name: /New table/i })).toBeNull();
    expect(screen.queryByTitle('Import CSV')).toBeNull();
    expect(screen.queryByTitle('Drop table')).toBeNull();
    expect(screen.getByRole('button', { name: /Console/i })).toBeTruthy();
    expect(screen.getByTitle('Export table')).toBeTruthy();

    const databasesListener = eventSourceInstances[0]?.listeners.get('databases');
    expect(databasesListener).toBeTypeOf('function');
    databasesListener?.(
      new MessageEvent('message', {
        data: JSON.stringify({
          ...listResponse,
          databases: [
            { ...owned, table_count: 9 },
            { ...linkedRead, owner_workspace_name: 'Reporting Updated', table_count: 4 },
            linkedWriteMissing,
          ],
        }),
      }),
    );

    await waitFor(() => {
      expect(screen.getByText(/Linked from Reporting Updated/i)).toBeTruthy();
    });
    expect(screen.queryByText(/Linked from Source Workspace/i)).toBeNull();
  });

  it('normalizes legacy owned database summaries for table loading and editing controls', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteDatabases.mockResolvedValue({
      ...listResponse,
      databases: [legacyOwned, linkedRead],
    });
    apiMock.listUserSpaceSqliteTables.mockResolvedValue({
      ...tableListResponse,
      database: legacyOwned,
    });

    renderModal();

    await user.click((await screen.findAllByRole('button', { name: /^app\.sqlite3/i }))[0]);

    await waitFor(() => {
      expect(apiMock.listUserSpaceSqliteTables).toHaveBeenCalledWith(
        'source-ws',
        'app.sqlite3',
        'source-ws',
      );
    });

    expect(screen.getByRole('button', { name: /New table/i })).toBeTruthy();
    expect(screen.getByTitle('Import CSV')).toBeTruthy();
    expect(screen.getByTitle('Drop table')).toBeTruthy();
  });

  it('opens missing linked read-write databases without table loading and exposes initialize and import actions', async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click((await screen.findAllByRole('button', { name: /^app\.sqlite3/i }))[2]);

    await waitFor(() => {
      expect(screen.getByText(/Linked from Operations/i)).toBeTruthy();
    });

    expect(apiMock.listUserSpaceSqliteTables).not.toHaveBeenCalledWith(
      'source-ws',
      'app.sqlite3',
      'target-write',
    );
    expect(screen.getAllByText(/Not initialized/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /Initialize app\.sqlite3/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /Import database/i })).toBeTruthy();
  });
});
