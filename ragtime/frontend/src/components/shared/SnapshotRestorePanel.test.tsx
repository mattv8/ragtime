import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import { StrictMode } from 'react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  SqliteHistoryBackup,
  SqliteHistoryListResponse,
  SqliteHistoryPreview,
  UserSpaceSnapshotTimeline,
} from '@/types';

vi.mock('@/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getUserSpaceSnapshotTimeline: vi.fn(),
      listUserSpaceSqliteHistory: vi.fn(),
      subscribeUserSpaceSqliteHistoryEvents: vi.fn(),
      restoreUserSpaceSnapshot: vi.fn(),
      previewUserSpaceSqliteHistory: vi.fn(),
      restoreUserSpaceSqliteHistory: vi.fn(),
      recoverUserSpaceSqliteHistoryMaintenance: vi.fn(),
      downloadUserSpaceSqliteHistory: vi.fn(),
    },
  };
});

import { api, ApiError } from '@/api/client';
import { SnapshotRestorePanel, type SnapshotRestorePanelProps } from './SnapshotRestorePanel';

const apiMock = vi.mocked(api);

const snapshot = {
  id: 'snap',
  workspace_id: 'ws',
  branch_id: 'branch',
  branch_name: 'main',
  is_current: false,
  can_rename: true,
  can_delete: true,
  created_at: '2026-01-01T00:00:00Z',
  file_count: 3,
};
const timeline: UserSpaceSnapshotTimeline = {
  workspace_id: 'ws',
  current_snapshot_id: null,
  current_branch_id: 'branch',
  has_previous: true,
  has_next: false,
  snapshots: [snapshot],
  branches: [],
};
const backup = (overrides: Partial<SqliteHistoryBackup> = {}): SqliteHistoryBackup => ({
  id: 'b1',
  workspace_id: 'ws',
  database_name: 'app.sqlite3',
  created_at: '2026-01-02T00:00:00Z',
  trigger: 'snapshot',
  snapshot_id: 'snap',
  snapshot_git_commit_hash: null,
  status: 'ready',
  size_bytes: 1,
  sha256: 'abc',
  error: null,
  can_restore: true,
  can_delete: false,
  capture_job_id: null,
  ...overrides,
});
const one = backup();
const two = backup({ id: 'b2', database_name: 'orders.sqlite3' });
const preview = (
  value: SqliteHistoryBackup,
  overrides: Partial<SqliteHistoryPreview> = {},
): SqliteHistoryPreview => ({
  preview_id: `p-${value.id}`,
  backup_id: value.id,
  database_name: value.database_name,
  mode: 'merge',
  conflict_policy: 'keep_current',
  tables: [
    {
      name: 'items',
      inserted: 2,
      updated: 3,
      deleted: 4,
      unchanged: 5,
      conflicts: 1,
      conflict_samples: [],
    },
  ],
  migrations_applied: ['001_init.sql'],
  warnings: ['Merge may restore deleted rows.'],
  blockers: [],
  can_apply: true,
  expires_at: '2026-01-03T00:00:00Z',
  ...overrides,
});
const history = (
  overrides: Partial<SqliteHistoryListResponse> = {},
): SqliteHistoryListResponse => ({
  workspace_id: 'ws',
  backups: [one, two],
  can_manage: true,
  interrupted_maintenance: null,
  ...overrides,
});
const restoreResult = {
  restored_snapshot_id: 'snap',
  file_count: 3,
  current_branch_id: 'branch',
  has_previous: true,
  has_next: false,
};
const databaseResult = (id: string) => ({
  operation_id: `op-${id}`,
  restored_backup_id: id,
  safety_backup_id: `safe-${id}`,
  runtime_stopped: true as const,
  status: 'completed' as const,
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function eventSource(): EventSource {
  return {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    close: vi.fn(),
  } as unknown as EventSource;
}

const defaultProps: SnapshotRestorePanelProps = {
  workspaceId: 'ws',
  snapshotId: 'snap',
  defaultScope: 'code',
  allowCodeRestore: true,
  allowDatabaseRestore: true,
};

function mount(props: Partial<SnapshotRestorePanelProps> = {}) {
  return render(<SnapshotRestorePanel {...defaultProps} {...props} />);
}

async function chooseDatabaseScope(user = userEvent.setup()) {
  mount();
  await user.click(await screen.findByLabelText('Database only'));
  return user;
}

async function prepareDatabases(user = userEvent.setup()) {
  await user.click(await screen.findByRole('button', { name: 'Prepare database previews' }));
  await screen.findByRole('button', { name: 'Confirm database restore' });
  return user;
}

describe('SnapshotRestorePanel', () => {
  beforeEach(() => {
    apiMock.getUserSpaceSnapshotTimeline.mockReset();
    apiMock.listUserSpaceSqliteHistory.mockReset();
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReset();
    apiMock.restoreUserSpaceSnapshot.mockReset();
    apiMock.previewUserSpaceSqliteHistory.mockReset();
    apiMock.restoreUserSpaceSqliteHistory.mockReset();
    apiMock.recoverUserSpaceSqliteHistoryMaintenance.mockReset();
    apiMock.downloadUserSpaceSqliteHistory.mockReset();
    apiMock.getUserSpaceSnapshotTimeline.mockResolvedValue(timeline);
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(history());
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue(eventSource());
    apiMock.restoreUserSpaceSnapshot.mockResolvedValue(restoreResult);
    apiMock.previewUserSpaceSqliteHistory.mockImplementation(async (_workspaceId, id) =>
      preview(id === one.id ? one : two),
    );
    apiMock.restoreUserSpaceSqliteHistory.mockImplementation(async (_workspaceId, previewId) =>
      databaseResult(previewId === 'p-b1' ? one.id : two.id),
    );
    apiMock.recoverUserSpaceSqliteHistoryMaintenance.mockResolvedValue({
      operation_id: 'maintenance',
      status: 'completed',
      runtime_stopped: true,
    });
    apiMock.downloadUserSpaceSqliteHistory.mockResolvedValue(undefined);
  });

  afterEach(cleanup);

  it('loads restore options under React StrictMode effect replay', async () => {
    render(
      <StrictMode>
        <SnapshotRestorePanel {...defaultProps} />
      </StrictMode>,
    );

    expect(await screen.findByLabelText('Code only')).toBeTruthy();
    expect(await screen.findByLabelText('Database only')).toBeTruthy();
  });

  it('announces initial loading and marks the panel busy', async () => {
    const pendingTimeline = deferred<UserSpaceSnapshotTimeline>();
    apiMock.getUserSpaceSnapshotTimeline.mockReturnValueOnce(pendingTimeline.promise);
    mount();

    expect(screen.getByRole('status').textContent).toMatch(/loading restore options/i);
    expect(
      document.querySelector('[data-snapshot-restore-panel="snap"]')?.getAttribute('aria-busy'),
    ).toBe('true');
    pendingTimeline.resolve(timeline);
    expect(await screen.findByLabelText('Code only')).toBeTruthy();
  });

  it('announces preparing and applying phases with their semantic phase hook', async () => {
    const user = await chooseDatabaseScope();
    const pendingPreview = deferred<SqliteHistoryPreview>();
    apiMock.previewUserSpaceSqliteHistory.mockReturnValueOnce(pendingPreview.promise);
    await user.click(screen.getByRole('button', { name: 'Prepare database previews' }));

    const preparing = await screen.findByText(/preparing database previews/i);
    expect(preparing.getAttribute('data-snapshot-restore-phase')).toBe('preparing');
    expect(
      document.querySelector('[data-snapshot-restore-panel="snap"]')?.getAttribute('aria-busy'),
    ).toBe('true');
    pendingPreview.resolve(preview(one));
    await screen.findByRole('button', { name: 'Confirm database restore' });

    const pendingRestore = deferred<ReturnType<typeof databaseResult>>();
    apiMock.restoreUserSpaceSqliteHistory.mockReturnValueOnce(pendingRestore.promise);
    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));
    const applying = await screen.findByText(/applying database restore/i);
    expect(applying.getAttribute('data-snapshot-restore-phase')).toBe('applying');
    pendingRestore.resolve(databaseResult(one.id));
  });

  it('announces restoring-code and recovery phases while the panel is busy', async () => {
    const user = userEvent.setup();
    const pendingCodeRestore = deferred<typeof restoreResult>();
    apiMock.restoreUserSpaceSnapshot.mockReturnValueOnce(pendingCodeRestore.promise);
    mount();
    await user.click(await screen.findByRole('button', { name: 'Restore code' }));

    const restoring = await screen.findByText(/restoring code snapshot/i);
    expect(restoring.getAttribute('data-snapshot-restore-phase')).toBe('restoring-code');
    expect(
      document.querySelector('[data-snapshot-restore-panel="snap"]')?.getAttribute('aria-busy'),
    ).toBe('true');
    pendingCodeRestore.resolve(restoreResult);
    await screen.findByText(/code restored/i);

    const pendingRecovery = deferred<{
      operation_id: string;
      status: string;
      runtime_stopped: boolean;
    }>();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(
      history({
        interrupted_maintenance: {
          state: 'invalid',
          operation_id: 'maintenance',
          can_complete: true,
          can_abort: true,
          detail: 'Runtime stopped.',
        },
      }),
    );
    apiMock.recoverUserSpaceSqliteHistoryMaintenance.mockReturnValueOnce(pendingRecovery.promise);
    mount({ defaultScope: 'database' });
    await user.click(await screen.findByRole('button', { name: 'Complete maintenance' }));

    const recovering = await screen.findByText(/recovering database maintenance/i);
    expect(recovering.getAttribute('data-snapshot-restore-phase')).toBe('recovering');
    expect(
      document
        .querySelectorAll('[data-snapshot-restore-panel="snap"]')[1]
        ?.getAttribute('aria-busy'),
    ).toBe('true');
    pendingRecovery.resolve({
      operation_id: 'maintenance',
      status: 'completed',
      runtime_stopped: true,
    });
  });

  it('uses only ready exact-snapshot backups and preselects only the initial database', async () => {
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(
      history({
        backups: [
          one,
          two,
          backup({ id: 'newer-wrong', created_at: '2026-02-01T00:00:00Z', snapshot_id: 'other' }),
          backup({ id: 'failed', database_name: 'failed.sqlite3', status: 'failed' }),
        ],
      }),
    );
    mount({ initialBackupId: two.id, defaultScope: 'database' });

    expect(((await screen.findByLabelText(/orders.sqlite3/)) as HTMLInputElement).checked).toBe(
      true,
    );
    expect((screen.getByLabelText(/app.sqlite3/) as HTMLInputElement).checked).toBe(false);
    expect(screen.queryByText(/failed.sqlite3/)).toBeNull();
    expect(screen.queryByText(/newer-wrong/)).toBeNull();
  });

  it('falls back from database scope to code-only when no exact ready backup exists', async () => {
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(
      history({ backups: [backup({ snapshot_id: 'other' })] }),
    );
    mount({ defaultScope: 'database' });

    expect(((await screen.findByLabelText('Code only')) as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByLabelText('Database only')).toBeNull();
    expect(screen.getByRole('status').textContent).toMatch(/no associated database restore point/i);
  });

  it('falls back from deleted code to database-only and never offers code restore', async () => {
    apiMock.getUserSpaceSnapshotTimeline.mockResolvedValue({ ...timeline, snapshots: [] });
    mount();

    expect(((await screen.findByLabelText('Database only')) as HTMLInputElement).checked).toBe(
      true,
    );
    expect(screen.queryByLabelText('Code only')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Restore code' })).toBeNull();
  });

  it('does not read privileged database history when database restore is disallowed', async () => {
    mount({ allowDatabaseRestore: false });

    await screen.findByLabelText('Code only');
    expect(apiMock.listUserSpaceSqliteHistory).not.toHaveBeenCalled();
  });

  it('uses unique radio groups for multiple mounted panels', async () => {
    render(
      <>
        <SnapshotRestorePanel {...defaultProps} />
        <SnapshotRestorePanel {...defaultProps} />
      </>,
    );

    const codeRadios = await screen.findAllByLabelText('Code only');
    expect((codeRadios[0] as HTMLInputElement).name).not.toBe(
      (codeRadios[1] as HTMLInputElement).name,
    );
  });

  it('keeps one busy sequence across code restore, callback refresh, history preflight, and previews', async () => {
    const user = userEvent.setup();
    const refresh = deferred<void>();
    const busy = vi.fn();
    mount({ onCodeRestored: () => refresh.promise, onBusyChange: busy });
    await user.click(await screen.findByLabelText('Code and database'));
    await user.click(screen.getByRole('button', { name: /Restore code and prepare/ }));
    await waitFor(() => expect(apiMock.restoreUserSpaceSnapshot).toHaveBeenCalledOnce());

    expect(apiMock.previewUserSpaceSqliteHistory).not.toHaveBeenCalled();
    expect(busy).toHaveBeenLastCalledWith(true);
    refresh.resolve();
    await waitFor(() => expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledTimes(2));
    expect(busy.mock.calls.map(([value]) => value)).toEqual([false, true, false]);
  });

  it('retains code success and permits database preparation retry without replaying code', async () => {
    const user = userEvent.setup();
    apiMock.previewUserSpaceSqliteHistory.mockRejectedValueOnce(new Error('quota exceeded'));
    mount();
    await user.click(await screen.findByLabelText('Code and database'));
    await user.click(screen.getByRole('button', { name: /Restore code and prepare/ }));
    await screen.findByText(/quota exceeded/i);

    await user.click(screen.getByRole('button', { name: 'Prepare database previews' }));
    await waitFor(() => expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledTimes(4));
    expect(apiMock.restoreUserSpaceSnapshot).toHaveBeenCalledOnce();
    expect(screen.queryByRole('button', { name: /Restore code and prepare/ })).toBeNull();
  });

  it('keeps code scope available after a database-only preview without claiming code success', async () => {
    const user = await chooseDatabaseScope();
    await prepareDatabases(user);

    await user.click(screen.getByLabelText('Code only'));

    expect((screen.getByLabelText('Code only') as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole('button', { name: 'Restore code' })).toBeTruthy();
    expect(screen.queryByText(/Code restored/i)).toBeNull();
  });

  it('never exposes database preparation after a code-only restore', async () => {
    const user = userEvent.setup();
    mount();
    await user.click(await screen.findByRole('button', { name: 'Restore code' }));
    await screen.findByText(/Code restored/i);

    expect(screen.queryByRole('button', { name: /Restore code/i })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Prepare database previews' })).toBeNull();
  });

  it('makes an uncertain code outcome terminal and prevents replay', async () => {
    const user = userEvent.setup();
    apiMock.restoreUserSpaceSnapshot.mockRejectedValueOnce(new Error('gateway timeout'));
    mount();
    await user.click(await screen.findByRole('button', { name: 'Restore code' }));

    expect((await screen.findByRole('alert')).textContent).toMatch(/outcome is uncertain/i);
    expect(screen.queryByRole('button', { name: 'Restore code' })).toBeNull();
    expect(apiMock.restoreUserSpaceSnapshot).toHaveBeenCalledOnce();
  });

  it('does not restore code when database authority is revoked during its safety preflight', async () => {
    const user = userEvent.setup();
    const pendingHistory = deferred<SqliteHistoryListResponse>();
    const revoke = { current: null as EventListener | null };
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
        if (type === 'access_revoked' && typeof listener === 'function') revoke.current = listener;
      }),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    } as unknown as EventSource);
    mount();
    await screen.findByRole('button', { name: 'Restore code' });
    apiMock.listUserSpaceSqliteHistory.mockReturnValueOnce(pendingHistory.promise);

    await user.click(screen.getByRole('button', { name: 'Restore code' }));
    await waitFor(() => expect(apiMock.listUserSpaceSqliteHistory).toHaveBeenCalledTimes(2));
    revoke.current?.(new Event('access_revoked'));
    pendingHistory.resolve(history());

    await waitFor(() => expect(apiMock.restoreUserSpaceSnapshot).not.toHaveBeenCalled());
    expect(
      (screen.getByRole('button', { name: 'Restore code' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('rechecks maintenance immediately before code-only mutation and blocks when it appears', async () => {
    const user = userEvent.setup();
    mount();
    await screen.findByRole('button', { name: 'Restore code' });
    apiMock.listUserSpaceSqliteHistory.mockResolvedValueOnce(
      history({
        interrupted_maintenance: {
          state: 'active',
          operation_id: null,
          can_complete: false,
          can_abort: false,
          detail: 'Maintenance acquired.',
        },
      }),
    );

    await user.click(screen.getByRole('button', { name: 'Restore code' }));

    expect(apiMock.restoreUserSpaceSnapshot).not.toHaveBeenCalled();
    expect(await screen.findByText('Maintenance acquired.')).toBeTruthy();
  });

  it('allows narrowing but not expanding the approved database set after code succeeds', async () => {
    const user = userEvent.setup();
    apiMock.previewUserSpaceSqliteHistory.mockRejectedValue(new Error('quota exceeded'));
    mount({ initialBackupId: one.id });
    await user.click(await screen.findByLabelText('Code and database'));
    await user.click(screen.getByRole('button', { name: /Restore code and prepare/ }));
    await screen.findByText(/quota exceeded/i);

    const unapproved = screen.getByLabelText(/orders.sqlite3/) as HTMLInputElement;
    expect(unapproved.checked).toBe(false);
    await user.click(unapproved);
    expect(unapproved.checked).toBe(false);

    const approved = screen.getByLabelText(/app.sqlite3/) as HTMLInputElement;
    await user.click(approved);
    expect(approved.checked).toBe(false);
  });

  it('allows interrupted maintenance recovery while blocking every restore action', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(
      history({
        interrupted_maintenance: {
          state: 'invalid',
          operation_id: 'maintenance',
          can_complete: true,
          can_abort: true,
          detail: 'Runtime stopped.',
        },
      }),
    );
    mount({ defaultScope: 'database' });

    const complete = await screen.findByRole('button', { name: 'Complete maintenance' });
    expect((complete as HTMLButtonElement).disabled).toBe(false);
    expect(
      (screen.getByRole('button', { name: 'Prepare database previews' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    await user.click(complete);
    await waitFor(() =>
      expect(apiMock.recoverUserSpaceSqliteHistoryMaintenance).toHaveBeenCalledWith(
        'ws',
        'maintenance',
        'complete',
      ),
    );
  });

  it('shows per-database preview failures so the failed database can be deselected and retried', async () => {
    const user = await chooseDatabaseScope();
    apiMock.previewUserSpaceSqliteHistory.mockImplementation(async (_workspaceId, id) => {
      if (id === two.id) throw new Error('quota exceeded');
      return preview(one);
    });
    await user.click(screen.getByRole('button', { name: 'Prepare database previews' }));

    expect(await screen.findByText(/orders.sqlite3.*quota exceeded/i)).toBeTruthy();
    await user.click(screen.getByLabelText(/orders.sqlite3/));
    apiMock.previewUserSpaceSqliteHistory.mockResolvedValue(preview(one));
    await user.click(screen.getByRole('button', { name: 'Prepare database previews' }));
    expect(
      (
        (await screen.findByRole('button', {
          name: 'Confirm database restore',
        })) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  it('uses overwrite enum and renders destructive warnings, migrations, and table counts', async () => {
    const user = await chooseDatabaseScope();
    await user.selectOptions(screen.getByLabelText('Restore mode'), 'overwrite');
    expect(screen.getByRole('alert').textContent).toMatch(/removes current-only data/i);
    await prepareDatabases(user);

    expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledWith(
      'ws',
      one.id,
      expect.objectContaining({ mode: 'overwrite' }),
    );
    expect(screen.getAllByText(/001_init.sql/)).toHaveLength(2);
    expect(screen.getAllByText(/2 inserted, 3 updated, 4 deleted, 1 conflicts/)).toHaveLength(2);
  });

  it('retains successful receipts, marks remaining work, and cannot apply completed previews again', async () => {
    const user = await chooseDatabaseScope();
    await prepareDatabases(user);
    apiMock.restoreUserSpaceSqliteHistory
      .mockResolvedValueOnce(databaseResult(one.id))
      .mockRejectedValueOnce(new Error('connection lost'));
    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));

    const receipts = await screen.findByRole('status', { name: 'Database restore results' });
    expect(within(receipts).getByText(/app.sqlite3.*restored/i)).toBeTruthy();
    expect(within(receipts).getByText(/orders.sqlite3.*outcome uncertain/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Confirm database restore' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Prepare database previews' })).toBeNull();
  });

  it('retains a completed receipt but stops before the next database after access revocation', async () => {
    const user = userEvent.setup();
    const firstRestore = deferred<ReturnType<typeof databaseResult>>();
    const revoke = { current: null as EventListener | null };
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn((type: string, listener: EventListenerOrEventListenerObject) => {
        if (type === 'access_revoked' && typeof listener === 'function') revoke.current = listener;
      }),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    } as unknown as EventSource);
    mount({ defaultScope: 'database' });
    await prepareDatabases(user);
    apiMock.restoreUserSpaceSqliteHistory
      .mockReturnValueOnce(firstRestore.promise)
      .mockResolvedValueOnce(databaseResult(two.id));

    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));
    await waitFor(() => expect(apiMock.restoreUserSpaceSqliteHistory).toHaveBeenCalledOnce());
    revoke.current?.(new Event('access_revoked'));
    firstRestore.resolve(databaseResult(one.id));

    const results = await screen.findByRole('status', { name: 'Database restore results' });
    expect(within(results).getByText(/app.sqlite3.*restored/i)).toBeTruthy();
    expect(apiMock.restoreUserSpaceSqliteHistory).toHaveBeenCalledOnce();
  });

  it('treats 409 as preview invalidation but treats 5xx as an unknown terminal outcome', async () => {
    const user = await chooseDatabaseScope();
    await prepareDatabases(user);
    apiMock.restoreUserSpaceSqliteHistory.mockRejectedValueOnce(new ApiError('stale', 409));
    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));
    expect(await screen.findByText(/preview is stale/i)).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Prepare database previews' }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);

    await prepareDatabases(user);
    apiMock.restoreUserSpaceSqliteHistory.mockRejectedValueOnce(new ApiError('server', 500));
    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));
    expect(await screen.findByText(/outcome is uncertain/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Prepare database previews' })).toBeNull();
  });

  it('fails closed on fresh-history 403 and does not perform a database request', async () => {
    const user = await chooseDatabaseScope();
    apiMock.listUserSpaceSqliteHistory.mockRejectedValueOnce(new ApiError('forbidden', 403));
    await user.click(screen.getByRole('button', { name: 'Prepare database previews' }));

    await waitFor(() => expect(screen.queryByLabelText('Database only')).toBeNull());
    expect(apiMock.previewUserSpaceSqliteHistory).not.toHaveBeenCalled();
  });

  it('resets busy for a new context and ignores the old action completion', async () => {
    const user = userEvent.setup();
    const oldPreview = deferred<SqliteHistoryPreview>();
    const newPreview = deferred<SqliteHistoryPreview>();
    const busy = vi.fn();
    apiMock.previewUserSpaceSqliteHistory.mockReturnValueOnce(oldPreview.promise);
    const view = mount({ defaultScope: 'database', initialBackupId: one.id, onBusyChange: busy });
    await user.click(await screen.findByRole('button', { name: 'Prepare database previews' }));
    await waitFor(() => expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledOnce());

    apiMock.getUserSpaceSnapshotTimeline.mockResolvedValue({
      ...timeline,
      workspace_id: 'ws-2',
      snapshots: [{ ...snapshot, id: 'snap-2', workspace_id: 'ws-2' }],
    });
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue(
      history({
        workspace_id: 'ws-2',
        backups: [backup({ workspace_id: 'ws-2', snapshot_id: 'snap-2' })],
      }),
    );
    view.rerender(
      <SnapshotRestorePanel
        {...defaultProps}
        workspaceId="ws-2"
        snapshotId="snap-2"
        defaultScope="database"
        initialBackupId={one.id}
        onBusyChange={busy}
      />,
    );
    const newPrepare = await screen.findByRole('button', { name: 'Prepare database previews' });
    expect((newPrepare as HTMLButtonElement).disabled).toBe(false);
    apiMock.previewUserSpaceSqliteHistory.mockReturnValueOnce(newPreview.promise);
    await user.click(newPrepare);
    await waitFor(() => expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledTimes(2));

    oldPreview.resolve(preview(one));
    await waitFor(() => expect(busy).toHaveBeenLastCalledWith(true));
    expect(
      (screen.getByRole('button', { name: 'Prepare database previews' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    newPreview.resolve(preview(backup({ workspace_id: 'ws-2', snapshot_id: 'snap-2' })));
    await waitFor(() => expect(busy).toHaveBeenLastCalledWith(false));
  });

  it('does not continue requests or leak receipts after the context changes mid-await', async () => {
    const user = userEvent.setup();
    const pending = deferred<SqliteHistoryPreview>();
    apiMock.previewUserSpaceSqliteHistory.mockReturnValue(pending.promise);
    const view = mount({ defaultScope: 'database' });
    await user.click(await screen.findByRole('button', { name: 'Prepare database previews' }));
    await waitFor(() => expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledOnce());

    view.rerender(
      <SnapshotRestorePanel {...defaultProps} workspaceId="ws-2" snapshotId="snap-2" />,
    );
    pending.resolve(preview(one));
    await waitFor(() => expect(apiMock.getUserSpaceSnapshotTimeline).toHaveBeenCalledWith('ws-2'));
    expect(apiMock.previewUserSpaceSqliteHistory).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Confirm database restore' })).toBeNull();
  });

  it('reports callback refresh failure separately and does not replay a successful restore', async () => {
    const user = userEvent.setup();
    mount({ onCodeRestored: async () => Promise.reject(new Error('refresh unavailable')) });
    await user.click(await screen.findByRole('button', { name: 'Restore code' }));

    expect((await screen.findByRole('alert')).textContent).toMatch(
      /restore succeeded.*refresh failed/i,
    );
    expect(apiMock.restoreUserSpaceSnapshot).toHaveBeenCalledOnce();
    expect(screen.queryByRole('button', { name: 'Restore code' })).toBeNull();
  });

  it('reports safety-download failures instead of silently swallowing them', async () => {
    const user = await chooseDatabaseScope();
    await prepareDatabases(user);
    await user.click(screen.getByRole('button', { name: 'Confirm database restore' }));
    apiMock.downloadUserSpaceSqliteHistory.mockRejectedValueOnce(new Error('download denied'));
    await user.click(
      await screen.findByRole('button', { name: /Download app.sqlite3 safety backup/ }),
    );

    expect((await screen.findByRole('alert')).textContent).toMatch(/download denied/i);
  });
});
