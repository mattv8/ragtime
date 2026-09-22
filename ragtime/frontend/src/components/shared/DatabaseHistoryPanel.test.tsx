import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({
  listUserSpaceSqliteHistory: vi.fn(),
  captureUserSpaceSqliteHistory: vi.fn(),
  enqueueUserSpaceSqliteBackup: vi.fn(),
  listUserSpaceSqliteBackupJobs: vi.fn(),
  cancelUserSpaceSqliteBackupJob: vi.fn(),
  downloadUserSpaceSqliteHistory: vi.fn(),
  deleteUserSpaceSqliteHistory: vi.fn(),
  previewUserSpaceSqliteHistory: vi.fn(),
  restoreUserSpaceSqliteHistory: vi.fn(),
  recoverUserSpaceSqliteHistoryMaintenance: vi.fn(),
  subscribeUserSpaceSqliteHistoryEvents: vi.fn(),
}));

vi.mock('@/api/client', () => ({
  api: apiMock,
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public status: number,
    ) {
      super(message);
    }
  },
}));

import { ApiError } from '@/api/client';
import { DatabaseHistoryPanel } from './DatabaseHistoryPanel';

const backup = {
  id: 'backup-1',
  workspace_id: 'ws-1',
  database_name: 'app.sqlite3',
  created_at: '2026-09-17T10:00:00Z',
  trigger: 'snapshot' as const,
  snapshot_id: 'snap-1',
  snapshot_git_commit_hash: 'abc',
  status: 'ready' as const,
  size_bytes: 1024,
  sha256: 'hash',
  error: null,
  can_restore: true,
  can_delete: true,
};

const secondBackup = {
  ...backup,
  id: 'backup-2',
  database_name: 'orders.sqlite3',
};

const captureJob = {
  id: 'job-1',
  workspace_id: 'ws-1',
  requested_by_id: null,
  trigger: 'manual' as const,
  database_names: ['app.sqlite3'],
  snapshot_id: null,
  snapshot_git_commit_hash: null,
  request_key: 'request-1',
  status: 'pending' as const,
  created_at: '2026-09-17T10:00:00Z',
  available_at: '2026-09-17T10:00:00Z',
  started_at: null,
  finished_at: null,
  updated_at: '2026-09-17T10:00:00Z',
  heartbeat_at: null,
  cancel_requested: false,
  completed_databases: 0,
  total_databases: 1,
  backup_ids: [],
  error_message: null,
};

describe('DatabaseHistoryPanel', () => {
  beforeEach(() => {
    Object.values(apiMock).forEach((mock) => mock.mockReset());
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup],
      can_manage: true,
    });
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({ jobs: [] });
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockImplementation(() => ({
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    }));
    apiMock.enqueueUserSpaceSqliteBackup.mockResolvedValue({ job: captureJob });
    apiMock.previewUserSpaceSqliteHistory.mockResolvedValue({
      preview_id: 'preview-1',
      backup_id: backup.id,
      database_name: backup.database_name,
      mode: 'merge',
      conflict_policy: 'keep_current',
      migrations_applied: ['001.sql'],
      warnings: [],
      blockers: [],
      can_apply: true,
      expires_at: '2026-09-17T10:15:00Z',
      tables: [
        {
          name: 'items',
          inserted: 1,
          updated: 0,
          deleted: 0,
          unchanged: 2,
          conflicts: 1,
          conflict_samples: [
            { key: { id: 1 }, current: { name: 'current' }, backup: { name: 'backup' } },
          ],
        },
      ],
    });
    apiMock.restoreUserSpaceSqliteHistory.mockResolvedValue({
      operation_id: 'operation-1',
      restored_backup_id: backup.id,
      safety_backup_id: 'safety-1',
      runtime_stopped: true,
      status: 'completed',
    });
  });
  afterEach(cleanup);

  it('uses exact workspace and snapshot context and completes a merge restore receipt', async () => {
    const user = userEvent.setup();
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        snapshotId="snap-1"
        ownerOrAdmin
        triggerLabel="Snapshot database history"
        hostId="snapshot-snap-1"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Snapshot database history' }));
    await waitFor(() =>
      expect(apiMock.listUserSpaceSqliteHistory).toHaveBeenCalledWith('ws-1', {
        databaseName: undefined,
        snapshotId: 'snap-1',
      }),
    );
    expect(screen.getByText('Exact snapshot snap-1')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    expect(screen.getByText(/Merge can resurrect/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await screen.findByText(/items: 1 inserted/);
    await user.selectOptions(
      screen.getByLabelText(/Table conflict policy \(items\)/),
      'use_backup',
    );
    expect(screen.queryByText('Actual restore preview')).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await user.click(await screen.findByRole('button', { name: 'Confirm restore' }));
    await screen.findByText('Database restored');
    expect(screen.getByText(/Safety backup: safety-1/)).toBeTruthy();
    expect(screen.getByText(/runtime is stopped/i)).toBeTruthy();
  });

  it('does not expose history to a non-owner host', () => {
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin={false} hostId="workspace" />);
    expect(screen.queryByRole('button', { name: 'Database history' })).toBeNull();
  });

  it('starts a fresh restore workflow when another backup is selected after a receipt', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup, secondBackup],
      can_manage: true,
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await user.click(await screen.findByRole('button', { name: 'Confirm restore' }));
    await screen.findByText('Database restored');

    await user.click(screen.getByRole('button', { name: /Restore orders\.sqlite3 backup/ }));

    expect(screen.queryByText('Database restored')).toBeNull();
    expect(screen.getByRole('heading', { name: 'Restore orders.sqlite3' })).toBeTruthy();
  });

  it('uses its host identity to keep history hooks unique and hides merge policies for overwrite', async () => {
    const user = userEvent.setup();
    render(
      <>
        <DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />
        <DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="snapshot-snap-1" />
      </>,
    );

    const triggers = screen.getAllByRole('button', { name: 'Database history' });
    expect(triggers[0].getAttribute('data-history-host')).toBe('workspace');
    expect(triggers[1].getAttribute('data-history-host')).toBe('snapshot-snap-1');
    expect(triggers[0].getAttribute('data-history-panel')).not.toBe(
      triggers[1].getAttribute('data-history-panel'),
    );

    await user.click(triggers[0]);
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.selectOptions(screen.getByLabelText('Mode'), 'overwrite');

    expect(screen.queryByLabelText('Default conflict policy')).toBeNull();
  });

  it('explains that snapshot history was not captured when its exact filter is empty', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [],
      can_manage: true,
    });
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        snapshotId="snap-old"
        ownerOrAdmin
        hostId="snapshot-snap-old"
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Database history' }));

    expect(await screen.findByText(/not captured for this snapshot/i)).toBeTruthy();
  });

  it('shows queued capture jobs while the history request is still pending', async () => {
    const user = userEvent.setup();
    let resolveHistory!: (value: {
      workspace_id: string;
      backups: [];
      can_manage: boolean;
    }) => void;
    apiMock.listUserSpaceSqliteHistory.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveHistory = resolve;
        }),
    );
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({ jobs: [captureJob] });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() =>
      expect(
        screen.getByRole('dialog').querySelector('[data-history-capture-job="job-1"]'),
      ).toBeTruthy(),
    );
    resolveHistory({ workspace_id: 'ws-1', backups: [], can_manage: true });
  });

  it('uses the history event stream for silent coalesced refreshes instead of polling', async () => {
    const user = userEvent.setup();
    const listeners = new Map<string, EventListener>();
    const source = {
      addEventListener: vi.fn((type: string, listener: EventListener) =>
        listeners.set(type, listener),
      ),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    };
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue(source);
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        snapshotId="snap-1"
        ownerOrAdmin
        hostId="workspace"
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() =>
      expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(1),
    );
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledWith('ws-1', {
      databaseName: 'app.sqlite3',
      snapshotId: 'snap-1',
    });
    await waitFor(() => expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledTimes(1));

    listeners.get('history_changed')?.(new Event('history_changed'));
    listeners.get('history_changed')?.(new Event('history_changed'));

    await waitFor(() => expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledTimes(3));
    expect(apiMock.listUserSpaceSqliteHistory).toHaveBeenCalledTimes(3);
  });

  it('closes the stream on context changes and access revocation', async () => {
    const user = userEvent.setup();
    const listeners = new Map<string, EventListener>();
    const firstSource = {
      addEventListener: vi.fn((type: string, listener: EventListener) =>
        listeners.set(type, listener),
      ),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    };
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue(firstSource);
    const { rerender } = render(
      <DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />,
    );

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() =>
      expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(1),
    );
    rerender(<DatabaseHistoryPanel workspaceId="ws-2" ownerOrAdmin hostId="workspace" />);
    expect(firstSource.close).toHaveBeenCalled();
    expect(firstSource.removeEventListener).toHaveBeenCalledWith(
      'history_changed',
      expect.any(Function),
    );
    expect(firstSource.removeEventListener).toHaveBeenCalledWith(
      'access_revoked',
      expect.any(Function),
    );

    await waitFor(() =>
      expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2),
    );
    listeners.get('access_revoked')?.(new Event('access_revoked'));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('rejects an older history response after an event refresh', async () => {
    const user = userEvent.setup();
    const listeners = new Map<string, EventListener>();
    let resolveInitial!: (value: {
      workspace_id: string;
      backups: (typeof backup)[];
      can_manage: boolean;
    }) => void;
    apiMock.listUserSpaceSqliteHistory
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveInitial = resolve;
          }),
      )
      .mockResolvedValueOnce({ workspace_id: 'ws-1', backups: [secondBackup], can_manage: true });
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn((type: string, listener: EventListener) =>
        listeners.set(type, listener),
      ),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() => expect(listeners.get('history_changed')).toBeTruthy());
    listeners.get('history_changed')?.(new Event('history_changed'));
    await within(screen.getByRole('dialog')).findByRole('heading', { name: 'orders.sqlite3' });

    resolveInitial({ workspace_id: 'ws-1', backups: [backup], can_manage: true });
    await Promise.resolve();
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByRole('heading', { name: 'orders.sqlite3' })).toBeTruthy();
    expect(within(dialog).queryByRole('heading', { name: 'app.sqlite3' })).toBeNull();
  });

  it('refreshes loaded history without re-entering the loading state', async () => {
    const user = userEvent.setup();
    const listeners = new Map<string, EventListener>();
    let resolveRefresh!: (value: {
      workspace_id: string;
      backups: (typeof backup)[];
      can_manage: boolean;
    }) => void;
    apiMock.listUserSpaceSqliteHistory
      .mockResolvedValueOnce({ workspace_id: 'ws-1', backups: [backup], can_manage: true })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveRefresh = resolve;
          }),
      );
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn((type: string, listener: EventListener) =>
        listeners.set(type, listener),
      ),
      removeEventListener: vi.fn(),
      close: vi.fn(),
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await within(screen.getByRole('dialog')).findByRole('heading', { name: 'app.sqlite3' });
    listeners.get('history_changed')?.(new Event('history_changed'));
    expect(
      within(screen.getByRole('dialog')).getByRole('heading', { name: 'app.sqlite3' }),
    ).toBeTruthy();
    expect(screen.queryByText('Loading database history…')).toBeNull();

    resolveRefresh({ workspace_id: 'ws-1', backups: [secondBackup], can_manage: true });
    await within(screen.getByRole('dialog')).findByRole('heading', { name: 'orders.sqlite3' });
  });

  it.each(['event responses first', 'initial responses first'])(
    'clears both loading rows when %s race the initial event refresh',
    async (resolutionOrder) => {
      const user = userEvent.setup();
      const listeners = new Map<string, EventListener>();
      let resolveInitialHistory!: (value: {
        workspace_id: string;
        backups: (typeof backup)[];
        can_manage: boolean;
      }) => void;
      let resolveEventHistory!: (value: {
        workspace_id: string;
        backups: (typeof backup)[];
        can_manage: boolean;
      }) => void;
      let resolveInitialJobs!: (value: { jobs: (typeof captureJob)[] }) => void;
      let resolveEventJobs!: (value: { jobs: (typeof captureJob)[] }) => void;
      apiMock.listUserSpaceSqliteHistory
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveInitialHistory = resolve;
            }),
        )
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveEventHistory = resolve;
            }),
        );
      apiMock.listUserSpaceSqliteBackupJobs
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveInitialJobs = resolve;
            }),
        )
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              resolveEventJobs = resolve;
            }),
        );
      apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
        addEventListener: vi.fn((type: string, listener: EventListener) =>
          listeners.set(type, listener),
        ),
        removeEventListener: vi.fn(),
        close: vi.fn(),
      });
      render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

      await user.click(screen.getByRole('button', { name: 'Database history' }));
      await waitFor(() => expect(listeners.get('history_changed')).toBeTruthy());
      listeners.get('history_changed')?.(new Event('history_changed'));

      const resolveInitial = () => {
        resolveInitialHistory({ workspace_id: 'ws-1', backups: [backup], can_manage: true });
        resolveInitialJobs({ jobs: [] });
      };
      const resolveEvent = () => {
        resolveEventHistory({ workspace_id: 'ws-1', backups: [secondBackup], can_manage: true });
        resolveEventJobs({ jobs: [] });
      };
      if (resolutionOrder === 'event responses first') {
        resolveEvent();
        await within(screen.getByRole('dialog')).findByRole('heading', { name: 'orders.sqlite3' });
        resolveInitial();
      } else {
        resolveInitial();
        await Promise.resolve();
        resolveEvent();
      }

      await waitFor(() => {
        expect(screen.queryByText('Loading database history…')).toBeNull();
        expect(screen.queryByText('Loading capture jobs…')).toBeNull();
      });
      expect(
        within(screen.getByRole('dialog')).getByRole('heading', { name: 'orders.sqlite3' }),
      ).toBeTruthy();
    },
  );

  it('orders priority bands and database groups deterministically, and preserves backup actions', async () => {
    const user = userEvent.setup();
    const mixedBackups = [
      {
        ...backup,
        id: 'hourly-new',
        database_name: 'hourly-new.sqlite3',
        created_at: '2026-09-17T12:00:00Z',
        trigger: 'scheduled' as const,
        snapshot_id: null,
      },
      {
        ...backup,
        id: 'snapshot-old',
        database_name: 'snapshot-old.sqlite3',
        created_at: '2026-09-17T08:00:00Z',
      },
      {
        ...backup,
        id: 'manual-new',
        database_name: 'manual-new.sqlite3',
        created_at: '2026-09-17T11:00:00Z',
        trigger: 'manual' as const,
        snapshot_id: null,
      },
      {
        ...backup,
        id: 'restore-safety',
        database_name: 'restore.sqlite3',
        created_at: '2026-09-17T10:30:00Z',
        trigger: 'pre_restore' as const,
        snapshot_id: null,
      },
      {
        ...backup,
        id: 'snapshot-tie-a',
        database_name: 'snapshot-tie-a.sqlite3',
        created_at: '2026-09-17T10:00:00+02:00',
      },
      {
        ...backup,
        id: 'snapshot-tie-y',
        database_name: 'snapshot-tie-y.sqlite3',
        created_at: '2026-09-17T09:00:00Z',
      },
      {
        ...backup,
        id: 'snapshot-tie-z',
        database_name: 'snapshot-tie-z.sqlite3',
        created_at: '2026-09-17T11:00:00+02:00',
      },
    ];
    const originalOrder = mixedBackups.map((item) => item.id);
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: mixedBackups,
      can_manage: true,
    });
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({
      jobs: [
        { ...captureJob, id: 'failed-job', status: 'failed', error_message: 'Capture failed' },
        captureJob,
        {
          ...captureJob,
          id: 'completed-job',
          status: 'completed',
          finished_at: '2026-09-17T12:00:00Z',
          backup_ids: ['hourly-new'],
        },
      ],
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    const dialog = screen.getByRole('dialog');
    await within(dialog).findByRole('heading', { name: 'snapshot-old.sqlite3' });

    const bands = Array.from(dialog.querySelectorAll<HTMLElement>('[data-history-band]'));
    expect(bands.map((band) => band.dataset.historyBand)).toEqual([
      'attention',
      'in-progress',
      'recoverable',
      'activity',
    ]);
    expect(bands[0]?.getAttribute('aria-labelledby')).toMatch(/^db-hist-band-attention-/);
    expect(bands[2]?.getAttribute('aria-labelledby')).toMatch(/^db-hist-band-recoverable-/);

    const groups = Array.from(
      dialog.querySelectorAll<HTMLElement>('[data-history-database-group]'),
    );
    expect(groups.map((group) => group.dataset.historyDatabaseGroup)).toEqual([
      'hourly-new.sqlite3',
      'manual-new.sqlite3',
      'restore.sqlite3',
      'snapshot-old.sqlite3',
      'snapshot-tie-a.sqlite3',
      'snapshot-tie-y.sqlite3',
      'snapshot-tie-z.sqlite3',
    ]);
    expect(groups[0]?.getAttribute('aria-labelledby')).toMatch(/^db-hist-db-hourly-new-sqlite3-/);
    expect(
      Array.from(dialog.querySelectorAll<HTMLElement>('[data-history-backup]')).map(
        (item) => item.dataset.historyBackup,
      ),
    ).toEqual([
      'hourly-new',
      'manual-new',
      'restore-safety',
      'snapshot-old',
      'snapshot-tie-a',
      'snapshot-tie-y',
      'snapshot-tie-z',
    ]);
    expect(dialog.querySelectorAll('[data-history-backup="snapshot-old"]').length).toBe(1);
    expect(
      dialog.querySelector<HTMLElement>('[data-history-backup="snapshot-old"]')?.dataset
        .historyTrigger,
    ).toBe('snapshot');
    for (const id of ['snapshot-old', 'snapshot-tie-a', 'snapshot-tie-y', 'snapshot-tie-z']) {
      expect(
        dialog.querySelector(`[data-history-backup="${id}"] .database-history-trigger-badge`)
          ?.textContent,
      ).toBe('Code snapshot');
    }
    expect(
      dialog.querySelector(
        '.database-history-backup[data-history-backup="manual-new"] .database-history-trigger-badge',
      )?.textContent,
    ).toBe('Manual');
    expect(
      dialog.querySelector(
        '.database-history-backup[data-history-backup="restore-safety"] .database-history-trigger-badge',
      )?.textContent,
    ).toBe('Before restore');
    expect(
      dialog.querySelector(
        '.database-history-backup[data-history-backup="hourly-new"] .database-history-trigger-badge',
      )?.textContent,
    ).toBe('Hourly');
    expect(within(dialog).getAllByText('Snapshot snap-1')).toHaveLength(4);
    expect(mixedBackups.map((item) => item.id)).toEqual(originalOrder);

    await user.click(screen.getByRole('button', { name: 'Delete snapshot-old.sqlite3 backup' }));
    expect(apiMock.deleteUserSpaceSqliteHistory).toHaveBeenCalledWith('ws-1', 'snapshot-old');
  });

  it('omits empty backup groups', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [{ ...backup, id: 'hourly-only', trigger: 'scheduled' as const, snapshot_id: null }],
      can_manage: true,
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await within(screen.getByRole('dialog')).findByRole('heading', { name: 'app.sqlite3' });

    const dialog = screen.getByRole('dialog');
    expect(dialog.querySelectorAll('[data-history-database-group]')).toHaveLength(1);
    expect(dialog.querySelector('[data-history-database-group="app.sqlite3"]')).toBeTruthy();
  });

  it('labels an empty active-job scope as all workspace databases', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({
      jobs: [{ ...captureJob, database_names: [] }],
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));

    const dialog = screen.getByRole('dialog');
    const job = await waitFor(() => {
      const card = dialog.querySelector('[data-history-capture-job="job-1"]');
      expect(card).toBeTruthy();
      return card as HTMLElement;
    });
    expect(within(job).getByText('All workspace databases')).toBeTruthy();
    expect(within(job).queryByText('All databases')).toBeNull();
  });

  it('keeps completed empty scheduled captures in collapsed activity with their no-change outcome', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({
      jobs: [
        {
          ...captureJob,
          id: 'scheduled-no-change',
          trigger: 'scheduled',
          status: 'completed',
          database_names: [],
          finished_at: '2026-09-17T11:00:00Z',
          backup_ids: [],
        },
      ],
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));

    const dialog = screen.getByRole('dialog');
    const activity = await waitFor(() => {
      const band = dialog.querySelector<HTMLElement>('[data-history-band="activity"]');
      expect(band).toBeTruthy();
      return band as HTMLElement;
    });
    const details = activity.querySelector('details');
    expect(details?.open).toBe(false);
    expect(within(activity).getByText('Capture activity (1 run)')).toBeTruthy();
    const row = activity.querySelector<HTMLElement>(
      '[data-history-activity-job="scheduled-no-change"]',
    );
    expect(row).toBeTruthy();
    expect(within(row as HTMLElement).getByText('Hourly backup check')).toBeTruthy();
    expect(within(row as HTMLElement).getByText('All workspace databases')).toBeTruthy();
    expect(within(row as HTMLElement).getByText('No new restore point created')).toBeTruthy();
    // Activity rows are terminal and must not add focusable controls inside the
    // collapsed <details>, otherwise the dialog Tab trap would stall on hidden elements.
    expect(
      (row as HTMLElement).querySelectorAll('button, a, input, select, textarea, [tabindex]'),
    ).toHaveLength(0);
  });

  it('shows older restore points per database only after its toggle is expanded', async () => {
    const user = userEvent.setup();
    const backups = Array.from({ length: 6 }, (_, index) => ({
      ...backup,
      id: `app-${index + 1}`,
      created_at: `2026-09-${String(10 + index).padStart(2, '0')}T10:00:00Z`,
    }));
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups,
      can_manage: true,
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));

    const dialog = screen.getByRole('dialog');
    await waitFor(() => expect(dialog.querySelectorAll('[data-history-backup]').length).toBe(5));
    const showOlder = within(dialog).getByRole('button', { name: 'Show 1 older restore point' });
    expect(showOlder.getAttribute('data-history-show-older')).toBe('app.sqlite3');
    await user.click(showOlder);
    expect(dialog.querySelectorAll('[data-history-backup]')).toHaveLength(6);
    expect(within(dialog).getByRole('button', { name: 'Show fewer restore points' })).toBeTruthy();
  });

  it('surfaces capture-job errors even when there are no active jobs', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteBackupJobs.mockRejectedValue(new Error('jobs unavailable'));
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));

    const dialog = screen.getByRole('dialog');
    expect((await within(dialog).findByRole('alert')).textContent).toContain('jobs unavailable');
    expect(dialog.querySelector('[data-history-band="in-progress"]')).toBeNull();
  });

  it('renders null safety_backup_id truthfully when restore omits it', async () => {
    const user = userEvent.setup();
    apiMock.restoreUserSpaceSqliteHistory.mockResolvedValue({
      operation_id: 'operation-1',
      restored_backup_id: backup.id,
      safety_backup_id: null,
      runtime_stopped: true,
      status: 'completed',
    });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await user.click(await screen.findByRole('button', { name: 'Confirm restore' }));
    await screen.findByText('Database restored');

    // Should show a truthful message when safety_backup_id is null, not a bare "null"
    // Either shows "No safety backup created" or similar message, not the bare string "null"
    const receiptText = screen.getByRole('status').textContent || '';
    expect(receiptText).toContain('Database restored');
    expect(receiptText).not.toContain('null');
  });

  it('discards a stale restore preview after a 409 response', async () => {
    const user = userEvent.setup();
    apiMock.restoreUserSpaceSqliteHistory.mockRejectedValue(new ApiError('stale', 409));
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await user.click(await screen.findByRole('button', { name: 'Confirm restore' }));

    expect(await screen.findByText(/preview is stale/i)).toBeTruthy();
    expect(screen.queryByText('Actual restore preview')).toBeNull();
  });

  it('dismisses dialog with Escape key', async () => {
    const user = userEvent.setup();
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    expect(screen.getByRole('dialog')).toBeTruthy();

    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('enqueues capture-now once while the enqueue request is in flight', async () => {
    const user = userEvent.setup();
    let resolveCapture: ((value: { job: typeof captureJob }) => void) | undefined;
    const capturePromise = new Promise<{ job: typeof captureJob }>((resolve) => {
      resolveCapture = resolve;
    });
    apiMock.enqueueUserSpaceSqliteBackup.mockImplementation(() => capturePromise);

    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        ownerOrAdmin
        hostId="workspace"
      />,
    );

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    const captureButton = screen.getByRole('button', { name: /Capture now/ }) as HTMLButtonElement;

    // First click should be allowed and disable the button
    expect(captureButton.disabled).toBe(false);
    await user.click(captureButton);

    // Button should be disabled after click
    await waitFor(() => {
      expect(captureButton.disabled).toBe(true);
    });

    // Second click should not increment call count while button is disabled
    await user.click(captureButton);

    expect(apiMock.enqueueUserSpaceSqliteBackup).toHaveBeenCalledTimes(1);

    // Resolve the promise and button should be enabled again
    resolveCapture?.({ job: captureJob });
  });

  it('uses a distinct request ID for each deliberate manual capture', async () => {
    const user = userEvent.setup();
    apiMock.enqueueUserSpaceSqliteBackup
      .mockResolvedValueOnce({ job: captureJob })
      .mockResolvedValueOnce({ job: { ...captureJob, id: 'job-2', request_key: 'request-2' } });
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        ownerOrAdmin
        hostId="workspace"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    const captureButton = await screen.findByRole('button', { name: 'Capture now' });
    await user.click(captureButton);
    await user.click(captureButton);
    await waitFor(() => expect(apiMock.enqueueUserSpaceSqliteBackup).toHaveBeenCalledTimes(2));
    const firstRequestId = apiMock.enqueueUserSpaceSqliteBackup.mock.calls[0]?.[2];
    const secondRequestId = apiMock.enqueueUserSpaceSqliteBackup.mock.calls[1]?.[2];
    expect(firstRequestId).toEqual(expect.any(String));
    expect(secondRequestId).toEqual(expect.any(String));
    expect(firstRequestId).not.toBe(secondRequestId);
  });

  it('keeps an accepted capture job when restore selection changes during enqueue', async () => {
    const user = userEvent.setup();
    let resolveEnqueue!: (value: { job: typeof captureJob }) => void;
    apiMock.enqueueUserSpaceSqliteBackup.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveEnqueue = resolve;
        }),
    );
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        ownerOrAdmin
        hostId="workspace"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: 'Capture now' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    resolveEnqueue({ job: captureJob });
    await waitFor(() =>
      expect(
        screen.getByRole('dialog').querySelector('[data-history-capture-job="job-1"]'),
      ).toBeTruthy(),
    );
    expect(
      (screen.getByRole('button', { name: 'Capture now' }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  it('does not let an older poll erase an accepted capture job', async () => {
    const user = userEvent.setup();
    let resolveJobs!: (value: { jobs: [] }) => void;
    apiMock.listUserSpaceSqliteBackupJobs.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveJobs = resolve;
        }),
    );
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        ownerOrAdmin
        hostId="workspace"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() => expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: 'Capture now' }));
    await waitFor(() =>
      expect(
        screen.getByRole('dialog').querySelector('[data-history-capture-job="job-1"]'),
      ).toBeTruthy(),
    );
    resolveJobs({ jobs: [] });
    await Promise.resolve();
    expect(
      screen.getByRole('dialog').querySelector('[data-history-capture-job="job-1"]'),
    ).toBeTruthy();
  });

  it('shows scoped capture jobs and keeps cancel busy per job', async () => {
    const user = userEvent.setup();
    let resolveCancel!: (value: { job: typeof captureJob }) => void;
    apiMock.listUserSpaceSqliteBackupJobs.mockResolvedValue({ jobs: [captureJob] });
    apiMock.cancelUserSpaceSqliteBackupJob.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCancel = resolve;
        }),
    );
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        databaseName="app.sqlite3"
        snapshotId="snap-1"
        ownerOrAdmin
        hostId="snapshot-snap-1"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() =>
      expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledWith('ws-1', {
        databaseName: 'app.sqlite3',
        snapshotId: 'snap-1',
      }),
    );
    const cancel = await screen.findByRole('button', { name: 'Cancel' });
    await user.click(cancel);
    expect(cancel.hasAttribute('disabled')).toBe(true);
    resolveCancel({ job: captureJob });
  });

  it('ignores a capture-jobs response that arrives after the dialog closes', async () => {
    const user = userEvent.setup();
    let resolveJobs!: (value: { jobs: (typeof captureJob)[] }) => void;
    apiMock.listUserSpaceSqliteBackupJobs
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveJobs = resolve;
          }),
      )
      .mockResolvedValue({ jobs: [] });
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() => expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole('button', { name: 'Close database history' }));
    resolveJobs({ jobs: [captureJob] });
    await Promise.resolve();

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await waitFor(() => expect(apiMock.listUserSpaceSqliteBackupJobs).toHaveBeenCalledTimes(2));
    expect(
      screen.getByRole('dialog').querySelector('[data-history-capture-job="job-1"]'),
    ).toBeNull();
  });

  it('prevents duplicate delete requests with local busy state', async () => {
    const user = userEvent.setup();
    let resolveDelete: (() => void) | undefined;
    const deletePromise = new Promise<void>((resolve) => {
      resolveDelete = resolve;
    });
    apiMock.deleteUserSpaceSqliteHistory.mockImplementation(() => deletePromise);

    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    const deleteButton = screen.getByRole('button', {
      name: /Delete app\.sqlite3 backup/,
    }) as HTMLButtonElement;

    // First click should be allowed
    expect(deleteButton.disabled).toBe(false);
    await user.click(deleteButton);

    // Button should be disabled after click
    await waitFor(() => {
      expect(deleteButton.disabled).toBe(true);
    });

    // Second click should not increment call count while button is disabled
    await user.click(deleteButton);

    expect(apiMock.deleteUserSpaceSqliteHistory).toHaveBeenCalledTimes(1);

    // Resolve the promise and button should be enabled again
    resolveDelete?.();
  });

  it('includes table name in merge table conflict policy label', async () => {
    const user = userEvent.setup();
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);

    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.selectOptions(screen.getByLabelText('Mode'), 'merge');
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));

    await screen.findByText(/items: 1 inserted/);

    // Label should include table name when selecting conflict policy
    const label = screen.getByText(/Table conflict policy \(items\)/);
    expect(label).toBeTruthy();
  });

  it('invalidates a delayed preview when its mode, default policy, table policy, backup, or workspace changes', async () => {
    const user = userEvent.setup();
    const backupPreview = {
      preview_id: 'late-preview',
      backup_id: backup.id,
      database_name: backup.database_name,
      mode: 'merge' as const,
      conflict_policy: 'keep_current' as const,
      migrations_applied: [],
      warnings: [],
      blockers: [],
      can_apply: true,
      expires_at: null,
      tables: [
        {
          name: 'items',
          inserted: 0,
          updated: 0,
          deleted: 0,
          unchanged: 0,
          conflicts: 1,
          conflict_samples: [],
        },
      ],
    };
    const delayed = (() => {
      let resolve!: (value: typeof backupPreview) => void;
      const promise = new Promise<typeof backupPreview>((done) => {
        resolve = done;
      });
      return { promise, resolve };
    })();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup, secondBackup],
      can_manage: true,
    });
    apiMock.previewUserSpaceSqliteHistory.mockReturnValueOnce(delayed.promise);
    const { rerender } = render(
      <DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    await user.click(screen.getByRole('button', { name: 'Prepare preview' }));
    await user.selectOptions(screen.getByLabelText('Mode'), 'overwrite');
    delayed.resolve(backupPreview);
    await Promise.resolve();
    expect(screen.queryByText('Actual restore preview')).toBeNull();

    await user.selectOptions(screen.getByLabelText('Mode'), 'merge');
    await user.selectOptions(screen.getByLabelText('Default conflict policy'), 'use_backup');
    await user.click(screen.getByRole('button', { name: /Restore orders\.sqlite3 backup/ }));
    rerender(<DatabaseHistoryPanel workspaceId="ws-2" ownerOrAdmin hostId="workspace" />);
    expect(screen.queryByLabelText('Mode')).toBeNull();
  });

  it('keeps delete state per backup, clears a deleted source, and reports download failures', async () => {
    const user = userEvent.setup();
    let resolveFirstDelete!: () => void;
    let resolveSecondDelete!: () => void;
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup, secondBackup],
      can_manage: true,
    });
    apiMock.deleteUserSpaceSqliteHistory
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            resolveFirstDelete = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            resolveSecondDelete = resolve;
          }),
      );
    apiMock.downloadUserSpaceSqliteHistory.mockRejectedValue(new Error('download unavailable'));
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);
    await user.click(screen.getByRole('button', { name: 'Database history' }));
    await user.click(screen.getByRole('button', { name: /Restore app\.sqlite3 backup/ }));
    const firstDelete = screen.getByRole('button', {
      name: /Delete app\.sqlite3 backup/,
    }) as HTMLButtonElement;
    const secondDelete = screen.getByRole('button', {
      name: /Delete orders\.sqlite3 backup/,
    }) as HTMLButtonElement;
    await user.click(firstDelete);
    await waitFor(() => expect(firstDelete.disabled).toBe(true));
    expect(secondDelete.disabled).toBe(false);
    await user.click(secondDelete);
    await user.click(screen.getByRole('button', { name: /Download app\.sqlite3 backup/ }));
    expect(await screen.findByText('download unavailable')).toBeTruthy();
    resolveFirstDelete();
    resolveSecondDelete();
    await waitFor(() => expect(screen.queryByLabelText('Mode')).toBeNull());
  });

  it('traps focus, restores its trigger, and blocks unsafe actions during active recovery', async () => {
    const user = userEvent.setup();
    let resolveRecovery!: () => void;
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup],
      can_manage: true,
      interrupted_maintenance: {
        state: 'interrupted',
        operation_id: 'op-1',
        detail: 'Interrupted',
        can_complete: true,
        can_abort: true,
      },
    });
    apiMock.recoverUserSpaceSqliteHistoryMaintenance.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveRecovery = resolve;
        }),
    );
    render(<DatabaseHistoryPanel workspaceId="ws-1" ownerOrAdmin hostId="workspace" />);
    const trigger = screen.getByRole('button', { name: 'Database history' });
    await user.click(trigger);
    const close = screen.getByRole('button', { name: 'Close database history' });
    await waitFor(() => expect(document.activeElement).toBe(close));
    close.focus();
    await user.keyboard('{Shift>}{Tab}{/Shift}');
    expect(screen.getByRole('dialog').contains(document.activeElement)).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Complete' }));
    expect(screen.getByText(/maintenance is active/i)).toBeTruthy();
    expect(
      screen.getByRole('button', { name: /Delete app\.sqlite3 backup/ }).hasAttribute('disabled'),
    ).toBe(true);
    expect(
      screen.getByRole('button', { name: 'Close database history' }).hasAttribute('disabled'),
    ).toBe(true);
    resolveRecovery();
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: 'Close database history' }).hasAttribute('disabled'),
      ).toBe(false),
    );
    await user.click(screen.getByRole('button', { name: 'Close database history' }));
    expect(document.activeElement).toBe(trigger);
  });

  it('renders active maintenance without recovery actions and blocks destructive controls', async () => {
    const user = userEvent.setup();
    apiMock.listUserSpaceSqliteHistory.mockResolvedValue({
      workspace_id: 'ws-1',
      backups: [backup],
      can_manage: true,
      interrupted_maintenance: {
        state: 'active',
        operation_id: null,
        detail: 'A restore is still running',
        can_complete: false,
        can_abort: false,
      },
    });
    render(
      <DatabaseHistoryPanel
        workspaceId="ws-1"
        ownerOrAdmin
        databaseName="app.sqlite3"
        hostId="workspace"
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Database history' }));

    expect(await screen.findByText(/Active database maintenance is in progress/i)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Complete' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Abort' })).toBeNull();
    expect(screen.getByRole('button', { name: /Capture now/ }).hasAttribute('disabled')).toBe(true);
    expect(
      screen.getByRole('button', { name: /Delete app\.sqlite3 backup/ }).hasAttribute('disabled'),
    ).toBe(true);
  });
});
