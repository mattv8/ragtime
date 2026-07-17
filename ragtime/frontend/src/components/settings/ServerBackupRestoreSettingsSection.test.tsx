import { cleanup, render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ServerBackupRestoreSettingsSection } from './ServerBackupRestoreSettingsSection';

declare const require: (id: string) => unknown;

const apiMock = vi.hoisted(() => ({
  createServerBackupJob: vi.fn(),
  getActiveServerBackupJobs: vi.fn(),
  getServerBackupJob: vi.fn(),
  cancelServerBackupJob: vi.fn(),
  downloadServerBackup: vi.fn(),
  listServerBackupExports: vi.fn(),
  deleteServerBackupJob: vi.fn(),
  uploadServerBackupArchive: vi.fn(),
  createServerRestoreJob: vi.fn(),
  getServerRestoreJob: vi.fn(),
  commitServerRestoreJob: vi.fn(),
  recoverServerDeploymentEnvironment: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('../DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({
    onDelete,
    buttonText = 'Delete',
    deleting = false,
  }: {
    onDelete: () => void;
    buttonText?: string;
    deleting?: boolean;
  }) => {
    const React = require('react') as typeof import('react');
    const [confirming, setConfirming] = React.useState(false);
    return (
      <button
        type="button"
        onClick={() => {
          if (confirming) {
            onDelete();
            return;
          }
          setConfirming(true);
        }}
      >
        {deleting ? 'Deleting...' : confirming ? 'Confirm?' : buttonText}
      </button>
    );
  },
}));

const PASSWORD_POLICY = {
  export_password_min_length: 12,
  export_password_require_uppercase: true,
  export_password_require_lowercase: true,
  export_password_require_number: true,
  export_password_require_special: true,
} as const;

function renderSection(
  props: Partial<React.ComponentProps<typeof ServerBackupRestoreSettingsSection>> = {},
) {
  return render(
    <ServerBackupRestoreSettingsSection
      open
      onToggle={vi.fn()}
      settings={PASSWORD_POLICY}
      {...props}
    />,
  );
}

function mockIntervals() {
  const callbacks = new Map<number, () => void | Promise<void>>();
  const activeIds: number[] = [];
  let nextId = 1;
  vi.spyOn(window, 'setInterval').mockImplementation(((callback: TimerHandler) => {
    const id = nextId;
    nextId += 1;
    callbacks.set(id, callback as () => void | Promise<void>);
    activeIds.push(id);
    return id as unknown as number;
  }) as typeof window.setInterval);
  vi.spyOn(window, 'clearInterval').mockImplementation((id?: number) => {
    if (typeof id !== 'number') {
      return;
    }
    const index = activeIds.indexOf(id);
    if (index >= 0) {
      activeIds.splice(index, 1);
    }
  });
  return {
    activeCount: () => activeIds.length,
    runLatest: async () => {
      const latestId = activeIds[activeIds.length - 1];
      if (latestId == null) {
        return;
      }
      await callbacks.get(latestId)?.();
    },
  };
}

describe('ServerBackupRestoreSettingsSection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: null,
      restore_job: null,
    });
    apiMock.listServerBackupExports.mockResolvedValue({ exports: [] });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('renders the encrypt backup control as a toggle switch rather than a plain checkbox', () => {
    renderSection();

    const encryptInput = screen.getByLabelText('Encrypt backup archive') as HTMLInputElement;
    expect(encryptInput.type).toBe('checkbox');
    expect(encryptInput.closest('.toggle-switch')).not.toBeNull();
    expect(encryptInput.closest('.checkbox-label')).toBeNull();
  });

  it('starts encrypted and unencrypted backup jobs with backend-shaped requests and accessible controls', async () => {
    const user = userEvent.setup();
    apiMock.createServerBackupJob
      .mockResolvedValueOnce({
        id: 'backup-enc',
        status: 'pending',
        progress: 0,
        message: 'Queued',
      })
      .mockResolvedValueOnce({
        id: 'backup-plain',
        status: 'pending',
        progress: 0,
        message: 'Queued',
      });

    renderSection();

    const backupTab = screen.getByRole('tab', { name: 'Backup' });
    const restoreTab = screen.getByRole('tab', { name: 'Restore' });
    expect(backupTab.getAttribute('aria-selected')).toBe('true');
    expect(restoreTab.getAttribute('aria-selected')).toBe('false');
    expect(screen.getByRole('radiogroup', { name: 'Backup scope' })).toBeTruthy();
    expect(screen.queryByRole('progressbar', { name: 'Backup progress' })).toBe(null);
    expect(screen.queryByLabelText('Restore archive')).toBe(null);
    expect((screen.getByLabelText('Encrypt backup archive') as HTMLInputElement).checked).toBe(
      true,
    );

    await user.type(screen.getByLabelText('Backup password'), 'StrongPass1!');
    await user.type(screen.getByLabelText('Confirm backup password'), 'StrongPass1!');
    await user.click(screen.getByRole('button', { name: 'Start Backup' }));

    expect(apiMock.createServerBackupJob).toHaveBeenNthCalledWith(1, {
      scope: 'full',
      encrypt: true,
      password: 'StrongPass1!',
    });

    await user.click(screen.getByLabelText('Encrypt backup archive'));
    await user.click(screen.getByRole('radio', { name: 'Files only' }));
    await user.click(screen.getByRole('button', { name: 'Start Backup' }));

    expect(apiMock.createServerBackupJob).toHaveBeenNthCalledWith(2, {
      scope: 'files',
      encrypt: false,
      password: undefined,
    });
  });

  it('switches between backup and restore tabs with accessible tab semantics', async () => {
    const user = userEvent.setup();

    renderSection();

    const tablist = screen.getByRole('tablist', { name: 'Backup and restore modes' });
    expect(tablist).toBeTruthy();
    const backupTab = screen.getByRole('tab', { name: 'Backup' });
    const restoreTab = screen.getByRole('tab', { name: 'Restore' });
    expect(backupTab.getAttribute('aria-selected')).toBe('true');
    expect(restoreTab.getAttribute('aria-selected')).toBe('false');
    expect(screen.getByLabelText('Backup password')).toBeTruthy();
    expect(screen.queryByLabelText('Restore archive')).toBe(null);

    await user.click(restoreTab);

    expect(backupTab.getAttribute('aria-selected')).toBe('false');
    expect(restoreTab.getAttribute('aria-selected')).toBe('true');
    expect(screen.getByLabelText('Restore archive')).toBeTruthy();
    expect(screen.queryByLabelText('Backup password')).toBe(null);
    expect(screen.queryByRole('radiogroup', { name: 'Restore scope' })).toBe(null);
  });

  it('keeps backup passwords in the secondary column and renders action rows outside the two-column grid', async () => {
    const user = userEvent.setup();

    const { container } = renderSection();

    const backupPanel = container.querySelector('#server-backup-panel');
    const backupGrid = backupPanel?.querySelector('.server-backup-grid');
    const backupSecondaryColumn = backupPanel?.querySelector('.server-backup-column--secondary');
    const backupPassword = screen.getByLabelText('Backup password');
    const backupConfirm = screen.getByLabelText('Confirm backup password');
    const backupActionRow = backupPanel?.querySelector('.server-backup-actions-row');

    expect(backupSecondaryColumn?.contains(backupPassword)).toBe(true);
    expect(backupSecondaryColumn?.contains(backupConfirm)).toBe(true);
    expect(backupGrid?.contains(backupActionRow ?? null)).toBe(false);
    expect(backupActionRow?.classList.contains('form-actions')).toBe(false);

    await user.click(screen.getByRole('tab', { name: 'Restore' }));

    const restorePanel = container.querySelector('#server-restore-panel');
    const restoreGrid = restorePanel?.querySelector('.server-backup-grid');
    const restoreActionRow = restorePanel?.querySelector('.server-backup-actions-row');

    expect(restoreGrid?.contains(restoreActionRow ?? null)).toBe(false);
    expect(restoreActionRow?.classList.contains('form-actions')).toBe(false);
  });

  it('moves selection and DOM focus across tabs with arrow, home, and end keys', async () => {
    const user = userEvent.setup();

    renderSection();

    const backupTab = screen.getByRole('tab', { name: 'Backup' });
    const restoreTab = screen.getByRole('tab', { name: 'Restore' });

    backupTab.focus();
    expect(document.activeElement).toBe(backupTab);

    await user.keyboard('{ArrowRight}');
    expect(backupTab.getAttribute('aria-selected')).toBe('false');
    expect(restoreTab.getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(restoreTab);

    await user.keyboard('{ArrowLeft}');
    expect(backupTab.getAttribute('aria-selected')).toBe('true');
    expect(restoreTab.getAttribute('aria-selected')).toBe('false');
    expect(document.activeElement).toBe(backupTab);

    await user.keyboard('{End}');
    expect(restoreTab.getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(restoreTab);

    await user.keyboard('{Home}');
    expect(backupTab.getAttribute('aria-selected')).toBe('true');
    expect(document.activeElement).toBe(backupTab);
  });

  it('cancels a running backup, downloads only after completed, and keeps polling until delivered', async () => {
    const intervals = mockIntervals();
    const user = userEvent.setup();
    const onEncryptedArtifactDelivered = vi.fn();
    apiMock.createServerBackupJob.mockResolvedValue({
      id: 'backup-1',
      status: 'pending',
      progress: 0,
      message: 'Queued',
      encrypt: true,
    });
    apiMock.cancelServerBackupJob.mockResolvedValue({
      id: 'backup-1',
      status: 'cancelled',
      progress: 0,
      message: 'Cancelled',
    });
    apiMock.getServerBackupJob
      .mockResolvedValueOnce({
        id: 'backup-1',
        status: 'running',
        progress: 35,
        message: 'Creating archive',
        encrypt: true,
      })
      .mockResolvedValueOnce({
        id: 'backup-1',
        status: 'completed',
        progress: 100,
        message: 'Ready to download',
        encrypt: true,
      })
      .mockResolvedValueOnce({
        id: 'backup-1',
        status: 'delivered',
        progress: 100,
        message: 'Delivered',
        encrypt: true,
        delivered_at: '2026-07-16T00:00:00Z',
      });
    apiMock.downloadServerBackup.mockResolvedValue(undefined);

    renderSection({ onEncryptedArtifactDelivered });

    await user.type(screen.getByLabelText('Backup password'), 'StrongPass1!');
    await user.type(screen.getByLabelText('Confirm backup password'), 'StrongPass1!');
    await user.click(screen.getByRole('button', { name: 'Start Backup' }));

    expect(intervals.activeCount()).toBe(1);
    expect(screen.getByRole('button', { name: 'Cancel Backup' })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Cancel Backup' }));
    expect(apiMock.cancelServerBackupJob).toHaveBeenCalledWith('backup-1');

    await user.click(screen.getByRole('button', { name: 'Start Backup' }));
    await act(async () => {
      await intervals.runLatest();
    });

    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Download Backup' })).toBe(null);

    await act(async () => {
      await intervals.runLatest();
    });

    await user.click(screen.getByRole('button', { name: 'Download Backup' }));
    expect(apiMock.downloadServerBackup).toHaveBeenCalledWith('backup-1');
    await act(async () => {
      await intervals.runLatest();
    });

    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(3);
    expect(onEncryptedArtifactDelivered).toHaveBeenCalledTimes(1);

    const callsBeforeTerminalCheck = apiMock.getServerBackupJob.mock.calls.length;
    if (intervals.activeCount() > 0) {
      await act(async () => {
        await intervals.runLatest();
      });
    }
    expect(apiMock.getServerBackupJob).toHaveBeenCalledTimes(3);
    expect(apiMock.getServerBackupJob.mock.calls.length).toBe(callsBeforeTerminalCheck);
  });

  it('restores an active backup job when the section remounts', async () => {
    const intervals = mockIntervals();
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: {
        id: 'backup-remounted',
        status: 'running',
        progress: 35,
        message: 'Creating backup archive',
        encrypt: true,
      },
      restore_job: null,
    });

    renderSection();

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();
    expect(screen.getByText('Creating backup archive')).toBeTruthy();
    expect(intervals.activeCount()).toBe(1);
  });

  it('renders backup progress details inside the status card instead of a standalone info message', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: {
        id: 'backup-detailed',
        status: 'running',
        progress: 42,
        phase: 'files_collect_start',
        message: 'Copying staged backup files',
        details: {
          processed_items: 12,
          total_items: 40,
          current_item: 'data/workspaces/demo/nested/report.csv',
        },
        encrypt: true,
      },
      restore_job: null,
    });

    const { container } = renderSection();

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    const statusCard = container.querySelector('.server-backup-status-card');
    expect(statusCard?.textContent).toContain('Copying staged backup files');
    expect(statusCard?.textContent).toContain('Copying data files');
    expect(statusCard?.textContent).toContain('42%');
    expect(statusCard?.textContent).toContain('12 / 40 files');
    expect(statusCard?.querySelector('.server-backup-current-item')?.textContent).toContain(
      'data/workspaces/demo/nested/report.csv',
    );
    expect(container.querySelector('.status-message.info')).toBe(null);
  });

  it('keeps the restore progress card visible for ready_for_commit jobs and shows detailed progress', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: null,
      restore_job: {
        id: 'restore-ready',
        status: 'ready_for_commit',
        progress: 40,
        phase: 'validation_complete',
        message: 'Validation complete. Review the manifest and confirm to continue.',
        details: {
          processed_items: 7,
          total_items: 12,
          current_item: 'data/workspaces/demo/backup-meta.json',
        },
        required_confirmation: 'RESTORE restore-ready',
        manifest: {
          format: 'tar.gz',
          version: 1,
          created_at: '2026-07-16T00:00:00Z',
          scope: 'full',
          encrypted: true,
          includes_managed_key: false,
          legacy_embedded_key: false,
        },
      },
    });

    const { container } = renderSection();

    await userEvent.setup().click(screen.getByRole('tab', { name: 'Restore' }));

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    const statusCard = container.querySelector('.server-backup-status-card');
    expect(screen.getByRole('progressbar', { name: 'Restore progress' })).toBeTruthy();
    expect(statusCard?.textContent).toContain('Validated — confirmation required');
    expect(statusCard?.textContent).toContain('40%');
    expect(statusCard?.textContent).toContain('7 / 12 files');
    expect(statusCard?.querySelector('.server-backup-current-item')?.textContent).toContain(
      'data/workspaces/demo/backup-meta.json',
    );
    expect(container.querySelector('.status-message.info')).toBe(null);
  });

  it('renders completed restore jobs with a restore-specific completion phase label', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: null,
      restore_job: {
        id: 'restore-complete',
        status: 'completed',
        phase: 'complete',
        progress: 100,
        message: 'Restore complete',
        manifest: {
          format: 'tar.gz',
          version: 1,
          created_at: '2026-07-16T00:00:00Z',
          scope: 'full',
          encrypted: true,
          includes_managed_key: false,
          legacy_embedded_key: false,
        },
      },
    });

    const { container } = renderSection();

    await userEvent.setup().click(screen.getByRole('tab', { name: 'Restore' }));

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    const statusCard = container.querySelector('.server-backup-status-card');
    expect(statusCard?.textContent).toContain('Restore completed');
    expect(statusCard?.textContent).not.toContain('Ready for download');
  });

  it('shows a reconnected failed backup job error outside the status card without duplicating it inside the card', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: {
        id: 'backup-failed',
        status: 'failed',
        phase: 'archive_build_start',
        progress: 68,
        message: 'Disk full while writing backup archive',
        error: 'Disk full while writing backup archive',
        encrypt: true,
      },
      restore_job: null,
    });

    const { container } = renderSection();

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    const statusCard = container.querySelector('.server-backup-status-card');

    expect(statusCard?.textContent).toContain('Building archive');
    expect(statusCard?.textContent).toContain('68%');
    expect(statusCard?.textContent).not.toContain('Disk full while writing backup archive');
    expect(container.querySelector('.status-message')).toBe(null);
  });

  it('shows a reconnected failed restore job error outside the status card without duplicating it inside the card', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: null,
      restore_job: {
        id: 'restore-failed',
        status: 'failed',
        phase: 'database_restore_start',
        progress: 73,
        message: 'Restore aborted due to checksum mismatch',
        error: 'Restore aborted due to checksum mismatch',
      },
    });

    const { container } = renderSection();

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    const statusCard = container.querySelector('.server-backup-status-card');

    expect(screen.getByRole('tab', { name: 'Restore' }).getAttribute('aria-selected')).toBe('true');
    expect(statusCard?.textContent).toContain('Restoring database');
    expect(statusCard?.textContent).toContain('73%');
    expect(statusCard?.textContent).not.toContain('Restore aborted due to checksum mismatch');
    expect(container.querySelector('.status-message')).toBe(null);
  });

  it('loads previous exports on mount and renders workspace-scm-style metadata and actions', async () => {
    apiMock.listServerBackupExports.mockResolvedValue({
      exports: [
        {
          job_id: 'backup-export-1',
          file_name: 'ragtime-backup-full-20260716T120000Z.ragbak',
          size_bytes: 2048,
          created_at: '2026-07-16T12:00:00Z',
          scope: 'full',
          encrypted: true,
          delivered_at: '2026-07-16T12:05:00Z',
        },
      ],
    });

    renderSection();

    await waitFor(() => {
      expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(1);
    });

    expect(screen.getByText('Previous exports')).toBeTruthy();
    expect(screen.getByText('ragtime-backup-full-20260716T120000Z.ragbak')).toBeTruthy();
    expect(screen.getByText('2.0 KB')).toBeTruthy();
    expect(screen.getByText('Scope: full')).toBeTruthy();
    expect(screen.getByText('Encrypted')).toBeTruthy();
    expect(screen.getByText(/^Delivered:/)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Delete' })).toBeTruthy();
  });

  it('marks previous exports as downloaded for the session', async () => {
    apiMock.listServerBackupExports
      .mockResolvedValueOnce({
        exports: [
          {
            job_id: 'backup-export-1',
            file_name: 'ragtime-backup-files-20260716T120000Z.tar.gz',
            size_bytes: 1024,
            created_at: '2026-07-16T12:00:00Z',
            scope: 'files',
            encrypted: false,
            delivered_at: null,
          },
        ],
      })
      .mockResolvedValueOnce({
        exports: [
          {
            job_id: 'backup-export-1',
            file_name: 'ragtime-backup-files-20260716T120000Z.tar.gz',
            size_bytes: 1024,
            created_at: '2026-07-16T12:00:00Z',
            scope: 'files',
            encrypted: false,
            delivered_at: null,
          },
        ],
      });
    apiMock.downloadServerBackup.mockResolvedValue(undefined);

    renderSection();

    await waitFor(() => {
      expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(1);
    });

    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: 'Download' }));
    expect(apiMock.downloadServerBackup).toHaveBeenCalledWith('backup-export-1');
    expect(screen.getByRole('button', { name: 'Downloaded' })).toBeTruthy();
  });

  it('removes a deleted export row without re-fetching the export list', async () => {
    apiMock.listServerBackupExports.mockResolvedValue({
      exports: [
        {
          job_id: 'backup-export-1',
          file_name: 'ragtime-backup-files-20260716T120000Z.tar.gz',
          size_bytes: 1024,
          created_at: '2026-07-16T12:00:00Z',
          scope: 'files',
          encrypted: false,
          delivered_at: null,
        },
      ],
    });
    apiMock.deleteServerBackupJob.mockResolvedValue({ success: true, job_id: 'backup-export-1' });

    renderSection();

    await waitFor(() => {
      expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(1);
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Delete' }));
    expect(screen.getByRole('button', { name: 'Confirm?' })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Confirm?' }));

    await waitFor(() => {
      expect(apiMock.deleteServerBackupJob).toHaveBeenCalledWith('backup-export-1');
    });
    expect(screen.queryByText('ragtime-backup-files-20260716T120000Z.tar.gz')).toBe(null);
    expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(1);
  });

  it('refreshes previous exports after an active backup job becomes completed', async () => {
    const intervals = mockIntervals();
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: {
        id: 'backup-active',
        status: 'running',
        progress: 80,
        message: 'Writing backup artifact',
        encrypt: true,
      },
      restore_job: null,
    });
    apiMock.listServerBackupExports.mockResolvedValueOnce({ exports: [] }).mockResolvedValueOnce({
      exports: [
        {
          job_id: 'backup-active',
          file_name: 'ragtime-backup-full-20260716T120000Z.ragbak',
          size_bytes: 4096,
          created_at: '2026-07-16T12:00:00Z',
          scope: 'full',
          encrypted: true,
          delivered_at: null,
        },
      ],
    });
    apiMock.getServerBackupJob.mockResolvedValue({
      id: 'backup-active',
      status: 'completed',
      phase: 'complete',
      progress: 100,
      message: 'Ready to download',
      encrypt: true,
    });

    renderSection();

    await waitFor(() => {
      expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(1);
    });

    await act(async () => {
      await intervals.runLatest();
    });

    await waitFor(() => {
      expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByText('Previous exports')).toBeTruthy();
    expect(screen.getByText('ragtime-backup-full-20260716T120000Z.ragbak')).toBeTruthy();
  });

  it('reports backup jobs and clears completed backup cards only after export confirmation without clearing newer jobs', async () => {
    vi.useFakeTimers();
    const onServerBackupJobObserved = vi.fn();

    apiMock.createServerBackupJob
      .mockResolvedValueOnce({
        id: 'backup-complete',
        status: 'completed',
        phase: 'complete',
        progress: 100,
        message: 'Ready to download',
        encrypt: true,
      })
      .mockResolvedValueOnce({
        id: 'backup-newer',
        status: 'running',
        phase: 'archive_build_start',
        progress: 50,
        message: 'Building archive',
        encrypt: true,
      });
    apiMock.downloadServerBackup.mockResolvedValue(undefined);
    apiMock.getServerBackupJob.mockResolvedValue({
      id: 'backup-complete',
      status: 'completed',
      phase: 'complete',
      progress: 100,
      message: 'Ready to download',
      encrypt: true,
    });
    apiMock.listServerBackupExports
      .mockResolvedValueOnce({ exports: [] })
      .mockResolvedValueOnce({ exports: [] })
      .mockResolvedValueOnce({
        exports: [
          {
            job_id: 'backup-complete',
            file_name: 'ragtime-backup-full-20260716T120000Z.ragbak',
            size_bytes: 4096,
            created_at: '2026-07-16T12:00:00Z',
            scope: 'full',
            encrypted: true,
            delivered_at: null,
          },
        ],
      });

    renderSection({ onServerBackupJobObserved });

    act(() => {
      (screen.getByLabelText('Encrypt backup archive') as HTMLInputElement).click();
    });
    act(() => {
      screen.getByRole('button', { name: 'Start Backup' }).click();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(onServerBackupJobObserved).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'backup-complete', status: 'completed' }),
    );
    expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(2);
    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();

    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();

    act(() => {
      screen.getByRole('button', { name: 'Download Backup' }).click();
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(apiMock.listServerBackupExports).toHaveBeenCalledTimes(3);

    await act(async () => {
      vi.advanceTimersByTime(2999);
    });
    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();

    act(() => {
      screen.getByRole('button', { name: 'Start Backup' }).click();
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(onServerBackupJobObserved).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ id: 'backup-newer', status: 'running' }),
    );

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();
    expect(screen.getAllByText('Building archive').length).toBeGreaterThan(0);
  });

  it('loads deployment environment values only on demand, copies per-variable values securely, and clears them on later restore actions', async () => {
    const user = userEvent.setup();
    const onServerRestoreJobObserved = vi.fn();
    const onServerOperationError = vi.fn();
    const writeText = vi.fn().mockResolvedValue(undefined);
    const appendedSecretTextareaValues: string[] = [];
    const originalAppendChild = document.body.appendChild.bind(document.body);
    const appendChildSpy = vi.spyOn(document.body, 'appendChild').mockImplementation(((
      node: Node,
    ) => {
      if (node instanceof HTMLTextAreaElement) {
        appendedSecretTextareaValues.push(node.value);
      }
      return originalAppendChild(node);
    }) as typeof document.body.appendChild);

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });

    apiMock.uploadServerBackupArchive
      .mockResolvedValueOnce({
        upload_id: 'upload-1',
        filename: 'backup-1.ragbak',
        size_bytes: 512,
      })
      .mockResolvedValueOnce({
        upload_id: 'upload-2',
        filename: 'backup-2.ragbak',
        size_bytes: 512,
      });
    apiMock.createServerRestoreJob
      .mockResolvedValueOnce({
        id: 'restore-1',
        status: 'ready_for_commit',
        progress: 100,
        message: 'Validation complete',
        required_confirmation: 'RESTORE restore-1',
        manifest: {
          format: 'ragbak',
          version: 1,
          created_at: '2026-07-16T00:00:00Z',
          scope: 'full',
          encrypted: true,
          includes_managed_key: true,
          legacy_embedded_key: false,
          deployment_environment_variables: [
            'APP_SECRET',
            'DATABASE_URL',
            'POSTGRES_PASSWORD',
            'MULTILINE_SECRET',
            'EMPTY_SECRET',
          ],
        },
      })
      .mockResolvedValueOnce({
        id: 'restore-2',
        status: 'ready_for_commit',
        progress: 100,
        message: 'Validation complete again',
        required_confirmation: 'RESTORE restore-2',
        manifest: {
          format: 'ragbak',
          version: 1,
          created_at: '2026-07-16T01:00:00Z',
          scope: 'full',
          encrypted: true,
          includes_managed_key: true,
          legacy_embedded_key: false,
          deployment_environment_variables: ['APP_SECRET'],
        },
      });
    apiMock.recoverServerDeploymentEnvironment
      .mockResolvedValueOnce({
        variables: {
          APP_SECRET: 'alpha"quote',
          DATABASE_URL: 'postgres://user:$password@db/ragtime',
          POSTGRES_PASSWORD: 'hunter\\2',
          MULTILINE_SECRET: 'line1\nline2',
          EMPTY_SECRET: '',
        },
        variable_names: [
          'APP_SECRET',
          'DATABASE_URL',
          'POSTGRES_PASSWORD',
          'MULTILINE_SECRET',
          'EMPTY_SECRET',
        ],
        warnings: ['Contains DATABASE_URL or POSTGRES_PASSWORD.'],
      })
      .mockResolvedValueOnce({
        variables: {
          APP_SECRET: 'alpha"quote',
          DATABASE_URL: 'postgres://user:$password@db/ragtime',
          POSTGRES_PASSWORD: 'hunter\\2',
          MULTILINE_SECRET: 'line1\nline2',
          EMPTY_SECRET: '',
        },
        variable_names: [
          'APP_SECRET',
          'DATABASE_URL',
          'POSTGRES_PASSWORD',
          'MULTILINE_SECRET',
          'EMPTY_SECRET',
        ],
        warnings: ['Contains DATABASE_URL or POSTGRES_PASSWORD.'],
      })
      .mockRejectedValueOnce(new Error('Recovery failed'));
    apiMock.commitServerRestoreJob.mockResolvedValue({
      id: 'restore-1',
      status: 'completed',
      phase: 'complete',
      progress: 100,
      message: 'Restore complete',
    });

    renderSection({ onServerRestoreJobObserved, onServerOperationError });

    await user.click(screen.getByRole('tab', { name: 'Restore' }));
    await user.upload(
      screen.getByLabelText('Restore archive'),
      new File(['backup'], 'backup-1.ragbak', { type: 'application/octet-stream' }),
    );
    await user.click(screen.getByRole('button', { name: 'Validate Restore Archive' }));

    expect(onServerRestoreJobObserved).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'restore-1', status: 'ready_for_commit' }),
    );
    expect(apiMock.recoverServerDeploymentEnvironment).not.toHaveBeenCalled();
    expect(screen.getByText('APP_SECRET')).toBeTruthy();
    expect(screen.getAllByText('DATABASE_URL').length).toBeGreaterThan(0);
    expect(screen.getByText('MULTILINE_SECRET')).toBeTruthy();
    expect(screen.getByText('EMPTY_SECRET')).toBeTruthy();
    expect(screen.getByText('5 variables').className).toContain('badge-muted');
    expect(screen.queryByText('alpha"quote')).toBe(null);
    expect(screen.queryByText('postgres://user:$password@db/ragtime')).toBe(null);
    expect(screen.queryByText('hunter\\2')).toBe(null);
    expect(screen.queryByText('line1')).toBe(null);
    expect(screen.queryByRole('button', { name: 'Copy recovered environment variables' })).toBe(
      null,
    );
    expect(screen.queryByRole('button', { name: 'Download .env file' })).toBe(null);
    expect(screen.getAllByLabelText('Cross-deployment warning').length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Do not apply recovered values to a different deployment/i),
    ).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Load values' }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(apiMock.recoverServerDeploymentEnvironment).toHaveBeenCalledWith('restore-1');
    expect(screen.getByRole('button', { name: 'Copy assignment for APP_SECRET' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy value for APP_SECRET' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy assignment for DATABASE_URL' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy value for DATABASE_URL' })).toBeTruthy();
    expect(
      screen.getByRole('button', { name: 'Copy assignment for POSTGRES_PASSWORD' }),
    ).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy value for POSTGRES_PASSWORD' })).toBeTruthy();
    expect(
      screen.getByRole('button', { name: 'Copy assignment for MULTILINE_SECRET' }),
    ).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy value for MULTILINE_SECRET' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy assignment for EMPTY_SECRET' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy value for EMPTY_SECRET' })).toBeTruthy();
    expect(screen.queryByLabelText('Recovered environment variable content')).toBe(null);
    expect(screen.queryByRole('button', { name: 'Copy recovered environment variables' })).toBe(
      null,
    );
    expect(screen.queryByRole('button', { name: 'Download .env file' })).toBe(null);
    expect(screen.queryByRole('button', { name: 'Hide environment variables' })).toBe(null);
    expect(screen.queryByText('alpha"quote')).toBe(null);
    expect(screen.queryByText('postgres://user:$password@db/ragtime')).toBe(null);
    expect(screen.queryByText('hunter\\2')).toBe(null);
    expect(screen.queryByText('line1')).toBe(null);

    await user.click(screen.getByRole('button', { name: 'Copy assignment for APP_SECRET' }));
    await user.click(screen.getByRole('button', { name: 'Copy value for APP_SECRET' }));
    await user.click(screen.getByRole('button', { name: 'Copy assignment for DATABASE_URL' }));
    await user.click(screen.getByRole('button', { name: 'Copy value for DATABASE_URL' }));
    await user.click(screen.getByRole('button', { name: 'Copy assignment for POSTGRES_PASSWORD' }));
    await user.click(screen.getByRole('button', { name: 'Copy value for POSTGRES_PASSWORD' }));
    await user.click(screen.getByRole('button', { name: 'Copy assignment for MULTILINE_SECRET' }));
    await user.click(screen.getByRole('button', { name: 'Copy value for MULTILINE_SECRET' }));
    await user.click(screen.getByRole('button', { name: 'Copy assignment for EMPTY_SECRET' }));
    await user.click(screen.getByRole('button', { name: 'Copy value for EMPTY_SECRET' }));

    expect(writeText.mock.calls.map(([value]) => value)).toEqual([
      'APP_SECRET="alpha\\"quote"',
      'alpha"quote',
      'DATABASE_URL="postgres://user:$password@db/ragtime"',
      'postgres://user:$password@db/ragtime',
      'POSTGRES_PASSWORD="hunter\\\\2"',
      'hunter\\2',
      'MULTILINE_SECRET="line1\\nline2"',
      'line1\nline2',
      'EMPTY_SECRET=""',
      '',
    ]);

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: undefined,
    });
    await user.click(screen.getByRole('button', { name: 'Copy value for APP_SECRET' }));
    expect(onServerOperationError).toHaveBeenCalledWith(
      'Clipboard copy is unavailable in this browser.',
    );
    expect(appendedSecretTextareaValues).toEqual([]);
    expect(screen.queryByText('alpha"quote')).toBe(null);

    await user.click(screen.getByRole('button', { name: 'Clear loaded values' }));
    expect(screen.queryByRole('button', { name: 'Copy assignment for APP_SECRET' })).toBe(null);

    await user.click(screen.getByRole('button', { name: 'Load values' }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getByRole('button', { name: 'Copy assignment for APP_SECRET' })).toBeTruthy();

    await user.upload(
      screen.getByLabelText('Restore archive'),
      new File(['backup'], 'backup-2.ragbak', { type: 'application/octet-stream' }),
    );
    await user.click(screen.getByRole('button', { name: 'Validate Restore Archive' }));

    expect(onServerRestoreJobObserved).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'restore-2', status: 'ready_for_commit' }),
    );
    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.queryByRole('button', { name: 'Copy assignment for APP_SECRET' })).toBe(null);

    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });
    await user.click(screen.getByRole('button', { name: 'Load values' }));
    await act(async () => {
      await Promise.resolve();
    });
    expect(onServerOperationError).toHaveBeenCalledWith('Recovery failed');
    expect(screen.queryByRole('button', { name: 'Copy assignment for APP_SECRET' })).toBe(null);
    expect(screen.queryByText('alpha"quote')).toBe(null);

    await user.type(screen.getByLabelText('Type the confirmation phrase'), 'RESTORE restore-2');
    apiMock.commitServerRestoreJob.mockResolvedValueOnce({
      id: 'restore-2',
      status: 'completed',
      phase: 'complete',
      progress: 100,
      message: 'Restore complete',
    });
    await user.click(screen.getByRole('button', { name: 'Confirm Restore' }));

    expect(onServerRestoreJobObserved).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'restore-2', status: 'completed' }),
    );

    appendChildSpy.mockRestore();
  });

  it('validates restore archives without a scope chooser and shows detected scope plus restore progress', async () => {
    const intervals = mockIntervals();
    const user = userEvent.setup();
    apiMock.uploadServerBackupArchive.mockResolvedValue({
      upload_id: 'upload-1',
      filename: 'legacy-backup.tar.gz',
      size_bytes: 1024,
    });
    apiMock.createServerRestoreJob.mockResolvedValue({
      id: 'restore-1',
      status: 'validating',
      progress: 0,
      message: 'Inspecting backup',
    });
    apiMock.getServerRestoreJob.mockResolvedValueOnce({
      id: 'restore-1',
      status: 'ready_for_commit',
      progress: 100,
      message: 'Validation complete',
      required_confirmation: 'RESTORE restore-1',
      requires_legacy_key_acknowledgement: true,
      manifest: {
        format: 'tar.gz',
        version: 1,
        created_at: '2026-07-16T00:00:00Z',
        scope: 'database',
        encrypted: false,
        includes_managed_key: false,
        legacy_embedded_key: true,
      },
    });
    apiMock.commitServerRestoreJob.mockResolvedValue({
      id: 'restore-1',
      status: 'completed',
      progress: 100,
      message: 'Restore complete',
    });

    renderSection();

    await user.click(screen.getByRole('tab', { name: 'Restore' }));

    expect(screen.queryByRole('radiogroup', { name: 'Restore scope' })).toBe(null);

    await user.upload(
      screen.getByLabelText('Restore archive'),
      new File(['backup'], 'legacy-backup.tar.gz', { type: 'application/gzip' }),
    );
    await user.click(screen.getByLabelText('Skip migrations after restore'));
    await user.click(screen.getByLabelText('Restore PostgreSQL data only'));
    await user.click(screen.getByLabelText('Replace existing server data'));
    await user.click(screen.getByLabelText('Mirror local admin accounts after restore'));
    await user.clear(screen.getByLabelText('Mirror local admin access from'));
    await user.type(screen.getByLabelText('Mirror local admin access from'), 'backup-admin');
    await user.type(screen.getByLabelText('Target local admin username'), 'local-admin');
    await user.click(screen.getByRole('button', { name: 'Validate Restore Archive' }));

    expect(apiMock.createServerRestoreJob).toHaveBeenCalledWith({
      upload_id: 'upload-1',
      password: undefined,
      skip_migrations: true,
      postgres_data_only: true,
      replace_data: true,
      mirror_local_admin_access: true,
      mirror_local_admin_from: 'backup-admin',
      local_admin_username: 'local-admin',
    });

    expect(screen.getByRole('progressbar', { name: 'Restore progress' })).toBeTruthy();
    expect(screen.getByText('Inspecting backup')).toBeTruthy();

    expect(intervals.activeCount()).toBe(1);
    await act(async () => {
      await intervals.runLatest();
    });

    expect(screen.getByText('Legacy embedded key detected')).toBeTruthy();
    const scopeRow = screen.getByText('Scope').closest('div');
    expect(scopeRow?.querySelector('dd')?.textContent).toBe('database');
    const commitButton = screen.getByRole('button', {
      name: 'Confirm Restore',
    }) as HTMLButtonElement;
    expect(commitButton.disabled).toBe(true);

    await user.type(screen.getByLabelText('Type the confirmation phrase'), 'RESTORE restore-1');
    await user.click(
      screen.getByLabelText('I understand this backup contains a legacy embedded encryption key'),
    );
    expect(commitButton.disabled).toBe(false);

    await user.click(commitButton);

    expect(apiMock.commitServerRestoreJob).toHaveBeenCalledWith('restore-1', {
      confirmation_text: 'RESTORE restore-1',
      acknowledge_legacy_key: true,
    });

    expect(apiMock.getServerRestoreJob).toHaveBeenCalledTimes(1);
  });

  it('reports explicit backup action errors through the app callback', async () => {
    const user = userEvent.setup();
    const onServerOperationError = vi.fn();

    renderSection({ onServerOperationError });

    await user.click(screen.getByRole('button', { name: 'Start Backup' }));

    expect(onServerOperationError).toHaveBeenCalledWith(
      'Backup password is required when encryption is enabled.',
    );
  });

  it('uses inline warnings plus direct grid-row wrappers and leaves no status-message elements', async () => {
    apiMock.getActiveServerBackupJobs.mockResolvedValue({
      backup_job: null,
      restore_job: {
        id: 'restore-warnings',
        status: 'ready_for_commit',
        progress: 100,
        phase: 'validation_complete',
        message: 'Validation complete',
        required_confirmation: 'RESTORE restore-warnings',
        restart_required: true,
        restart_state: 'Restart pending',
        manifest: {
          format: 'tar.gz',
          version: 1,
          created_at: '2026-07-16T00:00:00Z',
          scope: 'full',
          encrypted: true,
          includes_managed_key: false,
          legacy_embedded_key: true,
          deployment_environment_variables: ['DATABASE_URL'],
        },
      },
    });

    const { container } = renderSection();

    await userEvent.setup().click(screen.getByRole('tab', { name: 'Restore' }));

    await waitFor(() => {
      expect(apiMock.getActiveServerBackupJobs).toHaveBeenCalled();
    });

    expect(container.querySelector('.status-message')).toBe(null);
    expect(screen.getByText('Legacy embedded key detected').className).toContain(
      'server-backup-inline-warning',
    );
    expect(screen.getByText(/Restart required:/).className).toContain(
      'server-backup-inline-warning',
    );
    expect(screen.getByText(/Do not apply recovered values/i).className).toContain(
      'server-backup-inline-warning',
    );

    const grid = container.querySelector('#server-restore-panel .server-backup-grid');
    const statusRows = Array.from(container.querySelectorAll('.server-backup-status-row'));
    expect(statusRows.length).toBeGreaterThan(0);
    expect(statusRows.every((row) => row.parentElement === grid)).toBe(true);
  });
});
