import { cleanup, render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ServerBackupRestoreSettingsSection } from './ServerBackupRestoreSettingsSection';

const apiMock = vi.hoisted(() => ({
  createServerBackupJob: vi.fn(),
  getServerBackupJob: vi.fn(),
  cancelServerBackupJob: vi.fn(),
  downloadServerBackup: vi.fn(),
  uploadServerBackupArchive: vi.fn(),
  createServerRestoreJob: vi.fn(),
  getServerRestoreJob: vi.fn(),
  commitServerRestoreJob: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

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
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.useRealTimers();
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

    expect(screen.getByRole('radiogroup', { name: 'Backup scope' })).toBeTruthy();
    expect(screen.getByRole('progressbar', { name: 'Backup progress' })).toBeTruthy();
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

  it('defaults restore scope to archive scope and only sends scope_override after explicit admin selection', async () => {
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
      scope_override: undefined,
      skip_migrations: true,
      postgres_data_only: true,
      replace_data: true,
      mirror_local_admin_access: true,
      mirror_local_admin_from: 'backup-admin',
      local_admin_username: 'local-admin',
    });

    await user.click(screen.getByRole('radio', { name: 'Restore database only' }));
    await user.click(screen.getByRole('button', { name: 'Validate Restore Archive' }));

    expect(apiMock.createServerRestoreJob).toHaveBeenLastCalledWith({
      upload_id: 'upload-1',
      password: undefined,
      scope_override: 'database',
      skip_migrations: true,
      postgres_data_only: true,
      replace_data: true,
      mirror_local_admin_access: true,
      mirror_local_admin_from: 'backup-admin',
      local_admin_username: 'local-admin',
    });

    expect(intervals.activeCount()).toBe(1);
    await act(async () => {
      await intervals.runLatest();
    });

    expect(screen.getByText('Legacy embedded key detected')).toBeTruthy();
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
});
