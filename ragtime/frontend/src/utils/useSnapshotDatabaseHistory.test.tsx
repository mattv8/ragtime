import { act, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({
  listUserSpaceSqliteHistory: vi.fn(),
  subscribeUserSpaceSqliteHistoryEvents: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: apiMock,
  ApiError: class ApiError extends Error {
    status?: number;
  },
}));

import { useSnapshotDatabaseHistory } from './useSnapshotDatabaseHistory';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function Catalog({
  workspaceId,
  ownerOrAdmin = true,
}: {
  workspaceId: string | null;
  ownerOrAdmin?: boolean;
}) {
  const backups = useSnapshotDatabaseHistory(workspaceId, true, ownerOrAdmin);
  return <output>{backups.map(({ id }) => id).join(',')}</output>;
}

const READY_BACKUP = {
  id: 'backup',
  workspace_id: 'workspace',
  database_name: 'app.sqlite3',
  created_at: '2026-07-14T10:00:00Z',
  trigger: 'scheduled' as const,
  snapshot_id: null,
  snapshot_git_commit_hash: null,
  status: 'ready' as const,
  size_bytes: 1,
  sha256: null,
  error: null,
  can_restore: true,
  can_delete: true,
};

describe('useSnapshotDatabaseHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('does not apply an old workspace response after the user loses access', async () => {
    const first = deferred<{
      workspace_id: string;
      can_manage: boolean;
      backups: (typeof READY_BACKUP)[];
    }>();
    apiMock.listUserSpaceSqliteHistory.mockReturnValueOnce(first.promise);
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
    });

    const view = render(<Catalog workspaceId="workspace" />);
    view.rerender(<Catalog workspaceId="workspace" ownerOrAdmin={false} />);

    await act(async () => {
      first.resolve({ workspace_id: 'workspace', can_manage: true, backups: [READY_BACKUP] });
      await first.promise;
    });

    expect(screen.getByRole('status').textContent).toBe('');
  });

  it('does not request a catalog for users without owner/admin access', () => {
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
    });

    render(<Catalog workspaceId="workspace" ownerOrAdmin={false} />);

    expect(apiMock.listUserSpaceSqliteHistory).not.toHaveBeenCalled();
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).not.toHaveBeenCalled();
  });
});
