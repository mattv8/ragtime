import { describe, expect, it } from 'vitest';

import { buildSnapshotDatabaseWindows } from './snapshotDatabaseHistory';

function snapshot(id: string, createdAt: string) {
  return {
    id,
    workspace_id: 'workspace',
    branch_id: 'branch',
    branch_name: 'Main',
    is_current: false,
    can_rename: true,
    can_delete: true,
    created_at: createdAt,
    file_count: 1,
  };
}

function backup(id: string, createdAt: string, status: 'ready' | 'failed' = 'ready') {
  return {
    id,
    workspace_id: 'workspace',
    database_name: 'app.sqlite3',
    created_at: createdAt,
    trigger: 'scheduled' as const,
    snapshot_id: null,
    snapshot_git_commit_hash: null,
    status,
    size_bytes: 1,
    sha256: null,
    error: null,
    can_restore: true,
    can_delete: true,
  };
}

describe('buildSnapshotDatabaseWindows', () => {
  it('groups ready captures into preceding snapshot intervals without inferring links', () => {
    const backups = [
      backup('older', '2026-07-14T09:00:00Z'),
      backup('between', '2026-07-14T11:00:00Z'),
      backup('latest', '2026-07-14T13:00:00Z'),
      backup('failed', '2026-07-14T11:30:00Z', 'failed'),
      backup('bad', 'not-a-date'),
    ];
    const windows = buildSnapshotDatabaseWindows(
      [snapshot('second', '2026-07-14T12:00:00Z'), snapshot('first', '2026-07-14T10:00:00Z')],
      backups,
    );

    expect(windows.get('first')).toMatchObject({
      start: '2026-07-14T10:00:00Z',
      end: '2026-07-14T12:00:00Z',
      backups: [backups[1]],
    });
    expect(windows.get('second')).toMatchObject({
      start: '2026-07-14T12:00:00Z',
      end: undefined,
      backups: [backups[2]],
    });
  });

  it('uses a stable id tiebreak and assigns boundary captures to the newer interval', () => {
    const atBoundary = backup('boundary', '2026-07-14T10:00:00Z');
    const windows = buildSnapshotDatabaseWindows(
      [snapshot('z-newer', '2026-07-14T10:00:00Z'), snapshot('a-earlier', '2026-07-14T10:00:00Z')],
      [atBoundary],
    );

    expect(windows.get('a-earlier')?.backups).toEqual([]);
    expect(windows.get('z-newer')?.backups).toEqual([atBoundary]);
  });

  it('fails closed for invalid snapshot timestamps', () => {
    const windows = buildSnapshotDatabaseWindows(
      [snapshot('invalid', 'nope'), snapshot('valid', '2026-07-14T10:00:00Z')],
      [backup('capture', '2026-07-14T11:00:00Z')],
    );

    expect([...windows.keys()]).toEqual(['valid']);
    expect(windows.get('valid')?.backups.map(({ id }) => id)).toEqual(['capture']);
  });
});
