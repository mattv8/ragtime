import type { SqliteHistoryBackup, UserSpaceSnapshot } from '@/types';
import { parseUtcTimestampMs } from './databaseCaptureWindow';

export interface SnapshotDatabaseWindow {
  start: string;
  end?: string;
  backups: SqliteHistoryBackup[];
}

/**
 * Assign ready database captures to the code snapshot immediately preceding
 * them. This is strictly temporal: it does not infer or alter snapshot links.
 */
export function buildSnapshotDatabaseWindows(
  snapshots: UserSpaceSnapshot[],
  backups: SqliteHistoryBackup[],
): Map<string, SnapshotDatabaseWindow> {
  const orderedSnapshots = snapshots
    .map((snapshot) => ({ snapshot, timestamp: parseUtcTimestampMs(snapshot.created_at) }))
    .filter((item) => Number.isFinite(item.timestamp))
    .sort(
      (left, right) =>
        left.timestamp - right.timestamp || left.snapshot.id.localeCompare(right.snapshot.id),
    );

  const windows = new Map<string, SnapshotDatabaseWindow>();
  for (let index = 0; index < orderedSnapshots.length; index += 1) {
    const current = orderedSnapshots[index];
    const next = orderedSnapshots[index + 1];
    windows.set(current.snapshot.id, {
      start: current.snapshot.created_at,
      end: next?.snapshot.created_at,
      backups: [],
    });
  }

  for (const backup of backups) {
    if (backup.status !== 'ready') continue;
    const timestamp = parseUtcTimestampMs(backup.created_at);
    if (!Number.isFinite(timestamp)) continue;

    for (let index = orderedSnapshots.length - 1; index >= 0; index -= 1) {
      const current = orderedSnapshots[index];
      if (timestamp < current.timestamp) continue;
      const next = orderedSnapshots[index + 1];
      if (next && timestamp >= next.timestamp) continue;
      windows.get(current.snapshot.id)?.backups.push(backup);
      break;
    }
  }

  return windows;
}
