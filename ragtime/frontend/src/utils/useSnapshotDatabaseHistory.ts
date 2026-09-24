import { useEffect, useRef, useState } from 'react';

import { api, ApiError } from '@/api';
import type { SqliteHistoryBackup } from '@/types';
import { subscribeHistoryEvents } from './sqliteHistoryEventBus';

/** Loads the small, ready-only history catalog used to decorate snapshot rows. */
export function useSnapshotDatabaseHistory(
  workspaceId: string | null,
  enabled: boolean,
  ownerOrAdmin: boolean,
): SqliteHistoryBackup[] {
  const [catalog, setCatalog] = useState<{
    workspaceId: string;
    backups: SqliteHistoryBackup[];
  } | null>(null);
  const revisionRef = useRef(0);
  const requestRef = useRef(0);

  useEffect(() => {
    const revision = ++revisionRef.current;
    if (!workspaceId || !enabled || !ownerOrAdmin) {
      setCatalog(null);
      return;
    }

    let active = true;
    let revoked = false;
    let loading = false;
    let refreshQueued = false;
    const clear = () => {
      if (active && revision === revisionRef.current) setCatalog(null);
    };
    const load = async () => {
      if (!active || revoked) return;
      if (loading) {
        refreshQueued = true;
        return;
      }
      loading = true;
      const request = ++requestRef.current;
      try {
        const result = await api.listUserSpaceSqliteHistory(workspaceId);
        if (!active || revision !== revisionRef.current || request !== requestRef.current) return;
        if (!result.can_manage) {
          revoked = true;
          setCatalog(null);
          return;
        }
        setCatalog({
          workspaceId,
          backups: result.backups.filter((backup) => backup.status === 'ready'),
        });
      } catch (error) {
        if (!active || revision !== revisionRef.current || request !== requestRef.current) return;
        if (error instanceof ApiError && error.status === 403) {
          revoked = true;
          clear();
          return;
        }
        clear();
      } finally {
        loading = false;
        if (refreshQueued && active && !revoked) {
          refreshQueued = false;
          void load();
        }
      }
    };

    const unsubscribe = subscribeHistoryEvents(workspaceId, {
      onHistoryChanged: () => void load(),
      onAccessRevoked: () => {
        if (!active) return;
        revoked = true;
        ++requestRef.current;
        clear();
      },
    });
    void load();
    return () => {
      active = false;
      unsubscribe();
    };
  }, [workspaceId, enabled, ownerOrAdmin]);

  return enabled && ownerOrAdmin && catalog?.workspaceId === workspaceId ? catalog.backups : [];
}
