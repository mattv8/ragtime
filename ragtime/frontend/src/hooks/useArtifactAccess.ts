import { useCallback, useEffect, useSyncExternalStore } from 'react';
import { api } from '@/api';
import { ApiError } from '@/api/client';
import { sessionLifecycle } from '@/auth/sessionLifecycle';
import { ArtifactRequestError, artifactStateFromStatus } from '@/utils/modelArtifact';

export type ArtifactAccessState = 'ready' | ReturnType<typeof artifactStateFromStatus>;

type Entry = {
  state: ArtifactAccessState;
  revision: number;
  checked: boolean;
  pending?: Promise<void>;
  listeners: Set<() => void>;
};

const entries = new Map<string, Entry>();

export function clearArtifactAccessCache() {
  for (const [id, entry] of entries) {
    entry.revision += 1;
    entry.checked = false;
    entry.pending = undefined;
    if (entry.listeners.size === 0) entries.delete(id);
    else entry.listeners.forEach((listener) => listener());
  }
}

let lastUserId: string | null = null;
sessionLifecycle.subscribe((event) => {
  if (event.type === 'adopted' || event.type === 'renewed') {
    if (event.userId !== lastUserId) clearArtifactAccessCache();
    lastUserId = event.userId;
  } else if (['logout-finished', 'expired', 'anonymous'].includes(event.type)) {
    lastUserId = null;
    clearArtifactAccessCache();
  }
});

function getEntry(id: string): Entry {
  let entry = entries.get(id);
  if (!entry) {
    entry = { state: 'ready', revision: 0, checked: false, listeners: new Set() };
    entries.set(id, entry);
  }
  return entry;
}

function update(entry: Entry, state: ArtifactAccessState) {
  entry.state = state;
  entry.listeners.forEach((listener) => listener());
}

async function checkAccess(id: string, entry: Entry): Promise<void> {
  if (entry.pending) return entry.pending;
  const revision = entry.revision;
  entry.pending = (async () => {
    try {
      const metadata = await api.getModelArtifact(id);
      if (entry.revision === revision) update(entry, metadata.expired ? 'expired' : 'ready');
    } catch (error) {
      if (entry.revision === revision) {
        update(
          entry,
          error instanceof ApiError || error instanceof ArtifactRequestError
            ? artifactStateFromStatus(error.status)
            : 'error',
        );
      }
    } finally {
      if (entry.revision === revision) {
        entry.checked = entry.state !== 'error';
        entry.pending = undefined;
      }
    }
  })();
  return entry.pending;
}

export function useArtifactAccess(id: string, expiresAt: string) {
  const entry = getEntry(id);
  const subscribe = useCallback(
    (listener: () => void) => {
      entry.listeners.add(listener);
      return () => entry.listeners.delete(listener);
    },
    [entry],
  );
  const snapshot = useSyncExternalStore(
    subscribe,
    () => `${entry.revision}:${entry.state}`,
    () => `${entry.revision}:${entry.state}`,
  );
  const revision = snapshot.split(':')[0];
  const expired = Date.parse(expiresAt) < Date.now();
  useEffect(() => {
    if (!expired && !entry.checked) void checkAccess(id, entry);
  }, [entry, expired, id, revision]);
  return {
    state: expired ? ('expired' as const) : entry.state,
    refresh: () => checkAccess(id, entry),
    setState: (next: ArtifactAccessState) => update(entry, next),
  };
}
