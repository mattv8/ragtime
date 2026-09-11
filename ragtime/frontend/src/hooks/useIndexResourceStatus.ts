import { useEffect, useState } from 'react';
import { api } from '@/api';
import type { IndexResourceStatus } from '@/types';

interface ResourceState {
  data: IndexResourceStatus | null;
  stale: boolean;
}

let sharedState: ResourceState = { data: null, stale: false };
let timer: number | null = null;
let requestInFlight = false;
const listeners = new Set<(state: ResourceState) => void>();

function publish(next: ResourceState): void {
  sharedState = next;
  listeners.forEach((listener) => listener(next));
}

async function refresh(): Promise<void> {
  if (requestInFlight) return;
  requestInFlight = true;
  try {
    const data = await api.getIndexResourceStatus();
    publish({ data, stale: data.stale });
  } catch {
    // Keep the last good snapshot visible, but never pretend it is current.
    publish({ data: sharedState.data, stale: true });
  } finally {
    requestInFlight = false;
  }
}

function subscribe(listener: (state: ResourceState) => void): () => void {
  listeners.add(listener);
  listener(sharedState);
  if (timer === null) {
    void refresh();
    timer = window.setInterval(() => void refresh(), 2000);
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  };
}

/** Shares one visible-only, admin-authorized resource-status poller across consumers. */
export function useIndexResourceStatus(canView: boolean): ResourceState {
  const [state, setState] = useState<ResourceState>(sharedState);
  const [visible, setVisible] = useState(() => document.visibilityState === 'visible');

  useEffect(() => {
    const onVisibilityChange = () => setVisible(document.visibilityState === 'visible');
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  useEffect(() => {
    if (!canView || !visible) {
      return;
    }
    return subscribe(setState);
  }, [canView, visible]);

  return state;
}
