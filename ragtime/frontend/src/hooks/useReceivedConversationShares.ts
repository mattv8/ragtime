import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type { ReceivedConversationShare } from '@/types';

const PAGE_SIZE = 50;
const REFRESH_INTERVAL_MS = 30_000;

export interface UseReceivedConversationSharesOptions {
  userId: string | null | undefined;
  enabled: boolean;
}

export interface ReceivedConversationSharesState {
  shares: ReceivedConversationShare[];
  loading: boolean;
  error: Error | null;
  refresh: () => Promise<void>;
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error('Unable to load shared conversations');
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

/** Loads a complete, replacement snapshot of selected-user conversation shares. */
export function useReceivedConversationShares({
  userId,
  enabled,
}: UseReceivedConversationSharesOptions): ReceivedConversationSharesState {
  const [snapshot, setSnapshot] = useState<{
    ownerKey: string | null;
    shares: ReceivedConversationShare[];
  }>({ ownerKey: null, shares: [] });
  const [status, setStatus] = useState<{
    ownerKey: string | null;
    loading: boolean;
    error: Error | null;
  }>({ ownerKey: null, loading: false, error: null });
  const current = useRef({ userId, enabled });
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const inFlight = useRef<Promise<void> | null>(null);
  current.current = { userId, enabled };
  const ownerKey = enabled && userId ? userId : null;

  const cancel = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    inFlight.current = null;
    setStatus((previous) => ({ ...previous, loading: false }));
  }, []);

  const refresh = useCallback(async (allowHidden = true): Promise<void> => {
    if (!current.current.enabled || !current.current.userId) return;
    if (!allowHidden && document.visibilityState !== 'visible') return;
    if (inFlight.current) return inFlight.current;

    const requestGeneration = generation.current;
    const requestUserId = current.current.userId;
    const requestController = new AbortController();
    controller.current = requestController;
    setStatus({ ownerKey: requestUserId, loading: true, error: null });

    const request = (async () => {
      try {
        const snapshot: ReceivedConversationShare[] = [];
        const ids = new Set<string>();
        const requestedCursors = new Set<string>();
        let cursorCreatedAt: string | undefined;
        let cursorId: string | undefined;

        while (true) {
          const cursorKey = `${cursorCreatedAt ?? ''}\u0000${cursorId ?? ''}`;
          if (requestedCursors.has(cursorKey)) break;
          requestedCursors.add(cursorKey);
          const page = await api.listReceivedConversationShares(
            { limit: PAGE_SIZE, cursorCreatedAt, cursorId },
            requestController.signal,
          );
          if (
            requestGeneration !== generation.current ||
            !current.current.enabled ||
            current.current.userId !== requestUserId
          ) {
            return;
          }
          for (const item of page) {
            if (!ids.has(item.id)) {
              ids.add(item.id);
              snapshot.push(item);
            }
          }
          if (page.length < PAGE_SIZE) break;

          const last = page[page.length - 1];
          const nextCursorKey = `${last.created_at}\u0000${last.id}`;
          if (requestedCursors.has(nextCursorKey)) {
            throw new Error('Received conversation share pagination cursor did not advance');
          }
          cursorCreatedAt = last.created_at;
          cursorId = last.id;
        }

        if (
          requestGeneration === generation.current &&
          current.current.enabled &&
          current.current.userId === requestUserId
        ) {
          setSnapshot({ ownerKey: requestUserId, shares: snapshot });
          setStatus({ ownerKey: requestUserId, loading: false, error: null });
        }
      } catch (requestError) {
        if (
          requestGeneration !== generation.current ||
          current.current.userId !== requestUserId ||
          isAbort(requestError)
        ) {
          return;
        }
        setStatus({ ownerKey: requestUserId, loading: false, error: asError(requestError) });
      } finally {
        if (requestGeneration === generation.current) {
          controller.current = null;
          inFlight.current = null;
          setStatus((previous) =>
            previous.ownerKey === requestUserId ? { ...previous, loading: false } : previous,
          );
        }
      }
    })();
    inFlight.current = request;
    return request;
  }, []);

  useEffect(() => {
    cancel();
    setSnapshot({ ownerKey: null, shares: [] });
    setStatus({ ownerKey: null, loading: false, error: null });
    if (enabled && userId && document.visibilityState === 'visible') void refresh(false);
    return cancel;
  }, [cancel, enabled, refresh, userId]);

  useEffect(() => {
    if (!enabled || !userId) return;
    const refreshIfVisible = () => {
      if (document.visibilityState === 'visible') void refresh(false);
    };
    const interval = window.setInterval(refreshIfVisible, REFRESH_INTERVAL_MS);
    window.addEventListener('focus', refreshIfVisible);
    document.addEventListener('visibilitychange', refreshIfVisible);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener('focus', refreshIfVisible);
      document.removeEventListener('visibilitychange', refreshIfVisible);
    };
  }, [enabled, refresh, userId]);

  return {
    shares: snapshot.ownerKey === ownerKey ? snapshot.shares : [],
    loading: status.ownerKey === ownerKey ? status.loading : false,
    error: status.ownerKey === ownerKey ? status.error : null,
    refresh: () => refresh(true),
  };
}
