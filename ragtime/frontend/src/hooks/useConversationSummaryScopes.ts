import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type { ConversationSummary } from '@/types';
import {
  cursorFromLastRow,
  hasMorePages,
  type ConversationCursor,
} from '@/utils/conversationLoading';

const PAGE_SIZE = 50;
type OwnerScope = 'self' | 'others';

export interface ConversationSummaryScopeState {
  loading: boolean;
  complete: boolean;
  error: Error | null;
  cursor: ConversationCursor | null;
}

export interface UseConversationSummaryScopesOptions {
  enabled: boolean;
  cutoffIso: string;
  resetKey: string;
  onPage: (rows: ConversationSummary[]) => void;
}

export interface ConversationSummaryScopesState {
  scopeStates: Record<OwnerScope, ConversationSummaryScopeState>;
  loading: boolean;
  retryFailed: () => Promise<void>;
}

const initialScope = (): ConversationSummaryScopeState => ({
  loading: false,
  complete: false,
  error: null,
  cursor: null,
});

const initialScopes = (): Record<OwnerScope, ConversationSummaryScopeState> => ({
  self: initialScope(),
  others: initialScope(),
});

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error('Unable to load conversations');
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function yieldPage(): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, 0));
}

/** Sequential, abortable owner-scope summary paging for the standalone sidebar. */
export function useConversationSummaryScopes({
  enabled,
  cutoffIso,
  resetKey,
  onPage,
}: UseConversationSummaryScopesOptions): ConversationSummaryScopesState {
  const [scopeStates, setScopeStates] = useState(initialScopes);
  const [pending, setPending] = useState(false);
  const scopeRef = useRef(scopeStates);
  const onPageRef = useRef(onPage);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const running = useRef(false);
  const current = useRef({ enabled, cutoffIso, resetKey });
  current.current = { enabled, cutoffIso, resetKey };
  onPageRef.current = onPage;

  const publish = useCallback(
    (
      update: (
        previous: Record<OwnerScope, ConversationSummaryScopeState>,
      ) => Record<OwnerScope, ConversationSummaryScopeState>,
    ) => {
      const next = update(scopeRef.current);
      scopeRef.current = next;
      setScopeStates(next);
    },
    [],
  );

  const cancel = useCallback(() => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    running.current = false;
    setPending(false);
    publish((previous) => ({
      self: { ...previous.self, loading: false },
      others: { ...previous.others, loading: false },
    }));
  }, [publish]);

  const run = useCallback(
    async (onlyFailed = false): Promise<void> => {
      if (!current.current.enabled || running.current) return;
      const scopes: OwnerScope[] = onlyFailed
        ? (['self', 'others'] as const).filter((scope) => {
            const state = scopeRef.current[scope];
            return !state.complete && state.error !== null;
          })
        : ['self', 'others'];
      if (!scopes.length) return;

      const token = generation.current;
      const requestController = new AbortController();
      controller.current = requestController;
      running.current = true;
      setPending(true);
      try {
        for (const scope of scopes) {
          while (true) {
            const snapshot = scopeRef.current[scope];
            if (snapshot.complete) break;
            if (token !== generation.current || !current.current.enabled) return;
            publish((previous) => ({
              ...previous,
              [scope]: { ...previous[scope], loading: true, error: null },
            }));
            try {
              const rows = await api.listConversationSummaries(
                undefined,
                {
                  since: current.current.cutoffIso,
                  limit: PAGE_SIZE,
                  cursorUpdatedAt: snapshot.cursor?.cursorUpdatedAt,
                  cursorId: snapshot.cursor?.cursorId,
                  owner_scope: scope,
                },
                requestController.signal,
              );
              if (token !== generation.current || !current.current.enabled) return;
              onPageRef.current(rows);
              const complete = !hasMorePages(rows, PAGE_SIZE);
              publish((previous) => ({
                ...previous,
                [scope]: {
                  ...previous[scope],
                  loading: false,
                  complete,
                  error: null,
                  cursor: complete
                    ? previous[scope].cursor
                    : cursorFromLastRow(rows[rows.length - 1]),
                },
              }));
              if (complete) break;
              await yieldPage();
            } catch (error) {
              if (token !== generation.current || isAbort(error)) return;
              publish((previous) => ({
                ...previous,
                [scope]: { ...previous[scope], loading: false, error: toError(error) },
              }));
              break;
            }
          }
        }
      } finally {
        if (token === generation.current) {
          running.current = false;
          controller.current = null;
          setPending(false);
        }
      }
    },
    [publish],
  );

  useEffect(() => {
    cancel();
    scopeRef.current = initialScopes();
    setScopeStates(scopeRef.current);
    if (current.current.enabled) void run();
    return cancel;
  }, [cancel, cutoffIso, resetKey, run]);

  useEffect(() => {
    if (enabled && !running.current) void run();
    if (!enabled) cancel();
  }, [cancel, enabled, run]);

  const retryFailed = useCallback(async (): Promise<void> => run(true), [run]);

  return { scopeStates, loading: pending, retryFailed };
}
