import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type {
  Conversation,
  ConversationMessageWindow,
  ConversationWindowEntry,
  ConversationWindowMetadata,
} from '@/types';

const FIRST_PAGE_LIMIT = 20;
const FIRST_PAGE_TIMEOUT_MS = 10_000;

export interface UseConversationWindowOptions {
  conversationId: string | null;
  enabled: boolean;
  workspaceId?: string;
}

export interface ConversationWindowState {
  metadata: ConversationWindowMetadata | null;
  entries: ConversationWindowEntry[];
  revision: string | null;
  totalMessageCount: number;
  olderCursor: string | null;
  initialLoading: boolean;
  olderLoading: boolean;
  firstPageSettled: boolean;
  initialError: Error | null;
  olderError: Error | null;
  fullConversation: Conversation | null;
  reload: () => Promise<void>;
  loadOlder: () => Promise<void>;
  loadMessage: (index: number) => Promise<void>;
  ensureFullConversation: () => Promise<Conversation>;
  adoptFullConversation: (
    conversation: Conversation,
    options?: { pendingSelection?: boolean },
  ) => void;
  updateMetadata: (metadata: ConversationWindowMetadata) => void;
}

type WindowData = Omit<
  ConversationWindowState,
  | 'reload'
  | 'loadOlder'
  | 'loadMessage'
  | 'ensureFullConversation'
  | 'adoptFullConversation'
  | 'updateMetadata'
>;

const emptyState = (): WindowData => ({
  metadata: null,
  entries: [],
  revision: null,
  totalMessageCount: 0,
  olderCursor: null,
  initialLoading: false,
  olderLoading: false,
  firstPageSettled: false,
  initialError: null,
  olderError: null,
  fullConversation: null,
});

function toError(error: unknown): Error {
  return error instanceof Error ? error : new Error('Unable to load conversation messages');
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function isConflict(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'status' in error && error.status === 409;
}

function mergeEntries(
  existing: ConversationWindowEntry[],
  incoming: ConversationWindowEntry[],
): ConversationWindowEntry[] {
  const byIndex = new Map(existing.map((entry) => [entry.index, entry]));
  for (const entry of incoming) {
    const current = byIndex.get(entry.index);
    if (!current || entry.state === 'ready') byIndex.set(entry.index, entry);
  }
  return [...byIndex.values()].sort((left, right) => left.index - right.index);
}

function metadataFromConversation(conversation: Conversation): ConversationWindowMetadata {
  const { messages: _messages, ...metadata } = conversation;
  return metadata;
}

function yieldPaint(): Promise<void> {
  return new Promise((resolve) => {
    const fallback = window.setTimeout(resolve, 50);
    if (typeof requestAnimationFrame !== 'function') return;
    requestAnimationFrame(() => {
      window.clearTimeout(fallback);
      resolve();
    });
  });
}

function stableMessageKey(message: Conversation['messages'][number], occurrence: number): string {
  if (message.message_id) return message.message_id;
  const source = `${message.role}\u0000${message.timestamp}\u0000${JSON.stringify(message.content)}`;
  let hash = 2166136261;
  for (let index = 0; index < source.length; index += 1) {
    hash = Math.imul(hash ^ source.charCodeAt(index), 16777619);
  }
  return `legacy:${(hash >>> 0).toString(16)}:${occurrence}`;
}

function fullEntries(conversation: Conversation): ConversationWindowEntry[] {
  const occurrences = new Map<string, number>();
  return conversation.messages.map((message, index) => {
    const source = `${message.role}\u0000${message.timestamp}\u0000${JSON.stringify(message.content)}`;
    const occurrence = occurrences.get(source) ?? 0;
    occurrences.set(source, occurrence + 1);
    return {
      index,
      key: stableMessageKey(message, occurrence),
      state: 'ready' as const,
      message,
      preview: null,
    };
  });
}

/** Loads a bounded message window; callers must explicitly request full transcript hydration. */
export function useConversationWindow({
  conversationId,
  enabled,
  workspaceId,
}: UseConversationWindowOptions): ConversationWindowState {
  const [state, setState] = useState<WindowData>(emptyState);
  const generation = useRef(0);
  const current = useRef({ conversationId, workspaceId, enabled });
  const controllers = useRef(new Set<AbortController>());
  const pagePromise = useRef<Promise<void> | null>(null);
  const loadPageRef = useRef<
    ((cursor: string, firstPage: boolean, token?: number) => Promise<void>) | null
  >(null);
  const fullPromise = useRef<Promise<Conversation> | null>(null);
  const messagePromises = useRef(new Map<number, Promise<void>>());
  const retriedConflict = useRef(false);
  const pageTimeouts = useRef(new Set<number>());
  const adoptedSelectionId = useRef<string | null>(null);

  // Keep imperative calls made before effects see the current render's selection.
  current.current = { conversationId, workspaceId, enabled };

  const abortRequests = useCallback(() => {
    controllers.current.forEach((controller) => controller.abort());
    controllers.current.clear();
    pageTimeouts.current.forEach((timeout) => window.clearTimeout(timeout));
    pageTimeouts.current.clear();
    pagePromise.current = null;
    fullPromise.current = null;
    messagePromises.current.clear();
  }, []);

  const invalidate = useCallback(() => {
    generation.current += 1;
    abortRequests();
    adoptedSelectionId.current = null;
    setState(emptyState());
  }, [abortRequests]);

  const applyWindow = useCallback((window: ConversationMessageWindow, replace = false) => {
    if (window.legacy_conversation) {
      const conversation = window.legacy_conversation;
      setState((previous) => ({
        ...previous,
        metadata: metadataFromConversation(conversation),
        entries: window.entries.length ? window.entries : fullEntries(conversation),
        revision: window.revision,
        totalMessageCount: conversation.messages.length,
        olderCursor: null,
        fullConversation: conversation,
      }));
      return;
    }
    setState((previous) => ({
      ...previous,
      metadata:
        previous.metadata?.id === window.conversation.id &&
        Date.parse(previous.metadata.updated_at || '') >
          Date.parse(window.conversation.updated_at || '')
          ? previous.metadata
          : window.conversation,
      entries: replace
        ? mergeEntries([], window.entries)
        : mergeEntries(previous.entries, window.entries),
      revision: window.revision,
      totalMessageCount: window.total_message_count,
      olderCursor: window.has_more ? window.next_cursor : null,
    }));
  }, []);

  const reload = useCallback(
    async (preserveConflictRetry = false): Promise<void> => {
      const snapshot = current.current;
      if (!snapshot.enabled || !snapshot.conversationId) return;
      adoptedSelectionId.current = null;
      const token = ++generation.current;
      abortRequests();
      if (!preserveConflictRetry) retriedConflict.current = false;
      setState({ ...emptyState(), initialLoading: true });
      const controller = new AbortController();
      controllers.current.add(controller);
      try {
        const latest = await api.getConversationLatestExchange(
          snapshot.conversationId,
          snapshot.workspaceId,
          { signal: controller.signal },
        );
        if (token !== generation.current) return;
        applyWindow(latest, true);
        if (latest.legacy_conversation) {
          setState((previous) => ({ ...previous, initialLoading: false, firstPageSettled: true }));
          return;
        }
        setState((previous) => ({ ...previous, initialLoading: false }));
        await yieldPaint();
        if (token !== generation.current) return;
        const cursor = latest.next_cursor;
        if (!cursor) {
          setState((previous) => ({ ...previous, firstPageSettled: true }));
          return;
        }
        await loadPageRef.current?.(cursor, true, token);
      } catch (error) {
        if (token === generation.current && !isAbort(error)) {
          setState((previous) => ({
            ...previous,
            initialLoading: false,
            initialError: toError(error),
            firstPageSettled: true,
          }));
        }
      } finally {
        controllers.current.delete(controller);
      }
    },
    [abortRequests, applyWindow],
  );

  const loadPage = useCallback(
    async (cursor: string, firstPage: boolean, token = generation.current): Promise<void> => {
      if (pagePromise.current) return pagePromise.current;
      const snapshot = current.current;
      if (!snapshot.enabled || !snapshot.conversationId) return;
      const id = snapshot.conversationId;
      const controller = new AbortController();
      controllers.current.add(controller);
      let timeout: number | undefined;
      const request = (async () => {
        setState((previous) =>
          firstPage ? previous : { ...previous, olderLoading: true, olderError: null },
        );
        try {
          const pageRequest = api.getConversationMessageWindow(
            id,
            { cursor, limit: FIRST_PAGE_LIMIT },
            snapshot.workspaceId,
            { signal: controller.signal },
          );
          const abortRequest = new Promise<'aborted'>((resolve) => {
            controller.signal.addEventListener('abort', () => resolve('aborted'), { once: true });
          });
          const timeoutRequest = firstPage
            ? new Promise<'timed_out'>((resolve) => {
                timeout = window.setTimeout(() => resolve('timed_out'), FIRST_PAGE_TIMEOUT_MS);
                pageTimeouts.current.add(timeout);
              })
            : null;
          const page = await Promise.race([
            pageRequest,
            abortRequest,
            ...(timeoutRequest ? [timeoutRequest] : []),
          ]);
          if (page === 'timed_out') {
            controller.abort();
            if (token === generation.current) {
              setState((previous) => ({
                ...previous,
                olderLoading: false,
                firstPageSettled: true,
                olderError: new Error('Loading earlier messages timed out. Retry to continue.'),
              }));
            }
            return;
          }
          if (page === 'aborted') return;
          if (token !== generation.current) return;
          applyWindow(page);
          setState((previous) => ({
            ...previous,
            olderLoading: false,
            firstPageSettled: firstPage ? true : previous.firstPageSettled,
          }));
        } catch (error) {
          if (token !== generation.current || isAbort(error)) return;
          if (isConflict(error) && !retriedConflict.current) {
            retriedConflict.current = true;
            await reload(true);
            return;
          }
          const actionable = isConflict(error)
            ? new Error('Conversation changed; reload messages to continue.')
            : toError(error);
          setState((previous) => ({
            ...previous,
            olderLoading: false,
            firstPageSettled: firstPage ? true : previous.firstPageSettled,
            olderError: actionable,
          }));
        } finally {
          if (timeout !== undefined) {
            window.clearTimeout(timeout);
            pageTimeouts.current.delete(timeout);
          }
          controllers.current.delete(controller);
        }
      })();
      pagePromise.current = request;
      try {
        await request;
      } finally {
        if (pagePromise.current === request) pagePromise.current = null;
      }
    },
    [applyWindow, reload],
  );
  loadPageRef.current = loadPage;

  useEffect(() => {
    if (enabled && conversationId && adoptedSelectionId.current === conversationId) {
      adoptedSelectionId.current = null;
    } else if (enabled && conversationId) void reload();
    else setState(emptyState());
    return () => {
      const next = current.current;
      if (
        adoptedSelectionId.current === next.conversationId &&
        next.enabled &&
        next.workspaceId === workspaceId
      ) {
        return;
      }
      invalidate();
    };
  }, [conversationId, enabled, invalidate, reload, workspaceId]);

  useEffect(() => () => invalidate(), [invalidate]);

  const loadOlder = useCallback(async (): Promise<void> => {
    const cursor = state.olderCursor;
    if (!cursor || state.fullConversation) return;
    return loadPage(cursor, false);
  }, [loadPage, state.fullConversation, state.olderCursor]);

  const loadMessage = useCallback(
    async (index: number): Promise<void> => {
      if (messagePromises.current.has(index)) return messagePromises.current.get(index)!;
      const snapshot = current.current;
      const revision = state.revision;
      if (!snapshot.enabled || !snapshot.conversationId || !revision) return;
      const token = generation.current;
      const controller = new AbortController();
      controllers.current.add(controller);
      const request = api
        .getConversationWindowMessage(
          snapshot.conversationId,
          index,
          revision,
          snapshot.workspaceId,
          { signal: controller.signal },
        )
        .then((entry) => {
          if (token !== generation.current) return;
          setState((previous) => ({
            ...previous,
            entries: mergeEntries(previous.entries, [entry]),
          }));
        })
        .catch(async (error) => {
          if (token !== generation.current || isAbort(error)) return;
          if (isConflict(error) && !retriedConflict.current) {
            retriedConflict.current = true;
            await reload(true);
            return;
          }
          setState((previous) => ({ ...previous, olderError: toError(error) }));
        })
        .finally(() => {
          controllers.current.delete(controller);
          if (messagePromises.current.get(index) === request) messagePromises.current.delete(index);
        });
      messagePromises.current.set(index, request);
      return request;
    },
    [reload, state.revision],
  );

  const adoptFullConversation = useCallback(
    (conversation: Conversation, options?: { pendingSelection?: boolean }): void => {
      const snapshot = current.current;
      const pendingNewSelection =
        options?.pendingSelection === true && snapshot.conversationId !== conversation.id;
      if (
        !snapshot.enabled ||
        (!pendingNewSelection && snapshot.conversationId !== conversation.id)
      ) {
        return;
      }
      generation.current += 1;
      abortRequests();
      adoptedSelectionId.current = pendingNewSelection ? conversation.id : null;
      setState((previous) => ({
        ...previous,
        metadata: metadataFromConversation(conversation),
        entries: fullEntries(conversation),
        totalMessageCount: conversation.messages.length,
        olderCursor: null,
        initialLoading: false,
        olderLoading: false,
        firstPageSettled: true,
        initialError: null,
        olderError: null,
        fullConversation: conversation,
      }));
    },
    [abortRequests],
  );

  const updateMetadata = useCallback((metadata: ConversationWindowMetadata): void => {
    const snapshot = current.current;
    if (!snapshot.enabled || snapshot.conversationId !== metadata.id) return;
    setState((previous) =>
      previous.metadata?.id === metadata.id ? { ...previous, metadata } : previous,
    );
  }, []);

  const ensureFullConversation = useCallback(async (): Promise<Conversation> => {
    if (state.fullConversation) return state.fullConversation;
    if (fullPromise.current) return fullPromise.current;
    const snapshot = current.current;
    if (!snapshot.enabled || !snapshot.conversationId) throw new Error('No conversation selected');
    const token = ++generation.current;
    abortRequests();
    setState((previous) => ({
      ...previous,
      initialLoading: false,
      olderLoading: false,
      firstPageSettled: true,
    }));
    const controller = new AbortController();
    controllers.current.add(controller);
    const request = api
      .getConversation(snapshot.conversationId, snapshot.workspaceId, controller.signal)
      .then((conversation) => {
        if (
          token !== generation.current ||
          current.current.conversationId !== conversation.id ||
          !current.current.enabled
        ) {
          throw new Error('Conversation selection changed');
        }
        adoptFullConversation(conversation);
        return conversation;
      })
      .finally(() => {
        controllers.current.delete(controller);
        if (fullPromise.current === request) fullPromise.current = null;
      });
    fullPromise.current = request;
    return request;
  }, [abortRequests, adoptFullConversation, state.fullConversation]);

  return {
    ...state,
    reload,
    loadOlder,
    loadMessage,
    ensureFullConversation,
    adoptFullConversation,
    updateMetadata,
  };
}
