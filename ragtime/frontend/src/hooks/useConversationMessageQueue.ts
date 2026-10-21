import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError, api } from '@/api';
import type { QueuedMessageItem, SendMessageRequest } from '@/types';

export class QueueRaceError extends Error {}

export interface QueueState {
  conversationId?: string | null;
  items: QueuedMessageItem[];
  activeTaskId: string | null;
  held: boolean;
  loading: boolean;
  error: Error | null;
}

const EMPTY_STATE: QueueState = {
  items: [],
  activeTaskId: null,
  held: false,
  loading: false,
  error: null,
};

export function useConversationMessageQueue(conversationId: string | null, workspaceId?: string) {
  const [state, setState] = useState<QueueState>(EMPTY_STATE);
  const requestVersion = useRef(0);
  const hasLoadedConversation = useRef<string | null>(null);

  const load = useCallback(async () => {
    const version = ++requestVersion.current;
    if (!conversationId) {
      setState({ ...EMPTY_STATE, conversationId: null });
      return;
    }
    setState((current) => ({
      ...current,
      conversationId,
      loading: hasLoadedConversation.current !== conversationId,
      error: null,
    }));
    try {
      const queue = await api.getConversationMessageQueue(conversationId, workspaceId);
      if (version !== requestVersion.current) return;
      setState({
        conversationId,
        items: queue.items,
        activeTaskId: queue.active_task_id,
        held: queue.held,
        loading: false,
        error: null,
      });
      hasLoadedConversation.current = conversationId;
    } catch (error) {
      if (version !== requestVersion.current) return;
      setState((current) => ({
        ...current,
        loading: false,
        error: error instanceof Error ? error : new Error('Could not load queue'),
      }));
    }
  }, [conversationId, workspaceId]);

  useEffect(() => {
    setState({ ...EMPTY_STATE, conversationId });
    hasLoadedConversation.current = null;
    void load();
    const onFocus = () => void load();
    window.addEventListener('focus', onFocus);
    return () => {
      requestVersion.current += 1;
      window.removeEventListener('focus', onFocus);
    };
  }, [conversationId, load]);

  const enqueue = useCallback(
    async (draft: SendMessageRequest & { segments?: unknown[] }) => {
      if (!conversationId) throw new Error('No conversation selected');
      const response = await api.enqueueConversationMessage(conversationId, draft, workspaceId);
      await load();
      return response;
    },
    [conversationId, load, workspaceId],
  );

  const deleteItem = useCallback(
    async (itemId: string) => {
      if (!conversationId) throw new QueueRaceError('Conversation changed');
      try {
        const item = await api.deleteConversationMessageQueueItem(
          conversationId,
          itemId,
          workspaceId,
        );
        await load();
        return item;
      } catch (error) {
        if (error instanceof ApiError && (error.status === 404 || error.status === 409)) {
          void load();
          throw new QueueRaceError('Queued message has already started');
        }
        throw error;
      }
    },
    [conversationId, load, workspaceId],
  );

  const dispatch = useCallback(async () => {
    if (!conversationId) return;
    const response = await api.dispatchConversationMessageQueue(conversationId, workspaceId);
    await load();
    return response;
  }, [conversationId, load, workspaceId]);

  const scopedState = state.conversationId === conversationId ? state : EMPTY_STATE;
  return { state: scopedState, enqueue, deleteItem, dispatch, refetch: load };
}
