import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Conversation, ConversationMessageWindow } from '@/types';

const { api } = vi.hoisted(() => ({
  api: {
    getConversationLatestExchange: vi.fn(),
    getConversationMessageWindow: vi.fn(),
    getConversationWindowMessage: vi.fn(),
    getConversation: vi.fn(),
  },
}));

vi.mock('@/api', () => ({ api }));

import { useConversationWindow } from './useConversationWindow';

const conversation = (id = 'one'): Conversation =>
  ({
    id,
    title: id,
    model: 'model',
    messages: [{ role: 'user', content: id, timestamp: '2026-01-01T00:00:00Z' }],
    total_tokens: 1,
    active_task_id: null,
    tool_output_mode: 'default',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  }) as Conversation;

const window = (id = 'one', cursor: string | null = 'older'): ConversationMessageWindow => {
  const full = conversation(id);
  const { messages: _messages, ...metadata } = full;
  return {
    conversation: metadata,
    revision: `revision-${id}`,
    total_message_count: 3,
    entries: [
      {
        index: 2,
        key: `${id}-2`,
        state: 'deferred',
        message: null,
        preview: {
          role: 'assistant',
          content: 'preview',
          timestamp: '2026-01-01T00:00:00Z',
          content_truncated: true,
          has_details: true,
        },
      },
    ],
    next_cursor: cursor,
    has_more: cursor !== null,
    legacy_conversation: null,
  };
};

describe('useConversationWindow', () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('commits latest previews before one automatic first page and settles the sidebar gate', async () => {
    api.getConversationLatestExchange.mockResolvedValueOnce(window());
    api.getConversationMessageWindow.mockResolvedValueOnce({ ...window('one', null), entries: [] });
    const { result } = renderHook(() =>
      useConversationWindow({ conversationId: 'one', enabled: true }),
    );

    await waitFor(() => expect(result.current.entries[0]?.state).toBe('deferred'));
    await waitFor(() => expect(api.getConversationMessageWindow).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(result.current.firstPageSettled).toBe(true));

    expect(api.getConversationMessageWindow).toHaveBeenCalledWith(
      'one',
      { cursor: 'older', limit: 20 },
      undefined,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('ignores a stale latest response after navigation changes', async () => {
    let resolveOld!: (value: ConversationMessageWindow) => void;
    api.getConversationLatestExchange.mockImplementationOnce(
      () => new Promise<ConversationMessageWindow>((resolve) => (resolveOld = resolve)),
    );
    api.getConversationLatestExchange.mockResolvedValueOnce(window('two', null));
    const { result, rerender } = renderHook(
      ({ id }) => useConversationWindow({ conversationId: id, enabled: true }),
      { initialProps: { id: 'one' } },
    );

    rerender({ id: 'two' });
    await act(async () => resolveOld(window('one', null)));
    await waitFor(() => expect(result.current.metadata?.id).toBe('two'));
    expect(result.current.entries[0]?.key).toBe('two-2');
  });

  it('clears state and ignores an abort-ignoring response when disabled', async () => {
    let resolveLatest!: (value: ConversationMessageWindow) => void;
    api.getConversationLatestExchange.mockImplementationOnce(
      () => new Promise<ConversationMessageWindow>((resolve) => (resolveLatest = resolve)),
    );
    const { result, rerender } = renderHook(
      ({ enabled }) => useConversationWindow({ conversationId: 'one', enabled }),
      { initialProps: { enabled: true } },
    );

    rerender({ enabled: false });
    await act(async () => resolveLatest(window('one', null)));
    expect(result.current.metadata).toBeNull();
    expect(result.current.entries).toEqual([]);
  });

  it('deduplicates explicit full hydration and adopts the complete response', async () => {
    api.getConversationLatestExchange.mockResolvedValueOnce(window('one', null));
    let resolveFull!: (value: Conversation) => void;
    api.getConversation.mockImplementationOnce(
      () => new Promise<Conversation>((resolve) => (resolveFull = resolve)),
    );
    const { result } = renderHook(() =>
      useConversationWindow({ conversationId: 'one', enabled: true }),
    );
    await waitFor(() => expect(result.current.revision).toBe('revision-one'));

    let first!: Promise<Conversation>;
    let second!: Promise<Conversation>;
    act(() => {
      first = result.current.ensureFullConversation();
      second = result.current.ensureFullConversation();
    });
    expect(api.getConversation).toHaveBeenCalledTimes(1);
    await act(async () => resolveFull(conversation('one')));
    await expect(first).resolves.toEqual(conversation('one'));
    await expect(second).resolves.toEqual(conversation('one'));
    expect(result.current.fullConversation?.id).toBe('one');
    expect(result.current.entries[0]?.state).toBe('ready');
  });

  it('loads a deferred entry once and preserves its absolute index', async () => {
    api.getConversationLatestExchange.mockResolvedValueOnce(window('one', null));
    api.getConversationWindowMessage.mockResolvedValueOnce({
      index: 2,
      key: 'one-2',
      state: 'ready',
      message: conversation('one').messages[0],
      preview: null,
    });
    const { result } = renderHook(() =>
      useConversationWindow({ conversationId: 'one', enabled: true }),
    );
    await waitFor(() => expect(result.current.entries).toHaveLength(1));
    await act(async () =>
      Promise.all([result.current.loadMessage(2), result.current.loadMessage(2)]),
    );
    expect(api.getConversationWindowMessage).toHaveBeenCalledTimes(1);
    expect(result.current.entries[0]).toMatchObject({ index: 2, state: 'ready' });
  });

  it('settles the first-page gate with an actionable timeout error', async () => {
    vi.useFakeTimers();
    api.getConversationLatestExchange.mockResolvedValueOnce(window());
    api.getConversationMessageWindow.mockImplementationOnce(
      () => new Promise<ConversationMessageWindow>(() => undefined),
    );
    const { result } = renderHook(() =>
      useConversationWindow({ conversationId: 'one', enabled: true }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
      vi.advanceTimersByTime(16);
      await Promise.resolve();
    });
    expect(api.getConversationMessageWindow).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(10_000);
      await Promise.resolve();
    });
    expect(result.current.firstPageSettled).toBe(true);
    expect(result.current.olderError?.message).toContain('timed out');
  });

  it('restarts once on a stale page revision and surfaces a later conflict', async () => {
    api.getConversationLatestExchange
      .mockResolvedValueOnce(window('one'))
      .mockResolvedValueOnce(window('one'));
    api.getConversationMessageWindow
      .mockRejectedValueOnce(Object.assign(new Error('stale'), { status: 409 }))
      .mockRejectedValueOnce(Object.assign(new Error('stale again'), { status: 409 }));
    const { result } = renderHook(() =>
      useConversationWindow({ conversationId: 'one', enabled: true }),
    );

    await waitFor(() => expect(api.getConversationLatestExchange).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(result.current.olderError?.message).toContain('Conversation changed'),
    );
    expect(api.getConversationMessageWindow).toHaveBeenCalledTimes(2);
  });
});
