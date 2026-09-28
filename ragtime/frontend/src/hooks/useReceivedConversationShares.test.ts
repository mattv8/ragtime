import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReceivedConversationShare } from '@/types';

const { api } = vi.hoisted(() => ({ api: { listReceivedConversationShares: vi.fn() } }));
vi.mock('@/api', () => ({ api }));

import { useReceivedConversationShares } from './useReceivedConversationShares';

const PAGE_SIZE = 50;
const share = (id: string, second = 0) =>
  ({
    id,
    created_at: `2026-01-01T00:00:${String(second).padStart(2, '0')}Z`,
  }) as ReceivedConversationShare;

describe('useReceivedConversationShares', () => {
  beforeEach(() => {
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('replaces a complete deduplicated multi-page snapshot', async () => {
    const firstPage = Array.from({ length: PAGE_SIZE }, (_, index) =>
      share(String(index), 59 - index),
    );
    api.listReceivedConversationShares
      .mockResolvedValueOnce(firstPage)
      .mockResolvedValueOnce([share('0', 10), share('last', 9)]);
    const { result } = renderHook(() =>
      useReceivedConversationShares({ userId: 'u1', enabled: true }),
    );

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.shares.map(({ id }) => id)).toEqual([
      ...firstPage.map(({ id }) => id),
      'last',
    ]);
    const lastFirstPageShare = firstPage[firstPage.length - 1];
    expect(api.listReceivedConversationShares.mock.calls[1][0]).toMatchObject({
      limit: PAGE_SIZE,
      cursorCreatedAt: lastFirstPageShare?.created_at,
      cursorId: lastFirstPageShare?.id,
    });
  });

  it('stops on a nonadvancing cursor rather than fetching forever', async () => {
    const existing = share('existing', 60);
    const page = Array.from({ length: PAGE_SIZE }, () => share('same', 59));
    api.listReceivedConversationShares.mockResolvedValueOnce([existing]).mockResolvedValue(page);
    const { result } = renderHook(() =>
      useReceivedConversationShares({ userId: 'u1', enabled: true }),
    );

    await waitFor(() => expect(result.current.shares).toEqual([existing]));
    act(() => {
      void result.current.refresh();
    });
    await waitFor(() => expect(result.current.error?.message).toMatch(/did not advance/));
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(3);
    expect(result.current.shares).toEqual([existing]);
  });

  it('replaces prior rows after a refresh so revoked shares disappear', async () => {
    api.listReceivedConversationShares
      .mockResolvedValueOnce([share('gone', 1)])
      .mockResolvedValueOnce([]);
    const { result } = renderHook(() =>
      useReceivedConversationShares({ userId: 'u1', enabled: true }),
    );

    await waitFor(() => expect(result.current.shares).toEqual([share('gone', 1)]));
    await act(async () => result.current.refresh());
    expect(result.current.shares).toEqual([]);
  });

  it('keeps the prior snapshot on error and allows retry', async () => {
    const failure = new Error('offline');
    api.listReceivedConversationShares
      .mockResolvedValueOnce([share('saved', 1)])
      .mockRejectedValueOnce(failure)
      .mockResolvedValueOnce([share('new', 2)]);
    const { result } = renderHook(() =>
      useReceivedConversationShares({ userId: 'u1', enabled: true }),
    );

    await waitFor(() => expect(result.current.shares).toEqual([share('saved', 1)]));
    await act(async () => result.current.refresh());
    expect(result.current.error).toBe(failure);
    expect(result.current.shares).toEqual([share('saved', 1)]);
    await act(async () => result.current.refresh());
    expect(result.current.error).toBeNull();
    expect(result.current.shares).toEqual([share('new', 2)]);
  });

  it('refreshes on visible polling, focus, and visibility return but not while hidden', async () => {
    vi.useFakeTimers();
    api.listReceivedConversationShares.mockResolvedValue([]);
    renderHook(() => useReceivedConversationShares({ userId: 'u1', enabled: true }));
    await act(async () => {});
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(1);

    await act(async () => vi.advanceTimersByTimeAsync(30_000));
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(2);
    await act(async () => window.dispatchEvent(new Event('focus')));
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(3);

    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    await act(async () => document.dispatchEvent(new Event('visibilitychange')));
    await act(async () => vi.advanceTimersByTimeAsync(30_000));
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(3);

    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    await act(async () => document.dispatchEvent(new Event('visibilitychange')));
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(4);
  });

  it('masks all prior-account state on every account-switch render', async () => {
    let resolveOld!: (shares: ReceivedConversationShare[]) => void;
    api.listReceivedConversationShares
      .mockResolvedValueOnce([share('old', 1)])
      .mockImplementationOnce(
        () => new Promise<ReceivedConversationShare[]>((resolve) => (resolveOld = resolve)),
      )
      .mockResolvedValueOnce([share('new', 2)]);
    const renders: Array<{
      shares: ReceivedConversationShare[];
      loading: boolean;
      error: Error | null;
    }> = [];
    const { result, rerender } = renderHook(
      ({ userId }) => {
        const state = useReceivedConversationShares({ userId, enabled: true });
        renders.push({ shares: state.shares, loading: state.loading, error: state.error });
        return state;
      },
      { initialProps: { userId: 'old' } },
    );

    await waitFor(() => expect(result.current.shares).toEqual([share('old', 1)]));
    act(() => {
      void result.current.refresh();
    });
    const oldSignal = api.listReceivedConversationShares.mock.calls[1][1] as AbortSignal;
    rerender({ userId: 'new' });
    expect(oldSignal.aborted).toBe(true);
    expect(result.current.shares).toEqual([]);
    expect(renders.slice(-2)).toContainEqual({ shares: [], loading: false, error: null });
    await waitFor(() => expect(result.current.shares).toEqual([share('new', 2)]));
    await act(async () => resolveOld([share('old', 3)]));
    expect(result.current.shares).toEqual([share('new', 2)]);
  });

  it('masks a prior-account error and loading state before the new account effect runs', async () => {
    let rejectOld!: (error: Error) => void;
    api.listReceivedConversationShares.mockImplementationOnce(
      () => new Promise<ReceivedConversationShare[]>((_, reject) => (rejectOld = reject)),
    );
    const { result, rerender } = renderHook(
      ({ userId, enabled }) => useReceivedConversationShares({ userId, enabled }),
      { initialProps: { userId: 'old', enabled: true } },
    );

    await act(async () => rejectOld(new Error('old account offline')));
    await waitFor(() => expect(result.current.error?.message).toBe('old account offline'));
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    rerender({ userId: 'new', enabled: true });
    expect(result.current).toMatchObject({ shares: [], loading: false, error: null });
    rerender({ userId: 'new', enabled: false });
    expect(result.current).toMatchObject({ shares: [], loading: false, error: null });
  });

  it('does not overlap a pending refresh and clears on disable', async () => {
    let resolve!: (shares: ReceivedConversationShare[]) => void;
    api.listReceivedConversationShares.mockImplementationOnce(
      () => new Promise<ReceivedConversationShare[]>((done) => (resolve = done)),
    );
    const { result, rerender } = renderHook(
      ({ enabled }) => useReceivedConversationShares({ userId: 'u1', enabled }),
      { initialProps: { enabled: true } },
    );

    act(() => {
      void result.current.refresh();
      void result.current.refresh();
    });
    expect(api.listReceivedConversationShares).toHaveBeenCalledTimes(1);
    rerender({ enabled: false });
    expect(result.current).toMatchObject({ shares: [], loading: false, error: null });
    expect((api.listReceivedConversationShares.mock.calls[0][1] as AbortSignal).aborted).toBe(true);
    await act(async () => resolve([share('late', 1)]));
    expect(result.current.shares).toEqual([]);
  });

  it('aborts and ignores requests after unmount', async () => {
    let resolve!: (shares: ReceivedConversationShare[]) => void;
    api.listReceivedConversationShares.mockImplementationOnce(
      () => new Promise<ReceivedConversationShare[]>((done) => (resolve = done)),
    );
    const { unmount } = renderHook(() =>
      useReceivedConversationShares({ userId: 'u1', enabled: true }),
    );
    const signal = api.listReceivedConversationShares.mock.calls[0][1] as AbortSignal;
    unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => resolve([share('late')]));
  });
});
