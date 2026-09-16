import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ConversationSummary } from '@/types';

const { api } = vi.hoisted(() => ({ api: { listConversationSummaries: vi.fn() } }));
vi.mock('@/api', () => ({ api }));

import { useConversationSummaryScopes } from './useConversationSummaryScopes';

const row = (id: string): ConversationSummary =>
  ({ id, updated_at: `2026-01-01T00:00:0${id}Z` }) as ConversationSummary;

describe('useConversationSummaryScopes', () => {
  afterEach(() => vi.clearAllMocks());

  it('loads self completely before others with independent owner_scope requests', async () => {
    api.listConversationSummaries
      .mockResolvedValueOnce([row('1')])
      .mockResolvedValueOnce([row('2')]);
    const onPage = vi.fn();
    const { result } = renderHook(() =>
      useConversationSummaryScopes({
        enabled: true,
        cutoffIso: '2026-01-01T00:00:00Z',
        resetKey: 'a',
        onPage,
      }),
    );

    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(api.listConversationSummaries).toHaveBeenNthCalledWith(
      1,
      undefined,
      expect.objectContaining({ owner_scope: 'self', limit: 50 }),
      expect.any(AbortSignal),
    );
    expect(api.listConversationSummaries).toHaveBeenNthCalledWith(
      2,
      undefined,
      expect.objectContaining({ owner_scope: 'others', limit: 50 }),
      expect.any(AbortSignal),
    );
    expect(onPage).toHaveBeenCalledTimes(2);
  });

  it('continues to others after self fails and retries self from its saved cursor', async () => {
    const error = new Error('self failed');
    api.listConversationSummaries
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce([row('2')])
      .mockResolvedValueOnce([row('1')]);
    const { result } = renderHook(() =>
      useConversationSummaryScopes({
        enabled: true,
        cutoffIso: 'cutoff',
        resetKey: 'a',
        onPage: vi.fn(),
      }),
    );

    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(result.current.scopeStates.self.error).toBe(error);
    await act(async () => result.current.retryFailed());
    await waitFor(() => expect(result.current.scopeStates.self.complete).toBe(true));
    expect(api.listConversationSummaries).toHaveBeenCalledTimes(3);
    expect(api.listConversationSummaries.mock.calls[2][1]).toMatchObject({ owner_scope: 'self' });
  });

  it('retries every remaining self page after its initial failure without rereading others', async () => {
    const error = new Error('self failed');
    const firstRetryPage = Array.from({ length: 50 }, (_, index) => row(String(index)));
    const secondRetryPage = Array.from({ length: 50 }, (_, index) => row(String(index + 50)));
    api.listConversationSummaries
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce(firstRetryPage)
      .mockResolvedValueOnce(secondRetryPage)
      .mockResolvedValueOnce([row('100')]);
    const { result } = renderHook(() =>
      useConversationSummaryScopes({
        enabled: true,
        cutoffIso: 'cutoff',
        resetKey: 'a',
        onPage: vi.fn(),
      }),
    );

    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(result.current.scopeStates.self.error).toBe(error);

    await act(async () => result.current.retryFailed());

    await waitFor(() => expect(result.current.scopeStates.self.complete).toBe(true));
    expect(result.current.scopeStates.self.error).toBeNull();
    expect(api.listConversationSummaries).toHaveBeenCalledTimes(5);
    expect(api.listConversationSummaries.mock.calls.slice(2)).toEqual([
      [
        undefined,
        expect.objectContaining({
          owner_scope: 'self',
          cursorUpdatedAt: undefined,
          cursorId: undefined,
        }),
        expect.any(AbortSignal),
      ],
      [
        undefined,
        expect.objectContaining({
          owner_scope: 'self',
          cursorUpdatedAt: '2026-01-01T00:00:049Z',
          cursorId: '49',
        }),
        expect.any(AbortSignal),
      ],
      [
        undefined,
        expect.objectContaining({
          owner_scope: 'self',
          cursorUpdatedAt: '2026-01-01T00:00:099Z',
          cursorId: '99',
        }),
        expect.any(AbortSignal),
      ],
    ]);
  });

  it('uses a saved cursor on the next page and never rereads completed scopes', async () => {
    const firstPage = Array.from({ length: 50 }, (_, index) => row(String(index)));
    api.listConversationSummaries
      .mockResolvedValueOnce(firstPage)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([]);
    const { result } = renderHook(() =>
      useConversationSummaryScopes({
        enabled: true,
        cutoffIso: 'cutoff',
        resetKey: 'a',
        onPage: vi.fn(),
      }),
    );

    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(api.listConversationSummaries.mock.calls[1][1]).toMatchObject({
      owner_scope: 'self',
      cursorId: '49',
    });
    await act(async () => result.current.retryFailed());
    expect(api.listConversationSummaries).toHaveBeenCalledTimes(3);
  });

  it('ignores an abort-ignoring response after reset and completes the new scopes', async () => {
    let resolveSelf!: (rows: ConversationSummary[]) => void;
    api.listConversationSummaries.mockImplementationOnce(
      () => new Promise<ConversationSummary[]>((resolve) => (resolveSelf = resolve)),
    );
    api.listConversationSummaries.mockResolvedValue([]);
    const onPage = vi.fn();
    const { result, rerender } = renderHook(
      ({ enabled, resetKey }) =>
        useConversationSummaryScopes({ enabled, cutoffIso: 'cutoff', resetKey, onPage }),
      { initialProps: { enabled: true, resetKey: 'a' } },
    );

    rerender({ enabled: false, resetKey: 'a' });
    await act(async () => resolveSelf([row('stale')]));
    expect(result.current.scopeStates.self.complete).toBe(false);
    rerender({ enabled: true, resetKey: 'b' });
    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(api.listConversationSummaries).toHaveBeenCalledTimes(3);
    expect(result.current.scopeStates.self).toMatchObject({
      complete: true,
      cursor: null,
      error: null,
    });
    expect(result.current.scopeStates.others).toMatchObject({ complete: true, error: null });
    expect(onPage).not.toHaveBeenCalledWith([row('stale')]);
  });

  it('resumes a disabled scope from its saved cursor without resetting progress', async () => {
    const firstPage = Array.from({ length: 50 }, (_, index) => row(String(index)));
    let resolveSecondPage!: (rows: ConversationSummary[]) => void;
    api.listConversationSummaries
      .mockResolvedValueOnce(firstPage)
      .mockImplementationOnce(
        () => new Promise<ConversationSummary[]>((resolve) => (resolveSecondPage = resolve)),
      )
      .mockResolvedValue([]);
    const { result, rerender } = renderHook(
      ({ enabled }) =>
        useConversationSummaryScopes({
          enabled,
          cutoffIso: 'cutoff',
          resetKey: 'a',
          onPage: vi.fn(),
        }),
      { initialProps: { enabled: true } },
    );

    await waitFor(() => expect(api.listConversationSummaries).toHaveBeenCalledTimes(2));
    rerender({ enabled: false });
    await act(async () => resolveSecondPage([row('stale')]));
    expect(result.current.scopeStates.self.cursor?.cursorId).toBe('49');
    expect(result.current.scopeStates.self.complete).toBe(false);

    rerender({ enabled: true });
    await waitFor(() => expect(result.current.scopeStates.others.complete).toBe(true));
    expect(api.listConversationSummaries).toHaveBeenNthCalledWith(
      3,
      undefined,
      expect.objectContaining({ owner_scope: 'self', cursorId: '49' }),
      expect.any(AbortSignal),
    );
  });
});
