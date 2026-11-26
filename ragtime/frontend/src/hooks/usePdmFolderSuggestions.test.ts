import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePdmFolderSuggestions } from './usePdmFolderSuggestions';

const listing = vi.hoisted(() => vi.fn());
vi.mock('../utils/pdmFolderListing', () => ({
  listAllPdmFolderChildren: listing,
}));

const initial = { toolId: 'tool', pdmToolId: 'pdm', parent: '\\', enabled: true };
const advance = async (ms = 150) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

describe('usePdmFolderSuggestions', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    listing.mockReset().mockResolvedValue({ items: [], total: 0 });
  });
  afterEach(() => vi.useRealTimers());

  it('debounces changes and ignores stale responses', async () => {
    let finish!: (value: { items: never[]; total: number }) => void;
    listing.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    rerender({ ...initial, parent: '\\B' });
    await advance();
    await act(async () => finish({ items: [], total: 99 }));
    expect(result.current.total).toBe(0);
    expect(listing).toHaveBeenCalledTimes(2);
  });

  it('does not fetch while disabled or without a parent', async () => {
    const { rerender } = renderHook(usePdmFolderSuggestions, {
      initialProps: { ...initial, enabled: false, parent: null as string | null },
    });
    await advance();
    rerender({ ...initial, parent: null });
    await advance();
    expect(listing).not.toHaveBeenCalled();
  });

  it('reuses fresh cache and silently revalidates stale cache', async () => {
    listing.mockResolvedValue({ items: [], total: 7 });
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    rerender({ ...initial, parent: '\\B' });
    await advance();
    rerender(initial);
    expect(result.current.total).toBe(7);
    await advance();
    expect(listing).toHaveBeenCalledTimes(2);
    await advance(60_000);
    rerender({ ...initial, parent: '\\B' });
    expect(result.current.total).toBe(7);
    expect(result.current.status).toBe('ready');
    await advance();
    expect(listing).toHaveBeenCalledTimes(3);
  });

  it('retries errors and clears the cache when the tool changes', async () => {
    listing.mockRejectedValueOnce(Object.assign(new Error('offline'), { code: 'offline' }));
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    expect(result.current.error).toEqual({ message: 'offline', code: 'offline' });
    act(() => result.current.retry());
    await advance();
    expect(result.current.status).toBe('ready');
    rerender({ ...initial, toolId: 'other' });
    await advance();
    rerender(initial);
    await advance();
    expect(listing).toHaveBeenCalledTimes(4);
  });

  it('keeps stale cached results visible while revalidation is pending', async () => {
    listing.mockResolvedValueOnce({ items: [], total: 7 });
    let finish!: (value: { items: never[]; total: number }) => void;
    listing.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    rerender({ ...initial, enabled: false });
    await advance(60_000);
    rerender(initial);
    expect(result.current.status).toBe('ready');
    expect(result.current.total).toBe(7);
    await advance();
    expect(result.current.status).toBe('ready');
    expect(result.current.total).toBe(7);
    await act(async () => finish({ items: [], total: 9 }));
    expect(result.current.total).toBe(9);
  });

  it('does not cache errors when returning to a parent', async () => {
    listing.mockRejectedValueOnce(Object.assign(new Error('offline'), { code: 'offline' }));
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    expect(result.current.status).toBe('error');
    rerender({ ...initial, parent: '\\B' });
    await advance();
    rerender(initial);
    await advance();
    expect(listing).toHaveBeenCalledTimes(3);
    expect(result.current.status).toBe('ready');
  });

  it('clears cached results when the linked PDM tool changes and changes back', async () => {
    listing.mockResolvedValueOnce({ items: [], total: 7 });
    const { result, rerender } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    expect(result.current.total).toBe(7);
    rerender({ ...initial, pdmToolId: 'other' });
    expect(result.current.total).toBe(0);
    await advance();
    rerender(initial);
    expect(result.current.total).toBe(0);
    await advance();
    expect(listing).toHaveBeenCalledTimes(3);
  });

  it('aborts a pending request on unmount and ignores its response', async () => {
    let finish!: (value: { items: never[]; total: number }) => void;
    listing.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const { unmount } = renderHook(usePdmFolderSuggestions, { initialProps: initial });
    await advance();
    const signal = listing.mock.calls[0][3] as AbortSignal;
    unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => finish({ items: [], total: 1 }));
  });
});
