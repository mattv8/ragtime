import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { usePdmTools } from './usePdmTools';

const client = vi.hoisted(() => ({ listToolConfigs: vi.fn(), getPdmIndexStatus: vi.fn() }));
vi.mock('../api/client', () => ({ api: client }));

const status = (job: string | null) => ({
  last_indexed: null,
  document_count: 0,
  embedding_count: 0,
  current_job: job ? { status: job } : null,
});

describe('usePdmTools', () => {
  beforeEach(() => {
    client.listToolConfigs.mockReset().mockResolvedValue([
      { id: 'a', tool_type: 'solidworks_pdm', enabled: false },
      { id: 'b', tool_type: 'solidworks' },
    ]);
    client.getPdmIndexStatus.mockReset().mockResolvedValue(status(null));
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('filters the list and loads the selected status', async () => {
    const { result } = renderHook(() => usePdmTools('a'));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.tools.map((tool) => tool.id)).toEqual(['a']);
    await waitFor(() => expect(result.current.indexStatus).toEqual(status(null)));
  });

  it('polls active jobs, pauses while hidden and stops on unmount', async () => {
    vi.useFakeTimers();
    let hidden = false;
    vi.spyOn(document, 'hidden', 'get').mockImplementation(() => hidden);
    client.getPdmIndexStatus.mockResolvedValue(status('indexing'));
    const { unmount } = renderHook(() => usePdmTools('a'));
    await act(async () => {});
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(2);
    hidden = true;
    act(() => document.dispatchEvent(new Event('visibilitychange')));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(2);
    hidden = false;
    act(() => document.dispatchEvent(new Event('visibilitychange')));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(client.getPdmIndexStatus.mock.calls.length).toBeGreaterThan(2);
    unmount();
    const count = client.getPdmIndexStatus.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(count);
  });

  it('does not apply a stale tool response', async () => {
    let resolveOld!: (value: ReturnType<typeof status>) => void;
    client.getPdmIndexStatus.mockImplementation((id: string) =>
      id === 'a'
        ? new Promise((resolve) => {
            resolveOld = resolve;
          })
        : Promise.resolve(status('indexing')),
    );
    const { result, rerender } = renderHook(({ id }) => usePdmTools(id), {
      initialProps: { id: 'a' },
    });
    rerender({ id: 'b' });
    await waitFor(() => expect(result.current.indexStatus).toEqual(status('indexing')));
    await act(async () => resolveOld(status(null)));
    expect(result.current.indexStatus).toEqual(status('indexing'));
  });

  it('coalesces refreshes for the same tool while a request is in flight', async () => {
    let resolveOld!: (value: ReturnType<typeof status>) => void;
    client.getPdmIndexStatus
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveOld = resolve;
          }),
      )
      .mockResolvedValue(status('indexing'));
    const { result } = renderHook(() => usePdmTools('a'));
    let refresh!: Promise<void>;
    act(() => {
      refresh = result.current.refreshStatus();
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(1);
    await act(async () => {
      resolveOld(status(null));
      await refresh;
    });
    expect(result.current.indexStatus).toEqual(status(null));
  });

  it('waits for slow polls to settle before scheduling the next poll', async () => {
    vi.useFakeTimers();
    let resolvePoll!: (value: ReturnType<typeof status>) => void;
    client.getPdmIndexStatus
      .mockResolvedValueOnce(status('indexing'))
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolvePoll = resolve;
          }),
      )
      .mockResolvedValue(status(null));
    const { result, unmount } = renderHook(() => usePdmTools('a'));
    await act(async () => {});
    await act(async () => {
      await vi.advanceTimersByTimeAsync(9000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(2);
    await act(async () => resolvePoll(status('indexing')));
    expect(result.current.indexStatus).toEqual(status('indexing'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(3);
    unmount();
  });

  it('stops polling when the job reaches a terminal state', async () => {
    vi.useFakeTimers();
    client.getPdmIndexStatus
      .mockResolvedValueOnce(status('pending'))
      .mockResolvedValue(status(null));
    const { unmount } = renderHook(() => usePdmTools('a'));
    await act(async () => {});
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(9000);
    });
    expect(client.getPdmIndexStatus).toHaveBeenCalledTimes(2);
    unmount();
  });
});
