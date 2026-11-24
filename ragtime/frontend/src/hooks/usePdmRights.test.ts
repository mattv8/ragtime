import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PdmRightsResponse } from '../api/solidworks';
import { usePdmRights } from './usePdmRights';

const api = vi.hoisted(() => ({ checkPdmRights: vi.fn() }));
vi.mock('../api/solidworks', () => ({ solidworksApi: api }));

const response: PdmRightsResponse = {
  checked: true,
  user: 'Alice',
  root: {
    path: '\\',
    checked: true,
    exists: true,
    rights: { read: true, add_files: false, check_out: false, add_folders: false },
    write_ready: false,
  },
  folders: [],
  error: null,
};
const initial = {
  toolId: 'tool',
  pdmToolId: 'pdm',
  folders: ['\\A'],
  allowChanges: true,
  enabled: true,
};
const advance = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(400);
  });
};

describe('usePdmRights', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    api.checkPdmRights.mockReset().mockResolvedValue(response);
  });
  afterEach(() => vi.useRealTimers());

  it('debounces rapid changes and sends only the latest folders', async () => {
    const { rerender, result } = renderHook((props) => usePdmRights(props), {
      initialProps: initial,
    });
    rerender({ ...initial, folders: ['\\B'] });
    rerender({ ...initial, folders: ['\\C'] });
    expect(result.current.loading).toBe(true);
    expect(api.checkPdmRights).not.toHaveBeenCalled();
    await advance();
    expect(api.checkPdmRights).toHaveBeenCalledTimes(1);
    expect(api.checkPdmRights).toHaveBeenCalledWith(
      'tool',
      { pdm_tool_id: 'pdm', folders: ['\\C'] },
      expect.any(AbortSignal),
    );
    expect(result.current.state.root).toBe('ok');
  });

  it('sends no folders when changes are off', async () => {
    renderHook(() => usePdmRights({ ...initial, allowChanges: false }));
    await advance();
    expect(api.checkPdmRights.mock.calls[0][1]).toEqual({ pdm_tool_id: 'pdm', folders: [] });
  });

  it('ignores stale responses and aborts on change and unmount', async () => {
    let resolveOld!: (value: PdmRightsResponse) => void;
    api.checkPdmRights.mockImplementation(
      () =>
        new Promise<PdmRightsResponse>((resolve) => {
          resolveOld = resolve;
        }),
    );
    const { result, rerender, unmount } = renderHook((props) => usePdmRights(props), {
      initialProps: initial,
    });
    await advance();
    const firstSignal = api.checkPdmRights.mock.calls[0][2] as AbortSignal;
    rerender({ ...initial, pdmToolId: 'other' });
    expect(firstSignal.aborted).toBe(true);
    await act(async () => resolveOld(response));
    expect(result.current.state.root).toBe('not_checked');
    expect(result.current.loading).toBe(true);
    await advance();
    const secondSignal = api.checkPdmRights.mock.calls[1][2] as AbortSignal;
    unmount();
    expect(secondSignal.aborted).toBe(true);
  });

  it.each([
    { ...initial, enabled: false },
    { ...initial, toolId: null },
    { ...initial, pdmToolId: null },
  ])('does not request when disabled or missing an id', async (props) => {
    const { result } = renderHook(() => usePdmRights(props));
    await advance();
    act(() => result.current.recheck());
    expect(api.checkPdmRights).not.toHaveBeenCalled();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('reports response and thrown errors', async () => {
    api.checkPdmRights
      .mockResolvedValueOnce({ ...response, error: { code: 'x', message: 'failed' } })
      .mockRejectedValueOnce(new Error('offline'));
    const { result } = renderHook(() => usePdmRights(initial));
    await advance();
    expect(result.current.error).toBe('failed');
    await act(async () => result.current.recheck());
    expect(result.current.error).toBe('offline');
  });

  it('rechecks immediately with stable callback identity and cancels debounce', async () => {
    const { result, rerender } = renderHook((props) => usePdmRights(props), {
      initialProps: initial,
    });
    const recheck = result.current.recheck;
    rerender({ ...initial, folders: ['\\B'] });
    expect(result.current.recheck).toBe(recheck);
    await act(async () => result.current.recheck());
    expect(api.checkPdmRights).toHaveBeenCalledTimes(1);
    await advance();
    expect(api.checkPdmRights).toHaveBeenCalledTimes(1);
  });
});
