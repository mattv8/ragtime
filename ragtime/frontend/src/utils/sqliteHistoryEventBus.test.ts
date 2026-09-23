import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({ subscribeUserSpaceSqliteHistoryEvents: vi.fn() }));

vi.mock('@/api', () => ({ api: apiMock }));

import { subscribeHistoryEvents } from './sqliteHistoryEventBus';

function source(readyState = 1) {
  const listeners = new Map<string, EventListener>();
  return {
    listeners,
    source: {
      addEventListener: vi.fn((name: string, listener: EventListener) =>
        listeners.set(name, listener),
      ),
      close: vi.fn(),
      readyState,
    } as unknown as EventSource,
  };
}

describe('subscribeHistoryEvents', () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.useRealTimers());

  it('shares one unfiltered workspace connection and retains it until final unsubscribe', () => {
    const eventSource = source();
    apiMock.subscribeUserSpaceSqliteHistoryEvents.mockReturnValue(eventSource.source);
    const first = vi.fn();
    const second = vi.fn();
    const unsubscribeFirst = subscribeHistoryEvents('ws-a', { onHistoryChanged: first });
    const unsubscribeSecond = subscribeHistoryEvents('ws-a', { onHistoryChanged: second });

    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledOnce();
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledWith('ws-a');
    eventSource.listeners.get('history_changed')?.(new Event('history_changed'));
    expect(first).toHaveBeenCalledOnce();
    expect(second).toHaveBeenCalledOnce();

    unsubscribeFirst();
    expect(eventSource.source.close).not.toHaveBeenCalled();
    eventSource.listeners.get('history_changed')?.(new Event('history_changed'));
    expect(second).toHaveBeenCalledTimes(2);
    unsubscribeSecond();
    expect(eventSource.source.close).toHaveBeenCalledOnce();
  });

  it('isolates workspaces and replaces a revoked workspace subscription', () => {
    const first = source();
    const second = source();
    const replacement = source();
    apiMock.subscribeUserSpaceSqliteHistoryEvents
      .mockReturnValueOnce(first.source)
      .mockReturnValueOnce(second.source)
      .mockReturnValueOnce(replacement.source);
    const revoked = vi.fn();
    const unsubscribeFirst = subscribeHistoryEvents('ws-a', { onAccessRevoked: revoked });
    const unsubscribeSecond = subscribeHistoryEvents('ws-b', {});

    first.listeners.get('access_revoked')?.(new Event('access_revoked'));
    expect(revoked).toHaveBeenCalledOnce();
    expect(first.source.close).toHaveBeenCalledOnce();
    const unsubscribeReplacement = subscribeHistoryEvents('ws-a', {});

    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(3);
    expect(second.source.close).not.toHaveBeenCalled();
    unsubscribeFirst();
    unsubscribeSecond();
    unsubscribeReplacement();
    expect(second.source.close).toHaveBeenCalledOnce();
  });

  it('reconnects a closed source after a bounded exponential backoff without losing listeners', () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const first = source();
    const replacement = source();
    Object.defineProperty(first.source, 'readyState', { value: 2, configurable: true });
    apiMock.subscribeUserSpaceSqliteHistoryEvents
      .mockReturnValueOnce(first.source)
      .mockReturnValueOnce(replacement.source);
    const changed = vi.fn();
    const unsubscribeFirst = subscribeHistoryEvents('ws-a', { onHistoryChanged: changed });

    first.listeners.get('error')?.(new Event('error'));
    expect(first.source.close).toHaveBeenCalledOnce();
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(999);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(1);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2);
    replacement.listeners.get('history_changed')?.(new Event('history_changed'));
    expect(changed).toHaveBeenCalledOnce();

    const unsubscribeSecond = subscribeHistoryEvents('ws-a', { onHistoryChanged: vi.fn() });
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2);
    unsubscribeFirst();
    unsubscribeSecond();
    expect(replacement.source.close).toHaveBeenCalledOnce();
  });

  it('backs off repeated closed failures and resets the delay after opening', () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const first = source(2);
    const second = source(2);
    const third = source();
    const fourth = source();
    apiMock.subscribeUserSpaceSqliteHistoryEvents
      .mockReturnValueOnce(first.source)
      .mockReturnValueOnce(second.source)
      .mockReturnValueOnce(third.source)
      .mockReturnValueOnce(fourth.source);

    const unsubscribe = subscribeHistoryEvents('ws-a', {});
    first.listeners.get('error')?.(new Event('error'));
    vi.advanceTimersByTime(1_000);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2);
    second.listeners.get('error')?.(new Event('error'));
    vi.advanceTimersByTime(1_999);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(1);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(3);
    third.listeners.get('open')?.(new Event('open'));
    Object.defineProperty(third.source, 'readyState', { value: 2, configurable: true });
    third.listeners.get('error')?.(new Event('error'));
    vi.advanceTimersByTime(999);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(3);
    vi.advanceTimersByTime(1);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(4);
    unsubscribe();
  });

  it('does not duplicate or retain a reconnect timer after final unsubscribe or revocation', () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const first = source(2);
    const revoked = source(2);
    apiMock.subscribeUserSpaceSqliteHistoryEvents
      .mockReturnValueOnce(first.source)
      .mockReturnValueOnce(revoked.source);

    const unsubscribe = subscribeHistoryEvents('ws-a', {});
    first.listeners.get('error')?.(new Event('error'));
    const unsubscribeSecond = subscribeHistoryEvents('ws-a', {});
    unsubscribe();
    unsubscribeSecond();
    vi.advanceTimersByTime(30_000);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledOnce();

    const unsubscribeRevoked = subscribeHistoryEvents('ws-b', {});
    revoked.listeners.get('error')?.(new Event('error'));
    revoked.listeners.get('access_revoked')?.(new Event('access_revoked'));
    vi.advanceTimersByTime(30_000);
    expect(apiMock.subscribeUserSpaceSqliteHistoryEvents).toHaveBeenCalledTimes(2);
    unsubscribeRevoked();
  });
});
