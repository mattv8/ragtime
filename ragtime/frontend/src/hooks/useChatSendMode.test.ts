import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { useChatSendMode } from './useChatSendMode';

const keyFor = (userId: string) => `ragtime:composer:sendMode:${encodeURIComponent(userId)}`;
let values = new Map<string, string>();
const storage: Storage = {
  getItem: (key) => values.get(key) ?? null,
  setItem: (key, value) => void values.set(key, String(value)),
  removeItem: (key) => void values.delete(key),
  clear: () => values.clear(),
  key: (index) => [...values.keys()][index] ?? null,
  get length() {
    return values.size;
  },
};
beforeEach(() => {
  values = new Map();
  Object.defineProperty(window, 'localStorage', { configurable: true, value: storage });
});
afterEach(() => values.clear());

describe('useChatSendMode', () => {
  it('uses the stored per-user mode, defaulting invalid values to Ctrl+Enter', () => {
    storage.setItem(keyFor('person/a'), 'enter');
    storage.setItem(keyFor('invalid'), 'nope');
    expect(renderHook(() => useChatSendMode('person/a')).result.current[0]).toBe('enter');
    expect(renderHook(() => useChatSendMode('invalid')).result.current[0]).toBe('ctrl-enter');
  });
  it('persists updates and synchronizes mounted same-tab instances', () => {
    const first = renderHook(() => useChatSendMode('same user'));
    const second = renderHook(() => useChatSendMode('same user'));
    act(() => first.result.current[1]('enter'));
    expect(storage.getItem(keyFor('same user'))).toBe('enter');
    expect(second.result.current[0]).toBe('enter');
  });
  it('updates for a matching cross-tab storage event and changes users safely', () => {
    const { result, rerender } = renderHook(({ userId }) => useChatSendMode(userId), {
      initialProps: { userId: 'first' },
    });
    act(() =>
      window.dispatchEvent(
        new StorageEvent('storage', { key: keyFor('first'), newValue: 'button' }),
      ),
    );
    expect(result.current[0]).toBe('button');
    storage.setItem(keyFor('second'), 'enter');
    rerender({ userId: 'second' });
    expect(result.current[0]).toBe('enter');
  });

  it('never exposes the prior user mode while rendering a new user', () => {
    storage.setItem(keyFor('first'), 'enter');
    const renders: string[] = [];
    const { rerender } = renderHook(
      ({ userId }) => {
        const [mode] = useChatSendMode(userId);
        renders.push(mode);
        return mode;
      },
      { initialProps: { userId: 'first' } },
    );

    const rendersBeforeUserChange = renders.length;
    rerender({ userId: 'second' });

    expect(renders.slice(rendersBeforeUserChange)).not.toContain('enter');
    expect(renders[renders.length - 1]).toBe('ctrl-enter');
  });

  it('resets on a localStorage clear event and rejects invalid same-document modes', () => {
    storage.setItem(keyFor('first'), 'enter');
    const { result } = renderHook(() => useChatSendMode('first'));
    act(() =>
      window.dispatchEvent(
        new CustomEvent('ragtime:composer:send-mode-change', {
          detail: { key: keyFor('first'), mode: 'invalid' },
        }),
      ),
    );
    expect(result.current[0]).toBe('ctrl-enter');

    act(() => window.dispatchEvent(new StorageEvent('storage', { key: null })));
    expect(result.current[0]).toBe('ctrl-enter');
  });
});
