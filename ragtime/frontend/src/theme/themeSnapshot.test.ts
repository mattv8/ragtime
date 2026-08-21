import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getThemeSnapshot, subscribeToThemeChanges } from './themeSnapshot';

type MediaListener = (event: MediaQueryListEvent) => void;

function installMatchMediaStub(initialMatches: boolean) {
  let matches = initialMatches;
  const listeners = new Set<MediaListener>();
  const mediaQueryList = {
    matches,
    media: '(prefers-color-scheme: light)',
    onchange: null,
    addEventListener: vi.fn((_type: string, listener: MediaListener) => {
      listeners.add(listener);
    }),
    removeEventListener: vi.fn((_type: string, listener: MediaListener) => {
      listeners.delete(listener);
    }),
    addListener: vi.fn((listener: MediaListener) => {
      listeners.add(listener);
    }),
    removeListener: vi.fn((listener: MediaListener) => {
      listeners.delete(listener);
    }),
    dispatchEvent: vi.fn(() => true),
  } as unknown as MediaQueryList;

  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: vi.fn(() => mediaQueryList),
  });

  return {
    setMatches(nextMatches: boolean) {
      matches = nextMatches;
      Object.assign(mediaQueryList, { matches });
      const event = { matches, media: mediaQueryList.media } as MediaQueryListEvent;
      for (const listener of listeners) {
        listener(event);
      }
      mediaQueryList.onchange?.(event);
    },
  };
}

async function flushObservers() {
  await Promise.resolve();
  await Promise.resolve();
}

describe('themeSnapshot', () => {
  beforeEach(() => {
    document.documentElement.removeAttribute('data-theme-pack');
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    document.documentElement.removeAttribute('data-theme-pack');
    document.documentElement.removeAttribute('data-theme');
  });

  it('returns the effective pack and mode from root attributes and system preference', () => {
    installMatchMediaStub(true);
    document.documentElement.setAttribute('data-theme-pack', 'modern');

    expect(getThemeSnapshot()).toEqual({ pack: 'modern', mode: 'light' });
  });

  it('returns the same snapshot object until the effective pack or mode changes', async () => {
    const matchMediaControl = installMatchMediaStub(false);
    document.documentElement.setAttribute('data-theme-pack', 'modern');

    const firstSnapshot = getThemeSnapshot();
    const secondSnapshot = getThemeSnapshot();

    expect(secondSnapshot).toBe(firstSnapshot);

    const unsubscribe = subscribeToThemeChanges(() => {});
    document.documentElement.setAttribute('data-theme', 'light');
    await flushObservers();

    const thirdSnapshot = getThemeSnapshot();
    expect(thirdSnapshot).toEqual({ pack: 'modern', mode: 'light' });
    expect(thirdSnapshot).not.toBe(firstSnapshot);
    expect(getThemeSnapshot()).toBe(thirdSnapshot);

    document.documentElement.removeAttribute('data-theme');
    await flushObservers();
    matchMediaControl.setMatches(true);

    const fourthSnapshot = getThemeSnapshot();
    expect(fourthSnapshot).toEqual({ pack: 'modern', mode: 'light' });

    unsubscribe();
  });

  it('notifies one shared subscription for root and matchMedia changes', async () => {
    const matchMediaControl = installMatchMediaStub(false);
    const listener = vi.fn();
    const unsubscribe = subscribeToThemeChanges(listener);

    document.documentElement.setAttribute('data-theme-pack', 'modern');
    await flushObservers();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(getThemeSnapshot()).toEqual({ pack: 'modern', mode: 'dark' });

    document.documentElement.setAttribute('data-theme', 'light');
    await flushObservers();
    expect(listener).toHaveBeenCalledTimes(2);
    expect(getThemeSnapshot()).toEqual({ pack: 'modern', mode: 'light' });

    matchMediaControl.setMatches(true);
    expect(listener).toHaveBeenCalledTimes(2);

    document.documentElement.removeAttribute('data-theme');
    await flushObservers();
    expect(listener).toHaveBeenCalledTimes(2);

    matchMediaControl.setMatches(false);
    expect(listener).toHaveBeenCalledTimes(3);
    expect(getThemeSnapshot()).toEqual({ pack: 'modern', mode: 'dark' });

    unsubscribe();
    document.documentElement.setAttribute('data-theme-pack', 'serif');
    await flushObservers();
    expect(listener).toHaveBeenCalledTimes(3);
  });

  it('shares one matchMedia subscription across listeners and cleans it up once', () => {
    const { setMatches: _setMatches } = installMatchMediaStub(false);
    const matchMedia = vi.mocked(window.matchMedia);

    const listenerA = vi.fn();
    const listenerB = vi.fn();
    const unsubscribeA = subscribeToThemeChanges(listenerA);
    const unsubscribeB = subscribeToThemeChanges(listenerB);

    const mediaQueryList = matchMedia.mock.results[0]?.value as MediaQueryList;

    expect(matchMedia).toHaveBeenCalledTimes(1);
    expect(mediaQueryList.addEventListener).toHaveBeenCalledTimes(1);

    unsubscribeA();
    expect(mediaQueryList.removeEventListener).not.toHaveBeenCalled();

    unsubscribeB();
    expect(mediaQueryList.removeEventListener).toHaveBeenCalledTimes(1);
  });
});
