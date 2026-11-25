import { act, renderHook } from '@testing-library/react';
import { createRef } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { computeFitLines, useFitLineClamp } from './useFitLineClamp';

describe('computeFitLines', () => {
  it.each([
    [340, 150, 19, 9],
    [300, 230, 19, 3],
    [300, 295, 19, 1],
    [0, 150, 19, null],
    [300, 150, 0, null],
    [300, NaN, 19, null],
  ])('fits whole lines for %s, %s, %s', (capPx, othersPx, lineHeightPx, expected) => {
    expect(computeFitLines({ capPx, othersPx, lineHeightPx, minLines: 1, maxLines: 9 })).toBe(
      expected,
    );
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('useFitLineClamp', () => {
  it('uses the CSS fallback when ResizeObserver is unavailable', () => {
    vi.stubGlobal('ResizeObserver', undefined);
    const cardRef = createRef<HTMLDivElement>();
    const regionRef = createRef<HTMLDivElement>();
    const textRef = createRef<HTMLParagraphElement>();
    Object.assign(cardRef, { current: document.createElement('div') });
    Object.assign(regionRef, { current: document.createElement('div') });
    Object.assign(textRef, { current: document.createElement('p') });
    const { result } = renderHook(() =>
      useFitLineClamp({ cardRef, regionRef, textRef, enabled: true }),
    );
    expect(result.current).toEqual({ lines: null, isClamped: false });
    expect(cardRef.current?.style.getPropertyValue('--tool-card-description-lines')).toBe('');
  });

  it('retains the last fit at zero width, avoids redundant writes, and clears on disable', () => {
    let resize: ResizeObserverCallback = () => {};
    const observe = vi.fn();
    const disconnect = vi.fn();
    class ResizeObserverStub {
      constructor(callback: ResizeObserverCallback) {
        resize = callback;
      }
      observe = observe;
      disconnect = disconnect;
    }
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
    let scheduled: FrameRequestCallback | undefined;
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      scheduled = callback;
      return 1;
    });
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    const card = document.createElement('div');
    const region = document.createElement('div');
    const text = document.createElement('p');
    const header = document.createElement('div');
    card.append(header, region);
    region.append(text);
    const cardRef = createRef<HTMLDivElement>();
    const regionRef = createRef<HTMLDivElement>();
    const textRef = createRef<HTMLParagraphElement>();
    Object.assign(cardRef, { current: card });
    Object.assign(regionRef, { current: region });
    Object.assign(textRef, { current: text });
    let width = 300;
    vi.spyOn(card, 'getBoundingClientRect').mockImplementation(() => ({ width }) as DOMRect);
    vi.spyOn(header, 'getBoundingClientRect').mockReturnValue({ height: 230 } as DOMRect);
    Object.defineProperties(text, {
      scrollHeight: { value: 80, configurable: true },
      clientHeight: { value: 40, configurable: true },
    });
    vi.spyOn(window, 'getComputedStyle').mockImplementation(
      (element) =>
        ({
          paddingTop: '0',
          paddingBottom: '0',
          borderTopWidth: '0',
          borderBottomWidth: '0',
          rowGap: '0',
          lineHeight: element === text ? '19px' : 'normal',
        }) as CSSStyleDeclaration,
    );
    const write = vi.spyOn(card.style, 'setProperty');
    const { result, rerender } = renderHook(
      ({ enabled }) => useFitLineClamp({ cardRef, regionRef, textRef, enabled }),
      { initialProps: { enabled: true } },
    );
    const flush = () => {
      const callback = scheduled;
      scheduled = undefined;
      act(() => callback?.(0));
    };
    flush();
    expect(result.current.lines).toBe(3);
    expect(result.current.isClamped).toBe(true);
    expect(observe).toHaveBeenCalledWith(card);
    expect(observe).toHaveBeenCalledWith(header);
    expect(observe).not.toHaveBeenCalledWith(region);
    expect(card.style.getPropertyValue('--tool-card-description-lines')).toBe('3');
    act(() => resize([], {} as ResizeObserver));
    flush();
    expect(write).toHaveBeenCalledTimes(1);
    width = 0;
    act(() => resize([], {} as ResizeObserver));
    flush();
    expect(result.current.lines).toBe(3);
    rerender({ enabled: false });
    expect(card.style.getPropertyValue('--tool-card-description-lines')).toBe('');
    expect(result.current.lines).toBeNull();
    expect(disconnect).toHaveBeenCalled();
  });
});
