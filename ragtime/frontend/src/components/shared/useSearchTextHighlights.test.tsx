import { renderHook, waitFor } from '@testing-library/react';
import { createRef, type MutableRefObject } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { useSearchTextHighlights } from './useSearchTextHighlights';

class TestHighlight {
  ranges = new Set<Range>();

  add(...ranges: Range[]) {
    ranges.forEach((range) => this.ranges.add(range));
  }
}

const highlightRegistry = new Map<string, TestHighlight>();
const originalHighlight = globalThis.Highlight;
const originalCss = globalThis.CSS;

function installHighlightApi() {
  Object.defineProperty(globalThis, 'Highlight', { configurable: true, value: TestHighlight });
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    value: { highlights: highlightRegistry },
  });
}

function renderHighlights(
  root: HTMLElement,
  terms: readonly string[],
  includeSelector = '[data-highlightable]',
  excludeSelector = '[data-no-highlight]',
) {
  const rootRef = createRef<HTMLElement>() as MutableRefObject<HTMLElement | null>;
  rootRef.current = root;
  return renderHook(
    ({ currentTerms }) =>
      useSearchTextHighlights({
        rootRef,
        terms: currentTerms,
        highlightName: 'settings-search',
        includeSelector,
        excludeSelector,
      }),
    { initialProps: { currentTerms: terms } },
  );
}

function rangeTexts() {
  return [...(highlightRegistry.get('settings-search')?.ranges ?? [])].map((range) =>
    range.toString(),
  );
}

afterEach(() => {
  highlightRegistry.clear();
  Object.defineProperty(globalThis, 'Highlight', { configurable: true, value: originalHighlight });
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    value: originalCss,
  });
  vi.restoreAllMocks();
});

describe('useSearchTextHighlights', () => {
  it('highlights literal case-insensitive text while treating whitespace and ampersands as equivalent', async () => {
    installHighlightApi();
    const root = document.createElement('section');
    root.innerHTML = '<p data-highlightable>API  Keys &amp; <strong>Tokens</strong></p>';

    renderHighlights(root, ['api keys and tokens', 'keys & tokens', '[tokens]']);

    await waitFor(() => expect(rangeTexts()).toEqual(['API  Keys & Tokens', 'Keys & Tokens']));
  });

  it('includes selected text ancestors and excludes unsafe descendants', async () => {
    installHighlightApi();
    const root = document.createElement('section');
    root.innerHTML = `
      <article data-highlightable><span>Visible setting</span></article>
      <article data-highlightable data-no-highlight>Hidden setting</article>
      <input value="setting" />
    `;

    renderHighlights(root, ['setting']);

    await waitFor(() => expect(rangeTexts()).toEqual(['setting']));
  });

  it('updates ranges for new terms and relevant DOM mutations', async () => {
    installHighlightApi();
    const root = document.createElement('section');
    root.innerHTML = '<p data-highlightable>First setting</p>';
    const hook = renderHighlights(root, ['first']);

    await waitFor(() => expect(rangeTexts()).toEqual(['First']));
    hook.rerender({ currentTerms: ['second'] });
    root.querySelector('p')!.textContent = 'Second setting';

    await waitFor(() => expect(rangeTexts()).toEqual(['Second']));
  });

  it('does nothing when the Custom Highlight API is unavailable', () => {
    const root = document.createElement('section');
    root.innerHTML = '<p data-highlightable>Visible setting</p>';

    expect(() => renderHighlights(root, ['setting'])).not.toThrow();
    expect(highlightRegistry.size).toBe(0);
  });

  it('cancels scheduled work, disconnects observation, and removes its registry entry on cleanup', async () => {
    installHighlightApi();
    const disconnect = vi.spyOn(MutationObserver.prototype, 'disconnect');
    const cancelFrame = vi.spyOn(window, 'cancelAnimationFrame');
    const root = document.createElement('section');
    root.innerHTML = '<p data-highlightable>Visible setting</p>';
    const hook = renderHighlights(root, ['setting']);

    await waitFor(() => expect(highlightRegistry.has('settings-search')).toBe(true));
    root.querySelector('p')!.textContent = 'Updated setting';
    await Promise.resolve();
    hook.unmount();

    expect(disconnect).toHaveBeenCalled();
    expect(cancelFrame).toHaveBeenCalled();
    expect(highlightRegistry.has('settings-search')).toBe(false);
  });
});
