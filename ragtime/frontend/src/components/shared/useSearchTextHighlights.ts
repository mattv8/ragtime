import { useEffect, type RefObject } from 'react';

interface SearchTextHighlightsOptions {
  rootRef: RefObject<HTMLElement>;
  terms: readonly string[];
  highlightName: string;
  includeSelector: string;
  excludeSelector: string;
}

interface HighlightInstance {
  add(...ranges: Range[]): void;
}

interface HighlightRegistry {
  set(name: string, highlight: HighlightInstance): void;
  delete(name: string): boolean;
}

interface HighlightConstructor {
  new (): HighlightInstance;
}

interface HighlightApi {
  Highlight: HighlightConstructor;
  highlights: HighlightRegistry;
}

interface TextPosition {
  startNode: Text;
  startOffset: number;
  endNode: Text;
  endOffset: number;
}

interface NormalizedText {
  text: string;
  positions: TextPosition[];
}

// Marks where excluded DOM text was skipped so matches cannot bridge it.
const SEGMENT_BOUNDARY = '\u0000';

function getHighlightApi(): HighlightApi | null {
  const browser = globalThis as typeof globalThis & {
    Highlight?: unknown;
    CSS?: { highlights?: unknown };
  };
  if (typeof browser.Highlight !== 'function' || !browser.CSS?.highlights) {
    return null;
  }

  return {
    Highlight: browser.Highlight as HighlightConstructor,
    highlights: browser.CSS.highlights as HighlightRegistry,
  };
}

function normalizeTerms(terms: readonly string[]): string[] {
  return [...new Set(terms.map(normalizeValue).filter(Boolean))];
}

function normalizeValue(value: string): string {
  let normalized = '';
  let previousWasWhitespace = false;

  for (const character of value.replace(/&/g, ' and ').trim()) {
    if (character.trim() === '') {
      if (!previousWasWhitespace && normalized) {
        normalized += ' ';
      }
      previousWasWhitespace = true;
      continue;
    }

    normalized += character.toLowerCase();
    previousWasWhitespace = false;
  }

  return normalized.trim();
}

function getIncludedElements(root: HTMLElement, includeSelector: string): HTMLElement[] {
  const included = [
    ...(root.matches(includeSelector) ? [root] : []),
    ...Array.from(root.querySelectorAll<HTMLElement>(includeSelector)),
  ];

  return included.filter(
    (element) => !included.some((ancestor) => ancestor !== element && ancestor.contains(element)),
  );
}

function isExcluded(textNode: Text, root: HTMLElement, excludeSelector: string): boolean {
  const parent = textNode.parentElement;
  const excluded = parent?.closest(excludeSelector);
  return Boolean(excluded && root.contains(excluded));
}

function normalizeElementText(
  element: HTMLElement,
  root: HTMLElement,
  excludeSelector: string,
): NormalizedText {
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
  const positions: TextPosition[] = [];
  let normalized = '';
  let previousWasWhitespace = false;

  const appendSpace = (position: TextPosition) => {
    if (!previousWasWhitespace && normalized) {
      normalized += ' ';
      positions.push(position);
      previousWasWhitespace = true;
    }
  };
  const appendValue = (value: string, position: TextPosition) => {
    normalized += value;
    for (let index = 0; index < value.length; index += 1) {
      positions.push(position);
    }
    previousWasWhitespace = false;
  };
  const appendBoundary = (textNode: Text) => {
    normalized += SEGMENT_BOUNDARY;
    positions.push({ startNode: textNode, startOffset: 0, endNode: textNode, endOffset: 0 });
    previousWasWhitespace = false;
  };

  let textNode = walker.nextNode() as Text | null;
  while (textNode) {
    if (isExcluded(textNode, root, excludeSelector)) {
      appendBoundary(textNode);
    } else {
      let offset = 0;
      for (const character of textNode.data) {
        const position: TextPosition = {
          startNode: textNode,
          startOffset: offset,
          endNode: textNode,
          endOffset: offset + character.length,
        };
        offset += character.length;

        if (character.trim() === '') {
          appendSpace(position);
        } else if (character === '&') {
          appendSpace(position);
          appendValue('and', position);
          appendSpace(position);
        } else {
          appendValue(character.toLowerCase(), position);
        }
      }
    }
    textNode = walker.nextNode() as Text | null;
  }

  return { text: normalized.trimEnd(), positions };
}

function addTermRanges(highlight: HighlightInstance, text: NormalizedText, term: string): void {
  let start = text.text.indexOf(term);
  while (start !== -1) {
    const first = text.positions[start];
    const last = text.positions[start + term.length - 1];
    if (first && last) {
      const range = document.createRange();
      range.setStart(first.startNode, first.startOffset);
      range.setEnd(last.endNode, last.endOffset);
      highlight.add(range);
    }
    start = text.text.indexOf(term, start + term.length);
  }
}

function updateHighlights(
  root: HTMLElement,
  terms: readonly string[],
  includeSelector: string,
  excludeSelector: string,
  api: HighlightApi,
  highlightName: string,
): void {
  const highlight = new api.Highlight();
  for (const element of getIncludedElements(root, includeSelector)) {
    const text = normalizeElementText(element, root, excludeSelector);
    for (const term of terms) {
      addTermRanges(highlight, text, term);
    }
  }
  api.highlights.set(highlightName, highlight);
}

export function useSearchTextHighlights({
  rootRef,
  terms,
  highlightName,
  includeSelector,
  excludeSelector,
}: SearchTextHighlightsOptions): void {
  useEffect(() => {
    const root = rootRef.current;
    const api = getHighlightApi();
    const normalizedTerms = normalizeTerms(terms);
    if (!root || !api || !normalizedTerms.length) {
      return;
    }

    let frame: number | null = null;
    const update = () => {
      frame = null;
      updateHighlights(root, normalizedTerms, includeSelector, excludeSelector, api, highlightName);
    };
    const scheduleUpdate = () => {
      if (frame !== null) return;
      frame = window.requestAnimationFrame(update);
    };
    const observer = new MutationObserver(scheduleUpdate);

    scheduleUpdate();
    observer.observe(root, { childList: true, characterData: true, subtree: true });

    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      observer.disconnect();
      api.highlights.delete(highlightName);
    };
  }, [excludeSelector, highlightName, includeSelector, rootRef, terms]);
}
