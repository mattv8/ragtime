import { useLayoutEffect, useState } from 'react';

/**
 * Detects whether an element's content is horizontally truncated (clipped by
 * `text-overflow: ellipsis` / `overflow: hidden`).
 *
 * Returns a callback ref to attach to the target element and a boolean that is
 * `true` only when the element's full content does not fit within its box.
 *
 * The measurement re-runs whenever the element resizes (via `ResizeObserver`,
 * so it stays correct through responsive layout changes) and whenever the
 * optional `dependency` (typically the rendered text) changes.
 *
 * Implementation note: the observer is created and torn down inside the same
 * effect. Splitting them (create in a callback ref, disconnect in an effect
 * cleanup) breaks under React StrictMode, whose mount -> cleanup -> remount
 * cycle re-runs effects without re-invoking the ref, permanently
 * disconnecting the observer.
 *
 * Use this to suppress tooltips/popovers that would only repeat text already
 * fully visible on screen.
 */
export function useIsTruncated<T extends HTMLElement = HTMLElement>(
  dependency?: unknown,
): [(node: T | null) => void, boolean] {
  const [node, setNode] = useState<T | null>(null);
  const [isTruncated, setIsTruncated] = useState(false);

  useLayoutEffect(() => {
    if (!node) {
      setIsTruncated(false);
      return;
    }

    const measure = () => {
      // A 1px tolerance avoids sub-pixel rounding false positives.
      setIsTruncated(node.scrollWidth - node.clientWidth > 1);
    };

    measure();

    // Web-font swaps can change glyph metrics (and thus scrollWidth) without
    // resizing the element's layout box, which ResizeObserver would miss.
    let cancelled = false;
    if (typeof document !== 'undefined' && document.fonts?.ready) {
      document.fonts.ready.then(() => {
        if (!cancelled) measure();
      });
    }

    if (typeof ResizeObserver === 'undefined') {
      return () => {
        cancelled = true;
      };
    }
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, [node, dependency]);

  return [setNode, isTruncated];
}
