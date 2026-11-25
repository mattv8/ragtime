import { type RefObject, useLayoutEffect, useState } from 'react';

interface FitLinesInput {
  capPx: number;
  othersPx: number;
  lineHeightPx: number;
  minLines: number;
  maxLines: number;
}

export function computeFitLines({
  capPx,
  othersPx,
  lineHeightPx,
  minLines,
  maxLines,
}: FitLinesInput): number | null {
  if (
    [capPx, othersPx, lineHeightPx, minLines, maxLines].some(Number.isNaN) ||
    capPx <= 0 ||
    lineHeightPx <= 0
  ) {
    return null;
  }
  return Math.max(minLines, Math.min(maxLines, Math.floor((capPx - othersPx) / lineHeightPx)));
}

interface FitLineClampOptions {
  cardRef: RefObject<HTMLElement>;
  regionRef: RefObject<HTMLElement>;
  textRef: RefObject<HTMLElement>;
  enabled: boolean;
  minLines?: number;
  maxLines?: number;
}

export function useFitLineClamp({
  cardRef,
  regionRef,
  textRef,
  enabled,
  minLines = 1,
  maxLines = 9,
}: FitLineClampOptions): { lines: number | null; isClamped: boolean } {
  const [lines, setLines] = useState<number | null>(null);
  const [isClamped, setIsClamped] = useState(false);

  useLayoutEffect(() => {
    const card = cardRef.current;
    const region = regionRef.current;
    const text = textRef.current;
    if (!card) return;
    if (!enabled) {
      card.style.removeProperty('--tool-card-description-lines');
      setLines(null);
      setIsClamped(false);
      return;
    }
    if (!region || !text || typeof ResizeObserver === 'undefined') return;

    let frame = 0;
    let cancelled = false;
    const measure = () => {
      frame = 0;
      const cardStyle = getComputedStyle(card);
      const children = Array.from(card.children);
      const numeric = (value: string) => Number.parseFloat(value) || 0;
      const othersPx =
        numeric(cardStyle.paddingTop) +
        numeric(cardStyle.paddingBottom) +
        numeric(cardStyle.borderTopWidth) +
        numeric(cardStyle.borderBottomWidth) +
        numeric(cardStyle.rowGap) * Math.max(0, children.length - 1) +
        children.reduce(
          (total, child) => total + (child === region ? 0 : child.getBoundingClientRect().height),
          0,
        );
      const next = computeFitLines({
        capPx: card.getBoundingClientRect().width,
        othersPx,
        lineHeightPx: Number.parseFloat(getComputedStyle(text).lineHeight),
        minLines,
        maxLines,
      });
      if (next === null) return;
      if (card.style.getPropertyValue('--tool-card-description-lines') !== String(next)) {
        card.style.setProperty('--tool-card-description-lines', String(next));
      }
      setLines(next);
      setIsClamped(text.scrollHeight > text.clientHeight + 1);
    };
    const schedule = () => {
      if (!frame && !cancelled) frame = requestAnimationFrame(measure);
    };
    const resize = new ResizeObserver(schedule);
    const observeChildren = () => {
      resize.disconnect();
      resize.observe(card);
      Array.from(card.children).forEach((child) => {
        if (child !== region) resize.observe(child);
      });
      resize.observe(text);
      schedule();
    };
    const mutation = new MutationObserver(observeChildren);
    mutation.observe(card, { childList: true });
    mutation.observe(text, { childList: true, characterData: true, subtree: true });
    observeChildren();
    document.fonts?.ready.then(schedule);
    return () => {
      cancelled = true;
      if (frame) cancelAnimationFrame(frame);
      resize.disconnect();
      mutation.disconnect();
    };
  }, [cardRef, regionRef, textRef, enabled, minLines, maxLines]);

  return { lines, isClamped };
}
