import { useLayoutEffect, useState } from 'react';
import type { RefObject } from 'react';
import {
  computeAnchoredMenuPosition,
  type AnchoredMenuPlacement,
  type AnchoredMenuPosition,
} from '@/utils/anchoredMenuPosition';

interface UseAnchoredMenuPositionOptions {
  open: boolean;
  triggerRef: RefObject<HTMLElement>;
  menuRef: RefObject<HTMLElement>;
  placement: AnchoredMenuPlacement;
  measureKey?: unknown;
}

function positionsEqual(
  previous: AnchoredMenuPosition | null,
  next: AnchoredMenuPosition,
): boolean {
  return (
    previous?.top === next.top &&
    previous.left === next.left &&
    previous.maxHeight === next.maxHeight &&
    previous.maxWidth === next.maxWidth
  );
}

export function useAnchoredMenuPosition({
  open,
  triggerRef,
  menuRef,
  placement,
  measureKey,
}: UseAnchoredMenuPositionOptions): AnchoredMenuPosition | null {
  const [position, setPosition] = useState<AnchoredMenuPosition | null>(null);

  useLayoutEffect(() => {
    if (!open) {
      setPosition((current) => (current === null ? current : null));
      return;
    }

    const measure = () => {
      const trigger = triggerRef.current;
      const menu = menuRef.current;
      if (!trigger || !menu) return;
      const next = computeAnchoredMenuPosition(
        trigger.getBoundingClientRect(),
        menu.getBoundingClientRect(),
        placement,
      );
      setPosition((current) => (positionsEqual(current, next) ? current : next));
    };

    measure();
    window.addEventListener('scroll', measure, true);
    window.addEventListener('resize', measure);
    const observer =
      typeof ResizeObserver === 'undefined' ? undefined : new ResizeObserver(measure);
    if (observer && menuRef.current) observer.observe(menuRef.current);

    return () => {
      window.removeEventListener('scroll', measure, true);
      window.removeEventListener('resize', measure);
      observer?.disconnect();
    };
  }, [menuRef, measureKey, open, placement, triggerRef]);

  return position;
}
