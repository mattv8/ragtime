import { useEffect, useRef, type RefObject } from 'react';

export function getFocusableElements(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  );
}

export function useDialogFocusTrap(
  containerRef: RefObject<HTMLDivElement | null>,
  initialFocusRef: RefObject<HTMLElement | null>,
  { disableEscape, onEscape }: { disableEscape?: boolean; onEscape?: () => void } = {},
) {
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const disableEscapeRef = useRef(disableEscape);
  const onEscapeRef = useRef(onEscape);

  useEffect(() => {
    disableEscapeRef.current = disableEscape;
  }, [disableEscape]);

  useEffect(() => {
    onEscapeRef.current = onEscape;
  }, [onEscape]);

  useEffect(() => {
    returnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusTimer = window.setTimeout(() => {
      initialFocusRef.current?.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      const container = containerRef.current;
      if (!container) return;

      if (event.key === 'Escape') {
        if (!disableEscapeRef.current) {
          event.preventDefault();
          onEscapeRef.current?.();
        }
        return;
      }

      if (event.key !== 'Tab') {
        return;
      }

      const focusableElements = getFocusableElements(container);
      if (focusableElements.length === 0) {
        event.preventDefault();
        return;
      }

      const activeElement = document.activeElement as HTMLElement | null;
      const currentIndex = activeElement ? focusableElements.indexOf(activeElement) : -1;
      const nextIndex = event.shiftKey
        ? currentIndex <= 0
          ? focusableElements.length - 1
          : currentIndex - 1
        : currentIndex === -1 || currentIndex === focusableElements.length - 1
          ? 0
          : currentIndex + 1;

      event.preventDefault();
      focusableElements[nextIndex]?.focus();
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener('keydown', handleKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [containerRef, initialFocusRef, disableEscapeRef, onEscapeRef]);
}
