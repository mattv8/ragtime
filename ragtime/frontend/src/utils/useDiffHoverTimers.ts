import { useCallback, useMemo, useRef } from 'react';

const DEFAULT_HOVER_DELAY_MS = 500;
const DEFAULT_DISMISS_DELAY_MS = 500;

interface UseDiffHoverTimersOptions {
  hoverDelayMs?: number;
  dismissDelayMs?: number;
  onDismiss: () => void;
}

export interface DiffHoverTimers {
  startHover: (loadCallback: () => void) => void;
  endHover: () => void;
  dismiss: () => void;
  scheduleDismiss: () => void;
  cancelDismiss: () => void;
  overlayMouseEnter: () => void;
  overlayMouseLeave: () => void;
  overlayClick: () => void;
}

export function useDiffHoverTimers(options: UseDiffHoverTimersOptions): DiffHoverTimers {
  const {
    hoverDelayMs = DEFAULT_HOVER_DELAY_MS,
    dismissDelayMs = DEFAULT_DISMISS_DELAY_MS,
    onDismiss,
  } = options;

  const hoverTimerRef = useRef<number | null>(null);
  const dismissTimerRef = useRef<number | null>(null);
  const pinnedRef = useRef(false);
  const enteredOverlayRef = useRef(false);
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  const cancelDismiss = useCallback(() => {
    if (dismissTimerRef.current !== null) {
      window.clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
  }, []);

  const dismiss = useCallback(() => {
    pinnedRef.current = false;
    enteredOverlayRef.current = false;
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    cancelDismiss();
    onDismissRef.current();
  }, [cancelDismiss]);

  const scheduleDismiss = useCallback(() => {
    if (pinnedRef.current) return;
    cancelDismiss();
    dismissTimerRef.current = window.setTimeout(() => {
      dismissTimerRef.current = null;
      onDismissRef.current();
    }, dismissDelayMs);
  }, [cancelDismiss, dismissDelayMs]);

  const startHover = useCallback((loadCallback: () => void) => {
    cancelDismiss();
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
    }
    hoverTimerRef.current = window.setTimeout(() => {
      hoverTimerRef.current = null;
      loadCallback();
    }, hoverDelayMs);
  }, [cancelDismiss, hoverDelayMs]);

  const endHover = useCallback(() => {
    if (hoverTimerRef.current !== null) {
      window.clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    if (enteredOverlayRef.current) {
      scheduleDismiss();
    }
  }, [scheduleDismiss]);

  const overlayMouseEnter = useCallback(() => {
    enteredOverlayRef.current = true;
    cancelDismiss();
  }, [cancelDismiss]);

  const overlayMouseLeave = useCallback(() => {
    scheduleDismiss();
  }, [scheduleDismiss]);

  const overlayClick = useCallback(() => {
    pinnedRef.current = true;
  }, []);

  return useMemo(() => ({
    startHover,
    endHover,
    dismiss,
    scheduleDismiss,
    cancelDismiss,
    overlayMouseEnter,
    overlayMouseLeave,
    overlayClick,
  }), [startHover, endHover, dismiss, scheduleDismiss, cancelDismiss, overlayMouseEnter, overlayMouseLeave, overlayClick]);
}
