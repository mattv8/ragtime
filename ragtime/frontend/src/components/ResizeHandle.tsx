import { useCallback, useEffect, useRef } from 'react';
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown } from 'lucide-react';

interface ResizeHandleProps {
  /** 'horizontal' = dragging left/right, 'vertical' = dragging up/down */
  direction: 'horizontal' | 'vertical';
  /** Accessible name for the separator */
  ariaLabel: string;
  /** Current pane size or split percentage */
  value: number;
  /** Minimum expanded size */
  min: number;
  /** Maximum expanded size */
  max: number;
  /** Unit for announcing the current value */
  valueUnit: 'pixels' | 'percent';
  /** Called continuously during drag with the delta in px from drag start */
  onResize: (delta: number) => void;
  /** Called for absolute moves such as Home, End, and collapse/restore */
  onResizeTo: (value: number) => void;
  /** Optional className override */
  className?: string;
  /**
   * Which side adjacent to this handle is currently collapsed.
   * 'before' = the pane before (left/top), 'after' = the pane after (right/bottom), undefined = nothing collapsed.
   */
  collapsed?: 'before' | 'after';
  /** Optional collapse/restore support for Enter and pointer restore */
  collapsible?: {
    side: 'before' | 'after';
    restoreValue: number;
  };
  /** Called when a drag gesture ends or collapsed handle is activated */
  onResizeEnd?: () => void;
  /**
   * Selector, resolved within the handle's parent, for the panes the sash bar runs beside.
   * Gaps between the matched panes become breaks in the bar. Presentation only; themes opt
   * in by masking the bar with `--resize-handle-bar-mask`.
   */
  barSegmentSelector?: string;
}

type BarSegment = { start: number; end: number };

function buildBarSegmentMask(
  handle: HTMLElement,
  targets: Element[],
  axis: 'x' | 'y',
): string | null {
  const handleRect = handle.getBoundingClientRect();
  const origin = axis === 'y' ? handleRect.top : handleRect.left;
  const length = axis === 'y' ? handleRect.height : handleRect.width;
  if (length <= 0) return null;

  const segments: BarSegment[] = [];
  for (const target of targets) {
    const rect = target.getBoundingClientRect();
    const size = axis === 'y' ? rect.height : rect.width;
    if (size <= 0) continue;
    const start = Math.max(0, Math.round((axis === 'y' ? rect.top : rect.left) - origin));
    const end = Math.min(length, Math.round((axis === 'y' ? rect.bottom : rect.right) - origin));
    if (end > start) segments.push({ start, end });
  }
  segments.sort((a, b) => a.start - b.start);

  const merged: BarSegment[] = [];
  for (const segment of segments) {
    const last = merged[merged.length - 1];
    if (last && segment.start <= last.end) last.end = Math.max(last.end, segment.end);
    else merged.push({ ...segment });
  }
  if (merged.length < 2) return null;

  const stops = ['#000 0'];
  for (let i = 0; i < merged.length - 1; i += 1) {
    const gapStart = merged[i].end;
    const gapEnd = merged[i + 1].start;
    stops.push(
      `#000 ${gapStart}px`,
      `transparent ${gapStart}px`,
      `transparent ${gapEnd}px`,
      `#000 ${gapEnd}px`,
    );
  }
  stops.push('#000 100%');
  return `linear-gradient(to ${axis === 'y' ? 'bottom' : 'right'}, ${stops.join(', ')})`;
}

/** Keeps `--resize-handle-bar-mask` in sync with the panes matched by `selector`. */
function useBarSegmentMask(
  handleRef: React.RefObject<HTMLDivElement | null>,
  direction: 'horizontal' | 'vertical',
  selector: string | undefined,
) {
  useEffect(() => {
    const handle = handleRef.current;
    const root = handle?.parentElement;
    if (
      !selector ||
      !handle ||
      !root ||
      typeof ResizeObserver === 'undefined' ||
      typeof MutationObserver === 'undefined'
    ) {
      return;
    }

    const axis = direction === 'horizontal' ? 'y' : 'x';
    let targets: Element[] = [];
    let frame: number | null = null;

    const applyMask = () => {
      frame = null;
      const mask = buildBarSegmentMask(handle, targets, axis);
      if (mask) handle.style.setProperty('--resize-handle-bar-mask', mask);
      else handle.style.removeProperty('--resize-handle-bar-mask');
    };
    const scheduleMask = () => {
      if (frame === null) frame = window.requestAnimationFrame(applyMask);
    };

    const resizeObserver = new ResizeObserver(scheduleMask);
    resizeObserver.observe(handle);

    const refreshTargets = () => {
      const next = Array.from(root.querySelectorAll(selector));
      if (next.length === targets.length && next.every((el, i) => el === targets[i])) return;
      targets.forEach((el) => resizeObserver.unobserve(el));
      next.forEach((el) => resizeObserver.observe(el));
      targets = next;
      scheduleMask();
    };

    // Streaming content mutates the subtree constantly, so only re-query when a
    // tracked pane left the DOM or a newly added node could contain one.
    const mutationObserver = new MutationObserver((records) => {
      const stale =
        targets.some((el) => !el.isConnected) ||
        records.some((record) =>
          Array.from(record.addedNodes).some(
            (node) =>
              node instanceof Element &&
              (node.matches(selector) || node.querySelector(selector) !== null),
          ),
        );
      if (stale) refreshTargets();
    });
    mutationObserver.observe(root, { childList: true, subtree: true });

    refreshTargets();

    return () => {
      mutationObserver.disconnect();
      resizeObserver.disconnect();
      if (frame !== null) window.cancelAnimationFrame(frame);
      handle.style.removeProperty('--resize-handle-bar-mask');
    };
  }, [direction, handleRef, selector]);
}

export function ResizeHandle({
  direction,
  ariaLabel,
  value,
  min,
  max,
  valueUnit,
  onResize,
  onResizeTo,
  className,
  collapsed,
  collapsible,
  onResizeEnd,
  barSegmentSelector,
}: ResizeHandleProps) {
  const handleRef = useRef<HTMLDivElement>(null);
  useBarSegmentMask(handleRef, direction, barSegmentSelector);
  const startPos = useRef(0);
  const isDragging = useRef(false);
  const pendingDelta = useRef(0);
  const resizeFrame = useRef<number | null>(null);
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  const flushPendingResize = useCallback(() => {
    resizeFrame.current = null;
    const delta = pendingDelta.current;
    pendingDelta.current = 0;
    if (delta !== 0) {
      onResizeRef.current(delta);
    }
  }, []);

  const cancelPendingResize = useCallback(() => {
    if (resizeFrame.current !== null) {
      window.cancelAnimationFrame(resizeFrame.current);
      resizeFrame.current = null;
    }
    pendingDelta.current = 0;
  }, []);

  useEffect(() => {
    return () => {
      cancelPendingResize();
      if (isDragging.current) {
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        isDragging.current = false;
      }
    };
  }, [cancelPendingResize]);

  useEffect(() => {
    if (collapsed && isDragging.current) {
      cancelPendingResize();
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      isDragging.current = false;
    }
  }, [cancelPendingResize, collapsed]);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      if (collapsed && collapsible) {
        onResizeTo(collapsible.restoreValue);
      }
      e.currentTarget.setPointerCapture(e.pointerId);
      startPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
      isDragging.current = true;
    },
    [collapsed, collapsible, direction, onResizeTo],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
      e.preventDefault();
      const pos = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = pos - startPos.current;
      startPos.current = pos;
      pendingDelta.current += delta;
      if (resizeFrame.current === null) {
        resizeFrame.current = window.requestAnimationFrame(flushPendingResize);
      }
    },
    [direction, flushPendingResize],
  );

  const finishPointerDrag = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      try {
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
          e.currentTarget.releasePointerCapture(e.pointerId);
        }
      } catch {
        // Ignore capture release errors
      }

      if (isDragging.current) {
        if (resizeFrame.current !== null || pendingDelta.current !== 0) {
          flushPendingResize();
        }
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        isDragging.current = false;
        onResizeEnd?.();
      }
    },
    [flushPendingResize, onResizeEnd],
  );

  const cls = className ?? `resize-handle resize-handle-${direction}`;
  const isCollapsed = Boolean(collapsed);
  const separatorOrientation = direction === 'horizontal' ? 'vertical' : 'horizontal';
  const ariaValueNow = isCollapsed ? 0 : value;
  const roundedValue = Math.round(value);
  const ariaValueText = isCollapsed
    ? 'Collapsed'
    : valueUnit === 'percent'
      ? `${roundedValue}%`
      : `${roundedValue} pixels`;

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      let handled = false;
      const step = event.shiftKey ? 32 : 8;

      if (direction === 'horizontal') {
        if (event.key === 'ArrowLeft') {
          onResize(-step);
          handled = true;
        } else if (event.key === 'ArrowRight') {
          onResize(step);
          handled = true;
        }
      } else if (event.key === 'ArrowUp') {
        onResize(-step);
        handled = true;
      } else if (event.key === 'ArrowDown') {
        onResize(step);
        handled = true;
      }

      if (!handled && event.key === 'Home') {
        onResizeTo(min);
        handled = true;
      } else if (!handled && event.key === 'End') {
        onResizeTo(max);
        handled = true;
      } else if (!handled && event.key === 'Enter' && collapsible) {
        onResizeTo(isCollapsed ? collapsible.restoreValue : 0);
        handled = true;
      }

      if (!handled) return;
      event.preventDefault();
      event.stopPropagation();
      onResizeEnd?.();
    },
    [collapsible, direction, isCollapsed, max, min, onResize, onResizeEnd, onResizeTo],
  );

  let CollapsedIcon: typeof ChevronLeft | null = null;
  if (isCollapsed && direction === 'horizontal') {
    CollapsedIcon = collapsed === 'before' ? ChevronRight : ChevronLeft;
  } else if (isCollapsed) {
    CollapsedIcon = collapsed === 'before' ? ChevronDown : ChevronUp;
  }

  return (
    <div
      ref={handleRef}
      className={isCollapsed ? `${cls} resize-handle-collapsed` : cls}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishPointerDrag}
      onPointerCancel={finishPointerDrag}
      onKeyDown={handleKeyDown}
      role="separator"
      aria-label={ariaLabel}
      aria-orientation={separatorOrientation}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={ariaValueNow}
      aria-valuetext={ariaValueText}
      tabIndex={0}
      title={isCollapsed ? 'Drag, click, or press Enter to restore pane' : undefined}
      style={{ touchAction: 'none' }}
      data-value-unit={valueUnit}
      data-collapsed-side={collapsed}
    >
      <span className="resize-handle-grip" aria-hidden="true">
        <span className="resize-handle-grip-dot" />
        <span className="resize-handle-grip-dot" />
        <span className="resize-handle-grip-dot" />
      </span>
      {CollapsedIcon && <CollapsedIcon size={14} className="resize-handle-chevron" />}
    </div>
  );
}
