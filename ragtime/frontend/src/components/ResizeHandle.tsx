import { useCallback, useRef, useEffect } from 'react';
import { ChevronLeft, ChevronRight, ChevronUp, ChevronDown } from 'lucide-react';

interface ResizeHandleProps {
  /** 'horizontal' = dragging left/right, 'vertical' = dragging up/down */
  direction: 'horizontal' | 'vertical';
  /** Called continuously during drag with the delta in px from drag start */
  onResize: (delta: number) => void;
  /** Optional className override */
  className?: string;
  /**
   * Which side adjacent to this handle is currently collapsed.
   * 'before' = the pane before (left/top), 'after' = the pane after (right/bottom), undefined = nothing collapsed.
   */
  collapsed?: 'before' | 'after';
  /** Called when the user activates the collapsed handle to restore a pane */
  onExpand?: () => void;
  /** Called when a drag gesture ends or collapsed handle is activated */
  onResizeEnd?: () => void;
}

export function ResizeHandle({
  direction,
  onResize,
  className,
  collapsed,
  onExpand,
  onResizeEnd,
}: ResizeHandleProps) {
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
      e.currentTarget.setPointerCapture(e.pointerId);
      startPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
      isDragging.current = true;
      if (collapsed) {
        onExpand?.();
      }
    },
    [collapsed, direction, onExpand],
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
  let CollapsedIcon: typeof ChevronLeft | null = null;
  if (collapsed && direction === 'horizontal') {
    CollapsedIcon = collapsed === 'before' ? ChevronRight : ChevronLeft;
  } else if (collapsed) {
    CollapsedIcon = collapsed === 'before' ? ChevronDown : ChevronUp;
  }

  return (
    <div
      className={collapsed ? `${cls} resize-handle-collapsed` : cls}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishPointerDrag}
      onPointerCancel={finishPointerDrag}
      title={collapsed ? 'Drag or click to expand pane' : undefined}
      style={{ touchAction: 'none' }}
    >
      {CollapsedIcon && <CollapsedIcon size={14} className="resize-handle-chevron" />}
    </div>
  );
}
