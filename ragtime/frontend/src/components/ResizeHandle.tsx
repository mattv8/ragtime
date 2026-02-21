import { useCallback, useRef } from 'react';
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
}

export function ResizeHandle({ direction, onResize, className, collapsed, onExpand }: ResizeHandleProps) {
  const startPos = useRef(0);
  const onResizeRef = useRef(onResize);
  onResizeRef.current = onResize;

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (collapsed) return; // don't allow dragging when collapsed
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      startPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    },
    [direction, collapsed],
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
      e.preventDefault();
      const pos = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = pos - startPos.current;
      startPos.current = pos;
      onResizeRef.current(delta);
    },
    [direction],
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
      e.currentTarget.releasePointerCapture(e.pointerId);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    },
    [],
  );

  const cls = className ?? `resize-handle resize-handle-${direction}`;

  if (collapsed) {
    // Determine chevron icon: points toward the collapsed pane (click to expand it)
    let Icon: typeof ChevronLeft;
    if (direction === 'horizontal') {
      Icon = collapsed === 'before' ? ChevronRight : ChevronLeft;
    } else {
      Icon = collapsed === 'before' ? ChevronDown : ChevronUp;
    }

    return (
      <div
        className={`${cls} resize-handle-collapsed`}
        onPointerDown={(e) => {
          e.preventDefault();
          onExpand?.();
        }}
        title="Expand pane"
      >
        <Icon size={14} className="resize-handle-chevron" />
      </div>
    );
  }

  return (
    <div
      className={cls}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      style={{ touchAction: 'none' }}
    />
  );
}
