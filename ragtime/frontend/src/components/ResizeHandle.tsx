import { useCallback, useEffect, useRef } from 'react';

interface ResizeHandleProps {
  /** 'horizontal' = dragging left/right, 'vertical' = dragging up/down */
  direction: 'horizontal' | 'vertical';
  /** Called continuously during drag with the delta in px from drag start */
  onResize: (delta: number) => void;
  /** Optional className override */
  className?: string;
}

export function ResizeHandle({ direction, onResize, className }: ResizeHandleProps) {
  const startPos = useRef(0);
  const dragging = useRef(false);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragging.current) return;
      e.preventDefault();
      const pos = direction === 'horizontal' ? e.clientX : e.clientY;
      const delta = pos - startPos.current;
      startPos.current = pos;
      onResize(delta);
    },
    [direction, onResize],
  );

  const handleMouseUp = useCallback(() => {
    dragging.current = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  }, []);

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      dragging.current = true;
      startPos.current = direction === 'horizontal' ? e.clientX : e.clientY;
      document.body.style.cursor = direction === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    },
    [direction],
  );

  const cls = className ?? `resize-handle resize-handle-${direction}`;

  return <div className={cls} onMouseDown={handleMouseDown} />;
}
