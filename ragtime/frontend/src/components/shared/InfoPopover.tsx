import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Info } from 'lucide-react';
import { Popover } from '../Popover';

export interface InfoPopoverProps {
  /** Id of the trigger button; the popover content gets `${id}-content` */
  id: string;
  /** Accessible name of the trigger button */
  label: string;
  /** Popover content */
  children: ReactNode;
  /** Side of the trigger the popover opens on */
  position?: 'top' | 'bottom' | 'left' | 'right';
  /** Trigger size: 'sm' (18px) or 'md' (28px) */
  size?: 'sm' | 'md';
}

const ICON_SIZE = { sm: 12, md: 14 } as const;

export function InfoPopover({
  id,
  label,
  children,
  position = 'bottom',
  size = 'sm',
}: InfoPopoverProps) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const contentId = `${id}-content`;

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      triggerRef.current?.focus();
    };

    const handleMouseDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target)) return;
      if (document.getElementById(contentId)?.contains(target)) return;
      setOpen(false);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, [open, contentId]);

  return (
    <Popover
      trigger="click"
      show={open}
      position={position}
      content={
        <div id={contentId} className="info-popover-content" data-info-popover={id}>
          {children}
        </div>
      }
    >
      <button
        ref={triggerRef}
        type="button"
        id={id}
        className="info-popover-trigger"
        data-size={size}
        aria-label={label}
        aria-expanded={open}
        aria-controls={open ? contentId : undefined}
        aria-describedby={open ? contentId : undefined}
        onClick={() => setOpen((value) => !value)}
      >
        <Info size={ICON_SIZE[size]} aria-hidden="true" />
      </button>
    </Popover>
  );
}
