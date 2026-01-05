import { useState, useRef, useEffect, ReactNode } from 'react';

interface PopoverProps {
  /** The trigger element that shows/hides the popover */
  children: ReactNode;
  /** Content to display in the popover */
  content: ReactNode;
  /** Position relative to the trigger element */
  position?: 'top' | 'bottom' | 'left' | 'right';
  /** Whether to show the popover (controlled mode) */
  show?: boolean;
  /** Trigger mode: hover or click */
  trigger?: 'hover' | 'click';
  /** Additional class name for the popover container */
  className?: string;
  /** Whether the popover is disabled (won't show) */
  disabled?: boolean;
}

export function Popover({
  children,
  content,
  position = 'top',
  show: controlledShow,
  trigger = 'hover',
  className = '',
  disabled = false,
}: PopoverProps) {
  const [internalShow, setInternalShow] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);

  const isControlled = controlledShow !== undefined;
  const isVisible = isControlled ? controlledShow : internalShow;

  // Close on click outside for click trigger
  useEffect(() => {
    if (trigger !== 'click' || !isVisible) return;

    const handleClickOutside = (e: MouseEvent) => {
      if (
        triggerRef.current &&
        !triggerRef.current.contains(e.target as Node) &&
        popoverRef.current &&
        !popoverRef.current.contains(e.target as Node)
      ) {
        setInternalShow(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [trigger, isVisible]);

  const handleMouseEnter = () => {
    if (trigger === 'hover' && !disabled) {
      setInternalShow(true);
    }
  };

  const handleMouseLeave = () => {
    if (trigger === 'hover') {
      setInternalShow(false);
    }
  };

  const handleClick = () => {
    if (trigger === 'click' && !disabled) {
      setInternalShow(!internalShow);
    }
  };

  return (
    <div
      className={`popover-container ${className}`}
      ref={triggerRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    >
      {children}
      {isVisible && !disabled && (
        <div
          ref={popoverRef}
          className={`popover popover-${position}`}
          role="tooltip"
        >
          <div className="popover-content">
            {content}
          </div>
          <div className="popover-arrow" />
        </div>
      )}
    </div>
  );
}

/**
 * A simpler wrapper that shows a popover only when disabled
 * Useful for explaining why a button is disabled
 */
interface DisabledPopoverProps {
  /** The element to wrap */
  children: ReactNode;
  /** Message to show when disabled */
  message: string;
  /** Whether the wrapped element is disabled */
  disabled: boolean;
  /** Position of the popover */
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export function DisabledPopover({
  children,
  message,
  disabled,
  position = 'right',
}: DisabledPopoverProps) {
  if (!disabled) {
    return <>{children}</>;
  }

  return (
    <Popover
      content={message}
      position={position}
      trigger="hover"
    >
      <span style={{ display: 'inline-block' }}>
        {children}
      </span>
    </Popover>
  );
}
