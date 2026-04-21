import { useState, useRef, useEffect, useLayoutEffect, useCallback, useMemo, type CSSProperties, type FocusEvent, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

const POPOVER_GAP = 8; // px between trigger edge and popover
const HOVER_CLOSE_DELAY_MS = 80;
const VIEWPORT_PADDING = 12;
const ARROW_INSET = 14;

type PopoverPosition = NonNullable<PopoverProps['position']>;

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

interface ComputedPos {
  top: number;
  left: number;
  placement: PopoverPosition;
  visibility: 'hidden' | 'visible';
  arrowLeft?: number;
  arrowTop?: number;
}

type PopoverStyle = CSSProperties & {
  '--popover-arrow-left'?: string;
  '--popover-arrow-top'?: string;
};

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

function getOppositePosition(position: PopoverPosition): PopoverPosition {
  switch (position) {
    case 'top':
      return 'bottom';
    case 'bottom':
      return 'top';
    case 'left':
      return 'right';
    case 'right':
      return 'left';
  }
}

function computeFallbackPos(rect: DOMRect, position: PopoverPosition): ComputedPos {
  switch (position) {
    case 'bottom':
      return {
        top: rect.bottom + POPOVER_GAP,
        left: rect.left,
        placement: position,
        visibility: 'hidden',
      };
    case 'top':
      return {
        top: rect.top - POPOVER_GAP,
        left: rect.left,
        placement: position,
        visibility: 'hidden',
      };
    case 'right':
      return {
        top: rect.top,
        left: rect.right + POPOVER_GAP,
        placement: position,
        visibility: 'hidden',
      };
    case 'left':
      return {
        top: rect.top,
        left: rect.left - POPOVER_GAP,
        placement: position,
        visibility: 'hidden',
      };
  }
}

function computePopoverPos(
  rect: DOMRect,
  popoverRect: DOMRect | null,
  preferredPosition: PopoverPosition,
): ComputedPos {
  if (!popoverRect) {
    return computeFallbackPos(rect, preferredPosition);
  }

  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const spaces = {
    top: rect.top - VIEWPORT_PADDING,
    bottom: viewportHeight - rect.bottom - VIEWPORT_PADDING,
    left: rect.left - VIEWPORT_PADDING,
    right: viewportWidth - rect.right - VIEWPORT_PADDING,
  };

  let placement = preferredPosition;
  const opposite = getOppositePosition(preferredPosition);

  if (preferredPosition === 'top' || preferredPosition === 'bottom') {
    const needed = popoverRect.height + POPOVER_GAP;
    if (spaces[preferredPosition] < needed && spaces[opposite] > spaces[preferredPosition]) {
      placement = opposite;
    }
  } else {
    const needed = popoverRect.width + POPOVER_GAP;
    if (spaces[preferredPosition] < needed && spaces[opposite] > spaces[preferredPosition]) {
      placement = opposite;
    }
  }

  if (placement === 'top' || placement === 'bottom') {
    const unclampedLeft = rect.left + (rect.width / 2) - (popoverRect.width / 2);
    const left = clamp(
      unclampedLeft,
      VIEWPORT_PADDING,
      viewportWidth - VIEWPORT_PADDING - popoverRect.width,
    );
    const unclampedTop = placement === 'bottom'
      ? rect.bottom + POPOVER_GAP
      : rect.top - POPOVER_GAP - popoverRect.height;
    const top = clamp(
      unclampedTop,
      VIEWPORT_PADDING,
      viewportHeight - VIEWPORT_PADDING - popoverRect.height,
    );

    return {
      top,
      left,
      placement,
      visibility: 'visible',
      arrowLeft: clamp(rect.left + (rect.width / 2) - left, ARROW_INSET, popoverRect.width - ARROW_INSET),
    };
  }

  const unclampedLeft = placement === 'right'
    ? rect.right + POPOVER_GAP
    : rect.left - POPOVER_GAP - popoverRect.width;
  const left = clamp(
    unclampedLeft,
    VIEWPORT_PADDING,
    viewportWidth - VIEWPORT_PADDING - popoverRect.width,
  );
  const unclampedTop = rect.top + (rect.height / 2) - (popoverRect.height / 2);
  const top = clamp(
    unclampedTop,
    VIEWPORT_PADDING,
    viewportHeight - VIEWPORT_PADDING - popoverRect.height,
  );

  return {
    top,
    left,
    placement,
    visibility: 'visible',
    arrowTop: clamp(rect.top + (rect.height / 2) - top, ARROW_INSET, popoverRect.height - ARROW_INSET),
  };
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
  const [pos, setPos] = useState<ComputedPos | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const hoverCloseTimeoutRef = useRef<number | null>(null);
  const recomputeRafRef = useRef<number | null>(null);

  const isControlled = controlledShow !== undefined;
  const isVisible = isControlled ? controlledShow : internalShow;
  const shouldRender = isVisible && !disabled;

  const clearHoverCloseTimeout = useCallback(() => {
    if (hoverCloseTimeoutRef.current !== null) {
      window.clearTimeout(hoverCloseTimeoutRef.current);
      hoverCloseTimeoutRef.current = null;
    }
  }, []);

  const openHoverPopover = useCallback(() => {
    clearHoverCloseTimeout();
    if (!isControlled) {
      setInternalShow(true);
    }
  }, [clearHoverCloseTimeout, isControlled]);

  const closeHoverPopover = useCallback(() => {
    clearHoverCloseTimeout();
    if (!isControlled) {
      setInternalShow(false);
    }
  }, [clearHoverCloseTimeout, isControlled]);

  const scheduleHoverClose = useCallback(() => {
    if (trigger !== 'hover') {
      return;
    }
    clearHoverCloseTimeout();
    hoverCloseTimeoutRef.current = window.setTimeout(() => {
      hoverCloseTimeoutRef.current = null;
      if (!isControlled) {
        setInternalShow(false);
      }
    }, HOVER_CLOSE_DELAY_MS);
  }, [clearHoverCloseTimeout, isControlled, trigger]);

  const recomputePos = useCallback(() => {
    if (!triggerRef.current) return;
    setPos(
      computePopoverPos(
        triggerRef.current.getBoundingClientRect(),
        popoverRef.current ? popoverRef.current.getBoundingClientRect() : null,
        position,
      ),
    );
  }, [position]);

  // Throttle to once per animation frame (aligns with paint cycle)
  const scheduleRecomputePos = useCallback(() => {
    if (recomputeRafRef.current !== null) return;
    recomputeRafRef.current = window.requestAnimationFrame(() => {
      recomputeRafRef.current = null;
      recomputePos();
    });
  }, [recomputePos]);

  // Callback ref on the popover element: remeasures synchronously once the portal mounts,
  // transitioning from the hidden-fallback pass to the measured visible pass.
  const setPopoverRef = useCallback((node: HTMLDivElement | null) => {
    popoverRef.current = node;
    if (node && triggerRef.current) {
      recomputePos();
    }
  }, [recomputePos]);

  // Compute position synchronously when visibility/position changes
  useLayoutEffect(() => {
    if (shouldRender) {
      recomputePos();
    } else {
      setPos(null);
    }
  }, [shouldRender, recomputePos]);

  // Reposition on scroll / resize while visible (debounced to 16ms = 1 frame)
  useEffect(() => {
    if (!shouldRender) return;
    window.addEventListener('scroll', scheduleRecomputePos, true);
    window.addEventListener('resize', scheduleRecomputePos);
    return () => {
      window.removeEventListener('scroll', scheduleRecomputePos, true);
      window.removeEventListener('resize', scheduleRecomputePos);
    };
  }, [shouldRender, scheduleRecomputePos]);

  useEffect(() => {
    return () => {
      clearHoverCloseTimeout();
      if (recomputeRafRef.current !== null) {
        window.cancelAnimationFrame(recomputeRafRef.current);
        recomputeRafRef.current = null;
      }
    };
  }, [clearHoverCloseTimeout]);

  // Close on outside click for click trigger (but not if clicked inside popover)
  useEffect(() => {
    if (trigger !== 'click' || !isVisible) return;
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      const isClickOnTrigger = triggerRef.current && triggerRef.current.contains(target);
      const isClickOnPopover = popoverRef.current && popoverRef.current.contains(target);
      if (!isClickOnTrigger && !isClickOnPopover) {
        setInternalShow(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [trigger, isVisible]);

  const handleMouseEnter = () => {
    if (trigger === 'hover' && !disabled) {
      openHoverPopover();
    }
  };
  const handleMouseLeave = () => {
    scheduleHoverClose();
  };
  const handleClick = () => {
    if (trigger === 'click' && !disabled) setInternalShow((v) => !v);
  };
  const handleFocus = () => {
    if (trigger === 'hover' && !disabled) {
      openHoverPopover();
    }
  };
  const handleBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (trigger !== 'hover') {
      return;
    }
    const nextTarget = event.relatedTarget;
    if (
      nextTarget instanceof Node
      && (
        triggerRef.current?.contains(nextTarget)
        || popoverRef.current?.contains(nextTarget)
      )
    ) {
      return;
    }
    closeHoverPopover();
  };

  const popoverStyle = useMemo<PopoverStyle | undefined>(() => {
    if (!pos) return undefined;
    return {
      top: pos.top,
      left: pos.left,
      visibility: pos.visibility,
      '--popover-arrow-left': pos.arrowLeft != null ? `${pos.arrowLeft}px` : undefined,
      '--popover-arrow-top': pos.arrowTop != null ? `${pos.arrowTop}px` : undefined,
    };
  }, [pos]);

  const popoverPortal =
    shouldRender && pos
      ? createPortal(
          <div
            ref={setPopoverRef}
            className={`popover popover-${pos.placement}`}
            role="tooltip"
            style={popoverStyle}
            onMouseEnter={trigger === 'hover' ? clearHoverCloseTimeout : undefined}
            onMouseLeave={trigger === 'hover' ? scheduleHoverClose : undefined}
          >
            <div className="popover-arrow" />
            <div className="popover-content">{content}</div>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <div
        ref={triggerRef}
        className={`popover-container ${className}`.trim()}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onFocus={handleFocus}
        onBlur={handleBlur}
      >
        {children}
      </div>
      {popoverPortal}
    </>
  );
}

/**
 * A simpler wrapper that shows a popover only when disabled.
 * Useful for explaining why a button is disabled.
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
    <Popover content={message} position={position} trigger="hover">
      <span style={{ display: 'inline-block' }}>{children}</span>
    </Popover>
  );
}
