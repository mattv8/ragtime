import {
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  useCallback,
  useMemo,
  type CSSProperties,
  type FocusEvent,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

const POPOVER_GAP = 8; // px between trigger edge and popover
const HOVER_CLOSE_DELAY_MS = 80;
const VIEWPORT_PADDING = 12;
const ARROW_INSET = 14;
const FOLLOW_CURSOR_TOP_OFFSET = -8;

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
  /** Additional inline styles for the popover container */
  style?: CSSProperties;
  /** Whether the popover is disabled (won't show) */
  disabled?: boolean;
  /** Delay before showing on hover (ms) */
  openDelayMs?: number;
  /** Whether the popover should appear at the cursor coordinates instead of anchoring to the trigger bounds */
  followCursor?: boolean;
  /** Require the pointer to stay still for this many ms before opening on hover */
  requireHoverIdleMs?: number;
  /** CSS selector for elements that should not trigger the popover */
  ignoreSelector?: string;
  /** Whether focus should open hover-triggered popovers */
  focusTrigger?: boolean;
  /** Whether the popover should render one z-index above its trigger */
  zIndexAboveTrigger?: boolean;
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

function getZIndexAboveElement(element: HTMLElement | null): number | undefined {
  if (!element) return undefined;
  const zIndex = Number.parseInt(window.getComputedStyle(element).zIndex, 10);
  return Number.isFinite(zIndex) ? zIndex + 1 : undefined;
}

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
    const unclampedLeft = rect.left + rect.width / 2 - popoverRect.width / 2;
    const left = clamp(
      unclampedLeft,
      VIEWPORT_PADDING,
      viewportWidth - VIEWPORT_PADDING - popoverRect.width,
    );
    const unclampedTop =
      placement === 'bottom'
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
      arrowLeft: clamp(
        rect.left + rect.width / 2 - left,
        ARROW_INSET,
        popoverRect.width - ARROW_INSET,
      ),
    };
  }

  const unclampedLeft =
    placement === 'right' ? rect.right + POPOVER_GAP : rect.left - POPOVER_GAP - popoverRect.width;
  const left = clamp(
    unclampedLeft,
    VIEWPORT_PADDING,
    viewportWidth - VIEWPORT_PADDING - popoverRect.width,
  );
  const unclampedTop = rect.top + rect.height / 2 - popoverRect.height / 2;
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
    arrowTop: clamp(
      rect.top + rect.height / 2 - top,
      ARROW_INSET,
      popoverRect.height - ARROW_INSET,
    ),
  };
}

export function Popover({
  children,
  content,
  position = 'top',
  show: controlledShow,
  trigger = 'hover',
  className = '',
  style,
  disabled = false,
  openDelayMs = 0,
  followCursor = false,
  requireHoverIdleMs = 0,
  ignoreSelector,
  focusTrigger = true,
  zIndexAboveTrigger = false,
  ...rest
}: PopoverProps & Omit<React.HTMLAttributes<HTMLDivElement>, 'content'>) {
  const [internalShow, setInternalShow] = useState(false);
  const [pos, setPos] = useState<ComputedPos | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const hoverCloseTimeoutRef = useRef<number | null>(null);
  const hoverOpenTimeoutRef = useRef<number | null>(null);
  const recomputeRafRef = useRef<number | null>(null);
  const cursorPosRef = useRef<{ x: number; y: number } | null>(null);

  const isControlled = controlledShow !== undefined;
  const isVisible = isControlled ? controlledShow : internalShow;
  const shouldRender = isVisible && !disabled;

  const isIgnored = useCallback(
    (target: EventTarget | null) => {
      if (!ignoreSelector || !(target instanceof Element)) return false;
      return !!target.closest(ignoreSelector);
    },
    [ignoreSelector],
  );

  const clearHoverCloseTimeout = useCallback(() => {
    if (hoverCloseTimeoutRef.current !== null) {
      window.clearTimeout(hoverCloseTimeoutRef.current);
      hoverCloseTimeoutRef.current = null;
    }
  }, []);

  const clearHoverOpenTimeout = useCallback(() => {
    if (hoverOpenTimeoutRef.current !== null) {
      window.clearTimeout(hoverOpenTimeoutRef.current);
      hoverOpenTimeoutRef.current = null;
    }
  }, []);

  const openHoverPopover = useCallback(() => {
    clearHoverCloseTimeout();
    if (!isControlled) {
      setInternalShow(true);
    }
  }, [clearHoverCloseTimeout, isControlled]);

  const scheduleHoverOpen = useCallback(
    (delayMs: number) => {
      clearHoverOpenTimeout();
      if (delayMs > 0) {
        hoverOpenTimeoutRef.current = window.setTimeout(() => {
          hoverOpenTimeoutRef.current = null;
          openHoverPopover();
        }, delayMs);
      } else {
        openHoverPopover();
      }
    },
    [clearHoverOpenTimeout, openHoverPopover],
  );

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

    let targetRect = triggerRef.current.getBoundingClientRect();
    if (followCursor && cursorPosRef.current) {
      const y = cursorPosRef.current.y + FOLLOW_CURSOR_TOP_OFFSET;
      targetRect = {
        top: y,
        bottom: y,
        left: cursorPosRef.current.x,
        right: cursorPosRef.current.x,
        width: 0,
        height: 0,
        x: cursorPosRef.current.x,
        y,
        toJSON: () => {},
      } as DOMRect;
    }

    setPos(
      computePopoverPos(
        targetRect,
        popoverRef.current ? popoverRef.current.getBoundingClientRect() : null,
        position,
      ),
    );
  }, [position, followCursor]);

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
  const setPopoverRef = useCallback(
    (node: HTMLDivElement | null) => {
      popoverRef.current = node;
      if (node && triggerRef.current) {
        recomputePos();
      }
    },
    [recomputePos],
  );

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
      clearHoverOpenTimeout();
      if (recomputeRafRef.current !== null) {
        window.cancelAnimationFrame(recomputeRafRef.current);
        recomputeRafRef.current = null;
      }
    };
  }, [clearHoverCloseTimeout, clearHoverOpenTimeout]);

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

  const handleMouseEnter = (e: React.MouseEvent) => {
    if (trigger === 'hover' && !disabled) {
      if (isIgnored(e.target)) return;
      clearHoverCloseTimeout();
      if (followCursor) {
        cursorPosRef.current = { x: e.clientX, y: e.clientY };
      }
      if (requireHoverIdleMs > 0) {
        if (!isControlled && internalShow) {
          setInternalShow(false);
        }
        scheduleHoverOpen(requireHoverIdleMs);
      } else {
        scheduleHoverOpen(openDelayMs);
      }
    }
  };
  const handleMouseLeave = () => {
    clearHoverOpenTimeout();
    cursorPosRef.current = null;
    scheduleHoverClose();
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isIgnored(e.target)) {
      clearHoverOpenTimeout();
      cursorPosRef.current = null;
      scheduleHoverClose();
      return;
    }

    if (followCursor) {
      cursorPosRef.current = { x: e.clientX, y: e.clientY };
    }

    if (trigger === 'hover' && !disabled && requireHoverIdleMs > 0) {
      clearHoverCloseTimeout();
      if (!isControlled && internalShow) {
        setInternalShow(false);
      }
      scheduleHoverOpen(requireHoverIdleMs);
      return;
    }

    if (trigger === 'hover' && !disabled) {
      clearHoverCloseTimeout();
      if (!isControlled && !internalShow && hoverOpenTimeoutRef.current === null) {
        scheduleHoverOpen(openDelayMs);
      }
    }

    if (followCursor && isVisible) {
      scheduleRecomputePos();
    }
  };
  const handleClick = () => {
    if (trigger === 'click' && !disabled) setInternalShow((v) => !v);
  };
  const handleFocus = () => {
    if (focusTrigger && trigger === 'hover' && !disabled) {
      openHoverPopover();
    }
  };
  const handleBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (trigger !== 'hover') {
      return;
    }
    const nextTarget = event.relatedTarget;
    if (
      nextTarget instanceof Node &&
      (triggerRef.current?.contains(nextTarget) || popoverRef.current?.contains(nextTarget))
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
      zIndex: zIndexAboveTrigger ? getZIndexAboveElement(triggerRef.current) : undefined,
      '--popover-arrow-left': pos.arrowLeft != null ? `${pos.arrowLeft}px` : undefined,
      '--popover-arrow-top': pos.arrowTop != null ? `${pos.arrowTop}px` : undefined,
    };
  }, [pos, zIndexAboveTrigger]);

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
        style={style}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onMouseMove={handleMouseMove}
        onClick={handleClick}
        onFocus={handleFocus}
        onBlur={handleBlur}
        {...rest}
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
