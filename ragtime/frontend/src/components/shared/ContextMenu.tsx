import { forwardRef, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, MutableRefObject, ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { computeAnchoredMenuPosition } from '@/utils/anchoredMenuPosition';
import { Icon } from '../Icon';
import type { IconType } from '../Icon';

export interface ContextMenuItem {
  label: string;
  type?: 'default' | 'checkbox' | 'toggle';
  checked?: boolean;
  disabled?: boolean;
  description?: ReactNode;
  icon?: IconType;
  onSelect: () => void;
}

/** The element a menu belongs to, in viewport coordinates. */
export interface ContextMenuAnchorRect {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

/** Where the menu opens: a pointer position, or the element it belongs to. */
export type ContextMenuPlacement =
  | { x: number; y: number; anchorRect?: never }
  | { anchorRect: ContextMenuAnchorRect; x?: never; y?: never };

interface ContextMenuCommonProps {
  items: ContextMenuItem[];
  autoFocus?: boolean;
  onClose?: () => void;
  ariaLabel?: string;
}

export type ContextMenuProps = ContextMenuCommonProps & ContextMenuPlacement;

const ESTIMATED_MENU_WIDTH = 200;
const ESTIMATED_ITEM_HEIGHT = 52;
const POINTER_GAP = 8;

/**
 * A pointer becomes a box with inverted edges: a bottom-start menu then opens 8px right of and
 * below the pointer, and when flipped it ends 8px left of and above it.
 */
function getPointerAnchorRect(x: number, y: number): ContextMenuAnchorRect {
  return {
    left: x + POINTER_GAP,
    right: x - POINTER_GAP,
    top: y - POINTER_GAP,
    bottom: y + POINTER_GAP,
  };
}

/** Items with a tab stop; disabled items use tabIndex -1 and are skipped. */
function getEnabledItems(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(container.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement && child.tabIndex >= 0,
  );
}

export const ContextMenu = forwardRef<HTMLDivElement, ContextMenuProps>(
  ({ items, x, y, anchorRect, autoFocus = false, onClose, ariaLabel }, forwardedRef) => {
    const anchor = anchorRect ?? getPointerAnchorRect(x ?? 0, y ?? 0);
    const anchorLeft = anchor.left;
    const anchorRight = anchor.right;
    const anchorTop = anchor.top;
    const anchorBottom = anchor.bottom;

    const containerRef = useRef<HTMLDivElement | null>(null);

    const setRefs = useCallback(
      (node: HTMLDivElement | null) => {
        containerRef.current = node;
        if (typeof forwardedRef === 'function') {
          forwardedRef(node);
        } else if (forwardedRef) {
          (forwardedRef as MutableRefObject<HTMLDivElement | null>).current = node;
        }
      },
      [forwardedRef],
    );

    useEffect(() => {
      if (!autoFocus) return;
      getEnabledItems(containerRef.current)[0]?.focus();
    }, [autoFocus]);

    const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
      const enabled = getEnabledItems(containerRef.current);
      if (enabled.length === 0) return;
      const currentIndex = enabled.findIndex((element) => element.contains(event.target as Node));
      const lastIndex = enabled.length - 1;

      const focusAt = (index: number) => {
        event.preventDefault();
        enabled[index]?.focus();
      };

      switch (event.key) {
        case 'ArrowDown':
          focusAt(currentIndex < 0 || currentIndex === lastIndex ? 0 : currentIndex + 1);
          break;
        case 'ArrowUp':
          focusAt(currentIndex <= 0 ? lastIndex : currentIndex - 1);
          break;
        case 'Home':
          focusAt(0);
          break;
        case 'End':
          focusAt(lastIndex);
          break;
        case 'Escape':
          onClose?.();
          break;
        case 'Tab':
          if (event.shiftKey ? currentIndex === 0 : currentIndex === lastIndex) {
            onClose?.();
          }
          break;
      }
    };

    // First paint uses an estimate so the menu is never hidden; the measured size then places it.
    const [placement, setPlacement] = useState(() =>
      computeAnchoredMenuPosition(
        { left: anchorLeft, right: anchorRight, top: anchorTop, bottom: anchorBottom },
        { width: ESTIMATED_MENU_WIDTH, height: items.length * ESTIMATED_ITEM_HEIGHT },
        'bottom-start',
      ),
    );
    useLayoutEffect(() => {
      const menu = containerRef.current;
      if (!menu) return;
      // Measure at the viewport origin without the previous height cap, so neither shrink-to-fit
      // near the right edge nor an earlier placement can understate the menu's size.
      const { left, top, maxHeight } = menu.style;
      Object.assign(menu.style, { left: '0px', top: '0px', maxHeight: '' });
      const { width, height } = menu.getBoundingClientRect();
      Object.assign(menu.style, { left, top, maxHeight });
      setPlacement(
        computeAnchoredMenuPosition(
          { left: anchorLeft, right: anchorRight, top: anchorTop, bottom: anchorBottom },
          { width, height },
          'bottom-start',
        ),
      );
    }, [anchorLeft, anchorRight, anchorTop, anchorBottom, items.length]);

    if (items.length === 0) {
      return null;
    }

    return createPortal(
      <div
        ref={setRefs}
        className="userspace-tool-context-menu"
        role={ariaLabel ? 'group' : undefined}
        aria-label={ariaLabel}
        onKeyDown={handleKeyDown}
        style={{
          position: 'fixed',
          top: placement.top,
          left: placement.left,
          maxHeight: placement.maxHeight,
          zIndex: 9001,
        }}
      >
        {items.map((item, index) => {
          const isCheckbox =
            item.type === 'checkbox' || (item.type !== 'toggle' && item.checked !== undefined);
          const isToggle = item.type === 'toggle';
          return (
            <div
              key={index}
              className={`userspace-tool-context-menu-item ${item.disabled ? 'disabled' : ''} ${isCheckbox || isToggle ? '' : 'userspace-tool-context-menu-item-action'}`}
              onMouseDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                if (!item.disabled) {
                  item.onSelect();
                }
              }}
              onKeyDown={(event) => {
                if ((event.key === 'Enter' || event.key === ' ') && !item.disabled) {
                  event.preventDefault();
                  item.onSelect();
                }
              }}
              role="button"
              tabIndex={item.disabled ? -1 : 0}
              aria-checked={isCheckbox || isToggle ? item.checked : undefined}
            >
              {isToggle ? (
                <label
                  className="toggle-switch"
                  style={{ pointerEvents: 'none', margin: '2px 0 0 0', flexShrink: 0 }}
                >
                  <input
                    type="checkbox"
                    tabIndex={-1}
                    checked={item.checked}
                    disabled={item.disabled}
                    readOnly
                  />
                  <span className="toggle-slider"></span>
                </label>
              ) : isCheckbox ? (
                <input
                  type="checkbox"
                  tabIndex={-1}
                  checked={item.checked}
                  disabled={item.disabled}
                  readOnly
                />
              ) : null}
              {item.icon && <Icon name={item.icon} size={16} />}
              <div className="userspace-tool-context-menu-text">
                <span className="userspace-tool-context-menu-label">{item.label}</span>
                {item.description && (
                  <span className="userspace-tool-context-menu-description">
                    {item.description}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>,
      document.body,
    );
  },
);

ContextMenu.displayName = 'ContextMenu';
