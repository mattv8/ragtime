import { forwardRef, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, MutableRefObject, ReactNode } from 'react';
import { createPortal } from 'react-dom';
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

export interface ContextMenuProps {
  items: ContextMenuItem[];
  x: number;
  y: number;
  autoFocus?: boolean;
  onClose?: () => void;
  ariaLabel?: string;
}

const MENU_PADDING = 8;
const ESTIMATED_MENU_WIDTH = 200;
const ESTIMATED_ITEM_HEIGHT = 52;

/** Anchor-relative position, flipped left and lifted up when the menu would leave the viewport. */
function getMenuPosition(x: number, y: number, menuHeight: number): { x: number; y: number } {
  let nextX = x + MENU_PADDING;
  let nextY = y + MENU_PADDING;
  if (typeof window !== 'undefined') {
    if (nextX + ESTIMATED_MENU_WIDTH > window.innerWidth - MENU_PADDING) {
      nextX = Math.max(MENU_PADDING, x - MENU_PADDING);
    }
    if (nextY + menuHeight > window.innerHeight - MENU_PADDING) {
      nextY = Math.max(MENU_PADDING, window.innerHeight - menuHeight - MENU_PADDING);
    }
  }
  return { x: nextX, y: nextY };
}

/** Items with a tab stop; disabled items use tabIndex -1 and are skipped. */
function getEnabledItems(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(container.children).filter(
    (child): child is HTMLElement => child instanceof HTMLElement && child.tabIndex >= 0,
  );
}

export const ContextMenu = forwardRef<HTMLDivElement, ContextMenuProps>(
  ({ items, x, y, autoFocus = false, onClose, ariaLabel }, forwardedRef) => {
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

    // First paint uses an estimate; the measured height then keeps the bottom edge on screen.
    const [position, setPosition] = useState(() =>
      getMenuPosition(x, y, items.length * ESTIMATED_ITEM_HEIGHT),
    );
    useLayoutEffect(() => {
      const menu = containerRef.current;
      if (!menu) return;
      setPosition(getMenuPosition(x, y, menu.getBoundingClientRect().height));
    }, [x, y, items.length]);

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
          top: position.y,
          left: position.x,
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
