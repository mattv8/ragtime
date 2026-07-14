import { forwardRef, useCallback, useMemo } from 'react';
import type { MutableRefObject, ReactNode } from 'react';
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
}

export const ContextMenu = forwardRef<HTMLDivElement, ContextMenuProps>(
  ({ items, x, y }, forwardedRef) => {
    const setRefs = useCallback(
      (node: HTMLDivElement | null) => {
        if (typeof forwardedRef === 'function') {
          forwardedRef(node);
        } else if (forwardedRef) {
          (forwardedRef as MutableRefObject<HTMLDivElement | null>).current = node;
        }
      },
      [forwardedRef],
    );

    const position = useMemo(() => {
      const PADDING = 8;
      let nextX = x + PADDING;
      let nextY = y + PADDING;
      if (typeof window !== 'undefined') {
        if (nextX + 200 > window.innerWidth - PADDING) {
          nextX = Math.max(PADDING, x - PADDING);
        }
        if (nextY + items.length * 52 > window.innerHeight - PADDING) {
          nextY = Math.max(PADDING, window.innerHeight - items.length * 52 - PADDING);
        }
      }
      return { x: nextX, y: nextY };
    }, [items.length, x, y]);

    if (items.length === 0) {
      return null;
    }

    return createPortal(
      <div
        ref={setRefs}
        className="userspace-tool-context-menu"
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
                  <input type="checkbox" checked={item.checked} disabled={item.disabled} readOnly />
                  <span className="toggle-slider"></span>
                </label>
              ) : isCheckbox ? (
                <input
                  type="checkbox"
                  checked={item.checked}
                  disabled={item.disabled}
                  onChange={(event) => {
                    event.stopPropagation();
                    if (!item.disabled) {
                      item.onSelect();
                    }
                  }}
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
