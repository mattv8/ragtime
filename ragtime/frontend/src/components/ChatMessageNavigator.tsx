import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react';

const POINTER_LEAVE_CLOSE_DELAY_MS = 150;

export interface ChatMessageNavigationEntry {
  key: string;
  messageIndex: number;
  preview: string;
}

export interface ChatMessageNavigatorProps {
  entries: ChatMessageNavigationEntry[];
  activeKey: string | null;
  onNavigate: (entry: ChatMessageNavigationEntry) => void;
}

function getMaxScroll(element: HTMLElement): number {
  return Math.max(0, element.scrollHeight - element.clientHeight);
}

function clampScroll(value: number, max: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > max) {
    return max;
  }
  return value;
}

function normalizeWheelDelta(deltaY: number, deltaMode: number, pageHeight: number): number {
  if (deltaMode === WheelEvent.DOM_DELTA_LINE) {
    return deltaY * 16;
  }
  if (deltaMode === WheelEvent.DOM_DELTA_PAGE) {
    return deltaY * pageHeight;
  }
  return deltaY;
}

export function ChatMessageNavigator({
  entries,
  activeKey,
  onNavigate,
}: ChatMessageNavigatorProps): ReactElement | null {
  const hasNavigatorEntries = entries.length >= 2;
  const [isOpen, setIsOpen] = useState(false);
  const [previewedEntryKey, setPreviewedEntryKey] = useState<string | null>(null);
  const navigatorRef = useRef<HTMLElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const ticksRef = useRef<HTMLDivElement | null>(null);
  const itemElementsRef = useRef<Map<string, HTMLButtonElement>>(new Map());
  const tickElementsRef = useRef<Map<string, HTMLButtonElement>>(new Map());
  const skipNextFocusOpenRef = useRef(false);
  const closeTimeoutRef = useRef<number | null>(null);

  const activeEntry = useMemo(
    () => entries.find((entry) => entry.key === activeKey) ?? null,
    [activeKey, entries],
  );

  const syncTicksToList = useCallback(() => {
    const list = listRef.current;
    const ticks = ticksRef.current;
    if (!list || !ticks) {
      return;
    }

    const listMax = getMaxScroll(list);
    const ticksMax = getMaxScroll(ticks);
    const progress = listMax > 0 ? list.scrollTop / listMax : 0;
    ticks.scrollTop = ticksMax * progress;
  }, []);

  const applyWheelDelta = useCallback(
    (deltaY: number, deltaMode: number) => {
      const list = listRef.current;
      if (!list) {
        return;
      }

      const nextScrollTop = clampScroll(
        list.scrollTop + normalizeWheelDelta(deltaY, deltaMode, list.clientHeight),
        getMaxScroll(list),
      );
      list.scrollTop = nextScrollTop;
      syncTicksToList();
    },
    [syncTicksToList],
  );

  const openNavigator = useCallback(() => {
    if (closeTimeoutRef.current !== null) {
      window.clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setIsOpen(true);
  }, []);

  const closeNavigator = useCallback(() => {
    if (closeTimeoutRef.current !== null) {
      window.clearTimeout(closeTimeoutRef.current);
      closeTimeoutRef.current = null;
    }
    setPreviewedEntryKey(null);
    setIsOpen(false);
  }, []);

  const focusTickForEntry = useCallback(
    (entryKey: string | null) => {
      if (entryKey) {
        const tick = tickElementsRef.current.get(entryKey);
        if (tick) {
          tick.focus();
          return;
        }
      }

      const firstEntry = entries[0];
      if (!firstEntry) {
        return;
      }

      tickElementsRef.current.get(firstEntry.key)?.focus();
    },
    [entries],
  );

  useEffect(() => {
    if (!hasNavigatorEntries) {
      if (closeTimeoutRef.current !== null) {
        window.clearTimeout(closeTimeoutRef.current);
        closeTimeoutRef.current = null;
      }
      setIsOpen(false);
      skipNextFocusOpenRef.current = false;
    }
  }, [hasNavigatorEntries]);

  useEffect(() => {
    return () => {
      if (closeTimeoutRef.current !== null) {
        window.clearTimeout(closeTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!isOpen || !activeEntry) {
      return;
    }

    const activeItem = itemElementsRef.current.get(activeEntry.key);
    const activeTick = tickElementsRef.current.get(activeEntry.key);

    activeItem?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
    activeTick?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
  }, [activeEntry, isOpen]);

  useEffect(() => {
    if (!hasNavigatorEntries) {
      return;
    }

    const navigator = navigatorRef.current;
    if (!navigator) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      applyWheelDelta(event.deltaY, event.deltaMode);
    };

    navigator.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      navigator.removeEventListener('wheel', handleWheel);
    };
  }, [applyWheelDelta, hasNavigatorEntries]);

  if (!hasNavigatorEntries) {
    return null;
  }

  return (
    <nav
      ref={navigatorRef}
      aria-label="User message navigation"
      className={`chat-message-navigator${isOpen ? ' is-open' : ''}`}
      onBlurCapture={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
          return;
        }
        skipNextFocusOpenRef.current = false;
        closeNavigator();
      }}
      onFocusCapture={() => {
        if (skipNextFocusOpenRef.current) {
          skipNextFocusOpenRef.current = false;
          return;
        }
        openNavigator();
      }}
      onKeyDownCapture={(event) => {
        if (event.key !== 'Escape') {
          return;
        }
        event.preventDefault();
        const focusedEntryKey =
          (document.activeElement as HTMLElement | null)?.dataset.entryKey ?? null;
        const focusRestoreKey = activeKey ?? focusedEntryKey;
        skipNextFocusOpenRef.current = true;
        closeNavigator();
        focusTickForEntry(focusRestoreKey);
      }}
      onMouseEnter={() => {
        openNavigator();
      }}
      onMouseLeave={() => {
        if (!navigatorRef.current?.contains(document.activeElement)) {
          closeTimeoutRef.current = window.setTimeout(() => {
            closeTimeoutRef.current = null;
            closeNavigator();
          }, POINTER_LEAVE_CLOSE_DELAY_MS);
        }
      }}
    >
      <div className="chat-message-navigator-trigger">
        <span className="chat-message-navigator-rail">
          <span ref={ticksRef} className="chat-message-navigator-ticks">
            {entries.map((entry, index) => {
              const isActive = entry.key === activeKey;
              return (
                <button
                  key={entry.key}
                  aria-current={isActive ? 'location' : undefined}
                  aria-label={`Jump to user message ${index + 1}: ${entry.preview}`}
                  className={`chat-message-navigator-tick${isActive ? ' is-active' : ''}`}
                  data-entry-key={entry.key}
                  ref={(element) => {
                    if (element) {
                      tickElementsRef.current.set(entry.key, element);
                      return;
                    }
                    tickElementsRef.current.delete(entry.key);
                  }}
                  type="button"
                  onBlur={() => {
                    setPreviewedEntryKey(null);
                  }}
                  onClick={() => {
                    onNavigate(entry);
                  }}
                  onFocus={() => {
                    setPreviewedEntryKey(entry.key);
                  }}
                  onMouseEnter={() => {
                    setPreviewedEntryKey(entry.key);
                  }}
                  onMouseLeave={() => {
                    setPreviewedEntryKey(null);
                  }}
                />
              );
            })}
          </span>
        </span>
      </div>
      <div aria-hidden={!isOpen} className="chat-message-navigator-popover">
        <div ref={listRef} className="chat-message-navigator-list" onScroll={syncTicksToList}>
          {entries.map((entry, index) => {
            const isActive = entry.key === activeKey;
            const isPreviewed = entry.key === previewedEntryKey;
            return (
              <button
                key={entry.key}
                aria-current={isActive ? 'location' : undefined}
                aria-label={`Jump to user message ${index + 1}: ${entry.preview}`}
                className={`chat-message-navigator-item${isActive ? ' is-active' : ''}${isPreviewed ? ' is-previewed' : ''}`}
                data-entry-key={entry.key}
                ref={(element) => {
                  if (element) {
                    itemElementsRef.current.set(entry.key, element);
                    return;
                  }
                  itemElementsRef.current.delete(entry.key);
                }}
                tabIndex={isOpen ? 0 : -1}
                type="button"
                onClick={() => {
                  onNavigate(entry);
                }}
              >
                <span className="chat-message-navigator-item-text">{entry.preview}</span>
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
