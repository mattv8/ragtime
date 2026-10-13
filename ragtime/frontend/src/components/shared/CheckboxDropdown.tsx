import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useId,
  type CSSProperties,
  type KeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, X } from 'lucide-react';

export interface CheckboxDropdownOption {
  id: string;
  label: string;
  description?: string;
  badge?: string | string[];
  disabled?: boolean;
  checked?: boolean;
}

export interface CheckboxDropdownProps {
  options: CheckboxDropdownOption[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  /** Label shown above the option list inside the panel */
  searchPlaceholder?: string;
  ariaLabel?: string;
  searchAriaLabel?: string;
  searchThreshold?: number;
  triggerLabel?: string;
  id?: string;
}

const PANEL_MARGIN = 8;
const PANEL_OFFSET = 4;
const PANEL_MAX_HEIGHT = 360;
const PANEL_MIN_WIDTH = 220;
const FOCUSABLE_SELECTOR =
  'input:not(:disabled), button:not(:disabled), [tabindex]:not([tabindex="-1"]):not([aria-disabled="true"])';

export function CheckboxDropdown({
  options,
  selectedIds,
  onChange,
  placeholder = 'Select…',
  disabled = false,
  searchPlaceholder = 'Search…',
  ariaLabel,
  searchAriaLabel = 'Filter groups',
  searchThreshold = 5,
  triggerLabel,
  id,
}: CheckboxDropdownProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [panelStyle, setPanelStyle] = useState<CSSProperties | null>(null);
  const [naturalPanelHeight, setNaturalPanelHeight] = useState(PANEL_MAX_HEIGHT);
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const generatedPanelId = useId();
  const panelId = id ? `${id}-panel` : generatedPanelId;

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const isOptionChecked = useCallback(
    (option: CheckboxDropdownOption) => selectedSet.has(option.id) || Boolean(option.checked),
    [selectedSet],
  );

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => {
      const badgeText = Array.isArray(o.badge) ? o.badge.join(' ') : (o.badge ?? '');
      const haystack = `${o.label} ${o.description ?? ''} ${badgeText}`.toLowerCase();
      return haystack.includes(q);
    });
  }, [options, search]);

  const computePosition = useCallback(() => {
    if (!wrapRef.current) return;
    const rect = wrapRef.current.getBoundingClientRect();
    const spaceBelow = Math.max(0, window.innerHeight - rect.bottom - PANEL_MARGIN - PANEL_OFFSET);
    const spaceAbove = Math.max(0, rect.top - PANEL_MARGIN - PANEL_OFFSET);
    const useAbove = spaceBelow < naturalPanelHeight && spaceAbove > spaceBelow;
    const availableHeight = Math.max(0, useAbove ? spaceAbove : spaceBelow);
    const maxHeight = Math.min(availableHeight, PANEL_MAX_HEIGHT);
    const panelWidth = Math.min(
      Math.max(rect.width, PANEL_MIN_WIDTH),
      Math.max(0, window.innerWidth - PANEL_MARGIN * 2),
    );
    const left = Math.min(
      Math.max(PANEL_MARGIN, rect.left),
      Math.max(PANEL_MARGIN, window.innerWidth - panelWidth - PANEL_MARGIN),
    );
    setPanelStyle({
      position: 'fixed',
      ...(useAbove
        ? { bottom: window.innerHeight - rect.top + PANEL_OFFSET, top: 'auto' }
        : { top: rect.bottom + PANEL_OFFSET }),
      left,
      width: panelWidth,
      maxHeight,
      // This aligns the portal with the theme's tooltip layer: above modals, below toasts.
      zIndex: 'var(--z-tooltip)',
    });
  }, [naturalPanelHeight]);

  useLayoutEffect(() => {
    if (!open) return;

    const panel = panelRef.current;
    const list = panel?.querySelector<HTMLElement>('.chk-dropdown-list');
    if (!panel || !list) return;

    const measuredHeight = panel.offsetHeight - list.clientHeight + list.scrollHeight;
    if (measuredHeight > 0) {
      setNaturalPanelHeight((currentHeight) =>
        currentHeight === Math.min(measuredHeight, PANEL_MAX_HEIGHT)
          ? currentHeight
          : Math.min(measuredHeight, PANEL_MAX_HEIGHT),
      );
    }
  }, [filtered, open]);

  useEffect(() => {
    if (!open) return;
    computePosition();
    window.addEventListener('scroll', computePosition, true);
    window.addEventListener('resize', computePosition);
    return () => {
      window.removeEventListener('scroll', computePosition, true);
      window.removeEventListener('resize', computePosition);
    };
  }, [open, computePosition]);

  useEffect(() => {
    if (open) {
      setTimeout(() => {
        panelRef.current?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)?.focus();
      }, 0);
    } else {
      setSearch('');
      setPanelStyle(null);
      setNaturalPanelHeight(PANEL_MAX_HEIGHT);
    }
  }, [open]);

  useEffect(() => {
    if (disabled && open) {
      setOpen(false);
      requestAnimationFrame(() => {
        const dialog = triggerRef.current?.closest<HTMLElement>('[role="dialog"]');
        const fallback = Array.from(
          dialog?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [],
        ).find((element) => element !== triggerRef.current);
        fallback?.focus();
      });
    }
  }, [disabled, open]);

  const restoreTriggerFocus = useCallback(() => {
    requestAnimationFrame(() => triggerRef.current?.focus());
  }, []);

  const closeAndRestoreFocus = useCallback(() => {
    setOpen(false);
    restoreTriggerFocus();
  }, [restoreTriggerFocus]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleDown(e: MouseEvent) {
      const target = e.target as Node;
      if (!wrapRef.current?.contains(target) && !panelRef.current?.contains(target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleDown);
    return () => document.removeEventListener('mousedown', handleDown);
  }, [open]);

  function toggle(option: CheckboxDropdownOption) {
    if (option.disabled) {
      return;
    }
    const { id } = option;
    if (selectedSet.has(id)) {
      onChange(selectedIds.filter((x) => x !== id));
    } else {
      onChange([...selectedIds, id]);
    }
  }

  const checkedLabels = options.filter(isOptionChecked).map((o) => o.label);
  const visibleTriggerLabel =
    triggerLabel ?? (checkedLabels.length === 0 ? placeholder : checkedLabels.join(', '));

  function handlePanelKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      closeAndRestoreFocus();
      return;
    }

    if (event.key !== 'Tab') return;

    event.stopPropagation();

    const focusableElements = Array.from(
      panelRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [],
    );
    const focusedIndex = focusableElements.indexOf(document.activeElement as HTMLElement);
    const reachesBoundary =
      focusedIndex !== -1 &&
      (event.shiftKey ? focusedIndex === 0 : focusedIndex === focusableElements.length - 1);

    if (reachesBoundary) {
      event.preventDefault();
      event.stopPropagation();
      closeAndRestoreFocus();
    }
  }

  return (
    <div id={id} className="chk-dropdown-wrap" ref={wrapRef}>
      <button
        ref={triggerRef}
        type="button"
        className={`chk-dropdown-trigger${open ? ' open' : ''}${disabled ? ' disabled' : ''}`}
        onClick={() => {
          if (disabled) return;
          if (!open) computePosition();
          setOpen((v) => !v);
        }}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={ariaLabel}
        aria-controls={panelId}
      >
        <span className="chk-dropdown-trigger-label" title={visibleTriggerLabel}>
          {visibleTriggerLabel}
        </span>
        <ChevronDown size={14} className={`chk-dropdown-chevron${open ? ' rotated' : ''}`} />
      </button>

      {open &&
        panelStyle &&
        createPortal(
          <div
            ref={panelRef}
            className="chk-dropdown-panel model-selector-dropdown-inner"
            style={panelStyle}
            id={panelId}
            role="dialog"
            aria-label={ariaLabel ?? placeholder}
            onKeyDown={handlePanelKeyDown}
          >
            {options.length > searchThreshold && (
              <div className="model-selector-search">
                <input
                  ref={searchRef}
                  type="text"
                  placeholder={searchPlaceholder}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="model-selector-search-input"
                  aria-label={searchAriaLabel}
                />
                {search && (
                  <button
                    type="button"
                    className="model-selector-search-clear"
                    onClick={() => {
                      setSearch('');
                      searchRef.current?.focus();
                    }}
                    title="Clear search"
                    aria-label="Clear search"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            )}
            <div className="chk-dropdown-list">
              {filtered.length === 0 ? (
                <span className="chk-dropdown-empty">No matches</span>
              ) : (
                filtered.map((opt) => {
                  const checked = isOptionChecked(opt);
                  const descriptionId = `${panelId}-${opt.id}-description`;
                  const hasDescription = Boolean(opt.badge || opt.description);
                  return (
                    <label
                      key={opt.id}
                      className={`checkbox-label chk-dropdown-item${opt.disabled ? ' chk-dropdown-item-disabled' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggle(opt)}
                        disabled={disabled || opt.disabled}
                        aria-label={opt.label}
                        aria-describedby={hasDescription ? descriptionId : undefined}
                      />
                      <span className="chk-dropdown-option-text">
                        <span className="chk-dropdown-option-main">
                          <span className="chk-dropdown-option-label">{opt.label}</span>
                          {opt.badge && (
                            <span id={descriptionId} className="chk-dropdown-option-badge">
                              {Array.isArray(opt.badge) ? opt.badge.join(', ') : opt.badge}
                            </span>
                          )}
                        </span>
                        {opt.description && (
                          <span
                            id={opt.badge ? undefined : descriptionId}
                            className="chk-dropdown-option-description"
                          >
                            {opt.description}
                          </span>
                        )}
                      </span>
                    </label>
                  );
                })
              )}
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
