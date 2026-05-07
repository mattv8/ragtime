import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
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

interface CheckboxDropdownProps {
  options: CheckboxDropdownOption[];
  selectedIds: string[];
  onChange: (ids: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  /** Label shown above the option list inside the panel */
  searchPlaceholder?: string;
}

export function CheckboxDropdown({
  options,
  selectedIds,
  onChange,
  placeholder = 'Select…',
  disabled = false,
  searchPlaceholder = 'Search…',
}: CheckboxDropdownProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [panelStyle, setPanelStyle] = useState<CSSProperties | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

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
    const MARGIN = 8;
    const maxHeight = Math.max(120, window.innerHeight - rect.bottom - MARGIN);
    setPanelStyle({
      position: 'fixed',
      top: rect.bottom + 4,
      left: rect.left,
      width: Math.max(rect.width, 220),
      maxHeight,
      zIndex: 9100,
    });
  }, []);

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

  // Focus search when opening
  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 0);
    } else {
      setSearch('');
      setPanelStyle(null);
    }
  }, [open]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleDown(e: MouseEvent) {
      const target = e.target as Node;
      if (
        !wrapRef.current?.contains(target)
        && !panelRef.current?.contains(target)
      ) {
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
  const triggerLabel = checkedLabels.length === 0
    ? placeholder
    : checkedLabels.join(', ');

  return (
    <div className="chk-dropdown-wrap" ref={wrapRef}>
      <button
        type="button"
        className={`chk-dropdown-trigger${open ? ' open' : ''}${disabled ? ' disabled' : ''}`}
        onClick={() => {
          if (disabled) return;
          if (!open) computePosition();
          setOpen((v) => !v);
        }}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="chk-dropdown-trigger-label" title={triggerLabel}>{triggerLabel}</span>
        <ChevronDown size={14} className={`chk-dropdown-chevron${open ? ' rotated' : ''}`} />
      </button>

      {open && panelStyle && createPortal(
        <div
          ref={panelRef}
          className="chk-dropdown-panel model-selector-dropdown-inner"
          style={panelStyle}
        >
          {options.length > 5 && (
            <div className="model-selector-search">
              <input
                ref={searchRef}
                type="text"
                placeholder={searchPlaceholder}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="model-selector-search-input"
                aria-label="Filter groups"
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
                return (
                <label key={opt.id} className={`checkbox-label chk-dropdown-item${opt.disabled ? ' chk-dropdown-item-disabled' : ''}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(opt)}
                    disabled={disabled || opt.disabled}
                  />
                  <span className="chk-dropdown-option-text">
                    <span className="chk-dropdown-option-main">
                      <span className="chk-dropdown-option-label">{opt.label}</span>
                      {opt.badge && (Array.isArray(opt.badge)
                        ? opt.badge.map((b) => <span key={b} className="chk-dropdown-option-badge">{b}</span>)
                        : <span className="chk-dropdown-option-badge">{opt.badge}</span>
                      )}
                    </span>
                    {opt.description && <span className="chk-dropdown-option-description">{opt.description}</span>}
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
