import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { normalizeProviderAlias } from '@/utils/modelProviders';

// Generic model interface that both AvailableModel and LLMModel satisfy
interface BaseModel {
  id: string;
  name: string;
  group?: string;
  is_latest?: boolean;
}

interface ModelSelectorProps<T extends BaseModel> {
  models: T[];
  selectedModelId: string;
  onModelChange: (modelId: string) => void;
  getModelSelectionKey?: (model: T) => string;
  disabled?: boolean;
  loading?: boolean;
  placeholder?: string;
  /** Variant: 'compact' for chat header, 'full' for settings forms */
  variant?: 'compact' | 'full';
  triggerIcon?: ReactNode;
  triggerClassName?: string;
}

interface GroupedModels<T extends BaseModel> {
  group: string;
  latestModel: T;
  otherModels: T[];
}

function inferCompactFamilyLabel(model: BaseModel): string | null {
  const id = model.id.toLowerCase();
  if (id.includes('claude-haiku-4-5') || id.includes('claude-haiku-4.5')) {
    return 'Haiku';
  }
  if (id.includes('claude-haiku-4')) {
    return 'Haiku';
  }
  if (id.includes('claude-3-5-haiku') || id.includes('claude-3.5-haiku') || id.includes('claude-3-haiku')) {
    return 'Haiku';
  }
  if (id.includes('claude-sonnet-4')) {
    return 'Claude Sonnet 4';
  }
  if (id.includes('claude-opus-4')) {
    return 'Claude Opus 4';
  }
  if (id.includes('claude')) {
    return 'Claude';
  }
  return null;
}

function inferCompactFamilyLabelFromId(modelId: string): string | null {
  const scopedId = modelId
    .toLowerCase()
    .replace(/^.*::/, '');
  const slashIndex = scopedId.indexOf('/');
  const normalizedId = slashIndex > 0
    && ['anthropic', 'github_copilot'].includes(normalizeProviderAlias(scopedId.slice(0, slashIndex)) || '')
    ? scopedId.slice(slashIndex + 1)
    : scopedId;
  return inferCompactFamilyLabel({
    id: normalizedId,
    name: modelId,
  });
}

/**
 * Model selector dropdown with expandable submenus on hover.
 *
 * Shows only the latest model per group by default. When hovering over a group,
 * an expandable submenu opens to the right allowing selection of specific model versions.
 */
export function ModelSelector<T extends BaseModel>({
  models,
  selectedModelId,
  onModelChange,
  getModelSelectionKey,
  disabled,
  loading,
  placeholder = 'Select model',
  variant = 'compact',
  triggerIcon,
  triggerClassName,
}: ModelSelectorProps<T>) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [submenuPosition, setSubmenuPosition] = useState<{ top: number; left: number } | null>(null);
  const [dropdownPosition, setDropdownPosition] = useState<{ top: number; left: number } | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const expandTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const collapseTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const groupRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  const selectionKeyFor = useCallback((model: T): string => {
    return getModelSelectionKey ? getModelSelectionKey(model) : model.id;
  }, [getModelSelectionKey]);

  // Group models and identify latest in each group
  const groupedModels = useMemo((): GroupedModels<T>[] => {
    const groups: Record<string, T[]> = {};

    models.forEach(model => {
      const group = model.group || 'Other';
      if (!groups[group]) groups[group] = [];
      groups[group].push(model);
    });

    return Object.entries(groups).map(([group, groupModels]) => {
      // Find the latest model (marked by backend) or fallback to first
      const latestModel = groupModels.find(m => m.is_latest) || groupModels[0];
      const otherModels = groupModels.filter(m => m.id !== latestModel.id);

      return { group, latestModel, otherModels };
    }).sort((a, b) => {
      // Sort by group name, putting "Other" groups at the end
      if (a.group.startsWith('Other') && !b.group.startsWith('Other')) return 1;
      if (!a.group.startsWith('Other') && b.group.startsWith('Other')) return -1;
      return a.group.localeCompare(b.group);
    });
  }, [models]);

  // Filter groups for display
  const displayGroups = useMemo(() => {
    return groupedModels.filter(group => group.otherModels.length > 0 || !group.group.startsWith('Other'));
  }, [groupedModels]);

  // Flat filtered list when search is active. Searches across every model in
  // every group (including non-latest variants) so users can jump straight to
  // a specific version without expanding submenus.
  const filteredModels = useMemo((): T[] => {
    const needle = searchQuery.trim().toLowerCase();
    if (!needle) return [];
    return models.filter((model) => {
      const haystack = [model.name, model.id, model.group || '']
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [models, searchQuery]);
  const isSearching = searchQuery.trim().length > 0;

  // Get expanded group data
  const expandedGroupData = useMemo(() => {
    if (!expandedGroup) return null;
    return displayGroups.find(g => g.group === expandedGroup) || null;
  }, [expandedGroup, displayGroups]);

  // Find current selection display
  const selectedModel = useMemo(() => {
    return models.find((m) => selectionKeyFor(m) === selectedModelId);
  }, [models, selectedModelId, selectionKeyFor]);

  // Display text for the button
  const displayText = useMemo(() => {
    if (!selectedModel) {
      return inferCompactFamilyLabelFromId(selectedModelId) || selectedModelId || placeholder;
    }
    // Prefer the family/group label when available.
    // For full variant we only do this when the group has multiple versions,
    // so singleton groups still show the concrete model name.
    const selectedGroup = selectedModel.group
      ? groupedModels.find(group => group.group === selectedModel.group)
      : undefined;
    const hasMultipleVersions = !!selectedGroup && selectedGroup.otherModels.length > 0;
    if (selectedModel.group && !selectedModel.group.startsWith('Other')) {
      if (variant === 'compact' || hasMultipleVersions) {
        return selectedModel.group;
      }
    }
    if (variant === 'compact') {
      const inferred = inferCompactFamilyLabel(selectedModel);
      if (inferred) return inferred;
    }
    return selectedModel.name;
  }, [selectedModel, selectedModelId, placeholder, variant, groupedModels]);

  // Compute and track fixed dropdown position so it draws over iframes without layout shift
  const computeDropdownPosition = useCallback(() => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setDropdownPosition({ top: rect.bottom, left: rect.left });
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    computeDropdownPosition();
    window.addEventListener('scroll', computeDropdownPosition, true);
    window.addEventListener('resize', computeDropdownPosition);
    return () => {
      window.removeEventListener('scroll', computeDropdownPosition, true);
      window.removeEventListener('resize', computeDropdownPosition);
    };
  }, [isOpen, computeDropdownPosition]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setExpandedGroup(null);
        setSubmenuPosition(null);
        setSearchQuery('');
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  // Reset search and focus the input each time the dropdown opens.
  useEffect(() => {
    if (!isOpen) {
      setSearchQuery('');
      return;
    }
    // Defer focus so the trigger's click event finishes before stealing focus.
    const handle = setTimeout(() => {
      searchInputRef.current?.focus();
    }, 0);
    return () => clearTimeout(handle);
  }, [isOpen]);

  // Clean up timeouts on unmount
  useEffect(() => {
    return () => {
      if (expandTimeoutRef.current) clearTimeout(expandTimeoutRef.current);
      if (collapseTimeoutRef.current) clearTimeout(collapseTimeoutRef.current);
    };
  }, []);

  const handleGroupMouseEnter = useCallback((group: string, hasSubmodels: boolean) => {
    if (!hasSubmodels) {
      setExpandedGroup(null);
      setSubmenuPosition(null);
      return;
    }

    // Clear any pending collapse
    if (collapseTimeoutRef.current) {
      clearTimeout(collapseTimeoutRef.current);
      collapseTimeoutRef.current = null;
    }
    // Delay expansion slightly for better UX
    expandTimeoutRef.current = setTimeout(() => {
      // Calculate submenu position using fixed coordinates (to escape overflow:hidden)
      const groupEl = groupRefs.current.get(group);
      const dropdownEl = dropdownRef.current;
      if (groupEl && dropdownEl) {
        const groupRect = groupEl.getBoundingClientRect();
        const dropdownRect = dropdownEl.getBoundingClientRect();
        // Position fixed: top aligns with group, left is right edge of dropdown
        setSubmenuPosition({
          top: groupRect.top,
          left: dropdownRect.right + 2
        });
      }
      setExpandedGroup(group);
    }, 150);
  }, []);

  const handleGroupMouseLeave = useCallback(() => {
    // Clear any pending expansion
    if (expandTimeoutRef.current) {
      clearTimeout(expandTimeoutRef.current);
      expandTimeoutRef.current = null;
    }
    // Delay collapse to allow moving to submenu
    collapseTimeoutRef.current = setTimeout(() => {
      setExpandedGroup(null);
      setSubmenuPosition(null);
    }, 250);
  }, []);

  const handleSubmenuMouseEnter = useCallback(() => {
    // Clear collapse timeout when entering submenu
    if (collapseTimeoutRef.current) {
      clearTimeout(collapseTimeoutRef.current);
      collapseTimeoutRef.current = null;
    }
  }, []);

  const handleSubmenuMouseLeave = useCallback(() => {
    // Collapse when leaving submenu
    collapseTimeoutRef.current = setTimeout(() => {
      setExpandedGroup(null);
      setSubmenuPosition(null);
    }, 150);
  }, []);

  const handleSelectModel = useCallback((modelId: string) => {
    onModelChange(modelId);
    setIsOpen(false);
    setExpandedGroup(null);
    setSubmenuPosition(null);
    setSearchQuery('');
  }, [onModelChange]);

  const handleSelectGroup = useCallback((group: GroupedModels<T>) => {
    // Select the latest model of this group
    handleSelectModel(selectionKeyFor(group.latestModel));
  }, [handleSelectModel, selectionKeyFor]);

  const setGroupRef = useCallback((group: string, el: HTMLDivElement | null) => {
    if (el) {
      groupRefs.current.set(group, el);
    } else {
      groupRefs.current.delete(group);
    }
  }, []);

  if (models.length === 0) {
    const emptyTriggerClassName = variant === 'full'
      ? 'model-selector-trigger model-selector-trigger-full'
      : 'model-selector-trigger';
    const emptyTriggerClasses = `${emptyTriggerClassName}${triggerClassName ? ` ${triggerClassName}` : ''}`;
    return (
      <div className={`model-selector is-disabled ${variant === 'full' ? 'model-selector-full' : ''}`}>
        <button
          type="button"
          className={emptyTriggerClasses}
          disabled
          title={loading ? 'Loading models...' : placeholder}
        >
          {triggerIcon ? <span className="model-selector-icon" aria-hidden="true">{triggerIcon}</span> : null}
          <span className="model-selector-text">
            {loading
              ? <><MiniLoadingSpinner variant="icon" size={12} title="Loading models..." />{' '}Loading...</>
              : (selectedModelId || placeholder)}
          </span>
          <span className="model-selector-arrow">▾</span>
        </button>
      </div>
    );
  }

  const triggerButtonClassName = variant === 'full'
    ? 'model-selector-trigger model-selector-trigger-full'
    : 'model-selector-trigger';

  const triggerClasses = `${triggerButtonClassName}${triggerClassName ? ` ${triggerClassName}` : ''}`;

  return (
    <div
      ref={containerRef}
      className={`model-selector ${disabled ? 'is-disabled' : ''} ${variant === 'full' ? 'model-selector-full' : ''}`}
    >
      <button
        type="button"
        className={triggerClasses}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        title="Select model"
      >
        {triggerIcon ? <span className="model-selector-icon" aria-hidden="true">{triggerIcon}</span> : null}
        <span className="model-selector-text">{displayText}</span>
        <span className="model-selector-arrow">▾</span>
      </button>

      {isOpen && (
        <div
          className="model-selector-dropdown"
          ref={dropdownRef}
          style={dropdownPosition ? { top: dropdownPosition.top, left: dropdownPosition.left } : undefined}
        >
          {/* Inline search filter — only shown when there's something to filter */}
          {models.length > 1 && (
            <div className="model-selector-search">
              <input
                ref={searchInputRef}
                type="text"
                className="model-selector-search-input"
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') {
                    e.preventDefault();
                    if (searchQuery) {
                      setSearchQuery('');
                    } else {
                      setIsOpen(false);
                    }
                  }
                }}
                aria-label="Filter models"
              />
              {searchQuery && (
                <button
                  type="button"
                  className="model-selector-search-clear"
                  onClick={() => {
                    setSearchQuery('');
                    searchInputRef.current?.focus();
                  }}
                  title="Clear search"
                  aria-label="Clear search"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          )}

          {/* Scrollable inner container for main menu items */}
          <div className="model-selector-dropdown-inner">
            {isSearching ? (
              filteredModels.length === 0 ? (
                <div className="model-selector-empty">No models match "{searchQuery.trim()}"</div>
              ) : (
                filteredModels.map((model) => {
                  const key = selectionKeyFor(model);
                  const isSelected = selectedModelId === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`model-selector-item ${isSelected ? 'is-selected' : ''}`}
                      onClick={() => handleSelectModel(key)}
                      title={model.id}
                    >
                      <span className="model-selector-item-name">{model.name}</span>
                      {model.group && !model.group.startsWith('Other') && (
                        <span className="model-selector-expand-indicator" aria-hidden="true">
                          {model.group}
                        </span>
                      )}
                    </button>
                  );
                })
              )
            ) : (
              displayGroups.map((group) => {
              const hasSubmodels = group.otherModels.length > 0;
              const isExpanded = expandedGroup === group.group;

              return (
                <div
                  key={group.group}
                  ref={(el) => setGroupRef(group.group, el)}
                  className={`model-selector-group ${isExpanded ? 'is-expanded' : ''}`}
                  onMouseEnter={() => handleGroupMouseEnter(group.group, hasSubmodels)}
                  onMouseLeave={handleGroupMouseLeave}
                >
                  <button
                    type="button"
                    className={`model-selector-item model-selector-group-item ${
                      selectedModel?.group === group.group ? 'is-selected' : ''
                    }`}
                    onClick={() => handleSelectGroup(group)}
                  >
                    <span className="model-selector-item-name">{group.group}</span>
                    {hasSubmodels && (
                      <span className="model-selector-expand-indicator">›</span>
                    )}
                  </button>
                </div>
              );
            })
            )}
          </div>

          {/* Submenu rendered with position:fixed to escape overflow:hidden */}
          {!isSearching && expandedGroupData && expandedGroupData.otherModels.length > 0 && submenuPosition && (
            <div
              className="model-selector-submenu"
              style={{ top: submenuPosition.top, left: submenuPosition.left }}
              onMouseEnter={handleSubmenuMouseEnter}
              onMouseLeave={handleSubmenuMouseLeave}
            >
              {/* Latest model at top of submenu */}
              <button
                type="button"
                className={`model-selector-item model-selector-subitem ${
                  selectedModelId === selectionKeyFor(expandedGroupData.latestModel) ? 'is-selected' : ''
                }`}
                onClick={() => handleSelectModel(selectionKeyFor(expandedGroupData.latestModel))}
              >
                <span className="model-selector-item-name">{expandedGroupData.latestModel.name}</span>
                <span className="model-selector-latest-badge">Latest</span>
              </button>

              {/* Other models in the group */}
              {expandedGroupData.otherModels.map(model => (
                <button
                  key={selectionKeyFor(model)}
                  type="button"
                  className={`model-selector-item model-selector-subitem ${
                    selectedModelId === selectionKeyFor(model) ? 'is-selected' : ''
                  }`}
                  onClick={() => handleSelectModel(selectionKeyFor(model))}
                >
                  <span className="model-selector-item-name">{model.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
