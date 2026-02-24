import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { ReactNode } from 'react';

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
  disabled?: boolean;
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
  disabled,
  placeholder = 'Select model',
  variant = 'compact',
  triggerIcon,
  triggerClassName,
}: ModelSelectorProps<T>) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedGroup, setExpandedGroup] = useState<string | null>(null);
  const [submenuPosition, setSubmenuPosition] = useState<{ top: number; left: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const expandTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const collapseTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const groupRefs = useRef<Map<string, HTMLDivElement>>(new Map());

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

  // Get expanded group data
  const expandedGroupData = useMemo(() => {
    if (!expandedGroup) return null;
    return displayGroups.find(g => g.group === expandedGroup) || null;
  }, [expandedGroup, displayGroups]);

  // Find current selection display
  const selectedModel = useMemo(() => {
    return models.find(m => m.id === selectedModelId);
  }, [models, selectedModelId]);

  // Display text for the button
  const displayText = useMemo(() => {
    if (!selectedModel) return selectedModelId || placeholder;
    // For compact variant, if selected model is a "latest" model, show group name
    if (variant === 'compact' && selectedModel.is_latest && selectedModel.group) {
      return selectedModel.group;
    }
    return selectedModel.name;
  }, [selectedModel, selectedModelId, placeholder, variant]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setExpandedGroup(null);
        setSubmenuPosition(null);
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
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
  }, [onModelChange]);

  const handleSelectGroup = useCallback((group: GroupedModels<T>) => {
    // Select the latest model of this group
    handleSelectModel(group.latestModel.id);
  }, [handleSelectModel]);

  const setGroupRef = useCallback((group: string, el: HTMLDivElement | null) => {
    if (el) {
      groupRefs.current.set(group, el);
    } else {
      groupRefs.current.delete(group);
    }
  }, []);

  if (models.length === 0) {
    return (
      <span className="chat-model-badge">{selectedModelId || placeholder}</span>
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
        <div className="model-selector-dropdown" ref={dropdownRef}>
          {/* Scrollable inner container for main menu items */}
          <div className="model-selector-dropdown-inner">
            {displayGroups.map((group) => {
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
            })}
          </div>

          {/* Submenu rendered with position:fixed to escape overflow:hidden */}
          {expandedGroupData && expandedGroupData.otherModels.length > 0 && submenuPosition && (
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
                  selectedModelId === expandedGroupData.latestModel.id ? 'is-selected' : ''
                }`}
                onClick={() => handleSelectModel(expandedGroupData.latestModel.id)}
              >
                <span className="model-selector-item-name">{expandedGroupData.latestModel.name}</span>
                <span className="model-selector-latest-badge">Latest</span>
              </button>

              {/* Other models in the group */}
              {expandedGroupData.otherModels.map(model => (
                <button
                  key={model.id}
                  type="button"
                  className={`model-selector-item model-selector-subitem ${
                    selectedModelId === model.id ? 'is-selected' : ''
                  }`}
                  onClick={() => handleSelectModel(model.id)}
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
