import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { CHAT_MODEL_PROVIDER_LABELS } from '@/utils/modelDisplay';
import { normalizeProviderAlias } from '@/utils/modelProviders';

// Generic model interface that both AvailableModel and LLMModel satisfy
interface BaseModel {
  id: string;
  name: string;
  provider?: string;
  group?: string;
  model_provider_label?: string;
  model_family?: string;
  display_name?: string;
  model_variant?: string;
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

/**
 * Generic node in the hover-menu tree. A node is either a grouping node with
 * `children` (further submenus) or a leaf-level node whose `models` are the
 * concrete model rows shown in the deepest submenu.
 */
interface MenuNode<T extends BaseModel> {
  key: string;
  label: string;
  subtitle?: string;
  latestModel: T;
  children?: MenuNode<T>[];
  models?: T[];
}

type MenuSide = 'left' | 'right';

type MenuPosition =
  | { top: number; side: 'right'; left: number }
  | { top: number; side: 'left'; right: number };

const ESTIMATED_SUBMENU_WIDTH = 240;
const SUBMENU_GAP = 2;

function hostProviderLabel(provider: string | undefined): string {
  const normalized = normalizeProviderAlias(provider || '') || provider || '';
  return CHAT_MODEL_PROVIDER_LABELS[normalized] || normalized || '';
}

function modelDisplayName(model: BaseModel): string {
  return model.display_name || model.name || model.id;
}

function modelFamilyLabel(model: BaseModel): string {
  return model.model_family || model.group || 'Other';
}

function modelProviderLabel(model: BaseModel): string {
  return model.model_provider_label || hostProviderLabel(model.provider) || 'Other';
}

function sortOtherLast(a: string, b: string): number {
  if (a.startsWith('Other') && !b.startsWith('Other')) return 1;
  if (!a.startsWith('Other') && b.startsWith('Other')) return -1;
  return a.localeCompare(b);
}

function labelsMatch(a: string | undefined, b: string | undefined): boolean {
  return (a || '').trim().toLowerCase() === (b || '').trim().toLowerCase();
}

function modelVariantLabel(model: BaseModel): string {
  return model.model_variant || modelDisplayName(model);
}

function shouldGroupModelVariants<T extends BaseModel>(groups: Map<string, T[]>, totalModels: number): boolean {
  if (totalModels < 12) return false;
  return groups.size > 1 && groups.size < totalModels;
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
  const [expandedPath, setExpandedPath] = useState<string[]>([]);
  const [submenuPositions, setSubmenuPositions] = useState<MenuPosition[]>([]);
  const [rootChildSide, setRootChildSide] = useState<MenuSide>('right');
  const [dropdownPosition, setDropdownPosition] = useState<{ top: number; left: number } | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);
  const expandTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const collapseTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Refs to item elements per depth so we can measure for submenu positioning.
  const itemRefs = useRef<Map<number, Map<string, HTMLDivElement>>>(new Map());
  // Refs to the submenu container elements at each depth (depth = submenu index + 1).
  const submenuRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  const selectionKeyFor = useCallback((model: T): string => {
    return getModelSelectionKey ? getModelSelectionKey(model) : model.id;
  }, [getModelSelectionKey]);

  const isModelSelected = useCallback((model: T): boolean => {
    return selectedModelId === selectionKeyFor(model);
  }, [selectedModelId, selectionKeyFor]);

  // Build host → provider → family → models tree. Collapse the host level when
  // exactly one host is in play so the dropdown stays compact.
  const menuTree = useMemo((): MenuNode<T>[] => {
    const byHost = new Map<string, Map<string, Map<string, T[]>>>();
    const hostLabels = new Map<string, string>();
    const providerLabels = new Map<string, string>();

    models.forEach((model) => {
      const host = hostProviderLabel(model.provider) || 'Other';
      const hostKey = host.toLowerCase();
      const provider = modelProviderLabel(model);
      const providerKey = provider.toLowerCase();
      const family = modelFamilyLabel(model);

      hostLabels.set(hostKey, host);
      providerLabels.set(`${hostKey}::${providerKey}`, provider);

      if (!byHost.has(hostKey)) byHost.set(hostKey, new Map());
      const providers = byHost.get(hostKey)!;
      if (!providers.has(providerKey)) providers.set(providerKey, new Map());
      const families = providers.get(providerKey)!;
      if (!families.has(family)) families.set(family, []);
      families.get(family)!.push(model);
    });

    const buildFamilyNodes = (
      hostKey: string,
      providerKey: string,
      familiesByName: Map<string, T[]>,
    ): MenuNode<T>[] => {
      return [...familiesByName.entries()].map(([family, familyModels]) => {
        const latestModel = familyModels.find(m => m.is_latest) || familyModels[0];
        const modelsByVariant = new Map<string, T[]>();

        familyModels.forEach((model) => {
          const variant = modelVariantLabel(model);
          if (!modelsByVariant.has(variant)) modelsByVariant.set(variant, []);
          modelsByVariant.get(variant)!.push(model);
        });

        const children = shouldGroupModelVariants(modelsByVariant, familyModels.length)
          ? [...modelsByVariant.entries()].map(([variant, variantModels]) => ({
              key: `${hostKey}::${providerKey}::${family}::${variant.toLowerCase()}`,
              label: variant,
              latestModel: variantModels.find(m => m.is_latest) || variantModels[0],
              models: variantModels,
            }))
          : undefined;

        return {
          key: `${hostKey}::${providerKey}::${family}`,
          label: family,
          latestModel,
          ...(children ? { children } : { models: familyModels }),
        };
      }).sort((a, b) => sortOtherLast(a.label, b.label));
    };

    const buildProviderNodes = (
      hostKey: string,
      providersByKey: Map<string, Map<string, T[]>>,
    ): MenuNode<T>[] => {
      return [...providersByKey.entries()].map(([providerKey, familiesByName]) => {
        const children = buildFamilyNodes(hostKey, providerKey, familiesByName);
        const providerLabel = providerLabels.get(`${hostKey}::${providerKey}`) || providerKey;
        const onlyChild = children.length === 1 ? children[0] : undefined;
        if (onlyChild && labelsMatch(providerLabel, onlyChild.label)) {
          return {
            ...onlyChild,
            key: `${hostKey}::${providerKey}`,
            label: providerLabel,
          };
        }

        const latestModel = children.find(c => c.latestModel.is_latest)?.latestModel
          || children[0]?.latestModel
          || models[0];
        return {
          key: `${hostKey}::${providerKey}`,
          label: providerLabel,
          latestModel,
          children,
        };
      }).sort((a, b) => sortOtherLast(a.label, b.label));
    };

    const buildHostChildren = (
      hostKey: string,
      providersByKey: Map<string, Map<string, T[]>>,
    ): MenuNode<T>[] => {
      const providerEntries = [...providersByKey.entries()];
      if (providerEntries.length === 1) {
        const [[providerKey, familiesByName]] = providerEntries;
        const hostLabel = hostLabels.get(hostKey) || hostKey;
        const providerLabel = providerLabels.get(`${hostKey}::${providerKey}`) || providerKey;

        if (labelsMatch(hostLabel, providerLabel)) {
          return buildFamilyNodes(hostKey, providerKey, familiesByName);
        }
      }

      return buildProviderNodes(hostKey, providersByKey);
    };

    const hostEntries = [...byHost.entries()];

    // Collapse host tier when there's only one configured host.
    if (hostEntries.length <= 1) {
      const [hostKey, providersByKey] = hostEntries[0] || ['', new Map()];
      return buildHostChildren(hostKey, providersByKey);
    }

    return hostEntries.map(([hostKey, providersByKey]) => {
      const children = buildHostChildren(hostKey, providersByKey);
      const latestModel = children[0]?.latestModel || models[0];
      return {
        key: hostKey,
        label: hostLabels.get(hostKey) || hostKey,
        latestModel,
        children,
      };
    }).sort((a, b) => sortOtherLast(a.label, b.label));
  }, [models]);

  // Resolve the chain of nodes corresponding to the current expanded path so
  // rendering can pull each level's child list directly.
  const expandedNodes = useMemo((): MenuNode<T>[] => {
    const chain: MenuNode<T>[] = [];
    let level: MenuNode<T>[] | undefined = menuTree;
    for (const key of expandedPath) {
      if (!level) break;
      const next: MenuNode<T> | undefined = level.find(n => n.key === key);
      if (!next) break;
      chain.push(next);
      level = next.children;
    }
    return chain;
  }, [menuTree, expandedPath]);

  // Flat filtered list when search is active. Searches across every model in
  // every group (including non-latest variants) so users can jump straight to
  // a specific version without expanding submenus.
  const filteredModels = useMemo((): T[] => {
    const needle = searchQuery.trim().toLowerCase();
    if (!needle) return [];
    return models.filter((model) => {
      const haystack = [
        modelDisplayName(model),
        model.name,
        model.id,
        model.group || '',
        model.model_family || '',
        model.model_provider_label || '',
        hostProviderLabel(model.provider),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }, [models, searchQuery]);
  const isSearching = searchQuery.trim().length > 0;

  // Find current selection display
  const selectedModel = useMemo(() => {
    return models.find((m) => selectionKeyFor(m) === selectedModelId);
  }, [models, selectedModelId, selectionKeyFor]);

  // Display text for the button
  const displayText = useMemo(() => {
    if (!selectedModel) {
      return inferCompactFamilyLabelFromId(selectedModelId) || selectedModelId || placeholder;
    }
    const selectedKey = selectionKeyFor(selectedModel);
    // Find the leaf family node containing this model so we can decide whether
    // to show the family label or the concrete model name.
    const findLeaf = (nodes: MenuNode<T>[]): MenuNode<T> | undefined => {
      for (const node of nodes) {
        if (node.models?.some(m => selectionKeyFor(m) === selectedKey)) return node;
        if (node.children) {
          const found = findLeaf(node.children);
          if (found) return found;
        }
      }
      return undefined;
    };
    const leaf = findLeaf(menuTree);
    const hasMultipleVersions = !!leaf && (leaf.models?.length || 0) > 1;
    const familyLabel = modelFamilyLabel(selectedModel);
    if (familyLabel && !familyLabel.startsWith('Other')) {
      if (variant === 'compact' || hasMultipleVersions) {
        return familyLabel;
      }
    }
    if (variant === 'compact') {
      const inferred = inferCompactFamilyLabel(selectedModel);
      if (inferred) return inferred;
    }
    return modelDisplayName(selectedModel);
  }, [selectedModel, selectedModelId, placeholder, variant, menuTree, selectionKeyFor]);

  // Compute and track fixed dropdown position so it draws over iframes without layout shift
  const computeDropdownPosition = useCallback(() => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setDropdownPosition({ top: rect.bottom, left: rect.left });
  }, []);

  // Decide whether the root-level child submenu will open left or right based
  // on space around the dropdown. Drives root-row chevron direction.
  const computeRootChildSide = useCallback(() => {
    const dropdownEl = dropdownRef.current;
    if (!dropdownEl) return;
    const rect = dropdownEl.getBoundingClientRect();
    const spaceOnRight = window.innerWidth - rect.right;
    const spaceOnLeft = rect.left;
    const openLeft = spaceOnRight < ESTIMATED_SUBMENU_WIDTH + SUBMENU_GAP && spaceOnLeft > spaceOnRight;
    setRootChildSide(openLeft ? 'left' : 'right');
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    computeDropdownPosition();
    // Defer side measurement until the dropdown is rendered.
    const raf = requestAnimationFrame(computeRootChildSide);
    window.addEventListener('scroll', computeDropdownPosition, true);
    window.addEventListener('resize', computeDropdownPosition);
    window.addEventListener('resize', computeRootChildSide);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('scroll', computeDropdownPosition, true);
      window.removeEventListener('resize', computeDropdownPosition);
      window.removeEventListener('resize', computeRootChildSide);
    };
  }, [isOpen, computeDropdownPosition, computeRootChildSide]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (
        containerRef.current
        && !containerRef.current.contains(target)
        && !dropdownRef.current?.contains(target)
      ) {
        setIsOpen(false);
        setExpandedPath([]);
        setSubmenuPositions([]);
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

  // Compute submenu position when hovering an item at `level` whose children
  // should open in `preferredSide`. Falls back to the opposite side if there's
  // not enough room.
  const computeSubmenuPosition = useCallback(
    (level: number, nodeKey: string, preferredSide: MenuSide): MenuPosition | null => {
      const itemEl = itemRefs.current.get(level)?.get(nodeKey);
      // The parent container is the dropdown (level 0) or the submenu at depth = level.
      const parentEl = level === 0 ? dropdownRef.current : submenuRefs.current.get(level);
      if (!itemEl || !parentEl) return null;
      const itemRect = itemEl.getBoundingClientRect();
      const parentRect = parentEl.getBoundingClientRect();
      const spaceOnRight = window.innerWidth - parentRect.right;
      const spaceOnLeft = parentRect.left;
      let side: MenuSide = preferredSide;
      if (side === 'right' && spaceOnRight < ESTIMATED_SUBMENU_WIDTH + SUBMENU_GAP && spaceOnLeft > spaceOnRight) {
        side = 'left';
      } else if (side === 'left' && spaceOnLeft < ESTIMATED_SUBMENU_WIDTH + SUBMENU_GAP && spaceOnRight > spaceOnLeft) {
        side = 'right';
      }
      if (side === 'right') {
        return { top: itemRect.top, side: 'right', left: parentRect.right + SUBMENU_GAP };
      }
      return { top: itemRect.top, side: 'left', right: window.innerWidth - parentRect.left + SUBMENU_GAP };
    },
    [],
  );

  // Side at which children of `level` will open. Level 0 uses rootChildSide;
  // deeper levels inherit from the submenu that contains them.
  const childSideForLevel = useCallback(
    (level: number): MenuSide => {
      if (level === 0) return rootChildSide;
      const parentSubmenu = submenuPositions[level - 1];
      return parentSubmenu?.side || rootChildSide;
    },
    [rootChildSide, submenuPositions],
  );

  const cancelCollapse = useCallback(() => {
    if (collapseTimeoutRef.current) {
      clearTimeout(collapseTimeoutRef.current);
      collapseTimeoutRef.current = null;
    }
  }, []);

  const cancelExpand = useCallback(() => {
    if (expandTimeoutRef.current) {
      clearTimeout(expandTimeoutRef.current);
      expandTimeoutRef.current = null;
    }
  }, []);

  const scheduleCollapseAll = useCallback(() => {
    cancelExpand();
    collapseTimeoutRef.current = setTimeout(() => {
      setExpandedPath([]);
      setSubmenuPositions([]);
    }, 500);
  }, [cancelExpand]);

  const handleItemMouseEnter = useCallback(
    (level: number, node: MenuNode<T>) => {
      cancelCollapse();
      cancelExpand();
      const hasChildren = (node.children?.length || 0) > 0 || (node.models?.length || 0) > 0;
      if (!hasChildren) {
        // Defer collapsing deeper levels so diagonal mouse movement toward a
        // sibling submenu doesn't immediately destroy the open submenu.
        collapseTimeoutRef.current = setTimeout(() => {
          setExpandedPath(prev => prev.slice(0, level));
          setSubmenuPositions(prev => prev.slice(0, level));
        }, 120);
        return;
      }
      const delay = level === 0 ? 150 : 120;
      expandTimeoutRef.current = setTimeout(() => {
        const preferred = childSideForLevel(level);
        const position = computeSubmenuPosition(level, node.key, preferred);
        setExpandedPath(prev => [...prev.slice(0, level), node.key]);
        setSubmenuPositions(prev => {
          const next = prev.slice(0, level);
          if (position) next.push(position);
          return next;
        });
      }, delay);
    },
    [cancelCollapse, cancelExpand, childSideForLevel, computeSubmenuPosition],
  );

  const handleSubmenuMouseEnter = useCallback(() => {
    cancelCollapse();
  }, [cancelCollapse]);

  const handleSubmenuMouseLeave = useCallback(() => {
    scheduleCollapseAll();
  }, [scheduleCollapseAll]);

  const handleSelectModel = useCallback((modelId: string) => {
    onModelChange(modelId);
    setIsOpen(false);
    setExpandedPath([]);
    setSubmenuPositions([]);
    setSearchQuery('');
  }, [onModelChange]);

  const handleSelectNode = useCallback((node: MenuNode<T>) => {
    handleSelectModel(selectionKeyFor(node.latestModel));
  }, [handleSelectModel, selectionKeyFor]);

  const setItemRef = useCallback((level: number, nodeKey: string, el: HTMLDivElement | null) => {
    let levelMap = itemRefs.current.get(level);
    if (!levelMap) {
      levelMap = new Map();
      itemRefs.current.set(level, levelMap);
    }
    if (el) {
      levelMap.set(nodeKey, el);
    } else {
      levelMap.delete(nodeKey);
    }
  }, []);

  const setSubmenuRef = useCallback((depth: number, el: HTMLDivElement | null) => {
    if (el) {
      submenuRefs.current.set(depth, el);
    } else {
      submenuRefs.current.delete(depth);
    }
  }, []);

  const containsSelectedModel = useCallback((node: MenuNode<T>): boolean => {
    if (node.models?.some(isModelSelected)) return true;
    return !!node.children?.some(containsSelectedModel);
  }, [isModelSelected]);

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

  // Render a single grouping row (host/provider/family). Used at every depth
  // for non-leaf nodes; leaf model rows are rendered separately below.
  const renderGroupRow = (node: MenuNode<T>, level: number, classNameBase: string) => {
    const isExpanded = expandedPath[level] === node.key;
    const isSelected = containsSelectedModel(node);
    const hasChildren = (node.children?.length || 0) > 0 || (node.models?.length || 0) > 0;
    const childSide = childSideForLevel(level);
    const chevron = childSide === 'left' ? '‹' : '›';
    const directionClass = hasChildren ? ` has-${childSide}-submenu` : '';
    return (
      <div
        key={node.key}
        ref={(el) => setItemRef(level, node.key, el)}
        className={`model-selector-group ${isExpanded ? 'is-expanded' : ''}${isSelected ? ' is-selected' : ''}`}
        onMouseEnter={() => handleItemMouseEnter(level, node)}
      >
        <button
          type="button"
          className={`model-selector-item ${classNameBase}${directionClass} ${isSelected ? 'is-selected' : ''}`}
          onClick={() => handleSelectNode(node)}
        >
          <span className="model-selector-item-main">
            <span className="model-selector-item-name">{node.label}</span>
            {node.subtitle && (
              <span className="model-selector-item-meta">{node.subtitle}</span>
            )}
          </span>
          {hasChildren && (
            <span className="model-selector-expand-indicator" aria-hidden="true">{chevron}</span>
          )}
        </button>
      </div>
    );
  };

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

      {isOpen && createPortal(
        <div
          className="model-selector-dropdown"
          ref={dropdownRef}
          style={dropdownPosition ? { top: dropdownPosition.top, left: dropdownPosition.left } : undefined}
          onMouseEnter={cancelCollapse}
          onMouseLeave={scheduleCollapseAll}
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

          {/* Scrollable inner container for root menu items */}
          <div className="model-selector-dropdown-inner">
            {isSearching ? (
              filteredModels.length === 0 ? (
                <div className="model-selector-empty">No models match "{searchQuery.trim()}"</div>
              ) : (
                filteredModels.map((model) => {
                  const key = selectionKeyFor(model);
                  const isSelected = isModelSelected(model);
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`model-selector-item ${isSelected ? 'is-selected' : ''}`}
                      onClick={() => handleSelectModel(key)}
                      title={model.id}
                    >
                      <span className="model-selector-item-main">
                        <span className="model-selector-item-name">{modelDisplayName(model)}</span>
                        {(model.model_provider_label || model.model_family || model.group) && (
                          <span className="model-selector-item-meta">
                            {[model.model_provider_label, model.model_family || model.group].filter(Boolean).join(' / ')}
                          </span>
                        )}
                      </span>
                      {modelFamilyLabel(model) && !modelFamilyLabel(model).startsWith('Other') && (
                        <span className="model-selector-expand-indicator" aria-hidden="true">
                          {modelFamilyLabel(model)}
                        </span>
                      )}
                    </button>
                  );
                })
              )
            ) : (
              menuTree.map((node) => renderGroupRow(node, 0, 'model-selector-group-item'))
            )}
          </div>

          {/* One submenu portal per open level. Each shows the children (or
              leaf models) of the corresponding expanded node. */}
          {!isSearching && expandedNodes.map((node, idx) => {
            const depth = idx + 1;
            const position = submenuPositions[idx];
            if (!position) return null;
            const childChevronSide = position.side;
            const chevron = childChevronSide === 'left' ? '‹' : '›';
            const submenuClass = `model-selector-submenu${position.side === 'left' ? ' opens-left' : ''}`;
            const isLeaf = !node.children && !!node.models;
            return (
              <div
                key={`${node.key}-${depth}`}
                ref={(el) => setSubmenuRef(depth, el)}
                className={submenuClass}
                style={position.side === 'left'
                  ? { top: position.top, right: position.right }
                  : { top: position.top, left: position.left }}
                onMouseEnter={handleSubmenuMouseEnter}
                onMouseLeave={handleSubmenuMouseLeave}
              >
                <div className="model-selector-submenu-inner">
                {isLeaf
                  ? (node.models || []).map((model) => (
                      <button
                        key={selectionKeyFor(model)}
                        type="button"
                        className={`model-selector-item model-selector-subitem ${
                          isModelSelected(model) ? 'is-selected' : ''
                        }`}
                        onClick={() => handleSelectModel(selectionKeyFor(model))}
                      >
                        <span className="model-selector-item-name">{modelDisplayName(model)}</span>
                        {model.is_latest && <span className="model-selector-latest-badge">Latest</span>}
                      </button>
                    ))
                  : (node.children || []).map((child) => {
                      const isExpanded = expandedPath[depth] === child.key;
                      const isSelected = containsSelectedModel(child);
                      const hasChildren = (child.children?.length || 0) > 0 || (child.models?.length || 0) > 0;
                      const directionClass = hasChildren ? ` has-${childChevronSide}-submenu` : '';
                      return (
                        <div
                          key={child.key}
                          ref={(el) => setItemRef(depth, child.key, el)}
                          className={`model-selector-group ${isExpanded ? 'is-expanded' : ''}${isSelected ? ' is-selected' : ''}`}
                          onMouseEnter={() => handleItemMouseEnter(depth, child)}
                        >
                          <button
                            type="button"
                            className={`model-selector-item model-selector-subitem${directionClass} ${
                              isSelected ? 'is-selected' : ''
                            }`}
                            onClick={() => handleSelectNode(child)}
                          >
                            <span className="model-selector-item-name">{child.label}</span>
                            {hasChildren && (
                              <span className="model-selector-expand-indicator" aria-hidden="true">{chevron}</span>
                            )}
                          </button>
                        </div>
                      );
                    })}
                </div>
              </div>
            );
          })}
        </div>,
        document.body,
      )}
    </div>
  );
}
