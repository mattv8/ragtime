import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Settings, ChevronRight, X } from 'lucide-react';
import {
  getEffectiveUserSpaceToolIdSet,
  getSelectableUserSpaceToolIds,
  getUserSpaceGroupCheckState,
  isUserSpaceToolAvailable,
  normalizeToolSelectionMode,
  setUserSpaceToolSelectionForTools,
  toggleUserSpaceToolGroupSelection,
  toggleUserSpaceToolSelection,
  type UserSpaceToolSelection,
} from '@/utils/userSpaceTools';
import type { ToolSelectionMode } from '@/types';

interface ToolSelectorTool {
  id: string;
  name: string;
  tool_type: string;
  description?: string | null;
  group_id?: string | null;
  group_name?: string | null;
  available?: boolean;
  disabled_reason?: string | null;
}

export interface ToolGroupInfo {
  id: string;
  name: string;
}

interface ToolSelectorDropdownProps {
  availableTools: ToolSelectorTool[];
  selectedToolIds: Set<string>;
  toolSelectionMode?: ToolSelectionMode;
  onSelectionChange: (selection: UserSpaceToolSelection) => void;
  builtInTools?: ToolSelectorTool[];
  selectedBuiltInToolIds?: Set<string>;
  onToggleBuiltInTool?: (toolId: string) => void;
  onBulkBuiltInToggle?: (selected: boolean) => void;
  /** Selected tool group IDs. When a group is selected, all its tools are effectively enabled. */
  selectedToolGroupIds?: Set<string>;
  toolGroups?: ToolGroupInfo[];
  /** Which direction the dropdown opens. 'down' (default) for toolbars, 'up' for inline chat input. */
  openDirection?: 'down' | 'up';
  disabled?: boolean;
  readOnly?: boolean;
  saving?: boolean;
  title?: string;
  workspaceBuiltInSectionLabel?: string;
  /** When provided, renders a "Show tool calls" toggle at the bottom of the dropdown. */
  showToolCalls?: boolean;
  onToggleToolCalls?: (value: boolean) => void;
}

export function ToolSelectorDropdown({
  availableTools,
  selectedToolIds,
  toolSelectionMode,
  onSelectionChange,
  builtInTools = [],
  selectedBuiltInToolIds,
  onToggleBuiltInTool,
  onBulkBuiltInToggle,
  selectedToolGroupIds,
  toolGroups,
  openDirection = 'down',
  disabled = false,
  readOnly = false,
  saving = false,
  title = 'Tools',
  workspaceBuiltInSectionLabel = 'Workspace Tools',
  showToolCalls,
  onToggleToolCalls,
}: ToolSelectorDropdownProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [dropdownPosition, setDropdownPosition] = useState<{
    top: number;
    left: number;
    minWidth: number;
    maxHeight: number;
  } | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Compute fixed position so the dropdown draws over iframes without layout shift
  const computeDropdownPosition = useCallback(() => {
    if (!dropdownRef.current) return;
    const rect = dropdownRef.current.getBoundingClientRect();
    const MARGIN = 8;
    if (openDirection === 'up') {
      const maxHeight = Math.max(80, rect.top - MARGIN);
      setDropdownPosition({ top: rect.top, left: rect.right, minWidth: rect.width, maxHeight });
    } else {
      const maxHeight = Math.max(80, window.innerHeight - rect.bottom - MARGIN);
      setDropdownPosition({ top: rect.bottom, left: rect.right, minWidth: rect.width, maxHeight });
    }
  }, [openDirection]);

  useEffect(() => {
    if (!showDropdown) return;
    computeDropdownPosition();
    window.addEventListener('scroll', computeDropdownPosition, true);
    window.addEventListener('resize', computeDropdownPosition);
    return () => {
      window.removeEventListener('scroll', computeDropdownPosition, true);
      window.removeEventListener('resize', computeDropdownPosition);
    };
  }, [showDropdown, computeDropdownPosition]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(target) &&
        !menuRef.current?.contains(target)
      ) {
        setShowDropdown(false);
        setExpandedGroupId(null);
        setSearchQuery('');
      }
    }

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showDropdown]);

  useEffect(() => {
    if (!showDropdown) {
      setSearchQuery('');
      return;
    }
    const handle = setTimeout(() => searchInputRef.current?.focus(), 0);
    return () => clearTimeout(handle);
  }, [showDropdown]);

  // Build grouped structure
  const { groups, ungroupedTools } = useMemo(() => {
    const groupMap = new Map<string, { name: string; tools: ToolSelectorTool[] }>();
    const ungrouped: ToolSelectorTool[] = [];

    // Initialise groups from toolGroups prop if available
    if (toolGroups) {
      for (const g of toolGroups) {
        groupMap.set(g.id, { name: g.name, tools: [] });
      }
    }

    for (const tool of availableTools) {
      if (tool.group_id) {
        let entry = groupMap.get(tool.group_id);
        if (!entry) {
          entry = { name: tool.group_name || 'Group', tools: [] };
          groupMap.set(tool.group_id, entry);
        }
        entry.tools.push(tool);
      } else {
        ungrouped.push(tool);
      }
    }

    // Remove groups with 0 tools
    const groupList = Array.from(groupMap.entries())
      .filter(([, v]) => v.tools.length > 0)
      .map(([id, v]) => ({ id, name: v.name, tools: v.tools }));

    return { groups: groupList, ungroupedTools: ungrouped };
  }, [availableTools, toolGroups]);

  const hasGroups = groups.length > 0;
  const builtInSelectedIds = selectedBuiltInToolIds ?? new Set<string>();
  const builtInSelectedCount = builtInTools.filter((tool) =>
    builtInSelectedIds.has(tool.id),
  ).length;
  const totalToolCount = getSelectableUserSpaceToolIds(availableTools).length + builtInTools.length;
  const selection = useMemo<UserSpaceToolSelection>(
    () => ({
      mode: normalizeToolSelectionMode(toolSelectionMode),
      toolIds: Array.from(selectedToolIds),
      toolGroupIds: Array.from(selectedToolGroupIds ?? new Set<string>()),
    }),
    [selectedToolIds, selectedToolGroupIds, toolSelectionMode],
  );
  const effectiveToolIds = useMemo(
    () => getEffectiveUserSpaceToolIdSet(selection, availableTools),
    [selection, availableTools],
  );

  // Effective selected count: direct + group-expanded
  const effectiveSelectedCount = useMemo(() => {
    const visibleSelectedCount = availableTools.filter(
      (tool) => isUserSpaceToolAvailable(tool) && effectiveToolIds.has(tool.id),
    ).length;
    return visibleSelectedCount + builtInSelectedCount;
  }, [availableTools, builtInSelectedCount, effectiveToolIds]);

  const searchNeedle = searchQuery.trim().toLowerCase();
  const isSearching = searchNeedle.length > 0;
  const toolMatchesSearch = useCallback(
    (tool: ToolSelectorTool) => {
      if (!searchNeedle) return true;
      return [tool.name, tool.tool_type, tool.description || '', tool.group_name || '']
        .join(' ')
        .toLowerCase()
        .includes(searchNeedle);
    },
    [searchNeedle],
  );
  const filteredBuiltInTools = useMemo(
    () => builtInTools.filter(toolMatchesSearch),
    [builtInTools, toolMatchesSearch],
  );
  const filteredWorkspaceBuiltInTools = useMemo(
    () => filteredBuiltInTools.filter((tool) => tool.tool_type === 'workspace-built-in'),
    [filteredBuiltInTools],
  );
  const filteredChatBuiltInTools = useMemo(
    () => filteredBuiltInTools.filter((tool) => tool.tool_type !== 'workspace-built-in'),
    [filteredBuiltInTools],
  );
  const filteredGroups = useMemo(() => {
    if (!isSearching) return groups;
    return groups
      .map((group) => {
        const groupMatches = group.name.toLowerCase().includes(searchNeedle);
        const tools = groupMatches ? group.tools : group.tools.filter(toolMatchesSearch);
        return { ...group, tools };
      })
      .filter((group) => group.tools.length > 0);
  }, [groups, isSearching, searchNeedle, toolMatchesSearch]);
  const filteredUngroupedTools = useMemo(
    () => (isSearching ? ungroupedTools.filter(toolMatchesSearch) : ungroupedTools),
    [isSearching, toolMatchesSearch, ungroupedTools],
  );
  const visibleAvailableToolIds = useMemo(
    () => [
      ...filteredGroups.flatMap((group) => group.tools.map((tool) => tool.id)),
      ...filteredUngroupedTools.map((tool) => tool.id),
    ],
    [filteredGroups, filteredUngroupedTools],
  );
  const visibleSelectableToolIds = useMemo(() => {
    const visibleIds = new Set(visibleAvailableToolIds);
    return availableTools
      .filter((tool) => visibleIds.has(tool.id) && isUserSpaceToolAvailable(tool))
      .map((tool) => tool.id);
  }, [availableTools, visibleAvailableToolIds]);
  const visibleBuiltInToolIds = useMemo(
    () => filteredBuiltInTools.map((tool) => tool.id),
    [filteredBuiltInTools],
  );
  const allVisibleCount = visibleSelectableToolIds.length + visibleBuiltInToolIds.length;
  const allVisibleSelected =
    allVisibleCount > 0 &&
    visibleSelectableToolIds.every((toolId) => effectiveToolIds.has(toolId)) &&
    visibleBuiltInToolIds.every((toolId) => builtInSelectedIds.has(toolId));

  // Group checkbox state
  const getGroupCheckState = (
    groupId: string,
    tools: ToolSelectorTool[],
  ): 'all' | 'some' | 'none' => {
    return getUserSpaceGroupCheckState(selection, availableTools, groupId, tools);
  };

  const handleGroupToggle = (groupId: string, tools: ToolSelectorTool[]) => {
    if (readOnly || disabled) return;
    const selectableTools = tools.filter(isUserSpaceToolAvailable);
    if (selectableTools.length === 0) return;
    onSelectionChange(toggleUserSpaceToolGroupSelection(selection, availableTools, groupId, tools));
  };

  const handleBulkToggle = () => {
    if (readOnly || disabled || allVisibleCount === 0) return;
    const nextSelected = !allVisibleSelected;
    onSelectionChange(
      setUserSpaceToolSelectionForTools(
        selection,
        availableTools,
        isSearching ? visibleSelectableToolIds : getSelectableUserSpaceToolIds(availableTools),
        nextSelected,
      ),
    );
    if (onBulkBuiltInToggle) {
      onBulkBuiltInToggle(nextSelected);
    }
  };

  const renderToolItem = (tool: ToolSelectorTool) => {
    const toolAvailable = isUserSpaceToolAvailable(tool);
    const checked = toolAvailable && effectiveToolIds.has(tool.id);
    const reason = tool.disabled_reason || 'No recent heartbeat';
    return (
      <label
        key={tool.id}
        className={`checkbox-label userspace-tool-item ${toolAvailable ? '' : 'userspace-tool-item-disabled'}`}
        title={toolAvailable ? undefined : reason}
      >
        <input
          type="checkbox"
          checked={checked}
          onChange={() => {
            if (toolAvailable)
              onSelectionChange(toggleUserSpaceToolSelection(selection, availableTools, tool.id));
          }}
          disabled={readOnly || disabled || !toolAvailable}
        />
        <span>
          <strong>{tool.name}</strong>
          <small className="userspace-muted">
            {toolAvailable ? tool.tool_type : `${tool.tool_type} - Offline`}
          </small>
        </span>
      </label>
    );
  };

  const renderBuiltInToolItem = (tool: ToolSelectorTool) => (
    <label key={tool.id} className="checkbox-label userspace-tool-item userspace-tool-item-builtin">
      <input
        type="checkbox"
        checked={builtInSelectedIds.has(tool.id)}
        onChange={() => onToggleBuiltInTool?.(tool.id)}
        disabled={readOnly || disabled || !onToggleBuiltInTool}
      />
      <span>{tool.name}</span>
    </label>
  );

  return (
    <div className="userspace-tool-picker-wrap" ref={dropdownRef}>
      <button
        className={`btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn ${showDropdown ? 'active' : ''}`}
        onClick={() => setShowDropdown(!showDropdown)}
        title={`${title} (${effectiveSelectedCount}/${totalToolCount} selected)`}
        disabled={disabled}
      >
        <Settings size={14} />
        <span className="tool-count-badge">{effectiveSelectedCount}</span>
      </button>
      {showDropdown &&
        dropdownPosition &&
        createPortal(
          <div
            ref={menuRef}
            className="userspace-tool-dropdown"
            style={{
              top: openDirection === 'up' ? undefined : dropdownPosition.top,
              bottom:
                openDirection === 'up' ? `calc(100vh - ${dropdownPosition.top}px)` : undefined,
              left: dropdownPosition.left,
              minWidth: dropdownPosition.minWidth,
              maxHeight: dropdownPosition.maxHeight,
              transform: 'translateX(-100%)',
            }}
          >
            <div className="userspace-tool-dropdown-title">
              <h4>{title}</h4>
              <div className="userspace-tool-title-actions">
                {saving && <span className="userspace-muted userspace-tool-saving">Saving...</span>}
                <button
                  type="button"
                  className="userspace-tool-bulk-toggle"
                  onClick={handleBulkToggle}
                  disabled={readOnly || disabled || allVisibleCount === 0}
                >
                  {allVisibleSelected
                    ? isSearching
                      ? 'Deselect visible'
                      : 'Deselect all'
                    : isSearching
                      ? 'Select visible'
                      : 'Select all'}
                </button>
              </div>
            </div>
            {readOnly && <p className="userspace-muted">Read-only access</p>}
            {(availableTools.length > 1 || builtInTools.length > 1) && (
              <div className="model-selector-search userspace-tool-search">
                <input
                  ref={searchInputRef}
                  type="text"
                  className="model-selector-search-input"
                  placeholder="Search tools..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      e.preventDefault();
                      if (searchQuery) {
                        setSearchQuery('');
                      } else {
                        setShowDropdown(false);
                      }
                    }
                  }}
                  aria-label="Filter tools"
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
            <div className="userspace-tool-list model-selector-dropdown-inner">
              {builtInTools.length > 0 && (
                <div className="userspace-tool-builtins">
                  {filteredWorkspaceBuiltInTools.length > 0 && (
                    <>
                      <div className="userspace-tool-section-label">
                        {workspaceBuiltInSectionLabel}
                      </div>
                      {filteredWorkspaceBuiltInTools.map(renderBuiltInToolItem)}
                    </>
                  )}
                  {filteredChatBuiltInTools.length > 0 && (
                    <>
                      <div className="userspace-tool-section-label">Built-in</div>
                      {filteredChatBuiltInTools.map(renderBuiltInToolItem)}
                    </>
                  )}
                </div>
              )}
              {builtInTools.length > 0 && availableTools.length > 0 && (
                <div className="userspace-tool-divider" />
              )}
              {hasGroups &&
                filteredGroups.map((group) => {
                  const checkState = getGroupCheckState(group.id, group.tools);
                  const isExpanded = expandedGroupId === group.id;
                  return (
                    <div key={group.id} className="tool-group-section">
                      <div
                        className={`tool-group-header ${isExpanded ? 'expanded' : ''}`}
                        onClick={() => setExpandedGroupId(isExpanded ? null : group.id)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            setExpandedGroupId(isExpanded ? null : group.id);
                          }
                        }}
                        aria-expanded={isExpanded}
                        aria-label={`Tool group: ${group.name}`}
                      >
                        <input
                          type="checkbox"
                          className="tool-group-checkbox"
                          checked={checkState === 'all'}
                          ref={(el) => {
                            if (el) el.indeterminate = checkState === 'some';
                          }}
                          onChange={() => handleGroupToggle(group.id, group.tools)}
                          disabled={
                            readOnly || disabled || !group.tools.some(isUserSpaceToolAvailable)
                          }
                          onClick={(e) => e.stopPropagation()}
                          aria-label={`Select all tools in ${group.name}`}
                        />
                        <span className="tool-group-name">{group.name}</span>
                        <span className="tool-group-count">{group.tools.length}</span>
                        <ChevronRight
                          size={14}
                          className={`tool-group-chevron ${isExpanded ? 'rotated' : ''}`}
                        />
                      </div>
                      {isExpanded && (
                        <div className="tool-group-submenu">{group.tools.map(renderToolItem)}</div>
                      )}
                    </div>
                  );
                })}
              {filteredUngroupedTools.map(renderToolItem)}
              {filteredBuiltInTools.length === 0 &&
                filteredGroups.length === 0 &&
                filteredUngroupedTools.length === 0 && (
                  <div className="model-selector-empty">No tools match "{searchQuery.trim()}"</div>
                )}
            </div>
            {onToggleToolCalls !== undefined && showToolCalls !== undefined && (
              <div className="userspace-tool-calls-toggle">
                <label className="chat-toggle-control" title="Show/hide tool calls in messages">
                  <span className="chat-toggle-label">Show tool calls</span>
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={showToolCalls}
                      onChange={(e) => onToggleToolCalls(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </label>
              </div>
            )}
          </div>,
          document.body,
        )}
    </div>
  );
}
