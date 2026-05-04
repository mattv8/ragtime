import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Settings, ChevronRight } from 'lucide-react';

interface ToolSelectorTool {
  id: string;
  name: string;
  tool_type: string;
  description?: string | null;
  group_id?: string | null;
  group_name?: string | null;
}

export interface ToolGroupInfo {
  id: string;
  name: string;
}

interface ToolSelectorDropdownProps {
  availableTools: ToolSelectorTool[];
  selectedToolIds: Set<string>;
  onToggleTool: (toolId: string) => void;
  builtInTools?: ToolSelectorTool[];
  selectedBuiltInToolIds?: Set<string>;
  onToggleBuiltInTool?: (toolId: string) => void;
  /** Selected tool group IDs. When a group is selected, all its tools are effectively enabled. */
  selectedToolGroupIds?: Set<string>;
  onToggleToolGroup?: (groupId: string) => void;
  toolGroups?: ToolGroupInfo[];
  /** Which direction the dropdown opens. 'down' (default) for toolbars, 'up' for inline chat input. */
  openDirection?: 'down' | 'up';
  disabled?: boolean;
  readOnly?: boolean;
  saving?: boolean;
  title?: string;
  /** When provided, renders a "Show tool calls" toggle at the bottom of the dropdown. */
  showToolCalls?: boolean;
  onToggleToolCalls?: (value: boolean) => void;
}

export function ToolSelectorDropdown({
  availableTools,
  selectedToolIds,
  onToggleTool,
  builtInTools = [],
  selectedBuiltInToolIds,
  onToggleBuiltInTool,
  selectedToolGroupIds,
  onToggleToolGroup,
  toolGroups,
  openDirection = 'down',
  disabled = false,
  readOnly = false,
  saving = false,
  title = 'Tools',
  showToolCalls,
  onToggleToolCalls,
}: ToolSelectorDropdownProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null);
  const [dropdownPosition, setDropdownPosition] = useState<{ top: number; left: number; minWidth: number } | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Compute fixed position so the dropdown draws over iframes without layout shift
  const computeDropdownPosition = useCallback(() => {
    if (!dropdownRef.current) return;
    const rect = dropdownRef.current.getBoundingClientRect();
    if (openDirection === 'up') {
      // Position above the trigger button; actual height unknown until rendered, use 0 as placeholder
      setDropdownPosition({ top: rect.top, left: rect.right, minWidth: rect.width });
    } else {
      setDropdownPosition({ top: rect.bottom, left: rect.right, minWidth: rect.width });
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
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
        setExpandedGroupId(null);
      }
    }

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
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
  const builtInSelectedCount = builtInTools.filter((tool) => builtInSelectedIds.has(tool.id)).length;
  const totalToolCount = availableTools.length + builtInTools.length;

  // Effective selected count: direct + group-expanded
  const effectiveSelectedCount = useMemo(() => {
    const ids = new Set(selectedToolIds);
    if (selectedToolGroupIds) {
      for (const tool of availableTools) {
        if (tool.group_id && selectedToolGroupIds.has(tool.group_id)) {
          ids.add(tool.id);
        }
      }
    }
    return ids.size + builtInSelectedCount;
  }, [selectedToolIds, selectedToolGroupIds, availableTools, builtInSelectedCount]);

  // Group checkbox state
  const getGroupCheckState = (groupId: string, tools: ToolSelectorTool[]): 'all' | 'some' | 'none' => {
    if (selectedToolGroupIds?.has(groupId)) return 'all';
    const selected = tools.filter((t) => selectedToolIds.has(t.id)).length;
    if (selected === 0) return 'none';
    if (selected === tools.length) return 'all';
    return 'some';
  };

  const handleGroupToggle = (groupId: string, tools: ToolSelectorTool[]) => {
    if (readOnly || saving || disabled) return;
    if (onToggleToolGroup) {
      // Use group-level selection
      onToggleToolGroup(groupId);
    } else {
      // Fallback: toggle all individual tools in the group
      const state = getGroupCheckState(groupId, tools);
      if (state === 'all') {
        // Deselect all tools in this group
        for (const t of tools) {
          if (selectedToolIds.has(t.id)) onToggleTool(t.id);
        }
      } else {
        // Select all tools in this group
        for (const t of tools) {
          if (!selectedToolIds.has(t.id)) onToggleTool(t.id);
        }
      }
    }
  };

  const renderToolItem = (tool: ToolSelectorTool) => (
    <label key={tool.id} className="checkbox-label userspace-tool-item">
      <input
        type="checkbox"
        checked={
          selectedToolIds.has(tool.id) ||
          !!(tool.group_id && selectedToolGroupIds?.has(tool.group_id))
        }
        onChange={() => onToggleTool(tool.id)}
        disabled={saving || readOnly || disabled}
      />
      <span>
        <strong>{tool.name}</strong>
        <small className="userspace-muted">{tool.tool_type}</small>
      </span>
    </label>
  );

  const renderBuiltInToolItem = (tool: ToolSelectorTool) => (
    <label key={tool.id} className="checkbox-label userspace-tool-item userspace-tool-item-builtin">
      <input
        type="checkbox"
        checked={builtInSelectedIds.has(tool.id)}
        onChange={() => onToggleBuiltInTool?.(tool.id)}
        disabled={saving || readOnly || disabled || !onToggleBuiltInTool}
      />
      <span>
        <strong>{tool.name}</strong>
        <small className="userspace-muted">Built-in</small>
      </span>
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
      {showDropdown && dropdownPosition && (
        <div
          className="userspace-tool-dropdown"
          style={{
            top: openDirection === 'up' ? undefined : dropdownPosition.top,
            bottom: openDirection === 'up' ? `calc(100vh - ${dropdownPosition.top}px)` : undefined,
            left: dropdownPosition.left,
            minWidth: dropdownPosition.minWidth,
            transform: 'translateX(-100%)',
          }}
        >
          <h4>{title}</h4>
          {readOnly && <p className="userspace-muted">Read-only access</p>}
          <div className="userspace-tool-list">
            {builtInTools.length > 0 && (
              <div className="userspace-tool-builtins">
                <div className="userspace-tool-section-label">Built-in</div>
                {builtInTools.map(renderBuiltInToolItem)}
              </div>
            )}
            {builtInTools.length > 0 && availableTools.length > 0 && (
              <div className="userspace-tool-divider" />
            )}
            {hasGroups && groups.map((group) => {
              const checkState = getGroupCheckState(group.id, group.tools);
              const isExpanded = expandedGroupId === group.id;
              return (
                <div key={group.id} className="tool-group-section">
                  <div
                    className={`tool-group-header ${isExpanded ? 'expanded' : ''}`}
                    onMouseEnter={() => setExpandedGroupId(group.id)}
                    onMouseLeave={() => setExpandedGroupId(null)}
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
                      disabled={saving || readOnly || disabled}
                      onClick={(e) => e.stopPropagation()}
                      aria-label={`Select all tools in ${group.name}`}
                    />
                    <span className="tool-group-name">{group.name}</span>
                    <span className="tool-group-count">{group.tools.length}</span>
                    <ChevronRight size={14} className={`tool-group-chevron ${isExpanded ? 'rotated' : ''}`} />
                  </div>
                  {isExpanded && (
                    <div
                      className="tool-group-submenu"
                      onMouseEnter={() => setExpandedGroupId(group.id)}
                      onMouseLeave={() => setExpandedGroupId(null)}
                    >
                      {group.tools.map(renderToolItem)}
                    </div>
                  )}
                </div>
              );
            })}
            {ungroupedTools.map(renderToolItem)}
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
        </div>
      )}
    </div>
  );
}
