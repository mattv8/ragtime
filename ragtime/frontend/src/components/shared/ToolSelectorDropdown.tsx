import { useEffect, useRef, useState } from 'react';
import { Settings } from 'lucide-react';

interface ToolSelectorDropdownProps {
  availableTools: Array<{
    id: string;
    name: string;
    tool_type: string;
    description?: string | null;
  }>;
  selectedToolIds: Set<string>;
  onToggleTool: (toolId: string) => void;
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
  disabled = false,
  readOnly = false,
  saving = false,
  title = 'Tools',
  showToolCalls,
  onToggleToolCalls,
}: ToolSelectorDropdownProps) {
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    }

    if (showDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }
  }, [showDropdown]);

  return (
    <div className="userspace-tool-picker-wrap" ref={dropdownRef}>
      <button
        className={`btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn ${showDropdown ? 'active' : ''}`}
        onClick={() => setShowDropdown(!showDropdown)}
        title={`${title} (${selectedToolIds.size}/${availableTools.length} selected)`}
        disabled={disabled}
      >
        <Settings size={14} />
        <span className="tool-count-badge">{selectedToolIds.size}</span>
      </button>
      {showDropdown && (
        <div className="userspace-tool-dropdown">
          <h4>{title}</h4>
          {readOnly && <p className="userspace-muted">Read-only access</p>}
          <div className="userspace-tool-list">
            {availableTools.map((tool) => (
              <label key={tool.id} className="checkbox-label userspace-tool-item">
                <input
                  type="checkbox"
                  checked={selectedToolIds.has(tool.id)}
                  onChange={() => onToggleTool(tool.id)}
                  disabled={saving || readOnly || disabled}
                />
                <span>
                  <strong>{tool.name}</strong>
                  <small className="userspace-muted">{tool.tool_type}</small>
                </span>
              </label>
            ))}
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
