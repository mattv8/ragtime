import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { api } from '@/api';
import type { ToolConfig, ToolGroup, HeartbeatStatus, SchemaIndexStats, SchemaIndexJob } from '@/types';
import { TOOL_TYPE_INFO } from '@/types';
import { ToolWizard } from './ToolWizard';
import { Icon, getToolIconType } from './Icon';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { AnimatedCreateButton } from './AnimatedCreateButton';
import { IndexingPill } from './IndexingPill';

// Inline field being edited
type EditingField = 'name' | 'description' | null;

// Heartbeat polling interval (15 seconds)
const HEARTBEAT_INTERVAL = 15000;

interface ToolCardProps {
  tool: ToolConfig;
  heartbeat: HeartbeatStatus | null;
  onEdit: (tool: ToolConfig) => void;
  onDelete: (toolId: string) => void;
  onToggle: (toolId: string, enabled: boolean) => void;
  onTest: (toolId: string) => void;
  testing: boolean;
  onPdmReindex?: (toolId: string, fullReindex: boolean) => void;
  pdmIndexing?: boolean;
  onSchemaReindex?: (toolId: string, fullReindex: boolean) => void;
  schemaIndexing?: boolean;
  activeSchemaJob?: SchemaIndexJob | null;
  schemaStats?: SchemaIndexStats | null;
  onInlineUpdate?: (toolId: string, updates: { name?: string; description?: string }) => Promise<void>;
}

function ToolCard({ tool, heartbeat, onEdit, onDelete, onToggle, onTest, testing, onPdmReindex, pdmIndexing, onSchemaReindex, schemaIndexing, activeSchemaJob, schemaStats, onInlineUpdate }: ToolCardProps) {
  const typeInfo = TOOL_TYPE_INFO[tool.tool_type];

  // Inline editing state
  const [editingField, setEditingField] = useState<EditingField>(null);
  const [editName, setEditName] = useState(tool.name);
  const [editDescription, setEditDescription] = useState(tool.description || '');
  const [saving, setSaving] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const descTextareaRef = useRef<HTMLTextAreaElement>(null);

  // Check if schema indexing is enabled for this tool
  const hasSchemaIndexing = (tool.tool_type === 'postgres' || tool.tool_type === 'mssql' || tool.tool_type === 'mysql') &&
    (tool.connection_config as { schema_index_enabled?: boolean })?.schema_index_enabled === true;

  // Format memory size for display
  const formatMemory = (mb: number): string => {
    if (mb < 1) return `${Math.round(mb * 1024)} KB`;
    if (mb < 1024) return `${mb.toFixed(1)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
  };

  const getConnectionSummary = (): string => {
    const config = tool.connection_config;
    switch (tool.tool_type) {
      case 'postgres':
      case 'mysql':
        if ('host' in config && config.host) {
          const port = 'port' in config ? config.port : (tool.tool_type === 'mysql' ? 3306 : 5432);
          const database = 'database' in config ? config.database : '';
          return `${config.host}:${port}/${database}`;
        }
        return `Container: ${'container' in config ? config.container : 'N/A'}`;
      case 'mssql':
      case 'solidworks_pdm':
        if ('host' in config && config.host) {
          const port = 'port' in config ? config.port : 1433;
          const database = 'database' in config ? config.database : '';
          return `${config.host}:${port}/${database}`;
        }
        return 'MSSQL connection';
      case 'odoo_shell':
        if ('mode' in config && config.mode === 'ssh') {
          if ('ssh_host' in config && 'ssh_user' in config) {
            const port = 'ssh_port' in config ? config.ssh_port : 22;
            return `${config.ssh_user}@${config.ssh_host}:${port}`;
          }
          return 'SSH connection';
        }
        return `Container: ${'container' in config ? config.container : 'N/A'}`;
      case 'ssh_shell':
        if ('host' in config && 'user' in config) {
          const port = 'port' in config ? config.port : 22;
          return `${config.user}@${config.host}:${port}`;
        }
        return 'SSH connection';
      case 'filesystem_indexer':
        if ('paths' in config && Array.isArray(config.paths)) {
          return `${config.paths.length} path(s)`;
        }
        return 'Filesystem indexer';
      default:
        return 'Unknown';
    }
  };

  // Determine heartbeat display status
  const getHeartbeatDisplay = () => {
    if (!tool.enabled) {
      return { status: 'disabled', label: 'Disabled', icon: <Icon name="circle" size={16} /> };
    }
    if (!heartbeat) {
      return { status: 'checking', label: 'Checking...', icon: <Icon name="loader" size={16} /> };
    }
    if (heartbeat.alive) {
      const latency = heartbeat.latency_ms ? `${Math.round(heartbeat.latency_ms)}ms` : '';
      return { status: 'alive', label: latency || 'Connected', icon: <Icon name="check" size={16} /> };
    }
    return { status: 'dead', label: heartbeat.error || 'Disconnected', icon: <Icon name="close" size={16} /> };
  };

  const heartbeatDisplay = getHeartbeatDisplay();

  // Auto-resize textarea to fit content
  const autoResizeTextarea = (textarea: HTMLTextAreaElement | null) => {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;
  };

  // Focus input/textarea when entering edit mode
  useEffect(() => {
    if (editingField === 'name' && nameInputRef.current) {
      nameInputRef.current.focus();
      nameInputRef.current.select();
    } else if (editingField === 'description' && descTextareaRef.current) {
      descTextareaRef.current.focus();
      descTextareaRef.current.select();
      autoResizeTextarea(descTextareaRef.current);
    }
  }, [editingField]);

  // Sync local state when tool prop changes
  useEffect(() => {
    setEditName(tool.name);
    setEditDescription(tool.description || '');
  }, [tool.name, tool.description]);

  const handleStartEdit = (field: EditingField) => {
    setEditingField(field);
    if (field === 'name') {
      setEditName(tool.name);
    } else if (field === 'description') {
      setEditDescription(tool.description || '');
    }
  };

  const handleCancelEdit = () => {
    setEditingField(null);
    setEditName(tool.name);
    setEditDescription(tool.description || '');
  };

  const handleSaveEdit = async () => {
    if (!onInlineUpdate || saving) return;

    const updates: { name?: string; description?: string } = {};

    if (editingField === 'name' && editName.trim() !== tool.name) {
      if (!editName.trim()) {
        // Don't allow empty names
        handleCancelEdit();
        return;
      }
      updates.name = editName.trim();
    } else if (editingField === 'description' && editDescription !== (tool.description || '')) {
      updates.description = editDescription;
    }

    if (Object.keys(updates).length === 0) {
      setEditingField(null);
      return;
    }

    setSaving(true);
    try {
      await onInlineUpdate(tool.id, updates);
      setEditingField(null);
    } catch (err) {
      console.error('Failed to save:', err);
      // Keep edit mode open on error
    } finally {
      setSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      handleCancelEdit();
    } else if (e.key === 'Enter' && (editingField === 'name' || e.ctrlKey)) {
      e.preventDefault();
      handleSaveEdit();
    }
  };

  return (
    <div className={`tool-card ${!tool.enabled ? 'disabled' : ''}`}>
      <div className="tool-card-header">
        <div className="tool-card-icon"><Icon name={getToolIconType(typeInfo?.icon)} size={28} /></div>
        <div className="tool-card-header-content">
          <div className="tool-card-header-main">
            <div className="tool-card-title">
              {editingField === 'name' ? (
                <div className="inline-edit-field">
                  <input
                    ref={nameInputRef}
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onBlur={handleSaveEdit}
                    disabled={saving}
                    className="inline-edit-input"
                  />
                </div>
              ) : (
                <div className="editable-field-wrapper name-wrapper" onClick={() => handleStartEdit('name')}>
                  <h3>{tool.name}</h3>
                  <button
                    type="button"
                    className="inline-edit-btn"
                    onClick={(e) => { e.stopPropagation(); handleStartEdit('name'); }}
                    title="Edit name"
                  >
                    <Icon name="pencil" size={14} />
                  </button>
                </div>
              )}
            </div>
            <div className="tool-card-header-actions">
              <div className="tool-card-heartbeat">
                <span
                  className={`heartbeat-indicator ${heartbeatDisplay.status}`}
                  title={heartbeatDisplay.label}
                >
                  {heartbeatDisplay.icon}
                </span>
              </div>
              <div className="tool-card-status">
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={tool.enabled}
                    onChange={(e) => onToggle(tool.id, e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </div>
          </div>
          <code className="tool-card-connection-sub">{getConnectionSummary()}</code>
        </div>
      </div>

      {editingField === 'description' ? (
        <div className="inline-edit-field description-edit">
          <textarea
            ref={descTextareaRef}
            value={editDescription}
            onChange={(e) => {
              setEditDescription(e.target.value);
              autoResizeTextarea(e.target);
            }}
            onKeyDown={handleKeyDown}
            onBlur={handleSaveEdit}
            disabled={saving}
            className="inline-edit-textarea"
            rows={1}
            placeholder="Description for AI..."
          />
        </div>
      ) : (
        <div
          className="editable-field-wrapper description-wrapper"
          onClick={() => handleStartEdit('description')}
        >
          {tool.description ? (
            <p className="tool-card-description">{tool.description}</p>
          ) : (
            <p className="tool-card-description placeholder">Add description for AI...</p>
          )}
          <button
            type="button"
            className="inline-edit-btn"
            onClick={(e) => { e.stopPropagation(); handleStartEdit('description'); }}
            title="Edit description"
          >
            <Icon name="pencil" size={14} />
          </button>
        </div>
      )}

      {/* Show heartbeat error if connection failed */}
      {heartbeat && !heartbeat.alive && tool.enabled && (
        <div className="tool-card-heartbeat-error">
          <span className="error-icon">
            <Icon name="alert-circle" size={16} />
          </span>
          <span>{heartbeat.error || 'Connection failed'}</span>
        </div>
      )}

      {((tool.allow_write) || ((tool.tool_type === 'ssh_shell' || tool.tool_type === 'odoo_shell') && (tool.connection_config as any)?.working_directory) || activeSchemaJob) && (
        <div className="tool-card-constraints">
          <IndexingPill
            activeJob={activeSchemaJob}
            progressLabelPrefix="Indexing"
          />
          {tool.allow_write && (
            <span className="write-enabled-pill">
              <Icon name="alert-triangle" size={12} />
              Write enabled
            </span>
          )}
          {((tool.tool_type === 'ssh_shell' || tool.tool_type === 'odoo_shell') && (tool.connection_config as any)?.working_directory) && (
            <span className="constrained-path" title={`Constrained to ${(tool.connection_config as any).working_directory}`}>
              <Icon name="folder" size={14} />
              <span className="path-text">{(tool.connection_config as any).working_directory}</span>
            </span>
          )}
        </div>
      )}

      {/* Schema index stats */}
      {hasSchemaIndexing && schemaStats && schemaStats.embedding_count > 0 && (
        <div className="tool-card-schema-stats">
          <span className="schema-stats-label">Schema Index:</span>
          <span className="schema-stats-value">
            {schemaStats.embedding_count} tables
            {schemaStats.estimated_memory_mb != null && (
              <span className="schema-stats-memory" title="Estimated pgvector storage size">
                ({formatMemory(schemaStats.estimated_memory_mb)})
              </span>
            )}
          </span>
        </div>
      )}

      <div className="tool-card-actions">
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onTest(tool.id)}
          disabled={testing}
        >
          {testing ? 'Testing...' : 'Test'}
        </button>
        {tool.tool_type === 'solidworks_pdm' && onPdmReindex && (
          <>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onPdmReindex(tool.id, false)}
              disabled={pdmIndexing}
              title="Index new and changed documents"
            >
              {pdmIndexing ? 'Indexing...' : 'Index'}
            </button>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onPdmReindex(tool.id, true)}
              disabled={pdmIndexing}
              title="Re-index all documents from scratch"
            >
              Full Re-index
            </button>
          </>
        )}
        {hasSchemaIndexing && onSchemaReindex && (
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => onSchemaReindex(tool.id, true)}
            disabled={schemaIndexing}
            title="Re-index database schema"
          >
            {schemaIndexing ? 'Indexing...' : 'Re-index Schema'}
          </button>
        )}
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onEdit(tool)}
        >
          Edit
        </button>
        <DeleteConfirmButton
          onDelete={() => onDelete(tool.id)}
          className="btn btn-sm btn-danger"
          title="Delete tool"
        />
      </div>
    </div>
  );
}

interface ToolsPanelProps {
  onSchemaJobTriggered?: () => void;
  schemaJobs?: SchemaIndexJob[];
}

export function ToolsPanel({ onSchemaJobTriggered, schemaJobs = [] }: ToolsPanelProps) {
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showWizard, setShowWizard] = useState(false);
  const [editingTool, setEditingTool] = useState<ToolConfig | null>(null);
  const [testingToolId, setTestingToolId] = useState<string | null>(null);
  const [pdmIndexingToolId, setPdmIndexingToolId] = useState<string | null>(null);
  const [schemaIndexingToolId, setSchemaIndexingToolId] = useState<string | null>(null);
  const [heartbeats, setHeartbeats] = useState<Record<string, HeartbeatStatus>>({});
  const [schemaStats, setSchemaStats] = useState<Record<string, SchemaIndexStats>>({});

  const [editingGroupId, setEditingGroupId] = useState<string | null>(null);
  const [editingGroupName, setEditingGroupName] = useState('');
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadTools = useCallback(async () => {
    try {
      setLoading(true);
      const [data, groupData] = await Promise.all([
        api.listToolConfigs(),
        api.listToolGroups(),
      ]);
      // Filter out filesystem_indexer tools - they're shown in the Indexer tab
      const connectionTools = data.filter(t => t.tool_type !== 'filesystem_indexer');
      setTools(connectionTools);
      setGroups(groupData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tools');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load schema stats for tools with schema indexing enabled
  const loadSchemaStats = useCallback(async (toolList: ToolConfig[]) => {
    const schemaTools = toolList.filter(t =>
      (t.tool_type === 'postgres' || t.tool_type === 'mssql' || t.tool_type === 'mysql') &&
      (t.connection_config as { schema_index_enabled?: boolean })?.schema_index_enabled === true
    );

    if (schemaTools.length === 0) return;

    const statsMap: Record<string, SchemaIndexStats> = {};
    await Promise.all(
      schemaTools.map(async (tool) => {
        try {
          const stats = await api.getSchemaIndexStats(tool.id);
          statsMap[tool.id] = stats;
        } catch (err) {
          console.warn(`Failed to load schema stats for ${tool.name}:`, err);
        }
      })
    );
    setSchemaStats(statsMap);
  }, []);

  // Fetch heartbeat status for all enabled tools
  const fetchHeartbeats = useCallback(async () => {
    try {
      const response = await api.getToolHeartbeats();
      setHeartbeats(response.statuses);
    } catch (err) {
      // Silently fail on heartbeat errors - don't disrupt the UI
      console.warn('Heartbeat check failed:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadTools();
  }, [loadTools]);

  // Load schema stats when tools change
  useEffect(() => {
    if (tools.length > 0) {
      loadSchemaStats(tools);
    }
  }, [tools, loadSchemaStats]);

  // Heartbeat polling
  useEffect(() => {
    // Initial heartbeat check after tools load
    if (!loading && tools.length > 0) {
      fetchHeartbeats();
    }

    // Set up interval for periodic heartbeat checks
    heartbeatTimerRef.current = setInterval(() => {
      if (!showWizard) {
        fetchHeartbeats();
      }
    }, HEARTBEAT_INTERVAL);

    return () => {
      if (heartbeatTimerRef.current) {
        clearInterval(heartbeatTimerRef.current);
      }
    };
  }, [loading, tools.length, showWizard, fetchHeartbeats]);

  const handleAddTool = () => {
    setEditingTool(null);
    setShowWizard(true);
  };

  const handleEditTool = (tool: ToolConfig) => {
    setEditingTool(tool);
    setShowWizard(true);
  };

  const handleWizardClose = () => {
    setShowWizard(false);
    setEditingTool(null);
  };

  const handleWizardSave = async () => {
    setShowWizard(false);
    setEditingTool(null);
    await loadTools();
    setSuccess('Tool configuration saved successfully');
    setTimeout(() => setSuccess(null), 3000);
  };

  const handleDeleteTool = async (toolId: string) => {
    try {
      await api.deleteToolConfig(toolId);
      await loadTools();
      setSuccess('Tool deleted successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete tool');
    }
  };

  const handleToggleTool = async (toolId: string, enabled: boolean) => {
    try {
      await api.toggleToolConfig(toolId, enabled);
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle tool');
    }
  };

  const handleTestTool = async (toolId: string) => {
    try {
      setTestingToolId(toolId);
      const result = await api.testSavedToolConnection(toolId);
      await loadTools();
      if (result.success) {
        setSuccess('Connection test successful');
      } else {
        setError(`Connection test failed: ${result.message}`);
      }
      setTimeout(() => {
        setSuccess(null);
        setError(null);
      }, 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setTestingToolId(null);
    }
  };

  const handlePdmReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setPdmIndexingToolId(toolId);
      setSuccess(fullReindex ? 'Starting full PDM re-index...' : 'Starting PDM index update...');
      await api.triggerPdmIndex(toolId, fullReindex);
      setSuccess(fullReindex ? 'Full PDM re-index started' : 'PDM index update started');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger PDM index');
      setTimeout(() => setError(null), 5000);
    } finally {
      setPdmIndexingToolId(null);
    }
  };

  const handleSchemaReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setSchemaIndexingToolId(toolId);
      setSuccess('Starting schema re-index...');
      await api.triggerSchemaIndex(toolId, fullReindex);
      setSuccess('Schema re-index started');
      // Notify parent to refresh schema jobs list
      onSchemaJobTriggered?.();
      // Refresh schema stats after a short delay to allow job to start
      setTimeout(() => loadSchemaStats(tools), 2000);
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger schema index');
      setTimeout(() => setError(null), 5000);
    } finally {
      setSchemaIndexingToolId(null);
    }
  };

  const handleInlineUpdate = async (toolId: string, updates: { name?: string; description?: string }) => {
    try {
      await api.updateToolConfig(toolId, updates);
      await loadTools();
      if (updates.name) {
        setSuccess('Tool name updated');
      } else {
        setSuccess('Description updated');
      }
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update');
      setTimeout(() => setError(null), 3000);
      throw err; // Re-throw to let ToolCard know the save failed
    }
  };

  // ---- Tool Group handlers ----

  const handleCreateGroup = async () => {
    try {
      const created = await api.createToolGroup({ name: 'Untitled Group' });
      await loadTools();
      if (created?.id) {
        setEditingGroupId(created.id);
        setEditingGroupName('Untitled Group');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create group');
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleRenameGroup = async (groupId: string) => {
    const name = editingGroupName.trim();
    if (!name) return;
    try {
      await api.updateToolGroup(groupId, { name });
      setEditingGroupId(null);
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename group');
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleDeleteGroup = async (groupId: string) => {
    try {
      await api.deleteToolGroup(groupId);
      await loadTools();
      setSuccess('Group deleted');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete group');
      setTimeout(() => setError(null), 3000);
    }
  };

  const handleAssignGroup = async (toolId: string, groupId: string | null) => {
    try {
      await api.updateToolConfig(toolId, { group_id: groupId ?? '' });
      await loadTools();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to assign group');
      setTimeout(() => setError(null), 3000);
    }
  };

  // Group the tools for display — include ALL groups (even empty) as drop targets
  const { allGroups, ungroupedTools } = useMemo(() => {
    const grouped = new Map<string, { group: ToolGroup; tools: ToolConfig[] }>();
    const ungrouped: ToolConfig[] = [];

    for (const g of groups) {
      grouped.set(g.id, { group: g, tools: [] });
    }

    for (const tool of tools) {
      if (tool.group_id && grouped.has(tool.group_id)) {
        grouped.get(tool.group_id)!.tools.push(tool);
      } else {
        ungrouped.push(tool);
      }
    }

    return {
      allGroups: Array.from(grouped.values()),
      ungroupedTools: ungrouped,
    };
  }, [tools, groups]);

  // Selected group tab (null = show ungrouped / all)
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const groupContentRef = useRef<HTMLDivElement>(null);

  // Click outside the group content area clears the selected group
  useEffect(() => {
    if (!selectedGroupId) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (groupContentRef.current && !groupContentRef.current.contains(e.target as Node)) {
        setSelectedGroupId(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [selectedGroupId]);

  // Drag-and-drop state
  const [dragToolId, setDragToolId] = useState<string | null>(null);
  const [dragOverGroupId, setDragOverGroupId] = useState<string | null>(null);
  const [dragOverUngrouped, setDragOverUngrouped] = useState(false);

  const handleDragStart = useCallback((e: React.DragEvent, toolId: string) => {
    e.dataTransfer.setData('text/plain', toolId);
    e.dataTransfer.effectAllowed = 'move';
    setDragToolId(toolId);
  }, []);

  const handleDragEnd = useCallback(() => {
    setDragToolId(null);
    setDragOverGroupId(null);
    setDragOverUngrouped(false);
  }, []);

  const handleGroupDragOver = useCallback((e: React.DragEvent, groupId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverGroupId(groupId);
  }, []);

  const handleGroupDragLeave = useCallback((e: React.DragEvent, groupId: string) => {
    // Only clear if we actually left the group card (not entering a child element)
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setDragOverGroupId((current) => current === groupId ? null : current);
    }
  }, []);

  const handleGroupDrop = useCallback(async (e: React.DragEvent, groupId: string) => {
    e.preventDefault();
    const toolId = e.dataTransfer.getData('text/plain');
    setDragOverGroupId(null);
    setDragToolId(null);
    if (toolId) {
      await handleAssignGroup(toolId, groupId);
    }
  }, [handleAssignGroup]);

  const handleUngroupedDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverUngrouped(true);
  }, []);

  const handleUngroupedDragLeave = useCallback((e: React.DragEvent) => {
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) {
      setDragOverUngrouped(false);
    }
  }, []);

  const handleUngroupedDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const toolId = e.dataTransfer.getData('text/plain');
    setDragOverUngrouped(false);
    setDragToolId(null);
    if (toolId) {
      await handleAssignGroup(toolId, null);
    }
  }, [handleAssignGroup]);

  return (
    <div className="tools-panel">
      <div className="card">
        <div className="card-header">
          <h2>Tool Connections</h2>
          <AnimatedCreateButton
            isExpanded={showWizard}
            onClick={() => showWizard ? handleWizardClose() : handleAddTool()}
            label="Add Tool"
          />
        </div>

        {showWizard ? (
          <ToolWizard
            key={editingTool?.id ?? 'new'}
            existingTool={editingTool}
            onClose={handleWizardClose}
            onSave={handleWizardSave}
            embedded={true}
          />
        ) : (
          <>
        {error && <div className="error-banner">{error}</div>}
        {success && <div className="success-banner">{success}</div>}

        <p className="fieldset-help">
          Configure connections to databases, shells, and other tools that the AI can use during conversations.
          Each tool can have multiple instances (e.g., production and staging databases). To group a tool, create a group and drag the tool into it.
          Deleting a group does not delete the tools.
        </p>

        {loading ? (
          <p className="muted">Loading tools...</p>
        ) : tools.length === 0 && groups.length === 0 ? (
          <div className="empty-state">
            <p>No tools configured yet.</p>
            <p className="muted">
              Click "Add Tool" to set up your first connection.
            </p>
          </div>
        ) : (
          <>
            {/* Group tabs + tool grid */}
            <div ref={groupContentRef} className={`tool-group-content${selectedGroupId ? ' has-selection' : ''}`}>
            <div className={`tool-group-tabs${dragToolId ? ' dragging' : ''}`}>
              {allGroups.length > 0 && allGroups.map(({ group, tools: groupTools }) => {
                const isActive = selectedGroupId === group.id;
                const isDragTarget = dragOverGroupId === group.id;
                return (
                  <div
                    key={group.id}
                    className={`tool-group-tab${isActive ? ' active' : ''}${isDragTarget ? ' drag-over' : ''}`}
                    onClick={() => setSelectedGroupId(isActive ? null : group.id)}
                    onDragOver={(e) => handleGroupDragOver(e, group.id)}
                    onDragLeave={(e) => handleGroupDragLeave(e, group.id)}
                    onDrop={(e) => handleGroupDrop(e, group.id)}
                  >
                      {editingGroupId === group.id ? (
                        <input
                          className="tool-group-tab-input"
                          value={editingGroupName}
                          onChange={(e) => setEditingGroupName(e.target.value)}
                          onBlur={() => handleRenameGroup(group.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRenameGroup(group.id);
                            if (e.key === 'Escape') setEditingGroupId(null);
                          }}
                          onClick={(e) => e.stopPropagation()}
                          autoFocus
                        />
                      ) : (
                        <span className="tool-group-tab-name">{group.name}</span>
                      )}
                      <span className="tool-group-tab-count">{groupTools.length}</span>
                      <span
                        className="tool-group-tab-rename"
                        onClick={(e) => { e.stopPropagation(); setEditingGroupId(group.id); setEditingGroupName(group.name); }}
                        title="Rename group"
                      >
                        &#9998;
                      </span>
                      <DeleteConfirmButton
                        onDelete={() => { handleDeleteGroup(group.id); if (selectedGroupId === group.id) setSelectedGroupId(null); }}
                        className="tool-group-tab-delete"
                        title="Delete group"
                      />
                  </div>
                );
              })}
              <button
                className="tool-group-tab tool-group-tab-add"
                onClick={handleCreateGroup}
                title="Create a new group"
              >
                + Add Group
              </button>
              {/* "Ungrouped" drop target — visible when dragging from within a group */}
              {dragToolId && selectedGroupId && (
                <div
                  className={`tool-group-tab tool-group-tab-ungrouped${dragOverUngrouped ? ' drag-over' : ''}`}
                  onDragOver={handleUngroupedDragOver}
                  onDragLeave={handleUngroupedDragLeave}
                  onDrop={handleUngroupedDrop}
                >
                  Ungrouped
                </div>
              )}
            </div>

            {/* Tool cards — filtered by selected group */}
            {(() => {
              const selectedGroup = selectedGroupId
                ? allGroups.find(g => g.group.id === selectedGroupId)
                : null;
              const visibleTools = selectedGroup
                ? selectedGroup.tools
                : ungroupedTools;
              const isUngroupedView = !selectedGroupId;

              if (visibleTools.length === 0 && selectedGroupId) {
                return (
                  <div className="tool-group-active-panel">
                    <p className="muted" style={{ textAlign: 'center', margin: 0 }}>No tools in this group yet. Drag tools here to add them.</p>
                  </div>
                );
              }

              return (
                <>
                  {selectedGroupId && selectedGroup ? (
                    <div className="tool-group-active-panel">
                      <div className="tools-grid">
                        {visibleTools.map((tool) => (
                          <div
                            key={tool.id}
                            draggable
                            onDragStart={(e) => handleDragStart(e, tool.id)}
                            onDragEnd={handleDragEnd}
                            className={`tool-card-drag-wrap${dragToolId === tool.id ? ' dragging' : ''}`}
                          >
                            <ToolCard
                              tool={tool}
                              heartbeat={heartbeats[tool.id] || null}
                              onEdit={handleEditTool}
                              onDelete={handleDeleteTool}
                              onToggle={handleToggleTool}
                              onTest={handleTestTool}
                              testing={testingToolId === tool.id}
                              onPdmReindex={handlePdmReindex}
                              pdmIndexing={pdmIndexingToolId === tool.id}
                              onSchemaReindex={handleSchemaReindex}
                              schemaIndexing={schemaIndexingToolId === tool.id}
                              activeSchemaJob={schemaJobs.find(j => j.tool_config_id === tool.id && (j.status === 'pending' || j.status === 'indexing'))}
                              schemaStats={schemaStats[tool.id] || null}
                              onInlineUpdate={handleInlineUpdate}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                  <div
                    className={`tool-group-section-panel${isUngroupedView && dragToolId ? ' tool-ungrouped-drop-zone' : ''}${isUngroupedView && dragOverUngrouped ? ' drag-over' : ''}`}
                    onDragOver={isUngroupedView && dragToolId ? handleUngroupedDragOver : undefined}
                    onDragLeave={isUngroupedView && dragToolId ? handleUngroupedDragLeave : undefined}
                    onDrop={isUngroupedView && dragToolId ? handleUngroupedDrop : undefined}
                  >
                  <div className="tools-grid">
                    {visibleTools.map((tool) => (
                      <div
                        key={tool.id}
                        draggable
                        onDragStart={(e) => handleDragStart(e, tool.id)}
                        onDragEnd={handleDragEnd}
                        className={`tool-card-drag-wrap${dragToolId === tool.id ? ' dragging' : ''}`}
                      >
                        <ToolCard
                          tool={tool}
                          heartbeat={heartbeats[tool.id] || null}
                          onEdit={handleEditTool}
                          onDelete={handleDeleteTool}
                          onToggle={handleToggleTool}
                          onTest={handleTestTool}
                          testing={testingToolId === tool.id}
                          onPdmReindex={handlePdmReindex}
                          pdmIndexing={pdmIndexingToolId === tool.id}
                          onSchemaReindex={handleSchemaReindex}
                          schemaIndexing={schemaIndexingToolId === tool.id}
                          activeSchemaJob={schemaJobs.find(j => j.tool_config_id === tool.id && (j.status === 'pending' || j.status === 'indexing'))}
                          schemaStats={schemaStats[tool.id] || null}
                          onInlineUpdate={handleInlineUpdate}
                        />
                      </div>
                    ))}
                  </div>
                </div>
                  )}
                </>
              );
            })()}
            </div>
          </>
        )}
        </>
        )}
      </div>

      {!showWizard && tools.length === 0 && (
        <div className="card">
          <h2>About Tools</h2>
          <p className="fieldset-help">
            Tools give the AI assistant the ability to query your systems directly during conversations.
            When you ask a question, the AI can use these tools to fetch real-time data.
          </p>

          <div className="tool-types-info">
            {Object.entries(TOOL_TYPE_INFO).map(([type, info]) => (
              <div key={type} className="tool-type-info">
                <span className="tool-type-icon">
                  <Icon name={getToolIconType(info.icon)} size={20} />
                </span>
                <div>
                  <strong>{info.name}</strong>
                  <p>{info.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
