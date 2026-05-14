import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { api } from '@/api';
import type { ToolConfig, ToolGroup, HeartbeatStatus, SchemaIndexStats, SchemaIndexJob, UserspaceMountSource, MountSourceAffectedWorkspacesResponse } from '@/types';
import { TOOL_TYPE_INFO } from '@/types';
import { ToolWizard } from './ToolWizard';
import { MountSourceWizard } from './MountSourceWizard';
import { Icon, getToolIconType } from './Icon';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { AnimatedCreateButton } from './AnimatedCreateButton';
import { IndexingPill } from './IndexingPill';
import { useToast, ToastContainer } from './shared/Toast';
import { HardDrive, Trash2, Pencil, X } from 'lucide-react';
import { resolveSourceDisplayPath } from '@/utils/mountPaths';

// Inline field being edited
type EditingField = 'name' | 'description' | null;

// Heartbeat polling interval (15 seconds)
const HEARTBEAT_INTERVAL = 15000;
const DRAG_REORDER_PREVIEW_DELAY_MS = 80;
type DragPreviewTarget = { toolId: string; insertBefore: boolean };

function hasSchemaIndexingEnabled(tool: ToolConfig): boolean {
  return (tool.tool_type === 'postgres' || tool.tool_type === 'mssql' || tool.tool_type === 'mysql') &&
    (tool.connection_config as { schema_index_enabled?: boolean })?.schema_index_enabled === true;
}

function getToolWorkingDirectory(tool: ToolConfig): string | null {
  if (tool.tool_type !== 'ssh_shell' && tool.tool_type !== 'odoo_shell') {
    return null;
  }

  return (tool.connection_config as { working_directory?: string })?.working_directory || null;
}

function formatMountSourceInterval(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) { const m = Math.floor(seconds / 60); const s = seconds % 60; return s > 0 ? `${m}m ${s}s` : `${m}m`; }
  if (seconds < 86400) { const h = Math.floor(seconds / 3600); const m = Math.floor((seconds % 3600) / 60); return m > 0 ? `${h}h ${m}m` : `${h}h`; }
  const d = Math.floor(seconds / 86400); const h = Math.floor((seconds % 86400) / 3600);
  return h > 0 ? `${d}d ${h}h` : `${d}d`;
}

function isAutoSyncMountSource(source: UserspaceMountSource): boolean {
  return source.source_type === 'ssh'
    || source.source_type === 'microsoft_drive'
    || source.source_type === 'google_drive';
}

function getSuggestedGroupName(tool: ToolConfig | null | undefined): string {
  if (!tool) {
    return 'New Group';
  }

  const toolTypeName = TOOL_TYPE_INFO[tool.tool_type]?.name ?? 'Tool';
  return `${tool.name} ${toolTypeName}`;
}

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
  const hasSchemaIndexing = hasSchemaIndexingEnabled(tool);
  const workingDirectory = getToolWorkingDirectory(tool);

  // Inline editing state
  const [editingField, setEditingField] = useState<EditingField>(null);
  const [editName, setEditName] = useState(tool.name);
  const [editDescription, setEditDescription] = useState(tool.description || '');
  const [saving, setSaving] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const descTextareaRef = useRef<HTMLTextAreaElement>(null);

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
      case 'influxdb':
        if ('host' in config && config.host) {
          const port = 'port' in config ? config.port : 8086;
          const scheme = 'use_https' in config && config.use_https ? 'https' : 'http';
          const bucket = 'bucket' in config && config.bucket ? config.bucket : '(no bucket)';
          return `${scheme}://${config.host}:${port}/${bucket}`;
        }
        return 'InfluxDB connection';
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

      {(tool.allow_write || workingDirectory || activeSchemaJob) && (
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
          {workingDirectory && (
            <span className="constrained-path" title={`Constrained to ${workingDirectory}`}>
              <Icon name="folder" size={14} />
              <span className="path-text">{workingDirectory}</span>
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
  highlightSection?: string | null;
  onHighlightComplete?: () => void;
}

export function ToolsPanel({ onSchemaJobTriggered, schemaJobs = [], highlightSection, onHighlightComplete }: ToolsPanelProps) {
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [groups, setGroups] = useState<ToolGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [toasts, toast] = useToast();
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

  // Mount source state
  const [mountSources, setMountSources] = useState<UserspaceMountSource[]>([]);
  const [mountSourceDeletingId, setMountSourceDeletingId] = useState<string | null>(null);
  const [showMountSourceWizard, setShowMountSourceWizard] = useState(false);
  const [editingMountSource, setEditingMountSource] = useState<UserspaceMountSource | null>(null);
  const [disableConfirmation, setDisableConfirmation] = useState<{
    source: UserspaceMountSource;
    affected: MountSourceAffectedWorkspacesResponse | null;
    loading: boolean;
  } | null>(null);

  const loadTools = useCallback(async () => {
    try {
      setLoading(true);
      const [data, groupData, sources] = await Promise.all([
        api.listToolConfigs(),
        api.listToolGroups(),
        api.listUserspaceMountSources(),
      ]);
      // Filter out filesystem_indexer tools - they're shown in the Indexer tab
      const connectionTools = data.filter(t => t.tool_type !== 'filesystem_indexer');
      setTools(connectionTools);
      setGroups(groupData);
      setMountSources(sources);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load tools');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load schema stats for tools with schema indexing enabled
  const loadSchemaStats = useCallback(async (toolList: ToolConfig[]) => {
    const schemaTools = toolList.filter(hasSchemaIndexingEnabled);

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

  // Scroll to and highlight section when navigated from another view
  useEffect(() => {
    if (highlightSection && !loading) {
      const element = document.getElementById(`tools-${highlightSection}`);
      if (element) {
        element.classList.add('highlight-setting');
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        const timer = setTimeout(() => {
          element.classList.remove('highlight-setting');
          onHighlightComplete?.();
        }, 2000);
        return () => clearTimeout(timer);
      }
    }
  }, [highlightSection, loading, onHighlightComplete]);

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
    toast.success('Tool configuration saved successfully');
  };

  const patchToolInState = (toolId: string, updates: Partial<ToolConfig>) => {
    setTools((current) =>
      current.map((tool) => tool.id === toolId ? { ...tool, ...updates } : tool)
    );
  };

  const replaceToolInState = (updatedTool: ToolConfig) => {
    setTools((current) => current.map((tool) => tool.id === updatedTool.id ? updatedTool : tool));
  };

  const startEditingGroup = useCallback((groupId: string, name: string) => {
    setEditingGroupId(groupId);
    setEditingGroupName(name);
  }, []);

  const createSuggestedGroup = useCallback(async (tool: ToolConfig | null | undefined) => {
    const suggestedName = getSuggestedGroupName(tool);
    const created = await api.createToolGroup({ name: suggestedName });

    setGroups((current) => current.some((group) => group.id === created.id) ? current : [...current, created]);

    if (created.id) {
      startEditingGroup(created.id, suggestedName);
    }

    return { created, suggestedName };
  }, [startEditingGroup]);

  const handleDeleteTool = async (toolId: string) => {
    try {
      await api.deleteToolConfig(toolId);
      setTools((current) => current.filter((tool) => tool.id !== toolId));
      setHeartbeats((current) => {
        const next = { ...current };
        delete next[toolId];
        return next;
      });
      setSchemaStats((current) => {
        const next = { ...current };
        delete next[toolId];
        return next;
      });
      toast.success('Tool deleted successfully');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete tool');
    }
  };

  const handleToggleTool = async (toolId: string, enabled: boolean) => {
    try {
      const result = await api.toggleToolConfig(toolId, enabled);
      patchToolInState(toolId, { enabled: result.enabled });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to toggle tool');
    }
  };

  const handleTestTool = async (toolId: string) => {
    try {
      setTestingToolId(toolId);
      const result = await api.testSavedToolConnection(toolId);
      await loadTools();
      setHeartbeats((current) => ({
        ...current,
        [toolId]: {
          tool_id: toolId,
          alive: result.success,
          latency_ms: null,
          error: result.success ? null : result.message,
          checked_at: new Date().toISOString(),
        },
      }));
      if (result.success) {
        toast.success('Connection test successful');
      } else {
        toast.error(`Connection test failed: ${result.message}`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setTestingToolId(null);
    }
  };

  const handlePdmReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setPdmIndexingToolId(toolId);
      toast.success(fullReindex ? 'Starting full PDM re-index...' : 'Starting PDM index update...');
      await api.triggerPdmIndex(toolId, fullReindex);
      toast.success(fullReindex ? 'Full PDM re-index started' : 'PDM index update started');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to trigger PDM index');
    } finally {
      setPdmIndexingToolId(null);
    }
  };

  const handleSchemaReindex = async (toolId: string, fullReindex: boolean) => {
    try {
      setSchemaIndexingToolId(toolId);
      toast.success('Starting schema re-index...');
      await api.triggerSchemaIndex(toolId, fullReindex);
      toast.success('Schema re-index started');
      // Notify parent to refresh schema jobs list
      onSchemaJobTriggered?.();
      // Refresh schema stats after a short delay to allow job to start
      setTimeout(() => loadSchemaStats(tools), 2000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to trigger schema index');
    } finally {
      setSchemaIndexingToolId(null);
    }
  };

  const handleInlineUpdate = async (toolId: string, updates: { name?: string; description?: string }) => {
    try {
      const updatedTool = await api.updateToolConfig(toolId, updates);
      replaceToolInState(updatedTool);
      if (updates.name) {
        toast.success('Tool name updated');
      } else {
        toast.success('Description updated');
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update');
      throw err; // Re-throw to let ToolCard know the save failed
    }
  };

  // ---- Tool Group handlers ----

  const handleCreateGroup = async () => {
    try {
      await createSuggestedGroup(ungroupedTools[0]);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create group');
    }
  };

  const handleRenameGroup = async (groupId: string) => {
    const name = editingGroupName.trim();
    if (!name) return;
    try {
      const updatedGroup = await api.updateToolGroup(groupId, { name });
      setGroups((current) =>
        current.map((group) => group.id === groupId ? updatedGroup : group)
      );
      setEditingGroupId(null);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to rename group');
    }
  };

  const handleDeleteGroup = async (groupId: string) => {
    try {
      await api.deleteToolGroup(groupId);
      setGroups((current) => current.filter((group) => group.id !== groupId));
      setTools((current) =>
        current.map((tool) => tool.group_id === groupId ? { ...tool, group_id: null } : tool)
      );
      toast.success('Group deleted');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete group');
    }
  };

  const handleAssignGroup = async (toolId: string, groupId: string | null) => {
    try {
      const previousTool = tools.find((tool) => tool.id === toolId);
      const previousGroupId = previousTool?.group_id ?? null;
      const updatedTool = await api.updateToolConfig(toolId, { group_id: groupId ?? '' });
      replaceToolInState(updatedTool);

      await deleteEmptyGroupIfNeeded(toolId, previousGroupId, updatedTool.group_id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to assign group');
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

  const activeSchemaJobsByToolId = useMemo(() => {
    const jobsByToolId: Record<string, SchemaIndexJob> = {};

    for (const job of schemaJobs) {
      if ((job.status === 'pending' || job.status === 'indexing') && !jobsByToolId[job.tool_config_id]) {
        jobsByToolId[job.tool_config_id] = job;
      }
    }

    return jobsByToolId;
  }, [schemaJobs]);

  // Selected group tab (null = show ungrouped / all)
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const groupContentRef = useRef<HTMLDivElement>(null);

  const selectedGroup = useMemo(() => {
    if (!selectedGroupId) {
      return null;
    }

    return allGroups.find(({ group }) => group.id === selectedGroupId) || null;
  }, [allGroups, selectedGroupId]);

  const visibleTools = selectedGroup ? selectedGroup.tools : ungroupedTools;
  const showSelectedGroupEmptyState = Boolean(selectedGroupId) && visibleTools.length === 0;

  const deleteEmptyGroupIfNeeded = useCallback(async (toolId: string, previousGroupId: string | null, nextGroupId: string | null | undefined) => {
    if (!previousGroupId || previousGroupId === nextGroupId) {
      return;
    }

    const remainingToolsInPreviousGroup = tools.filter(
      (tool) => tool.id !== toolId && tool.group_id === previousGroupId
    );

    if (remainingToolsInPreviousGroup.length > 0) {
      return;
    }

    await api.deleteToolGroup(previousGroupId);
    setGroups((current) => current.filter((group) => group.id !== previousGroupId));

    if (selectedGroupId === previousGroupId) {
      setSelectedGroupId(null);
    }
  }, [tools, selectedGroupId]);

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

  const handleGroupTabsClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (selectedGroupId && e.target === e.currentTarget) {
      setSelectedGroupId(null);
    }
  }, [selectedGroupId]);

  // ---------------------------------------------------------------------------
  // Mount source handlers
  // ---------------------------------------------------------------------------

  const handleMountSourceWizardSaved = useCallback((saved: UserspaceMountSource) => {
    setMountSources((current) => {
      const next = current.some((source) => source.id === saved.id)
        ? current.map((source) => source.id === saved.id ? saved : source)
        : [...current, saved];
      return [...next].sort((left, right) => left.name.localeCompare(right.name));
    });
    setShowMountSourceWizard(false);
    setEditingMountSource(null);
    toast.success(saved.id === editingMountSource?.id ? 'Mount source updated.' : 'Mount source created.', 5000);
  }, [editingMountSource]);

  const handleDeleteMountSource = useCallback(async (mountSourceId: string) => {
    setMountSourceDeletingId(mountSourceId);

    try {
      await api.deleteUserspaceMountSource(mountSourceId);
      setMountSources((current) => current.filter((source) => source.id !== mountSourceId));
      toast.success('Mount source deleted.', 5000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete mount source');
    } finally {
      setMountSourceDeletingId(null);
    }
  }, []);

  const handleToggleMountSourceEnabled = useCallback(async (source: UserspaceMountSource) => {
    const nextEnabled = !source.enabled;

    if (!nextEnabled && source.usage_count > 0) {
      setDisableConfirmation({ source, affected: null, loading: true });
      try {
        const affected = await api.getMountSourceAffectedWorkspaces(source.id);
        setDisableConfirmation({ source, affected, loading: false });
      } catch {
        setDisableConfirmation({ source, affected: null, loading: false });
      }
      return;
    }

    setMountSources((current) =>
      current.map((s) => s.id === source.id ? { ...s, enabled: nextEnabled } : s)
    );
    try {
      await api.updateUserspaceMountSource(source.id, { enabled: nextEnabled });
    } catch (err) {
      setMountSources((current) =>
        current.map((s) => s.id === source.id ? { ...s, enabled: source.enabled } : s)
      );
      toast.error(err instanceof Error ? err.message : 'Failed to update mount source');
    }
  }, []);

  const handleConfirmDisableMountSource = useCallback(async () => {
    if (!disableConfirmation) return;
    const { source } = disableConfirmation;
    setDisableConfirmation(null);
    setMountSources((current) =>
      current.map((s) => s.id === source.id ? { ...s, enabled: false } : s)
    );
    try {
      const updated = await api.updateUserspaceMountSource(source.id, { enabled: false });
      setMountSources((current) =>
        current.map((s) => s.id === source.id ? { ...updated, usage_count: s.usage_count } : s)
      );
    } catch (err) {
      setMountSources((current) =>
        current.map((s) => s.id === source.id ? { ...s, enabled: true } : s)
      );
      toast.error(err instanceof Error ? err.message : 'Failed to disable mount source');
    }
  }, [disableConfirmation]);

  // Drag-and-drop state
  const [dragToolId, setDragToolId] = useState<string | null>(null);
  const [dragOverGroupId, setDragOverGroupId] = useState<string | null>(null);
  const [dragOverUngrouped, setDragOverUngrouped] = useState(false);
  // Within-list reorder DnD state
  const [dragOverToolId, setDragOverToolId] = useState<string | null>(null);
  const [dragInsertBefore, setDragInsertBefore] = useState(true);
  // Group-by-stacking DnD state (hovering over center of another card)
  const [dragGroupTargetId, setDragGroupTargetId] = useState<string | null>(null);
  const dragPreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingDragPreviewRef = useRef<DragPreviewTarget | null>(null);
  const showUngroupedDropZone = !selectedGroupId && Boolean(dragToolId);

  const previewInsertIndex = useMemo(() => {
    if (!dragToolId || !dragOverToolId) {
      return null;
    }

    const targetIdx = visibleTools.findIndex((tool) => tool.id === dragOverToolId);
    if (targetIdx < 0) {
      return null;
    }

    const idx = Math.max(0, Math.min(dragInsertBefore ? targetIdx : targetIdx + 1, visibleTools.length));

    // Don't show a drop slot at a position that would result in no change
    const fromIdx = visibleTools.findIndex((tool) => tool.id === dragToolId);
    if (fromIdx >= 0 && (idx === fromIdx || idx === fromIdx + 1)) {
      return null;
    }

    return idx;
  }, [dragToolId, dragOverToolId, dragInsertBefore, visibleTools]);

  const clearDragPreviewTimer = useCallback(() => {
    if (dragPreviewTimerRef.current) {
      clearTimeout(dragPreviewTimerRef.current);
      dragPreviewTimerRef.current = null;
    }
  }, []);

  const clearDragPreview = useCallback(() => {
    clearDragPreviewTimer();
    pendingDragPreviewRef.current = null;
    setDragOverToolId(null);
    setDragGroupTargetId(null);
  }, [clearDragPreviewTimer]);

  const scheduleDragPreview = useCallback((target: DragPreviewTarget) => {
    clearDragPreviewTimer();
    pendingDragPreviewRef.current = target;
    setDragOverToolId(null);

    dragPreviewTimerRef.current = setTimeout(() => {
      const pending = pendingDragPreviewRef.current;
      if (!pending) {
        dragPreviewTimerRef.current = null;
        return;
      }

      setDragOverToolId(pending.toolId);
      setDragInsertBefore(pending.insertBefore);

      dragPreviewTimerRef.current = null;
      pendingDragPreviewRef.current = null;
    }, DRAG_REORDER_PREVIEW_DELAY_MS);
  }, [clearDragPreviewTimer]);

  const handleDragStart = useCallback((e: React.DragEvent, toolId: string) => {
    e.dataTransfer.setData('text/plain', toolId);
    e.dataTransfer.effectAllowed = 'move';
    setDragToolId(toolId);
  }, []);

  const handleDragEnd = useCallback(() => {
    clearDragPreview();
    setDragToolId(null);
    setDragOverGroupId(null);
    setDragOverUngrouped(false);
  }, [clearDragPreview]);

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
    clearDragPreview();
    setDragOverGroupId(null);
    setDragToolId(null);
    if (toolId) {
      await handleAssignGroup(toolId, groupId);
    }
  }, [handleAssignGroup, clearDragPreview]);

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
    clearDragPreview();
    setDragOverUngrouped(false);
    setDragToolId(null);
    if (toolId) {
      await handleAssignGroup(toolId, null);
    }
  }, [handleAssignGroup, clearDragPreview]);

  // ---- Within-list reorder handlers ----

  const handleToolCardDragOver = useCallback((e: React.DragEvent, targetToolId: string) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'move';

    if (!dragToolId || dragToolId === targetToolId) {
      return;
    }

    // Detect stack/group zone in the center of the card. Edge zones reorder.
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;
    const isCenter = relX >= 0.25 && relX <= 0.75 && relY >= 0.25 && relY <= 0.75;

    if (isCenter) {
      // Center hover → show stack/group indicator, suppress reorder slot
      if (dragGroupTargetId !== targetToolId) {
        clearDragPreviewTimer();
        pendingDragPreviewRef.current = null;
        setDragOverToolId(null);
        setDragGroupTargetId(targetToolId);
      }
      return;
    }

    // Edge hover → reorder mode; clear any group target
    if (dragGroupTargetId !== null) {
      setDragGroupTargetId(null);
    }

    const insertBefore = relX < 0.5;

    if (dragOverToolId === targetToolId && dragInsertBefore === insertBefore) {
      return;
    }

    const pending = pendingDragPreviewRef.current;
    if (pending?.toolId === targetToolId && pending.insertBefore === insertBefore) {
      return;
    }

    scheduleDragPreview({ toolId: targetToolId, insertBefore });
  }, [dragToolId, dragOverToolId, dragInsertBefore, dragGroupTargetId, clearDragPreviewTimer, scheduleDragPreview]);

  const handleToolCardDragLeave = useCallback((e: React.DragEvent) => {
    const related = e.relatedTarget as HTMLElement;
    if (
      !(e.currentTarget as HTMLElement).contains(related) &&
      (!related || !related.closest('.tool-reorder-drop-slot'))
    ) {
      clearDragPreview();
    }
  }, [clearDragPreview]);

  const reorderVisibleTools = useCallback(async (draggedId: string, insertIdx: number) => {
    try {
      const reordered = [...visibleTools];
      const fromIdx = reordered.findIndex(t => t.id === draggedId);

      let moved: ToolConfig;
      let safeInsertIdx = insertIdx;

      if (fromIdx >= 0) {
        [moved] = reordered.splice(fromIdx, 1);
        if (fromIdx < safeInsertIdx) {
          safeInsertIdx -= 1;
        }
      } else {
        const draggedTool = tools.find(t => t.id === draggedId);
        if (!draggedTool) return;
        moved = await api.updateToolConfig(draggedId, { group_id: selectedGroupId ?? '' });
      }

      safeInsertIdx = Math.max(0, Math.min(safeInsertIdx, reordered.length));
      reordered.splice(safeInsertIdx, 0, moved);

      // Persist a full global order so sort_order remains unique and stable across reloads.
      // Sending only the visible subset causes sort_order collisions with hidden tools.
      const visibleIds = visibleTools.map((t) => t.id);
      const visibleIdSet = new Set(visibleIds);
      const reorderedVisibleIds = reordered.map((t) => t.id);

      let fullOrderedIds: string[];

      if (fromIdx >= 0) {
        // Reorder within the currently visible subset while keeping non-visible tools in place.
        let nextVisibleIdx = 0;
        fullOrderedIds = tools.map((tool) => {
          if (!visibleIdSet.has(tool.id)) {
            return tool.id;
          }
          const replacement = reorderedVisibleIds[nextVisibleIdx];
          nextVisibleIdx += 1;
          return replacement;
        });
      } else {
        // Dragged from outside the visible subset: remove old position, then insert into
        // this subset's band at the resolved visible insertion index.
        fullOrderedIds = tools.map((t) => t.id).filter((id) => id !== draggedId);

        let globalInsertIdx = fullOrderedIds.length;
        if (visibleIds.length > 0) {
          if (safeInsertIdx >= visibleIds.length) {
            const anchorId = visibleIds[visibleIds.length - 1];
            const anchorIdx = fullOrderedIds.indexOf(anchorId);
            if (anchorIdx >= 0) {
              globalInsertIdx = anchorIdx + 1;
            }
          } else {
            const anchorId = visibleIds[safeInsertIdx];
            const anchorIdx = fullOrderedIds.indexOf(anchorId);
            if (anchorIdx >= 0) {
              globalInsertIdx = anchorIdx;
            }
          }
        }

        fullOrderedIds.splice(globalInsertIdx, 0, draggedId);
      }

      const updatedSortOrders = new Map(fullOrderedIds.map((id, i) => [id, i * 100]));
      setTools((current) => {
        const byId = new Map(current.map((tool) => [tool.id, tool]));
        byId.set(draggedId, {
          ...(byId.get(draggedId) ?? moved),
          ...moved,
          sort_order: updatedSortOrders.get(draggedId) ?? moved.sort_order,
        });

        return fullOrderedIds
          .map((id) => {
            const tool = byId.get(id);
            if (!tool) return null;
            const nextOrder = updatedSortOrders.get(id);
            return nextOrder === undefined ? tool : { ...tool, sort_order: nextOrder };
          })
          .filter((tool): tool is ToolConfig => tool !== null);
      });

      await api.reorderTools({ tool_ids: fullOrderedIds });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to reorder tools');
      await loadTools();
    }
  }, [visibleTools, tools, selectedGroupId, loadTools]);

  const handleToolCardDrop = useCallback(async (e: React.DragEvent, targetToolId: string) => {
    e.preventDefault();
    e.stopPropagation();
    const draggedId = e.dataTransfer.getData('text/plain');

    const localDragOverId = dragOverToolId;
    const localInsertBefore = localDragOverId ? dragInsertBefore : true;
    const localGroupTarget = dragGroupTargetId;

    clearDragPreview();
    setDragToolId(null);

    if (!draggedId || draggedId === targetToolId) return;

    // Center drop → create a new group containing both tools
    if (localGroupTarget === targetToolId) {
      try {
        const targetTool = tools.find((tool) => tool.id === targetToolId) ?? null;
        const { created: newGroup } = await createSuggestedGroup(targetTool);
        // Assign both tools to the new group (target first so it retains lower sort_order)
        await Promise.all([
          handleAssignGroup(targetToolId, newGroup.id),
          handleAssignGroup(draggedId, newGroup.id),
        ]);
        setSelectedGroupId(newGroup.id);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : 'Failed to create group');
      }
      return;
    }

    const resolvedTargetId = localDragOverId ?? targetToolId;

    const toIdx = visibleTools.findIndex((t) => t.id === resolvedTargetId);
    if (toIdx < 0) return;
    const insertIdx = localInsertBefore ? toIdx : toIdx + 1;
    await reorderVisibleTools(draggedId, insertIdx);
  }, [visibleTools, dragOverToolId, dragInsertBefore, dragGroupTargetId, reorderVisibleTools, clearDragPreview, handleAssignGroup, toast]);

  const handleSlotDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const draggedId = e.dataTransfer.getData('text/plain');
    const localDragOverId = dragOverToolId;
    const localInsertBefore = dragInsertBefore;

    clearDragPreview();
    setDragToolId(null);

    if (!draggedId || !localDragOverId || draggedId === localDragOverId) return;

    const toIdx = visibleTools.findIndex(t => t.id === localDragOverId);
    const insertIdx = localInsertBefore ? toIdx : toIdx + 1;
    await reorderVisibleTools(draggedId, insertIdx);
  }, [visibleTools, dragOverToolId, dragInsertBefore, reorderVisibleTools, clearDragPreview]);

  const handleGridDragOver = useCallback((e: React.DragEvent) => {
    if (!dragToolId) return;

    if ((e.target as HTMLElement).closest('.tool-card-drag-wrap, .tool-reorder-drop-slot')) {
      return;
    }

    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragGroupTargetId(null);

    // Hovering over the grid background itself, so push drop slot to the very end
    const lastTool = visibleTools[visibleTools.length - 1];
    if (lastTool && lastTool.id !== dragToolId) {
      // Guard against resetting the timer on every rapid dragover event:
      // check the pending ref (like handleToolCardDragOver does) so we only
      // schedule once and let the 80ms timer commit.
      const pending = pendingDragPreviewRef.current;
      if (pending?.toolId === lastTool.id && pending.insertBefore === false) {
        return;
      }
      if (dragOverToolId !== lastTool.id || dragInsertBefore !== false) {
        scheduleDragPreview({ toolId: lastTool.id, insertBefore: false });
      }
    }
  }, [dragToolId, dragOverToolId, dragInsertBefore, visibleTools, scheduleDragPreview]);

  const handleGridDrop = useCallback(async (e: React.DragEvent) => {
    if (!dragToolId) return;
    if ((e.target as HTMLElement).closest('.tool-card-drag-wrap, .tool-reorder-drop-slot')) {
      return;
    }

    e.preventDefault();
    const draggedId = e.dataTransfer.getData('text/plain');

    clearDragPreview();
    setDragToolId(null);

    if (!draggedId) return;

    const lastTool = visibleTools[visibleTools.length - 1];
    if (lastTool) {
      await reorderVisibleTools(draggedId, visibleTools.length);
    }
  }, [dragToolId, visibleTools, reorderVisibleTools, clearDragPreview]);

  useEffect(() => {
    return () => {
      clearDragPreviewTimer();
      pendingDragPreviewRef.current = null;
    };
  }, [clearDragPreviewTimer]);

  const renderToolCard = (tool: ToolConfig) => {
    const isBeingDragged = dragToolId === tool.id;
    const isGroupTarget = dragGroupTargetId === tool.id;
    return (
      <div
        key={tool.id}
        draggable
        onDragStart={(e) => handleDragStart(e, tool.id)}
        onDragEnd={handleDragEnd}
        onDragOver={(e) => handleToolCardDragOver(e, tool.id)}
        onDragLeave={handleToolCardDragLeave}
        onDrop={(e) => handleToolCardDrop(e, tool.id)}
        className={`tool-card-drag-wrap${isBeingDragged ? ' dragging' : ''}${isGroupTarget ? ' group-target' : ''}`}
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
          activeSchemaJob={activeSchemaJobsByToolId[tool.id] || null}
          schemaStats={schemaStats[tool.id] || null}
          onInlineUpdate={handleInlineUpdate}
        />
      </div>
    );
  };

  const handleDropSlotDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const renderDropSlot = (key: string) => (
    <div
      key={key}
      className="tool-reorder-drop-slot"
      aria-hidden="true"
      onDragOver={handleDropSlotDragOver}
      onDrop={handleSlotDrop}
    />
  );

  const renderToolCardsWithDropSlot = () => {
    const items: JSX.Element[] = [];
    const slotPrefix = `tool-reorder-drop-slot-${selectedGroupId ?? 'ungrouped'}`;

    visibleTools.forEach((tool, idx) => {
      if (previewInsertIndex === idx) {
        items.push(renderDropSlot(`${slotPrefix}-${idx}`));
      }
      items.push(renderToolCard(tool));
    });

    if (previewInsertIndex === visibleTools.length) {
      items.push(renderDropSlot(`${slotPrefix}-end`));
    }

    return items;
  };

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
        <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />

        <p className="fieldset-help">
          Configure connections to databases, shells, and other tools that the AI can use during conversations.
          Each tool can have multiple instances (e.g., production and staging databases). To group a tool, create a group and drag the tool into it.
          Deleting a group does not delete the tools.
        </p>

        {loading ? (
          <p className="muted">Loading tools...</p>
        ) : tools.length === 0 ? (
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
            <div
              className={`tool-group-tabs${dragToolId ? ' dragging' : ''}`}
              onClick={handleGroupTabsClick}
            >
              {allGroups.length > 0 && allGroups.map(({ group, tools: groupTools }) => {
                const isActive = selectedGroupId === group.id;
                const isDragTarget = dragOverGroupId === group.id;
                return (
                  <div
                    key={group.id}
                    className={`tool-group-tab${isActive ? ' active' : ''}${isDragTarget ? ' drag-over' : ''}${editingGroupId === group.id ? ' editing' : ''}`}
                    onClick={() => setSelectedGroupId(isActive ? null : group.id)}
                    onDragOver={(e) => handleGroupDragOver(e, group.id)}
                    onDragLeave={(e) => handleGroupDragLeave(e, group.id)}
                    onDrop={(e) => handleGroupDrop(e, group.id)}
                  >
                      {editingGroupId === group.id ? (
                        <textarea
                          className="tool-group-tab-input"
                          value={editingGroupName}
                          onChange={(e) => setEditingGroupName(e.target.value)}
                          onBlur={() => handleRenameGroup(group.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') { e.preventDefault(); handleRenameGroup(group.id); }
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
            {showSelectedGroupEmptyState ? (
              <div className="tool-group-active-panel">
                <p className="muted" style={{ textAlign: 'center', margin: 0 }}>No tools in this group yet. Drag tools here to add them.</p>
              </div>
            ) : selectedGroupId && selectedGroup ? (
              <div className="tool-group-active-panel">
                <div
                  className="tools-grid"
                  onDragOver={(e) => dragToolId ? handleGridDragOver(e) : undefined}
                  onDrop={(e) => dragToolId ? handleGridDrop(e) : undefined}
                >
                  {renderToolCardsWithDropSlot()}
                </div>
              </div>
            ) : (
              <div
                className={`tool-group-section-panel${showUngroupedDropZone ? ' tool-ungrouped-drop-zone' : ''}${dragOverUngrouped ? ' drag-over' : ''}`}
                onDragOver={(e) => {
                  if (showUngroupedDropZone) handleUngroupedDragOver(e);
                  if (dragToolId) handleGridDragOver(e);
                }}
                onDragLeave={showUngroupedDropZone ? handleUngroupedDragLeave : undefined}
                onDrop={(e) => {
                  const draggedTool = tools.find(t => t.id === dragToolId);
                  if (showUngroupedDropZone && draggedTool?.group_id) handleUngroupedDrop(e);
                  else if (dragToolId) handleGridDrop(e);
                }}
              >
                <div className="tools-grid">
                  {renderToolCardsWithDropSlot()}
                </div>
              </div>
            )}
            </div>
          </>
        )}
        </>
        )}
      </div>

      {/* Mounts section */}
      <div className="card" id="tools-mount-sources">
        <div className="card-header">
          <h2>Mounts</h2>
          <AnimatedCreateButton
            isExpanded={showMountSourceWizard}
            onClick={() => {
              if (showMountSourceWizard) {
                setShowMountSourceWizard(false);
                setEditingMountSource(null);
              } else {
                setEditingMountSource(null);
                setShowMountSourceWizard(true);
              }
            }}
            label="New Mount Source"
          />
        </div>

        {showMountSourceWizard ? (
          <MountSourceWizard
            existingSource={editingMountSource}
            existingNames={mountSources.map((s) => s.name)}
            onClose={() => { setShowMountSourceWizard(false); setEditingMountSource(null); }}
            onSaved={handleMountSourceWizardSaved}
            embedded={true}
          />
        ) : (
          <>
        <p className="fieldset-help">
          Define mount sources backed by SSH, filesystem, OneDrive, or Google Drive. Workspaces attach these sources without duplicating connection credentials.
        </p>

        <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))' }}>
          {mountSources.length > 0 ? mountSources.map((source) => (
            <div
              key={source.id}
              className="card"
              style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: '12px 14px', opacity: source.enabled ? 1 : 0.6 }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
                  <HardDrive size={14} style={{ flexShrink: 0 }} />
                  <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{source.name}</span>
                  <span className="muted" style={{ fontSize: '0.8rem', flexShrink: 0 }}>
                    {(() => {
                      switch (source.source_type) {
                        case 'ssh':
                          return 'SSH';
                        case 'microsoft_drive':
                          return 'OneDrive';
                        case 'google_drive':
                          return 'Google Drive';
                        default:
                          return source.mount_backend.replace('_', ' ');
                      }
                    })()}
                  </span>
                  {source.usage_count > 0 && (
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        padding: '1px 6px',
                        borderRadius: 8,
                        background: 'var(--color-accent)',
                        color: 'var(--color-bg)',
                        flexShrink: 0,
                      }}
                      title={`Used by ${source.usage_count} workspace${source.usage_count !== 1 ? 's' : ''}`}
                    >
                      {source.usage_count}
                    </span>
                  )}
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => { setEditingMountSource(source); setShowMountSourceWizard(true); }}
                  >
                    <Pencil size={12} />
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => void handleDeleteMountSource(source.id)}
                    disabled={mountSourceDeletingId === source.id}
                  >
                    <Trash2 size={12} />
                  </button>
                  <label className="toggle-switch" style={{ marginLeft: 2 }}>
                    <input
                      type="checkbox"
                      checked={source.enabled}
                      onChange={() => void handleToggleMountSourceEnabled(source)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </span>
              </div>
              {source.approved_paths.length > 0 && (
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {source.approved_paths.map((p) => {
                    const displayPath = resolveSourceDisplayPath(p, undefined, { sourceType: source.source_type });
                    return (
                      <code key={p} style={{ fontSize: '0.8rem', padding: '2px 8px', background: 'var(--color-bg-tertiary)', borderRadius: 4, border: '1px solid var(--color-border)' }}>
                        {displayPath === '/'
                          ? '/'
                          : displayPath.split('/').filter(Boolean).map((seg, i) => (
                            <span key={i}>
                              <span style={{ opacity: 0.5 }}>/</span>{seg}
                            </span>
                          ))
                        }
                      </code>
                    );
                  })}
                </div>
              )}
              {isAutoSyncMountSource(source) && source.sync_interval_seconds != null && (
                <span className="muted" style={{ fontSize: '0.75rem' }}>
                  Sync interval: {formatMountSourceInterval(source.sync_interval_seconds)}
                </span>
              )}
              {(source.source_type === 'microsoft_drive' || source.source_type === 'google_drive') && source.account_email && (
                <span className="muted" style={{ fontSize: '0.75rem' }}>
                  Account: {source.account_email}
                </span>
              )}
            </div>
          )) : (
            <p className="muted" style={{ margin: 0 }}>No mount sources configured yet.</p>
          )}
        </div>
          </>
        )}
      </div>

      {/* Mount Source Disable Confirmation Modal */}
      {disableConfirmation && (
        <div className="modal-overlay" onClick={() => setDisableConfirmation(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Disable Mount Source</h3>
              <button className="modal-close" onClick={() => setDisableConfirmation(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              {disableConfirmation.loading ? (
                <p className="muted">Loading affected workspaces...</p>
              ) : (
                <>
                  <p>
                    Disabling <strong>{disableConfirmation.source.name}</strong> will immediately stop
                    {disableConfirmation.source.source_type === 'ssh' ? ' SFTP sync' : ' mount access'} for
                    {' '}<strong>{disableConfirmation.affected?.total_mounts ?? disableConfirmation.source.usage_count}</strong> mount{(disableConfirmation.affected?.total_mounts ?? disableConfirmation.source.usage_count) !== 1 ? 's' : ''} across
                    {' '}<strong>{disableConfirmation.affected?.workspaces.length ?? '?'}</strong> workspace{(disableConfirmation.affected?.workspaces.length ?? 0) !== 1 ? 's' : ''}.
                  </p>
                  {disableConfirmation.affected && disableConfirmation.affected.workspaces.length > 0 && (
                    <div style={{ maxHeight: 200, overflowY: 'auto', margin: '12px 0', border: '1px solid var(--color-border)', borderRadius: 6 }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)' }}>
                            <th style={{ padding: '6px 10px', textAlign: 'left' }}>Workspace</th>
                            <th style={{ padding: '6px 10px', textAlign: 'right' }}>Mounts</th>
                          </tr>
                        </thead>
                        <tbody>
                          {disableConfirmation.affected.workspaces.map((ws) => (
                            <tr key={ws.workspace_id} style={{ borderBottom: '1px solid var(--color-border)' }}>
                              <td style={{ padding: '6px 10px' }}>{ws.workspace_name}</td>
                              <td style={{ padding: '6px 10px', textAlign: 'right' }}>{ws.mount_count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <p className="field-help" style={{ margin: '8px 0 0' }}>
                    {disableConfirmation.source.source_type === 'ssh'
                      ? 'Auto-sync will be stopped for all affected mounts. Re-enabling the source will allow mounts to resume syncing.'
                      : 'Workspace access to this mount will be interrupted. Re-enabling restores access without reconfiguration.'}
                  </p>
                </>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setDisableConfirmation(null)}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => void handleConfirmDisableMountSource()}
                disabled={disableConfirmation.loading}
              >
                Disable Mount Source
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
