import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api';
import type {
  McpRouteConfig,
  CreateMcpRouteRequest,
  UpdateMcpRouteRequest,
  ToolConfig,
  IndexInfo,
  AppSettings,
} from '@/types';

interface MCPRoutesPanelProps {
  onClose?: () => void;
}

// Route card component
interface RouteCardProps {
  route: McpRouteConfig;
  tools: ToolConfig[];
  onEdit: (route: McpRouteConfig) => void;
  onDelete: (routeId: string) => void;
  onToggle: (routeId: string, enabled: boolean) => void;
}

function RouteCard({ route, tools, onEdit, onDelete, onToggle }: RouteCardProps) {
  const selectedTools = tools.filter(t => route.tool_config_ids.includes(t.id));

  return (
    <div className={`tool-card ${!route.enabled ? 'disabled' : ''}`}>
      <div className="tool-card-header">
        <div className="tool-card-icon">🔌</div>
        <div className="tool-card-title">
          <h3>{route.name}</h3>
          <span className="tool-card-type">/mcp/{route.route_path}</span>
        </div>
        <div className="tool-card-status">
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={route.enabled}
              onChange={(e) => onToggle(route.id, e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
      </div>

      {route.description && (
        <p className="tool-card-description">{route.description}</p>
      )}

      <div className="tool-card-meta">
        {route.require_auth && route.has_password && <span className="write-enabled">Password protected</span>}
        {route.require_auth && !route.has_password && <span style={{ color: 'var(--color-warning, #f59e0b)' }}>Auth enabled (no password set)</span>}
        {route.include_knowledge_search && <span>Knowledge search</span>}
        {route.include_git_history && <span>Git history</span>}
        <span>{selectedTools.length} tool(s)</span>
      </div>

      {selectedTools.length > 0 && (
        <div className="tool-card-connection">
          <span className="connection-label">Tools:</span>
          <span>{selectedTools.map(t => t.name).join(', ')}</span>
        </div>
      )}

      <div className="tool-card-actions">
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onEdit(route)}
        >
          Edit
        </button>
        <button
          type="button"
          className="btn btn-sm btn-danger"
          onClick={() => onDelete(route.id)}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

// Route wizard/form
interface RouteWizardProps {
  editingRoute: McpRouteConfig | null;
  tools: ToolConfig[];
  documentIndexes: IndexInfo[];
  filesystemTools: ToolConfig[];
  schemaTools: ToolConfig[];
  aggregateSearch: boolean;
  onSave: (data: CreateMcpRouteRequest | UpdateMcpRouteRequest, routeId?: string) => Promise<void>;
  onCancel: () => void;
  saving: boolean;
}

function RouteWizard({
  editingRoute,
  tools,
  documentIndexes,
  filesystemTools,
  schemaTools,
  aggregateSearch,
  onSave,
  onCancel,
  saving,
}: RouteWizardProps) {
  const [name, setName] = useState(editingRoute?.name || '');
  const [routePath, setRoutePath] = useState(editingRoute?.route_path || '');
  const [description, setDescription] = useState(editingRoute?.description || '');
  const [requireAuth, setRequireAuth] = useState(editingRoute?.require_auth ?? false);
  // Pre-populate password if editing and one exists (backend decrypts it)
  const [authPassword, setAuthPassword] = useState(editingRoute?.auth_password || '');
  const [showPassword, setShowPassword] = useState(false);
  const [clearPassword, setClearPassword] = useState(false);

  // Document index selection (when aggregate is true, this is a single toggle; when false, per-index)
  const [includeKnowledgeSearch, setIncludeKnowledgeSearch] = useState(editingRoute?.include_knowledge_search ?? false);
  const [selectedDocIndexes, setSelectedDocIndexes] = useState<Set<string>>(
    new Set(editingRoute?.selected_document_indexes || [])
  );

  // Git history selection
  const [includeGitHistory, setIncludeGitHistory] = useState(editingRoute?.include_git_history ?? false);

  // Filesystem index selection (per tool)
  const [selectedFilesystemTools, setSelectedFilesystemTools] = useState<Set<string>>(
    new Set(editingRoute?.selected_filesystem_indexes || [])
  );

  // Schema index selection (per tool that has schema indexer)
  const [selectedSchemaTools, setSelectedSchemaTools] = useState<Set<string>>(
    new Set(editingRoute?.selected_schema_indexes || [])
  );

  // Database/shell tool selection
  const [selectedToolIds, setSelectedToolIds] = useState<Set<string>>(
    new Set(editingRoute?.tool_config_ids || [])
  );

  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    if (!routePath.trim()) {
      setError('Route path is required');
      return;
    }

    // Validate route path format (must match backend pattern: ^[a-z][a-z0-9_]*$)
    const pathRegex = /^[a-z][a-z0-9_]*$/;
    if (!pathRegex.test(routePath)) {
      setError('Route path must start with a lowercase letter and contain only lowercase letters, numbers, and underscores');
      return;
    }

    // Validate password if auth is required and no existing password
    if (requireAuth && !editingRoute?.has_password && !authPassword && !clearPassword) {
      setError('Password is required when authentication is enabled');
      return;
    }

    if (authPassword && authPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    try {
      if (editingRoute) {
        const updateData: UpdateMcpRouteRequest = {
          name: name.trim(),
          description: description.trim(),
          require_auth: requireAuth,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        if (authPassword) {
          updateData.auth_password = authPassword;
        }
        if (clearPassword) {
          updateData.clear_password = true;
        }
        await onSave(updateData, editingRoute.id);
      } else {
        const createData: CreateMcpRouteRequest = {
          name: name.trim(),
          route_path: routePath.trim().toLowerCase(),
          description: description.trim(),
          require_auth: requireAuth,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        if (authPassword) {
          createData.auth_password = authPassword;
        }
        await onSave(createData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save route');
    }
  };

  const toggleTool = (toolId: string) => {
    setSelectedToolIds(prev => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  const toggleDocIndex = (indexName: string) => {
    setSelectedDocIndexes(prev => {
      const next = new Set(prev);
      if (next.has(indexName)) {
        next.delete(indexName);
      } else {
        next.add(indexName);
      }
      return next;
    });
  };

  const toggleFilesystemTool = (toolId: string) => {
    setSelectedFilesystemTools(prev => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  const toggleSchemaTool = (toolId: string) => {
    setSelectedSchemaTools(prev => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  return (
    <div className="wizard-panel">
      <div className="wizard-header">
        <h3>{editingRoute ? 'Edit MCP Route' : 'Create MCP Route'}</h3>
        <button type="button" className="close-btn" onClick={onCancel}>✕</button>
      </div>

      <form onSubmit={handleSubmit} className="wizard-form">
        {error && <div className="error-banner">{error}</div>}

        <div className="form-group">
          <label>Route Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Custom Tools"
            required
          />
          <p className="field-help">A friendly name for this MCP route configuration.</p>
        </div>

        <div className="form-group">
          <label>Route Path</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span className="muted">/mcp/</span>
            <input
              type="text"
              value={routePath}
              onChange={(e) => setRoutePath(e.target.value)}
              placeholder="my_tools"
              disabled={!!editingRoute}
              required
              style={{ flex: 1 }}
            />
          </div>
          <p className="field-help">
            {editingRoute
              ? 'Route path cannot be changed after creation.'
              : 'URL path for this MCP endpoint. Use lowercase letters, numbers, and underscores.'}
          </p>
        </div>

        <div className="form-group">
          <label>Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Tools for specific use case..."
            rows={2}
          />
          <p className="field-help">Optional description for documentation purposes.</p>
        </div>

        <fieldset>
          <legend>Authentication</legend>
          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={requireAuth}
                onChange={(e) => {
                  setRequireAuth(e.target.checked);
                  if (!e.target.checked) {
                    setClearPassword(false);
                  }
                }}
                style={{ marginRight: '0.5rem' }}
              />
              <span>Require password authentication</span>
            </label>
            <p className="field-help">
              When enabled, clients must provide the password via Bearer authentication header.
            </p>
          </div>

          {requireAuth && (
            <div className="form-group">
              <label>
                Password
                {editingRoute?.has_password && !clearPassword && (
                  <span className="muted" style={{ marginLeft: '0.5rem', fontWeight: 'normal' }}>
                    (currently set)
                  </span>
                )}
              </label>
              {editingRoute?.has_password && !clearPassword ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    placeholder="Enter new password to change"
                    style={{ flex: 1 }}
                  />
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? 'Hide' : 'Show'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    onClick={() => setClearPassword(true)}
                  >
                    Clear
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    placeholder="Enter password (min 8 characters)"
                    required={requireAuth && !editingRoute?.has_password}
                    minLength={8}
                    style={{ flex: 1 }}
                  />
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? 'Hide' : 'Show'}
                  </button>
                  {clearPassword && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setClearPassword(false)}
                    >
                      Cancel
                    </button>
                  )}
                </div>
              )}
              <p className="field-help">
                MCP clients will use this password as the Bearer token. Minimum 8 characters.
              </p>
            </div>
          )}
        </fieldset>

        {/* Document Indexes Section */}
        <fieldset>
          <legend>Document Indexes</legend>
          <p className="fieldset-help">
            {aggregateSearch
              ? 'Enable the combined knowledge search tool for all document indexes.'
              : 'Select which document indexes to expose as individual search tools.'}
          </p>

          {aggregateSearch ? (
            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeKnowledgeSearch}
                  onChange={(e) => setIncludeKnowledgeSearch(e.target.checked)}
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Include Knowledge Search</span>
              </label>
              <p className="field-help">
                Exposes a single <code>search_knowledge</code> tool that searches all document indexes.
              </p>
            </div>
          ) : (
            <>
              {documentIndexes.length === 0 ? (
                <p className="muted">No document indexes available.</p>
              ) : (
                <div className="tool-selection-list">
                  {documentIndexes.map(idx => (
                    <label key={idx.name} className="checkbox-label tool-selection-item">
                      <input
                        type="checkbox"
                        checked={selectedDocIndexes.has(idx.name)}
                        onChange={() => toggleDocIndex(idx.name)}
                        style={{ marginRight: '0.5rem' }}
                      />
                      <span className="tool-selection-name">{idx.display_name || idx.name}</span>
                      <span className="tool-selection-type muted">{idx.source_type}</span>
                      {!idx.enabled && <span className="tool-selection-disabled">(disabled)</span>}
                    </label>
                  ))}
                </div>
              )}

              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={includeGitHistory}
                    onChange={(e) => setIncludeGitHistory(e.target.checked)}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span>Include Git History Search</span>
                </label>
                <p className="field-help">
                  Exposes git history search tools for indexed repositories.
                </p>
              </div>
            </>
          )}

          {aggregateSearch && (
            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={includeGitHistory}
                  onChange={(e) => setIncludeGitHistory(e.target.checked)}
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Include Git History Search</span>
              </label>
              <p className="field-help">
                Exposes git history search tools for indexed repositories.
              </p>
            </div>
          )}
        </fieldset>

        {/* Filesystem Indexes Section */}
        {filesystemTools.length > 0 && (
          <fieldset>
            <legend>Filesystem Indexes</legend>
            <p className="fieldset-help">
              Select which filesystem indexer tools to expose on this route.
            </p>

            <div className="tool-selection-list">
              {filesystemTools.map(tool => (
                <label key={tool.id} className="checkbox-label tool-selection-item">
                  <input
                    type="checkbox"
                    checked={selectedFilesystemTools.has(tool.id)}
                    onChange={() => toggleFilesystemTool(tool.id)}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span className="tool-selection-name">{tool.name}</span>
                  <span className="tool-selection-type muted">filesystem</span>
                  {!tool.enabled && <span className="tool-selection-disabled">(disabled)</span>}
                </label>
              ))}
            </div>
          </fieldset>
        )}

        {/* Schema Indexes Section (from database tools) */}
        {schemaTools.length > 0 && (
          <fieldset>
            <legend>Schema Indexes</legend>
            <p className="fieldset-help">
              Select which database schema indexes to expose on this route.
            </p>

            <div className="tool-selection-list">
              {schemaTools.map(tool => (
                <label key={tool.id} className="checkbox-label tool-selection-item">
                  <input
                    type="checkbox"
                    checked={selectedSchemaTools.has(tool.id)}
                    onChange={() => toggleSchemaTool(tool.id)}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span className="tool-selection-name">{tool.name} Schema</span>
                  <span className="tool-selection-type muted">{tool.tool_type}</span>
                  {!tool.enabled && <span className="tool-selection-disabled">(disabled)</span>}
                </label>
              ))}
            </div>
          </fieldset>
        )}

        {/* Tools Section (database/shell) */}
        <fieldset>
          <legend>Tools</legend>
          <p className="fieldset-help">
            Select which database and shell tools to expose on this route.
          </p>

          {tools.length === 0 ? (
            <p className="muted">No tools configured. Create tools in the Tools panel first.</p>
          ) : (
            <div className="tool-selection-list">
              {tools.map(tool => (
                <label key={tool.id} className="checkbox-label tool-selection-item">
                  <input
                    type="checkbox"
                    checked={selectedToolIds.has(tool.id)}
                    onChange={() => toggleTool(tool.id)}
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span className="tool-selection-name">{tool.name}</span>
                  <span className="tool-selection-type muted">{tool.tool_type}</span>
                  {!tool.enabled && <span className="tool-selection-disabled">(disabled)</span>}
                </label>
              ))}
            </div>
          )}
        </fieldset>

        <div className="wizard-actions">
          <button type="button" className="btn btn-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="btn" disabled={saving}>
            {saving ? 'Saving...' : editingRoute ? 'Update Route' : 'Create Route'}
          </button>
        </div>
      </form>
    </div>
  );
}

// Main panel
export function MCPRoutesPanel({ onClose }: MCPRoutesPanelProps) {
  const [routes, setRoutes] = useState<McpRouteConfig[]>([]);
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [documentIndexes, setDocumentIndexes] = useState<IndexInfo[]>([]);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [showWizard, setShowWizard] = useState(false);
  const [editingRoute, setEditingRoute] = useState<McpRouteConfig | null>(null);
  const [saving, setSaving] = useState(false);

  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  // Categorize tools
  const filesystemTools = tools.filter(t => t.tool_type === 'filesystem_indexer');
  const schemaTools = tools.filter(t => {
    if (!['postgres', 'mssql'].includes(t.tool_type)) return false;
    const config = t.connection_config as { schema_index_enabled?: boolean } | undefined;
    return config?.schema_index_enabled === true;
  });
  const otherTools = tools.filter(t =>
    t.tool_type !== 'filesystem_indexer'
  );

  // Load routes, tools, indexes, and settings
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [routesRes, toolsRes, indexesRes, settingsRes] = await Promise.all([
        api.listMcpRoutes(),
        api.listToolConfigs(),
        api.listIndexes(),
        api.getSettings(),
      ]);
      setRoutes(routesRes.routes);
      setTools(toolsRes);
      setDocumentIndexes(indexesRes);
      setSettings(settingsRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load MCP routes');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Save route (create or update)
  const handleSave = async (data: CreateMcpRouteRequest | UpdateMcpRouteRequest, routeId?: string) => {
    setSaving(true);
    try {
      if (routeId) {
        await api.updateMcpRoute(routeId, data as UpdateMcpRouteRequest);
        setSuccess('Route updated successfully');
      } else {
        await api.createMcpRoute(data as CreateMcpRouteRequest);
        setSuccess('Route created successfully');
      }
      setShowWizard(false);
      setEditingRoute(null);
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } finally {
      setSaving(false);
    }
  };

  // Toggle route enabled
  const handleToggle = async (routeId: string, enabled: boolean) => {
    try {
      await api.toggleMcpRoute(routeId, enabled);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle route');
    }
  };

  // Delete route
  const handleDelete = async (routeId: string) => {
    try {
      await api.deleteMcpRoute(routeId);
      setDeleteConfirm(null);
      setSuccess('Route deleted successfully');
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete route');
    }
  };

  // Edit route
  const handleEdit = (route: McpRouteConfig) => {
    setEditingRoute(route);
    setShowWizard(true);
  };

  // Create new route
  const handleCreate = () => {
    setEditingRoute(null);
    setShowWizard(true);
  };

  if (loading) {
    return (
      <>
        <div className="modal-header">
          <h3>MCP Routes</h3>
          {onClose && <button type="button" className="close-btn" onClick={onClose}>✕</button>}
        </div>
        <div className="modal-body">
          <p className="muted">Loading routes...</p>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="modal-header">
        <h3>MCP Routes</h3>
        {onClose && <button type="button" className="close-btn" onClick={onClose}>✕</button>}
      </div>

      <div className="modal-body">
        <p className="field-help" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
          Create custom MCP endpoints that expose specific subsets of tools.
          The default <code>/mcp</code> route always exposes all enabled tools.
          Custom routes let you create focused tool configurations for different use cases.
        </p>

        {error && <div className="error-banner">{error}</div>}
        {success && <div className="success-banner">{success}</div>}

        {showWizard ? (
          <RouteWizard
            editingRoute={editingRoute}
            tools={otherTools}
            documentIndexes={documentIndexes}
            filesystemTools={filesystemTools}
            schemaTools={schemaTools}
            aggregateSearch={settings?.aggregate_search ?? true}
            onSave={handleSave}
            onCancel={() => {
              setShowWizard(false);
              setEditingRoute(null);
            }}
            saving={saving}
          />
        ) : (
          <>
            <div className="panel-actions" style={{ marginBottom: '1rem' }}>
              <button className="btn" onClick={handleCreate}>
                + Create Route
              </button>
            </div>

            {routes.length === 0 ? (
              <div className="empty-state">
                <p>No custom MCP routes configured.</p>
                <p className="muted">
                  Create a route to expose a specific subset of tools at a custom endpoint.
                </p>
              </div>
            ) : (
              <div className="tools-grid">
                {routes.map(route => (
                  <RouteCard
                    key={route.id}
                    route={route}
                    tools={tools}
                    onEdit={handleEdit}
                    onDelete={(id) => setDeleteConfirm(id)}
                    onToggle={handleToggle}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>

      {/* Delete confirmation modal */}
      {deleteConfirm && (
        <div className="modal-overlay" onClick={() => setDeleteConfirm(null)}>
          <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Delete</h3>
              <button type="button" className="close-btn" onClick={() => setDeleteConfirm(null)}>✕</button>
            </div>
            <div className="modal-body">
              <p>Are you sure you want to delete this MCP route?</p>
              <p className="muted">This action cannot be undone.</p>
            </div>
            <div className="modal-footer">
              <button
                className="btn btn-secondary"
                onClick={() => setDeleteConfirm(null)}
              >
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => handleDelete(deleteConfirm)}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
