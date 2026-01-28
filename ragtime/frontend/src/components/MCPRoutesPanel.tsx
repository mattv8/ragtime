import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api';
import type {
  McpRouteConfig,
  CreateMcpRouteRequest,
  UpdateMcpRouteRequest,
  McpDefaultRouteFilter,
  CreateMcpDefaultRouteFilterRequest,
  UpdateMcpDefaultRouteFilterRequest,
  ToolConfig,
  IndexInfo,
  AppSettings,
} from '@/types';
import { Icon } from './Icon';
import { DeleteConfirmButton } from './DeleteConfirmButton';

type PanelTab = 'custom-routes' | 'default-filters';

interface MCPRoutesPanelProps {
  onClose?: () => void;
  ldapConfigured?: boolean;
  ldapGroups?: { dn: string; name: string }[];
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
        <div className="tool-card-icon"><Icon name="plug" size={24} /></div>
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
        {route.require_auth && route.auth_method === 'oauth2' && <span className="write-enabled">OAuth2 (LDAP)</span>}
        {route.require_auth && route.auth_method === 'password' && route.has_password && <span className="write-enabled">Password protected</span>}
        {route.require_auth && route.auth_method === 'password' && !route.has_password && <span style={{ color: 'var(--color-warning, #f59e0b)' }}>Auth enabled (no password set)</span>}
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
        <DeleteConfirmButton
          onDelete={() => onDelete(route.id)}
          buttonText="Delete"
          className="btn btn-sm btn-danger"
        />
      </div>
    </div>
  );
}

// Default filter card component
interface DefaultFilterCardProps {
  filter: McpDefaultRouteFilter;
  tools: ToolConfig[];
  ldapGroups: { dn: string; name: string }[];
  onEdit: (filter: McpDefaultRouteFilter) => void;
  onDelete: (filterId: string) => void;
  onToggle: (filterId: string, enabled: boolean) => void;
}

function DefaultFilterCard({ filter, tools, ldapGroups, onEdit, onDelete, onToggle }: DefaultFilterCardProps) {
  const selectedTools = tools.filter(t => filter.tool_config_ids.includes(t.id));
  const groupName = ldapGroups.find(g => g.dn === filter.ldap_group_dn)?.name || filter.ldap_group_dn;

  return (
    <div className={`tool-card ${!filter.enabled ? 'disabled' : ''}`}>
      <div className="tool-card-header">
        <div className="tool-card-icon"><Icon name="users" size={24} /></div>
        <div className="tool-card-title">
          <h3>{filter.name}</h3>
          <span className="tool-card-type">Priority: {filter.priority}</span>
        </div>
        <div className="tool-card-status">
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={filter.enabled}
              onChange={(e) => onToggle(filter.id, e.target.checked)}
            />
            <span className="toggle-slider"></span>
          </label>
        </div>
      </div>

      <p className="tool-card-description">
        <strong>LDAP Group:</strong> {groupName}
      </p>

      {filter.description && (
        <p className="tool-card-description">{filter.description}</p>
      )}

      <div className="tool-card-meta">
        {filter.include_knowledge_search && <span>Knowledge search</span>}
        {filter.include_git_history && <span>Git history</span>}
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
          onClick={() => onEdit(filter)}
        >
          Edit
        </button>
        <DeleteConfirmButton
          onDelete={() => onDelete(filter.id)}
          buttonText="Delete"
          className="btn btn-sm btn-danger"
        />
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
  ldapConfigured: boolean;
  ldapGroups: { dn: string; name: string }[];
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
  ldapConfigured,
  ldapGroups,
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
  // Auth method: password or oauth2 (LDAP)
  const [authMethod, setAuthMethod] = useState<'password' | 'oauth2'>(editingRoute?.auth_method || 'password');
  // Allowed LDAP group for OAuth2 auth
  const [allowedLdapGroup, setAllowedLdapGroup] = useState(editingRoute?.allowed_ldap_group || '');

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

    // Validate password if auth is required and using password method
    if (requireAuth && authMethod === 'password' && !editingRoute?.has_password && !authPassword && !clearPassword) {
      setError('Password is required when password authentication is enabled');
      return;
    }

    if (authMethod === 'password' && authPassword && authPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    try {
      if (editingRoute) {
        const updateData: UpdateMcpRouteRequest = {
          name: name.trim(),
          description: description.trim(),
          require_auth: requireAuth,
          auth_method: authMethod,
          allowed_ldap_group: allowedLdapGroup.trim() || undefined,
          clear_allowed_ldap_group: !allowedLdapGroup.trim() && !!editingRoute.allowed_ldap_group,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        if (authMethod === 'password' && authPassword) {
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
          auth_method: authMethod,
          allowed_ldap_group: allowedLdapGroup.trim() || undefined,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        if (authMethod === 'password' && authPassword) {
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
              <span>Require authentication</span>
            </label>
            <p className="field-help">
              When enabled, clients must authenticate to access this route.
            </p>
          </div>

          {/* Auth method selection - only show when LDAP is configured and auth is enabled */}
          {requireAuth && ldapConfigured && (
            <div className="form-group">
              <label>Authentication Method</label>
              <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
                <label className="radio-label">
                  <input
                    type="radio"
                    name="route_auth_method"
                    value="password"
                    checked={authMethod === 'password'}
                    onChange={() => setAuthMethod('password')}
                  />
                  <span>Password</span>
                </label>
                <label className="radio-label">
                  <input
                    type="radio"
                    name="route_auth_method"
                    value="oauth2"
                    checked={authMethod === 'oauth2'}
                    onChange={() => setAuthMethod('oauth2')}
                  />
                  <span>OAuth2 (LDAP)</span>
                </label>
              </div>
              <p className="field-help">
                {authMethod === 'oauth2'
                  ? 'MCP clients authenticate with LDAP credentials via POST /auth/oauth2/token to get a Bearer token.'
                  : 'MCP clients use a static password as the Bearer token or MCP-Password header.'}
              </p>
            </div>
          )}

          {/* LDAP Group restriction - only for OAuth2 auth method */}
          {requireAuth && ldapConfigured && authMethod === 'oauth2' && (
            <div className="form-group">
              <label>Allowed LDAP Group (Optional)</label>
              <select
                value={allowedLdapGroup}
                onChange={(e) => setAllowedLdapGroup(e.target.value)}
              >
                <option value="">Any authenticated LDAP user</option>
                {ldapGroups.map((g) => (
                  <option key={g.dn} value={g.dn}>{g.name}</option>
                ))}
              </select>
              <p className="field-help">
                Restrict access to members of a specific LDAP group. Leave empty to allow all authenticated LDAP users.
              </p>
            </div>
          )}

          {/* Password field - only for password auth method */}
          {requireAuth && authMethod === 'password' && (
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
                    required={requireAuth && authMethod === 'password' && !editingRoute?.has_password}
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

// Default filter wizard/form
interface DefaultFilterWizardProps {
  editingFilter: McpDefaultRouteFilter | null;
  tools: ToolConfig[];
  documentIndexes: IndexInfo[];
  filesystemTools: ToolConfig[];
  schemaTools: ToolConfig[];
  aggregateSearch: boolean;
  ldapGroups: { dn: string; name: string }[];
  onSave: (data: CreateMcpDefaultRouteFilterRequest | UpdateMcpDefaultRouteFilterRequest, filterId?: string) => Promise<void>;
  onCancel: () => void;
  saving: boolean;
}

function DefaultFilterWizard({
  editingFilter,
  tools,
  documentIndexes,
  filesystemTools,
  schemaTools,
  aggregateSearch,
  ldapGroups,
  onSave,
  onCancel,
  saving,
}: DefaultFilterWizardProps) {
  const [name, setName] = useState(editingFilter?.name || '');
  const [description, setDescription] = useState(editingFilter?.description || '');
  const [priority, setPriority] = useState(editingFilter?.priority ?? 0);
  const [ldapGroupDn, setLdapGroupDn] = useState(editingFilter?.ldap_group_dn || '');

  // Document index selection
  const [includeKnowledgeSearch, setIncludeKnowledgeSearch] = useState(editingFilter?.include_knowledge_search ?? false);
  const [selectedDocIndexes, setSelectedDocIndexes] = useState<Set<string>>(
    new Set(editingFilter?.selected_document_indexes || [])
  );

  // Git history selection
  const [includeGitHistory, setIncludeGitHistory] = useState(editingFilter?.include_git_history ?? false);

  // Filesystem index selection
  const [selectedFilesystemTools, setSelectedFilesystemTools] = useState<Set<string>>(
    new Set(editingFilter?.selected_filesystem_indexes || [])
  );

  // Schema index selection
  const [selectedSchemaTools, setSelectedSchemaTools] = useState<Set<string>>(
    new Set(editingFilter?.selected_schema_indexes || [])
  );

  // Database/shell tool selection
  const [selectedToolIds, setSelectedToolIds] = useState<Set<string>>(
    new Set(editingFilter?.tool_config_ids || [])
  );

  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    if (!ldapGroupDn) {
      setError('LDAP group is required');
      return;
    }

    try {
      if (editingFilter) {
        const updateData: UpdateMcpDefaultRouteFilterRequest = {
          name: name.trim(),
          description: description.trim(),
          priority,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        await onSave(updateData, editingFilter.id);
      } else {
        const createData: CreateMcpDefaultRouteFilterRequest = {
          name: name.trim(),
          ldap_group_dn: ldapGroupDn,
          description: description.trim(),
          priority,
          include_knowledge_search: includeKnowledgeSearch,
          include_git_history: includeGitHistory,
          selected_document_indexes: Array.from(selectedDocIndexes),
          selected_filesystem_indexes: Array.from(selectedFilesystemTools),
          selected_schema_indexes: Array.from(selectedSchemaTools),
          tool_config_ids: Array.from(selectedToolIds),
        };
        await onSave(createData);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save filter');
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
        <h3>{editingFilter ? 'Edit Default Route Filter' : 'Create Default Route Filter'}</h3>
      </div>

      <form onSubmit={handleSubmit} className="wizard-form">
        {error && <div className="error-banner">{error}</div>}

        <p className="field-help" style={{ marginBottom: '1rem', padding: '0.75rem', backgroundColor: 'var(--color-surface)', borderRadius: '4px' }}>
          Default route filters control which tools are shown to users on the <code>/mcp</code> endpoint
          based on their LDAP group membership. When OAuth2 authentication is enabled on the default route,
          users will only see tools matching their group.
        </p>

        <div className="form-group">
          <label>Filter Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Engineering Tools"
            required
          />
          <p className="field-help">A friendly name for this filter configuration.</p>
        </div>

        <div className="form-group">
          <label>LDAP Group</label>
          <select
            value={ldapGroupDn}
            onChange={(e) => setLdapGroupDn(e.target.value)}
            disabled={!!editingFilter}
            required
          >
            <option value="">Select an LDAP group...</option>
            {ldapGroups.map((g) => (
              <option key={g.dn} value={g.dn}>{g.name}</option>
            ))}
          </select>
          <p className="field-help">
            {editingFilter
              ? 'LDAP group cannot be changed after creation.'
              : 'Users in this LDAP group will see only the tools configured below.'}
          </p>
        </div>

        <div className="form-group">
          <label>Priority</label>
          <input
            type="number"
            value={priority}
            onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
            min={0}
            max={100}
          />
          <p className="field-help">
            Higher priority filters take precedence when a user is in multiple groups (0-100).
          </p>
        </div>

        <div className="form-group">
          <label>Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Tools for engineering team..."
            rows={2}
          />
          <p className="field-help">Optional description for documentation purposes.</p>
        </div>

        {/* Document Indexes Section */}
        <fieldset>
          <legend>Document Indexes</legend>
          <p className="fieldset-help">
            {aggregateSearch
              ? 'Enable the combined knowledge search tool for all document indexes.'
              : 'Select which document indexes to expose.'}
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
                    </label>
                  ))}
                </div>
              )}
            </>
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
          </div>
        </fieldset>

        {/* Filesystem Indexes Section */}
        {filesystemTools.length > 0 && (
          <fieldset>
            <legend>Filesystem Indexes</legend>
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
                </label>
              ))}
            </div>
          </fieldset>
        )}

        {/* Schema Indexes Section */}
        {schemaTools.length > 0 && (
          <fieldset>
            <legend>Schema Indexes</legend>
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
                </label>
              ))}
            </div>
          </fieldset>
        )}

        {/* Tools Section */}
        <fieldset>
          <legend>Tools</legend>
          <p className="fieldset-help">
            Select which database and shell tools to expose to this group.
          </p>

          {tools.length === 0 ? (
            <p className="muted">No tools configured.</p>
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
            {saving ? 'Saving...' : editingFilter ? 'Update Filter' : 'Create Filter'}
          </button>
        </div>
      </form>
    </div>
  );
}

// Main panel
export function MCPRoutesPanel({ onClose, ldapConfigured = false, ldapGroups = [] }: MCPRoutesPanelProps) {
  const [activeTab, setActiveTab] = useState<PanelTab>('custom-routes');
  const [routes, setRoutes] = useState<McpRouteConfig[]>([]);
  const [filters, setFilters] = useState<McpDefaultRouteFilter[]>([]);
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [documentIndexes, setDocumentIndexes] = useState<IndexInfo[]>([]);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Route wizard state
  const [showRouteWizard, setShowRouteWizard] = useState(false);
  const [editingRoute, setEditingRoute] = useState<McpRouteConfig | null>(null);
  const [savingRoute, setSavingRoute] = useState(false);

  // Filter wizard state
  const [showFilterWizard, setShowFilterWizard] = useState(false);
  const [editingFilter, setEditingFilter] = useState<McpDefaultRouteFilter | null>(null);
  const [savingFilter, setSavingFilter] = useState(false);

  // Categorize tools
  const filesystemTools = tools.filter(t => t.tool_type === 'filesystem_indexer');
  const schemaTools = tools.filter(t => {
    if (!['postgres', 'mssql', 'mysql'].includes(t.tool_type)) return false;
    const config = t.connection_config as { schema_index_enabled?: boolean } | undefined;
    return config?.schema_index_enabled === true;
  });
  const otherTools = tools.filter(t =>
    t.tool_type !== 'filesystem_indexer'
  );

  // Load routes, filters, tools, indexes, and settings
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [routesRes, filtersRes, toolsRes, indexesRes, settingsRes] = await Promise.all([
        api.listMcpRoutes(),
        api.listMcpDefaultFilters(),
        api.listToolConfigs(),
        api.listIndexes(),
        api.getSettings(),
      ]);
      setRoutes(routesRes.routes);
      setFilters(filtersRes.filters);
      setTools(toolsRes);
      setDocumentIndexes(indexesRes);
      setSettings(settingsRes.settings);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load MCP configuration');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Route handlers
  const handleSaveRoute = async (data: CreateMcpRouteRequest | UpdateMcpRouteRequest, routeId?: string) => {
    setSavingRoute(true);
    try {
      if (routeId) {
        await api.updateMcpRoute(routeId, data as UpdateMcpRouteRequest);
        setSuccess('Route updated successfully');
      } else {
        await api.createMcpRoute(data as CreateMcpRouteRequest);
        setSuccess('Route created successfully');
      }
      setShowRouteWizard(false);
      setEditingRoute(null);
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } finally {
      setSavingRoute(false);
    }
  };

  const handleToggleRoute = async (routeId: string, enabled: boolean) => {
    try {
      await api.toggleMcpRoute(routeId, enabled);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle route');
    }
  };

  const handleDeleteRoute = async (routeId: string) => {
    try {
      await api.deleteMcpRoute(routeId);
      setSuccess('Route deleted successfully');
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete route');
    }
  };

  const handleEditRoute = (route: McpRouteConfig) => {
    setEditingRoute(route);
    setShowRouteWizard(true);
  };

  const handleCreateRoute = () => {
    setEditingRoute(null);
    setShowRouteWizard(true);
  };

  // Filter handlers
  const handleSaveFilter = async (data: CreateMcpDefaultRouteFilterRequest | UpdateMcpDefaultRouteFilterRequest, filterId?: string) => {
    setSavingFilter(true);
    try {
      if (filterId) {
        await api.updateMcpDefaultFilter(filterId, data as UpdateMcpDefaultRouteFilterRequest);
        setSuccess('Filter updated successfully');
      } else {
        await api.createMcpDefaultFilter(data as CreateMcpDefaultRouteFilterRequest);
        setSuccess('Filter created successfully');
      }
      setShowFilterWizard(false);
      setEditingFilter(null);
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } finally {
      setSavingFilter(false);
    }
  };

  const handleToggleFilter = async (filterId: string, enabled: boolean) => {
    try {
      await api.toggleMcpDefaultFilter(filterId, enabled);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle filter');
    }
  };

  const handleDeleteFilter = async (filterId: string) => {
    try {
      await api.deleteMcpDefaultFilter(filterId);
      setSuccess('Filter deleted successfully');
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete filter');
    }
  };

  const handleEditFilter = (filter: McpDefaultRouteFilter) => {
    setEditingFilter(filter);
    setShowFilterWizard(true);
  };

  const handleCreateFilter = () => {
    setEditingFilter(null);
    setShowFilterWizard(true);
  };

  const handleClose = () => {
    if (showRouteWizard) {
      setShowRouteWizard(false);
      setEditingRoute(null);
    } else if (showFilterWizard) {
      setShowFilterWizard(false);
      setEditingFilter(null);
    } else {
      onClose?.();
    }
  };

  if (loading) {
    return (
      <>
        <div className="modal-header">
          <h3>MCP Routes</h3>
          {onClose && <button type="button" className="close-btn" onClick={onClose}><Icon name="close" size={18} /></button>}
        </div>
        <div className="modal-body">
          <p className="muted">Loading configuration...</p>
        </div>
      </>
    );
  }

  const showWizard = showRouteWizard || showFilterWizard;
  // Only show default route filters tab when OAuth2 is configured on the default route
  const showDefaultFiltersTab = ldapConfigured && settings?.mcp_default_route_auth && settings?.mcp_default_route_auth_method === 'oauth2';

  return (
    <>
      <div className="modal-header">
        {/* Tab navigation in header when not in wizard mode */}
        {!showWizard && showDefaultFiltersTab ? (
          <div className="tab-navigation" style={{ display: 'flex', gap: '0.5rem', flex: 1 }}>
            <button
              type="button"
              className={`tab-button ${activeTab === 'custom-routes' ? 'active' : ''}`}
              onClick={() => setActiveTab('custom-routes')}
              style={{
                padding: '0.5rem 1rem',
                border: 'none',
                background: 'none',
                cursor: 'pointer',
                borderBottom: activeTab === 'custom-routes' ? '2px solid var(--color-primary)' : '2px solid transparent',
                color: activeTab === 'custom-routes' ? 'var(--color-primary)' : 'var(--color-text-muted)',
                fontWeight: activeTab === 'custom-routes' ? 600 : 400,
                fontSize: '1rem',
              }}
            >
              Custom Routes
            </button>
            <button
              type="button"
              className={`tab-button ${activeTab === 'default-filters' ? 'active' : ''}`}
              onClick={() => setActiveTab('default-filters')}
              style={{
                padding: '0.5rem 1rem',
                border: 'none',
                background: 'none',
                cursor: 'pointer',
                borderBottom: activeTab === 'default-filters' ? '2px solid var(--color-primary)' : '2px solid transparent',
                color: activeTab === 'default-filters' ? 'var(--color-primary)' : 'var(--color-text-muted)',
                fontWeight: activeTab === 'default-filters' ? 600 : 400,
                fontSize: '1rem',
              }}
            >
              Default Route Filters
            </button>
          </div>
        ) : (
          <h3>MCP Routes</h3>
        )}
        {onClose && <button type="button" className="close-btn" onClick={handleClose}><Icon name="close" size={18} /></button>}
      </div>

      <div className="modal-body">
        {error && <div className="error-banner">{error}</div>}
        {success && <div className="success-banner">{success}</div>}

          <>
            {activeTab === 'custom-routes' && (
              <>
                <p className="field-help" style={{ marginBottom: '1rem' }}>
                  Create custom MCP endpoints that expose specific subsets of tools at custom paths like <code>/mcp/my_tools</code>.
                </p>


                {!showRouteWizard && (
                  <div className="panel-actions" style={{ marginBottom: '1rem' }}>
                    <button className="btn" onClick={handleCreateRoute}>
                      + Add Custom Route
                    </button>
                  </div>
                )}

        {showRouteWizard ? (
          <RouteWizard
            editingRoute={editingRoute}
            tools={otherTools}
            documentIndexes={documentIndexes}
            filesystemTools={filesystemTools}
            schemaTools={schemaTools}
            aggregateSearch={settings?.aggregate_search ?? true}
            ldapConfigured={ldapConfigured}
            ldapGroups={ldapGroups}
            onSave={handleSaveRoute}
            onCancel={() => {
              setShowRouteWizard(false);
              setEditingRoute(null);
            }}
            saving={savingRoute}
          />
        ) : (
          <>

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
                        onEdit={handleEditRoute}
                        onDelete={handleDeleteRoute}
                        onToggle={handleToggleRoute}
                      />
                    ))}
                  </div>
                )}
                </>
                )}
              </>
            )}

            {activeTab === 'default-filters' && showDefaultFiltersTab && (
              <>
                <p className="field-help" style={{ marginBottom: '1rem' }}>
                  Configure LDAP group-based tool filtering for the default <code>/mcp</code> route.
                  Users will only see tools matching their LDAP group membership. Higher priority filters take precedence.
                </p>


                {!showFilterWizard && (
                  <div className="panel-actions" style={{ marginBottom: '1rem' }}>
                    <button className="btn" onClick={handleCreateFilter}>
                      + Add Group Filter
                    </button>
                  </div>
                )}

        {showFilterWizard ? (
          <DefaultFilterWizard
            editingFilter={editingFilter}
            tools={otherTools}
            documentIndexes={documentIndexes}
            filesystemTools={filesystemTools}
            schemaTools={schemaTools}
            aggregateSearch={settings?.aggregate_search ?? true}
            ldapGroups={ldapGroups}
            onSave={handleSaveFilter}
            onCancel={() => {
              setShowFilterWizard(false);
              setEditingFilter(null);
            }}
            saving={savingFilter}
          />
        ) : (
          <>

                {filters.length === 0 ? (
                  <div className="empty-state">
                    <p>No default route filters configured.</p>
                    <p className="muted">
                      Create a filter to restrict tool visibility based on LDAP group membership.
                      Without filters, all users see all tools on the default route.
                    </p>
                  </div>
                ) : (
                  <div className="tools-grid">
                    {filters.map(filter => (
                      <DefaultFilterCard
                        key={filter.id}
                        filter={filter}
                        tools={tools}
                        ldapGroups={ldapGroups}
                        onEdit={handleEditFilter}
                        onDelete={handleDeleteFilter}
                        onToggle={handleToggleFilter}
                      />
                    ))}
                  </div>
                )}
                </>
                )}
              </>
            )}
          </>
      </div>
    </>
  );
}
