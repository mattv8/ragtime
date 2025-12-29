import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/api';
import type { ToolConfig, HeartbeatStatus } from '@/types';
import { TOOL_TYPE_INFO } from '@/types';
import { ToolWizard } from './ToolWizard';

// Confirmation modal state
interface ConfirmationState {
  message: string;
  onConfirm: () => void;
  onCancel?: () => void;
}

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
}

function ToolCard({ tool, heartbeat, onEdit, onDelete, onToggle, onTest, testing }: ToolCardProps) {
  const typeInfo = TOOL_TYPE_INFO[tool.tool_type];

  const getConnectionSummary = (): string => {
    const config = tool.connection_config;
    switch (tool.tool_type) {
      case 'postgres':
        if ('host' in config && config.host) {
          return `${config.host}:${config.port || 5432}/${config.database || ''}`;
        }
        return `Container: ${config.container || 'N/A'}`;
      case 'odoo_shell':
        return `Container: ${'container' in config ? config.container : 'N/A'}`;
      case 'ssh_shell':
        if ('host' in config && 'user' in config) {
          return `${config.user}@${config.host}:${config.port || 22}`;
        }
        return 'SSH connection';
      default:
        return 'Unknown';
    }
  };

  // Determine heartbeat display status
  const getHeartbeatDisplay = () => {
    if (!tool.enabled) {
      return { status: 'disabled', label: 'Disabled', icon: '○' };
    }
    if (!heartbeat) {
      return { status: 'checking', label: 'Checking...', icon: '◌' };
    }
    if (heartbeat.alive) {
      const latency = heartbeat.latency_ms ? `${Math.round(heartbeat.latency_ms)}ms` : '';
      return { status: 'alive', label: latency || 'Connected', icon: '●' };
    }
    return { status: 'dead', label: heartbeat.error || 'Disconnected', icon: '●' };
  };

  const heartbeatDisplay = getHeartbeatDisplay();

  return (
    <div className={`tool-card ${!tool.enabled ? 'disabled' : ''}`}>
      <div className="tool-card-header">
        <div className="tool-card-icon">{typeInfo?.icon === 'database' ? '🗄️' : typeInfo?.icon === 'terminal' ? '💻' : '🖥️'}</div>
        <div className="tool-card-title">
          <h3>{tool.name}</h3>
          <span className="tool-card-type">{typeInfo?.name || tool.tool_type}</span>
        </div>
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

      {tool.description && (
        <p className="tool-card-description">{tool.description}</p>
      )}

      <div className="tool-card-connection">
        <span className="connection-label">Connection:</span>
        <code>{getConnectionSummary()}</code>
      </div>

      {/* Show heartbeat error if connection failed */}
      {heartbeat && !heartbeat.alive && tool.enabled && (
        <div className="tool-card-heartbeat-error">
          <span className="error-icon">✗</span>
          <span>{heartbeat.error || 'Connection failed'}</span>
        </div>
      )}

      {/* Show traditional test result only if no heartbeat available and tool is enabled */}
      {!heartbeat && tool.enabled && tool.last_test_at && (
        <div className={`tool-card-test-result ${tool.last_test_result ? 'success' : 'error'}`}>
          <span className="test-icon">{tool.last_test_result ? '✓' : '✗'}</span>
          <span>
            {tool.last_test_result ? 'Connection OK' : tool.last_test_error || 'Connection failed'}
          </span>
          <span className="test-time">
            {new Date(tool.last_test_at).toLocaleString()}
          </span>
        </div>
      )}

      <div className="tool-card-meta">
        <span>Timeout: {tool.timeout}s</span>
        <span>Max results: {tool.max_results}</span>
        {tool.allow_write && <span className="write-enabled">Write enabled</span>}
      </div>

      <div className="tool-card-actions">
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onTest(tool.id)}
          disabled={testing}
        >
          {testing ? 'Testing...' : 'Test'}
        </button>
        <button
          type="button"
          className="btn btn-sm"
          onClick={() => onEdit(tool)}
        >
          Edit
        </button>
        <button
          type="button"
          className="btn btn-sm btn-danger"
          onClick={() => onDelete(tool.id)}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

export function ToolsPanel() {
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showWizard, setShowWizard] = useState(false);
  const [editingTool, setEditingTool] = useState<ToolConfig | null>(null);
  const [testingToolId, setTestingToolId] = useState<string | null>(null);
  const [heartbeats, setHeartbeats] = useState<Record<string, HeartbeatStatus>>({});
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [confirmation, setConfirmation] = useState<ConfirmationState | null>(null);

  const loadTools = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.listToolConfigs();
      setTools(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tools');
    } finally {
      setLoading(false);
    }
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
    setConfirmation({
      message: 'Are you sure you want to delete this tool configuration?',
      onConfirm: async () => {
        setConfirmation(null);
        try {
          await api.deleteToolConfig(toolId);
          await loadTools();
          setSuccess('Tool deleted successfully');
          setTimeout(() => setSuccess(null), 3000);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to delete tool');
        }
      },
      onCancel: () => setConfirmation(null)
    });
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

  if (showWizard) {
    return (
      <ToolWizard
        existingTool={editingTool}
        onClose={handleWizardClose}
        onSave={handleWizardSave}
      />
    );
  }

  return (
    <div className="tools-panel">
      {/* Confirmation Modal */}
      {confirmation && (
        <div className="modal-overlay" onClick={() => confirmation.onCancel?.()}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Action</h3>
              <button className="modal-close" onClick={() => confirmation.onCancel?.()}>×</button>
            </div>
            <div className="modal-body">
              <p>{confirmation.message}</p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => confirmation.onCancel?.()}>
                Cancel
              </button>
              <button className="btn btn-danger" onClick={confirmation.onConfirm}>
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h2>Tool Connections</h2>
          <button type="button" className="btn" onClick={handleAddTool}>
            Add Tool
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}
        {success && <div className="success-banner">{success}</div>}

        <p className="fieldset-help">
          Configure connections to databases, shells, and other tools that the AI can use during conversations.
          Each tool can have multiple instances (e.g., production and staging databases).
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
          <div className="tools-grid">
            {tools.map((tool) => (
              <ToolCard
                key={tool.id}
                tool={tool}
                heartbeat={heartbeats[tool.id] || null}
                onEdit={handleEditTool}
                onDelete={handleDeleteTool}
                onToggle={handleToggleTool}
                onTest={handleTestTool}
                testing={testingToolId === tool.id}
              />
            ))}
          </div>
        )}
      </div>

      {tools.length === 0 && (
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
                  {info.icon === 'database' ? '🗄️' : info.icon === 'terminal' ? '💻' : '🖥️'}
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
