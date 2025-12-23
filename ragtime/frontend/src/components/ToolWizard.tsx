import { useState, useEffect, useRef } from 'react';
import { api, DockerContainer, DockerNetwork } from '@/api';
import type {
  ToolConfig,
  ToolType,
  CreateToolConfigRequest,
  PostgresConnectionConfig,
  OdooShellConnectionConfig,
  SSHShellConnectionConfig,
  ConnectionConfig,
} from '@/types';
import { TOOL_TYPE_INFO } from '@/types';

interface ToolWizardProps {
  existingTool: ToolConfig | null;
  onClose: () => void;
  onSave: () => void;
}

type WizardStep = 'type' | 'connection' | 'description' | 'options' | 'review';

const WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'description', 'options', 'review'];

function getStepTitle(step: WizardStep): string {
  switch (step) {
    case 'type':
      return 'Select Tool Type';
    case 'connection':
      return 'Configure Connection';
    case 'description':
      return 'Add Description';
    case 'options':
      return 'Set Options';
    case 'review':
      return 'Review & Save';
  }
}

export function ToolWizard({ existingTool, onClose, onSave }: ToolWizardProps) {
  const isEditing = existingTool !== null;
  const progressRef = useRef<HTMLDivElement>(null);

  // Wizard state
  const [currentStep, setCurrentStep] = useState<WizardStep>(isEditing ? 'connection' : 'type');
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [toolType, setToolType] = useState<ToolType>(existingTool?.tool_type || 'postgres');

  // Auto-scroll active step into view
  useEffect(() => {
    const activeStep = progressRef.current?.querySelector('.wizard-step.active');
    if (activeStep) {
      activeStep.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentStep]);
  const [name, setName] = useState(existingTool?.name || '');
  const [description, setDescription] = useState(existingTool?.description || '');
  const [maxResults, setMaxResults] = useState(existingTool?.max_results || 100);
  const [timeout, setTimeout] = useState(existingTool?.timeout || 30);
  const [allowWrite, setAllowWrite] = useState(existingTool?.allow_write || false);

  // Connection config state
  const [pgConnectionMode, setPgConnectionMode] = useState<'direct' | 'container'>(
    existingTool?.tool_type === 'postgres' && (existingTool.connection_config as PostgresConnectionConfig).container
      ? 'container'
      : 'direct'
  );
  const [postgresConfig, setPostgresConfig] = useState<PostgresConnectionConfig>(
    existingTool?.tool_type === 'postgres'
      ? (existingTool.connection_config as PostgresConnectionConfig)
      : { host: '', port: 5432, user: '', password: '', database: '', container: '' }
  );

  const [odooConfig, setOdooConfig] = useState<OdooShellConnectionConfig>(
    existingTool?.tool_type === 'odoo_shell'
      ? (existingTool.connection_config as OdooShellConnectionConfig)
      : { container: '', database: 'odoo', docker_network: '', config_path: '' }
  );

  // Docker discovery state
  const [dockerContainers, setDockerContainers] = useState<DockerContainer[]>([]);
  const [dockerNetworks, setDockerNetworks] = useState<DockerNetwork[]>([]);
  const [currentNetwork, setCurrentNetwork] = useState<string | null>(null);
  const [loadingDocker, setLoadingDocker] = useState(false);
  const [connectingNetwork, setConnectingNetwork] = useState(false);

  const [sshConfig, setSshConfig] = useState<SSHShellConnectionConfig>(
    existingTool?.tool_type === 'ssh_shell'
      ? (existingTool.connection_config as SSHShellConnectionConfig)
      : { host: '', port: 22, user: '', key_path: '', password: '', command_prefix: '' }
  );

  const getConnectionConfig = (): ConnectionConfig => {
    switch (toolType) {
      case 'postgres':
        return postgresConfig;
      case 'odoo_shell':
        return odooConfig;
      case 'ssh_shell':
        return sshConfig;
    }
  };

  const getCurrentStepIndex = () => WIZARD_STEPS.indexOf(currentStep);

  const canNavigateToStep = (stepIndex: number): boolean => {
    // Can always go back to previous steps
    if (stepIndex <= getCurrentStepIndex()) return true;
    // Can only go forward one step at a time, and current step must be valid
    if (stepIndex === getCurrentStepIndex() + 1 && canProceed()) return true;
    return false;
  };

  const goToStep = (step: WizardStep) => {
    const stepIndex = WIZARD_STEPS.indexOf(step);
    if (canNavigateToStep(stepIndex)) {
      setCurrentStep(step);
      setTestResult(null);
      setError(null);
    }
  };

  const goToNextStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex < WIZARD_STEPS.length - 1) {
      setCurrentStep(WIZARD_STEPS[currentIndex + 1]);
      setTestResult(null);
      setError(null);
    }
  };

  const goToPreviousStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex > 0) {
      // Skip type step if editing
      const prevIndex = currentIndex - 1;
      if (isEditing && WIZARD_STEPS[prevIndex] === 'type') {
        return;
      }
      setCurrentStep(WIZARD_STEPS[prevIndex]);
      setTestResult(null);
      setError(null);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setError(null);

    try {
      const result = await api.testToolConnection({
        tool_type: toolType,
        connection_config: getConnectionConfig(),
      });
      setTestResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      if (isEditing && existingTool) {
        await api.updateToolConfig(existingTool.id, {
          name,
          description,
          connection_config: getConnectionConfig(),
          max_results: maxResults,
          timeout,
          allow_write: allowWrite,
        });
      } else {
        const request: CreateToolConfigRequest = {
          name,
          tool_type: toolType,
          description,
          connection_config: getConnectionConfig(),
          max_results: maxResults,
          timeout,
          allow_write: allowWrite,
        };
        await api.createToolConfig(request);
      }
      onSave();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
      setSaving(false);
    }
  };

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'type':
        return true;
      case 'connection':
        return validateConnection();
      case 'description':
        return name.trim().length > 0;
      case 'options':
        return true;
      case 'review':
        return true;
    }
  };

  const validateConnection = (): boolean => {
    switch (toolType) {
      case 'postgres':
        // Either host or container must be specified
        return Boolean((postgresConfig.host && postgresConfig.user) || postgresConfig.container);
      case 'odoo_shell':
        return Boolean(odooConfig.container);
      case 'ssh_shell':
        return Boolean(sshConfig.host && sshConfig.user);
    }
  };

  const renderTypeSelection = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Select the type of tool connection you want to add:
      </p>
      <div className="tool-type-selection">
        {(Object.entries(TOOL_TYPE_INFO) as [ToolType, typeof TOOL_TYPE_INFO[ToolType]][]).map(
          ([type, info]) => (
            <button
              key={type}
              type="button"
              className={`tool-type-option ${toolType === type ? 'selected' : ''}`}
              onClick={() => setToolType(type)}
            >
              <span className="tool-type-option-icon">
                {info.icon === 'database' ? '🗄️' : info.icon === 'terminal' ? '💻' : '🖥️'}
              </span>
              <span className="tool-type-option-name">{info.name}</span>
              <span className="tool-type-option-desc">{info.description}</span>
            </button>
          )
        )}
      </div>
    </div>
  );

  const renderPostgresConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Choose your connection method:
        </p>

        <div className="connection-tabs">
          <button
            type="button"
            className={`connection-tab ${pgConnectionMode === 'direct' ? 'active' : ''}`}
            onClick={() => {
              setPgConnectionMode('direct');
              setPostgresConfig({ ...postgresConfig, container: '' });
            }}
          >
            Direct Connection
          </button>
          <button
            type="button"
            className={`connection-tab ${pgConnectionMode === 'container' ? 'active' : ''}`}
            onClick={() => {
              setPgConnectionMode('container');
              setPostgresConfig({ ...postgresConfig, host: '', user: '', password: '' });
            }}
          >
            Docker Container
          </button>
        </div>

        {pgConnectionMode === 'direct' ? (
          <div className="connection-panel">
            <div className="form-row">
              <div className="form-group">
                <label>Host</label>
                <input
                  type="text"
                  value={postgresConfig.host || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, host: e.target.value })}
                  placeholder="localhost"
                />
              </div>
              <div className="form-group form-group-small">
                <label>Port</label>
                <input
                  type="number"
                  value={postgresConfig.port || 5432}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, port: parseInt(e.target.value) || 5432 })}
                  min={1}
                  max={65535}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>User</label>
                <input
                  type="text"
                  value={postgresConfig.user || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, user: e.target.value })}
                  placeholder="postgres"
                />
              </div>
              <div className="form-group">
                <label>Password</label>
                <input
                  type="password"
                  value={postgresConfig.password || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, password: e.target.value })}
                  placeholder="********"
                />
              </div>
            </div>
            <div className="form-group">
              <label>Database</label>
              <input
                type="text"
                value={postgresConfig.database || ''}
                onChange={(e) => setPostgresConfig({ ...postgresConfig, database: e.target.value })}
                placeholder="mydb"
              />
            </div>
          </div>
        ) : pgConnectionMode === 'container' ? (
          <div className="connection-panel">
            <div className="form-group">
              <label>Container Name</label>
              <input
                type="text"
                value={postgresConfig.container || ''}
                onChange={(e) => setPostgresConfig({ ...postgresConfig, container: e.target.value })}
                placeholder="my-postgres-container"
              />
              <p className="field-help">
                Uses container's POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB environment variables.
              </p>
            </div>
            <div className="form-group">
              <label>Database (optional override)</label>
              <input
                type="text"
                value={postgresConfig.database || ''}
                onChange={(e) => setPostgresConfig({ ...postgresConfig, database: e.target.value })}
                placeholder="Leave empty to use POSTGRES_DB"
              />
            </div>
          </div>
        ) : null}
      </div>
    );
  };

  const renderOdooConnection = () => {
    const odooContainers = dockerContainers.filter(c => c.has_odoo);

    const handleDiscoverDocker = async () => {
      setLoadingDocker(true);
      try {
        const result = await api.discoverDocker();
        if (result.success) {
          setDockerContainers(result.containers);
          setDockerNetworks(result.networks);
          setCurrentNetwork(result.current_network);

          // Auto-select first Odoo container if none selected
          if (!odooConfig.container && result.containers.some(c => c.has_odoo)) {
            const firstOdoo = result.containers.find(c => c.has_odoo);
            if (firstOdoo) {
              setOdooConfig({
                ...odooConfig,
                container: firstOdoo.name,
                docker_network: firstOdoo.networks[0] || ''
              });
            }
          }
        }
      } catch (err) {
        console.error('Docker discovery failed:', err);
      } finally {
        setLoadingDocker(false);
      }
    };

    const handleConnectNetwork = async (networkName: string) => {
      setConnectingNetwork(true);
      try {
        const result = await api.connectToNetwork(networkName);
        if (result.success) {
          setCurrentNetwork(networkName);
          setOdooConfig({ ...odooConfig, docker_network: networkName });
        }
      } catch (err) {
        console.error('Network connection failed:', err);
      } finally {
        setConnectingNetwork(false);
      }
    };

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Connect to an Odoo instance running in a Docker container.
        </p>

        <div className="form-group">
          <div className="form-row" style={{ alignItems: 'flex-end' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Container Name</label>
              <input
                type="text"
                value={odooConfig.container}
                onChange={(e) => setOdooConfig({ ...odooConfig, container: e.target.value })}
                placeholder="odoo-server"
                list="odoo-containers"
              />
              <datalist id="odoo-containers">
                {odooContainers.map(c => (
                  <option key={c.name} value={c.name} />
                ))}
              </datalist>
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleDiscoverDocker}
              disabled={loadingDocker}
              style={{ marginBottom: '0.5rem' }}
            >
              {loadingDocker ? 'Scanning...' : 'Discover'}
            </button>
          </div>
          <p className="field-help">
            The Docker container running the Odoo server.
            {odooContainers.length > 0 && ` Found ${odooContainers.length} Odoo container(s).`}
          </p>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Database Name</label>
            <input
              type="text"
              value={odooConfig.database || 'odoo'}
              onChange={(e) => setOdooConfig({ ...odooConfig, database: e.target.value })}
              placeholder="odoo"
            />
            <p className="field-help">
              The Odoo database to connect to.
            </p>
          </div>
          <div className="form-group">
            <label>Config Path (optional)</label>
            <input
              type="text"
              value={odooConfig.config_path || ''}
              onChange={(e) => setOdooConfig({ ...odooConfig, config_path: e.target.value })}
              placeholder="Leave empty to use container defaults"
            />
            <p className="field-help">
              Path to odoo.conf inside the container. Leave empty if container uses environment variables.
            </p>
          </div>
        </div>

        {dockerNetworks.length > 0 && (
          <div className="form-group">
            <label>Docker Network</label>
            <div className="form-row" style={{ alignItems: 'flex-start' }}>
              <select
                value={odooConfig.docker_network || ''}
                onChange={(e) => setOdooConfig({ ...odooConfig, docker_network: e.target.value })}
                style={{ flex: 1 }}
              >
                <option value="">Select network...</option>
                {dockerNetworks.map(n => (
                  <option key={n.name} value={n.name}>
                    {n.name} ({n.containers.length} containers)
                    {n.name === currentNetwork ? ' - connected' : ''}
                  </option>
                ))}
              </select>
              {odooConfig.docker_network && odooConfig.docker_network !== currentNetwork && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => handleConnectNetwork(odooConfig.docker_network!)}
                  disabled={connectingNetwork}
                  style={{ marginLeft: '0.5rem' }}
                >
                  {connectingNetwork ? 'Connecting...' : 'Connect'}
                </button>
              )}
            </div>
            <p className="field-help">
              Connect Ragtime to the same Docker network as the Odoo container for direct access.
              {currentNetwork && ` Currently connected to: ${currentNetwork}`}
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderSSHConnection = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Connect to a remote server via SSH to run shell commands.
      </p>

      <div className="form-row">
        <div className="form-group">
          <label>Host</label>
          <input
            type="text"
            value={sshConfig.host}
            onChange={(e) => setSshConfig({ ...sshConfig, host: e.target.value })}
            placeholder="server.example.com"
          />
        </div>
        <div className="form-group form-group-small">
          <label>Port</label>
          <input
            type="number"
            value={sshConfig.port || 22}
            onChange={(e) => setSshConfig({ ...sshConfig, port: parseInt(e.target.value) || 22 })}
            min={1}
            max={65535}
          />
        </div>
      </div>

      <div className="form-group">
        <label>User</label>
        <input
          type="text"
          value={sshConfig.user}
          onChange={(e) => setSshConfig({ ...sshConfig, user: e.target.value })}
          placeholder="ubuntu"
        />
      </div>

      <div className="form-group">
        <label>SSH Key Path (optional)</label>
        <input
          type="text"
          value={sshConfig.key_path || ''}
          onChange={(e) => setSshConfig({ ...sshConfig, key_path: e.target.value })}
          placeholder="/home/user/.ssh/id_rsa"
        />
        <p className="field-help">
          Path to SSH private key. Leave empty to use default key or password.
        </p>
      </div>

      <div className="form-group">
        <label>Command Prefix (optional)</label>
        <input
          type="text"
          value={sshConfig.command_prefix || ''}
          onChange={(e) => setSshConfig({ ...sshConfig, command_prefix: e.target.value })}
          placeholder="cd /app && source venv/bin/activate && "
        />
        <p className="field-help">
          Commands to run before each tool command (e.g., activate virtualenv).
        </p>
      </div>
    </div>
  );

  const renderConnectionConfig = () => {
    const content = (() => {
      switch (toolType) {
        case 'postgres':
          return renderPostgresConnection();
        case 'odoo_shell':
          return renderOdooConnection();
        case 'ssh_shell':
          return renderSSHConnection();
      }
    })();

    return (
      <>
        {content}
        <div className="wizard-test-section">
          <button
            type="button"
            className={`btn ${testResult?.success ? 'btn-connected' : ''}`}
            onClick={handleTestConnection}
            disabled={testing || !validateConnection()}
          >
            {testing ? 'Testing...' : testResult?.success ? 'Connected' : 'Test Connection'}
          </button>
          {testResult && (
            <span className={`test-result ${testResult.success ? 'success' : 'error'}`}>
              {testResult.message}
            </span>
          )}
        </div>
      </>
    );
  };

  const renderDescription = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Give this tool a name and description. The description is provided to the AI model
        to help it understand when and how to use this tool.
      </p>

      <div className="form-group">
        <label>Name *</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., Production Database, Staging Odoo"
        />
        <p className="field-help">
          A short, descriptive name for this tool instance.
        </p>
      </div>

      <div className="form-group">
        <label>Description for AI</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what data is available through this connection and when the AI should use it. For example: 'Production PostgreSQL database containing sales orders, customer data, and inventory. Use for real-time business queries.'"
          rows={4}
        />
        <p className="field-help">
          This description is included in the system prompt to help the AI understand
          what this tool is for and when to use it. Be specific about the data available.
        </p>
      </div>
    </div>
  );

  const renderOptions = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Configure limits and security options for this tool.
      </p>

      <div className="form-row">
        <div className="form-group">
          <label>Max Results</label>
          <input
            type="number"
            value={maxResults}
            onChange={(e) => setMaxResults(parseInt(e.target.value) || 100)}
            min={1}
            max={1000}
          />
          <p className="field-help">
            Maximum number of rows/results returned per query.
          </p>
        </div>

        <div className="form-group">
          <label>Timeout (seconds)</label>
          <input
            type="number"
            value={timeout}
            onChange={(e) => setTimeout(parseInt(e.target.value) || 30)}
            min={1}
            max={300}
          />
          <p className="field-help">
            Maximum time to wait for a query to complete.
          </p>
        </div>
      </div>

      <fieldset>
        <legend>Security</legend>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={allowWrite}
            onChange={(e) => setAllowWrite(e.target.checked)}
          />
          Allow write operations (INSERT/UPDATE/DELETE)
        </label>
        {allowWrite && (
          <p className="warning-text">
            Warning: Write operations are enabled. The AI will be able to modify data.
          </p>
        )}
      </fieldset>
    </div>
  );

  const renderReview = () => {
    const typeInfo = TOOL_TYPE_INFO[toolType];

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Review your tool configuration before saving.
        </p>

        <div className="review-section">
          <h4>Tool Type</h4>
          <p>
            <span className="review-icon">
              {typeInfo?.icon === 'database' ? '🗄️' : typeInfo?.icon === 'terminal' ? '💻' : '🖥️'}
            </span>
            {typeInfo?.name}
          </p>
        </div>

        <div className="review-section">
          <h4>Name</h4>
          <p>{name}</p>
        </div>

        {description && (
          <div className="review-section">
            <h4>Description</h4>
            <p className="review-description">{description}</p>
          </div>
        )}

        <div className="review-section">
          <h4>Connection</h4>
          <pre className="review-config">
            {JSON.stringify(getConnectionConfig(), null, 2)}
          </pre>
        </div>

        <div className="review-section">
          <h4>Options</h4>
          <ul>
            <li>Max results: {maxResults}</li>
            <li>Timeout: {timeout}s</li>
            <li>Write operations: {allowWrite ? 'Enabled' : 'Disabled'}</li>
          </ul>
        </div>
      </div>
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 'type':
        return renderTypeSelection();
      case 'connection':
        return renderConnectionConfig();
      case 'description':
        return renderDescription();
      case 'options':
        return renderOptions();
      case 'review':
        return renderReview();
    }
  };

  return (
    <div className="card wizard-card">
      <div className="wizard-header">
        <h2>{isEditing ? 'Edit Tool' : 'Add Tool'}</h2>
        <button type="button" className="close-btn" onClick={onClose}>
          ✕
        </button>
      </div>

      {/* Progress indicator */}
      <div className="wizard-progress" ref={progressRef}>
        {WIZARD_STEPS.map((step, index) => {
          if (isEditing && step === 'type') return null;
          const stepIndex = WIZARD_STEPS.indexOf(step);
          const isNavigable = canNavigateToStep(stepIndex);
          return (
            <button
              key={step}
              type="button"
              className={`wizard-step ${currentStep === step ? 'active' : ''} ${
                getCurrentStepIndex() > index ? 'completed' : ''
              } ${isNavigable ? 'navigable' : ''}`}
              onClick={() => goToStep(step)}
              disabled={!isNavigable}
            >
              <span className="step-number">{isEditing ? index : index + 1}</span>
              <span className="step-title">{getStepTitle(step)}</span>
            </button>
          );
        })}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="wizard-body">{renderStepContent()}</div>

      <div className="wizard-footer">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={getCurrentStepIndex() === 0 || (isEditing && currentStep === 'connection') ? onClose : goToPreviousStep}
        >
          {getCurrentStepIndex() === 0 || (isEditing && currentStep === 'connection') ? 'Cancel' : 'Back'}
        </button>

        {currentStep === 'review' ? (
          <button
            type="button"
            className="btn"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Tool'}
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            onClick={goToNextStep}
            disabled={!canProceed()}
          >
            Continue
          </button>
        )}
      </div>
    </div>
  );
}
