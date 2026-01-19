import type { OllamaModel } from '@/types';

interface OllamaConnectionFormProps {
  /** Current protocol value */
  protocol: 'http' | 'https';
  /** Current host value */
  host: string;
  /** Current port value */
  port: number;
  /** Current selected model */
  model: string;
  /** Whether connected to Ollama server */
  connected: boolean;
  /** Whether currently connecting */
  connecting: boolean;
  /** Error message if connection failed */
  error: string | null;
  /** Available models after connection */
  models: OllamaModel[];
  /** Label for the model field */
  modelLabel?: string;
  /** Placeholder for manual model input */
  modelPlaceholder?: string;
  /** Help text shown when connected */
  connectedHelpText?: string;
  /** Help text shown when not connected */
  disconnectedHelpText?: string;
  /** Whether to filter for embedding models only */
  embeddingsOnly?: boolean;
  /** Called when protocol changes */
  onProtocolChange: (protocol: 'http' | 'https') => void;
  /** Called when host changes */
  onHostChange: (host: string) => void;
  /** Called when port changes */
  onPortChange: (port: number) => void;
  /** Called when model changes */
  onModelChange: (model: string) => void;
  /** Called when user clicks Fetch Models button */
  onFetchModels: () => void;
}

/**
 * Shared component for Ollama server connection form.
 * Used for both LLM and Embedding configuration in SettingsPanel.
 */
export function OllamaConnectionForm({
  protocol,
  host,
  port,
  model,
  connected,
  connecting,
  error,
  models,
  modelLabel = 'Model',
  modelPlaceholder = 'llama3',
  connectedHelpText = 'Select a model from your Ollama server.',
  disconnectedHelpText = 'Click "Fetch Models" to see available models, or enter manually.',
  onProtocolChange,
  onHostChange,
  onPortChange,
  onModelChange,
  onFetchModels,
}: OllamaConnectionFormProps) {
  return (
    <>
      <div className="form-row form-row-4">
        <div className="form-group form-group-small">
          <label>Protocol</label>
          <select
            value={protocol}
            onChange={(e) => onProtocolChange(e.target.value as 'http' | 'https')}
          >
            <option value="http">http://</option>
            <option value="https">https://</option>
          </select>
        </div>
        <div className="form-group">
          <label>Host / IP</label>
          <input
            type="text"
            value={host}
            onChange={(e) => onHostChange(e.target.value)}
            placeholder="localhost"
          />
        </div>
        <div className="form-group form-group-small">
          <label>Port</label>
          <input
            type="number"
            value={port}
            onChange={(e) => onPortChange(parseInt(e.target.value, 10) || 11434)}
            min={1}
            max={65535}
          />
        </div>
        <div className="form-group form-group-button">
          <button
            type="button"
            className={`btn btn-test ${connected ? 'btn-connected' : ''}`}
            onClick={onFetchModels}
            disabled={connecting}
          >
            {connecting ? 'Connecting...' : connected ? 'Connected' : 'Fetch Models'}
          </button>
        </div>
      </div>

      {connected && (
        <p className="field-help" style={{ color: 'var(--success-color, #28a745)' }}>
          {models.length} model(s) available
        </p>
      )}
      {error && <p className="field-error">{error}</p>}
      <p className="field-help">
        When running in Docker, use <code>host.docker.internal</code> instead of <code>localhost</code>.
      </p>

      {/* Model Selection */}
      <div className="form-group">
        <label>{modelLabel}</label>
        {connected && models.length > 0 ? (
          <select value={model} onChange={(e) => onModelChange(e.target.value)}>
            <option value="">Select a model...</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
                {m.size ? ` (${(m.size / 1024 / 1024 / 1024).toFixed(1)}GB)` : ''}
                {m.dimensions ? ` [${m.dimensions} dims]` : ''}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            placeholder={modelPlaceholder}
          />
        )}
        <p className="field-help">
          {connected
            ? (() => {
                const selectedModel = models.find(m => m.name === model);
                const dimInfo = selectedModel?.dimensions
                  ? ` Selected model outputs ${selectedModel.dimensions}-dimension vectors.`
                  : '';
                return `${connectedHelpText}${dimInfo}`;
              })()
            : disconnectedHelpText}
        </p>
      </div>
    </>
  );
}
