import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/api';
import type { AppSettings, UpdateSettingsRequest, OllamaModel, LLMModel, AvailableModel } from '@/types';

export function SettingsPanel() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Ollama connection state
  const [ollamaConnecting, setOllamaConnecting] = useState(false);
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);

  // LLM provider model fetching state
  const [llmModelsFetching, setLlmModelsFetching] = useState(false);
  const [llmModelsError, setLlmModelsError] = useState<string | null>(null);
  const [llmModels, setLlmModels] = useState<LLMModel[]>([]);
  const [llmModelsLoaded, setLlmModelsLoaded] = useState(false);

  // Model filter modal state
  const [showModelFilterModal, setShowModelFilterModal] = useState(false);
  const [allAvailableModels, setAllAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelFilterText, setModelFilterText] = useState('');

  // Form state
  const [formData, setFormData] = useState<UpdateSettingsRequest>({});

  // Track if we've already auto-tested Ollama
  const hasAutoTestedOllama = useRef(false);

  // Test Ollama connection
  const testOllamaConnection = useCallback(async (
    protocol: string,
    host: string,
    port: number
  ) => {
    setOllamaConnecting(true);
    setOllamaError(null);
    setOllamaConnected(false);
    setOllamaModels([]);

    try {
      const response = await api.testOllamaConnection({
        protocol: protocol || 'http',
        host: host || 'localhost',
        port: port || 11434,
      });

      if (response.success) {
        setOllamaConnected(true);
        setOllamaModels(response.models);
        setFormData((prev) => ({
          ...prev,
          ollama_base_url: response.base_url,
        }));
      } else {
        setOllamaError(response.message);
      }
    } catch (err) {
      setOllamaError(err instanceof Error ? err.message : 'Connection test failed');
    } finally {
      setOllamaConnecting(false);
    }
  }, []);

  // Fetch LLM models from provider API
  const fetchLlmModels = useCallback(async (
    provider: 'openai' | 'anthropic',
    apiKey: string
  ) => {
    if (!apiKey || apiKey.length < 10) {
      setLlmModelsError('Please enter a valid API key first');
      return;
    }

    setLlmModelsFetching(true);
    setLlmModelsError(null);
    setLlmModels([]);
    setLlmModelsLoaded(false);

    try {
      const response = await api.fetchLLMModels({
        provider,
        api_key: apiKey,
      });

      if (response.success) {
        setLlmModels(response.models);
        setLlmModelsLoaded(true);
        // Auto-select the default model if none is currently set or the current one isn't in the list
        if (response.default_model) {
          const currentModel = formData.llm_model;
          const modelExists = response.models.some((m) => m.id === currentModel);
          if (!currentModel || !modelExists) {
            setFormData((prev) => ({
              ...prev,
              llm_model: response.default_model,
            }));
          }
        }
      } else {
        setLlmModelsError(response.message);
      }
    } catch (err) {
      setLlmModelsError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setLlmModelsFetching(false);
    }
  }, [formData.llm_model]);

  // Open model filter modal and load all available models
  const openModelFilterModal = useCallback(async () => {
    setModelsLoading(true);
    setShowModelFilterModal(true);
    setModelFilterText('');

    try {
      const response = await api.getAllModels();
      setAllAvailableModels(response.models);

      // Initialize selected models from current settings or all models
      const allowedModels = response.allowed_models || [];
      if (allowedModels.length > 0) {
        setSelectedModels(new Set(allowedModels));
      } else {
        // Default to all models selected
        setSelectedModels(new Set(response.models.map(m => m.id)));
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  const toggleModel = (modelId: string) => {
    setSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });
  };

  const selectAllModels = () => {
    setSelectedModels(new Set(allAvailableModels.map(m => m.id)));
  };

  const deselectAllModels = () => {
    setSelectedModels(new Set());
  };

  const saveModelFilter = async () => {
    // If all models are selected, save empty array (means all allowed)
    const allSelected = selectedModels.size === allAvailableModels.length;
    const allowedModels = allSelected ? [] : Array.from(selectedModels);

    try {
      await api.updateSettings({ allowed_chat_models: allowedModels });
      setShowModelFilterModal(false);
      setSuccess('Model filter saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save model filter');
    }
  };

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.getSettings();
      setSettings(data);
      setFormData({
        // Embedding settings
        embedding_provider: data.embedding_provider,
        embedding_model: data.embedding_model,
        ollama_protocol: data.ollama_protocol,
        ollama_host: data.ollama_host,
        ollama_port: data.ollama_port,
        ollama_base_url: data.ollama_base_url,
        // LLM settings
        llm_provider: data.llm_provider,
        llm_model: data.llm_model,
        openai_api_key: data.openai_api_key,
        anthropic_api_key: data.anthropic_api_key,
      });
      // Reset Ollama connection state
      setOllamaConnected(false);
      setOllamaError(null);
      setOllamaModels([]);
      // Reset LLM models state
      setLlmModels([]);
      setLlmModelsError(null);
      setLlmModelsLoaded(false);
      setError(null);

      // Auto-test Ollama if using ollama embedding provider
      if (data.embedding_provider === 'ollama' && !hasAutoTestedOllama.current) {
        hasAutoTestedOllama.current = true;
        testOllamaConnection(
          data.ollama_protocol || 'http',
          data.ollama_host || 'localhost',
          data.ollama_port || 11434
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  }, [testOllamaConnection]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleTestOllamaConnection = async () => {
    await testOllamaConnection(
      formData.ollama_protocol || 'http',
      formData.ollama_host || 'localhost',
      formData.ollama_port || 11434
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSuccess(null);
    setError(null);

    try {
      // Compute ollama_base_url from separate fields
      const dataToSave = {
        ...formData,
        ollama_base_url: `${formData.ollama_protocol || 'http'}://${formData.ollama_host || 'localhost'}:${formData.ollama_port || 11434}`,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setSuccess('Settings saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <h2>Settings</h2>
        <p className="muted">Loading settings...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Settings</h2>

      {error && <div className="error-banner">{error}</div>}
      {success && <div className="success-banner">{success}</div>}

      <form onSubmit={handleSubmit}>
        {/* Embedding Configuration */}
        <fieldset>
          <legend>Embedding Configuration</legend>
          <p className="fieldset-help">
            Configure how document embeddings are generated for FAISS indexes.
          </p>

          <div className="form-row">
            <div className="form-group">
              <label>Provider</label>
              <select
                value={formData.embedding_provider || 'ollama'}
                onChange={(e) => {
                  const newProvider = e.target.value as 'ollama' | 'openai';
                  setFormData({
                    ...formData,
                    embedding_provider: newProvider,
                    // Set sensible default model when switching providers
                    embedding_model:
                      newProvider === 'ollama'
                        ? 'nomic-embed-text'
                        : 'text-embedding-3-small',
                  });
                  // Reset Ollama connection state when switching providers
                  if (newProvider !== 'ollama') {
                    setOllamaConnected(false);
                    setOllamaError(null);
                    setOllamaModels([]);
                  }
                }}
              >
                <option value="ollama">Ollama</option>
                <option value="openai">OpenAI</option>
              </select>
            </div>
          </div>

          {formData.embedding_provider === 'ollama' && (
            <>
              {/* Ollama Server Connection */}
              <div className="connection-section">
                <h4>Ollama Server Connection</h4>
                <div className="form-row form-row-3">
                  <div className="form-group form-group-small">
                    <label>Protocol</label>
                    <select
                      value={formData.ollama_protocol || 'http'}
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          ollama_protocol: e.target.value as 'http' | 'https',
                        });
                        // Reset connection when server settings change
                        setOllamaConnected(false);
                        setOllamaError(null);
                        setOllamaModels([]);
                      }}
                    >
                      <option value="http">http://</option>
                      <option value="https">https://</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Host / IP</label>
                    <input
                      type="text"
                      value={formData.ollama_host || ''}
                      onChange={(e) => {
                        setFormData({ ...formData, ollama_host: e.target.value });
                        // Reset connection when server settings change
                        setOllamaConnected(false);
                        setOllamaError(null);
                        setOllamaModels([]);
                      }}
                      placeholder="localhost"
                    />
                  </div>
                  <div className="form-group form-group-small">
                    <label>Port</label>
                    <input
                      type="number"
                      value={formData.ollama_port || 11434}
                      onChange={(e) => {
                        setFormData({
                          ...formData,
                          ollama_port: parseInt(e.target.value, 10) || 11434,
                        });
                        // Reset connection when server settings change
                        setOllamaConnected(false);
                        setOllamaError(null);
                        setOllamaModels([]);
                      }}
                      min={1}
                      max={65535}
                    />
                  </div>
                </div>

                <div className="connection-test-row">
                  <button
                    type="button"
                    className={`btn btn-test ${ollamaConnected ? 'btn-connected' : ''}`}
                    onClick={handleTestOllamaConnection}
                    disabled={ollamaConnecting}
                  >
                    {ollamaConnecting
                      ? 'Connecting...'
                      : ollamaConnected
                        ? 'Connected'
                        : 'Test Connection'}
                  </button>
                  {ollamaConnected && (
                    <span className="connection-status success">
                      {ollamaModels.length} model(s) available
                    </span>
                  )}
                  {ollamaError && (
                    <span className="connection-status error">{ollamaError}</span>
                  )}
                </div>
              </div>

              {/* Model Selection */}
              <div className="form-group">
                <label>Embedding Model</label>
                {ollamaConnected && ollamaModels.length > 0 ? (
                  <select
                    value={formData.embedding_model || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, embedding_model: e.target.value })
                    }
                  >
                    <option value="">Select a model...</option>
                    {ollamaModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                        {model.size
                          ? ` (${(model.size / 1024 / 1024 / 1024).toFixed(1)}GB)`
                          : ''}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={formData.embedding_model || ''}
                    onChange={(e) =>
                      setFormData({ ...formData, embedding_model: e.target.value })
                    }
                    placeholder="nomic-embed-text"
                  />
                )}
                <p className="field-help">
                  {ollamaConnected
                    ? 'Select an embedding model from your Ollama server.'
                    : 'Test the connection above to see available models, or enter manually.'}
                </p>
              </div>
            </>
          )}

          {formData.embedding_provider === 'openai' && (
            <>
              <div className="form-group">
                <label>Embedding Model</label>
                <input
                  type="text"
                  value={formData.embedding_model || ''}
                  onChange={(e) =>
                    setFormData({ ...formData, embedding_model: e.target.value })
                  }
                  placeholder="text-embedding-3-small"
                />
              </div>
              <p className="field-help">
                Requires OpenAI API key (configured below in LLM settings).
              </p>
            </>
          )}
        </fieldset>

        {/* LLM Configuration */}
        <fieldset>
          <legend>LLM Configuration (Chat/RAG)</legend>
          <p className="fieldset-help">
            Configure the language model used for answering questions and tool calls.
          </p>

          <div className="form-group">
            <label>Provider</label>
            <select
              value={formData.llm_provider || 'openai'}
              onChange={(e) => {
                const newProvider = e.target.value as 'openai' | 'anthropic';
                setFormData({
                  ...formData,
                  llm_provider: newProvider,
                  llm_model:
                    newProvider === 'anthropic'
                      ? 'claude-sonnet-4-20250514'
                      : 'gpt-4o',
                });
                // Reset LLM models when switching providers
                setLlmModels([]);
                setLlmModelsError(null);
                setLlmModelsLoaded(false);
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic (Claude)</option>
            </select>
          </div>

          {/* API Key - show appropriate one based on provider */}
          {formData.llm_provider === 'openai' || !formData.llm_provider ? (
            <div className="form-group">
              <label>OpenAI API Key</label>
              <div className="input-with-button">
                <input
                  type="password"
                  value={formData.openai_api_key || ''}
                  onChange={(e) => {
                    setFormData({ ...formData, openai_api_key: e.target.value });
                    // Reset models when API key changes
                    setLlmModels([]);
                    setLlmModelsError(null);
                    setLlmModelsLoaded(false);
                  }}
                  placeholder="sk-..."
                />
                <button
                  type="button"
                  className={`btn btn-test ${llmModelsLoaded && formData.llm_provider === 'openai' ? 'btn-connected' : ''}`}
                  onClick={() => fetchLlmModels('openai', formData.openai_api_key || '')}
                  disabled={llmModelsFetching || !formData.openai_api_key}
                >
                  {llmModelsFetching ? 'Fetching...' : llmModelsLoaded && formData.llm_provider === 'openai' ? 'Loaded' : 'Fetch Models'}
                </button>
              </div>
              {llmModelsError && formData.llm_provider === 'openai' && (
                <p className="field-error">{llmModelsError}</p>
              )}
              <p className="field-help">
                Required for OpenAI LLM and optionally for OpenAI embeddings.
              </p>
            </div>
          ) : (
            <div className="form-group">
              <label>Anthropic API Key</label>
              <div className="input-with-button">
                <input
                  type="password"
                  value={formData.anthropic_api_key || ''}
                  onChange={(e) => {
                    setFormData({ ...formData, anthropic_api_key: e.target.value });
                    // Reset models when API key changes
                    setLlmModels([]);
                    setLlmModelsError(null);
                    setLlmModelsLoaded(false);
                  }}
                  placeholder="sk-ant-..."
                />
                <button
                  type="button"
                  className={`btn btn-test ${llmModelsLoaded && formData.llm_provider === 'anthropic' ? 'btn-connected' : ''}`}
                  onClick={() => fetchLlmModels('anthropic', formData.anthropic_api_key || '')}
                  disabled={llmModelsFetching || !formData.anthropic_api_key}
                >
                  {llmModelsFetching ? 'Fetching...' : llmModelsLoaded && formData.llm_provider === 'anthropic' ? 'Loaded' : 'Fetch Models'}
                </button>
              </div>
              {llmModelsError && formData.llm_provider === 'anthropic' && (
                <p className="field-error">{llmModelsError}</p>
              )}
            </div>
          )}

          {/* Model Selection */}
          <div className="form-group">
            <label>Model</label>
            {llmModelsLoaded && llmModels.length > 0 ? (
              <select
                value={formData.llm_model || ''}
                onChange={(e) =>
                  setFormData({ ...formData, llm_model: e.target.value })
                }
              >
                <option value="">Select a model...</option>
                {llmModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={formData.llm_model || ''}
                onChange={(e) =>
                  setFormData({ ...formData, llm_model: e.target.value })
                }
                placeholder={
                  formData.llm_provider === 'anthropic'
                    ? 'claude-sonnet-4-20250514'
                    : 'gpt-4o'
                }
              />
            )}
            <p className="field-help">
              {llmModelsLoaded
                ? 'Select a model from your provider.'
                : 'Click "Fetch Models" above to see available models, or enter manually.'}
            </p>
          </div>

          {/* Show OpenAI key field for embeddings if using Anthropic for LLM */}
          {formData.llm_provider === 'anthropic' && formData.embedding_provider === 'openai' && (
            <div className="form-group">
              <label>OpenAI API Key (for embeddings)</label>
              <input
                type="password"
                value={formData.openai_api_key || ''}
                onChange={(e) =>
                  setFormData({ ...formData, openai_api_key: e.target.value })
                }
                placeholder="sk-..."
              />
              <p className="field-help">
                Required for OpenAI embeddings when using Anthropic for LLM.
              </p>
            </div>
          )}

          {/* Chat Model Filter */}
          <div className="form-group">
            <label>Chat Model Filter</label>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={openModelFilterModal}
            >
              Configure Allowed Models
            </button>
            <p className="field-help">
              Limit which models appear in the Chat view dropdown.
            </p>
          </div>
        </fieldset>

        <div className="form-actions">
          <button type="submit" disabled={saving}>
            {saving ? 'Saving...' : 'Save Settings'}
          </button>
          <button type="button" onClick={loadSettings} disabled={saving}>
            Reset
          </button>
        </div>
      </form>

      {settings?.updated_at && (
        <p className="muted" style={{ marginTop: '1rem', fontSize: '0.85rem' }}>
          Last updated: {new Date(settings.updated_at).toLocaleString()}
        </p>
      )}

      {/* Model Filter Modal */}
      {showModelFilterModal && (
        <div className="modal-overlay" onClick={() => setShowModelFilterModal(false)}>
          <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Configure Allowed Chat Models</h3>
              <button
                className="modal-close"
                onClick={() => setShowModelFilterModal(false)}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              {modelsLoading ? (
                <p className="muted">Loading available models...</p>
              ) : allAvailableModels.length === 0 ? (
                <p className="muted">
                  No models available. Please configure API keys and save settings first.
                </p>
              ) : (
                <>
                  <div className="model-filter-search">
                    <input
                      type="text"
                      placeholder="Filter models..."
                      value={modelFilterText}
                      onChange={(e) => setModelFilterText(e.target.value)}
                      autoFocus
                    />
                  </div>
                  <div className="model-filter-actions">
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={selectAllModels}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={deselectAllModels}
                    >
                      Deselect All
                    </button>
                    <span className="muted" style={{ marginLeft: 'auto' }}>
                      {selectedModels.size} of {allAvailableModels.length} selected
                    </span>
                  </div>
                  <div className="model-filter-list">
                    {allAvailableModels
                      .filter((model) =>
                        modelFilterText === '' ||
                        model.name.toLowerCase().includes(modelFilterText.toLowerCase()) ||
                        model.provider.toLowerCase().includes(modelFilterText.toLowerCase())
                      )
                      .map((model) => (
                      <label key={model.id} className="model-filter-item">
                        <input
                          type="checkbox"
                          checked={selectedModels.has(model.id)}
                          onChange={() => toggleModel(model.id)}
                        />
                        <span className="model-filter-name">{model.name}</span>
                        <span className="model-filter-provider">{model.provider}</span>
                      </label>
                    ))}
                  </div>
                </>
              )}
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowModelFilterModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn"
                onClick={saveModelFilter}
                disabled={modelsLoading || allAvailableModels.length === 0}
              >
                Save Filter
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
