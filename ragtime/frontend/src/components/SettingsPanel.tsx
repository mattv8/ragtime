import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/api';
import type { AppSettings, UpdateSettingsRequest, OllamaModel, LLMModel, EmbeddingModel, AvailableModel, LdapConfig } from '@/types';

/**
 * Format a DN for display like Active Directory tree view.
 * E.g., "OU=Users,OU=NYC,DC=example,DC=com" -> "NYC / Users"
 */
function formatDnForDisplay(dn: string, baseDn: string): string {
  // Parse DN components
  const parts = dn.split(',').map(p => p.trim());

  // Find the relative path from base DN
  const baseParts = baseDn.split(',').map(p => p.trim());
  const baseLength = baseParts.length;

  // If this is the base DN itself, show it specially
  if (dn === baseDn) {
    // Extract domain from DC components: DC=example,DC=com -> example.com
    const dcParts = baseParts.filter(p => p.toUpperCase().startsWith('DC='));
    const domain = dcParts.map(p => p.substring(3)).join('.');
    return `[Root] ${domain}`;
  }

  // Get the relative path (parts before the base DN)
  const relativeParts = parts.slice(0, parts.length - baseLength);

  // Build display string: show OU/CN names in reverse order (top to bottom)
  const names = relativeParts.map(part => {
    const [_type, ...valueParts] = part.split('=');
    return valueParts.join('='); // Handle values with = in them
  }).reverse();

  // Show path from parent to child
  return names.join(' / ');
}

interface SettingsPanelProps {
  onServerNameChange?: (name: string) => void;
}

export function SettingsPanel({ onServerNameChange }: SettingsPanelProps) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Section-specific saving states
  const [embeddingSaving, setEmbeddingSaving] = useState(false);
  const [llmSaving, setLlmSaving] = useState(false);

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

  // OpenAI embedding model fetching state
  const [embeddingModelsFetching, setEmbeddingModelsFetching] = useState(false);
  const [embeddingModelsError, setEmbeddingModelsError] = useState<string | null>(null);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [embeddingModelsLoaded, setEmbeddingModelsLoaded] = useState(false);

  // Model filter modal state
  const [showModelFilterModal, setShowModelFilterModal] = useState(false);
  const [allAvailableModels, setAllAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelFilterText, setModelFilterText] = useState('');

  // LDAP configuration state
  const [ldapConfig, setLdapConfig] = useState<LdapConfig | null>(null);
  const [ldapFormData, setLdapFormData] = useState({
    ldap_protocol: 'ldaps' as 'ldap' | 'ldaps',
    ldap_host: '',
    ldap_port: 636,
    allow_self_signed: false,
    bind_dn: '',
    bind_password: '',
    user_search_base: '',
    user_search_filter: '(|(sAMAccountName={username})(uid={username}))',
    admin_group_dn: '',
    user_group_dn: '',
  });
  const [ldapTesting, setLdapTesting] = useState(false);
  const [ldapTestResult, setLdapTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [ldapSaving, setLdapSaving] = useState(false);
  const [ldapDiscoveredOus, setLdapDiscoveredOus] = useState<string[]>([]);
  const [ldapDiscoveredGroups, setLdapDiscoveredGroups] = useState<{ dn: string; name: string }[]>([]);

  // Form state
  const [formData, setFormData] = useState<UpdateSettingsRequest>({});

  // Track if we've already auto-tested Ollama
  const hasAutoTestedOllama = useRef(false);

  // Test Ollama connection
  const testOllamaConnection = useCallback(async (
    protocol: 'http' | 'https',
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

  // Fetch embedding models from OpenAI API
  const fetchEmbeddingModels = useCallback(async (apiKey: string) => {
    if (!apiKey || apiKey.length < 10) {
      setEmbeddingModelsError('Please enter a valid OpenAI API key first');
      return;
    }

    setEmbeddingModelsFetching(true);
    setEmbeddingModelsError(null);
    setEmbeddingModels([]);
    setEmbeddingModelsLoaded(false);

    try {
      const response = await api.fetchEmbeddingModels({
        provider: 'openai',
        api_key: apiKey,
      });

      if (response.success) {
        setEmbeddingModels(response.models);
        setEmbeddingModelsLoaded(true);
        // Auto-select the default model if none is currently set or the current one isn't in the list
        if (response.default_model) {
          const currentModel = formData.embedding_model;
          const modelExists = response.models.some((m) => m.id === currentModel);
          if (!currentModel || !modelExists) {
            setFormData((prev) => ({
              ...prev,
              embedding_model: response.default_model,
            }));
          }
        }
      } else {
        setEmbeddingModelsError(response.message);
      }
    } catch (err) {
      setEmbeddingModelsError(err instanceof Error ? err.message : 'Failed to fetch embedding models');
    } finally {
      setEmbeddingModelsFetching(false);
    }
  }, [formData.embedding_model]);

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
        embedding_dimensions: data.embedding_dimensions,
        ollama_protocol: data.ollama_protocol,
        ollama_host: data.ollama_host,
        ollama_port: data.ollama_port,
        ollama_base_url: data.ollama_base_url,
        // LLM settings
        llm_provider: data.llm_provider,
        llm_model: data.llm_model,
        openai_api_key: data.openai_api_key,
        anthropic_api_key: data.anthropic_api_key,
        max_iterations: data.max_iterations,
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

      // Load LDAP configuration (non-blocking - don't await discovery)
      try {
        const ldapData = await api.getLdapConfig();
        setLdapConfig(ldapData);

        // Parse server_url into components
        let protocol: 'ldap' | 'ldaps' = 'ldaps';
        let host = '';
        let port = 636;
        if (ldapData.server_url) {
          const match = ldapData.server_url.match(/^(ldaps?):\/\/([^:]+)(?::(\d+))?$/);
          if (match) {
            protocol = match[1] as 'ldap' | 'ldaps';
            host = match[2];
            port = match[3] ? parseInt(match[3], 10) : (protocol === 'ldaps' ? 636 : 389);
          }
        }

        setLdapFormData({
          ldap_protocol: protocol,
          ldap_host: host,
          ldap_port: port,
          allow_self_signed: ldapData.allow_self_signed || false,
          bind_dn: ldapData.bind_dn || '',
          bind_password: '', // Never returned from server
          user_search_base: ldapData.user_search_base || '',
          user_search_filter: ldapData.user_search_filter || '(|(sAMAccountName={username})(uid={username}))',
          admin_group_dn: ldapData.admin_group_dn || '',
          user_group_dn: ldapData.user_group_dn || '',
        });

        // Auto-discover LDAP structure in background (non-blocking)
        if (ldapData.server_url && ldapData.bind_dn) {
          // Don't await - let it run async
          api.discoverLdapWithStoredCredentials()
            .then((discovery) => {
              if (discovery.success) {
                setLdapDiscoveredOus(discovery.user_ous);
                setLdapDiscoveredGroups(discovery.groups);
                setLdapTestResult({ success: true, message: `Connected. Found ${discovery.user_ous.length} OUs and ${discovery.groups.length} groups.` });
              }
            })
            .catch(() => {
              // Silent fail - user can still test connection manually
            });
        }
      } catch {
        // LDAP config may not exist yet, that's OK
        setLdapConfig(null);
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

  // Helper to build server URL from components
  const buildServerUrl = () => {
    const { ldap_protocol, ldap_host, ldap_port } = ldapFormData;
    if (!ldap_host) return '';
    return `${ldap_protocol}://${ldap_host}:${ldap_port}`;
  };

  // LDAP connection test and discovery
  const handleTestLdapConnection = async () => {
    const serverUrl = buildServerUrl();
    if (!serverUrl || !ldapFormData.bind_dn || !ldapFormData.bind_password) {
      setLdapTestResult({ success: false, message: 'Server, Bind DN, and Bind Password are required' });
      return;
    }

    setLdapTesting(true);
    setLdapTestResult(null);

    try {
      const response = await api.discoverLdap({
        server_url: serverUrl,
        bind_dn: ldapFormData.bind_dn,
        bind_password: ldapFormData.bind_password,
        allow_self_signed: ldapFormData.allow_self_signed,
      });

      setLdapDiscoveredOus(response.user_ous);
      setLdapDiscoveredGroups(response.groups);
      setLdapTestResult({ success: true, message: `Connected. Found ${response.user_ous.length} OUs and ${response.groups.length} groups.` });
    } catch (err) {
      setLdapTestResult({ success: false, message: err instanceof Error ? err.message : 'Connection failed' });
      setLdapDiscoveredOus([]);
      setLdapDiscoveredGroups([]);
    } finally {
      setLdapTesting(false);
    }
  };

  // Save LDAP configuration
  const handleSaveLdapConfig = async () => {
    setLdapSaving(true);
    setError(null);

    try {
      const serverUrl = buildServerUrl();
      const updated = await api.updateLdapConfig({
        server_url: serverUrl || undefined,
        allow_self_signed: ldapFormData.allow_self_signed,
        bind_dn: ldapFormData.bind_dn || undefined,
        bind_password: ldapFormData.bind_password || undefined,
        user_search_base: ldapFormData.user_search_base || undefined,
        user_search_filter: ldapFormData.user_search_filter || undefined,
        admin_group_dn: ldapFormData.admin_group_dn || undefined,
        user_group_dn: ldapFormData.user_group_dn || undefined,
      });
      setLdapConfig(updated);
      setSuccess('LDAP configuration saved successfully');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save LDAP configuration');
    } finally {
      setLdapSaving(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
  };

  // Save Server Branding
  const [brandingSaving, setBrandingSaving] = useState(false);
  const handleSaveBranding = async () => {
    setBrandingSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave = {
        server_name: formData.server_name,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      // Notify parent component of name change
      if (onServerNameChange && updated.server_name) {
        onServerNameChange(updated.server_name);
      }
      setSuccess('Server branding saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save branding settings');
    } finally {
      setBrandingSaving(false);
    }
  };

  // Save Embedding Configuration
  const handleSaveEmbedding = async () => {
    setEmbeddingSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave: UpdateSettingsRequest = {
        embedding_provider: formData.embedding_provider,
        embedding_model: formData.embedding_model,
        embedding_dimensions: formData.embedding_dimensions ?? null,
        ollama_protocol: formData.ollama_protocol,
        ollama_host: formData.ollama_host,
        ollama_port: formData.ollama_port,
        ollama_base_url: `${formData.ollama_protocol || 'http'}://${formData.ollama_host || 'localhost'}:${formData.ollama_port || 11434}`,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setSuccess('Embedding configuration saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save embedding settings');
    } finally {
      setEmbeddingSaving(false);
    }
  };

  // Save LLM Configuration
  const handleSaveLlm = async () => {
    setLlmSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave = {
        llm_provider: formData.llm_provider,
        llm_model: formData.llm_model,
        openai_api_key: formData.openai_api_key,
        anthropic_api_key: formData.anthropic_api_key,
        allowed_chat_models: formData.allowed_chat_models,
        max_iterations: formData.max_iterations,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setSuccess('LLM configuration saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save LLM settings');
    } finally {
      setLlmSaving(false);
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

      {/* API Endpoint Info */}
      <div className="api-info-box">
        <strong>OpenAI-Compatible API</strong>
        <p>
          Connect external clients (e.g., Open WebUI) using:
        </p>
        <code>{`${window.location.protocol}//${window.location.hostname}:8000/v1`}</code>
        <p className="field-help" style={{ marginTop: '0.5rem' }}>
          Model: <code>{(formData.server_name || settings?.server_name || 'Ragtime').toLowerCase().replace(/\s+/g, '-')}</code>. The <code>/v1</code> path is required for OpenAI API compatibility.
        </p>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {success && <div className="success-banner">{success}</div>}

      <form onSubmit={handleSubmit}>
        {/* Server Branding */}
        <fieldset>
          <legend>Server Branding</legend>
          <p className="fieldset-help">
            Customize the server name displayed in the UI, API model name, and MCP server identity.
          </p>

          <div className="form-group">
            <label>Server Name</label>
            <input
              type="text"
              value={formData.server_name ?? settings?.server_name ?? 'Ragtime'}
              onChange={(e) => setFormData({ ...formData, server_name: e.target.value })}
              placeholder="Ragtime"
            />
          </div>

          <div className="form-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSaveBranding}
              disabled={brandingSaving}
            >
              {brandingSaving ? 'Saving...' : 'Save Branding'}
            </button>
          </div>
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
                    // Also reset embedding models since they use the same key
                    setEmbeddingModels([]);
                    setEmbeddingModelsError(null);
                    setEmbeddingModelsLoaded(false);
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
                ? 'Select the model that will be used for chat completions and RAG responses.'
                : 'This model ID is sent to the provider API for all chat/RAG requests. Click "Fetch Models" to see available options.'}
            </p>
          </div>

          {/* Show OpenAI key field for embeddings if using Anthropic for LLM */}
          {formData.llm_provider === 'anthropic' && formData.embedding_provider === 'openai' && (
            <div className="form-group">
              <label>OpenAI API Key (for embeddings)</label>
              <input
                type="password"
                value={formData.openai_api_key || ''}
                onChange={(e) => {
                  setFormData({ ...formData, openai_api_key: e.target.value });
                  // Reset embedding models when key changes
                  setEmbeddingModels([]);
                  setEmbeddingModelsError(null);
                  setEmbeddingModelsLoaded(false);
                }}
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

          <div className="form-group">
            <label>Max Tool Iterations</label>
            <input
              type="number"
              min={1}
              max={50}
              value={formData.max_iterations ?? 15}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  max_iterations: Math.max(1, Math.min(50, parseInt(e.target.value, 10) || 1)),
                })
              }
            />
            <p className="field-help">
              Maximum number of agent tool-calling steps before stopping. Lower to keep runs short; higher to allow deeper reasoning.
            </p>
          </div>

          <div className="form-group" style={{ marginTop: '1rem' }}>
            <button
              type="button"
              className="btn"
              onClick={handleSaveLlm}
              disabled={llmSaving}
            >
              {llmSaving ? 'Saving...' : 'Save LLM Configuration'}
            </button>
          </div>
        </fieldset>

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
                  // Reset embedding models when switching away from OpenAI
                  if (newProvider !== 'openai') {
                    setEmbeddingModels([]);
                    setEmbeddingModelsError(null);
                    setEmbeddingModelsLoaded(false);
                  }
                }}
              >
                <option value="ollama">Ollama</option>
                <option value="openai">OpenAI</option>
              </select>
              <p className="field-help">
                Note: Anthropic does not offer embedding models. Use Ollama or OpenAI for document embeddings.
              </p>
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
                <div className="input-with-button">
                  {embeddingModelsLoaded && embeddingModels.length > 0 ? (
                    <select
                      value={formData.embedding_model || ''}
                      onChange={(e) =>
                        setFormData({ ...formData, embedding_model: e.target.value })
                      }
                    >
                      <option value="">Select a model...</option>
                      {embeddingModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name}
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
                      placeholder="text-embedding-3-small"
                    />
                  )}
                  <button
                    type="button"
                    className={`btn btn-test ${embeddingModelsLoaded ? 'btn-connected' : ''}`}
                    onClick={() => fetchEmbeddingModels(formData.openai_api_key || '')}
                    disabled={embeddingModelsFetching || !formData.openai_api_key}
                  >
                    {embeddingModelsFetching ? 'Fetching...' : embeddingModelsLoaded ? 'Loaded' : 'Fetch Models'}
                  </button>
                </div>
                {embeddingModelsError && (
                  <p className="field-error">{embeddingModelsError}</p>
                )}
                <p className="field-help">
                  {embeddingModelsLoaded
                    ? 'Select an embedding model from OpenAI (embedding models only).'
                    : 'Requires OpenAI API key (configured above). Click "Fetch Models" to see available embedding models.'}
                </p>
              </div>

              {/* Embedding Dimensions (only for text-embedding-3-* models) */}
              {formData.embedding_model?.startsWith('text-embedding-3') && (
                <div className="form-group">
                  <label>Embedding Dimensions</label>
                  <input
                    type="number"
                    min="256"
                    max="3072"
                    step="1"
                    value={formData.embedding_dimensions ?? ''}
                    onChange={(e) => {
                      const val = e.target.value ? parseInt(e.target.value, 10) : null;
                      setFormData({ ...formData, embedding_dimensions: val });
                    }}
                    placeholder="Default (model max)"
                  />
                  <p className="field-help">
                    Controls the output size of embeddings. Lower values = faster search and less storage,
                    but slightly reduced accuracy. <strong>Recommended: 1536</strong> for best balance.
                    Values over 2000 disable fast indexed search (pgvector limit). Changing this requires a full re-index of all filesystem indexes.
                  </p>
                </div>
              )}
            </>
          )}

          <div className="form-group" style={{ marginTop: '1rem' }}>
            <button
              type="button"
              className="btn"
              onClick={handleSaveEmbedding}
              disabled={embeddingSaving}
            >
              {embeddingSaving ? 'Saving...' : 'Save Embedding Configuration'}
            </button>
          </div>
        </fieldset>

        {/* LDAP Authentication Configuration */}
        <fieldset>
          <legend>LDAP Authentication</legend>
          <p className="fieldset-help">
            Configure LDAP/Active Directory authentication. Leave host empty to disable LDAP and use local admin only.
          </p>

          {/* Server Connection */}
          <div style={{ display: 'grid', gridTemplateColumns: '100px 1fr 70px', gap: '12px', marginBottom: '16px' }}>
            <div className="form-group">
              <label>Protocol</label>
              <select
                value={ldapFormData.ldap_protocol}
                onChange={(e) => {
                  const protocol = e.target.value as 'ldap' | 'ldaps';
                  const defaultPort = protocol === 'ldaps' ? 636 : 389;
                  setLdapFormData({ ...ldapFormData, ldap_protocol: protocol, ldap_port: defaultPort });
                }}
              >
                <option value="ldaps">ldaps://</option>
                <option value="ldap">ldap://</option>
              </select>
            </div>
            <div className="form-group">
              <label>Host</label>
              <input
                type="text"
                value={ldapFormData.ldap_host}
                onChange={(e) =>
                  setLdapFormData({ ...ldapFormData, ldap_host: e.target.value })
                }
                placeholder="ldap.example.com"
              />
            </div>
            <div className="form-group">
              <label>Port</label>
              <input
                type="number"
                value={ldapFormData.ldap_port}
                onChange={(e) =>
                  setLdapFormData({ ...ldapFormData, ldap_port: parseInt(e.target.value, 10) || 636 })
                }
              />
            </div>
          </div>

          {/* Self-signed certificate option - only show for ldaps */}
          {ldapFormData.ldap_protocol === 'ldaps' && (
            <div className="form-row">
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={ldapFormData.allow_self_signed}
                  onChange={(e) =>
                    setLdapFormData({ ...ldapFormData, allow_self_signed: e.target.checked })
                  }
                />
                Allow self-signed certificates
              </label>
              <p className="field-help" style={{ marginTop: '0.25rem' }}>
                Skip SSL certificate validation. Use only for testing or with internal CAs.
              </p>
            </div>
          )}

          {/* Bind Account */}
          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label>Bind DN / Username</label>
              <input
                type="text"
                value={ldapFormData.bind_dn}
                onChange={(e) =>
                  setLdapFormData({ ...ldapFormData, bind_dn: e.target.value })
                }
                placeholder="user@domain.com or CN=admin,DC=example,DC=com"
              />
              <p className="field-help">
                AD: user@domain.com or DOMAIN\user. OpenLDAP: full DN like cn=admin,dc=example,dc=com
              </p>
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Bind Password</label>
              <input
                type="password"
                value={ldapFormData.bind_password}
                onChange={(e) =>
                  setLdapFormData({ ...ldapFormData, bind_password: e.target.value })
                }
                placeholder={ldapConfig?.bind_dn ? '(password saved)' : 'Enter password'}
              />
            </div>
          </div>

          <div className="form-group">
            <div className="connection-test-row">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleTestLdapConnection}
                disabled={ldapTesting || !ldapFormData.ldap_host || !ldapFormData.bind_dn || !ldapFormData.bind_password}
              >
                {ldapTesting ? 'Testing...' : 'Test Connection & Discover'}
              </button>
              {ldapTestResult?.success && (
                <span className="connection-status success">
                  {ldapTestResult.message}
                </span>
              )}
              {ldapTestResult && !ldapTestResult.success && (
                <span className="connection-status error">{ldapTestResult.message}</span>
              )}
            </div>
          </div>

          {/* Show search config when we have discovered OUs (from test or auto-load) */}
          {ldapDiscoveredOus.length > 0 && (
            <>
              <div className="form-row">
                <div className="form-group" style={{ flex: 2 }}>
                  <label>User Search Base</label>
                  <select
                    value={ldapFormData.user_search_base}
                    onChange={(e) =>
                      setLdapFormData({ ...ldapFormData, user_search_base: e.target.value })
                    }
                  >
                    <option value="">Select a search base...</option>
                    {ldapDiscoveredOus.map((ou) => (
                      <option key={ou} value={ou}>
                        {formatDnForDisplay(ou, ldapDiscoveredOus[0] || ou)}
                      </option>
                    ))}
                  </select>
                  <p className="field-help">
                    Where to search for users. Select the root domain to search all users, or a specific OU to limit scope.
                  </p>
                  {ldapFormData.user_search_base && (
                    <p className="field-help" style={{ fontFamily: 'monospace', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                      DN: {ldapFormData.user_search_base}
                    </p>
                  )}
                </div>
              </div>

              <div className="form-row">
                <div className="form-group" style={{ flex: 2 }}>
                  <label>User Search Filter</label>
                  <input
                    type="text"
                    value={ldapFormData.user_search_filter}
                    onChange={(e) =>
                      setLdapFormData({ ...ldapFormData, user_search_filter: e.target.value })
                    }
                    placeholder="(|(sAMAccountName={username})(uid={username}))"
                  />
                  <p className="field-help">
                    LDAP filter to find users. Use {'{username}'} as placeholder.
                  </p>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Admin Group DN</label>
                  <select
                    value={ldapFormData.admin_group_dn}
                    onChange={(e) =>
                      setLdapFormData({ ...ldapFormData, admin_group_dn: e.target.value })
                    }
                  >
                    <option value="">No admin group (all users are standard)</option>
                    {ldapDiscoveredGroups.map((g) => (
                      <option key={g.dn} value={g.dn}>{g.name}</option>
                    ))}
                  </select>
                  <p className="field-help">
                    Members of this group get admin privileges
                  </p>
                </div>
                <div className="form-group">
                  <label>User Group DN (optional)</label>
                  <select
                    value={ldapFormData.user_group_dn}
                    onChange={(e) =>
                      setLdapFormData({ ...ldapFormData, user_group_dn: e.target.value })
                    }
                  >
                    <option value="">Any user can login</option>
                    {ldapDiscoveredGroups.map((g) => (
                      <option key={g.dn} value={g.dn}>{g.name}</option>
                    ))}
                  </select>
                  <p className="field-help">
                    If set, users must be members of this group to login
                  </p>
                </div>
              </div>
            </>
          )}

          {/* Show current config if loaded but not testing */}
          {ldapConfig?.server_url && !ldapTestResult?.success && (
            <div className="form-group">
              <div className="meta-pills">
                <span className="meta-pill">
                  <span className="meta-pill-label">Server</span>
                  <span className="meta-pill-value">{ldapConfig.server_url}</span>
                </span>
                {ldapConfig.user_search_base && (
                  <span className="meta-pill">
                    <span className="meta-pill-label">Base</span>
                    <span className="meta-pill-value">{ldapConfig.user_search_base}</span>
                  </span>
                )}
                {ldapConfig.admin_group_dn && (
                  <span className="meta-pill">
                    <span className="meta-pill-label">Admin Group</span>
                    <span className="meta-pill-value">{ldapConfig.admin_group_dn}</span>
                  </span>
                )}
              </div>
            </div>
          )}

          <div className="form-group">
            <button
              type="button"
              className="btn"
              onClick={handleSaveLdapConfig}
              disabled={ldapSaving}
            >
              {ldapSaving ? 'Saving...' : 'Save LDAP Configuration'}
            </button>
          </div>
        </fieldset>
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
