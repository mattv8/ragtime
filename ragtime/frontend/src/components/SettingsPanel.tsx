import { useState, useEffect, useCallback, useRef } from 'react';
import { Lock, LockOpen } from 'lucide-react';
import { api } from '@/api';
import type { AppSettings, UpdateSettingsRequest, OllamaModel, LLMModel, EmbeddingModel, AvailableModel, LdapConfig, McpRouteConfig, AuthStatus } from '@/types';
import { MCPRoutesPanel } from './MCPRoutesPanel';
import { OllamaConnectionForm } from './OllamaConnectionForm';
import { ModelSelector } from './ModelSelector';

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
  /** Setting ID to highlight and scroll to (e.g., 'sequential_index_loading') */
  highlightSetting?: string | null;
  /** Called after highlight animation completes to clear the param */
  onHighlightComplete?: () => void;
  /** Auth status for security warnings */
  authStatus?: AuthStatus | null;
}

export function SettingsPanel({ onServerNameChange, highlightSetting, onHighlightComplete, authStatus }: SettingsPanelProps) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Section-specific saving states
  const [embeddingSaving, setEmbeddingSaving] = useState(false);
  const [llmSaving, setLlmSaving] = useState(false);

  // Scroll to and highlight setting when highlightSetting changes
  useEffect(() => {
    if (highlightSetting && !loading) {
      const element = document.getElementById(`setting-${highlightSetting}`);
      if (element) {
        // If it's a details element, open it first
        if (element.tagName === 'DETAILS') {
          (element as HTMLDetailsElement).open = true;
        }
        // Add highlight class
        element.classList.add('highlight-setting');
        // Scroll into view with some padding
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Clear the highlight after animation
        const timer = setTimeout(() => {
          element.classList.remove('highlight-setting');
          onHighlightComplete?.();
        }, 2000);
        return () => clearTimeout(timer);
      }
    }
  }, [highlightSetting, loading, onHighlightComplete]);

  // Ollama connection state (for embeddings)
  const [ollamaConnecting, setOllamaConnecting] = useState(false);
  const [ollamaConnected, setOllamaConnected] = useState(false);
  const [ollamaError, setOllamaError] = useState<string | null>(null);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);

  // LLM Ollama connection state (separate from embedding Ollama)
  const [llmOllamaConnecting, setLlmOllamaConnecting] = useState(false);
  const [llmOllamaConnected, setLlmOllamaConnected] = useState(false);
  const [llmOllamaError, setLlmOllamaError] = useState<string | null>(null);
  const [llmOllamaModels, setLlmOllamaModels] = useState<OllamaModel[]>([]);

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

  // MCP Routes panel state
  const [showMcpRoutesPanel, setShowMcpRoutesPanel] = useState(false);
  const [mcpRoutes, setMcpRoutes] = useState<McpRouteConfig[]>([]);

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
  const hasAutoTestedLlmOllama = useRef(false);

  // Test Ollama connection (for embeddings)
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
        embeddings_only: true,  // Filter to embedding models only
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

  // Test LLM Ollama connection (separate from embeddings)
  const testLlmOllamaConnection = useCallback(async (
    protocol: 'http' | 'https',
    host: string,
    port: number
  ) => {
    setLlmOllamaConnecting(true);
    setLlmOllamaError(null);
    setLlmOllamaConnected(false);
    setLlmOllamaModels([]);

    try {
      const response = await api.testOllamaConnection({
        protocol: protocol || 'http',
        host: host || 'localhost',
        port: port || 11434,
      });

      if (response.success) {
        setLlmOllamaConnected(true);
        setLlmOllamaModels(response.models);
        setFormData((prev) => ({
          ...prev,
          llm_ollama_base_url: response.base_url,
        }));
      } else {
        setLlmOllamaError(response.message);
      }
    } catch (err) {
      setLlmOllamaError(err instanceof Error ? err.message : 'Connection test failed');
    } finally {
      setLlmOllamaConnecting(false);
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
      const { settings: data } = await api.getSettings();
      setSettings(data);
      setFormData({
        // Server branding
        server_name: data.server_name,
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
        llm_max_tokens: data.llm_max_tokens,
        llm_ollama_protocol: data.llm_ollama_protocol,
        llm_ollama_host: data.llm_ollama_host,
        llm_ollama_port: data.llm_ollama_port,
        llm_ollama_base_url: data.llm_ollama_base_url,
        openai_api_key: data.openai_api_key,
        anthropic_api_key: data.anthropic_api_key,
        max_iterations: data.max_iterations,
        // Search settings
        search_results_k: data.search_results_k,
        aggregate_search: data.aggregate_search,
        // Advanced search settings
        search_use_mmr: data.search_use_mmr,
        search_mmr_lambda: data.search_mmr_lambda,
        context_token_budget: data.context_token_budget,
        chunking_use_tokens: data.chunking_use_tokens,
        ivfflat_lists: data.ivfflat_lists,
        // API Tool Output settings
        tool_output_mode: data.tool_output_mode,
        // MCP settings
        mcp_enabled: data.mcp_enabled,
        mcp_default_route_auth: data.mcp_default_route_auth,
        mcp_default_route_auth_method: data.mcp_default_route_auth_method,
        mcp_default_route_allowed_group: data.mcp_default_route_allowed_group,
        mcp_default_route_password: data.mcp_default_route_password ?? '',
      });
      // Reset Ollama connection state (for embeddings)
      setOllamaConnected(false);
      setOllamaError(null);
      setOllamaModels([]);
      // Reset LLM Ollama connection state
      setLlmOllamaConnected(false);
      setLlmOllamaError(null);
      setLlmOllamaModels([]);
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

      // Auto-test LLM Ollama if using ollama LLM provider
      if (data.llm_provider === 'ollama' && !hasAutoTestedLlmOllama.current) {
        hasAutoTestedLlmOllama.current = true;
        testLlmOllamaConnection(
          data.llm_ollama_protocol || 'http',
          data.llm_ollama_host || 'localhost',
          data.llm_ollama_port || 11434
        );
      }

      // Load MCP routes (for summary display)
      try {
        const routesRes = await api.listMcpRoutes();
        setMcpRoutes(routesRes.routes);
      } catch {
        // MCP routes may fail silently
        setMcpRoutes([]);
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
        // Send empty string explicitly to clear these fields (don't convert to undefined)
        admin_group_dn: ldapFormData.admin_group_dn,
        user_group_dn: ldapFormData.user_group_dn,
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
      const dataToSave: Record<string, unknown> = {
        llm_provider: formData.llm_provider,
        llm_model: formData.llm_model,
        llm_max_tokens: formData.llm_max_tokens,
        openai_api_key: formData.openai_api_key,
        anthropic_api_key: formData.anthropic_api_key,
        allowed_chat_models: formData.allowed_chat_models,
        max_iterations: formData.max_iterations,
      };
      // Include LLM Ollama connection fields when using Ollama provider
      if (formData.llm_provider === 'ollama') {
        dataToSave.llm_ollama_protocol = formData.llm_ollama_protocol;
        dataToSave.llm_ollama_host = formData.llm_ollama_host;
        dataToSave.llm_ollama_port = formData.llm_ollama_port;
        dataToSave.llm_ollama_base_url = `${formData.llm_ollama_protocol || 'http'}://${formData.llm_ollama_host || 'localhost'}:${formData.llm_ollama_port || 11434}`;
      }
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

  // Save Search Configuration
  const [searchSaving, setSearchSaving] = useState(false);
  const handleSaveSearch = async () => {
    setSearchSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave = {
        search_results_k: formData.search_results_k,
        aggregate_search: formData.aggregate_search,
        // Advanced settings
        search_use_mmr: formData.search_use_mmr,
        search_mmr_lambda: formData.search_mmr_lambda,
        context_token_budget: formData.context_token_budget,
        chunking_use_tokens: formData.chunking_use_tokens,
        ivfflat_lists: formData.ivfflat_lists,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setSuccess('Search configuration saved. Restart the server to apply changes to search tools.');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save search settings');
    } finally {
      setSearchSaving(false);
    }
  };

  // Save MCP Configuration
  const [mcpSaving, setMcpSaving] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [showMcpPassword, setShowMcpPassword] = useState(false);
  const handleSaveMcp = async () => {
    setMcpSaving(true);
    setSuccess(null);
    setMcpError(null);

    // Validate password if provided (not empty string which clears, and not undefined which skips)
    const pwd = formData.mcp_default_route_password;
    const authMethod = formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password';
    if (authMethod === 'password' && pwd !== undefined && pwd !== '' && pwd.length < 8) {
      setMcpError('MCP password must be at least 8 characters');
      setMcpSaving(false);
      return;
    }

    try {
      const dataToSave: UpdateSettingsRequest = {
        mcp_enabled: formData.mcp_enabled,
        mcp_default_route_auth: formData.mcp_default_route_auth,
        mcp_default_route_auth_method: formData.mcp_default_route_auth_method,
        mcp_default_route_allowed_group: formData.mcp_default_route_allowed_group,
      };
      // Include password if it was modified
      if (formData.mcp_default_route_password !== undefined) {
        dataToSave.mcp_default_route_password = formData.mcp_default_route_password;
      }
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      // Update formData with the returned values
      setFormData(prev => ({
        ...prev,
        mcp_enabled: updated.mcp_enabled,
        mcp_default_route_auth: updated.mcp_default_route_auth,
        mcp_default_route_auth_method: updated.mcp_default_route_auth_method,
        mcp_default_route_allowed_group: updated.mcp_default_route_allowed_group,
        mcp_default_route_password: updated.mcp_default_route_password ?? '',
      }));
      setSuccess('MCP configuration saved.');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setMcpError(err instanceof Error ? err.message : 'Failed to save MCP settings');
    } finally {
      setMcpSaving(false);
    }
  };

  // Save Performance Configuration
  const [performanceSaving, setPerformanceSaving] = useState(false);
  const handleSavePerformance = async () => {
    setPerformanceSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave: UpdateSettingsRequest = {
        sequential_index_loading: formData.sequential_index_loading,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setFormData(prev => ({
        ...prev,
        sequential_index_loading: updated.sequential_index_loading,
      }));
      setSuccess('Performance settings saved. Changes take effect on next server restart.');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save performance settings');
    } finally {
      setPerformanceSaving(false);
    }
  };

  const getDisplayUrl = (path: string) => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port === '8001' ? '8000' : window.location.port;
    const host = port ? `${hostname}:${port}` : hostname;
    return `${protocol}//${host}${path}`;
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
      <div
        className={`api-info-box ${highlightSetting === 'api_key_info' ? 'highlight-setting' : ''}`}
        id="setting-api_key_info"
      >
        <strong>OpenAI-Compatible API</strong>
        <p>
          Connect external clients (e.g., Open WebUI) using:
        </p>
        <code>{getDisplayUrl('/v1')}</code>
        <p className="field-help" style={{ marginTop: '0.5rem' }}>
          Model: <code>{(formData.server_name || settings?.server_name || 'Ragtime').toLowerCase().replace(/\s+/g, '-')}</code>. The <code>/v1</code> path is required for OpenAI API compatibility.
        </p>
        {(!authStatus?.api_key_configured || window.location.protocol === 'http:') && (
          <div className="field-warning" style={{ marginTop: '0.75rem', padding: '0.75rem', backgroundColor: 'rgba(255, 193, 7, 0.15)', borderLeft: '3px solid #ffc107', borderRadius: '4px' }}>
            <strong>Security:</strong>
            {!authStatus?.api_key_configured && (
              <span> The API endpoint accepts an API Key for authentication (set via <code>API_KEY</code> environment variable).
              Without an API key, anyone with network access can use your LLM and tools.</span>
            )}
            {window.location.protocol === 'http:' && (
              <span> {authStatus?.api_key_configured ? '' : 'Additionally, y'}ou are currently accessing over HTTP - API keys and credentials will be transmitted in plaintext.
              Consider using HTTPS via a reverse proxy or setting <code>ENABLE_HTTPS=true</code>.</span>
            )}
          </div>
        )}
      </div>

      {/* MCP Routes Summary */}
      <div className="api-info-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong>MCP (Model Context Protocol)</strong>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setShowMcpRoutesPanel(true)}
          >
            Manage Routes
          </button>
        </div>
        <p style={{ marginTop: '0.5rem' }}>
          Connect AI assistants (Claude Desktop, VS Code, etc.) using:
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <code>{getDisplayUrl('/mcp')}</code>
            <span className="muted" style={{ fontSize: '0.85em' }}>(default - all tools)</span>
            {settings?.mcp_default_route_auth && (settings?.has_mcp_default_password || settings?.mcp_default_route_auth_method === 'oauth2') ? (
              <span title={settings?.mcp_default_route_auth_method === 'oauth2' ? "OAuth2 protected" : "Password protected"}>
                <Lock size={14} style={{ color: 'var(--success-color, #4caf50)' }} />
              </span>
            ) : (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: 'var(--error-color, #f44336)', fontSize: '0.8em' }}>
                <LockOpen size={14} /> unprotected
              </span>
            )}
          </div>
          {mcpRoutes.filter(r => r.enabled).map(route => {
            const isProtected = route.require_auth && (route.has_password || route.auth_method === 'oauth2');
            return (
              <div key={route.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <code>{getDisplayUrl(`/mcp/${route.route_path}`)}</code>
                <span className="muted" style={{ fontSize: '0.85em' }}>({route.name})</span>
                {isProtected ? (
                  <span title={route.auth_method === 'oauth2' ? 'OAuth2 (LDAP)' : 'Password protected'}>
                    <Lock size={14} style={{ color: 'var(--success-color, #4caf50)' }} />
                  </span>
                ) : (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', color: 'var(--error-color, #f44336)', fontSize: '0.8em' }}>
                    <LockOpen size={14} /> unprotected
                  </span>
                )}
              </div>
            );
          })}
        </div>
        {mcpRoutes.filter(r => !r.enabled).length > 0 && (
          <p className="field-help" style={{ marginTop: '0.5rem' }}>
            {mcpRoutes.filter(r => !r.enabled).length} disabled route(s) not shown.
          </p>
        )}
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
                const newProvider = e.target.value as 'openai' | 'anthropic' | 'ollama';
                setFormData({
                  ...formData,
                  llm_provider: newProvider,
                  llm_model:
                    newProvider === 'anthropic'
                      ? ''
                      : newProvider === 'ollama'
                        ? ''
                        : '',
                });
                // Reset LLM models when switching providers
                setLlmModels([]);
                setLlmModelsError(null);
                setLlmModelsLoaded(false);
                // Reset LLM Ollama state when switching away from Ollama
                if (newProvider !== 'ollama') {
                  setLlmOllamaConnected(false);
                  setLlmOllamaError(null);
                  setLlmOllamaModels([]);
                }
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="ollama">Ollama</option>
            </select>
          </div>

          {/* Ollama LLM Server Connection - only show when Ollama is selected */}
          {formData.llm_provider === 'ollama' && (
            <OllamaConnectionForm
              protocol={formData.llm_ollama_protocol || 'http'}
              host={formData.llm_ollama_host || ''}
              port={formData.llm_ollama_port || 11434}
              model={formData.llm_model || ''}
              connected={llmOllamaConnected}
              connecting={llmOllamaConnecting}
              error={llmOllamaError}
              models={llmOllamaModels}
              modelLabel="Model"
              modelPlaceholder=""
              connectedHelpText="Select an LLM from your Ollama server."
              disconnectedHelpText="Click &quot;Fetch Models&quot; to see available models, or enter manually."
              onProtocolChange={(protocol) => {
                setFormData({ ...formData, llm_ollama_protocol: protocol });
                setLlmOllamaConnected(false);
                setLlmOllamaError(null);
                setLlmOllamaModels([]);
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, llm_ollama_host: host });
                setLlmOllamaConnected(false);
                setLlmOllamaError(null);
                setLlmOllamaModels([]);
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, llm_ollama_port: port });
                setLlmOllamaConnected(false);
                setLlmOllamaError(null);
                setLlmOllamaModels([]);
              }}
              onModelChange={(model) => setFormData({ ...formData, llm_model: model })}
              onFetchModels={() => testLlmOllamaConnection(
                formData.llm_ollama_protocol || 'http',
                formData.llm_ollama_host || 'localhost',
                formData.llm_ollama_port || 11434
              )}
            />
          )}

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
                {window.location.protocol === 'http:' && (
                  <span style={{ color: '#b8860b' }}> Warning: API keys are transmitted in plaintext over HTTP.</span>
                )}
              </p>
            </div>
          ) : formData.llm_provider === 'anthropic' ? (
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
              {window.location.protocol === 'http:' && (
                <p className="field-help" style={{ color: '#b8860b' }}>
                  Warning: API keys are transmitted in plaintext over HTTP.
                </p>
              )}
            </div>
          ) : null}

          {/* Model Selection - for OpenAI and Anthropic only (Ollama has its own section above) */}
          {formData.llm_provider !== 'ollama' && (
            <div className="form-group">
              <label>Model</label>
              {llmModelsLoaded && llmModels.length > 0 ? (
                <ModelSelector
                  models={llmModels}
                  selectedModelId={formData.llm_model || ''}
                  onModelChange={(modelId) =>
                    setFormData({ ...formData, llm_model: modelId })
                  }
                  placeholder="Select a model..."
                  variant="full"
                />
              ) : (
                <input
                  type="text"
                  value={formData.llm_model || ''}
                  onChange={(e) =>
                    setFormData({ ...formData, llm_model: e.target.value })
                  }
                  placeholder="Select a model..."
                />
              )}
              <p className="field-help">
                {llmModelsLoaded
                  ? 'Select the model that will be used for chat completions and RAG responses.'
                  : 'This model ID is sent to the provider API for all chat/RAG requests. Click "Fetch Models" to see available options.'}
              </p>
            </div>
          )}

          {/* Show OpenAI key field for embeddings if using Anthropic or Ollama for LLM */}
          {(formData.llm_provider === 'anthropic' || formData.llm_provider === 'ollama') && formData.embedding_provider === 'openai' && (
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

          <div className="form-row">
            <div className="form-group" style={{ flex: 2 }}>
              <label>Max Output Tokens</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  style={{ flex: 1 }}
                  value={(() => {
                    const val = formData.llm_max_tokens || 4096;
                    if (val >= 100000) return 100;
                    const min = 500;
                    const max = 100000;
                    const scale = Math.log(max / min);
                    return (Math.log(val / min) / scale) * 100;
                  })()}
                  onChange={(e) => {
                    const slider = parseInt(e.target.value, 10);
                    const min = 500;
                    const max = 100000;
                    let val;
                    if (slider === 100) {
                       val = max;
                    } else {
                      const scale = Math.log(max / min);
                      val = Math.round(min * Math.exp((slider / 100) * scale));
                    }
                    setFormData({ ...formData, llm_max_tokens: val });
                  }}
                />
                <span style={{ minWidth: '80px', textAlign: 'right', fontFamily: 'monospace' }}>
                  {formData.llm_max_tokens && formData.llm_max_tokens >= 100000 ? 'LLM Max' : (formData.llm_max_tokens || 4096).toLocaleString()}
                </span>
              </div>
              <p className="field-help">
                Limit the length of the model's response.
              </p>
            </div>

            <div className="form-group" style={{ flex: 1 }}>
              <label>Max Tool Iterations</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="range"
                  min="1"
                  max="100"
                  step="1"
                  style={{ flex: 1 }}
                  value={formData.max_iterations ?? 30}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      max_iterations: parseInt(e.target.value, 10),
                    })
                  }
                />
                <span style={{ minWidth: '30px', textAlign: 'right', fontFamily: 'monospace' }}>
                  {formData.max_iterations ?? 30}
                </span>
              </div>
              <p className="field-help">
                Maximum number of agent tool-calling steps.
              </p>
            </div>
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
        <fieldset id="setting-embedding_config">
          <legend>Embedding Configuration</legend>
          <p className="fieldset-help">
            Configure how document embeddings are generated for FAISS indexes.
          </p>

          <div className="form-row">
            <div className="form-group" id="setting-embedding_provider">
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
            {/* Show embedding dimension info */}
            {(() => {
              // Get the dimension from the selected model if available
              const selectedOllamaModel = ollamaModels.find(m => m.name === formData.embedding_model);
              const selectedOpenAIModel = embeddingModels.find(m => m.id === formData.embedding_model);
              const selectedModelDimension = selectedOllamaModel?.dimensions || selectedOpenAIModel?.dimensions;
              const storedDimension = settings?.embedding_dimension;

              // Determine if there's a mismatch between stored and selected
              const hasMismatch = storedDimension && selectedModelDimension && storedDimension !== selectedModelDimension;
              // Use selected model dimension if available, otherwise fall back to stored
              const displayDimension = selectedModelDimension || storedDimension;

              return (
                <div className="form-group" style={{ flex: '0 0 auto', minWidth: '180px' }}>
                  <label>{selectedModelDimension ? 'Model Dimensions' : 'Current Dimensions'}</label>
                  <div style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: hasMismatch
                      ? 'var(--warning-bg, rgba(255, 152, 0, 0.1))'
                      : 'var(--bg-secondary, #1e1e1e)',
                    borderRadius: '4px',
                    border: `1px solid ${hasMismatch ? 'var(--warning-color, #ff9800)' : 'var(--border-color, #3c3c3c)'}`,
                    fontFamily: 'monospace',
                    fontSize: '1.1rem',
                    textAlign: 'center',
                  }}>
                    {displayDimension ? (
                      <>
                        {displayDimension.toLocaleString()}
                        {hasMismatch && (
                          <span style={{ color: 'var(--warning-color, #ff9800)', fontSize: '0.75rem', marginLeft: '0.25rem' }}>
                            (change)
                          </span>
                        )}
                      </>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>—</span>
                    )}
                  </div>
                  <p className="field-help">
                    {hasMismatch
                      ? `Indexes use ${storedDimension?.toLocaleString()} dims. Re-index required.`
                      : storedDimension
                        ? 'Matches existing indexes.'
                        : 'Will be set on first index.'}
                  </p>
                </div>
              );
            })()}
          </div>

          {formData.embedding_provider === 'ollama' && (
            <OllamaConnectionForm
              protocol={formData.ollama_protocol || 'http'}
              host={formData.ollama_host || ''}
              port={formData.ollama_port || 11434}
              model={formData.embedding_model || ''}
              connected={ollamaConnected}
              connecting={ollamaConnecting}
              error={ollamaError}
              models={ollamaModels}
              modelLabel="Embedding Model"
              modelPlaceholder="nomic-embed-text"
              connectedHelpText="Select an embedding model from your Ollama server."
              disconnectedHelpText="Click &quot;Fetch Models&quot; to see available models, or enter manually."
              onProtocolChange={(protocol) => {
                setFormData({ ...formData, ollama_protocol: protocol });
                setOllamaConnected(false);
                setOllamaError(null);
                setOllamaModels([]);
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, ollama_host: host });
                setOllamaConnected(false);
                setOllamaError(null);
                setOllamaModels([]);
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, ollama_port: port });
                setOllamaConnected(false);
                setOllamaError(null);
                setOllamaModels([]);
              }}
              onModelChange={(model) => setFormData({ ...formData, embedding_model: model })}
              onFetchModels={handleTestOllamaConnection}
            />
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
                          {model.name}{model.dimensions ? ` (${model.dimensions} dims)` : ''}
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
                    ? (() => {
                        const selectedModel = embeddingModels.find(m => m.id === formData.embedding_model);
                        const dimInfo = selectedModel?.dimensions
                          ? ` Selected model outputs ${selectedModel.dimensions}-dimension vectors.`
                          : '';
                        return `Select an embedding model from OpenAI.${dimInfo}`;
                      })()
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

        {/* Search Configuration */}
        <fieldset>
          <legend>Search Configuration</legend>
          <p className="fieldset-help">
            Configure how vector search behaves across your indexed knowledge bases.
          </p>

          <div className="form-group">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={formData.aggregate_search ?? settings?.aggregate_search ?? true}
                onChange={(e) =>
                  setFormData({ ...formData, aggregate_search: e.target.checked })
                }
                style={{ marginRight: '0.5rem' }}
              />
              <span>Aggregate search results (single tool)</span>
            </label>
            <p className="field-help">
              <strong>Enabled (default):</strong> A single <code>search_knowledge</code> tool searches all indexes.
              Results are combined and the AI receives context from all sources.<br />
              <strong>Disabled:</strong> Creates separate <code>search_&lt;index_name&gt;</code> tools for each index.
              The AI can choose which specific index to search, giving it granular control.
              Use this when you have distinct knowledge bases (e.g., code vs. docs) and want the AI to target searches.
            </p>
          </div>

          {/* Advanced Search Settings */}
          <details style={{ marginBottom: '16px' }} id="setting-search_advanced">
            <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Settings</summary>
            <div className="form-group">
              <label>Results per Search (k)</label>
              <input
                type="number"
                min={1}
                max={100}
                value={formData.search_results_k ?? settings?.search_results_k ?? 5}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    search_results_k: Math.max(1, Math.min(100, parseInt(e.target.value, 10) || 5)),
                  })
                }
              />
              <p className="field-help">
                Number of matching document chunks retrieved per vector search query (k).
                Lower values (3-5) are faster and cheaper but may miss relevant context.
                Higher values (10-20) provide more context but increase token usage and response time.
                Very high values (50+) may introduce noise from less relevant matches.
              </p>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={formData.search_use_mmr ?? settings?.search_use_mmr ?? true}
                  onChange={(e) =>
                    setFormData({ ...formData, search_use_mmr: e.target.checked })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Use MMR (Max Marginal Relevance)</span>
              </label>
              <p className="field-help">
                Reduces near-duplicate results by balancing relevance with diversity.
                Recommended for most use cases to get varied, high-quality context.
              </p>
            </div>

            {(formData.search_use_mmr ?? settings?.search_use_mmr ?? true) && (
              <div className="form-group">
                <label>MMR Diversity/Relevance (lambda: {formData.search_mmr_lambda ?? settings?.search_mmr_lambda ?? 0.5})</label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.1}
                  value={formData.search_mmr_lambda ?? settings?.search_mmr_lambda ?? 0.5}
                  onChange={(e) =>
                    setFormData({ ...formData, search_mmr_lambda: parseFloat(e.target.value) })
                  }
                  style={{ width: '100%' }}
                />
                <p className="field-help">
                  <strong>0 = Max diversity</strong> (most varied results) |
                  <strong> 1 = Max relevance</strong> (closest matches).
                  Default 0.5 provides a good balance.
                </p>
              </div>
            )}

            <div className="form-group">
              <label>Context Token Budget</label>
              <input
                type="number"
                min={0}
                max={32000}
                value={formData.context_token_budget ?? settings?.context_token_budget ?? 4000}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    context_token_budget: Math.max(0, Math.min(32000, parseInt(e.target.value, 10) || 0)),
                  })
                }
              />
              <p className="field-help">
                Maximum tokens for retrieved context sent to the LLM. Set to 0 for unlimited.
                Prevents context overflow for models with smaller context windows.
              </p>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={formData.chunking_use_tokens ?? settings?.chunking_use_tokens ?? true}
                  onChange={(e) =>
                    setFormData({ ...formData, chunking_use_tokens: e.target.checked })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Token-based chunking</span>
              </label>
              <p className="field-help">
                Use token-based chunking instead of character-based for more accurate
                chunk sizes aligned with model tokenization.
              </p>
            </div>

            <div className="form-group">
              <label>IVFFlat Lists (pgvector)</label>
              <input
                type="number"
                min={10}
                max={1000}
                value={formData.ivfflat_lists ?? settings?.ivfflat_lists ?? 100}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    ivfflat_lists: Math.max(10, Math.min(1000, parseInt(e.target.value, 10) || 100)),
                  })
                }
              />
              <p className="field-help">
                Index parameter for pgvector (filesystem indexes only).
                Higher values: slower build but faster queries for large datasets.
                Recommended: sqrt(number of embeddings). Default: 100.
              </p>
            </div>
          </details>

          <div className="form-group" style={{ marginTop: '1rem' }}>
            <button
              type="button"
              className="btn"
              onClick={handleSaveSearch}
              disabled={searchSaving}
            >
              {searchSaving ? 'Saving...' : 'Save Search Configuration'}
            </button>
          </div>
        </fieldset>

        {/* API Output Configuration */}
        <fieldset>
          <legend>API Output Configuration</legend>
          <p className="fieldset-help">
            Configure how tool call output is handled in OpenAI-compatible API responses (e.g., when using OpenWebUI or other clients).
            This does not affect MCP or the built-in chat interface.
          </p>

          <div className="form-group">
            <label>Tool Output Visibility</label>
            <select
              value={(formData.tool_output_mode ?? settings?.tool_output_mode) === 'default' ? 'show' : (formData.tool_output_mode ?? settings?.tool_output_mode ?? 'show')}
              onChange={(e) =>
                setFormData({ ...formData, tool_output_mode: e.target.value as any })
              }
            >
              <option value="show">Show (Always include output)</option>
              <option value="hide">Hide (Suppress output)</option>
              <option value="auto">Auto (AI decides)</option>
            </select>
            <p className="field-help">
              Controls whether tool execution details (inputs/outputs) are included in the streaming response.
              <strong>Hide</strong> is useful for cleaner output in clients that don't support tool visualization.
            </p>
          </div>

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={async () => {
                try {
                  const updated = await api.updateSettings({
                    tool_output_mode: formData.tool_output_mode,
                  });
                  setFormData((prev) => ({
                    ...prev,
                    tool_output_mode: updated.tool_output_mode,
                  }));
                  setSuccess('API output settings saved.');
                } catch (err) {
                  setError(err instanceof Error ? err.message : 'Failed to save settings');
                }
              }}
            >
              Save API Output Settings
            </button>
          </div>
        </fieldset>

        {/* MCP Configuration */}
        <fieldset>
          <legend>MCP Configuration</legend>
          <p className="fieldset-help">
            Configure Model Context Protocol (MCP) access and authentication settings.
          </p>

          <div className="form-group">
            <label className="chat-toggle-control" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={formData.mcp_enabled ?? settings?.mcp_enabled ?? false}
                  onChange={(e) =>
                    setFormData({ ...formData, mcp_enabled: e.target.checked })
                  }
                />
                <span className="toggle-slider"></span>
              </label>
              <span>Enable MCP Server</span>
            </label>
            <p className="field-help">
              When enabled, the MCP server endpoints (<code>/mcp</code> and custom routes) will be active.
              Disable to prevent all MCP access.
            </p>
          </div>

          {/* Only show other MCP settings when enabled */}
          {(formData.mcp_enabled ?? settings?.mcp_enabled ?? false) && (
            <>
              <div className="form-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth ?? false}
                    onChange={(e) =>
                      setFormData({ ...formData, mcp_default_route_auth: e.target.checked })
                    }
                    style={{ marginRight: '0.5rem' }}
                  />
                  <span>Require authentication for default /mcp route</span>
                </label>
                <p className="field-help">
                  When enabled, the default <code>/mcp</code> endpoint requires authentication.
                  {(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'oauth2'
                    ? ' MCP clients must authenticate via OAuth2 using the /auth/oauth2/token endpoint.'
                    : settings?.has_mcp_default_password
                      ? ' A password is configured - MCP clients should use this password as the Bearer token.'
                      : ' Set a password below to enable password-based authentication.'}
                </p>
              </div>

              {/* Auth method selection - only show when LDAP is configured and auth is enabled */}
              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && ldapConfig?.server_url && (
                <div className="form-group" style={{ marginTop: '1rem' }}>
                  <label>Authentication Method</label>
                  <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="mcp_auth_method"
                        value="password"
                        checked={(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'password'}
                        onChange={() => setFormData({ ...formData, mcp_default_route_auth_method: 'password' })}
                      />
                      <span>Password</span>
                    </label>
                    <label className="radio-label">
                      <input
                        type="radio"
                        name="mcp_auth_method"
                        value="oauth2"
                        checked={(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'oauth2'}
                        onChange={() => setFormData({ ...formData, mcp_default_route_auth_method: 'oauth2' })}
                      />
                      <span>OAuth2 (LDAP)</span>
                    </label>
                  </div>
                  <p className="field-help">
                    {(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'oauth2'
                      ? 'MCP clients authenticate with LDAP credentials via POST /auth/oauth2/token to get a Bearer token.'
                      : 'MCP clients use a static password as the Bearer token or MCP-Password header.'}
                  </p>
                </div>
              )}

              {/* LDAP Group restriction - only for OAuth2 auth method */}
              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) &&
               ldapConfig?.server_url &&
               (formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'oauth2' && (
                <div className="form-group" style={{ marginTop: '1rem' }}>
                  <label htmlFor="mcp-allowed-group">Allowed LDAP Group (Optional)</label>
                  <select
                    id="mcp-allowed-group"
                    value={formData.mcp_default_route_allowed_group ?? settings?.mcp_default_route_allowed_group ?? ''}
                    onChange={(e) =>
                      setFormData({ ...formData, mcp_default_route_allowed_group: e.target.value || null })
                    }
                    style={{ maxWidth: '500px' }}
                  >
                    <option value="">Any authenticated LDAP user</option>
                    {ldapDiscoveredGroups.map((g) => (
                      <option key={g.dn} value={g.dn}>{g.name}</option>
                    ))}
                  </select>
                  <p className="field-help">
                    Restrict access to members of a specific LDAP group. Leave empty to allow all authenticated LDAP users.
                  </p>
                </div>
              )}

              {/* Warning when auth is disabled */}
              {!(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && (
                <div className="field-warning" style={{ marginTop: '0.5rem', padding: '0.75rem', backgroundColor: 'rgba(255, 193, 7, 0.15)', borderLeft: '3px solid #ffc107', borderRadius: '4px' }}>
                  <strong>Security Notice:</strong> The <code>/mcp</code> endpoint is currently open without authentication.
                  Anyone with network access can invoke your configured tools. Consider enabling authentication if this
                  server is accessible beyond localhost or a trusted network.
                </div>
              )}

              {/* Password for default MCP route - only show for password auth method */}
              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) &&
               (formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'password' && (
                <div className="form-group" style={{ marginTop: '1rem' }}>
                  <label htmlFor="mcp-password">MCP Password</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type={showMcpPassword ? 'text' : 'password'}
                      id="mcp-password"
                      placeholder={settings?.has_mcp_default_password ? '••••••••' : 'Enter password (min 8 characters)'}
                      value={formData.mcp_default_route_password ?? ''}
                      onChange={(e) =>
                        setFormData({ ...formData, mcp_default_route_password: e.target.value })
                  }
                  style={{ flex: 1, maxWidth: '400px' }}
                />
                <button
                  type="button"
                  className="btn btn-small"
                  onClick={() => setShowMcpPassword(!showMcpPassword)}
                  title={showMcpPassword ? 'Hide password' : 'Show password'}
                >
                  {showMcpPassword ? 'Hide' : 'Show'}
                </button>
                {settings?.has_mcp_default_password && (
                  <button
                    type="button"
                    className="btn btn-small btn-secondary"
                    onClick={() => setFormData({ ...formData, mcp_default_route_password: '' })}
                    title="Clear password (submit empty to remove)"
                  >
                    Clear
                  </button>
                )}
              </div>
              <p className="field-help">
                {settings?.has_mcp_default_password
                  ? 'Password is set. Leave blank to keep current password, or enter a new one to change it. Clear and save to remove password protection.'
                  : 'Set a password that MCP clients will use as their Bearer token. Minimum 8 characters.'}
              </p>
              {window.location.protocol === 'http:' && (
                <div className="field-warning" style={{ marginTop: '0.5rem', padding: '0.5rem', backgroundColor: 'rgba(255, 193, 7, 0.15)', borderLeft: '3px solid #ffc107', borderRadius: '4px', fontSize: '0.85em' }}>
                  <strong>Security:</strong> You are accessing over HTTP. MCP passwords will be transmitted in plaintext.
                  Consider using HTTPS via a reverse proxy for production deployments.
                </div>
              )}
              {mcpError && <p className="field-error">{mcpError}</p>}
            </div>
          )}

          {/* Show MCP error when password field is not visible */}
          {!(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && mcpError && (
            <p className="field-error" style={{ marginTop: '0.5rem' }}>{mcpError}</p>
          )}
            </>
          )}

          <div className="form-group" style={{ marginTop: '1rem', display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              className="btn"
              onClick={handleSaveMcp}
              disabled={mcpSaving}
            >
              {mcpSaving ? 'Saving...' : 'Save MCP Configuration'}
            </button>
            {(formData.mcp_enabled ?? settings?.mcp_enabled ?? false) && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowMcpRoutesPanel(true)}
              >
                Manage Custom Routes
              </button>
            )}
          </div>
        </fieldset>

        {/* Performance Configuration */}
        <fieldset
          id="setting-sequential_index_loading"
          className={highlightSetting === 'sequential_index_loading' ? 'highlight-setting' : ''}
        >
          <legend>Performance</legend>
          <p className="fieldset-help">
            Configure memory and loading behavior for FAISS indexes.
          </p>

          <div className="form-group">
            <label className="chat-toggle-control" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={formData.sequential_index_loading ?? settings?.sequential_index_loading ?? false}
                  onChange={(e) =>
                    setFormData({ ...formData, sequential_index_loading: e.target.checked })
                  }
                />
                <span className="toggle-slider"></span>
              </label>
              <span>Sequential Index Loading</span>
            </label>
            <p className="field-help">
              <strong>Parallel (default):</strong> All indexes load simultaneously for faster startup,
              but peak RAM is ~1.8x total index size during deserialization.
            </p>
            <p className="field-help">
              <strong>Sequential:</strong> Indexes load one at a time (smallest first), reducing peak
              memory to ~1.8x the largest index. Useful when RAM is limited or OOM errors occur on startup.
            </p>
          </div>

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSavePerformance}
              disabled={performanceSaving}
            >
              {performanceSaving ? 'Saving...' : 'Save Performance Settings'}
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
                    {(() => {
                      // Filter models first
                      const filtered = allAvailableModels.filter((model) =>
                        modelFilterText === '' ||
                        model.name.toLowerCase().includes(modelFilterText.toLowerCase()) ||
                        model.provider.toLowerCase().includes(modelFilterText.toLowerCase())
                      );

                      // Group them
                      const groups: Record<string, typeof filtered> = {};
                      filtered.forEach(m => {
                        const g = m.group || 'Other';
                        if (!groups[g]) groups[g] = [];
                        groups[g].push(m);
                      });

                      return Object.keys(groups).map(groupName => (
                         <div key={groupName} className="model-group">
                           <div className="model-group-header">{groupName}</div>
                           {groups[groupName].map((model) => (
                              <label key={model.id} className="model-filter-item" style={{
                                paddingLeft: '1rem',
                                backgroundColor: model.is_latest ? 'rgba(0,0,0,0.03)' : undefined,
                                fontWeight: model.is_latest ? 500 : undefined
                                }}>
                                <input
                                  type="checkbox"
                                  checked={selectedModels.has(model.id)}
                                  onChange={() => toggleModel(model.id)}
                                />
                                <span className="model-filter-name">
                                  {model.name}
                                  {model.is_latest && <span style={{ marginLeft: '6px', fontSize: '0.7em', padding: '1px 4px', borderRadius: '4px', background: '#e0e0e0', color: '#555' }}>LATEST</span>}
                                </span>
                              </label>
                           ))}
                         </div>
                      ));
                    })()}
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

      {/* MCP Routes Panel Modal */}
      {showMcpRoutesPanel && (
        <div className="modal-overlay" onClick={() => setShowMcpRoutesPanel(false)}>
          <div className="modal-content modal-large" onClick={(e) => e.stopPropagation()}>
            <MCPRoutesPanel
              ldapConfigured={!!ldapConfig?.server_url}
              ldapGroups={ldapDiscoveredGroups}
              onClose={async () => {
              setShowMcpRoutesPanel(false);
              // Refresh routes list
              try {
                const routesRes = await api.listMcpRoutes();
                setMcpRoutes(routesRes.routes);
              } catch {
                // Silent fail
              }
            }} />
          </div>
        </div>
      )}
    </div>
  );
}
