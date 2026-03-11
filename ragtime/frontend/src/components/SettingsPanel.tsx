import { LdapGroupSelect } from './LdapGroupSelect';
import { useState, useEffect, useCallback, useRef } from 'react';
import { Lock, LockOpen, Info, Search, Clipboard, ExternalLink, X } from 'lucide-react';
import { api } from '@/api';
import type { AppSettings, UpdateSettingsRequest, OllamaModel, OllamaVisionModel, LLMModel, EmbeddingModel, AvailableModel, LdapConfig, McpRouteConfig, AuthStatus, CopilotAuthStatusResponse } from '@/types';
import { MCPRoutesPanel } from './MCPRoutesPanel';
import { OllamaConnectionForm } from './OllamaConnectionForm';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';

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

function normalizeSettingsSearchText(value: string): string {
  return value.trim().toLowerCase();
}

function settingsTextMatchesQuery(text: string | null | undefined, queries: string[]): boolean {
  if (queries.length === 0) {
    return true;
  }
  const normalized = normalizeSettingsSearchText(text || '');
  return queries.some((q) => normalized.includes(q));
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
  const { refresh: refreshModels } = useAvailableModels();
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [settingsFilterTags, setSettingsFilterTags] = useState<string[]>([]);
  const [settingsFilterInput, setSettingsFilterInput] = useState('');
  const [debouncedFilterInput, setDebouncedFilterInput] = useState('');
  const [settingsFilterHasMatches, setSettingsFilterHasMatches] = useState(true);
  const settingsFilterInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedFilterInput(settingsFilterInput), 200);
    return () => clearTimeout(timer);
  }, [settingsFilterInput]);

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

  // GitHub Copilot auth state
  const [copilotAuthStatus, setCopilotAuthStatus] = useState<CopilotAuthStatusResponse | null>(null);
  const [copilotConnecting, setCopilotConnecting] = useState(false);
  const [copilotDeviceCode, setCopilotDeviceCode] = useState<string>('');
  const [copilotVerificationUri, setCopilotVerificationUri] = useState<string>('');
  const [copilotRequestId, setCopilotRequestId] = useState<string | null>(null);
  const [copilotCodeCopied, setCopilotCodeCopied] = useState(false);
  const [copilotWizardVisible, setCopilotWizardVisible] = useState(false);
  const [copilotWizardStep, setCopilotWizardStep] = useState<1 | 2 | 3>(1);
  const [copilotAuthMode, setCopilotAuthMode] = useState<'oauth' | 'pat'>('oauth');
  const copilotPollTimerRef = useRef<number | null>(null);
  const copilotPollGenerationRef = useRef(0);

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

  // OpenAPI model filter modal state
  const [showOpenapiModelModal, setShowOpenapiModelModal] = useState(false);
  const [selectedOpenapiModels, setSelectedOpenapiModels] = useState<Set<string>>(new Set());
  const [openapiModelsLoading, setOpenapiModelsLoading] = useState(false);
  const [openapiModelFilterText, setOpenapiModelFilterText] = useState('');
  const [openapiAvailableModels, setOpenapiAvailableModels] = useState<AvailableModel[]>([]);

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
  const settingsFormRef = useRef<HTMLFormElement | null>(null);

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
    provider: 'openai' | 'anthropic' | 'github_copilot',
    apiKey?: string,
    options?: {
      authMode?: 'oauth' | 'pat';
      includeDirectoryModels?: boolean;
      includeAnthropicModels?: boolean;
      includeGoogleModels?: boolean;
    }
  ) => {
    if ((provider === 'openai' || provider === 'anthropic') && (!apiKey || apiKey.length < 10)) {
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
        auth_mode: options?.authMode,
        include_directory_models: options?.includeDirectoryModels,
        include_anthropic_models: options?.includeAnthropicModels,
        include_google_models: options?.includeGoogleModels,
      });

      if (response.success) {
        setLlmModels(response.models);
        setLlmModelsLoaded(true);
        // Auto-select a default model without capturing formData in callback deps.
        if (response.default_model) {
          setFormData((prev) => {
            const currentModel = prev.llm_model;
            const modelExists = response.models.some((m) => m.id === currentModel);
            if (!currentModel || !modelExists) {
              return {
                ...prev,
                llm_model: response.default_model,
              };
            }
            return prev;
          });
        }
      } else {
        setLlmModelsError(response.message);
      }
    } catch (err) {
      setLlmModelsError(err instanceof Error ? err.message : 'Failed to fetch models');
    } finally {
      setLlmModelsFetching(false);
    }
  }, []);

  const refreshCopilotStatus = useCallback(async () => {
    try {
      const status = await api.getCopilotAuthStatus();
      setCopilotAuthStatus(status);
      setFormData((prev) => ({
        ...prev,
        github_copilot_base_url: status.base_url,
        github_copilot_enterprise_url: status.enterprise_url ?? null,
      }));
      return status;
    } catch {
      setCopilotAuthStatus(null);
      return null;
    }
  }, []);

  const clearCopilotPollTimer = useCallback(() => {
    copilotPollGenerationRef.current += 1;
    if (copilotPollTimerRef.current !== null) {
      window.clearTimeout(copilotPollTimerRef.current);
      copilotPollTimerRef.current = null;
    }
  }, []);

  const pollCopilotDeviceFlow = useCallback(async (requestId: string, delaySeconds: number, generation: number) => {
    clearCopilotPollTimer();
    copilotPollGenerationRef.current = generation;
    copilotPollTimerRef.current = window.setTimeout(async () => {
      if (copilotPollGenerationRef.current !== generation) {
        return;
      }

      try {
        const response = await api.pollCopilotDeviceFlow({ request_id: requestId });
        if (response.status === 'pending') {
          await pollCopilotDeviceFlow(requestId, response.retry_after_seconds || 5, generation);
          return;
        }

        if (response.status === 'connected') {
          setCopilotConnecting(false);
          setCopilotRequestId(null);
          setCopilotDeviceCode('');
          setCopilotVerificationUri('');
          setCopilotCodeCopied(false);
          setCopilotWizardVisible(false);
          setCopilotWizardStep(1);
          await refreshCopilotStatus();
          setSuccess('GitHub Copilot connected successfully');
          setTimeout(() => setSuccess(null), 3000);
          const selectedProvider = formData.llm_provider || 'openai';
          if (selectedProvider === 'github_copilot') {
            await fetchLlmModels('github_copilot', undefined, {
              authMode: copilotAuthMode,
              includeDirectoryModels: true,
              includeAnthropicModels: true,
              includeGoogleModels: true,
            });
          }
          return;
        }

        setCopilotConnecting(false);
        setCopilotRequestId(null);
        setCopilotWizardVisible(false);
        setCopilotWizardStep(1);
        setLlmModelsError(response.message || 'GitHub Copilot authorization failed');
      } catch (err) {
        setCopilotConnecting(false);
        setCopilotRequestId(null);
        setCopilotWizardVisible(false);
        setCopilotWizardStep(1);
        const status = typeof err === 'object' && err !== null && 'status' in err
          ? (err as { status?: number }).status
          : undefined;
        if (status === 404) {
          setLlmModelsError('GitHub Copilot authorization session expired or server reloaded. Click Connect again.');
        } else {
          setLlmModelsError(err instanceof Error ? err.message : 'GitHub Copilot authorization failed');
        }
      }
    }, Math.max(delaySeconds, 1) * 1000);
  }, [
    clearCopilotPollTimer,
    fetchLlmModels,
    formData.llm_provider,
    refreshCopilotStatus,
    copilotAuthMode,
  ]);

  const startCopilotDeviceFlow = useCallback(async () => {
    setLlmModelsError(null);
    setCopilotConnecting(true);
    clearCopilotPollTimer();
    setCopilotCodeCopied(false);
    setCopilotWizardVisible(false);
    setCopilotWizardStep(1);

    try {
      const response = await api.startCopilotDeviceFlow({ deployment_type: 'github.com' });
      if (!response.verification_uri) {
        throw new Error('GitHub did not return an authorization URL');
      }
      setCopilotRequestId(response.request_id);
      setCopilotDeviceCode(response.user_code);
      setCopilotVerificationUri(response.verification_uri);
      setCopilotWizardVisible(true);
      setCopilotWizardStep(1);
      const pollGeneration = copilotPollGenerationRef.current + 1;
      copilotPollGenerationRef.current = pollGeneration;
      await pollCopilotDeviceFlow(response.request_id, response.interval || 5, pollGeneration);
    } catch (err) {
      setCopilotConnecting(false);
      setCopilotRequestId(null);
      setCopilotWizardVisible(false);
      setCopilotWizardStep(1);
      setLlmModelsError(err instanceof Error ? err.message : 'Failed to start GitHub Copilot authorization');
    }
  }, [clearCopilotPollTimer, pollCopilotDeviceFlow]);

  const clearCopilotAuth = useCallback(async () => {
    clearCopilotPollTimer();
    setCopilotConnecting(false);
    setCopilotRequestId(null);
    setCopilotDeviceCode('');
    setCopilotVerificationUri('');
    setCopilotCodeCopied(false);
    setCopilotWizardVisible(false);
    setCopilotWizardStep(1);
    try {
      await api.clearCopilotAuth();
      await refreshCopilotStatus();
      setLlmModels([]);
      setLlmModelsLoaded(false);
      setSuccess('GitHub Copilot connection removed');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setLlmModelsError(err instanceof Error ? err.message : 'Failed to clear GitHub Copilot auth');
    }
  }, [clearCopilotPollTimer, refreshCopilotStatus]);

  const copyCopilotDeviceCode = useCallback(async () => {
    if (!copilotDeviceCode) {
      return;
    }

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(copilotDeviceCode);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = copilotDeviceCode;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'absolute';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setSuccess('Device code copied');
      setCopilotCodeCopied(true);
      setTimeout(() => setCopilotCodeCopied(false), 2000);
      setTimeout(() => setSuccess(null), 2000);
    } catch {
      setLlmModelsError('Unable to copy device code. Please copy it manually.');
    }
  }, [copilotDeviceCode]);

  const openCopilotAuthorizationPage = useCallback(() => {
    if (!copilotVerificationUri) {
      return;
    }
    window.open(copilotVerificationUri, '_blank');
    setCopilotWizardStep(3);
  }, [copilotVerificationUri]);

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

  // --- Shared model-fetching helpers for Chat & OpenAPI modal ---
  const fetchModelsForModal = useCallback(async (): Promise<{ models: AvailableModel[]; response: Awaited<ReturnType<typeof api.getAllModels>> }> => {
    const response = await api.getAllModels();
    let models: AvailableModel[] = response.models.map((model) => ({
      ...model,
      provider: model.provider === 'github_models' ? 'github_copilot' : model.provider,
    }));

    const copilotPatToken = (formData.github_models_api_token || settings?.github_models_api_token || '').trim();
    const copilotConnected = Boolean(copilotAuthStatus?.connected || settings?.has_github_copilot_auth);
    const hasSelectedAuth = copilotAuthMode === 'pat' ? Boolean(copilotPatToken) : copilotConnected;
    if (hasSelectedAuth) {
      const githubResponse = await api.fetchLLMModels({
        provider: 'github_copilot',
        auth_mode: copilotAuthMode,
        include_directory_models: true,
        include_anthropic_models: true,
        include_google_models: true,
      });
      if (githubResponse.success) {
        const contextLimitById = new Map(models.map((m) => [m.id, m.context_limit]));
        const nonGithubModels = models.filter((m) => m.provider !== 'github_copilot');
        const githubModels: AvailableModel[] = githubResponse.models.map((m) => ({
          id: m.id,
          name: m.name,
          provider: 'github_copilot',
          context_limit: contextLimitById.get(m.id) ?? 200000,
          max_output_tokens: m.max_output_tokens,
          group: m.group,
          is_latest: m.is_latest,
        }));
        models = [...nonGithubModels, ...githubModels];
      }
    }

    return { models, response };
  }, [
    copilotAuthMode,
    copilotAuthStatus?.connected,
    formData.github_models_api_token,
    settings?.github_models_api_token,
    settings?.has_github_copilot_auth,
  ]);

  const initSelectedFromAllowed = (models: AvailableModel[], allowedModels: string[]): Set<string> => {
    const toScopedKey = (model: AvailableModel): string => `${model.provider}::${model.id}`;
    if (allowedModels.length > 0) {
      const hasScopedEntries = allowedModels.some((value) => value.includes('::'));
      if (hasScopedEntries) {
        return new Set(allowedModels);
      }
      const legacyIds = new Set(allowedModels);
      return new Set(
        models.filter((model) => legacyIds.has(model.id)).map((model) => toScopedKey(model))
      );
    }
    return new Set(models.map((m) => toScopedKey(m)));
  };

  // Open model filter modal and load all available models
  const openModelFilterModal = useCallback(async () => {
    setModelsLoading(true);
    setShowModelFilterModal(true);
    setModelFilterText('');

    try {
      const { models, response } = await fetchModelsForModal();
      setAllAvailableModels(models);
      const allowedModels = response.allowed_models || [];
      setSelectedModels(initSelectedFromAllowed(models, allowedModels));
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setModelsLoading(false);
    }
  }, [fetchModelsForModal]);

  const toggleModel = (model: AvailableModel) => {
    const selectionKey = `${model.provider}::${model.id}`;
    setSelectedModels(prev => {
      const next = new Set(prev);
      if (next.has(selectionKey)) {
        next.delete(selectionKey);
      } else {
        // Keep one provider binding per model ID so provider priority is explicit.
        for (const key of Array.from(next)) {
          const delimiter = key.indexOf('::');
          const keyModelId = delimiter >= 0 ? key.slice(delimiter + 2) : key;
          if (keyModelId === model.id) {
            next.delete(key);
          }
        }
        next.add(selectionKey);
      }
      return next;
    });
  };

  const selectAllModels = () => {
    setSelectedModels(new Set(allAvailableModels.map((m) => `${m.provider}::${m.id}`)));
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
      refreshModels();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save model filter');
    }
  };

  const openOpenapiModelModal = useCallback(async () => {
    setOpenapiModelsLoading(true);
    setShowOpenapiModelModal(true);
    setOpenapiModelFilterText('');

    try {
      const { models, response } = await fetchModelsForModal();
      setOpenapiAvailableModels(models);
      const allowedOpenapiModels = response.allowed_openapi_models || [];
      setSelectedOpenapiModels(initSelectedFromAllowed(models, allowedOpenapiModels));
    } catch (err) {
      console.error('Failed to load models:', err);
    } finally {
      setOpenapiModelsLoading(false);
    }
  }, [fetchModelsForModal]);

  const toggleOpenapiModel = (model: AvailableModel) => {
    const selectionKey = `${model.provider}::${model.id}`;
    setSelectedOpenapiModels(prev => {
      const next = new Set(prev);
      if (next.has(selectionKey)) {
        next.delete(selectionKey);
      } else {
        for (const key of Array.from(next)) {
          const delimiter = key.indexOf('::');
          const keyModelId = delimiter >= 0 ? key.slice(delimiter + 2) : key;
          if (keyModelId === model.id) {
            next.delete(key);
          }
        }
        next.add(selectionKey);
      }
      return next;
    });
  };

  const selectAllOpenapiModels = () => {
    setSelectedOpenapiModels(new Set(openapiAvailableModels.map((m) => `${m.provider}::${m.id}`)));
  };

  const deselectAllOpenapiModels = () => {
    setSelectedOpenapiModels(new Set());
  };

  const saveOpenapiModelFilter = async () => {
    const allSelected = selectedOpenapiModels.size === openapiAvailableModels.length;
    const allowedModels = allSelected ? [] : Array.from(selectedOpenapiModels);

    try {
      await api.updateSettings({ allowed_openapi_models: allowedModels });
      setShowOpenapiModelModal(false);
      setSuccess('OpenAPI model filter saved');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save OpenAPI model filter');
    }
  };

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const { settings: data } = await api.getSettings();
      setSettings(data);
      const normalizedLlmProvider = data.llm_provider === 'github_models' ? 'github_copilot' : data.llm_provider;
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
        llm_provider: normalizedLlmProvider,
        llm_model: data.llm_model,
        llm_max_tokens: data.llm_max_tokens,
        llm_ollama_protocol: data.llm_ollama_protocol,
        llm_ollama_host: data.llm_ollama_host,
        llm_ollama_port: data.llm_ollama_port,
        llm_ollama_base_url: data.llm_ollama_base_url,
        openai_api_key: data.openai_api_key,
        anthropic_api_key: data.anthropic_api_key,
        github_models_api_token: data.github_models_api_token,
        github_copilot_base_url: data.github_copilot_base_url,
        github_copilot_enterprise_url: data.github_copilot_enterprise_url,
        max_iterations: data.max_iterations,
        // Token optimization settings
        max_tool_output_chars: data.max_tool_output_chars,
        scratchpad_window_size: data.scratchpad_window_size,
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
        // OCR settings
        default_ocr_mode: data.default_ocr_mode,
        default_ocr_vision_model: data.default_ocr_vision_model,
        ocr_concurrency_limit: data.ocr_concurrency_limit,
        // User Space settings
        snapshot_retention_days: data.snapshot_retention_days,
        // OpenAPI model settings
        openapi_sync_chat_models: data.openapi_sync_chat_models,
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
      clearCopilotPollTimer();
      setCopilotConnecting(false);
      setCopilotRequestId(null);
      setCopilotDeviceCode('');
      setCopilotVerificationUri('');
      setError(null);
      setCopilotAuthMode(data.github_models_api_token ? 'pat' : 'oauth');

      const copilotStatus = await refreshCopilotStatus();

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

      if (normalizedLlmProvider === 'github_copilot') {
        if (data.github_models_api_token || copilotStatus?.connected) {
          fetchLlmModels('github_copilot', undefined, {
            authMode: data.github_models_api_token ? 'pat' : 'oauth',
            includeDirectoryModels: true,
            includeAnthropicModels: true,
            includeGoogleModels: true,
          });
        }
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
  }, [
    clearCopilotPollTimer,
    fetchLlmModels,
    refreshCopilotStatus,
    testLlmOllamaConnection,
    testOllamaConnection,
  ]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  useEffect(() => {
    const normalizedProvider = formData.llm_provider === 'github_models' ? 'github_copilot' : formData.llm_provider;
    if (normalizedProvider !== 'github_copilot') {
      return;
    }

    const copilotPatToken = (formData.github_models_api_token || settings?.github_models_api_token || '').trim();
    const hasCopilotAuth = copilotAuthMode === 'pat'
      ? Boolean(copilotPatToken)
      : Boolean(copilotAuthStatus?.connected || settings?.has_github_copilot_auth);
    if (!hasCopilotAuth) {
      return;
    }

    void fetchLlmModels('github_copilot', undefined, {
      authMode: copilotAuthMode,
      includeDirectoryModels: true,
      includeAnthropicModels: true,
      includeGoogleModels: true,
    });

    if (showModelFilterModal) {
      void openModelFilterModal();
    }
  }, [
    copilotAuthMode,
    copilotAuthStatus?.connected,
    fetchLlmModels,
    formData.github_models_api_token,
    formData.llm_provider,
    openModelFilterModal,
    settings?.github_models_api_token,
    settings?.has_github_copilot_auth,
    showModelFilterModal,
  ]);

  useEffect(() => {
    return () => {
      clearCopilotPollTimer();
    };
  }, [clearCopilotPollTimer]);

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
      const normalizedProvider = formData.llm_provider === 'github_models' ? 'github_copilot' : formData.llm_provider;
      const dataToSave: Record<string, unknown> = {
        llm_provider: normalizedProvider,
        llm_model: formData.llm_model,
        llm_max_tokens: formData.llm_max_tokens,
        openai_api_key: formData.openai_api_key,
        anthropic_api_key: formData.anthropic_api_key,
        github_models_api_token: formData.github_models_api_token,
        github_copilot_base_url: formData.github_copilot_base_url,
        github_copilot_enterprise_url: formData.github_copilot_enterprise_url,
        allowed_chat_models: formData.allowed_chat_models,
        max_iterations: formData.max_iterations,
        // OpenAPI model settings
        openapi_sync_chat_models: formData.openapi_sync_chat_models,
        // Token optimization settings
        max_tool_output_chars: formData.max_tool_output_chars,
        scratchpad_window_size: formData.scratchpad_window_size,
        context_token_budget: formData.context_token_budget,
      };

      if (normalizedProvider === 'github_copilot') {
        if (copilotAuthMode === 'pat') {
          dataToSave.github_copilot_access_token = '';
          dataToSave.github_copilot_refresh_token = '';
          dataToSave.github_copilot_token_expires_at = null;
        } else {
          dataToSave.github_models_api_token = '';
        }
      }
      // Include LLM Ollama connection fields when using Ollama provider
      if (normalizedProvider === 'ollama') {
        dataToSave.llm_ollama_protocol = formData.llm_ollama_protocol;
        dataToSave.llm_ollama_host = formData.llm_ollama_host;
        dataToSave.llm_ollama_port = formData.llm_ollama_port;
        dataToSave.llm_ollama_base_url = `${formData.llm_ollama_protocol || 'http'}://${formData.llm_ollama_host || 'localhost'}:${formData.llm_ollama_port || 11434}`;
      }
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setSuccess('LLM configuration saved');
      refreshModels();
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

  // OCR Configuration
  const [ocrSaving, setOcrSaving] = useState(false);

  // User Space Configuration
  const [userspaceSaving, setUserspaceSaving] = useState(false);
  const [visionModels, setVisionModels] = useState<OllamaVisionModel[]>([]);
  const [visionModelsLoading, setVisionModelsLoading] = useState(false);
  const [visionModelsError, setVisionModelsError] = useState<string | null>(null);
  const [showOcrRecommendations, setShowOcrRecommendations] = useState(false);

  // Fetch vision models when OCR mode is set to 'ollama'
  const fetchVisionModels = useCallback(async () => {
    if (!formData.ollama_protocol || !formData.ollama_host || !formData.ollama_port) {
      return;
    }

    setVisionModelsLoading(true);
    setVisionModelsError(null);

    try {
      const response = await api.getOllamaVisionModels({
        protocol: formData.ollama_protocol as 'http' | 'https',
        host: formData.ollama_host,
        port: formData.ollama_port,
      });

      if (response.success) {
        setVisionModels(response.models);
        if (response.models.length === 0) {
          setVisionModelsError('No vision-capable models found. Pull a vision model like llava or granite3.2-vision.');
        }
      } else {
        setVisionModelsError(response.message);
      }
    } catch (err) {
      setVisionModelsError(err instanceof Error ? err.message : 'Failed to fetch vision models');
    } finally {
      setVisionModelsLoading(false);
    }
  }, [formData.ollama_protocol, formData.ollama_host, formData.ollama_port]);

  // Auto-fetch vision models when OCR mode changes to 'ollama'
  useEffect(() => {
    if (formData.default_ocr_mode === 'ollama' && visionModels.length === 0 && !visionModelsLoading) {
      fetchVisionModels();
    }
  }, [formData.default_ocr_mode, fetchVisionModels, visionModels.length, visionModelsLoading]);

  const handleSaveOcr = async () => {
    setOcrSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave: UpdateSettingsRequest = {
        default_ocr_mode: formData.default_ocr_mode,
        default_ocr_vision_model: formData.default_ocr_vision_model,
        ocr_concurrency_limit: formData.ocr_concurrency_limit,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setFormData(prev => ({
        ...prev,
        default_ocr_mode: updated.default_ocr_mode,
        default_ocr_vision_model: updated.default_ocr_vision_model,
        ocr_concurrency_limit: updated.ocr_concurrency_limit,
      }));
      setSuccess('OCR settings saved.');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save OCR settings');
    } finally {
      setOcrSaving(false);
    }
  };

  const handleSaveUserspace = async () => {
    setUserspaceSaving(true);
    setSuccess(null);
    setError(null);

    try {
      const dataToSave: UpdateSettingsRequest = {
        snapshot_retention_days: formData.snapshot_retention_days,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setFormData(prev => ({
        ...prev,
        snapshot_retention_days: updated.snapshot_retention_days,
      }));
      setSuccess('User Space settings saved.');
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save User Space settings');
    } finally {
      setUserspaceSaving(false);
    }
  };

  const getDisplayUrl = (path: string) => {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = window.location.port === '8001' ? '8000' : window.location.port;
    const host = port ? `${hostname}:${port}` : hostname;
    return `${protocol}//${host}${path}`;
  };

  useEffect(() => {
    const form = settingsFormRef.current;
    if (!form) {
      return;
    }

    const liveInput = normalizeSettingsSearchText(debouncedFilterInput);
    const queries = [...settingsFilterTags.map(normalizeSettingsSearchText), ...(liveInput ? [liveInput] : [])].filter(Boolean);
    const infoCards = form.parentElement?.querySelectorAll<HTMLElement>('[data-settings-filter-card="true"]') || [];

    if (queries.length === 0) {
      infoCards.forEach((card) => {
        card.style.display = '';
      });

      const fieldsets = form.querySelectorAll<HTMLElement>('fieldset');
      fieldsets.forEach((fieldset) => {
        fieldset.style.display = '';
        fieldset.querySelectorAll<HTMLElement>('.form-group, .form-actions, details, .fieldset-help').forEach((element) => {
          element.style.display = '';
        });

        fieldset.querySelectorAll<HTMLDetailsElement>('details[data-filter-opened="true"]').forEach((details) => {
          details.open = false;
          details.removeAttribute('data-filter-opened');
        });
      });

      setSettingsFilterHasMatches(true);
      return;
    }

    let hasAnyMatches = false;

    infoCards.forEach((card) => {
      const cardText = card.textContent || '';
      const isMatch = settingsTextMatchesQuery(cardText, queries);
      card.style.display = isMatch ? '' : 'none';
      if (isMatch) {
        hasAnyMatches = true;
      }
    });

    const fieldsets = form.querySelectorAll<HTMLElement>('fieldset');
    fieldsets.forEach((fieldset) => {
      const legendText = fieldset.querySelector('legend')?.textContent || '';
      const helpText = fieldset.querySelector('.fieldset-help')?.textContent || '';
      const fieldsetTextMatch = settingsTextMatchesQuery(`${legendText} ${helpText}`, queries);

      let visibleFormGroupCount = 0;
      const formGroups = Array.from(fieldset.querySelectorAll<HTMLElement>('.form-group'));
      formGroups.forEach((group) => {
        const labelText = group.querySelector('label')?.textContent || '';
        const groupText = group.textContent || '';
        const isMatch = fieldsetTextMatch || settingsTextMatchesQuery(`${labelText} ${groupText}`, queries);
        group.style.display = isMatch ? '' : 'none';
        if (isMatch) {
          visibleFormGroupCount += 1;
        }
      });

      const detailsElements = Array.from(fieldset.querySelectorAll<HTMLDetailsElement>('details'));
      detailsElements.forEach((details) => {
        const summaryText = details.querySelector('summary')?.textContent || '';
        const detailText = details.textContent || '';
        const hasVisibleChild = Array.from(details.querySelectorAll<HTMLElement>('.form-group')).some((group) => group.style.display !== 'none');
        const detailsMatch = fieldsetTextMatch || hasVisibleChild || settingsTextMatchesQuery(`${summaryText} ${detailText}`, queries);
        details.style.display = detailsMatch ? '' : 'none';
        if (detailsMatch && hasVisibleChild && !details.open) {
          details.open = true;
          details.setAttribute('data-filter-opened', 'true');
        }
      });

      const fieldsetHelp = fieldset.querySelector<HTMLElement>('.fieldset-help');
      if (fieldsetHelp) {
        const helpMatch = fieldsetTextMatch || settingsTextMatchesQuery(fieldsetHelp.textContent, queries);
        fieldsetHelp.style.display = helpMatch ? '' : 'none';
      }

      const formActions = Array.from(fieldset.querySelectorAll<HTMLElement>('.form-actions'));
      formActions.forEach((actions) => {
        actions.style.display = visibleFormGroupCount > 0 || fieldsetTextMatch ? '' : 'none';
      });

      const showFieldset = fieldsetTextMatch || visibleFormGroupCount > 0;
      fieldset.style.display = showFieldset ? '' : 'none';
      if (showFieldset) {
        hasAnyMatches = true;
      }
    });

    setSettingsFilterHasMatches(hasAnyMatches);
  }, [settingsFilterTags, debouncedFilterInput, loading]);

  if (loading) {
    return (
      <div className="card">
        <h2>Settings</h2>
        <p className="muted">Loading settings...</p>
      </div>
    );
  }

  const openAiConfigured = Boolean((formData.openai_api_key ?? settings?.openai_api_key)?.trim());
  const claudeConfigured = Boolean((formData.anthropic_api_key ?? settings?.anthropic_api_key)?.trim());
  const copilotConfigured = Boolean(copilotAuthStatus?.connected ?? settings?.has_github_copilot_auth);
  const copilotPatToken = (formData.github_models_api_token ?? settings?.github_models_api_token ?? '').trim();
  const hasCopilotPatToken = Boolean(copilotPatToken);
  const copilotPatConfigured = Boolean(
    copilotPatToken
  );
  const ollamaConfigured = Boolean(
    (formData.llm_ollama_protocol ?? settings?.llm_ollama_protocol) &&
    (formData.llm_ollama_host ?? settings?.llm_ollama_host)?.trim() &&
    (formData.llm_ollama_port ?? settings?.llm_ollama_port)
  );
  const embeddingOpenAiConfigured = Boolean((formData.openai_api_key ?? settings?.openai_api_key)?.trim());
  const embeddingOllamaConfigured = Boolean(
    (formData.ollama_protocol ?? settings?.ollama_protocol) &&
    (formData.ollama_host ?? settings?.ollama_host)?.trim() &&
    (formData.ollama_port ?? settings?.ollama_port)
  );

  return (
    <div className="card">
      <h2>Settings</h2>

      <div className="settings-filter-search" role="search" aria-label="Filter settings" onClick={() => settingsFilterInputRef.current?.focus()}>
        <Search size={16} className="settings-filter-search-icon" aria-hidden="true" />
        {settingsFilterTags.map((tag, i) => (
          <span key={i} className="settings-filter-tag">
            {tag}
            <button
              type="button"
              className="settings-filter-tag-remove"
              onClick={(e) => { e.stopPropagation(); setSettingsFilterTags((prev) => prev.filter((_, idx) => idx !== i)); }}
              aria-label={`Remove filter: ${tag}`}
            >
              <X size={12} />
            </button>
          </span>
        ))}
        <input
          ref={settingsFilterInputRef}
          type="text"
          placeholder={settingsFilterTags.length === 0 ? 'Filter settings by keyword...' : ''}
          value={settingsFilterInput}
          onChange={(e) => {
            const val = e.target.value;
            if (val.endsWith(',')) {
              const tag = val.slice(0, -1).trim();
              if (tag && !settingsFilterTags.includes(tag)) {
                setSettingsFilterTags((prev) => [...prev, tag]);
              }
              setSettingsFilterInput('');
            } else {
              setSettingsFilterInput(val);
            }
          }}
          onKeyDown={(e) => {
            if (e.key === 'Tab' && settingsFilterInput.trim()) {
              e.preventDefault();
              const tag = settingsFilterInput.trim();
              if (!settingsFilterTags.includes(tag)) {
                setSettingsFilterTags((prev) => [...prev, tag]);
              }
              setSettingsFilterInput('');
            }
            if (e.key === 'Backspace' && !settingsFilterInput && settingsFilterTags.length > 0) {
              setSettingsFilterTags((prev) => prev.slice(0, -1));
            }
          }}
          onBlur={() => {
            const tag = settingsFilterInput.trim();
            if (tag && !settingsFilterTags.includes(tag)) {
              setSettingsFilterTags((prev) => [...prev, tag]);
            }
            setSettingsFilterInput('');
          }}
          aria-label="Filter settings by keyword"
        />
        {(settingsFilterTags.length > 0 || settingsFilterInput.trim()) && (
          <button
            type="button"
            className="settings-filter-clear"
            onClick={(e) => { e.stopPropagation(); setSettingsFilterTags([]); setSettingsFilterInput(''); }}
            aria-label="Clear all filters"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {!settingsFilterHasMatches && (settingsFilterTags.length > 0 || debouncedFilterInput.trim()) && (
        <p className="muted settings-filter-empty">No settings match the current filters.</p>
      )}

      {/* API Endpoint Info */}
      <div
        className={`api-info-box ${highlightSetting === 'api_key_info' ? 'highlight-setting' : ''}`}
        id="setting-api_key_info"
        data-settings-filter-card="true"
      >
        <strong>OpenAI-Compatible API</strong>
        <p>
          Connect external clients (e.g., Open WebUI) using:
        </p>
        <code>{getDisplayUrl('/v1')}</code>
        <p className="field-help" style={{ marginTop: '0.5rem' }}>
          Default model: <code>{formData.llm_model || settings?.llm_model || 'not configured'}</code>. <code>/v1/models</code> returns {formData.openapi_sync_chat_models !== false ? 'your Chat Models selection' : 'a separately configured OpenAPI models list'}.
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
      <div className="api-info-box" data-settings-filter-card="true">
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

      <form ref={settingsFormRef} onSubmit={handleSubmit}>
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

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0, marginTop: 'var(--space-md)' }}>
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
          <legend className="legend-with-status">
            <span>LLM Configuration (Chat/RAG)</span>
            <span className="legend-divider" aria-hidden="true" />
            <span className="llm-provider-status-inline" aria-label="LLM provider configuration status">
              <span className="llm-provider-status-item" title={openAiConfigured ? 'OpenAI configured' : 'OpenAI not configured'}>
                <span
                  className={`llm-provider-status-dot ${openAiConfigured ? 'configured' : ''}`}
                  aria-label={openAiConfigured ? 'OpenAI configured' : 'OpenAI not configured'}
                />
                <span className="llm-provider-status-label">OpenAI</span>
              </span>
              <span className="llm-provider-status-item" title={claudeConfigured ? 'Claude configured' : 'Claude not configured'}>
                <span
                  className={`llm-provider-status-dot ${claudeConfigured ? 'configured' : ''}`}
                  aria-label={claudeConfigured ? 'Claude configured' : 'Claude not configured'}
                />
                <span className="llm-provider-status-label">Claude</span>
              </span>
              <span className="llm-provider-status-item" title={ollamaConfigured ? 'Ollama configured' : 'Ollama not configured'}>
                <span
                  className={`llm-provider-status-dot ${ollamaConfigured ? 'configured' : ''}`}
                  aria-label={ollamaConfigured ? 'Ollama configured' : 'Ollama not configured'}
                />
                <span className="llm-provider-status-label">Ollama</span>
              </span>
              <span className="llm-provider-status-item" title={(copilotConfigured || copilotPatConfigured) ? 'GitHub Copilot configured' : 'GitHub Copilot not configured'}>
                <span
                  className={`llm-provider-status-dot ${(copilotConfigured || copilotPatConfigured) ? 'configured' : ''}`}
                  aria-label={(copilotConfigured || copilotPatConfigured) ? 'GitHub Copilot configured' : 'GitHub Copilot not configured'}
                />
                <span className="llm-provider-status-label">Copilot</span>
              </span>
            </span>
          </legend>
          <p className="fieldset-help">
            Configure the language model used for answering questions and tool calls.
          </p>

          <div className="form-group">
            <label>Provider</label>
            <select
              value={formData.llm_provider || 'openai'}
              onChange={(e) => {
                const newProvider = e.target.value as 'openai' | 'anthropic' | 'ollama' | 'github_copilot';
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

                if (newProvider === 'github_copilot' && ((copilotAuthMode === 'oauth' && (copilotAuthStatus?.connected || settings?.has_github_copilot_auth)) || (copilotAuthMode === 'pat' && hasCopilotPatToken))) {
                  fetchLlmModels('github_copilot', undefined, {
                    authMode: copilotAuthMode,
                    includeDirectoryModels: true,
                    includeAnthropicModels: true,
                    includeGoogleModels: true,
                  });
                }
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="ollama">Ollama</option>
              <option value="github_copilot">GitHub Copilot</option>
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
          ) : formData.llm_provider === 'github_copilot' ? (
            <div className="form-group">
              <label>GitHub Copilot Connection</label>
              <div className="form-row" style={{ marginBottom: '0.75rem' }}>
                <label style={{ marginRight: '0.75rem' }}>Authentication</label>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  <label style={{ display: 'inline-flex', gap: '0.35rem', alignItems: 'center', marginBottom: 0 }}>
                    <input
                      type="radio"
                      name="copilot-auth-mode"
                      checked={copilotAuthMode === 'oauth'}
                      onChange={() => {
                        setCopilotAuthMode('oauth');
                        setFormData((prev) => ({ ...prev, github_models_api_token: '' }));
                        setLlmModels([]);
                        setLlmModelsError(null);
                        setLlmModelsLoaded(false);
                      }}
                    />
                    OAuth (GitHub device login)
                  </label>
                  <label style={{ display: 'inline-flex', gap: '0.35rem', alignItems: 'center', marginBottom: 0 }}>
                    <input
                      type="radio"
                      name="copilot-auth-mode"
                      checked={copilotAuthMode === 'pat'}
                      onChange={() => {
                        setCopilotAuthMode('pat');
                        clearCopilotPollTimer();
                        setCopilotConnecting(false);
                        setCopilotRequestId(null);
                        setCopilotWizardVisible(false);
                        setCopilotWizardStep(1);
                        setLlmModels([]);
                        setLlmModelsError(null);
                        setLlmModelsLoaded(false);
                      }}
                    />
                    PAT (Copilot models)
                  </label>
                </div>
              </div>
              {copilotAuthMode === 'pat' && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <input
                    type="password"
                    value={formData.github_models_api_token || ''}
                    onChange={(e) => {
                      setFormData({ ...formData, github_models_api_token: e.target.value });
                      setLlmModels([]);
                      setLlmModelsError(null);
                      setLlmModelsLoaded(false);
                    }}
                    placeholder="github_pat_..."
                  />
                  <p className="field-help">
                    Use a fine-grained GitHub token with the `Models:read` permission. Stored encrypted in backend settings.
                  </p>
                </div>
              )}
              <div className="input-with-button input-with-actions" style={{ gap: '0.5rem', flexWrap: 'wrap' }}>
                {copilotAuthMode === 'oauth' && (
                  <button
                    type="button"
                    className={`btn btn-test ${copilotConfigured ? 'btn-connected' : ''}`}
                    onClick={startCopilotDeviceFlow}
                    disabled={copilotConnecting}
                  >
                    {copilotConnecting ? 'Preparing...' : copilotConfigured ? 'Reauthorize' : 'Authorize'}
                  </button>
                )}
                <button
                  type="button"
                  className={`btn btn-test ${llmModelsLoaded && formData.llm_provider === 'github_copilot' ? 'btn-connected' : ''}`}
                  onClick={() => fetchLlmModels('github_copilot', undefined, {
                    authMode: copilotAuthMode,
                    includeDirectoryModels: true,
                    includeAnthropicModels: true,
                    includeGoogleModels: true,
                  })}
                  disabled={llmModelsFetching || (copilotAuthMode === 'oauth' ? !copilotConfigured : !hasCopilotPatToken)}
                >
                  {llmModelsFetching ? 'Fetching...' : llmModelsLoaded && formData.llm_provider === 'github_copilot' ? 'Loaded' : 'Fetch Models'}
                </button>
                {copilotAuthMode === 'oauth' && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={clearCopilotAuth}
                    disabled={copilotConnecting || !copilotConfigured}
                  >
                    Disconnect
                  </button>
                )}
              </div>
              {copilotAuthMode === 'oauth' && copilotWizardVisible && copilotRequestId && copilotDeviceCode && copilotVerificationUri && (
                <div
                  className="field-help"
                  style={{
                    marginTop: '0.75rem',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: '0.5rem' }}>GitHub Authorization</div>
                  {copilotWizardStep === 1 && (
                    <div>
                      <div><strong>Step 1: Copy your device code</strong></div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.45rem', flexWrap: 'wrap' }}>
                        <code style={{ fontSize: '1.1rem', fontWeight: 700, letterSpacing: '0.08em', padding: '0.35rem 0.55rem' }}>
                          {copilotDeviceCode}
                        </code>
                        <button
                          type="button"
                          onClick={copyCopilotDeviceCode}
                          aria-label="Copy device code"
                          title="Copy device code"
                          className="btn btn-sm btn-secondary"
                          style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '0.35rem' }}
                        >
                          <Clipboard size={16} />
                        </button>
                        {copilotCodeCopied && <span className="muted">Copied</span>}
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() => setCopilotWizardStep(2)}
                          disabled={!copilotCodeCopied}
                          style={{ marginLeft: '0.25rem' }}
                        >
                          Continue
                        </button>
                      </div>
                    </div>
                  )}

                  {copilotWizardStep === 2 && (
                    <div>
                      <div><strong>Step 2: Open the authorization page</strong></div>
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={openCopilotAuthorizationPage}
                        style={{
                          marginTop: '0.45rem',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.4rem',
                          fontSize: '1.05rem',
                          fontWeight: 700,
                        }}
                      >
                        Open GitHub Authorization
                        <ExternalLink size={16} />
                      </button>
                      <div className="muted" style={{ marginTop: '0.45rem' }}>{copilotVerificationUri}</div>
                      <div style={{ marginTop: '0.65rem' }}>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => setCopilotWizardStep(1)}
                        >
                          Back
                        </button>
                      </div>
                    </div>
                  )}

                  {copilotWizardStep === 3 && (
                    <div>
                      <div><strong>Step 3: Complete authorization in GitHub</strong></div>
                      <div className="muted" style={{ marginTop: '0.45rem' }}>
                        After you approve access in GitHub, Ragtime will connect automatically.
                      </div>
                      <div style={{ marginTop: '0.65rem' }}>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={openCopilotAuthorizationPage}
                        >
                          Reopen Authorization Page
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
              {llmModelsError && formData.llm_provider === 'github_copilot' && (
                <p className="field-error">{llmModelsError}</p>
              )}
              <p className="field-help">
                {copilotAuthMode === 'oauth'
                  ? 'OAuth uses GitHub device authorization and is required to access models included with your GitHub Copilot subscription.'
                  : 'PAT mode uses your personal GitHub token (Models:read) with the GitHub Models API. PAT mode does not grant Copilot subscription model access.'}
              </p>
            </div>
          ) : null}

          <div className="form-row">
            {/* Chat Model Filter */}
            <div className="form-group" style={{ flex: 1 }}>
              <label>Chat Models</label>
              <button
                type="button"
                className="btn btn-secondary settings-control-height"
                onClick={openModelFilterModal}
              >
                Configure Chat Models
              </button>
              <p className="field-help">
                Limit which models appear in the Chat view dropdown. Includes all configured providers (OpenAI, Anthropic, Ollama, GitHub Copilot).
              </p>
            </div>

            {/* OpenAPI Models configuration */}
            <div className="form-group" style={{ flex: 1 }}>
              <label>OpenAPI Models</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.9em', margin: 0, whiteSpace: 'nowrap' }}>
                  <input
                    type="checkbox"
                    checked={formData.openapi_sync_chat_models !== false}
                    onChange={(e) =>
                      setFormData({ ...formData, openapi_sync_chat_models: e.target.checked })
                    }
                  />
                  Mirror Chat Models
                </label>
                {formData.openapi_sync_chat_models === false && (
                  <button
                    type="button"
                    className="btn btn-secondary settings-control-height"
                    onClick={openOpenapiModelModal}
                  >
                    Configure OpenAPI Models
                  </button>
                )}
              </div>
              <p className="field-help">
                {formData.openapi_sync_chat_models !== false
                  ? 'The /v1/models endpoint returns the same models as Chat Models above.'
                  : 'Configure a separate list of models exposed via the /v1/models endpoint for external clients.'}
              </p>
            </div>
          </div>

          {/* Show OpenAI key field for embeddings if using Anthropic or Ollama for LLM */}
          {(formData.llm_provider === 'anthropic' || formData.llm_provider === 'ollama' || formData.llm_provider === 'github_copilot') && formData.embedding_provider === 'openai' && (
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

          {/* Advanced Context & Token Settings */}
          <details style={{ marginBottom: '16px' }} id="setting-llm_advanced">
            <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Settings</summary>

            <div className="form-row">
              <div className="form-group" style={{ flex: 2 }}>
                <label>Max Output Tokens</label>
                {(() => {
                  const selectedLlmModel = llmModels.find(m => m.id === formData.llm_model);
                  const selectedAvailableModel = allAvailableModels.find(m => m.id === formData.llm_model);
                  const modelMax = selectedLlmModel?.max_output_tokens
                    || selectedAvailableModel?.max_output_tokens
                    || 100000;
                  const sliderMax = modelMax;
                  const sliderMin = 500;
                  const hasModelInfo = !!(selectedLlmModel?.max_output_tokens || selectedAvailableModel?.max_output_tokens);

                  return (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          style={{ flex: 1 }}
                          value={(() => {
                            const val = formData.llm_max_tokens || 4096;
                            if (val >= sliderMax) return 100;
                            const scale = Math.log(sliderMax / sliderMin);
                            return Math.max(0, Math.min(100, (Math.log(val / sliderMin) / scale) * 100));
                          })()}
                          onChange={(e) => {
                            const slider = parseInt(e.target.value, 10);
                            let val;
                            if (slider === 100) {
                              val = sliderMax;
                            } else {
                              const scale = Math.log(sliderMax / sliderMin);
                              val = Math.round(sliderMin * Math.exp((slider / 100) * scale));
                            }
                            setFormData({ ...formData, llm_max_tokens: val });
                          }}
                        />
                        <span style={{ minWidth: '80px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                          {formData.llm_max_tokens && formData.llm_max_tokens >= sliderMax ? 'LLM Max' : (formData.llm_max_tokens || 4096).toLocaleString()}
                        </span>
                      </div>
                      <p className="field-help">
                        Limit the length of the model's response.{hasModelInfo ? ` (Model max: ${modelMax.toLocaleString()})` : ''}
                      </p>
                    </>
                  );
                })()}
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
                  <span style={{ minWidth: '30px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {formData.max_iterations ?? 30}
                  </span>
                </div>
                <p className="field-help">
                  Maximum number of agent tool-calling steps.
                </p>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Max Tool Output (chars)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="0"
                    max="50000"
                    step="1000"
                    style={{ flex: 1 }}
                    value={formData.max_tool_output_chars ?? 5000}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        max_tool_output_chars: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '60px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.max_tool_output_chars ?? 5000) === 0 ? 'Off' : `${((formData.max_tool_output_chars ?? 5000) / 1000).toFixed(0)}K`}
                  </span>
                </div>
                <p className="field-help">
                  Cap on each tool response before truncation (0 = no limit).
                  Lower values curb token growth during multi-step tool loops.
                </p>
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label>Context Window (steps)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="0"
                    max="30"
                    step="1"
                    style={{ flex: 1 }}
                    value={formData.scratchpad_window_size ?? 6}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        scratchpad_window_size: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '40px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.scratchpad_window_size ?? 6) === 0 ? 'All' : formData.scratchpad_window_size ?? 6}
                  </span>
                </div>
                <p className="field-help">
                  Number of recent tool steps kept in full detail; older steps are compressed (0 = keep all).
                  Smaller windows reduce input tokens in long conversations.
                </p>
              </div>
            </div>

            <div className="form-group">
              <label>Context Token Budget</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="range"
                  min="0"
                  max="32000"
                  step="500"
                  style={{ flex: 1 }}
                  value={formData.context_token_budget ?? settings?.context_token_budget ?? 4000}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      context_token_budget: parseInt(e.target.value, 10),
                    })
                  }
                />
                <span style={{ minWidth: '60px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {(formData.context_token_budget ?? settings?.context_token_budget ?? 4000) === 0 ? 'Off' : `${((formData.context_token_budget ?? settings?.context_token_budget ?? 4000) / 1000).toFixed(1)}K`}
                </span>
              </div>
              <p className="field-help">
                Cap on retrieved context tokens fed to the LLM per request (0 = unlimited).
                Lower values reduce input token usage; higher values give the model more knowledge to draw from.
              </p>
            </div>
          </details>

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
          <legend className="legend-with-status">
            <span>Embedding Configuration</span>
            <span className="legend-divider" aria-hidden="true" />
            <span className="llm-provider-status-inline" aria-label="Embedding provider configuration status">
              <span className="llm-provider-status-item" title={embeddingOpenAiConfigured ? 'OpenAI configured' : 'OpenAI not configured'}>
                <span
                  className={`llm-provider-status-dot ${embeddingOpenAiConfigured ? 'configured' : ''}`}
                  aria-label={embeddingOpenAiConfigured ? 'OpenAI configured' : 'OpenAI not configured'}
                />
                <span className="llm-provider-status-label">OpenAI</span>
              </span>
              <span className="llm-provider-status-item" title={embeddingOllamaConfigured ? 'Ollama configured' : 'Ollama not configured'}>
                <span
                  className={`llm-provider-status-dot ${embeddingOllamaConfigured ? 'configured' : ''}`}
                  aria-label={embeddingOllamaConfigured ? 'Ollama configured' : 'Ollama not configured'}
                />
                <span className="llm-provider-status-label">Ollama</span>
              </span>
            </span>
          </legend>
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
                    fontFamily: 'var(--font-mono)',
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
                <div className="form-group" style={{ flex: 1 }}>
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
                    <p className="field-help" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                      DN: {ldapFormData.user_search_base}
                    </p>
                  )}
                </div>

                <div className="form-group" style={{ flex: 1 }}>
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
                Document chunks retrieved per query (k).
                Lower (3-5) is faster; higher (10-20) gives more context but costs more tokens.
              </p>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
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
                </p>

                {(formData.search_use_mmr ?? settings?.search_use_mmr ?? true) && (
                  <div style={{ marginTop: '0.5rem' }}>
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
                      <strong>0 = Max diversity</strong> |
                      <strong> 1 = Max relevance</strong>.
                      Default 0.5 provides a good balance.
                    </p>
                  </div>
                )}
              </div>

              <div className="form-group" style={{ flex: 1 }}>
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

                <div style={{ marginTop: '0.5rem' }}>
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
                    pgvector index parameter. Higher = faster queries for large datasets.
                    Recommended: sqrt(num embeddings). Default: 100.
                  </p>
                </div>
              </div>
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
                    <div style={{ maxWidth: '500px' }}>
                      <LdapGroupSelect
                        id="mcp-allowed-group"
                        value={formData.mcp_default_route_allowed_group ?? settings?.mcp_default_route_allowed_group ?? ''}
                        onChange={(value) =>
                          setFormData({ ...formData, mcp_default_route_allowed_group: value || null })
                        }
                        groups={ldapDiscoveredGroups}
                        emptyOptionLabel="Any authenticated LDAP user"
                      />
                    </div>
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

        {/* OCR Configuration */}
        <fieldset id="setting-ocr">
          <legend>OCR Settings</legend>
          <p className="fieldset-help">
            Configure default OCR (Optical Character Recognition) mode for extracting text from images during indexing.
          </p>

          <div className="form-row" style={formData.default_ocr_mode === 'ollama' ? { display: 'flex', flexWrap: 'nowrap', gap: 'var(--space-md)' } : undefined}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Default OCR Mode</label>
              <select
                value={formData.default_ocr_mode || 'disabled'}
                onChange={(e) => {
                  const newMode = e.target.value as 'disabled' | 'tesseract' | 'ollama';
                  setFormData({ ...formData, default_ocr_mode: newMode });
                  // Clear vision model error when changing mode
                  if (newMode !== 'ollama') {
                    setVisionModelsError(null);
                  }
                }}
              >
                <option value="disabled">Disabled (skip images)</option>
                <option value="tesseract">Tesseract (fast, traditional OCR)</option>
                <option value="ollama">Ollama Vision (semantic OCR with AI)</option>
              </select>
              <p className="field-help">
                {formData.default_ocr_mode === 'disabled' && (
                  <>Image files will be skipped during indexing.</>
                )}
                {formData.default_ocr_mode === 'tesseract' && (
                  <>Fast traditional OCR using Tesseract. Good for screenshots and scanned documents with clear text.</>
                )}
                {formData.default_ocr_mode === 'ollama' && (
                  <>
                    Semantic OCR using Ollama vision models. Better at understanding complex layouts, handwriting, and context.
                  </>
                )}
              </p>
            </div>

            {formData.default_ocr_mode === 'ollama' && (
              <div className="form-group" style={{ flex: 1 }}>
                <label>Vision Model</label>
                {visionModelsLoading ? (
                  <p className="muted">Loading vision models...</p>
                ) : visionModelsError ? (
                  <div>
                    <p className="error-text" style={{ marginBottom: '8px' }}>{visionModelsError}</p>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={fetchVisionModels}
                    >
                      Retry
                    </button>
                  </div>
                ) : visionModels.length > 0 ? (
                  <select
                    value={formData.default_ocr_vision_model || ''}
                    onChange={(e) => setFormData({ ...formData, default_ocr_vision_model: e.target.value || null })}
                  >
                    <option value="">Select a vision model</option>
                    {visionModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name}
                        {model.parameter_size && ` (${model.parameter_size})`}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div>
                    <p className="muted">No vision models loaded.</p>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={fetchVisionModels}
                    >
                      Load Vision Models
                    </button>
                  </div>
                )}
                <p className="field-help">
                  Select an Ollama vision model for semantic OCR.
                </p>
              </div>
            )}

            {formData.default_ocr_mode === 'ollama' && (
              <div className="form-group" style={{ flex: '0 0 120px' }}>
                <label>Concurrency</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={formData.ocr_concurrency_limit ?? 1}
                  onChange={(e) => setFormData({ ...formData, ocr_concurrency_limit: Math.max(1, Math.min(10, parseInt(e.target.value) || 1)) })}
                  style={{ width: '80px' }}
                />
                <p className="field-help">
                  Parallel OCR requests. Higher values use more VRAM.
                </p>
              </div>
            )}
          </div>

          {formData.default_ocr_mode === 'ollama' && (
            <div className="form-group" style={{ marginTop: '-0.5rem', marginBottom: '1rem' }}>
              <p className="field-help">
                <span style={{ color: 'var(--warning-color, #b58900)' }}>
                  <strong>Performance note:</strong> Vision models are 3-15x slower than Tesseract depending on model size.
                  <button
                    type="button"
                    onClick={() => setShowOcrRecommendations(!showOcrRecommendations)}
                    title="View model recommendations"
                    style={{
                      background: 'none',
                      border: 'none',
                      cursor: 'pointer',
                      marginLeft: '4px',
                      padding: 0,
                      color: 'inherit',
                      verticalAlign: 'middle',
                      display: 'inline-flex',
                      alignItems: 'center',
                    }}
                  >
                    <Info size="1em" />
                  </button>
                </span>
                {showOcrRecommendations && (
                  <div style={{
                    marginTop: '12px',
                    padding: '12px',
                    backgroundColor: 'var(--input-bg, var(--bg-secondary, #f5f5f5))',
                    border: '1px solid var(--border-color, #ddd)',
                    borderRadius: '6px',
                    fontSize: '0.9em',
                    color: 'var(--text-color, inherit)',
                  }}>
                    <strong>Recommended:</strong> <code>llama3.2-vision</code> (10.7B)<br />
                    Best balance of speed and accuracy, ~6x slower than Tesseract.
                    <br /><br />
                    <strong>Other options:</strong><br />
                    <code>qwen3-vl</code> (8.8B) - Highest accuracy, cleanest output, but ~14x slower.<br />
                    <code>llava</code> (7B) - Fastest (~2x slower), but may hallucinate on document OCR.
                  </div>
                )}
              </p>
            </div>
          )}

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSaveOcr}
              disabled={ocrSaving || (formData.default_ocr_mode === 'ollama' && !formData.default_ocr_vision_model)}
            >
              {ocrSaving ? 'Saving...' : 'Save OCR Settings'}
            </button>
            {formData.default_ocr_mode === 'ollama' && !formData.default_ocr_vision_model && (
              <span className="muted" style={{ marginLeft: '1rem' }}>
                Select a vision model to save
              </span>
            )}
          </div>
        </fieldset>

        {/* User Space Configuration */}
        <fieldset>
          <legend>User Space</legend>

          <div className="form-group">
            <div className="form-row">
              <label>Snapshot Retention</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <input
                  type="number"
                  min={0}
                  max={3650}
                  value={formData.snapshot_retention_days ?? 0}
                  onChange={(e) => setFormData({ ...formData, snapshot_retention_days: Math.max(0, Math.min(3650, parseInt(e.target.value) || 0)) })}
                  style={{ width: '5rem' }}
                />
                <span>days</span>
              </div>
            </div>
            <p className="muted">
              How long to keep workspace snapshots. Set to 0 for unlimited retention.
              Snapshots older than this window will not appear in the snapshot list.
            </p>
          </div>

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleSaveUserspace}
              disabled={userspaceSaving}
            >
              {userspaceSaving ? 'Saving...' : 'Save User Space Settings'}
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
                  No models available. Please configure API keys or Ollama connection and save settings first.
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
                  <p className="field-help" style={{ margin: '0 0 0.5rem 0' }}>
                    If the same model exists from multiple providers, selecting a row sets the provider to use for that model.
                  </p>
                  <div className="model-filter-list">
                    {(() => {
                      // Filter models by name, id, and provider
                      const filterLower = modelFilterText.toLowerCase();
                      const filtered = allAvailableModels.filter((model) =>
                        modelFilterText === '' ||
                        model.name.toLowerCase().includes(filterLower) ||
                        model.id.toLowerCase().includes(filterLower) ||
                        model.provider.toLowerCase().includes(filterLower)
                      );

                      // Group by provider first, then by model group within provider
                      // Collect unique providers in order of appearance
                      const providerOrder: string[] = [];
                      const providerGroups: Record<string, Record<string, typeof filtered>> = {};
                      filtered.forEach(m => {
                        if (!providerGroups[m.provider]) {
                          providerGroups[m.provider] = {};
                          providerOrder.push(m.provider);
                        }
                        const g = m.group || 'Other';
                        if (!providerGroups[m.provider][g]) providerGroups[m.provider][g] = [];
                        providerGroups[m.provider][g].push(m);
                      });

                      const providerLabels: Record<string, string> = {
                        openai: 'OpenAI',
                        anthropic: 'Anthropic',
                        ollama: 'Ollama',
                        github_copilot: 'GitHub Copilot',
                        github_models: 'GitHub Copilot',
                      };

                      return providerOrder.map(provider => (
                        <div key={provider}>
                          <div className="model-group-header" style={{
                            position: 'sticky',
                            top: 0,
                            zIndex: 1,
                            padding: '6px 8px',
                            fontWeight: 600,
                            fontSize: '0.85em',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                            borderBottom: '1px solid var(--border-color, #3c3c3c)',
                            background: 'var(--bg-primary, #1e1e1e)',
                          }}>
                            {providerLabels[provider] || provider}
                          </div>
                          {Object.keys(providerGroups[provider]).map(groupName => (
                            <div key={groupName} className="model-group">
                              <div className="model-group-header" style={{ paddingLeft: '0.5rem', fontSize: '0.8em' }}>{groupName}</div>
                              {providerGroups[provider][groupName].map((model) => (
                                <label key={`${model.provider}:${model.id}`} className="model-filter-item" style={{
                                  paddingLeft: '1rem',
                                  backgroundColor: model.is_latest ? 'rgba(0,0,0,0.03)' : undefined,
                                  fontWeight: model.is_latest ? 500 : undefined
                                }}>
                                  <input
                                    type="checkbox"
                                    checked={selectedModels.has(`${model.provider}::${model.id}`)}
                                    onChange={() => toggleModel(model)}
                                  />
                                  <span className="model-filter-name">
                                    {model.id !== model.name ? model.id : model.name}
                                    <span style={{ marginLeft: '6px', fontSize: '0.7em', padding: '1px 4px', borderRadius: '4px', background: 'var(--bg-secondary, #2d2d2d)', color: 'var(--text-muted, #888)' }}>
                                      via {providerLabels[model.provider] || model.provider}
                                    </span>
                                    {model.is_latest && <span style={{ marginLeft: '6px', fontSize: '0.7em', padding: '1px 4px', borderRadius: '4px', background: '#e0e0e0', color: '#555' }}>LATEST</span>}
                                  </span>
                                </label>
                              ))}
                            </div>
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
      {showOpenapiModelModal && (
        <div className="modal-overlay" onClick={() => setShowOpenapiModelModal(false)}>
          <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Configure OpenAPI Models</h3>
              <button
                className="modal-close"
                onClick={() => setShowOpenapiModelModal(false)}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              {openapiModelsLoading ? (
                <p className="muted">Loading available models...</p>
              ) : openapiAvailableModels.length === 0 ? (
                <p className="muted">
                  No models available. Please configure API keys or Ollama connection and save settings first.
                </p>
              ) : (
                <>
                  <div className="model-filter-search">
                    <input
                      type="text"
                      placeholder="Filter models..."
                      value={openapiModelFilterText}
                      onChange={(e) => setOpenapiModelFilterText(e.target.value)}
                      autoFocus
                    />
                  </div>
                  <div className="model-filter-actions">
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={selectAllOpenapiModels}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={deselectAllOpenapiModels}
                    >
                      Deselect All
                    </button>
                    <span className="muted" style={{ marginLeft: 'auto' }}>
                      {selectedOpenapiModels.size} of {openapiAvailableModels.length} selected
                    </span>
                  </div>
                  <p className="field-help" style={{ margin: '0 0 0.5rem 0' }}>
                    Select models to expose via the <code>/v1/models</code> endpoint for external clients (e.g., Open WebUI).
                  </p>
                  <div className="model-filter-list">
                    {(() => {
                      const filterLower = openapiModelFilterText.toLowerCase();
                      const filtered = openapiAvailableModels.filter((model) =>
                        openapiModelFilterText === '' ||
                        model.name.toLowerCase().includes(filterLower) ||
                        model.id.toLowerCase().includes(filterLower) ||
                        model.provider.toLowerCase().includes(filterLower)
                      );

                      const providerOrder: string[] = [];
                      const providerGroups: Record<string, Record<string, typeof filtered>> = {};
                      filtered.forEach(m => {
                        if (!providerGroups[m.provider]) {
                          providerGroups[m.provider] = {};
                          providerOrder.push(m.provider);
                        }
                        const g = m.group || 'Other';
                        if (!providerGroups[m.provider][g]) providerGroups[m.provider][g] = [];
                        providerGroups[m.provider][g].push(m);
                      });

                      const providerLabels: Record<string, string> = {
                        openai: 'OpenAI',
                        anthropic: 'Anthropic',
                        ollama: 'Ollama',
                        github_copilot: 'GitHub Copilot',
                        github_models: 'GitHub Copilot',
                      };

                      return providerOrder.map(provider => (
                        <div key={provider}>
                          <div className="model-group-header" style={{
                            position: 'sticky',
                            top: 0,
                            zIndex: 1,
                            padding: '6px 8px',
                            fontWeight: 600,
                            fontSize: '0.85em',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                            borderBottom: '1px solid var(--border-color, #3c3c3c)',
                            background: 'var(--bg-primary, #1e1e1e)',
                          }}>
                            {providerLabels[provider] || provider}
                          </div>
                          {Object.keys(providerGroups[provider]).map(groupName => (
                            <div key={groupName} className="model-group">
                              <div className="model-group-header" style={{ paddingLeft: '0.5rem', fontSize: '0.8em' }}>{groupName}</div>
                              {providerGroups[provider][groupName].map((model) => (
                                <label key={`${model.provider}:${model.id}`} className="model-filter-item" style={{
                                  paddingLeft: '1rem',
                                  backgroundColor: model.is_latest ? 'rgba(0,0,0,0.03)' : undefined,
                                  fontWeight: model.is_latest ? 500 : undefined
                                }}>
                                  <input
                                    type="checkbox"
                                    checked={selectedOpenapiModels.has(`${model.provider}::${model.id}`)}
                                    onChange={() => toggleOpenapiModel(model)}
                                  />
                                  <span className="model-filter-name">
                                    {model.id !== model.name ? model.id : model.name}
                                    <span style={{ marginLeft: '6px', fontSize: '0.7em', padding: '1px 4px', borderRadius: '4px', background: 'var(--bg-secondary, #2d2d2d)', color: 'var(--text-muted, #888)' }}>
                                      via {providerLabels[model.provider] || model.provider}
                                    </span>
                                    {model.is_latest && <span style={{ marginLeft: '6px', fontSize: '0.7em', padding: '1px 4px', borderRadius: '4px', background: '#e0e0e0', color: '#555' }}>LATEST</span>}
                                  </span>
                                </label>
                              ))}
                            </div>
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
                onClick={() => setShowOpenapiModelModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn"
                onClick={saveOpenapiModelFilter}
                disabled={openapiModelsLoading || openapiAvailableModels.length === 0}
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
