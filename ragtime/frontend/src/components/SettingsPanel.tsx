import { LdapGroupSelect } from './LdapGroupSelect';
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Lock, LockOpen, Info, Search, ExternalLink, X, Eye, EyeOff, Pencil } from 'lucide-react';
import { api } from '@/api';
import type { AppSettings, UpdateSettingsRequest, OllamaModel, VisionModel, LLMModel, EmbeddingModel, AvailableModel, LdapConfig, McpRouteConfig, AuthStatus, AuthProviderConfig, AuthGroup, LdapUserProfile, CopilotAuthStatusResponse, UserSpacePreviewSettingsResponse, LlmProviderWire, UpsertUserSpaceWorkspaceEnvVarRequest, UserSpaceWorkspaceEnvVar, User, OcrMode, OcrProvider } from '@/types';
import { MCPRoutesPanel } from './MCPRoutesPanel';
import { OllamaConnectionForm } from './OllamaConnectionForm';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { Popover } from './Popover';
import { InlineCopyButton } from './shared/InlineCopyButton';
import { UserSpaceEnvVarsModal } from './shared/UserSpaceEnvVarsModal';
import { UserSpaceRuntimeRestartPanel } from './shared/UserSpaceRuntimeRestartPanel';
import { AuthAdminModalHost } from './shared/AuthAdminModals';
import { ModelFilterModal } from './ModelFilterModal';
import { CheckboxDropdown } from './shared/CheckboxDropdown';
import { OCR_PROVIDER_LABELS } from './OcrVectorStoreFields';
import { renderApiKeySecurityWarning, renderHttpSecurityWarning } from './shared/securityWarnings';
import { useToast, ToastContainer } from './shared/Toast';

import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import {
  buildUserSpacePreviewSandboxAttribute,
  getUserSpacePreviewSandboxFlagValues,
  normalizeUserSpacePreviewSandboxFlags,
} from '@/utils/userspacePreview/sandbox';
import { formatBytes } from '@/utils';
import {
  SQLITE_IMPORT_DEFAULT_MAX_BYTES,
  sqliteImportBytesToSlider,
  sliderToSqliteImportBytes,
} from '@/utils/sqliteImport';
import {
  MOUNT_SYNC_DEFAULT_SECONDS,
  formatMountSyncInterval,
  mountSyncIntervalToSlider,
  sliderToMountSyncInterval,
} from '@/utils/mountSyncIntervals';
import { CHAT_MODEL_PROVIDER_LABELS, parseScopedModelIdentifier } from '@/utils/modelDisplay';
import {
  PROVIDER_CONNECTIONS,
  buildProviderBaseUrl,
  normalizeProviderAlias,
  providersEquivalent,
  type ProviderConnectionDescriptor,
} from '@/utils/modelProviders';

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

function generateMcpCredentialValue(length: number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789';
  const randomValues = new Uint8Array(length);

  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(randomValues);
  } else {
    for (let index = 0; index < randomValues.length; index += 1) {
      randomValues[index] = Math.floor(Math.random() * alphabet.length);
    }
  }

  return Array.from(randomValues, (value) => alphabet[value % alphabet.length]).join('');
}

function generateMcpClientId(): string {
  return `cid-${generateMcpCredentialValue(12).toLowerCase()}`;
}

function generateMcpSecret(): string {
  return generateMcpCredentialValue(32);
}

function normalizeSettingsSearchText(value: string): string {
  return value.trim().toLowerCase();
}

const SETTINGS_FILTER_QUERY_PARAM = 'search';

function readSettingsFilterStateFromUrl(): { input: string; tags: string[] } {
  const params = new URLSearchParams(window.location.search);
  const rawSearch = params.get(SETTINGS_FILTER_QUERY_PARAM) || '';
  const tags = rawSearch
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  return { input: '', tags };
}

function writeSettingsFilterStateToUrl(input: string, tags: string[]): void {
  const params = new URLSearchParams(window.location.search);
  const searchValue = [...tags, input]
    .map((value) => value.trim())
    .filter(Boolean)
    .join(',');

  if (searchValue) {
    params.set(SETTINGS_FILTER_QUERY_PARAM, searchValue);
  } else {
    params.delete(SETTINGS_FILTER_QUERY_PARAM);
  }

  const nextSearch = params.toString();
  const nextUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}${window.location.hash}`;
  window.history.replaceState(null, '', nextUrl);
}

function getCloudOAuthCallbackUrl(): string {
  return new URL('/indexes/userspace/cloud-oauth/callback', window.location.origin).toString();
}

function renderCloudDriveOAuthSetupPopover(callbackUrl: string): JSX.Element {
  return (
    <div style={{ display: 'grid', gap: 8, maxWidth: 360 }}>
      <strong style={{ fontSize: '0.85rem' }}>Cloud drive OAuth setup</strong>
      <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>Register provider OAuth apps with this redirect URI:</span>
      <div className="cloud-oauth-callback-row">
        <code className="cloud-oauth-callback-code">{callbackUrl}</code>
        <InlineCopyButton
          copyText={callbackUrl}
          className="cloud-oauth-callback-copy"
          title="Copy redirect URI"
          ariaLabel="Copy redirect URI"
          copiedTitle="Redirect URI copied"
          copiedAriaLabel="Redirect URI copied"
          iconSize={12}
        />
      </div>
      <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
        <strong>Google Drive:</strong> enable the Google Drive API (<code>drive.googleapis.com</code>) for the OAuth client project and add scopes <code>https://www.googleapis.com/auth/drive</code> and <code>https://www.googleapis.com/auth/userinfo.email</code>.
      </span>
      <span style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>
        <strong>OneDrive/SharePoint:</strong> set <code>CLOUD_MOUNT_MICROSOFT_TENANT_ID</code> to your Azure Directory tenant ID or primary tenant domain for single-tenant apps, then add Microsoft Graph delegated permissions <code>offline_access</code>, <code>User.Read</code>, <code>Files.ReadWrite.All</code>, and <code>Sites.ReadWrite.All</code>. Tenant policy may require admin consent.
      </span>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <a href="https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade" target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>Microsoft apps</a>
        <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>Google credentials</a>
        <a href="https://console.cloud.google.com/apis/library/drive.googleapis.com" target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>Google Drive API</a>
      </div>
    </div>
  );
}

function settingsTextMatchesQuery(text: string | null | undefined, queries: string[]): boolean {
  if (queries.length === 0) {
    return true;
  }
  const normalized = normalizeSettingsSearchText(text || '');
  return queries.some((q) => normalized.includes(q));
}

function isSettingsSaveControlButton(button: HTMLButtonElement): boolean {
  return normalizeSettingsSearchText(button.textContent || '').includes('save');
}

const DEFAULT_OLLAMA_PORT = PROVIDER_CONNECTIONS.ollamaEmbedding.defaultPort;
const DEFAULT_OLLAMA_HOST = PROVIDER_CONNECTIONS.ollamaEmbedding.defaultHost;
const DEFAULT_OLLAMA_PROTOCOL = PROVIDER_CONNECTIONS.ollamaEmbedding.defaultProtocol;
const DEFAULT_OLLAMA_BASE_URL = PROVIDER_CONNECTIONS.ollamaEmbedding.defaultBaseUrl;
const DEFAULT_LLAMA_CPP_CHAT_PORT = PROVIDER_CONNECTIONS.llamaCppLlm.defaultPort;
const DEFAULT_LLAMA_CPP_EMBEDDING_PORT = PROVIDER_CONNECTIONS.llamaCppEmbedding.defaultPort;
const DEFAULT_LLAMA_CPP_HOST = PROVIDER_CONNECTIONS.llamaCppLlm.defaultHost;
const DEFAULT_LLAMA_CPP_PROTOCOL = PROVIDER_CONNECTIONS.llamaCppLlm.defaultProtocol;
const DEFAULT_LMSTUDIO_PORT = PROVIDER_CONNECTIONS.lmstudioLlm.defaultPort;
const DEFAULT_LMSTUDIO_HOST = PROVIDER_CONNECTIONS.lmstudioLlm.defaultHost;
const DEFAULT_LMSTUDIO_PROTOCOL = PROVIDER_CONNECTIONS.lmstudioLlm.defaultProtocol;
const DEFAULT_OMLX_PORT = PROVIDER_CONNECTIONS.omlxLlm.defaultPort;
const DEFAULT_OMLX_HOST = PROVIDER_CONNECTIONS.omlxLlm.defaultHost;
const DEFAULT_OMLX_PROTOCOL = PROVIDER_CONNECTIONS.omlxLlm.defaultProtocol;

const COPILOT_MODEL_FETCH_OPTIONS = {
  includeDirectoryModels: true,
  includeAnthropicModels: true,
  includeGoogleModels: true,
} as const;

function normalizeLlmProvider(provider: LlmProviderWire | null | undefined): Exclude<LlmProviderWire, 'github_models'> | null | undefined {
  return normalizeProviderAlias(provider) as Exclude<LlmProviderWire, 'github_models'> | null | undefined;
}

function buildOllamaBaseUrl(protocol?: 'http' | 'https' | null, host?: string | null, port?: number | null): string {
  return buildProviderBaseUrl(PROVIDER_CONNECTIONS.ollamaEmbedding, protocol, host, port);
}

function buildLocalBaseUrl(
  protocol: 'http' | 'https' | null | undefined,
  host: string | null | undefined,
  port: number | null | undefined,
  connection: ProviderConnectionDescriptor,
): string {
  return buildProviderBaseUrl(connection, protocol, host, port);
}

function isUnsetDefaultOllamaConnection(
  protocol: string | null | undefined,
  host: string | null | undefined,
  port: number | null | undefined,
  baseUrl: string | null | undefined,
): boolean {
  return (host || '').trim() === DEFAULT_OLLAMA_HOST
    && (protocol || DEFAULT_OLLAMA_PROTOCOL) === DEFAULT_OLLAMA_PROTOCOL
    && (port || DEFAULT_OLLAMA_PORT) === DEFAULT_OLLAMA_PORT
    && (baseUrl || DEFAULT_OLLAMA_BASE_URL) === DEFAULT_OLLAMA_BASE_URL;
}

function sanitizeOllamaDefaults(settings: AppSettings): AppSettings {
  const sanitized = { ...settings };

  if (isUnsetDefaultOllamaConnection(
    settings.ollama_protocol,
    settings.ollama_host,
    settings.ollama_port,
    settings.ollama_base_url,
  )) {
    sanitized.ollama_host = '';
  }

  if (isUnsetDefaultOllamaConnection(
    settings.llm_ollama_protocol,
    settings.llm_ollama_host,
    settings.llm_ollama_port,
    settings.llm_ollama_base_url,
  )) {
    sanitized.llm_ollama_host = '';
  }

  return sanitized;
}

function parseLdapServerUrl(serverUrl: string | null | undefined): { protocol: 'ldap' | 'ldaps'; host: string; port: number } {
  const defaults = { protocol: 'ldaps' as const, host: '', port: 636 };
  if (!serverUrl) {
    return defaults;
  }

  const match = serverUrl.match(/^(ldaps?):\/\/([^:]+)(?::(\d+))?$/);
  if (!match) {
    return defaults;
  }

  const protocol = match[1] as 'ldap' | 'ldaps';
  return {
    protocol,
    host: match[2],
    port: match[3] ? parseInt(match[3], 10) : (protocol === 'ldaps' ? 636 : 389),
  };
}

function getScopedModelId(selectionKey: string): string {
  const delimiter = selectionKey.indexOf('::');
  return delimiter >= 0 ? selectionKey.slice(delimiter + 2) : selectionKey;
}

function toggleScopedModelSelection(currentSelection: Set<string>, model: AvailableModel): Set<string> {
  const selectionKey = toScopedModelIdentifier(model);
  const nextSelection = new Set(currentSelection);

  if (nextSelection.has(selectionKey)) {
    nextSelection.delete(selectionKey);
    return nextSelection;
  }

  for (const key of Array.from(nextSelection)) {
    if (getScopedModelId(key) === model.id) {
      nextSelection.delete(key);
    }
  }

  nextSelection.add(selectionKey);
  return nextSelection;
}

const AUTH_PROVIDER_OPTIONS = [
  {
    value: 'local_managed',
    label: 'Internal Users',
    description: 'Manage users and groups stored in the local database. Authentication uses local password hashes.',
  },
  {
    value: 'ldap',
    label: 'LDAP / Active Directory',
    description: 'Configure the LDAP/AD server connection, lazy-sync behavior, and optionally pre-import LDAP identities into the local cache.',
  },
] as const;

function toScopedModelIdentifier(model: AvailableModel): string {
  return `${model.provider}::${model.id}`;
}

function formatModelIdentifierForDisplay(identifier: string | null | undefined, models: AvailableModel[]): string {
  const { provider, modelId } = parseScopedModelIdentifier(identifier);
  if (!modelId) {
    return 'not configured';
  }

  const exactMatch = models.find((m) => (
    m.id === modelId
    && provider
    && providersEquivalent(m.provider, provider)
  ));
  if (exactMatch) {
    const label = CHAT_MODEL_PROVIDER_LABELS[exactMatch.provider] || exactMatch.provider;
    return `${exactMatch.id} (${label})`;
  }

  const unscopedMatch = models.find((m) => m.id === modelId);
  if (unscopedMatch) {
    const label = CHAT_MODEL_PROVIDER_LABELS[unscopedMatch.provider] || unscopedMatch.provider;
    return `${unscopedMatch.id} (${label})`;
  }

  if (provider) {
    const label = CHAT_MODEL_PROVIDER_LABELS[provider] || provider;
    return `${modelId} (${label})`;
  }
  return modelId;
}

function getEmbeddingSettingsFormData(data: AppSettings): Pick<UpdateSettingsRequest,
  'embedding_provider'
  | 'embedding_model'
  | 'embedding_dimensions'
  | 'ollama_protocol'
  | 'ollama_host'
  | 'ollama_port'
  | 'ollama_base_url'
  | 'llama_cpp_protocol'
  | 'llama_cpp_host'
  | 'llama_cpp_port'
  | 'llama_cpp_base_url'
  | 'lmstudio_protocol'
  | 'lmstudio_host'
  | 'lmstudio_port'
  | 'lmstudio_base_url'
  | 'lmstudio_api_key'
  | 'omlx_protocol'
  | 'omlx_host'
  | 'omlx_port'
  | 'omlx_base_url'
  | 'omlx_api_key'
  | 'ollama_embedding_timeout_seconds'
  | 'default_ocr_mode'
  | 'default_ocr_provider'
  | 'default_ocr_vision_model'
  | 'ocr_concurrency_limit'
> {
  return {
    embedding_provider: data.embedding_provider,
    embedding_model: data.embedding_model,
    embedding_dimensions: data.embedding_dimensions,
    ollama_protocol: data.ollama_protocol,
    ollama_host: data.ollama_host,
    ollama_port: data.ollama_port,
    ollama_base_url: data.ollama_base_url,
    llama_cpp_protocol: data.llama_cpp_protocol,
    llama_cpp_host: data.llama_cpp_host,
    llama_cpp_port: data.llama_cpp_port,
    llama_cpp_base_url: data.llama_cpp_base_url,
    lmstudio_protocol: data.lmstudio_protocol,
    lmstudio_host: data.lmstudio_host,
    lmstudio_port: data.lmstudio_port,
    lmstudio_base_url: data.lmstudio_base_url,
    lmstudio_api_key: data.lmstudio_api_key,
    omlx_protocol: data.omlx_protocol,
    omlx_host: data.omlx_host,
    omlx_port: data.omlx_port,
    omlx_base_url: data.omlx_base_url,
    omlx_api_key: data.omlx_api_key,
    ollama_embedding_timeout_seconds: data.ollama_embedding_timeout_seconds,
    default_ocr_mode: data.default_ocr_mode,
    default_ocr_provider: data.default_ocr_provider || 'ollama',
    default_ocr_vision_model: data.default_ocr_vision_model,
    ocr_concurrency_limit: data.ocr_concurrency_limit,
  };
}

interface SettingsPanelProps {
  currentUser?: User | null;
  onServerNameChange?: (name: string) => void;
  onAuthenticatedWebglBackgroundChange?: (enabled: boolean) => void;
  /** Setting ID to highlight and scroll to (e.g., 'sequential_index_loading') */
  highlightSetting?: string | null;
  /** Called after highlight animation completes to clear the param */
  onHighlightComplete?: () => void;
  /** Auth status for security warnings */
  authStatus?: AuthStatus | null;
}

export function SettingsPanel({ currentUser, onServerNameChange, onAuthenticatedWebglBackgroundChange, highlightSetting, onHighlightComplete, authStatus }: SettingsPanelProps) {
  const { refresh: refreshModels } = useAvailableModels();
  const initialSettingsFilterState = useMemo(() => readSettingsFilterStateFromUrl(), []);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [userspacePreviewSettings, setUserspacePreviewSettings] = useState<UserSpacePreviewSettingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [toasts, toast] = useToast();
  const [settingsFilterTags, setSettingsFilterTags] = useState<string[]>(initialSettingsFilterState.tags);
  const [settingsFilterInput, setSettingsFilterInput] = useState(initialSettingsFilterState.input);
  const [debouncedFilterInput, setDebouncedFilterInput] = useState('');
  const [settingsFilterHasMatches, setSettingsFilterHasMatches] = useState(true);
  const settingsFilterInputRef = useRef<HTMLInputElement | null>(null);
  const [activeAuthProviderValue, setActiveAuthProviderValue] = useState<typeof AUTH_PROVIDER_OPTIONS[number]['value']>('local_managed');

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedFilterInput(settingsFilterInput), 200);
    return () => clearTimeout(timer);
  }, [settingsFilterInput]);

  useEffect(() => {
    writeSettingsFilterStateToUrl(settingsFilterInput, settingsFilterTags);
  }, [settingsFilterInput, settingsFilterTags]);

  useEffect(() => {
    if (loading) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      settingsFilterInputRef.current?.focus();
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [loading]);

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
  const [lmstudioModelActionLoading, setLmstudioModelActionLoading] = useState(false);

  // Model filter modal state (chat)
  const [showModelFilterModal, setShowModelFilterModal] = useState(false);
  const [allAvailableModels, setAllAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [modelsLoading, setModelsLoading] = useState(false);
  const [filteredChatModels, setFilteredChatModels] = useState<AvailableModel[]>([]);
  const [automaticDefaultChatModel, setAutomaticDefaultChatModel] = useState<string | null>(null);
  const [chatModelsLoading, setChatModelsLoading] = useState(false);

  // OpenAPI model filter modal state
  const [showOpenapiModelModal, setShowOpenapiModelModal] = useState(false);
  const [selectedOpenapiModels, setSelectedOpenapiModels] = useState<Set<string>>(new Set());
  const [openapiModelsLoading, setOpenapiModelsLoading] = useState(false);
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
    user_search_filter: '(uid={username})',
    admin_group_dns: [] as string[],
    user_group_dns: [] as string[],
  });
  const [ldapTesting, setLdapTesting] = useState(false);
  const [ldapTestResult, setLdapTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [ldapDiscoveredOus, setLdapDiscoveredOus] = useState<string[]>([]);
  const [ldapDiscoveredGroups, setLdapDiscoveredGroups] = useState<{ dn: string; name: string }[]>([]);
  const [authProviderConfig, setAuthProviderConfig] = useState<AuthProviderConfig | null>(null);
  const [authProviderConfigSaving, setAuthProviderConfigSaving] = useState(false);
  const [authGroups, setAuthGroups] = useState<AuthGroup[]>([]);
  const [showCreateLocalUserModal, setShowCreateLocalUserModal] = useState(false);
  const [showManageAuthGroupsModal, setShowManageAuthGroupsModal] = useState(false);
  const [ldapUserSearchName, setLdapUserSearchName] = useState('');
  const [ldapUserSearching, setLdapUserSearching] = useState(false);
  const [ldapUserSearchResults, setLdapUserSearchResults] = useState<LdapUserProfile[]>([]);
  const [showLdapUserSearchResults, setShowLdapUserSearchResults] = useState(false);
  const [suppressLdapUserSearchDropdown, setSuppressLdapUserSearchDropdown] = useState(false);
  const [ldapUserImporting, setLdapUserImporting] = useState(false);
  const [ldapUserPreview, setLdapUserPreview] = useState<LdapUserProfile | null>(null);
  const ldapUserSearchRequestSeqRef = useRef(0);
  const ldapUserSearchContainerRef = useRef<HTMLDivElement | null>(null);

  // Form state
  const [formData, setFormData] = useState<UpdateSettingsRequest>({});
  const settingsFormRef = useRef<HTMLFormElement | null>(null);

  // Track if we've already auto-tested Ollama
  const hasAutoTestedOllama = useRef(false);
  const hasAutoTestedLlmOllama = useRef(false);

  const resetEmbeddingOllamaState = useCallback(() => {
    setOllamaConnected(false);
    setOllamaError(null);
    setOllamaModels([]);
  }, []);

  const resetLlmOllamaState = useCallback(() => {
    setLlmOllamaConnected(false);
    setLlmOllamaError(null);
    setLlmOllamaModels([]);
  }, []);

  const resetLlmModelsState = useCallback(() => {
    setLlmModels([]);
    setLlmModelsError(null);
    setLlmModelsLoaded(false);
  }, []);

  const resetEmbeddingModelsState = useCallback(() => {
    setEmbeddingModels([]);
    setEmbeddingModelsError(null);
    setEmbeddingModelsLoaded(false);
  }, []);

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
        port: port || DEFAULT_OLLAMA_PORT,
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
        port: port || DEFAULT_OLLAMA_PORT,
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
    provider: 'openai' | 'anthropic' | 'openrouter' | 'llama_cpp' | 'lmstudio' | 'omlx' | 'github_copilot',
    apiKey?: string,
    options?: {
      authMode?: 'oauth' | 'pat';
      includeDirectoryModels?: boolean;
      includeAnthropicModels?: boolean;
      includeGoogleModels?: boolean;
      baseUrl?: string;
    }
  ) => {
    if ((provider === 'openai' || provider === 'anthropic' || provider === 'openrouter') && (!apiKey || apiKey.length < 10)) {
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
        base_url: options?.baseUrl,
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

  const fetchLocalLlmModels = useCallback(async (
    provider: 'llama_cpp' | 'lmstudio' | 'omlx',
    connection: ProviderConnectionDescriptor,
    protocol: 'http' | 'https' | null | undefined,
    host: string | null | undefined,
    port: number | null | undefined,
    apiKey?: string,
  ) => {
    await fetchLlmModels(provider, apiKey, {
      baseUrl: buildLocalBaseUrl(protocol, host, port, connection),
    });
  }, [fetchLlmModels]);

  const fetchLlamaCppLlmModels = useCallback(async () => {
    await fetchLocalLlmModels(
      'llama_cpp',
      PROVIDER_CONNECTIONS.llamaCppLlm,
      formData.llm_llama_cpp_protocol,
      formData.llm_llama_cpp_host,
      formData.llm_llama_cpp_port,
    );
  }, [fetchLocalLlmModels, formData.llm_llama_cpp_host, formData.llm_llama_cpp_port, formData.llm_llama_cpp_protocol]);

  const fetchLmstudioLlmModels = useCallback(async () => {
    await fetchLocalLlmModels(
      'lmstudio',
      PROVIDER_CONNECTIONS.lmstudioLlm,
      formData.llm_lmstudio_protocol,
      formData.llm_lmstudio_host,
      formData.llm_lmstudio_port,
      formData.lmstudio_api_key,
    );
  }, [fetchLocalLlmModels, formData.llm_lmstudio_host, formData.llm_lmstudio_port, formData.llm_lmstudio_protocol, formData.lmstudio_api_key]);

  const fetchOmlxLlmModels = useCallback(async () => {
    await fetchLocalLlmModels(
      'omlx',
      PROVIDER_CONNECTIONS.omlxLlm,
      formData.llm_omlx_protocol,
      formData.llm_omlx_host,
      formData.llm_omlx_port,
      formData.omlx_api_key,
    );
  }, [fetchLocalLlmModels, formData.llm_omlx_host, formData.llm_omlx_port, formData.llm_omlx_protocol, formData.omlx_api_key]);

  const fetchCopilotModels = useCallback(async () => {
    await fetchLlmModels('github_copilot', undefined, {
      authMode: copilotAuthMode,
      ...COPILOT_MODEL_FETCH_OPTIONS,
    });
  }, [copilotAuthMode, fetchLlmModels]);

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
          toast.success('GitHub Copilot connected successfully');
          const selectedProvider = formData.llm_provider || 'openai';
          if (selectedProvider === 'github_copilot') {
            await fetchCopilotModels();
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
    fetchCopilotModels,
    formData.llm_provider,
    refreshCopilotStatus,
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
      resetLlmModelsState();
      toast.success('GitHub Copilot connection removed');
    } catch (err) {
      setLlmModelsError(err instanceof Error ? err.message : 'Failed to clear GitHub Copilot auth');
    }
  }, [clearCopilotPollTimer, refreshCopilotStatus, resetLlmModelsState]);

  const handleCopilotDeviceCodeCopied = useCallback(() => {
    toast.success('Device code copied');
    setCopilotCodeCopied(true);
    window.setTimeout(() => setCopilotCodeCopied(false), 2000);
  }, [toast]);

  const handleCopilotDeviceCodeCopyError = useCallback(() => {
    setLlmModelsError('Unable to copy device code. Please copy it manually.');
  }, []);

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

  const fetchLocalEmbeddingModels = useCallback(async (
    provider: 'llama_cpp' | 'lmstudio' | 'omlx',
    connection: ProviderConnectionDescriptor,
    protocol: 'http' | 'https' | null | undefined,
    host: string | null | undefined,
    port: number | null | undefined,
    apiKey?: string,
  ) => {
    setEmbeddingModelsFetching(true);
    setEmbeddingModelsError(null);
    setEmbeddingModels([]);
    setEmbeddingModelsLoaded(false);

    try {
      const response = await api.fetchEmbeddingModels({
        provider,
        api_key: apiKey,
        base_url: buildLocalBaseUrl(protocol, host, port, connection),
        model: formData.embedding_model || undefined,
      });

      if (response.success) {
        setEmbeddingModels(response.models);
        setEmbeddingModelsLoaded(true);
        if (response.default_model) {
          setFormData((prev) => ({
            ...prev,
            embedding_model: prev.embedding_model || response.default_model,
          }));
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

  const fetchLlamaCppEmbeddingModels = useCallback(async () => {
    await fetchLocalEmbeddingModels(
      'llama_cpp',
      PROVIDER_CONNECTIONS.llamaCppEmbedding,
      formData.llama_cpp_protocol,
      formData.llama_cpp_host,
      formData.llama_cpp_port,
    );
  }, [fetchLocalEmbeddingModels, formData.llama_cpp_host, formData.llama_cpp_port, formData.llama_cpp_protocol]);

  const fetchLmstudioEmbeddingModels = useCallback(async () => {
    await fetchLocalEmbeddingModels(
      'lmstudio',
      PROVIDER_CONNECTIONS.lmstudioEmbedding,
      formData.lmstudio_protocol,
      formData.lmstudio_host,
      formData.lmstudio_port,
      formData.lmstudio_api_key,
    );
  }, [fetchLocalEmbeddingModels, formData.lmstudio_host, formData.lmstudio_port, formData.lmstudio_protocol, formData.lmstudio_api_key]);

  const fetchOmlxEmbeddingModels = useCallback(async () => {
    await fetchLocalEmbeddingModels(
      'omlx',
      PROVIDER_CONNECTIONS.omlxEmbedding,
      formData.omlx_protocol,
      formData.omlx_host,
      formData.omlx_port,
      formData.omlx_api_key,
    );
  }, [fetchLocalEmbeddingModels, formData.omlx_host, formData.omlx_port, formData.omlx_protocol, formData.omlx_api_key]);

  const getLmstudioChatBaseUrl = useCallback(() => buildLocalBaseUrl(
    formData.llm_lmstudio_protocol,
    formData.llm_lmstudio_host,
    formData.llm_lmstudio_port,
    PROVIDER_CONNECTIONS.lmstudioLlm,
  ), [formData.llm_lmstudio_host, formData.llm_lmstudio_port, formData.llm_lmstudio_protocol]);

  const getLmstudioEmbeddingBaseUrl = useCallback(() => buildLocalBaseUrl(
    formData.lmstudio_protocol,
    formData.lmstudio_host,
    formData.lmstudio_port,
    PROVIDER_CONNECTIONS.lmstudioEmbedding,
  ), [formData.lmstudio_host, formData.lmstudio_port, formData.lmstudio_protocol]);

  const loadSelectedLmstudioModel = useCallback(async (role: 'llm' | 'embedding') => {
    const model = ((role === 'llm' ? formData.llm_model : formData.embedding_model) || '').trim();
    if (!model) {
      const message = role === 'llm' ? 'Select an LM Studio chat model first' : 'Select an LM Studio embedding model first';
      role === 'llm' ? setLlmModelsError(message) : setEmbeddingModelsError(message);
      return;
    }

    setLmstudioModelActionLoading(true);
    role === 'llm' ? setLlmModelsError(null) : setEmbeddingModelsError(null);
    try {
      const response = await api.loadLmstudioModel({
        base_url: role === 'llm' ? getLmstudioChatBaseUrl() : getLmstudioEmbeddingBaseUrl(),
        model,
      });
      if (!response.success) {
        throw new Error(response.message);
      }
      toast.success('LM Studio model load requested');
      if (role === 'llm') {
        await fetchLmstudioLlmModels();
      } else {
        await fetchLmstudioEmbeddingModels();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load LM Studio model';
      role === 'llm' ? setLlmModelsError(message) : setEmbeddingModelsError(message);
    } finally {
      setLmstudioModelActionLoading(false);
    }
  }, [fetchLmstudioEmbeddingModels, fetchLmstudioLlmModels, formData.embedding_model, formData.llm_model, getLmstudioChatBaseUrl, getLmstudioEmbeddingBaseUrl, toast]);

  const unloadSelectedLmstudioModel = useCallback(async (role: 'llm' | 'embedding') => {
    const model = ((role === 'llm' ? formData.llm_model : formData.embedding_model) || '').trim();
    const modelInfo = role === 'llm'
      ? llmModels.find((item) => item.id === model)
      : embeddingModels.find((item) => item.id === model);
    const hasLoadedInstance = !!(modelInfo?.loaded_instances && modelInfo.loaded_instances.length > 0);
    if (!model && !hasLoadedInstance) {
      const message = role === 'llm' ? 'Select a loaded LM Studio chat model first' : 'Select a loaded LM Studio embedding model first';
      role === 'llm' ? setLlmModelsError(message) : setEmbeddingModelsError(message);
      return;
    }

    setLmstudioModelActionLoading(true);
    role === 'llm' ? setLlmModelsError(null) : setEmbeddingModelsError(null);
    try {
      // Pass only the model name so the backend unloads every instance of it.
      // (LM Studio supports multiple concurrent instances of the same model.)
      const response = await api.unloadLmstudioModel({
        base_url: role === 'llm' ? getLmstudioChatBaseUrl() : getLmstudioEmbeddingBaseUrl(),
        instance_id: undefined,
        model: model || undefined,
      });
      if (!response.success) {
        throw new Error(response.message);
      }
      toast.success('LM Studio model unload requested');
      if (role === 'llm') {
        await fetchLmstudioLlmModels();
      } else {
        await fetchLmstudioEmbeddingModels();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to unload LM Studio model';
      role === 'llm' ? setLlmModelsError(message) : setEmbeddingModelsError(message);
    } finally {
      setLmstudioModelActionLoading(false);
    }
  }, [embeddingModels, fetchLmstudioEmbeddingModels, fetchLmstudioLlmModels, formData.embedding_model, formData.llm_model, getLmstudioChatBaseUrl, getLmstudioEmbeddingBaseUrl, llmModels, toast]);

  // --- Shared model-fetching helpers for Chat & OpenAPI modal ---
  const fetchModelsForModal = useCallback(async (): Promise<{ models: AvailableModel[]; response: Awaited<ReturnType<typeof api.getAllModels>> }> => {
    const response = await api.getAllModels();
    let models: AvailableModel[] = response.models.map((model) => ({
      ...model,
      provider: normalizeLlmProvider(model.provider) as AvailableModel['provider'],
    }));

    const copilotPatToken = (formData.github_models_api_token || settings?.github_models_api_token || '').trim();
    const copilotConnected = Boolean(copilotAuthStatus?.connected || settings?.has_github_copilot_auth);
    const hasSelectedAuth = copilotAuthMode === 'pat' ? Boolean(copilotPatToken) : copilotConnected;
    if (hasSelectedAuth) {
      const githubResponse = await api.fetchLLMModels({
        provider: 'github_copilot',
        auth_mode: copilotAuthMode,
        include_directory_models: COPILOT_MODEL_FETCH_OPTIONS.includeDirectoryModels,
        include_anthropic_models: COPILOT_MODEL_FETCH_OPTIONS.includeAnthropicModels,
        include_google_models: COPILOT_MODEL_FETCH_OPTIONS.includeGoogleModels,
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
    setSelectedModels((prev) => toggleScopedModelSelection(prev, model));
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
      toast.success('Model filter saved');
      refreshModels();
      await refreshDefaultChatModelPreview();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save model filter');
    }
  };

  const openOpenapiModelModal = useCallback(async () => {
    setOpenapiModelsLoading(true);
    setShowOpenapiModelModal(true);

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
    setSelectedOpenapiModels((prev) => toggleScopedModelSelection(prev, model));
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
      toast.success('OpenAPI model filter saved');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save OpenAPI model filter');
    }
  };

  const refreshDefaultChatModelPreview = useCallback(async () => {
    setChatModelsLoading(true);
    try {
      const response = await api.getAvailableModels();
      setFilteredChatModels(response.models || []);
      setAutomaticDefaultChatModel(response.automatic_default_model || null);
    } catch {
      setFilteredChatModels([]);
      setAutomaticDefaultChatModel(null);
    } finally {
      setChatModelsLoading(false);
    }
  }, []);

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true);
      const [{ settings: rawSettings }, previewSettings, providerConfig, groups] = await Promise.all([
        api.getSettings(),
        api.getUserSpacePreviewSettings(),
        api.getAuthProviderConfig(),
        api.listAuthGroups(),
      ]);
      const data = sanitizeOllamaDefaults(rawSettings);
      setSettings(data);
      setUserspacePreviewSettings(previewSettings);
      setAuthProviderConfig(providerConfig);
      setAuthGroups(groups);
      const normalizedLlmProvider = normalizeLlmProvider(data.llm_provider);
      setFormData({
        // Server branding
        server_name: data.server_name,
        authenticated_webgl_background_enabled: data.authenticated_webgl_background_enabled ?? true,
        // Embedding settings
        ...getEmbeddingSettingsFormData(data),
        // LLM settings
        llm_provider: normalizedLlmProvider ?? undefined,
        llm_model: data.llm_model,
        llm_max_tokens: data.llm_max_tokens,
        image_payload_max_width: data.image_payload_max_width,
        image_payload_max_height: data.image_payload_max_height,
        image_payload_max_pixels: data.image_payload_max_pixels,
        image_payload_max_bytes: data.image_payload_max_bytes,
        llm_ollama_protocol: data.llm_ollama_protocol,
        llm_ollama_host: data.llm_ollama_host,
        llm_ollama_port: data.llm_ollama_port,
        llm_ollama_base_url: data.llm_ollama_base_url,
        llm_llama_cpp_protocol: data.llm_llama_cpp_protocol,
        llm_llama_cpp_host: data.llm_llama_cpp_host,
        llm_llama_cpp_port: data.llm_llama_cpp_port,
        llm_llama_cpp_base_url: data.llm_llama_cpp_base_url,
        llm_lmstudio_protocol: data.llm_lmstudio_protocol,
        llm_lmstudio_host: data.llm_lmstudio_host,
        llm_lmstudio_port: data.llm_lmstudio_port,
        llm_lmstudio_base_url: data.llm_lmstudio_base_url,
        llm_omlx_protocol: data.llm_omlx_protocol,
        llm_omlx_host: data.llm_omlx_host,
        llm_omlx_port: data.llm_omlx_port,
        llm_omlx_base_url: data.llm_omlx_base_url,
        openai_api_key: data.openai_api_key,
        anthropic_api_key: data.anthropic_api_key,
        openrouter_api_key: data.openrouter_api_key,
        github_models_api_token: data.github_models_api_token,
        github_copilot_base_url: data.github_copilot_base_url,
        github_copilot_enterprise_url: data.github_copilot_enterprise_url,
        default_chat_model: data.default_chat_model ?? null,
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
        mcp_default_route_client_id: data.mcp_default_route_client_id ?? '',
        mcp_default_route_password: data.mcp_default_route_password ?? '',
        // OCR settings
        default_ocr_mode: data.default_ocr_mode,
        default_ocr_provider: data.default_ocr_provider || 'ollama',
        default_ocr_vision_model: data.default_ocr_vision_model,
        ocr_concurrency_limit: data.ocr_concurrency_limit,
        userspace_preview_sandbox_flags: data.userspace_preview_sandbox_flags,
        userspace_duplicate_copy_files_default: data.userspace_duplicate_copy_files_default,
        userspace_duplicate_copy_metadata_default: data.userspace_duplicate_copy_metadata_default,
        userspace_duplicate_copy_chats_default: data.userspace_duplicate_copy_chats_default,
        userspace_duplicate_copy_mounts_default: data.userspace_duplicate_copy_mounts_default,
        userspace_mount_sync_interval_seconds: data.userspace_mount_sync_interval_seconds,
        userspace_sqlite_import_max_bytes: data.userspace_sqlite_import_max_bytes,

        // OpenAPI model settings
        openapi_sync_chat_models: data.openapi_sync_chat_models,
      });
      // Reset Ollama connection state (for embeddings)
      resetEmbeddingOllamaState();
      resetLlmOllamaState();
      resetLlmModelsState();
      clearCopilotPollTimer();
      setCopilotConnecting(false);
      setCopilotRequestId(null);
      setCopilotDeviceCode('');
      setCopilotVerificationUri('');
      toast.clear();
      setCopilotAuthMode(data.github_models_api_token ? 'pat' : 'oauth');

      // Form is ready — show the page immediately; remaining fetches are lazy.
      setLoading(false);

      // --- Lazy background fetches (non-blocking) ---

      // Copilot auth status + model list
      refreshCopilotStatus().then((copilotStatus) => {
        if (normalizedLlmProvider === 'github_copilot') {
          if (data.github_models_api_token || copilotStatus?.connected) {
            fetchLlmModels('github_copilot', undefined, {
              authMode: data.github_models_api_token ? 'pat' : 'oauth',
              ...COPILOT_MODEL_FETCH_OPTIONS,
            });
          }
        }
      }).catch(() => { /* copilot status is best-effort */ });

      // Auto-test Ollama if using ollama embedding provider
      if (data.embedding_provider === 'ollama' && !hasAutoTestedOllama.current) {
        hasAutoTestedOllama.current = true;
        if (data.ollama_host?.trim()) {
          testOllamaConnection(
            data.ollama_protocol || DEFAULT_OLLAMA_PROTOCOL,
            data.ollama_host,
            data.ollama_port || DEFAULT_OLLAMA_PORT,
          );
        }
      }

      // Auto-test LLM Ollama if using ollama LLM provider
      if (data.llm_provider === 'ollama' && !hasAutoTestedLlmOllama.current) {
        hasAutoTestedLlmOllama.current = true;
        if (data.llm_ollama_host?.trim()) {
          testLlmOllamaConnection(
            data.llm_ollama_protocol || DEFAULT_OLLAMA_PROTOCOL,
            data.llm_ollama_host,
            data.llm_ollama_port || DEFAULT_OLLAMA_PORT,
          );
        }
      }

      // Chat model preview (available-models)
      refreshDefaultChatModelPreview();

      // MCP routes (for summary display)
      api.listMcpRoutes()
        .then((routesRes) => setMcpRoutes(routesRes.routes))
        .catch(() => setMcpRoutes([]));

      // LDAP configuration
      api.getLdapConfig()
        .then((ldapData) => {
          setLdapConfig(ldapData);

          const { protocol, host, port } = parseLdapServerUrl(ldapData.server_url);

          setLdapFormData({
            ldap_protocol: protocol,
            ldap_host: host,
            ldap_port: port,
            allow_self_signed: ldapData.allow_self_signed || false,
            bind_dn: ldapData.bind_dn || '',
            bind_password: '', // Never returned from server
            user_search_base: ldapData.user_search_base || '',
            user_search_filter: ldapData.user_search_filter || '(uid={username})',
            admin_group_dns: ldapData.admin_group_dns || [],
            user_group_dns: ldapData.user_group_dns || [],
          });

          // Auto-discover LDAP structure in background
          if (ldapData.server_url && ldapData.bind_dn) {
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
        })
        .catch(() => {
          // LDAP config may not exist yet, that's OK
          setLdapConfig(null);
        });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load settings');
      setLoading(false);
    }
  }, [
    clearCopilotPollTimer,
    fetchLlmModels,
    refreshDefaultChatModelPreview,
    refreshCopilotStatus,
    resetEmbeddingOllamaState,
    resetLlmModelsState,
    resetLlmOllamaState,
    testLlmOllamaConnection,
    testOllamaConnection,
  ]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  useEffect(() => {
    const normalizedProvider = normalizeLlmProvider(formData.llm_provider);
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

    void fetchCopilotModels();

    if (showModelFilterModal) {
      void openModelFilterModal();
    }
  }, [
    copilotAuthMode,
    copilotAuthStatus?.connected,
    fetchCopilotModels,
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
      formData.ollama_port || DEFAULT_OLLAMA_PORT
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
    const normalizedBindDn = ldapFormData.bind_dn.trim();
    const typedBindPassword = ldapFormData.bind_password;
    const canReuseStoredCredentials = Boolean(
      !typedBindPassword
      && ldapConfig?.server_url
      && ldapConfig?.bind_dn
      && ldapConfig.server_url === serverUrl
      && ldapConfig.bind_dn.trim().toLowerCase() === normalizedBindDn.toLowerCase()
    );

    if (!serverUrl || !normalizedBindDn) {
      setLdapTestResult({ success: false, message: 'Server and Bind DN are required' });
      return;
    }

    if (!typedBindPassword && !canReuseStoredCredentials) {
      setLdapTestResult({
        success: false,
        message: 'Bind Password is required unless testing with unchanged saved LDAP credentials.',
      });
      return;
    }

    setLdapTesting(true);
    setLdapTestResult(null);

    try {
      const response = canReuseStoredCredentials
        ? await api.discoverLdapWithStoredCredentials()
        : await api.discoverLdap({
          server_url: serverUrl,
          bind_dn: normalizedBindDn,
          bind_password: typedBindPassword,
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

  const saveLdapConfig = async () => {
    const serverUrl = buildServerUrl();
    const updated = await api.updateLdapConfig({
      server_url: serverUrl || undefined,
      allow_self_signed: ldapFormData.allow_self_signed,
      bind_dn: ldapFormData.bind_dn || undefined,
      bind_password: ldapFormData.bind_password || undefined,
      user_search_base: ldapFormData.user_search_base || undefined,
      user_search_filter: ldapFormData.user_search_filter || undefined,
      admin_group_dns: ldapFormData.admin_group_dns,
      user_group_dns: ldapFormData.user_group_dns,
    });
    setLdapConfig(updated);
    setAuthGroups(await api.listAuthGroups());
  };

  const refreshAuthGroups = useCallback(async () => {
    try {
      setAuthGroups(await api.listAuthGroups());
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load Group Memberships');
    }
  }, [toast]);

  const openManageAuthGroupsModal = useCallback(() => {
    setShowManageAuthGroupsModal(true);
    void refreshAuthGroups();
  }, [refreshAuthGroups]);

  const closeManageAuthGroupsModal = useCallback(() => {
    setShowManageAuthGroupsModal(false);
  }, []);

  const handleSaveAuthProviderConfig = async () => {
    setAuthProviderConfigSaving(true);
    try {
      await saveLdapConfig();

      if (authProviderConfig) {
        const updated = await api.updateAuthProviderConfig(authProviderConfig);
        setAuthProviderConfig(updated);
      }

      toast.success('Authentication settings saved');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save authentication settings');
    } finally {
      setAuthProviderConfigSaving(false);
    }
  };

  const handleSelectLdapUserSuggestion = useCallback((profile: LdapUserProfile) => {
    setLdapUserPreview(profile);
    setLdapUserSearchName(profile.username || profile.email || '');
    setShowLdapUserSearchResults(false);
    setSuppressLdapUserSearchDropdown(true);
  }, []);

  const ldapUserSearchEnabled = Boolean(ldapFormData.ldap_host.trim() || ldapConfig?.server_url?.trim());

  useEffect(() => {
    const query = ldapUserSearchName.trim();
    if (suppressLdapUserSearchDropdown) {
      setShowLdapUserSearchResults(false);
      setLdapUserSearching(false);
      return;
    }

    if (!ldapUserSearchEnabled || query.length < 2) {
      setLdapUserSearchResults([]);
      setShowLdapUserSearchResults(false);
      setLdapUserSearching(false);
      return;
    }

    const requestSeq = ldapUserSearchRequestSeqRef.current + 1;
    ldapUserSearchRequestSeqRef.current = requestSeq;
    setLdapUserSearching(true);

    const timer = window.setTimeout(async () => {
      try {
        const response = await api.searchLdapUsers({ query, limit: 8 });
        if (ldapUserSearchRequestSeqRef.current !== requestSeq) {
          return;
        }
        setLdapUserSearchResults(response.users || []);
        setShowLdapUserSearchResults(true);
      } catch {
        if (ldapUserSearchRequestSeqRef.current !== requestSeq) {
          return;
        }
        setLdapUserSearchResults([]);
        setShowLdapUserSearchResults(true);
      } finally {
        if (ldapUserSearchRequestSeqRef.current === requestSeq) {
          setLdapUserSearching(false);
        }
      }
    }, 250);

    return () => {
      window.clearTimeout(timer);
    };
  }, [ldapUserSearchEnabled, ldapUserSearchName, suppressLdapUserSearchDropdown]);

  useEffect(() => {
    if (!showLdapUserSearchResults) {
      return;
    }

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node | null;
      if (!target) {
        return;
      }
      if (!ldapUserSearchContainerRef.current?.contains(target)) {
        setShowLdapUserSearchResults(false);
      }
    };

    window.addEventListener('mousedown', handleClickOutside);
    return () => {
      window.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showLdapUserSearchResults]);

  const handleImportLdapUser = async () => {
    setLdapUserImporting(true);
    try {
      const targetUsername = (ldapUserPreview?.username || ldapUserSearchName || '').trim();
      if (!targetUsername) {
        toast.error('Select an LDAP user first');
        return;
      }
      await api.importLdapUser({ username: targetUsername });
      toast.success('LDAP user imported into local cache');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to import LDAP user');
    } finally {
      setLdapUserImporting(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
  };

  // Save Server Branding
  const [brandingSaving, setBrandingSaving] = useState(false);
  const handleSaveBranding = async () => {
    setBrandingSaving(true);

    try {
      const normalizedServerName = (formData.server_name || '').trim();
      if (!normalizedServerName) {
        toast.error('Server name cannot be empty');
        return;
      }

      const dataToSave = {
        server_name: normalizedServerName,
        authenticated_webgl_background_enabled: formData.authenticated_webgl_background_enabled ?? settings?.authenticated_webgl_background_enabled ?? true,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setFormData((prev) => ({
        ...prev,
        server_name: normalizedServerName,
        authenticated_webgl_background_enabled: updated.authenticated_webgl_background_enabled ?? true,
      }));
      // Show restart guidance in the dismissable security banner.
      sessionStorage.setItem('ragtime_branding_restart_notice', 'true');
      const dismissedNoticesKey = 'ragtime_security_banner_dismissed_notices';
      const legacyDismissKey = 'ragtime_security_banner_dismissed';
      const brandingNoticeId = 'branding-restart';
      const existingDismissedNoticesRaw = sessionStorage.getItem(dismissedNoticesKey);
      if (existingDismissedNoticesRaw) {
        try {
          const existingDismissedNotices = JSON.parse(existingDismissedNoticesRaw);
          if (Array.isArray(existingDismissedNotices)) {
            const nextDismissedNotices = existingDismissedNotices.filter((n) => n !== brandingNoticeId);
            sessionStorage.setItem(dismissedNoticesKey, JSON.stringify(nextDismissedNotices));
          }
        } catch {
          // Ignore invalid session storage payload.
        }
      }
      sessionStorage.removeItem(legacyDismissKey);
      window.dispatchEvent(new CustomEvent('ragtime:branding-notice-updated'));
      // Notify parent component of name change
      if (onServerNameChange && updated.server_name) {
        onServerNameChange(updated.server_name);
      }
      onAuthenticatedWebglBackgroundChange?.(updated.authenticated_webgl_background_enabled ?? true);
      toast.success('Server branding saved');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save branding settings');
    } finally {
      setBrandingSaving(false);
    }
  };

  // Save Embedding Configuration
  const handleSaveEmbedding = async () => {
    setEmbeddingSaving(true);

    try {
      const dataToSave: UpdateSettingsRequest = {
        embedding_provider: formData.embedding_provider,
        embedding_model: formData.embedding_model,
        embedding_dimensions: formData.embedding_dimensions ?? null,
        ollama_protocol: formData.ollama_protocol,
        ollama_host: formData.ollama_host,
        ollama_port: formData.ollama_port,
        ollama_base_url: buildOllamaBaseUrl(
          formData.ollama_protocol,
          formData.ollama_host,
          formData.ollama_port,
        ),
        llama_cpp_protocol: formData.llama_cpp_protocol,
        llama_cpp_host: formData.llama_cpp_host,
        llama_cpp_port: formData.llama_cpp_port,
        llama_cpp_base_url: buildLocalBaseUrl(
          formData.llama_cpp_protocol,
          formData.llama_cpp_host,
          formData.llama_cpp_port,
          PROVIDER_CONNECTIONS.llamaCppEmbedding,
        ),
        lmstudio_protocol: formData.lmstudio_protocol,
        lmstudio_host: formData.lmstudio_host,
        lmstudio_port: formData.lmstudio_port,
        lmstudio_base_url: buildLocalBaseUrl(
          formData.lmstudio_protocol,
          formData.lmstudio_host,
          formData.lmstudio_port,
          PROVIDER_CONNECTIONS.lmstudioEmbedding,
        ),
        lmstudio_api_key: formData.lmstudio_api_key,
        omlx_protocol: formData.omlx_protocol,
        omlx_host: formData.omlx_host,
        omlx_port: formData.omlx_port,
        omlx_base_url: buildLocalBaseUrl(
          formData.omlx_protocol,
          formData.omlx_host,
          formData.omlx_port,
          PROVIDER_CONNECTIONS.omlxEmbedding,
        ),
        omlx_api_key: formData.omlx_api_key,
        ollama_embedding_timeout_seconds: formData.ollama_embedding_timeout_seconds,
        sequential_index_loading: formData.sequential_index_loading,
        default_ocr_mode: formData.default_ocr_mode,
        default_ocr_provider: formData.default_ocr_provider,
        default_ocr_vision_model: formData.default_ocr_vision_model,
        ocr_concurrency_limit: formData.ocr_concurrency_limit,
      };
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      setFormData((prev) => ({
        ...prev,
        ...getEmbeddingSettingsFormData(updated),
      }));
      toast.success('Embedding configuration saved');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save embedding settings');
    } finally {
      setEmbeddingSaving(false);
    }
  };

  // Save LLM Configuration
  const handleSaveLlm = async () => {
    setLlmSaving(true);

    try {
      const normalizedProvider = normalizeLlmProvider(formData.llm_provider);
      const dataToSave: Record<string, unknown> = {
        llm_provider: normalizedProvider,
        llm_model: formData.llm_model,
        llm_max_tokens: formData.llm_max_tokens,
        image_payload_max_width: formData.image_payload_max_width,
        image_payload_max_height: formData.image_payload_max_height,
        image_payload_max_pixels: formData.image_payload_max_pixels,
        image_payload_max_bytes: formData.image_payload_max_bytes,
        openai_api_key: formData.openai_api_key,
        anthropic_api_key: formData.anthropic_api_key,
        openrouter_api_key: formData.openrouter_api_key,
        github_models_api_token: formData.github_models_api_token,
        github_copilot_base_url: formData.github_copilot_base_url,
        github_copilot_enterprise_url: formData.github_copilot_enterprise_url,
        default_chat_model: formData.default_chat_model,
        allowed_chat_models: formData.allowed_chat_models,
        max_iterations: formData.max_iterations,
        // OpenAPI model settings
        openapi_sync_chat_models: formData.openapi_sync_chat_models,
        // Token optimization settings
        max_tool_output_chars: formData.max_tool_output_chars,
        scratchpad_window_size: formData.scratchpad_window_size,
        context_token_budget: formData.context_token_budget,
        // API output settings
        tool_output_mode: formData.tool_output_mode,
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
        dataToSave.llm_ollama_base_url = buildOllamaBaseUrl(
          formData.llm_ollama_protocol,
          formData.llm_ollama_host,
          formData.llm_ollama_port,
        );
      }
      if (normalizedProvider === 'llama_cpp') {
        dataToSave.llm_llama_cpp_protocol = formData.llm_llama_cpp_protocol;
        dataToSave.llm_llama_cpp_host = formData.llm_llama_cpp_host;
        dataToSave.llm_llama_cpp_port = formData.llm_llama_cpp_port;
        dataToSave.llm_llama_cpp_base_url = buildLocalBaseUrl(
          formData.llm_llama_cpp_protocol,
          formData.llm_llama_cpp_host,
          formData.llm_llama_cpp_port,
          PROVIDER_CONNECTIONS.llamaCppLlm,
        );
      }
      if (normalizedProvider === 'lmstudio') {
        dataToSave.llm_lmstudio_protocol = formData.llm_lmstudio_protocol;
        dataToSave.llm_lmstudio_host = formData.llm_lmstudio_host;
        dataToSave.llm_lmstudio_port = formData.llm_lmstudio_port;
        dataToSave.llm_lmstudio_base_url = buildLocalBaseUrl(
          formData.llm_lmstudio_protocol,
          formData.llm_lmstudio_host,
          formData.llm_lmstudio_port,
          PROVIDER_CONNECTIONS.lmstudioLlm,
        );
        dataToSave.lmstudio_api_key = formData.lmstudio_api_key;
      }
      if (normalizedProvider === 'omlx') {
        dataToSave.llm_omlx_protocol = formData.llm_omlx_protocol;
        dataToSave.llm_omlx_host = formData.llm_omlx_host;
        dataToSave.llm_omlx_port = formData.llm_omlx_port;
        dataToSave.llm_omlx_base_url = buildLocalBaseUrl(
          formData.llm_omlx_protocol,
          formData.llm_omlx_host,
          formData.llm_omlx_port,
          PROVIDER_CONNECTIONS.omlxLlm,
        );
        dataToSave.omlx_api_key = formData.omlx_api_key;
      }
      const updated = await api.updateSettings(dataToSave);
      setSettings(updated);
      toast.success('LLM configuration saved');
      refreshModels();
      await refreshDefaultChatModelPreview();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save LLM settings');
    } finally {
      setLlmSaving(false);
    }
  };

  // Save Search Configuration
  const [searchSaving, setSearchSaving] = useState(false);
  const handleSaveSearch = async () => {
    setSearchSaving(true);

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
      toast.success('Search configuration saved. Restart the server to apply changes to search tools.', 5000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save search settings');
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
    setMcpError(null);

    // Validate password if provided (not empty string which clears, and not undefined which skips)
    const pwd = formData.mcp_default_route_password;
    const authMethod = formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password';
    const authEnabled = formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth;
    const clientId = (formData.mcp_default_route_client_id ?? settings?.mcp_default_route_client_id ?? '').trim();
    if ((authMethod === 'password' || authMethod === 'client_credentials') && pwd !== undefined && pwd !== '' && pwd.length < 8) {
      setMcpError('MCP password must be at least 8 characters');
      setMcpSaving(false);
      return;
    }
    if (authEnabled && authMethod === 'client_credentials' && !clientId) {
      setMcpError('Client ID is required for client credentials authentication');
      setMcpSaving(false);
      return;
    }
    if (authEnabled && authMethod === 'client_credentials' && !settings?.has_mcp_default_password && (!pwd || pwd === '')) {
      setMcpError('Client secret is required for client credentials authentication');
      setMcpSaving(false);
      return;
    }

    try {
      const allowedGroup = authMethod === 'oauth2'
        ? (formData.mcp_default_route_allowed_group ?? settings?.mcp_default_route_allowed_group ?? null)
        : null;
      const dataToSave: UpdateSettingsRequest = {
        mcp_enabled: formData.mcp_enabled,
        mcp_default_route_auth: formData.mcp_default_route_auth,
        mcp_default_route_auth_method: formData.mcp_default_route_auth_method,
        mcp_default_route_allowed_group: allowedGroup,
      };
      if (authMethod === 'client_credentials') {
        dataToSave.mcp_default_route_client_id = clientId;
      } else if (settings?.mcp_default_route_client_id) {
        dataToSave.mcp_default_route_client_id = '';
      }
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
        mcp_default_route_client_id: updated.mcp_default_route_client_id ?? '',
        mcp_default_route_password: updated.mcp_default_route_password ?? '',
      }));
      toast.success('MCP configuration saved.', 5000);
    } catch (err) {
      setMcpError(err instanceof Error ? err.message : 'Failed to save MCP settings');
    } finally {
      setMcpSaving(false);
    }
  };



  // OCR Configuration
  const [userspaceSaving, setUserspaceSaving] = useState(false);
  const [showSandboxModal, setShowSandboxModal] = useState(false);
  const [showGlobalEnvVarsModal, setShowGlobalEnvVarsModal] = useState(false);
  const [globalEnvVars, setGlobalEnvVars] = useState<UserSpaceWorkspaceEnvVar[]>([]);
  const [globalEnvVarsLoading, setGlobalEnvVarsLoading] = useState(false);
  const [globalEnvVarsSaving, setGlobalEnvVarsSaving] = useState(false);


  const [visionModels, setVisionModels] = useState<VisionModel[]>([]);
  const [visionModelsLoading, setVisionModelsLoading] = useState(false);
  const [visionModelsError, setVisionModelsError] = useState<string | null>(null);
  const [showOcrRecommendations, setShowOcrRecommendations] = useState(false);

  const selectedOcrProvider = (formData.default_ocr_provider || 'ollama') as OcrProvider;
  const selectedOcrProviderLabel = OCR_PROVIDER_LABELS[selectedOcrProvider] || selectedOcrProvider;

  const fetchVisionModels = useCallback(async () => {
    const provider = (formData.default_ocr_provider || 'ollama') as OcrProvider;
    const request: Parameters<typeof api.getVisionModels>[0] = { provider };

    if (provider === 'ollama') {
      if (!formData.ollama_protocol || !formData.ollama_host || !formData.ollama_port) return;
      request.protocol = formData.ollama_protocol as 'http' | 'https';
      request.host = formData.ollama_host;
      request.port = formData.ollama_port;
    } else if (provider === 'openai') {
      request.api_key = formData.openai_api_key;
      if (!request.api_key || request.api_key.length < 10) {
        setVisionModelsError('Enter an OpenAI API key before loading vision models.');
        return;
      }
    } else if (provider === 'omlx') {
      request.base_url = buildLocalBaseUrl(formData.llm_omlx_protocol, formData.llm_omlx_host, formData.llm_omlx_port, PROVIDER_CONNECTIONS.omlxLlm);
      request.api_key = formData.omlx_api_key;
    } else if (provider === 'lmstudio') {
      request.base_url = buildLocalBaseUrl(formData.llm_lmstudio_protocol, formData.llm_lmstudio_host, formData.llm_lmstudio_port, PROVIDER_CONNECTIONS.lmstudioLlm);
      request.api_key = formData.lmstudio_api_key;
    } else if (provider === 'llama_cpp') {
      request.base_url = buildLocalBaseUrl(formData.llm_llama_cpp_protocol, formData.llm_llama_cpp_host, formData.llm_llama_cpp_port, PROVIDER_CONNECTIONS.llamaCppLlm);
    }

    setVisionModelsLoading(true);
    setVisionModelsError(null);

    try {
      const response = await api.getVisionModels(request);

      if (response.success) {
        setVisionModels(response.models);
        if (response.models.length === 0) {
          setVisionModelsError(`No vision-capable models found for ${OCR_PROVIDER_LABELS[provider] || provider}.`);
        }
      } else {
        setVisionModelsError(response.message);
      }
    } catch (err) {
      setVisionModelsError(err instanceof Error ? err.message : 'Failed to fetch vision models');
    } finally {
      setVisionModelsLoading(false);
    }
  }, [
    formData.default_ocr_provider,
    formData.ollama_protocol,
    formData.ollama_host,
    formData.ollama_port,
    formData.openai_api_key,
    formData.llm_omlx_protocol,
    formData.llm_omlx_host,
    formData.llm_omlx_port,
    formData.omlx_api_key,
    formData.llm_lmstudio_protocol,
    formData.llm_lmstudio_host,
    formData.llm_lmstudio_port,
    formData.lmstudio_api_key,
    formData.llm_llama_cpp_protocol,
    formData.llm_llama_cpp_host,
    formData.llm_llama_cpp_port,
  ]);

  // Auto-fetch vision models when OCR mode changes to semantic vision.
  useEffect(() => {
    if (formData.default_ocr_mode === 'vision' && visionModels.length === 0 && !visionModelsLoading) {
      fetchVisionModels();
    }
  }, [formData.default_ocr_mode, formData.default_ocr_provider, fetchVisionModels, visionModels.length, visionModelsLoading]);



  const effectiveUserSpacePreviewSandboxFlags = useMemo(
    () => {
      const fallbackFlags = userspacePreviewSettings?.userspace_preview_sandbox_default_flags ?? [];
      const allowedFlags = getUserSpacePreviewSandboxFlagValues(
        userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []
      );
      return normalizeUserSpacePreviewSandboxFlags(
        formData.userspace_preview_sandbox_flags
        ?? settings?.userspace_preview_sandbox_flags,
        allowedFlags,
        fallbackFlags,
      );
    },
    [
      formData.userspace_preview_sandbox_flags,
      settings?.userspace_preview_sandbox_flags,
      userspacePreviewSettings,
    ]
  );

  const userspacePreviewSandboxAttribute = useMemo(
    () => buildUserSpacePreviewSandboxAttribute(effectiveUserSpacePreviewSandboxFlags),
    [effectiveUserSpacePreviewSandboxFlags]
  );

  const setUserSpacePreviewSandboxFlags = useCallback((flags: string[]) => {
    const fallbackFlags = userspacePreviewSettings?.userspace_preview_sandbox_default_flags ?? [];
    const allowedFlags = getUserSpacePreviewSandboxFlagValues(
      userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []
    );
    setFormData((prev) => ({
      ...prev,
      userspace_preview_sandbox_flags: normalizeUserSpacePreviewSandboxFlags(
        flags,
        allowedFlags,
        fallbackFlags,
      ),
    }));
  }, [userspacePreviewSettings]);

  const handleToggleUserSpacePreviewSandboxFlag = useCallback((flag: string) => {
    const selected = new Set(effectiveUserSpacePreviewSandboxFlags);
    if (selected.has(flag)) {
      selected.delete(flag);
    } else {
      selected.add(flag);
    }
    setUserSpacePreviewSandboxFlags(Array.from(selected));
  }, [effectiveUserSpacePreviewSandboxFlags, setUserSpacePreviewSandboxFlags]);

  const handleSaveUserSpacePreviewSandbox = useCallback(async () => {
    setUserspaceSaving(true);

    try {
      const updated = await api.updateSettings({
        userspace_preview_sandbox_flags: effectiveUserSpacePreviewSandboxFlags,
      });
      setSettings(updated);
      setFormData((prev) => ({
        ...prev,
        userspace_preview_sandbox_flags: updated.userspace_preview_sandbox_flags,
      }));
      toast.success('User Space preview sandbox settings saved.', 5000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save User Space preview sandbox settings');
    } finally {
      setUserspaceSaving(false);
    }
  }, [effectiveUserSpacePreviewSandboxFlags]);

  const [staleBranchSaving, setStaleBranchSaving] = useState(false);
  const handleSaveStaleBranchThreshold = useCallback(async () => {
    setStaleBranchSaving(true);

    try {
      const updated = await api.updateSettings({
        snapshot_stale_branch_threshold: formData.snapshot_stale_branch_threshold,
        userspace_duplicate_copy_files_default: formData.userspace_duplicate_copy_files_default,
        userspace_duplicate_copy_metadata_default: formData.userspace_duplicate_copy_metadata_default,
        userspace_duplicate_copy_chats_default: formData.userspace_duplicate_copy_chats_default,
        userspace_duplicate_copy_mounts_default: formData.userspace_duplicate_copy_mounts_default,
        userspace_mount_sync_interval_seconds: formData.userspace_mount_sync_interval_seconds,
        userspace_sqlite_import_max_bytes: formData.userspace_sqlite_import_max_bytes,
      });
      setSettings(updated);
      setFormData((prev) => ({
        ...prev,
        snapshot_stale_branch_threshold: updated.snapshot_stale_branch_threshold,
        userspace_duplicate_copy_files_default: updated.userspace_duplicate_copy_files_default,
        userspace_duplicate_copy_metadata_default: updated.userspace_duplicate_copy_metadata_default,
        userspace_duplicate_copy_chats_default: updated.userspace_duplicate_copy_chats_default,
        userspace_duplicate_copy_mounts_default: updated.userspace_duplicate_copy_mounts_default,
        userspace_mount_sync_interval_seconds: updated.userspace_mount_sync_interval_seconds,
        userspace_sqlite_import_max_bytes: updated.userspace_sqlite_import_max_bytes,
      }));
      toast.success('User Space settings saved.', 5000);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save User Space settings');
    } finally {
      setStaleBranchSaving(false);
    }
  }, [
    formData.snapshot_stale_branch_threshold,
    formData.userspace_duplicate_copy_files_default,
    formData.userspace_duplicate_copy_metadata_default,
    formData.userspace_duplicate_copy_chats_default,
    formData.userspace_duplicate_copy_mounts_default,
    formData.userspace_mount_sync_interval_seconds,
    formData.userspace_sqlite_import_max_bytes,
  ]);

  const loadGlobalEnvVars = useCallback(async () => {
    setGlobalEnvVarsLoading(true);
    try {
      const vars = await api.listUserSpaceGlobalEnvVars();
      setGlobalEnvVars(vars);
      return vars;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load global environment variables');
      throw err;
    } finally {
      setGlobalEnvVarsLoading(false);
    }
  }, [toast]);

  const mergeGlobalEnvVar = useCallback((envVar: UserSpaceWorkspaceEnvVar, previousKey?: string) => {
    setGlobalEnvVars((current) => {
      const keysToReplace = new Set([previousKey, envVar.key].filter(Boolean));
      const next = current.filter((item) => !keysToReplace.has(item.key));
      return [...next, envVar].sort((a, b) => a.key.localeCompare(b.key));
    });
  }, []);

  const handleOpenGlobalEnvVarsModal = useCallback(async () => {
    setShowGlobalEnvVarsModal(true);
    await loadGlobalEnvVars();
  }, [loadGlobalEnvVars]);

  const handleCreateGlobalEnvVar = useCallback(async (request: UpsertUserSpaceWorkspaceEnvVarRequest) => {
    setGlobalEnvVarsSaving(true);
    try {
      const upserted = await api.upsertUserSpaceGlobalEnvVar(request);
      try {
        await loadGlobalEnvVars();
      } catch {
        mergeGlobalEnvVar(upserted, request.key);
      }
      toast.success('Global environment variable saved. Restart active runtimes to apply changes.');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save global environment variable');
      throw err;
    } finally {
      setGlobalEnvVarsSaving(false);
    }
  }, [loadGlobalEnvVars, mergeGlobalEnvVar, toast]);

  const handleUpdateGlobalEnvVar = useCallback(async (request: UpsertUserSpaceWorkspaceEnvVarRequest) => {
    setGlobalEnvVarsSaving(true);
    try {
      const upserted = await api.upsertUserSpaceGlobalEnvVar(request);
      try {
        await loadGlobalEnvVars();
      } catch {
        mergeGlobalEnvVar(upserted, request.key);
      }
      toast.success('Global environment variable updated. Restart active runtimes to apply changes.');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update global environment variable');
      throw err;
    } finally {
      setGlobalEnvVarsSaving(false);
    }
  }, [loadGlobalEnvVars, mergeGlobalEnvVar, toast]);

  const handleDeleteGlobalEnvVar = useCallback(async (key: string) => {
    setGlobalEnvVarsSaving(true);
    try {
      await api.deleteUserSpaceGlobalEnvVar(key);
      try {
        await loadGlobalEnvVars();
      } catch {
        setGlobalEnvVars((current) => current.filter((item) => item.key !== key));
      }
      toast.success('Global environment variable deleted. Restart active runtimes to apply changes.');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete global environment variable');
      throw err;
    } finally {
      setGlobalEnvVarsSaving(false);
    }
  }, [loadGlobalEnvVars, toast]);

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
      const saveActionGroups = Array.from(fieldset.querySelectorAll<HTMLElement>('.form-group')).filter((group) => (
        Array.from(group.querySelectorAll<HTMLButtonElement>('button')).some(isSettingsSaveControlButton)
      ));

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

      let visibleDetailsCount = 0;
      const detailsElements = Array.from(fieldset.querySelectorAll<HTMLDetailsElement>('details'));
      detailsElements.forEach((details) => {
        const summaryText = details.querySelector('summary')?.textContent || '';
        const detailText = details.textContent || '';
        const hasVisibleChild = Array.from(details.querySelectorAll<HTMLElement>('.form-group')).some((group) => group.style.display !== 'none');
        const detailsMatch = fieldsetTextMatch || hasVisibleChild || settingsTextMatchesQuery(`${summaryText} ${detailText}`, queries);
        details.style.display = detailsMatch ? '' : 'none';
        if (detailsMatch) {
          visibleDetailsCount += 1;
        }
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
      const showFieldset = fieldsetTextMatch || visibleFormGroupCount > 0 || visibleDetailsCount > 0;
      formActions.forEach((actions) => {
        actions.style.display = showFieldset ? '' : 'none';
      });

      // Some sections place Save buttons in .form-group blocks instead of .form-actions.
      saveActionGroups.forEach((group) => {
        group.style.display = showFieldset ? '' : 'none';
      });

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
  const openRouterConfigured = Boolean((formData.openrouter_api_key ?? settings?.openrouter_api_key)?.trim());
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
  const llamaCppConfigured = Boolean(
    (formData.llm_llama_cpp_protocol ?? settings?.llm_llama_cpp_protocol) &&
    (formData.llm_llama_cpp_host ?? settings?.llm_llama_cpp_host)?.trim() &&
    (formData.llm_llama_cpp_port ?? settings?.llm_llama_cpp_port)
  );
  const lmstudioConfigured = Boolean(
    (formData.llm_lmstudio_protocol ?? settings?.llm_lmstudio_protocol) &&
    (formData.llm_lmstudio_host ?? settings?.llm_lmstudio_host)?.trim() &&
    (formData.llm_lmstudio_port ?? settings?.llm_lmstudio_port)
  );
  const omlxConfigured = Boolean(
    (formData.llm_omlx_protocol ?? settings?.llm_omlx_protocol) &&
    (formData.llm_omlx_host ?? settings?.llm_omlx_host)?.trim() &&
    (formData.llm_omlx_port ?? settings?.llm_omlx_port)
  );
  const embeddingOpenAiConfigured = Boolean((formData.openai_api_key ?? settings?.openai_api_key)?.trim());
  const embeddingOllamaConfigured = Boolean(
    (formData.ollama_protocol ?? settings?.ollama_protocol) &&
    (formData.ollama_host ?? settings?.ollama_host)?.trim() &&
    (formData.ollama_port ?? settings?.ollama_port)
  );
  const embeddingLlamaCppConfigured = Boolean(
    (formData.llama_cpp_protocol ?? settings?.llama_cpp_protocol) &&
    (formData.llama_cpp_host ?? settings?.llama_cpp_host)?.trim() &&
    (formData.llama_cpp_port ?? settings?.llama_cpp_port)
  );
  const embeddingLmstudioConfigured = Boolean(
    (formData.lmstudio_protocol ?? settings?.lmstudio_protocol) &&
    (formData.lmstudio_host ?? settings?.lmstudio_host)?.trim() &&
    (formData.lmstudio_port ?? settings?.lmstudio_port)
  );
  const embeddingOmlxConfigured = Boolean(
    (formData.omlx_protocol ?? settings?.omlx_protocol) &&
    (formData.omlx_host ?? settings?.omlx_host)?.trim() &&
    (formData.omlx_port ?? settings?.omlx_port)
  );
  const activeAuthProvider = AUTH_PROVIDER_OPTIONS.find((provider) => provider.value === activeAuthProviderValue) || AUTH_PROVIDER_OPTIONS[0];
  const ldapConfigured = Boolean(ldapFormData.ldap_host.trim() || ldapConfig?.server_url?.trim());
  const ldapCanReuseStoredCredentials = Boolean(
    ldapConfig?.server_url
    && ldapConfig?.bind_dn
    && ldapConfig.server_url === buildServerUrl()
    && ldapConfig.bind_dn.trim().toLowerCase() === ldapFormData.bind_dn.trim().toLowerCase()
  );
  const ldapTestDisabled = ldapTesting
    || !ldapFormData.ldap_host.trim()
    || !ldapFormData.bind_dn.trim()
    || (!ldapFormData.bind_password && !ldapCanReuseStoredCredentials);
  const isAdmin = currentUser?.role === 'admin';
  const manualDefaultChatModel = (() => {
    if (formData.default_chat_model !== undefined) {
      return formData.default_chat_model ?? null;
    }
    return settings?.default_chat_model ?? null;
  })();
  const defaultChatModelOptions = filteredChatModels.map((model) => ({
    value: toScopedModelIdentifier(model),
    label: `${model.id} (${CHAT_MODEL_PROVIDER_LABELS[model.provider] || model.provider})`,
  }));
  const manualDefaultExistsInOptions = !!manualDefaultChatModel
    && defaultChatModelOptions.some((option) => option.value === manualDefaultChatModel);
  const effectiveDefaultChatModelDisplay = formatModelIdentifierForDisplay(
    manualDefaultExistsInOptions ? manualDefaultChatModel : automaticDefaultChatModel,
    filteredChatModels,
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
            if ((e.key === 'Tab' || e.key === 'Enter') && settingsFilterInput.trim()) {
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
          Connect external clients (e.g., Open WebUI) using <code>{getDisplayUrl('/v1')}</code>.
        </p>
        <p className="field-help" style={{ marginTop: '0.5rem' }}>
          Default model: <code>{effectiveDefaultChatModelDisplay}</code>. <code>/v1/models</code> returns {formData.openapi_sync_chat_models !== false ? 'your Chat Models selection' : 'a separately configured OpenAPI models list'}.
        </p>
        {(!authStatus?.api_key_configured || window.location.protocol === 'http:') && (
          <div className="field-warning" style={{ marginTop: '0.75rem', padding: '0.75rem', backgroundColor: 'rgba(255, 193, 7, 0.15)', borderLeft: '3px solid #ffc107', borderRadius: '4px' }}>
            <strong>Security:</strong>
            {!authStatus?.api_key_configured && (
              <span> {renderApiKeySecurityWarning()}</span>
            )}
            {window.location.protocol === 'http:' && (
              <span> {renderHttpSecurityWarning(!authStatus?.api_key_configured)}</span>
            )}
          </div>
        )}
      </div>

      <div className="api-info-box" data-settings-filter-card="true">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem' }}>
          <strong>Cloud Drive OAuth</strong>
          <Popover
            trigger="click"
            position="bottom"
            content={renderCloudDriveOAuthSetupPopover(getCloudOAuthCallbackUrl())}
          >
            <button
              type="button"
              className="btn btn-sm"
              aria-label="Cloud drive OAuth setup"
              title="Cloud drive OAuth setup"
              style={{ padding: '4px 7px' }}
            >
              <Info size={14} />
            </button>
          </Popover>
        </div>
        <p className="field-help" style={{ marginTop: '0.5rem' }}>
          OneDrive, SharePoint, and Google Drive userspace mounts require provider OAuth apps configured through environment variables.
        </p>
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
                <span title={settings?.mcp_default_route_auth_method === 'oauth2' ? 'OAuth2 protected' : settings?.mcp_default_route_auth_method === 'client_credentials' ? 'Client credentials protected' : 'Password protected'}>
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
                  <span title={route.auth_method === 'oauth2' ? 'OAuth2 (LDAP)' : route.auth_method === 'client_credentials' ? 'Client credentials' : 'Password protected'}>
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

      <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />

      <form
        ref={settingsFormRef}
        onSubmit={handleSubmit}
        autoComplete="off"
        data-form-type="other"
        data-bwignore="true"
        data-lpignore="true"
        data-1p-ignore="true"
      >
        {/* Server Branding */}
        <fieldset
          id="setting-server_branding"
          className={highlightSetting === 'server_branding' ? 'highlight-setting' : ''}
        >
          <legend>Server Branding</legend>
          <p className="fieldset-help">
            Customize the server name displayed in the UI, API model name, and MCP server identity.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '1rem',
              alignItems: 'start',
            }}
          >
            <div className="form-group">
              <label>Server Name</label>
              <input
                type="text"
                value={formData.server_name ?? settings?.server_name ?? 'Ragtime'}
                onChange={(e) => setFormData({ ...formData, server_name: e.target.value })}
                placeholder="Ragtime"
              />
            </div>

            <div className="form-group">
              <label className="chat-toggle-control" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={formData.authenticated_webgl_background_enabled ?? settings?.authenticated_webgl_background_enabled ?? true}
                    onChange={(e) => setFormData({
                      ...formData,
                      authenticated_webgl_background_enabled: e.target.checked,
                    })}
                  />
                  <span className="toggle-slider"></span>
                </label>
                <span>Animated Background After Login</span>
              </label>
              <p className="field-help">
                Show the WebGL gradient behind authenticated app pages. Disable this to use the static theme background after login.
              </p>
            </div>
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
              <span className="llm-provider-status-item" title={openRouterConfigured ? 'OpenRouter configured' : 'OpenRouter not configured'}>
                <span
                  className={`llm-provider-status-dot ${openRouterConfigured ? 'configured' : ''}`}
                  aria-label={openRouterConfigured ? 'OpenRouter configured' : 'OpenRouter not configured'}
                />
                <span className="llm-provider-status-label">OpenRouter</span>
              </span>
              <span className="llm-provider-status-item" title={ollamaConfigured ? 'Ollama configured' : 'Ollama not configured'}>
                <span
                  className={`llm-provider-status-dot ${ollamaConfigured ? 'configured' : ''}`}
                  aria-label={ollamaConfigured ? 'Ollama configured' : 'Ollama not configured'}
                />
                <span className="llm-provider-status-label">Ollama</span>
              </span>
              <span className="llm-provider-status-item" title={llamaCppConfigured ? 'llama.cpp configured' : 'llama.cpp not configured'}>
                <span
                  className={`llm-provider-status-dot ${llamaCppConfigured ? 'configured' : ''}`}
                  aria-label={llamaCppConfigured ? 'llama.cpp configured' : 'llama.cpp not configured'}
                />
                <span className="llm-provider-status-label">llama.cpp</span>
              </span>
              <span className="llm-provider-status-item" title={lmstudioConfigured ? 'LM Studio configured' : 'LM Studio not configured'}>
                <span
                  className={`llm-provider-status-dot ${lmstudioConfigured ? 'configured' : ''}`}
                  aria-label={lmstudioConfigured ? 'LM Studio configured' : 'LM Studio not configured'}
                />
                <span className="llm-provider-status-label">LM Studio</span>
              </span>
              <span className="llm-provider-status-item" title={omlxConfigured ? 'oMLX configured' : 'oMLX not configured'}>
                <span
                  className={`llm-provider-status-dot ${omlxConfigured ? 'configured' : ''}`}
                  aria-label={omlxConfigured ? 'oMLX configured' : 'oMLX not configured'}
                />
                <span className="llm-provider-status-label">oMLX</span>
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
            <div className="input-with-button input-with-actions">
            <select
              value={formData.llm_provider || 'openai'}
              onChange={(e) => {
                const newProvider = e.target.value as 'openai' | 'anthropic' | 'openrouter' | 'ollama' | 'llama_cpp' | 'lmstudio' | 'omlx' | 'github_copilot';
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
                resetLlmModelsState();
                if (newProvider !== 'ollama') {
                  resetLlmOllamaState();
                }

                if (newProvider === 'github_copilot' && ((copilotAuthMode === 'oauth' && (copilotAuthStatus?.connected || settings?.has_github_copilot_auth)) || (copilotAuthMode === 'pat' && hasCopilotPatToken))) {
                  fetchCopilotModels();
                }
              }}
            >
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="openrouter">OpenRouter</option>
              <option value="ollama">Ollama</option>
              <option value="llama_cpp">llama.cpp</option>
              <option value="lmstudio">LM Studio</option>
              <option value="omlx">oMLX</option>
              <option value="github_copilot">GitHub Copilot</option>
            </select>
            {/* Quick-fill from embedding Ollama when it has a real host */}
            {formData.llm_provider === 'ollama' && formData.embedding_provider === 'ollama' && formData.ollama_host?.trim() && (
              <button
                type="button"
                className="btn btn-test"
                onClick={() => {
                  setFormData({
                    ...formData,
                    llm_ollama_protocol: formData.ollama_protocol || DEFAULT_OLLAMA_PROTOCOL,
                    llm_ollama_host: formData.ollama_host || '',
                    llm_ollama_port: formData.ollama_port || DEFAULT_OLLAMA_PORT,
                  });
                  resetLlmOllamaState();
                }}
              >
                Use Embedding Server
              </button>
            )}
            {formData.llm_provider === 'lmstudio' && formData.embedding_provider === 'lmstudio' && formData.lmstudio_host?.trim() && (
              <button
                type="button"
                className="btn btn-test"
                onClick={() => {
                  setFormData({
                    ...formData,
                    llm_lmstudio_protocol: formData.lmstudio_protocol || DEFAULT_LMSTUDIO_PROTOCOL,
                    llm_lmstudio_host: formData.lmstudio_host || '',
                    llm_lmstudio_port: formData.lmstudio_port || DEFAULT_LMSTUDIO_PORT,
                  });
                  resetLlmModelsState();
                }}
              >
                Use Embedding Server
              </button>
            )}
            {formData.llm_provider === 'omlx' && formData.embedding_provider === 'omlx' && formData.omlx_host?.trim() && (
              <button
                type="button"
                className="btn btn-test"
                onClick={() => {
                  setFormData({
                    ...formData,
                    llm_omlx_protocol: formData.omlx_protocol || DEFAULT_OMLX_PROTOCOL,
                    llm_omlx_host: formData.omlx_host || '',
                    llm_omlx_port: formData.omlx_port || DEFAULT_OMLX_PORT,
                  });
                  resetLlmModelsState();
                }}
              >
                Use Embedding Server
              </button>
            )}
            </div>
          </div>

          {/* Ollama LLM Server Connection - only show when Ollama is selected */}
          {formData.llm_provider === 'ollama' && (
            <>
            <OllamaConnectionForm
              protocol={formData.llm_ollama_protocol || 'http'}
              host={formData.llm_ollama_host || ''}
              port={formData.llm_ollama_port || DEFAULT_OLLAMA_PORT}
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
                resetLlmOllamaState();
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, llm_ollama_host: host });
                resetLlmOllamaState();
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, llm_ollama_port: port });
                resetLlmOllamaState();
              }}
              onModelChange={(model) => setFormData({ ...formData, llm_model: model })}
              onFetchModels={() => testLlmOllamaConnection(
                formData.llm_ollama_protocol || 'http',
                formData.llm_ollama_host || 'localhost',
                formData.llm_ollama_port || DEFAULT_OLLAMA_PORT
              )}
            />
            </>
          )}

          {formData.llm_provider === 'llama_cpp' && (
            <>
            <OllamaConnectionForm
              protocol={formData.llm_llama_cpp_protocol || DEFAULT_LLAMA_CPP_PROTOCOL}
              host={formData.llm_llama_cpp_host || DEFAULT_LLAMA_CPP_HOST}
              port={formData.llm_llama_cpp_port || DEFAULT_LLAMA_CPP_CHAT_PORT}
              model={formData.llm_model || ''}
              connected={llmModelsLoaded && formData.llm_provider === 'llama_cpp'}
              connecting={llmModelsFetching}
              error={formData.llm_provider === 'llama_cpp' ? llmModelsError : null}
              models={llmModels.map((m) => ({ id: m.id, name: m.name, context_limit: m.context_limit }))}
              providerLabel="llama.cpp"
              defaultPort={DEFAULT_LLAMA_CPP_CHAT_PORT}
              hostPlaceholder={DEFAULT_LLAMA_CPP_HOST}
              modelLabel="Model"
              modelPlaceholder="my-chat-model"
              connectedHelpText="Select a model from your llama.cpp server."
              disconnectedHelpText="Click &quot;Fetch Models&quot; to discover the active llama.cpp model, or enter its alias manually."
              onProtocolChange={(protocol) => {
                setFormData({ ...formData, llm_llama_cpp_protocol: protocol });
                resetLlmModelsState();
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, llm_llama_cpp_host: host });
                resetLlmModelsState();
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, llm_llama_cpp_port: port });
                resetLlmModelsState();
              }}
              onModelChange={(model) => setFormData({ ...formData, llm_model: model })}
              onFetchModels={fetchLlamaCppLlmModels}
            />
            <p className="field-help">
              llama.cpp does not support load/unload over its HTTP API. Start the llama.cpp server with the desired model already loaded (for example, <code>llama-server -m model.gguf</code>); Ragtime will use whichever model the server is currently serving.
            </p>
            </>
          )}

          {formData.llm_provider === 'lmstudio' && (
            <>
              <div className="form-group">
                <label>LM Studio API Key</label>
                <input
                  type="password"
                  value={formData.lmstudio_api_key || ''}
                  onChange={(e) => setFormData({ ...formData, lmstudio_api_key: e.target.value })}
                  placeholder="sk-lm-... (optional)"
                  autoComplete="off"
                />
                <p className="form-help">Optional. Leave blank if LM Studio is running without authentication.</p>
              </div>
              <OllamaConnectionForm
                protocol={formData.llm_lmstudio_protocol || DEFAULT_LMSTUDIO_PROTOCOL}
                host={formData.llm_lmstudio_host || DEFAULT_LMSTUDIO_HOST}
                port={formData.llm_lmstudio_port || DEFAULT_LMSTUDIO_PORT}
                model={formData.llm_model || ''}
                connected={llmModelsLoaded && formData.llm_provider === 'lmstudio'}
                connecting={llmModelsFetching}
                error={formData.llm_provider === 'lmstudio' ? llmModelsError : null}
                models={llmModels.map((m) => ({
                  id: m.id,
                  name: m.name,
                  context_limit: m.context_limit,
                  loaded: m.loaded,
                }))}
                providerLabel="LM Studio"
                defaultPort={DEFAULT_LMSTUDIO_PORT}
                hostPlaceholder={DEFAULT_LMSTUDIO_HOST}
                modelLabel="Model"
                modelPlaceholder="gemma-4-31b-it-mlx"
                connectedHelpText="Select a chat-capable model from LM Studio."
                disconnectedHelpText="Click &quot;Fetch Models&quot; to discover LM Studio models, or enter a model key manually."
                onProtocolChange={(protocol) => {
                  setFormData({ ...formData, llm_lmstudio_protocol: protocol });
                  resetLlmModelsState();
                }}
                onHostChange={(host) => {
                  setFormData({ ...formData, llm_lmstudio_host: host });
                  resetLlmModelsState();
                }}
                onPortChange={(port) => {
                  setFormData({ ...formData, llm_lmstudio_port: port });
                  resetLlmModelsState();
                }}
                onModelChange={(model) => setFormData({ ...formData, llm_model: model })}
                onFetchModels={fetchLmstudioLlmModels}
                modelAction={(() => {
                  const selected = llmModels.find((m) => m.id === formData.llm_model);
                  const isLoaded = !!(selected?.loaded || (selected?.loaded_instances && selected.loaded_instances.length > 0));
                  if (!formData.llm_model) {
                    return (
                      <button type="button" className="btn btn-test" disabled>
                        Load Selected
                      </button>
                    );
                  }
                  return isLoaded ? (
                    <button type="button" className="btn btn-test" onClick={() => unloadSelectedLmstudioModel('llm')} disabled={lmstudioModelActionLoading}>
                      Unload Selected
                    </button>
                  ) : (
                    <button type="button" className="btn btn-test" onClick={() => loadSelectedLmstudioModel('llm')} disabled={lmstudioModelActionLoading}>
                      Load Selected
                    </button>
                  );
                })()}
              />
            </>
          )}

          {formData.llm_provider === 'omlx' && (
            <>
              <div className="form-group">
                <label>oMLX API Key</label>
                <input
                  type="password"
                  value={formData.omlx_api_key || ''}
                  onChange={(e) => setFormData({ ...formData, omlx_api_key: e.target.value })}
                  placeholder="optional"
                  autoComplete="off"
                />
                <p className="form-help">Optional. Leave blank if oMLX is running without authentication.</p>
              </div>
              <OllamaConnectionForm
                protocol={formData.llm_omlx_protocol || DEFAULT_OMLX_PROTOCOL}
                host={formData.llm_omlx_host || DEFAULT_OMLX_HOST}
                port={formData.llm_omlx_port || DEFAULT_OMLX_PORT}
                model={formData.llm_model || ''}
                connected={llmModelsLoaded && formData.llm_provider === 'omlx'}
                connecting={llmModelsFetching}
                error={formData.llm_provider === 'omlx' ? llmModelsError : null}
                models={llmModels.map((m) => ({
                  id: m.id,
                  name: m.name,
                  context_limit: m.context_limit,
                }))}
                providerLabel="oMLX"
                defaultPort={DEFAULT_OMLX_PORT}
                hostPlaceholder={DEFAULT_OMLX_HOST}
                modelLabel="Model"
                modelPlaceholder="qwen3-coder-next-8bit"
                connectedHelpText="Select a model from oMLX."
                disconnectedHelpText="Click &quot;Fetch Models&quot; to discover oMLX models, or enter a model id manually."
                onProtocolChange={(protocol) => {
                  setFormData({ ...formData, llm_omlx_protocol: protocol });
                  resetLlmModelsState();
                }}
                onHostChange={(host) => {
                  setFormData({ ...formData, llm_omlx_host: host });
                  resetLlmModelsState();
                }}
                onPortChange={(port) => {
                  setFormData({ ...formData, llm_omlx_port: port });
                  resetLlmModelsState();
                }}
                onModelChange={(model) => setFormData({ ...formData, llm_model: model })}
                onFetchModels={fetchOmlxLlmModels}
              />
              <p className="field-help">
                oMLX manages model loading in its own admin UI and serves selected models through its OpenAI-compatible API.
              </p>
            </>
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
                    resetLlmModelsState();
                    // Also reset embedding models since they use the same key
                    resetEmbeddingModelsState();
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
                    resetLlmModelsState();
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
          ) : formData.llm_provider === 'openrouter' ? (
            <div className="form-group">
              <label>OpenRouter API Key</label>
              <div className="input-with-button">
                <input
                  type="password"
                  value={formData.openrouter_api_key || ''}
                  onChange={(e) => {
                    setFormData({ ...formData, openrouter_api_key: e.target.value });
                    resetLlmModelsState();
                  }}
                  placeholder="sk-or-..."
                />
                <button
                  type="button"
                  className={`btn btn-test ${llmModelsLoaded && formData.llm_provider === 'openrouter' ? 'btn-connected' : ''}`}
                  onClick={() => fetchLlmModels('openrouter', formData.openrouter_api_key || '')}
                  disabled={llmModelsFetching || !formData.openrouter_api_key}
                >
                  {llmModelsFetching ? 'Fetching...' : llmModelsLoaded && formData.llm_provider === 'openrouter' ? 'Loaded' : 'Fetch Models'}
                </button>
              </div>
              {llmModelsError && formData.llm_provider === 'openrouter' && (
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
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  <label style={{ display: 'inline-flex', gap: '0.35rem', alignItems: 'center', marginBottom: 0 }}>
                    <input
                      type="radio"
                      name="copilot-auth-mode"
                      checked={copilotAuthMode === 'oauth'}
                      onChange={() => {
                        setCopilotAuthMode('oauth');
                        setFormData((prev) => ({ ...prev, github_models_api_token: '' }));
                        resetLlmModelsState();
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
                        resetLlmModelsState();
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
                      resetLlmModelsState();
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
                  onClick={() => fetchCopilotModels()}
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
                        <InlineCopyButton
                          copyText={copilotDeviceCode}
                          className="copilot-device-copy-btn"
                          title="Copy device code"
                          ariaLabel="Copy device code"
                          copiedTitle="Device code copied"
                          copiedAriaLabel="Device code copied"
                          iconSize={16}
                          feedbackMs={2000}
                          onCopySuccess={handleCopilotDeviceCodeCopied}
                          onCopyError={handleCopilotDeviceCodeCopyError}
                        />
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

          <div className="form-row-3">
            {/* Chat Model Filter */}
            <div className="form-group">
              <label>Chat Models</label>
              <button
                type="button"
                className="btn btn-secondary settings-control-height"
                onClick={openModelFilterModal}
              >
                Configure Chat Models
              </button>
              <p className="field-help">
                Limit which models appear in the Chat view dropdown. Includes all configured providers (OpenAI, Anthropic, OpenRouter, Ollama, llama.cpp, GitHub Copilot).
              </p>
            </div>

            {/* Default Chat Model configuration */}
            <div className="form-group">
              <label>Default Chat Model{chatModelsLoading && <>{' '}<MiniLoadingSpinner variant="icon" size={12} title="Loading models..." /></>}</label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <select
                  style={{ flex: 1 }}
                  value={manualDefaultChatModel ?? automaticDefaultChatModel ?? ''}
                  onChange={(e) => {
                    const selectedValue = e.target.value;
                    setFormData({
                      ...formData,
                      default_chat_model: selectedValue || null,
                    });
                  }}
                  disabled={chatModelsLoading || filteredChatModels.length === 0}
                >
                  {defaultChatModelOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                {manualDefaultChatModel && (
                  <button
                    type="button"
                    className="btn btn-secondary settings-control-height"
                    style={{ padding: '0 0.5rem', fontSize: '0.85em', whiteSpace: 'nowrap' }}
                    title="Reset to default model"
                    onClick={() => setFormData({ ...formData, default_chat_model: null })}
                  >
                    Reset
                  </button>
                )}
              </div>
              <p className="field-help">
                {manualDefaultChatModel
                  ? 'Manually selected. Click Reset to use the default.'
                  : 'Using the default model. Select a different model to override.'}
              </p>
            </div>

            {/* OpenAPI Models configuration */}
            <div className="form-group">
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

          {/* Show OpenAI key field for embeddings when the LLM uses another provider */}
          {(formData.llm_provider === 'anthropic' || formData.llm_provider === 'openrouter' || formData.llm_provider === 'ollama' || formData.llm_provider === 'llama_cpp' || formData.llm_provider === 'lmstudio' || formData.llm_provider === 'github_copilot') && formData.embedding_provider === 'openai' && (
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
                Required for OpenAI embeddings when using a different LLM provider.
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
                <label>Image Max Width (px)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="320"
                    max="4096"
                    step="16"
                    style={{ flex: 1 }}
                    value={formData.image_payload_max_width ?? 1024}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        image_payload_max_width: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '56px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.image_payload_max_width ?? 1024).toLocaleString()}
                  </span>
                </div>
                <p className="field-help">
                  Maximum width for inline image attachments before downsampling.
                </p>
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label>Image Max Height (px)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="240"
                    max="4096"
                    step="16"
                    style={{ flex: 1 }}
                    value={formData.image_payload_max_height ?? 1024}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        image_payload_max_height: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '56px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.image_payload_max_height ?? 1024).toLocaleString()}
                  </span>
                </div>
                <p className="field-help">
                  Maximum height for inline image attachments before downsampling.
                </p>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Image Max Pixels</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="76800"
                    max="8000000"
                    step="25600"
                    style={{ flex: 1 }}
                    value={formData.image_payload_max_pixels ?? 786432}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        image_payload_max_pixels: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '72px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.image_payload_max_pixels ?? 786432).toLocaleString()}
                  </span>
                </div>
                <p className="field-help">
                  Total pixel budget (width x height). Images exceeding this are scaled down proportionally.
                </p>
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label>Image Max Bytes</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="50000"
                    max="5000000"
                    step="10000"
                    style={{ flex: 1 }}
                    value={formData.image_payload_max_bytes ?? 350000}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        image_payload_max_bytes: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '72px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {(formData.image_payload_max_bytes ?? 350000).toLocaleString()}
                  </span>
                </div>
                <p className="field-help">
                  Max encoded size of each image. Larger images are re-compressed with lower JPEG quality.
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

            {/* API Output Configuration */}
            <div style={{ marginTop: '1rem' }}>
              <h4 style={{ margin: '0 0 4px' }}>API Output</h4>
              <p className="field-help" style={{ marginBottom: '12px' }}>
                Configure how tool call output is handled in OpenAI-compatible API responses (e.g., when using OpenWebUI or other clients).
                This does not affect MCP or the built-in chat interface.
              </p>

              <div className="form-row">
                <div className="form-group" style={{ flex: 1 }}>
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
              </div>
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
              <span className="llm-provider-status-item" title={embeddingLlamaCppConfigured ? 'llama.cpp configured' : 'llama.cpp not configured'}>
                <span
                  className={`llm-provider-status-dot ${embeddingLlamaCppConfigured ? 'configured' : ''}`}
                  aria-label={embeddingLlamaCppConfigured ? 'llama.cpp configured' : 'llama.cpp not configured'}
                />
                <span className="llm-provider-status-label">llama.cpp</span>
              </span>
              <span className="llm-provider-status-item" title={embeddingLmstudioConfigured ? 'LM Studio configured' : 'LM Studio not configured'}>
                <span
                  className={`llm-provider-status-dot ${embeddingLmstudioConfigured ? 'configured' : ''}`}
                  aria-label={embeddingLmstudioConfigured ? 'LM Studio configured' : 'LM Studio not configured'}
                />
                <span className="llm-provider-status-label">LM Studio</span>
              </span>
              <span className="llm-provider-status-item" title={embeddingOmlxConfigured ? 'oMLX configured' : 'oMLX not configured'}>
                <span
                  className={`llm-provider-status-dot ${embeddingOmlxConfigured ? 'configured' : ''}`}
                  aria-label={embeddingOmlxConfigured ? 'oMLX configured' : 'oMLX not configured'}
                />
                <span className="llm-provider-status-label">oMLX</span>
              </span>
            </span>
          </legend>
          <p className="fieldset-help">
            Configure how document embeddings are generated for FAISS indexes.
          </p>

          <div className="form-row">
            <div className="form-group" id="setting-embedding_provider">
              <label>Provider</label>
              <div className="input-with-button input-with-actions">
              <select
                value={formData.embedding_provider || 'ollama'}
                onChange={(e) => {
                  const newProvider = e.target.value as 'ollama' | 'openai' | 'llama_cpp' | 'lmstudio' | 'omlx';
                  setFormData({
                    ...formData,
                    embedding_provider: newProvider,
                    // Set sensible default model when switching providers
                    embedding_model:
                      newProvider === 'ollama'
                        ? 'nomic-embed-text'
                        : newProvider === 'llama_cpp'
                          ? ''
                          : newProvider === 'lmstudio'
                            ? ''
                            : newProvider === 'omlx'
                              ? ''
                            : 'text-embedding-3-small',
                  });
                  // Reset Ollama connection state when switching providers
                  if (newProvider !== 'ollama') {
                    resetEmbeddingOllamaState();
                  }
                  // Reset embedding models when switching away from OpenAI
                  if (newProvider !== 'openai') {
                    resetEmbeddingModelsState();
                  }
                }}
              >
                <option value="ollama">Ollama</option>
                <option value="llama_cpp">llama.cpp</option>
                <option value="lmstudio">LM Studio</option>
                <option value="omlx">oMLX</option>
                <option value="openai">OpenAI</option>
              </select>
              {/* Quick-fill from LLM Ollama when it has a real host */}
              {formData.embedding_provider === 'ollama' && formData.llm_provider === 'ollama' && formData.llm_ollama_host?.trim() && (
                <button
                  type="button"
                  className="btn btn-test"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      ollama_protocol: formData.llm_ollama_protocol || DEFAULT_OLLAMA_PROTOCOL,
                      ollama_host: formData.llm_ollama_host || '',
                      ollama_port: formData.llm_ollama_port || DEFAULT_OLLAMA_PORT,
                    });
                    resetEmbeddingOllamaState();
                  }}
                >
                  Use LLM Server
                </button>
              )}
              {formData.embedding_provider === 'lmstudio' && formData.llm_provider === 'lmstudio' && formData.llm_lmstudio_host?.trim() && (
                <button
                  type="button"
                  className="btn btn-test"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      lmstudio_protocol: formData.llm_lmstudio_protocol || DEFAULT_LMSTUDIO_PROTOCOL,
                      lmstudio_host: formData.llm_lmstudio_host || '',
                      lmstudio_port: formData.llm_lmstudio_port || DEFAULT_LMSTUDIO_PORT,
                    });
                    resetEmbeddingModelsState();
                  }}
                >
                  Use LLM Server
                </button>
              )}
              {formData.embedding_provider === 'omlx' && formData.llm_provider === 'omlx' && formData.llm_omlx_host?.trim() && (
                <button
                  type="button"
                  className="btn btn-test"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      omlx_protocol: formData.llm_omlx_protocol || DEFAULT_OMLX_PROTOCOL,
                      omlx_host: formData.llm_omlx_host || '',
                      omlx_port: formData.llm_omlx_port || DEFAULT_OMLX_PORT,
                    });
                    resetEmbeddingModelsState();
                  }}
                >
                  Use LLM Server
                </button>
              )}
              </div>
              <p className="field-help">
                Note: Anthropic does not offer embedding models. Use Ollama, llama.cpp, LM Studio, oMLX, or OpenAI for document embeddings.
              </p>
            </div>
            {/* Show embedding dimension info */}
            {(() => {
              // Get the dimension from the selected model if available
              const selectedOllamaModel = ollamaModels.find(m => m.name === formData.embedding_model);
              const selectedOpenAIModel = embeddingModels.find(m => m.id === formData.embedding_model);
              const selectedLlamaCppModel = embeddingModels.find(m => m.id === formData.embedding_model);
              const selectedLmstudioModel = embeddingModels.find(m => m.id === formData.embedding_model);
              const selectedModelDimension = selectedOllamaModel?.dimensions || selectedOpenAIModel?.dimensions || selectedLlamaCppModel?.dimensions || selectedLmstudioModel?.dimensions;
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
              port={formData.ollama_port || DEFAULT_OLLAMA_PORT}
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
                resetEmbeddingOllamaState();
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, ollama_host: host });
                resetEmbeddingOllamaState();
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, ollama_port: port });
                resetEmbeddingOllamaState();
              }}
              onModelChange={(model) => setFormData({ ...formData, embedding_model: model })}
              onFetchModels={handleTestOllamaConnection}
            />
          )}

          {formData.embedding_provider === 'llama_cpp' && (
            <>
            <OllamaConnectionForm
              protocol={formData.llama_cpp_protocol || DEFAULT_LLAMA_CPP_PROTOCOL}
              host={formData.llama_cpp_host || DEFAULT_LLAMA_CPP_HOST}
              port={formData.llama_cpp_port || DEFAULT_LLAMA_CPP_EMBEDDING_PORT}
              model={formData.embedding_model || ''}
              connected={embeddingModelsLoaded && formData.embedding_provider === 'llama_cpp'}
              connecting={embeddingModelsFetching}
              error={formData.embedding_provider === 'llama_cpp' ? embeddingModelsError : null}
              models={embeddingModels.map((m) => ({ id: m.id, name: m.name, dimensions: m.dimensions }))}
              providerLabel="llama.cpp"
              defaultPort={DEFAULT_LLAMA_CPP_EMBEDDING_PORT}
              hostPlaceholder={DEFAULT_LLAMA_CPP_HOST}
              modelLabel="Embedding Model"
              modelPlaceholder="my-embed-model"
              connectedHelpText="Select an embedding model from your llama.cpp server."
              disconnectedHelpText="Click &quot;Fetch Models&quot; to probe the llama.cpp embedding server, or enter the model alias manually."
              onProtocolChange={(protocol) => {
                setFormData({ ...formData, llama_cpp_protocol: protocol });
                resetEmbeddingModelsState();
              }}
              onHostChange={(host) => {
                setFormData({ ...formData, llama_cpp_host: host });
                resetEmbeddingModelsState();
              }}
              onPortChange={(port) => {
                setFormData({ ...formData, llama_cpp_port: port });
                resetEmbeddingModelsState();
              }}
              onModelChange={(model) => setFormData({ ...formData, embedding_model: model })}
              onFetchModels={fetchLlamaCppEmbeddingModels}
            />
            <p className="field-help">
              llama.cpp does not support load/unload over its HTTP API. Start the embedding server with <code>--embedding</code> and the desired model already loaded (for example, <code>llama-server --embedding -m embed-model.gguf</code>).
            </p>
            </>
          )}

          {formData.embedding_provider === 'lmstudio' && (
            <>
              <div className="form-group">
                <label>LM Studio API Key</label>
                <input
                  type="password"
                  value={formData.lmstudio_api_key || ''}
                  onChange={(e) => setFormData({ ...formData, lmstudio_api_key: e.target.value })}
                  placeholder="sk-lm-... (optional)"
                  autoComplete="off"
                />
                <p className="form-help">Optional. Leave blank if LM Studio is running without authentication.</p>
              </div>
              <OllamaConnectionForm
                protocol={formData.lmstudio_protocol || DEFAULT_LMSTUDIO_PROTOCOL}
                host={formData.lmstudio_host || DEFAULT_LMSTUDIO_HOST}
                port={formData.lmstudio_port || DEFAULT_LMSTUDIO_PORT}
                model={formData.embedding_model || ''}
                connected={embeddingModelsLoaded && formData.embedding_provider === 'lmstudio'}
                connecting={embeddingModelsFetching}
                error={formData.embedding_provider === 'lmstudio' ? embeddingModelsError : null}
                models={embeddingModels.map((m) => ({
                  id: m.id,
                  name: m.name,
                  dimensions: m.dimensions,
                  context_limit: m.context_limit,
                  loaded: m.loaded,
                }))}
                providerLabel="LM Studio"
                defaultPort={DEFAULT_LMSTUDIO_PORT}
                hostPlaceholder={DEFAULT_LMSTUDIO_HOST}
                modelLabel="Embedding Model"
                modelPlaceholder="text-embedding-nomic-embed-text-v1.5"
                connectedHelpText="Select an embedding model from LM Studio."
                disconnectedHelpText="Click &quot;Fetch Models&quot; to discover LM Studio embedding models, or enter a model key manually."
                onProtocolChange={(protocol) => {
                  setFormData({ ...formData, lmstudio_protocol: protocol });
                  resetEmbeddingModelsState();
                }}
                onHostChange={(host) => {
                  setFormData({ ...formData, lmstudio_host: host });
                  resetEmbeddingModelsState();
                }}
                onPortChange={(port) => {
                  setFormData({ ...formData, lmstudio_port: port });
                  resetEmbeddingModelsState();
                }}
                onModelChange={(model) => setFormData({ ...formData, embedding_model: model })}
                onFetchModels={fetchLmstudioEmbeddingModels}
                modelAction={(() => {
                  const selected = embeddingModels.find((m) => m.id === formData.embedding_model);
                  const isLoaded = !!(selected?.loaded || (selected?.loaded_instances && selected.loaded_instances.length > 0));
                  if (!formData.embedding_model) {
                    return (
                      <button type="button" className="btn btn-test" disabled>
                        Load Selected
                      </button>
                    );
                  }
                  return isLoaded ? (
                    <button type="button" className="btn btn-test" onClick={() => unloadSelectedLmstudioModel('embedding')} disabled={lmstudioModelActionLoading}>
                      Unload Selected
                    </button>
                  ) : (
                    <button type="button" className="btn btn-test" onClick={() => loadSelectedLmstudioModel('embedding')} disabled={lmstudioModelActionLoading}>
                      Load Selected
                    </button>
                  );
                })()}
              />
            </>
          )}

          {formData.embedding_provider === 'omlx' && (
            <>
              <div className="form-group">
                <label>oMLX API Key</label>
                <input
                  type="password"
                  value={formData.omlx_api_key || ''}
                  onChange={(e) => setFormData({ ...formData, omlx_api_key: e.target.value })}
                  placeholder="optional"
                  autoComplete="off"
                />
                <p className="form-help">Optional. Leave blank if oMLX is running without authentication.</p>
              </div>
              <OllamaConnectionForm
                protocol={formData.omlx_protocol || DEFAULT_OMLX_PROTOCOL}
                host={formData.omlx_host || DEFAULT_OMLX_HOST}
                port={formData.omlx_port || DEFAULT_OMLX_PORT}
                model={formData.embedding_model || ''}
                connected={embeddingModelsLoaded && formData.embedding_provider === 'omlx'}
                connecting={embeddingModelsFetching}
                error={formData.embedding_provider === 'omlx' ? embeddingModelsError : null}
                models={embeddingModels.map((m) => ({
                  id: m.id,
                  name: m.name,
                  dimensions: m.dimensions,
                  context_limit: m.context_limit,
                }))}
                providerLabel="oMLX"
                defaultPort={DEFAULT_OMLX_PORT}
                hostPlaceholder={DEFAULT_OMLX_HOST}
                modelLabel="Embedding Model"
                modelPlaceholder="bge-m3"
                connectedHelpText="Select an embedding model from oMLX."
                disconnectedHelpText="Click &quot;Fetch Models&quot; to discover oMLX models, or enter a model id manually."
                onProtocolChange={(protocol) => {
                  setFormData({ ...formData, omlx_protocol: protocol });
                  resetEmbeddingModelsState();
                }}
                onHostChange={(host) => {
                  setFormData({ ...formData, omlx_host: host });
                  resetEmbeddingModelsState();
                }}
                onPortChange={(port) => {
                  setFormData({ ...formData, omlx_port: port });
                  resetEmbeddingModelsState();
                }}
                onModelChange={(model) => setFormData({ ...formData, embedding_model: model })}
                onFetchModels={fetchOmlxEmbeddingModels}
              />
              <p className="field-help">
                oMLX uses its OpenAI-compatible embeddings endpoint; non-embedding models may appear but fail dimension probing.
              </p>
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

          {/* Advanced Embedding Settings */}
          <details style={{ marginBottom: '16px' }} id="setting-embedding_advanced">
            <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Settings</summary>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Embedding Timeout (seconds)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <input
                    type="range"
                    min="30"
                    max="600"
                    step="10"
                    style={{ flex: 1 }}
                    value={formData.ollama_embedding_timeout_seconds ?? settings?.ollama_embedding_timeout_seconds ?? 180}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        ollama_embedding_timeout_seconds: parseInt(e.target.value, 10),
                      })
                    }
                  />
                  <span style={{ minWidth: '48px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                    {formData.ollama_embedding_timeout_seconds ?? settings?.ollama_embedding_timeout_seconds ?? 180}s
                  </span>
                </div>
                <p className="field-help">
                  Maximum time allowed per embedding sub-batch for any embedding provider.
                  If a batch times out, it is automatically retried with a smaller batch size.
                  Increase for slow hardware or large embedding models.
                </p>
              </div>

              <div className="form-group" style={{ flex: 1 }} id="setting-sequential_index_loading">
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
                  <span>{(formData.sequential_index_loading ?? settings?.sequential_index_loading ?? false) ? 'Sequential Index Loading' : 'Parallel Index Loading'}</span>
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
            </div>

            {/* OCR Settings */}
            <div id="setting-ocr" style={{ marginTop: '1rem' }}>
              <h4 style={{ margin: '0 0 4px' }}>OCR Settings</h4>
              <p className="field-help" style={{ marginBottom: '12px' }}>
                Configure default OCR (Optical Character Recognition) mode for extracting text from images during indexing.
              </p>

              <div className="form-row" style={formData.default_ocr_mode === 'vision' ? { display: 'flex', flexWrap: 'nowrap', gap: 'var(--space-md)' } : undefined}>
                <div className="form-group" style={{ flex: 1 }}>
                  <label>Default OCR Mode</label>
                  <select
                    value={formData.default_ocr_mode || 'disabled'}
                    onChange={(e) => {
                      const newMode = e.target.value as OcrMode;
                      setFormData({
                        ...formData,
                        default_ocr_mode: newMode,
                        default_ocr_provider: formData.default_ocr_provider || 'ollama',
                      });
                      if (newMode !== 'vision') {
                        setVisionModelsError(null);
                      }
                    }}
                  >
                    <option value="disabled">Disabled (skip images)</option>
                    <option value="tesseract">Tesseract (fast, traditional OCR)</option>
                    <option value="vision">Vision Model (semantic OCR with AI)</option>
                  </select>
                  <p className="field-help">
                    {formData.default_ocr_mode === 'disabled' && (
                      <>Image files will be skipped during indexing.</>
                    )}
                    {formData.default_ocr_mode === 'tesseract' && (
                      <>Fast traditional OCR using Tesseract. Good for screenshots and scanned documents with clear text.</>
                    )}
                    {formData.default_ocr_mode === 'vision' && (
                      <>
                        Semantic OCR using a vision-capable model. Better at understanding complex layouts, handwriting, and context.
                      </>
                    )}
                  </p>
                </div>

                {formData.default_ocr_mode === 'vision' && (
                  <div className="form-group" style={{ flex: 1 }}>
                    <label>Vision OCR Provider</label>
                    <select
                      value={selectedOcrProvider}
                      onChange={(e) => {
                        const provider = e.target.value as OcrProvider;
                        setFormData({
                          ...formData,
                          default_ocr_mode: 'vision',
                          default_ocr_provider: provider,
                          default_ocr_vision_model: null,
                        });
                        setVisionModels([]);
                        setVisionModelsError(null);
                      }}
                    >
                      {Object.entries(OCR_PROVIDER_LABELS).map(([provider, label]) => (
                        <option key={provider} value={provider}>{label}</option>
                      ))}
                    </select>
                    <p className="field-help">Uses the provider connection configured above for chat or model serving.</p>
                  </div>
                )}

                {formData.default_ocr_mode === 'vision' && (
                  <div className="form-group" style={{ flex: 1 }}>
                    <label>Vision Model</label>
                    <div style={{ display: 'flex', gap: '8px' }}>
                      <select
                        value={formData.default_ocr_vision_model || ''}
                        onChange={(e) => setFormData({ ...formData, default_ocr_vision_model: e.target.value || null })}
                        style={{ flex: 1 }}
                      >
                        <option value="">Select a model</option>
                        {formData.default_ocr_vision_model
                          && !visionModels.some((model) => model.name === formData.default_ocr_vision_model) && (
                          <option value={formData.default_ocr_vision_model}>{formData.default_ocr_vision_model}</option>
                        )}
                        {visionModels.map((model) => (
                          <option key={`${model.provider || selectedOcrProvider}:${model.name}`} value={model.name}>
                            {model.name}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={fetchVisionModels}
                        disabled={visionModelsLoading}
                      >
                        {visionModelsLoading ? 'Loading...' : 'Load'}
                      </button>
                    </div>
                    {visionModelsError && (
                      <p className="error-text" style={{ marginBottom: '8px' }}>{visionModelsError}</p>
                    )}
                    <p className="field-help">
                      Select a {selectedOcrProviderLabel} vision model for semantic OCR. Load checks provider metadata without running a vision request.
                    </p>
                  </div>
                )}

                {formData.default_ocr_mode === 'vision' && (
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

              {formData.default_ocr_mode === 'vision' && (
                <div className="form-group" style={{ marginTop: '-0.5rem', marginBottom: '1rem' }}>
                  {selectedOcrProvider === 'openai' && (
                    <p className="field-help" style={{ marginBottom: '8px' }}>
                      <span style={{ color: 'var(--warning-color, #b58900)' }}>
                        <strong>API cost note:</strong> OpenAI vision OCR sends image content to the selected model for each processed image. Cost and latency vary by model, image size, and OCR concurrency.
                      </span>
                    </p>
                  )}
                  <p className="field-help">
                    <span style={{ color: 'var(--warning-color, #b58900)' }}>
                      <strong>Performance note:</strong> Vision models are usually slower than Tesseract depending on provider and model size.
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
                        Use the smallest vision-capable model that reliably reads your document style. Local models trade speed for privacy and cost control; hosted models are often easier to operate but depend on API limits.
                      </div>
                    )}
                  </p>
                </div>
              )}

            </div>
          </details>

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

        {/* Authentication Provider Configuration */}
        <fieldset>
          <legend className="legend-with-status">
            <span>Authentication Providers</span>
            <span className="legend-divider" aria-hidden="true" />
            <span className="llm-provider-status-inline" aria-label="Authentication provider configuration status">
              <span
                className="llm-provider-status-item"
                title={(authProviderConfig?.local_users_enabled ?? true) ? 'Internal users enabled' : 'Internal users disabled'}
              >
                <span
                  className={`llm-provider-status-dot ${(authProviderConfig?.local_users_enabled ?? true) ? 'configured' : ''}`}
                  aria-hidden="true"
                />
                <span className="llm-provider-status-label">INTERNAL</span>
              </span>
              <span
                className="llm-provider-status-item"
                title={ldapConfigured ? 'LDAP configured' : 'LDAP not configured'}
              >
                <span
                  className={`llm-provider-status-dot ${ldapConfigured ? 'configured' : ''}`}
                  aria-hidden="true"
                />
                <span className="llm-provider-status-label">LDAP</span>
              </span>
            </span>
          </legend>
          <p className="fieldset-help">
            Both providers can be configured independently. Login attempts are tried in order: the env-based local admin, then internal users, then LDAP. OAuth and PKCE clients continue to use the existing login and token endpoints.
          </p>

          <div className="form-row">
            <div className="form-group">
              <label>Configure</label>
              <select
                value={activeAuthProvider.value}
                onChange={(e) => setActiveAuthProviderValue(e.target.value as typeof AUTH_PROVIDER_OPTIONS[number]['value'])}
              >
                {AUTH_PROVIDER_OPTIONS.map((provider) => (
                  <option key={provider.value} value={provider.value}>{provider.label}</option>
                ))}
              </select>
              <p className="field-help">{activeAuthProvider.description}</p>
            </div>

            {authProviderConfig && (
              <div className="form-group">
                <label>Internal Users</label>
                <div className="auth-provider-toggle-control">
                  <label
                    className="toggle-switch"
                    title={authProviderConfig.local_users_enabled ? 'Enabled' : 'Disabled'}
                  >
                    <input
                      type="checkbox"
                      checked={authProviderConfig.local_users_enabled}
                      onChange={(e) => setAuthProviderConfig({ ...authProviderConfig, local_users_enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <span className="auth-provider-toggle-state">
                    {authProviderConfig.local_users_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <p className="field-help auth-provider-help-tight">When disabled, only the env-based local admin and LDAP (if configured) can sign in.</p>
              </div>
            )}
          </div>

          {activeAuthProvider.value === 'local_managed' && (
            <>
              <div className="form-group">
                <h4>Internal Users &amp; Groups</h4>
                <p className="fieldset-help auth-provider-help-tight">Create local users and groups from focused dialogs to keep this page compact.</p>
                <div className="auth-provider-actions-row auth-provider-launchers">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => {
                      setShowCreateLocalUserModal(true);
                    }}
                  >
                    Create Internal User
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={openManageAuthGroupsModal}
                  >
                    <Pencil size={16} />
                    Manage Group Memberships
                  </button>
                </div>
              </div>
            </>
          )}

          {activeAuthProvider.value === 'ldap' && (
            <>
              <h4 style={{ margin: '0 0 4px' }}>Server Connection</h4>
              <p className="fieldset-help">Connect to your LDAP or Active Directory server.</p>

              <div className="form-row-4" style={{ gridTemplateColumns: '110px 1fr 90px' }}>
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
                    onChange={(e) => setLdapFormData({ ...ldapFormData, ldap_host: e.target.value })}
                    placeholder="ldap.example.com"
                  />
                </div>
                <div className="form-group">
                  <label>Port</label>
                  <input
                    type="number"
                    value={ldapFormData.ldap_port}
                    onChange={(e) => setLdapFormData({ ...ldapFormData, ldap_port: parseInt(e.target.value, 10) || 636 })}
                  />
                </div>
              </div>

              {ldapFormData.ldap_protocol === 'ldaps' && (
                <div className="form-group">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={ldapFormData.allow_self_signed}
                      onChange={(e) => setLdapFormData({ ...ldapFormData, allow_self_signed: e.target.checked })}
                      style={{ marginRight: '0.5rem' }}
                    />
                    <span>Allow self-signed certificates</span>
                  </label>
                  <p className="field-help">Skip SSL certificate validation. Use only for testing or with internal CAs.</p>
                </div>
              )}

              <div className="form-row-3 ldap-bind-row">
                <div className="form-group">
                  <label>Bind DN / Username</label>
                  <input
                    type="text"
                    value={ldapFormData.bind_dn}
                    onChange={(e) => setLdapFormData({ ...ldapFormData, bind_dn: e.target.value })}
                    placeholder="user@domain.com or CN=admin,DC=example,DC=com"
                  />
                  <p className="field-help">AD: user@domain.com or DOMAIN\user. OpenLDAP: full DN like cn=admin,dc=example,dc=com</p>
                </div>
                <div className="form-group">
                  <label>Bind Password</label>
                  <input
                    type="password"
                    value={ldapFormData.bind_password}
                    onChange={(e) => setLdapFormData({ ...ldapFormData, bind_password: e.target.value })}
                    placeholder={ldapConfig?.bind_dn ? '(password saved)' : 'Enter password'}
                  />
                </div>
                <div className="form-group ldap-bind-test-group">
                  <label>Connection Test</label>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={handleTestLdapConnection}
                    disabled={ldapTestDisabled}
                  >
                    {ldapTesting ? 'Testing...' : 'Test Connection & Discover'}
                  </button>
                </div>
              </div>

              {ldapTestResult && (
                <p className={`fieldset-help ${ldapTestResult.success ? 'connection-status success' : 'connection-status error'}`}>
                  {ldapTestResult.message}
                </p>
              )}

              {ldapDiscoveredOus.length > 0 && (
                <>
                  <h4 style={{ margin: '1rem 0 4px' }}>Search Configuration</h4>
                  <p className="fieldset-help">Discovered from your directory. Refine where users and groups are looked up.</p>
                  <div className="form-row">
                    <div className="form-group">
                      <label>User Search Base</label>
                      <select
                        value={ldapFormData.user_search_base}
                        onChange={(e) => setLdapFormData({ ...ldapFormData, user_search_base: e.target.value })}
                      >
                        <option value="">Select a search base...</option>
                        {ldapDiscoveredOus.map((ou) => (
                          <option key={ou} value={ou}>
                            {formatDnForDisplay(ou, ldapDiscoveredOus[0] || ou)}
                          </option>
                        ))}
                      </select>
                      <p className="field-help">Where to search for users. Select the root domain to search all users, or a specific OU to limit scope.</p>
                      {ldapFormData.user_search_base && (
                        <p className="field-help" style={{ fontFamily: 'var(--font-mono)' }}>
                          DN: {ldapFormData.user_search_base}
                        </p>
                      )}
                    </div>
                    <div className="form-group">
                      <label>User Search Filter</label>
                      <input
                        type="text"
                        value={ldapFormData.user_search_filter}
                        onChange={(e) => setLdapFormData({ ...ldapFormData, user_search_filter: e.target.value })}
                        placeholder="(uid={username})"
                      />
                      <p className="field-help">LDAP filter to find users. Use {'{username}'} as placeholder.</p>
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label>Admin Group DNs</label>
                      <CheckboxDropdown
                        options={ldapDiscoveredGroups.map((group) => ({
                          id: group.dn,
                          label: group.name,
                          description: group.dn,
                        }))}
                        selectedIds={ldapFormData.admin_group_dns}
                        onChange={(ids) => setLdapFormData({ ...ldapFormData, admin_group_dns: ids })}
                        placeholder="No admin groups selected"
                        searchPlaceholder="Search LDAP groups..."
                      />
                      <p className="field-help">Members of any selected group get admin privileges.</p>
                    </div>
                    <div className="form-group">
                      <label>User Group DNs (optional)</label>
                      <CheckboxDropdown
                        options={ldapDiscoveredGroups.map((group) => ({
                          id: group.dn,
                          label: group.name,
                          description: group.dn,
                        }))}
                        selectedIds={ldapFormData.user_group_dns}
                        onChange={(ids) => setLdapFormData({ ...ldapFormData, user_group_dns: ids })}
                        placeholder="Any LDAP user can log in"
                        searchPlaceholder="Search LDAP groups..."
                      />
                      <p className="field-help">If set, users must be members of at least one selected group to login.</p>
                    </div>
                  </div>
                </>
              )}

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
                    {ldapConfig.admin_group_dns.length > 0 && (
                      <span className="meta-pill">
                        <span className="meta-pill-label">Admin Groups</span>
                        <span className="meta-pill-value">{ldapConfig.admin_group_dns.length}</span>
                      </span>
                    )}
                    {ldapConfig.user_group_dns.length > 0 && (
                      <span className="meta-pill">
                        <span className="meta-pill-label">Logon Gates</span>
                        <span className="meta-pill-value">{ldapConfig.user_group_dns.length}</span>
                      </span>
                    )}
                  </div>
                </div>
              )}

              <details style={{ marginBottom: '16px' }} id="setting-authentication_advanced">
                <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Settings</summary>

                <h4 style={{ margin: '1rem 0 4px' }}>Sync &amp; Cache</h4>
                <p className="fieldset-help">Controls how LDAP identities are projected into the local database. LDAP passwords are never cached; only identity, groups, and role projection.</p>
                {authProviderConfig && (
                  <div className="form-row">
                    <div className="form-group">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={authProviderConfig.ldap_lazy_sync_enabled}
                          onChange={(e) => setAuthProviderConfig({ ...authProviderConfig, ldap_lazy_sync_enabled: e.target.checked })}
                          style={{ marginRight: '0.5rem' }}
                        />
                        <span>Lazy sync LDAP identities on login</span>
                      </label>
                      <p className="field-help">When a user authenticates via LDAP, cache their identity and groups locally for the TTL below.</p>
                    </div>
                    <div className="form-group">
                      <label>Cache TTL (minutes)</label>
                      <input
                        type="number"
                        min={1}
                        max={10080}
                        value={authProviderConfig.cache_ttl_minutes}
                        onChange={(e) => setAuthProviderConfig({ ...authProviderConfig, cache_ttl_minutes: parseInt(e.target.value, 10) || 1 })}
                      />
                    </div>
                  </div>
                )}

                <h4 style={{ margin: '1rem 0 4px' }}>Pre-import LDAP User</h4>
                <p className="fieldset-help">Optionally cache an LDAP identity into the local database before they sign in &mdash; useful for assigning role or group overrides ahead of time. The user still authenticates with their LDAP password.</p>
                <div className="form-group">
                  <label>Search LDAP User</label>
                  <div className="ldap-user-search-row">
                    <div className="ldap-user-search-typeahead" ref={ldapUserSearchContainerRef}>
                      <input
                        type="text"
                        value={ldapUserSearchName}
                        onChange={(e) => {
                          setLdapUserSearchName(e.target.value);
                          setLdapUserPreview(null);
                        }}
                        onKeyUp={() => {
                          if (suppressLdapUserSearchDropdown) {
                            setSuppressLdapUserSearchDropdown(false);
                          }
                        }}
                        onFocus={() => {
                          if (!suppressLdapUserSearchDropdown && (ldapUserSearchResults.length > 0 || ldapUserSearchName.trim().length >= 2)) {
                            setShowLdapUserSearchResults(true);
                          }
                        }}
                        placeholder={ldapConfigured ? 'Search by username or email' : 'Configure LDAP first'}
                        disabled={!ldapConfigured}
                      />
                      {ldapUserSearching && (
                        <div className="ldap-user-search-status">Searching...</div>
                      )}
                      {showLdapUserSearchResults && ldapUserSearchEnabled && (
                        <div className="ldap-user-search-dropdown" role="listbox" aria-label="LDAP user search results">
                          {ldapUserSearchResults.length > 0 ? ldapUserSearchResults.map((profile) => (
                            <button
                              key={`${profile.source_dn || profile.username}-${profile.email || ''}`}
                              type="button"
                              className="ldap-user-search-option"
                              onClick={() => handleSelectLdapUserSuggestion(profile)}
                            >
                              <span className="ldap-user-search-option-title">{profile.display_name || profile.username}</span>
                              <span className="ldap-user-search-option-meta">
                                <span className="ldap-user-search-option-username">{profile.username}</span>
                                {profile.email && <span className="ldap-user-search-option-email">{profile.email}</span>}
                              </span>
                            </button>
                          )) : (
                            <div className="ldap-user-search-empty">No matching LDAP users</div>
                          )}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={handleImportLdapUser}
                      disabled={ldapUserImporting || !ldapConfigured || !ldapUserSearchName.trim()}
                    >
                      {ldapUserImporting ? 'Syncing...' : 'Sync User'}
                    </button>
                  </div>
                </div>
                {ldapUserPreview && (
                  <div className="form-group">
                    <div className="meta-pills">
                      <span className="meta-pill"><span className="meta-pill-label">User</span><span className="meta-pill-value">{ldapUserPreview.display_name || ldapUserPreview.username}</span></span>
                      <span className="meta-pill"><span className="meta-pill-label">Role</span><span className="meta-pill-value">{ldapUserPreview.role}</span></span>
                      <span className="meta-pill"><span className="meta-pill-label">Groups</span><span className="meta-pill-value">{ldapUserPreview.groups.length}</span></span>
                    </div>
                  </div>
                )}
              </details>
            </>
          )}

          <div className="form-group" style={{ marginTop: '1rem' }}>
            <h4 style={{ margin: '0 0 4px' }}>Current Group Memberships</h4>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: 'var(--space-sm)' }}>
              <p className="fieldset-help auth-provider-help-tight" style={{ margin: 0 }}>Internal groups are managed here. LDAP groups are projected from synced LDAP user memberships.</p>
            </div>
            {authGroups.length > 0 && (
              <div className="meta-pills auth-provider-pills">
                {authGroups.slice(0, 8).map((group) => {
                  const isLdap = group.provider === 'ldap';
                  const isAdmin = group.role === 'admin';
                  const isLogon = Boolean(group.is_logon_group);
                  const memberPreviews = group.member_previews ?? [];
                  return (
                    <Popover
                      key={group.id}
                      trigger="hover"
                      position="right"
                      content={
                        <div className="auth-group-members-popover">
                          <div className="auth-group-members-popover-header">{group.display_name} members</div>
                          {memberPreviews.length === 0 ? (
                            <div className="auth-group-members-popover-status">No members in this group.</div>
                          ) : (
                            <ul className="auth-group-members-popover-list">
                              {memberPreviews.map((member) => {
                                const displayName = member.display_name || member.username;
                                return (
                                  <li key={`${group.id}-${member.username}`} className="auth-group-members-popover-item">
                                    <span className="auth-group-member-display-name">{displayName}</span>
                                    <span className="auth-group-member-handle">@{member.username}</span>
                                  </li>
                                );
                              })}
                            </ul>
                          )}
                        </div>
                      }
                    >
                      <span className={`meta-pill auth-provider-pill auth-provider-pill-${group.provider}${isAdmin ? ' auth-provider-pill-admin' : ''}${isLogon ? ' auth-provider-pill-logon' : ''}`}>
                        <span className="meta-pill-value">{group.display_name}</span>
                        <span className="auth-provider-pill-count">({group.member_count})</span>
                        {isLdap && <span className="auth-provider-pill-source-badge">LDAP</span>}
                        {isAdmin && <span className="auth-provider-pill-admin-badge">Admin</span>}
                        {isLogon && <span className="auth-provider-pill-logon-badge">Logon</span>}
                      </span>
                    </Popover>
                  );
                })}
              </div>
            )}
            {authGroups.length === 0 && (
              <p className="muted" style={{ margin: 0 }}>No groups created yet.</p>
            )}
          </div>

          {authProviderConfig && (
            <div className="form-group">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleSaveAuthProviderConfig}
                disabled={authProviderConfigSaving}
              >
                {authProviderConfigSaving ? 'Saving...' : 'Save Authentication Policy'}
              </button>
            </div>
          )}
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
                    : (formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'client_credentials'
                      ? ' MCP clients must authenticate with a client ID and client secret, either via HTTP Basic or the per-route token endpoint.'
                      : settings?.has_mcp_default_password
                        ? ' A password is configured - MCP clients should use this password as the Bearer token.'
                        : ' Set a password below to enable password-based authentication.'}
                </p>
              </div>

              {/* Auth method selection - always show when auth is enabled. LDAP-only OAuth2 is conditional. */}
              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && (
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
                        value="client_credentials"
                        checked={(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'client_credentials'}
                        onChange={() => setFormData({ ...formData, mcp_default_route_auth_method: 'client_credentials' })}
                      />
                      <span>Client Credentials</span>
                    </label>
                    {ldapConfigured && (
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
                    )}
                  </div>
                  <p className="field-help">
                    {(formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'oauth2'
                      ? 'MCP clients authenticate with LDAP credentials via POST /auth/oauth2/token to get a Bearer token.'
                      : (formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'client_credentials'
                        ? 'MCP clients authenticate with client_id/client_secret over HTTP Basic, or exchange them at the token endpoint for a short-lived Bearer token.'
                        : 'MCP clients use a static password as the Bearer token or MCP-Password header.'}
                  </p>
                </div>
              )}

              {/* LDAP Group restriction - only for OAuth2 auth method */}
              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) &&
                ldapConfigured &&
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

              {(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) &&
                (formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'password') === 'client_credentials' && (
                  <>
                    <div className="form-group" style={{ marginTop: '1rem' }}>
                      <label htmlFor="mcp-client-id">Client ID</label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div className="settings-inline-copy-wrap" style={{ flex: 1, maxWidth: '400px' }}>
                          <input
                            type="text"
                            id="mcp-client-id"
                            placeholder="cid-..."
                            value={formData.mcp_default_route_client_id ?? settings?.mcp_default_route_client_id ?? ''}
                            onChange={(e) => setFormData({ ...formData, mcp_default_route_client_id: e.target.value })}
                            style={{ width: '100%', fontFamily: 'var(--font-mono)' }}
                          />
                          <InlineCopyButton
                            copyText={formData.mcp_default_route_client_id ?? settings?.mcp_default_route_client_id ?? ''}
                            className="settings-inline-copy"
                            disabled={!(formData.mcp_default_route_client_id ?? settings?.mcp_default_route_client_id ?? '')}
                            title="Copy client ID"
                            ariaLabel="Copy client ID"
                            copiedTitle="Client ID copied"
                            copiedAriaLabel="Client ID copied"
                            feedbackMs={2000}
                            onCopySuccess={() => toast.success('Client ID copied')}
                            onCopyError={() => setMcpError('Unable to copy client-id. Please copy it manually.')}
                          />
                        </div>
                        <button
                          type="button"
                          className="btn btn-small btn-secondary"
                          onClick={() => setFormData({ ...formData, mcp_default_route_client_id: generateMcpClientId() })}
                        >
                          Generate Client ID
                        </button>
                      </div>
                      <p className="field-help">
                        Public identifier for MCP clients. Use this with the client secret for HTTP Basic auth or token exchange.
                      </p>
                    </div>

                    <div className="form-group" style={{ marginTop: '1rem' }}>
                      <label htmlFor="mcp-client-secret">Client Secret</label>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div className="settings-inline-copy-wrap" style={{ flex: 1, maxWidth: '400px' }}>
                          <input
                            type={showMcpPassword ? 'text' : 'password'}
                            id="mcp-client-secret"
                            placeholder={settings?.has_mcp_default_password ? '••••••••' : 'Enter client secret (min 8 characters)'}
                            value={formData.mcp_default_route_password ?? ''}
                            onChange={(e) =>
                              setFormData({ ...formData, mcp_default_route_password: e.target.value })
                            }
                            style={{ width: '100%', fontFamily: 'var(--font-mono)' }}
                          />
                          <InlineCopyButton
                            copyText={formData.mcp_default_route_password ?? ''}
                            className="settings-inline-copy"
                            disabled={!(formData.mcp_default_route_password ?? '')}
                            title="Copy client secret"
                            ariaLabel="Copy client secret"
                            copiedTitle="Client secret copied"
                            copiedAriaLabel="Client secret copied"
                            feedbackMs={2000}
                            onCopySuccess={() => toast.success('Client secret copied')}
                            onCopyError={() => setMcpError('Unable to copy secret. Please copy it manually.')}
                          />
                          <button
                            type="button"
                            className="settings-inline-copy settings-inline-copy-secondary"
                            onClick={() => setShowMcpPassword(!showMcpPassword)}
                            title={showMcpPassword ? 'Hide client secret' : 'Show client secret'}
                            aria-label={showMcpPassword ? 'Hide client secret' : 'Show client secret'}
                          >
                            {showMcpPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                          </button>
                        </div>
                        <button
                          type="button"
                          className="btn btn-small btn-secondary"
                          onClick={() => setFormData({ ...formData, mcp_default_route_password: generateMcpSecret() })}
                        >
                          Generate Password
                        </button>
                        {settings?.has_mcp_default_password && (
                          <button
                            type="button"
                            className="btn btn-small btn-secondary"
                            onClick={() => setFormData({ ...formData, mcp_default_route_password: '' })}
                            title="Clear client secret (submit empty to remove)"
                          >
                            Clear
                          </button>
                        )}
                      </div>
                      <p className="field-help">
                        {settings?.has_mcp_default_password
                          ? 'Client secret is set. Leave blank to keep the current secret, or enter a new one to rotate it. Clear and save to remove client credentials protection.'
                          : 'Set a client secret for MCP clients. Minimum 8 characters.'}
                      </p>
                      {window.location.protocol === 'http:' && (
                        <div className="field-warning" style={{ marginTop: '0.5rem', padding: '0.5rem', backgroundColor: 'rgba(255, 193, 7, 0.15)', borderLeft: '3px solid #ffc107', borderRadius: '4px', fontSize: '0.85em' }}>
                          <strong>Security:</strong> You are accessing over HTTP. Client credentials will be transmitted in plaintext.
                          Consider using HTTPS via a reverse proxy for production deployments.
                        </div>
                      )}
                      {mcpError && <p className="field-error">{mcpError}</p>}
                    </div>
                  </>
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
                      <div className="settings-inline-copy-wrap" style={{ flex: 1, maxWidth: '400px' }}>
                        <input
                          type={showMcpPassword ? 'text' : 'password'}
                          id="mcp-password"
                          placeholder={settings?.has_mcp_default_password ? '••••••••' : 'Enter password (min 8 characters)'}
                          value={formData.mcp_default_route_password ?? ''}
                          onChange={(e) =>
                            setFormData({ ...formData, mcp_default_route_password: e.target.value })
                          }
                          style={{ width: '100%' }}
                        />
                        <InlineCopyButton
                          copyText={formData.mcp_default_route_password ?? ''}
                          className="settings-inline-copy"
                          disabled={!(formData.mcp_default_route_password ?? '')}
                          title="Copy password"
                          ariaLabel="Copy password"
                          copiedTitle="Password copied"
                          copiedAriaLabel="Password copied"
                          feedbackMs={2000}
                          onCopySuccess={() => toast.success('Password copied')}
                          onCopyError={() => setMcpError('Unable to copy secret. Please copy it manually.')}
                        />
                        <button
                          type="button"
                          className="settings-inline-copy settings-inline-copy-secondary"
                          onClick={() => setShowMcpPassword(!showMcpPassword)}
                          title={showMcpPassword ? 'Hide password' : 'Show password'}
                          aria-label={showMcpPassword ? 'Hide password' : 'Show password'}
                        >
                          {showMcpPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                        </button>
                      </div>
                      <button
                        type="button"
                        className="btn btn-small btn-secondary"
                        onClick={() => setFormData({ ...formData, mcp_default_route_password: generateMcpSecret() })}
                        title="Generate password"
                      >
                        Generate Password
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



        <fieldset id="setting-userspace">
          <legend>User Space</legend>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '1rem',
              alignItems: 'start',
            }}
          >
            <div>
              <h4 style={{ margin: '0 0 8px' }}>Global Environment Variables</h4>
              <p className="fieldset-help" style={{ marginBottom: 12 }}>
                Define admin-managed environment variables that are inherited by every workspace.
              </p>
              <div className="form-group">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => { void handleOpenGlobalEnvVarsModal(); }}
                >
                  Manage Global Env Vars
                </button>
              </div>
            </div>

            <div>
              <h4 style={{ margin: '0 0 8px' }}>Preview Sandbox</h4>
              <p className="fieldset-help" style={{ marginBottom: 12 }}>
                Control which HTML iframe sandbox flags are granted to User Space previews.
              </p>
              <div className="form-group">
                <p className="field-help">
                  <strong>{effectiveUserSpacePreviewSandboxFlags.length}</strong> of{' '}
                  {(userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []).length} sandbox flags enabled.
                  Sandbox attribute: <code>{userspacePreviewSandboxAttribute || '(empty)'}</code>
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowSandboxModal(true)}
                >
                  Configure Sandbox Flags
                </button>
              </div>
            </div>
          </div>

          <details style={{ marginBottom: '16px' }} id="setting-userspace_advanced">
            <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Settings</summary>

            <div className="form-group">
              <label>Workspace Duplication Defaults</label>
              <p className="field-help" style={{ marginTop: 0 }}>
                Control what the hover-duplicate action copies into the new workspace when no per-duplicate override is provided.
              </p>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={formData.userspace_duplicate_copy_files_default ?? settings?.userspace_duplicate_copy_files_default ?? true}
                  onChange={(event) => setFormData({
                    ...formData,
                    userspace_duplicate_copy_files_default: event.target.checked,
                  })}
                />
                Copy workspace files by default
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={formData.userspace_duplicate_copy_metadata_default ?? settings?.userspace_duplicate_copy_metadata_default ?? true}
                  onChange={(event) => setFormData({
                    ...formData,
                    userspace_duplicate_copy_metadata_default: event.target.checked,
                  })}
                />
                Copy metadata by default (description, SQLite mode, tool selections, env vars when accessible)
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={formData.userspace_duplicate_copy_chats_default ?? settings?.userspace_duplicate_copy_chats_default ?? false}
                  onChange={(event) => setFormData({
                    ...formData,
                    userspace_duplicate_copy_chats_default: event.target.checked,
                  })}
                />
                Copy chats by default
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
                <input
                  type="checkbox"
                  checked={formData.userspace_duplicate_copy_mounts_default ?? settings?.userspace_duplicate_copy_mounts_default ?? false}
                  onChange={(event) => setFormData({
                    ...formData,
                    userspace_duplicate_copy_mounts_default: event.target.checked,
                  })}
                />
                Copy mounts by default
              </label>
              <p className="field-help">
                Disable metadata copy to create a copy with the source files but fresh workspace settings. Chat and mount defaults are off by default.
              </p>
            </div>

            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}>
                <label>Mount Auto-Sync Interval</label>
                <p className="field-help" style={{ marginTop: 0 }}>
                  Global default used by SSH, OneDrive, and Google Drive workspace mounts when no mount source or workspace mount override is set.
                </p>
                {(() => {
                  const currentVal = formData.userspace_mount_sync_interval_seconds
                    ?? settings?.userspace_mount_sync_interval_seconds
                    ?? MOUNT_SYNC_DEFAULT_SECONDS;

                  return (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          style={{ flex: 1 }}
                          value={mountSyncIntervalToSlider(currentVal)}
                          onChange={(e) => {
                            const slider = parseInt(e.target.value, 10);
                            setFormData({
                              ...formData,
                              userspace_mount_sync_interval_seconds: sliderToMountSyncInterval(slider),
                            });
                          }}
                        />
                        <span style={{ minWidth: '64px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                          {formatMountSyncInterval(currentVal)}
                        </span>
                      </div>
                      <p className="field-help">
                        Range: 1 second to 30 days. Local workspace file changes still wake eligible mounts immediately.
                      </p>
                    </>
                  );
                })()}
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label>Stale Branch Threshold</label>
                <p className="field-help" style={{ marginTop: 0 }}>
                  Branches that fall behind the active head by this many snapshots are hidden from the timeline.
                </p>
                {(() => {
                  const sliderMin = 10;
                  const sliderMax = 500;
                  const currentVal = formData.snapshot_stale_branch_threshold ?? 50;
                  const isAll = currentVal === 0;

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
                            if (isAll) return 100;
                            const scale = Math.log(sliderMax / sliderMin);
                            return Math.max(0, Math.min(99, Math.round((Math.log(currentVal / sliderMin) / scale) * 99)));
                          })()}
                          onChange={(e) => {
                            const slider = parseInt(e.target.value, 10);
                            let val: number;
                            if (slider >= 100) {
                              val = 0; // "All" sentinel
                            } else {
                              const scale = Math.log(sliderMax / sliderMin);
                              val = Math.max(sliderMin, Math.round(sliderMin * Math.exp((slider / 99) * scale)));
                            }
                            setFormData({ ...formData, snapshot_stale_branch_threshold: val });
                          }}
                        />
                        <span style={{ minWidth: '48px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                          {isAll ? 'All' : currentVal}
                        </span>
                      </div>
                      <p className="field-help">
                        Default: 50. Set to "All" to show every branch regardless of age.
                      </p>
                    </>
                  );
                })()}
              </div>

              <div className="form-group" style={{ flex: 1 }}>
                <label>SQLite Import Size Limit</label>
                <p className="field-help" style={{ marginTop: 0 }}>
                  Maximum SQL dump upload accepted by the User Space SQLite import wizard.
                </p>
                {(() => {
                  const currentVal = formData.userspace_sqlite_import_max_bytes
                    ?? settings?.userspace_sqlite_import_max_bytes
                    ?? SQLITE_IMPORT_DEFAULT_MAX_BYTES;

                  return (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          style={{ flex: 1 }}
                          value={sqliteImportBytesToSlider(currentVal)}
                          onChange={(e) => {
                            const slider = parseInt(e.target.value, 10);
                            setFormData({
                              ...formData,
                              userspace_sqlite_import_max_bytes: sliderToSqliteImportBytes(slider),
                            });
                          }}
                        />
                        <span style={{ minWidth: '84px', textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                          {formatBytes(currentVal)}
                        </span>
                      </div>
                      <p className="field-help">
                        Range: 100 MB to 100 GB. Large imports are memory and disk intensive; use higher caps only for trusted dumps on hosts with enough headroom.
                      </p>
                    </>
                  );
                })()}
              </div>
            </div>
          </details>

          <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0 }}>
            <button
              type="button"
              className="btn"
              onClick={handleSaveStaleBranchThreshold}
              disabled={staleBranchSaving}
            >
              {staleBranchSaving ? 'Saving...' : 'Save User Space Settings'}
            </button>
          </div>
        </fieldset>


      </form>

      <AuthAdminModalHost
        createUserOpen={showCreateLocalUserModal}
        manageGroupsOpen={showManageAuthGroupsModal}
        authGroups={authGroups}
        onAuthGroupsChange={setAuthGroups}
        onCloseCreateUser={() => setShowCreateLocalUserModal(false)}
        onCloseManageGroups={closeManageAuthGroupsModal}
        toast={toast}
      />

      {settings?.updated_at && (
        <p className="muted" style={{ marginTop: '1rem', fontSize: '0.85rem' }}>
          Last updated: {new Date(settings.updated_at).toLocaleString()}
        </p>
      )}


      {/* Chat Models Filter Modal */}
      <ModelFilterModal
        isOpen={showModelFilterModal}
        title="Allowed Chat Models"
        onClose={() => setShowModelFilterModal(false)}
        allModels={allAvailableModels}
        modelsLoading={modelsLoading}
        selectedModels={selectedModels}
        allowedHelpText="Select host-specific model rows to allow them in chat. Selecting every row stores the default all-models setting."
        toggleModel={toggleModel}
        selectAll={selectAllModels}
        deselectAll={deselectAllModels}
        onSaveAllowed={saveModelFilter}
      />

      {/* OpenAPI Models Filter Modal */}
      <ModelFilterModal
        isOpen={showOpenapiModelModal}
        title="Allowed OpenAPI Models"
        onClose={() => setShowOpenapiModelModal(false)}
        allModels={openapiAvailableModels}
        modelsLoading={openapiModelsLoading}
        selectedModels={selectedOpenapiModels}
        allowedHelpText="Select host-specific model rows to expose via the /v1/models endpoint for external clients."
        toggleModel={toggleOpenapiModel}
        selectAll={selectAllOpenapiModels}
        deselectAll={deselectAllOpenapiModels}
        onSaveAllowed={saveOpenapiModelFilter}
      />

      {/* User Space Preview Sandbox Modal */}
      {showSandboxModal && (
        <div className="modal-overlay" onClick={() => setShowSandboxModal(false)}>
          <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>User Space Preview Sandbox</h3>
              <button
                className="modal-close"
                onClick={() => setShowSandboxModal(false)}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              <p className="field-help" style={{ margin: '0 0 0.75rem 0' }}>
                Control which HTML iframe sandbox flags are granted to User Space previews. Broader navigation and origin-related flags reduce iframe isolation.
              </p>
              <div className="model-filter-actions">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setUserSpacePreviewSandboxFlags(userspacePreviewSettings?.userspace_preview_sandbox_default_flags ?? [])}
                >
                  Use Default
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setUserSpacePreviewSandboxFlags(getUserSpacePreviewSandboxFlagValues(userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []))}
                >
                  Allow All
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={() => setUserSpacePreviewSandboxFlags([])}
                >
                  Clear All
                </button>
                <span className="muted" style={{ marginLeft: 'auto' }}>
                  {effectiveUserSpacePreviewSandboxFlags.length} of{' '}
                  {(userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []).length} enabled
                </span>
              </div>
              <div className="model-filter-list">
                {(userspacePreviewSettings?.userspace_preview_sandbox_flag_options ?? []).map((option) => {
                  const checked = effectiveUserSpacePreviewSandboxFlags.includes(option.value);
                  return (
                    <label key={option.value} className="model-filter-item">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => handleToggleUserSpacePreviewSandboxFlag(option.value)}
                      />
                      <span className="model-filter-name">
                        <strong>{option.value}</strong>
                        <span style={{ display: 'block', fontWeight: 400, fontSize: '0.85em', color: 'var(--text-muted, #888)' }}>
                          {option.description}
                        </span>
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowSandboxModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn"
                onClick={async () => {
                  await handleSaveUserSpacePreviewSandbox();
                  setShowSandboxModal(false);
                }}
                disabled={userspaceSaving}
              >
                {userspaceSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}

      <UserSpaceEnvVarsModal
        isOpen={showGlobalEnvVarsModal}
        title="Global Environment Variables"
        onClose={() => setShowGlobalEnvVarsModal(false)}
        envVars={globalEnvVars}
        loading={globalEnvVarsLoading}
        saving={globalEnvVarsSaving}
        canManage
        addLabel="Add global variable"
        extraContent={isAdmin ? (
          <UserSpaceRuntimeRestartPanel
            enabled={isAdmin}
            isVisible={showGlobalEnvVarsModal}
            description="Global environment variable changes are only picked up by active runtime containers after they restart."
            notifySuccess={toast.success}
            notifyError={toast.error}
          />
        ) : undefined}
        onCreateEnvVar={handleCreateGlobalEnvVar}
        onUpdateEnvVar={handleUpdateGlobalEnvVar}
        onDeleteEnvVar={handleDeleteGlobalEnvVar}
      />

      {/* MCP Routes Panel Modal */}
      {showMcpRoutesPanel && (
        <div className="modal-overlay" onClick={() => setShowMcpRoutesPanel(false)}>
          <div className="modal-content modal-large" onClick={(e) => e.stopPropagation()}>
            <MCPRoutesPanel
              ldapConfigured={ldapConfigured}
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
