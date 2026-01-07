/**
 * API client for Ragtime Indexer
 */

import type { IndexJob, IndexInfo, CreateIndexRequest, AppSettings, UpdateSettingsRequest, OllamaTestRequest, OllamaTestResponse, LLMModelsRequest, LLMModelsResponse, EmbeddingModelsRequest, EmbeddingModelsResponse, ToolConfig, CreateToolConfigRequest, UpdateToolConfigRequest, ToolTestRequest, ToolTestResponse, PostgresDiscoverRequest, PostgresDiscoverResponse, MssqlDiscoverRequest, MssqlDiscoverResponse, PdmDiscoverRequest, PdmDiscoverResponse, SSHKeyPairResponse, HeartbeatResponse, Conversation, CreateConversationRequest, SendMessageRequest, ChatMessage, AvailableModelsResponse, LoginRequest, LoginResponse, AuthStatus, User, LdapConfig, LdapDiscoverRequest, LdapDiscoverResponse, LdapBindDnLookupRequest, LdapBindDnLookupResponse, AnalyzeIndexRequest, IndexAnalysisResult, CheckRepoVisibilityRequest, RepoVisibilityResponse, FetchBranchesRequest, FetchBranchesResponse } from '@/types';

const API_BASE = '/indexes';
const AUTH_BASE = '/auth';

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }

  /**
   * Check if this is an authentication error (401)
   */
  isAuthError(): boolean {
    return this.status === 401;
  }

  /**
   * Check if this is a permission error (403)
   */
  isPermissionError(): boolean {
    return this.status === 403;
  }
}

/**
 * Wrapper for fetch that includes credentials and handles common options
 */
async function apiFetch(url: string, options: RequestInit = {}): Promise<Response> {
  return fetch(url, {
    ...options,
    credentials: 'include',
  });
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(
      data.detail || `Request failed with status ${response.status}`,
      response.status,
      data.detail
    );
  }
  return response.json();
}

export const api = {
  // ===========================================================================
  // Authentication
  // ===========================================================================

  /**
   * Get authentication status
   */
  async getAuthStatus(): Promise<AuthStatus> {
    const response = await apiFetch(`${AUTH_BASE}/status`);
    return handleResponse<AuthStatus>(response);
  },

  /**
   * Login with username and password
   */
  async login(request: LoginRequest): Promise<LoginResponse> {
    const response = await apiFetch(`${AUTH_BASE}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
       // Include cookies
    });
    return handleResponse<LoginResponse>(response);
  },

  /**
   * Logout current session
   */
  async logout(): Promise<void> {
    await fetch(`${AUTH_BASE}/logout`, {
      method: 'POST',

    });
  },

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<User> {
    const response = await apiFetch(`${AUTH_BASE}/me`, {

    });
    return handleResponse<User>(response);
  },

  /**
   * Get LDAP configuration (admin only)
   */
  async getLdapConfig(): Promise<LdapConfig> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/config`, {

    });
    return handleResponse<LdapConfig>(response);
  },

  /**
   * Discover LDAP structure (admin only)
   */
  async discoverLdap(request: LdapDiscoverRequest): Promise<LdapDiscoverResponse> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),

    });
    return handleResponse<LdapDiscoverResponse>(response);
  },

  /**
   * Discover LDAP structure using stored credentials (admin only)
   */
  async discoverLdapWithStoredCredentials(): Promise<LdapDiscoverResponse> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/discover`, {
      method: 'GET',
    });
    return handleResponse<LdapDiscoverResponse>(response);
  },

  /**
   * Look up bind DN from username (admin only)
   */
  async lookupBindDn(request: LdapBindDnLookupRequest): Promise<LdapBindDnLookupResponse> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/lookup-bind-dn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LdapBindDnLookupResponse>(response);
  },

  /**
   * Update LDAP configuration (admin only)
   */
  async updateLdapConfig(config: Partial<LdapConfig> & { bind_password?: string }): Promise<LdapConfig> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),

    });
    return handleResponse<LdapConfig>(response);
  },

  /**
   * List all users (admin only)
   */
  async listUsers(): Promise<User[]> {
    const response = await apiFetch(`${AUTH_BASE}/users`, {

    });
    return handleResponse<User[]>(response);
  },

  // ===========================================================================
  // Indexes
  // ===========================================================================

  /**
   * List all available indexes
   */
  async listIndexes(): Promise<IndexInfo[]> {
    const response = await apiFetch(API_BASE, {  });
    return handleResponse<IndexInfo[]>(response);
  },

  /**
   * Analyze a git repository before indexing
   * Returns estimated file counts, chunk counts, size, and suggested exclusions
   */
  async analyzeRepository(request: AnalyzeIndexRequest): Promise<IndexAnalysisResult> {
    const response = await apiFetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<IndexAnalysisResult>(response);
  },

  /**
   * Analyze an uploaded archive before indexing
   * Returns estimated file counts, chunk counts, size, and suggested exclusions
   */
  async analyzeUpload(formData: FormData): Promise<IndexAnalysisResult> {
    const response = await apiFetch(`${API_BASE}/upload/analyze`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<IndexAnalysisResult>(response);
  },

  /**
   * Check if a Git repository is publicly accessible.
   * Used to determine whether a token is needed for re-indexing.
   */
  async checkRepoVisibility(request: CheckRepoVisibilityRequest): Promise<RepoVisibilityResponse> {
    const response = await apiFetch(`${API_BASE}/check-visibility`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<RepoVisibilityResponse>(response);
  },

  /**
   * Fetch branches from a Git repository.
   * Uses stored token from existing index if available.
   */
  async fetchBranches(request: FetchBranchesRequest): Promise<FetchBranchesResponse> {
    const response = await apiFetch(`${API_BASE}/fetch-branches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<FetchBranchesResponse>(response);
  },

  /**
   * List all indexing jobs
   */
  async listJobs(): Promise<IndexJob[]> {
    const response = await apiFetch(`${API_BASE}/jobs`, {  });
    return handleResponse<IndexJob[]>(response);
  },

  /**
   * Get a specific job by ID
   */
  async getJob(jobId: string): Promise<IndexJob> {
    const response = await apiFetch(`${API_BASE}/jobs/${jobId}`, {  });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Cancel a pending or processing job
   */
  async cancelJob(jobId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/jobs/${jobId}/cancel`, {
      method: 'POST',

    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Cancel failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Retry a failed or stuck job
   */
  async retryJob(jobId: string, gitToken?: string): Promise<IndexJob> {
    const response = await apiFetch(`${API_BASE}/jobs/${jobId}/retry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ git_token: gitToken }),
    });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Delete a job record
   */
  async deleteJob(jobId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/jobs/${jobId}`, {
      method: 'DELETE',

    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Delete failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Upload an archive and create an index
   */
  async uploadAndIndex(formData: FormData): Promise<IndexJob> {
    const response = await apiFetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,

    });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Create an index from a git repository
   */
  async indexFromGit(request: CreateIndexRequest): Promise<IndexJob> {
    const response = await apiFetch(`${API_BASE}/git`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Delete an index by name
   */
  async deleteIndex(name: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Delete failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Toggle an index's enabled status for RAG context
   */
  async toggleIndex(name: string, enabled: boolean): Promise<{ enabled: boolean }> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/toggle`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    return handleResponse<{ enabled: boolean }>(response);
  },

  /**
   * Update an index's description for AI context
   */
  async updateIndexDescription(name: string, description: string): Promise<{ description: string }> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/description`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    });
    return handleResponse<{ description: string }>(response);
  },

  /**
   * Update an index's search weight for result prioritization
   */
  async updateIndexWeight(name: string, weight: number): Promise<{ weight: number }> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/weight`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weight }),
    });
    return handleResponse<{ weight: number }>(response);
  },

  /**
   * Update an index's configuration for next re-index (git indexes only)
   */
  async updateIndexConfig(name: string, config: import('@/types').UpdateIndexConfigRequest): Promise<{ git_branch: string; config_snapshot: import('@/types').IndexConfigSnapshot }> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/config`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse<{ git_branch: string; config_snapshot: import('@/types').IndexConfigSnapshot }>(response);
  },

  /**
   * Re-index a git-based index by pulling latest changes
   */
  async reindexFromGit(name: string, gitToken?: string): Promise<IndexJob> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/reindex`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ git_token: gitToken }),
    });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Download a FAISS index as a zip file
   */
  async downloadIndex(name: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/download`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Download failed',
        response.status,
        data.detail
      );
    }

    // Get the filename from the Content-Disposition header
    const contentDisposition = response.headers.get('content-disposition') || '';
    const filenameMatch = contentDisposition.match(/filename=([^;]+)/);
    const filename = filenameMatch ? filenameMatch[1] : `${name}_index.zip`;

    // Get the blob and create a download link
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  },

  /**
   * Get application settings
   */
  async getSettings(): Promise<AppSettings> {
    const response = await apiFetch(`${API_BASE}/settings`);
    return handleResponse<AppSettings>(response);
  },

  /**
   * Update application settings
   */
  async updateSettings(settings: UpdateSettingsRequest): Promise<AppSettings> {
    const response = await apiFetch(`${API_BASE}/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    return handleResponse<AppSettings>(response);
  },

  /**
   * Test Ollama server connection and get available models
   */
  async testOllamaConnection(request: OllamaTestRequest): Promise<OllamaTestResponse> {
    const response = await apiFetch(`${API_BASE}/ollama/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<OllamaTestResponse>(response);
  },

  /**
   * Fetch available models from an LLM provider
   */
  async fetchLLMModels(request: LLMModelsRequest): Promise<LLMModelsResponse> {
    const response = await apiFetch(`${API_BASE}/llm/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LLMModelsResponse>(response);
  },

  /**
   * Fetch available embedding models from a provider (OpenAI only)
   */
  async fetchEmbeddingModels(request: EmbeddingModelsRequest): Promise<EmbeddingModelsResponse> {
    const response = await apiFetch(`${API_BASE}/embedding/models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<EmbeddingModelsResponse>(response);
  },

  // =========================================================================
  // Tool Configuration API
  // =========================================================================

  /**
   * List all tool configurations
   */
  async listToolConfigs(enabledOnly: boolean = false): Promise<ToolConfig[]> {
    const params = enabledOnly ? '?enabled_only=true' : '';
    const response = await apiFetch(`${API_BASE}/tools${params}`);
    return handleResponse<ToolConfig[]>(response);
  },

  /**
   * Create a new tool configuration
   */
  async createToolConfig(config: CreateToolConfigRequest): Promise<ToolConfig> {
    const response = await apiFetch(`${API_BASE}/tools`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse<ToolConfig>(response);
  },

  /**
   * Get a specific tool configuration
   */
  async getToolConfig(toolId: string): Promise<ToolConfig> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}`);
    return handleResponse<ToolConfig>(response);
  },

  /**
   * Update a tool configuration
   */
  async updateToolConfig(toolId: string, updates: UpdateToolConfigRequest): Promise<ToolConfig> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return handleResponse<ToolConfig>(response);
  },

  /**
   * Delete a tool configuration
   */
  async deleteToolConfig(toolId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Delete failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Toggle a tool's enabled status
   */
  async toggleToolConfig(toolId: string, enabled: boolean): Promise<{ enabled: boolean }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/toggle?enabled=${enabled}`, {
      method: 'POST',
    });
    return handleResponse<{ enabled: boolean }>(response);
  },

  /**
   * Test a tool connection (without saving)
   */
  async testToolConnection(request: ToolTestRequest): Promise<ToolTestResponse> {
    const response = await apiFetch(`${API_BASE}/tools/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ToolTestResponse>(response);
  },

  /**
   * Discover available databases on a PostgreSQL server
   */
  async discoverPostgresDatabases(request: PostgresDiscoverRequest): Promise<PostgresDiscoverResponse> {
    const response = await apiFetch(`${API_BASE}/tools/postgres/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<PostgresDiscoverResponse>(response);
  },

  /**
   * Discover available databases on an MSSQL server
   */
  async discoverMssqlDatabases(request: MssqlDiscoverRequest): Promise<MssqlDiscoverResponse> {
    const response = await apiFetch(`${API_BASE}/tools/mssql/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<MssqlDiscoverResponse>(response);
  },

  /**
   * Discover PDM schema metadata (file extensions and variable names)
   */
  async discoverPdmSchema(request: PdmDiscoverRequest): Promise<PdmDiscoverResponse> {
    const response = await apiFetch(`${API_BASE}/tools/pdm/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<PdmDiscoverResponse>(response);
  },

  /**
   * Test a saved tool's connection
   */
  async testSavedToolConnection(toolId: string): Promise<ToolTestResponse> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/test`, {
      method: 'POST',
    });
    return handleResponse<ToolTestResponse>(response);
  },

  /**
   * Get heartbeat status for all enabled tools
   */
  async getToolHeartbeats(): Promise<HeartbeatResponse> {
    const response = await apiFetch(`${API_BASE}/tools/heartbeat`);
    return handleResponse<HeartbeatResponse>(response);
  },

  /**
   * Generate a new SSH keypair
   */
  async generateSSHKeypair(comment?: string, passphrase?: string): Promise<SSHKeyPairResponse> {
    const params = new URLSearchParams();
    if (comment) params.set('comment', comment);
    if (passphrase) params.set('passphrase', passphrase);
    const queryString = params.toString();
    const url = queryString
      ? `${API_BASE}/tools/ssh/generate-keypair?${queryString}`
      : `${API_BASE}/tools/ssh/generate-keypair`;
    const response = await apiFetch(url, { method: 'POST' });
    return handleResponse<SSHKeyPairResponse>(response);
  },

  // =========================================================================
  // Filesystem Indexer API
  // =========================================================================

  /**
   * Get filesystem indexing jobs for a tool
   */
  async getFilesystemJobs(toolId: string): Promise<import('@/types').FilesystemIndexJob[]> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/jobs`);
    return handleResponse<import('@/types').FilesystemIndexJob[]>(response);
  },

  /**
   * Get filesystem index stats for a tool
   */
  async getFilesystemStats(toolId: string): Promise<import('@/types').FilesystemIndexStats> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/stats`);
    return handleResponse<import('@/types').FilesystemIndexStats>(response);
  },

  /**
   * Start filesystem analysis for a tool (returns job for polling)
   */
  async startFilesystemAnalysis(toolId: string): Promise<import('@/types').FilesystemAnalysisJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/analyze`, {
      method: 'POST',
    });
    return handleResponse<import('@/types').FilesystemAnalysisJob>(response);
  },

  /**
   * Get filesystem analysis job status and results
   */
  async getFilesystemAnalysisJob(toolId: string, jobId: string): Promise<import('@/types').FilesystemAnalysisJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/analyze/${jobId}`);
    return handleResponse<import('@/types').FilesystemAnalysisJob>(response);
  },

  /**
   * Trigger filesystem indexing for a tool
   */
  async triggerFilesystemIndex(toolId: string, fullReindex?: boolean): Promise<import('@/types').FilesystemIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_reindex: fullReindex ?? false }),
    });
    return handleResponse<import('@/types').FilesystemIndexJob>(response);
  },

  /**
   * Cancel an active filesystem indexing job
   */
  async cancelFilesystemJob(toolId: string, jobId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/jobs/${jobId}/cancel`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  /**
   * Retry a failed or cancelled filesystem indexing job
   */
  async retryFilesystemJob(toolId: string, jobId: string): Promise<import('@/types').FilesystemIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/jobs/${jobId}/retry`, {
      method: 'POST',
    });
    return handleResponse<import('@/types').FilesystemIndexJob>(response);
  },

  /**
   * Delete filesystem index for a tool
   */
  async deleteFilesystemIndex(toolId: string): Promise<{ success: boolean; deleted_count: number }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/filesystem/index`, {
      method: 'DELETE',
    });
    return handleResponse<{ success: boolean; deleted_count: number }>(response);
  },

  /**
   * Discover current volume mounts in the container
   */
  async discoverMounts(): Promise<import('@/types').MountDiscoveryResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/mounts`);
    return handleResponse<import('@/types').MountDiscoveryResponse>(response);
  },

  /**
   * Browse a directory in the container filesystem
   */
  async browseFilesystem(path: string): Promise<import('@/types').BrowseResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/browse?path=${encodeURIComponent(path)}`);
    return handleResponse<import('@/types').BrowseResponse>(response);
  },

  /**
   * Discover NFS exports from a server
   */
  async discoverNfsExports(host: string): Promise<import('@/types').NFSDiscoveryResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/nfs/discover?host=${encodeURIComponent(host)}`);
    return handleResponse<import('@/types').NFSDiscoveryResponse>(response);
  },

  /**
   * Browse an NFS export
   */
  async browseNfsExport(host: string, exportPath: string, path: string = ''): Promise<import('@/types').BrowseResponse> {
    const params = new URLSearchParams({ host, export_path: exportPath });
    if (path) params.set('path', path);
    const response = await apiFetch(`${API_BASE}/filesystem/nfs/browse?${params}`);
    return handleResponse<import('@/types').BrowseResponse>(response);
  },

  /**
   * Discover SMB shares from a server
   */
  async discoverSmbShares(host: string, user?: string, password?: string, domain?: string): Promise<import('@/types').SMBDiscoveryResponse> {
    const params = new URLSearchParams({ host });
    if (user) params.set('user', user);
    if (password) params.set('password', password);
    if (domain) params.set('domain', domain);
    const response = await apiFetch(`${API_BASE}/filesystem/smb/discover?${params}`);
    return handleResponse<import('@/types').SMBDiscoveryResponse>(response);
  },

  /**
   * Browse an SMB share
   */
  async browseSmbShare(host: string, share: string, path: string = '', user?: string, password?: string, domain?: string): Promise<import('@/types').BrowseResponse> {
    const params = new URLSearchParams({ host, share });
    if (path) params.set('path', path);
    if (user) params.set('user', user);
    if (password) params.set('password', password);
    if (domain) params.set('domain', domain);
    const response = await apiFetch(`${API_BASE}/filesystem/smb/browse?${params}`);
    return handleResponse<import('@/types').BrowseResponse>(response);
  },

  /**
   * Discover Docker networks and containers
   */
  async discoverDocker(): Promise<DockerDiscoveryResponse> {
    const response = await apiFetch(`${API_BASE}/docker/discover`);
    return handleResponse<DockerDiscoveryResponse>(response);
  },

  /**
   * Connect ragtime container to a Docker network
   */
  async connectToNetwork(networkName: string): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/docker/connect-network?network_name=${encodeURIComponent(networkName)}`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  /**
   * Disconnect ragtime container from a Docker network
   */
  async disconnectFromNetwork(networkName: string): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/docker/disconnect-network?network_name=${encodeURIComponent(networkName)}`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  // =========================================================================
  // Schema Indexer API (SQL Database Schema Indexing)
  // =========================================================================

  /**
   * Trigger schema indexing for a SQL database tool
   */
  async triggerSchemaIndex(toolId: string, forceReindex?: boolean): Promise<import('@/types').SchemaIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_reindex: forceReindex ?? false }),
    });
    return handleResponse<import('@/types').SchemaIndexJob>(response);
  },

  /**
   * Get schema index status for a tool
   */
  async getSchemaIndexStatus(toolId: string): Promise<import('@/types').SchemaIndexStatusResponse> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/status`);
    return handleResponse<import('@/types').SchemaIndexStatusResponse>(response);
  },

  /**
   * Get a specific schema index job
   */
  async getSchemaIndexJob(toolId: string, jobId: string): Promise<import('@/types').SchemaIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/job/${jobId}`);
    return handleResponse<import('@/types').SchemaIndexJob>(response);
  },

  /**
   * Cancel an active schema indexing job
   */
  async cancelSchemaIndexJob(toolId: string, jobId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/job/${jobId}/cancel`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  /**
   * Delete schema index for a tool
   */
  async deleteSchemaIndex(toolId: string): Promise<{ success: boolean; deleted_count: number }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/index`, {
      method: 'DELETE',
    });
    return handleResponse<{ success: boolean; deleted_count: number }>(response);
  },

  /**
   * Get schema index stats for a tool
   */
  async getSchemaIndexStats(toolId: string): Promise<import('@/types').SchemaIndexStats> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/stats`);
    return handleResponse<import('@/types').SchemaIndexStats>(response);
  },

  /**
   * List all schema indexing jobs across all tools
   */
  async listSchemaJobs(limit?: number): Promise<import('@/types').SchemaIndexJob[]> {
    const params = limit ? `?limit=${limit}` : '';
    const response = await apiFetch(`${API_BASE}/schema/jobs${params}`);
    return handleResponse<import('@/types').SchemaIndexJob[]>(response);
  },

  /**
   * Retry a failed/cancelled schema indexing job
   */
  async retrySchemaJob(toolId: string, jobId: string): Promise<import('@/types').SchemaIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/job/${jobId}/retry`, {
      method: 'POST',
    });
    return handleResponse<import('@/types').SchemaIndexJob>(response);
  },

  // =========================================================================
  // SolidWorks PDM Indexer API (PDM Database Document Indexing)
  // =========================================================================

  /**
   * Trigger PDM document indexing for a SolidWorks PDM tool
   */
  async triggerPdmIndex(toolId: string, fullReindex?: boolean): Promise<import('@/types').PdmIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_reindex: fullReindex ?? false }),
    });
    return handleResponse<import('@/types').PdmIndexJob>(response);
  },

  /**
   * Get PDM index status for a tool
   */
  async getPdmIndexStatus(toolId: string): Promise<import('@/types').PdmIndexStatusResponse> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/status`);
    return handleResponse<import('@/types').PdmIndexStatusResponse>(response);
  },

  /**
   * Get a specific PDM index job
   */
  async getPdmIndexJob(toolId: string, jobId: string): Promise<import('@/types').PdmIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/job/${jobId}`);
    return handleResponse<import('@/types').PdmIndexJob>(response);
  },

  /**
   * Cancel an active PDM indexing job
   */
  async cancelPdmIndexJob(toolId: string, jobId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/job/${jobId}/cancel`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  /**
   * Delete PDM index for a tool
   */
  async deletePdmIndex(toolId: string): Promise<{ success: boolean; deleted_count: number }> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/index`, {
      method: 'DELETE',
    });
    return handleResponse<{ success: boolean; deleted_count: number }>(response);
  },

  /**
   * Get PDM index stats for a tool
   */
  async getPdmIndexStats(toolId: string): Promise<import('@/types').PdmIndexStats> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/stats`);
    return handleResponse<import('@/types').PdmIndexStats>(response);
  },

  /**
   * List all PDM indexing jobs across all tools
   */
  async listPdmJobs(limit?: number): Promise<import('@/types').PdmIndexJob[]> {
    const params = limit ? `?limit=${limit}` : '';
    const response = await apiFetch(`${API_BASE}/pdm/jobs${params}`);
    return handleResponse<import('@/types').PdmIndexJob[]>(response);
  },

  /**
   * Retry a failed/cancelled PDM indexing job
   */
  async retryPdmJob(toolId: string, jobId: string): Promise<import('@/types').PdmIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/pdm/job/${jobId}/retry`, {
      method: 'POST',
    });
    return handleResponse<import('@/types').PdmIndexJob>(response);
  },

  // =========================================================================
  // Chat/Conversation API
  // =========================================================================

  /**
   * Get available models from all configured LLM providers (filtered by allowed_chat_models)
   */
  async getAvailableModels(): Promise<AvailableModelsResponse> {
    const response = await apiFetch(`${API_BASE}/chat/available-models`);
    return handleResponse<AvailableModelsResponse>(response);
  },

  /**
   * Get ALL models from all configured LLM providers (unfiltered, for settings UI)
   */
  async getAllModels(): Promise<AvailableModelsResponse> {
    const response = await apiFetch(`${API_BASE}/chat/all-models`);
    return handleResponse<AvailableModelsResponse>(response);
  },

  /**
   * List all conversations
   */
  async listConversations(): Promise<Conversation[]> {
    const response = await apiFetch(`${API_BASE}/conversations`);
    return handleResponse<Conversation[]>(response);
  },

  /**
   * Create a new conversation
   */
  async createConversation(request?: CreateConversationRequest): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request || {}),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Get a specific conversation by ID
   */
  async getConversation(conversationId: string): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}`);
    return handleResponse<Conversation>(response);
  },

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Delete failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Update conversation title
   */
  async updateConversationTitle(conversationId: string, title: string): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/title`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Update conversation model
   */
  async updateConversationModel(conversationId: string, model: string): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/model`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Send a message to a conversation (non-streaming)
   */
  async sendMessage(conversationId: string, request: SendMessageRequest): Promise<{ message: ChatMessage; conversation: Conversation }> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<{ message: ChatMessage; conversation: Conversation }>(response);
  },

  /**
   * Send a message with streaming response
   * Returns an async generator that yields structured stream events
   */
  async *sendMessageStream(conversationId: string, message: string, signal?: AbortSignal): AsyncGenerator<import('@/types').StreamEvent, void, unknown> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/messages/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || `Request failed with status ${response.status}`,
        response.status,
        data.detail
      );
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new ApiError('No response body', 500);
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') {
            return;
          }
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta;

            // Check for tool call events
            if (delta?.tool_call) {
              const tc = delta.tool_call;
              if (tc.type === 'start') {
                yield {
                  type: 'tool_start',
                  toolCall: {
                    tool: tc.tool,
                    input: tc.input
                  }
                };
              } else if (tc.type === 'end') {
                yield {
                  type: 'tool_end',
                  toolCall: {
                    tool: tc.tool,
                    output: tc.output
                  }
                };
              }
            }
            // Check for content tokens
            else if (delta?.content) {
              yield {
                type: 'content',
                content: delta.content
              };
            }
            // Check for error finish_reason
            else if (parsed.choices?.[0]?.finish_reason === 'error') {
              yield {
                type: 'error',
                error: delta?.content || 'Unknown error occurred'
              };
            }
          } catch {
            // Skip invalid JSON
          }
        }
      }
    }
  },

  /**
   * Clear all messages in a conversation
   */
  async clearConversation(conversationId: string): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/clear`, {
      method: 'POST',
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Truncate conversation messages to keep only the first N messages.
   * Used when editing/resending a message.
   */
  async truncateConversation(conversationId: string, keepCount: number): Promise<Conversation> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/truncate?keep_count=${keepCount}`, {
      method: 'POST',
    });
    return handleResponse<Conversation>(response);
  },

  // ===========================================================================
  // Background Task API
  // ===========================================================================

  /**
   * Send a message to be processed in the background.
   * Returns a task object that can be polled for status.
   */
  async sendMessageBackground(conversationId: string, message: string): Promise<import('@/types').ChatTask> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/messages/background`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    return handleResponse<import('@/types').ChatTask>(response);
  },

  /**
   * Get the active task for a conversation, if any.
   */
  async getConversationActiveTask(conversationId: string): Promise<import('@/types').ChatTask | null> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/task`);
    if (response.status === 204 || response.status === 404) {
      return null;
    }
    const data = await handleResponse<import('@/types').ChatTask | null>(response);
    return data;
  },

  /**
   * Get a chat task by ID.
   * Use this to poll for task status and streaming state.
   * @param taskId The task ID to fetch
   * @param sinceVersion Optional version to enable delta polling - if provided and
   *                     version hasn't changed, streaming_state will be null to reduce data transfer
   */
  async getChatTask(taskId: string, sinceVersion?: number): Promise<import('@/types').ChatTask> {
    const url = sinceVersion !== undefined
      ? `${API_BASE}/tasks/${taskId}?since_version=${sinceVersion}`
      : `${API_BASE}/tasks/${taskId}`;
    const response = await apiFetch(url);
    return handleResponse<import('@/types').ChatTask>(response);
  },

  /**
   * Cancel a running chat task.
   */
  async cancelChatTask(taskId: string): Promise<import('@/types').ChatTask> {
    const response = await apiFetch(`${API_BASE}/tasks/${taskId}/cancel`, {
      method: 'POST',
    });
    return handleResponse<import('@/types').ChatTask>(response);
  },
};

// Docker discovery types
export interface DockerNetwork {
  name: string;
  driver: string;
  scope: string;
  containers: string[];
}

export interface DockerContainer {
  name: string;
  image: string;
  status: string;
  networks: string[];
  has_odoo: boolean;
}

export interface DockerDiscoveryResponse {
  success: boolean;
  message: string;
  networks: DockerNetwork[];
  containers: DockerContainer[];
  current_network: string | null;
  current_container: string | null;
}

export { ApiError };
