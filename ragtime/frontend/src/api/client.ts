/**
 * API client for Ragtime Indexer
 */

import type { IndexJob, IndexInfo, CreateIndexRequest, AppSettings, UpdateSettingsRequest, OllamaTestRequest, OllamaTestResponse, LLMModelsRequest, LLMModelsResponse, ToolConfig, CreateToolConfigRequest, UpdateToolConfigRequest, ToolTestRequest, ToolTestResponse, PostgresDiscoverRequest, PostgresDiscoverResponse, SSHKeyPairResponse, HeartbeatResponse, Conversation, CreateConversationRequest, SendMessageRequest, ChatMessage, AvailableModelsResponse, LoginRequest, LoginResponse, AuthStatus, User, LdapConfig, LdapDiscoverRequest, LdapDiscoverResponse, LdapBindDnLookupRequest, LdapBindDnLookupResponse } from '@/types';

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
