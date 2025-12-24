/**
 * API client for Ragtime Indexer
 */

import type { IndexJob, IndexInfo, CreateIndexRequest, AppSettings, UpdateSettingsRequest, OllamaTestRequest, OllamaTestResponse, LLMModelsRequest, LLMModelsResponse, ToolConfig, CreateToolConfigRequest, UpdateToolConfigRequest, ToolTestRequest, ToolTestResponse, HeartbeatResponse } from '@/types';

const API_BASE = '/indexes';

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
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
  /**
   * List all available indexes
   */
  async listIndexes(): Promise<IndexInfo[]> {
    const response = await fetch(API_BASE);
    return handleResponse<IndexInfo[]>(response);
  },

  /**
   * List all indexing jobs
   */
  async listJobs(): Promise<IndexJob[]> {
    const response = await fetch(`${API_BASE}/jobs`);
    return handleResponse<IndexJob[]>(response);
  },

  /**
   * Get a specific job by ID
   */
  async getJob(jobId: string): Promise<IndexJob> {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`);
    return handleResponse<IndexJob>(response);
  },

  /**
   * Cancel a pending or processing job
   */
  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/jobs/${jobId}/cancel`, {
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
    const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
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
    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<IndexJob>(response);
  },

  /**
   * Create an index from a git repository
   */
  async indexFromGit(request: CreateIndexRequest): Promise<IndexJob> {
    const response = await fetch(`${API_BASE}/git`, {
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
    const response = await fetch(`${API_BASE}/${encodeURIComponent(name)}`, {
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
    const response = await fetch(`${API_BASE}/${encodeURIComponent(name)}/toggle`, {
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
    const response = await fetch(`${API_BASE}/${encodeURIComponent(name)}/description`, {
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
    const response = await fetch(`${API_BASE}/settings`);
    return handleResponse<AppSettings>(response);
  },

  /**
   * Update application settings
   */
  async updateSettings(settings: UpdateSettingsRequest): Promise<AppSettings> {
    const response = await fetch(`${API_BASE}/settings`, {
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
    const response = await fetch(`${API_BASE}/ollama/test`, {
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
    const response = await fetch(`${API_BASE}/llm/models`, {
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
    const response = await fetch(`${API_BASE}/tools${params}`);
    return handleResponse<ToolConfig[]>(response);
  },

  /**
   * Create a new tool configuration
   */
  async createToolConfig(config: CreateToolConfigRequest): Promise<ToolConfig> {
    const response = await fetch(`${API_BASE}/tools`, {
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
    const response = await fetch(`${API_BASE}/tools/${toolId}`);
    return handleResponse<ToolConfig>(response);
  },

  /**
   * Update a tool configuration
   */
  async updateToolConfig(toolId: string, updates: UpdateToolConfigRequest): Promise<ToolConfig> {
    const response = await fetch(`${API_BASE}/tools/${toolId}`, {
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
    const response = await fetch(`${API_BASE}/tools/${toolId}`, {
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
    const response = await fetch(`${API_BASE}/tools/${toolId}/toggle?enabled=${enabled}`, {
      method: 'POST',
    });
    return handleResponse<{ enabled: boolean }>(response);
  },

  /**
   * Test a tool connection (without saving)
   */
  async testToolConnection(request: ToolTestRequest): Promise<ToolTestResponse> {
    const response = await fetch(`${API_BASE}/tools/test`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ToolTestResponse>(response);
  },

  /**
   * Test a saved tool's connection
   */
  async testSavedToolConnection(toolId: string): Promise<ToolTestResponse> {
    const response = await fetch(`${API_BASE}/tools/${toolId}/test`, {
      method: 'POST',
    });
    return handleResponse<ToolTestResponse>(response);
  },

  /**
   * Get heartbeat status for all enabled tools
   */
  async getToolHeartbeats(): Promise<HeartbeatResponse> {
    const response = await fetch(`${API_BASE}/tools/heartbeat`);
    return handleResponse<HeartbeatResponse>(response);
  },

  /**
   * Discover Docker networks and containers
   */
  async discoverDocker(): Promise<DockerDiscoveryResponse> {
    const response = await fetch(`${API_BASE}/docker/discover`);
    return handleResponse<DockerDiscoveryResponse>(response);
  },

  /**
   * Connect ragtime container to a Docker network
   */
  async connectToNetwork(networkName: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE}/docker/connect-network?network_name=${encodeURIComponent(networkName)}`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  /**
   * Disconnect ragtime container from a Docker network
   */
  async disconnectFromNetwork(networkName: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${API_BASE}/docker/disconnect-network?network_name=${encodeURIComponent(networkName)}`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
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
