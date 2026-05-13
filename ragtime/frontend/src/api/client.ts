/**
 * API client for Ragtime Indexer
 */

import type { IndexJob, IndexInfo, CreateIndexRequest, AppSettings, GetSettingsResponse, UpdateSettingsRequest, OllamaTestRequest, OllamaTestResponse, OllamaVisionModelsRequest, OllamaVisionModelsResponse, VisionModelsRequest, VisionModelsResponse, LLMModelsRequest, LLMModelsResponse, EmbeddingModelsRequest, EmbeddingModelsResponse, LmStudioModelLoadRequest, LmStudioModelUnloadRequest, LmStudioModelActionResponse, ToolConfig, CreateToolConfigRequest, UpdateToolConfigRequest, ReorderToolsRequest, ToolTestRequest, ToolTestResponse, ToolGroup, CreateToolGroupRequest, UpdateToolGroupRequest, PostgresDiscoverRequest, PostgresDiscoverResponse, MssqlDiscoverRequest, MssqlDiscoverResponse, MysqlDiscoverRequest, MysqlDiscoverResponse, InfluxdbDiscoverRequest, InfluxdbDiscoverResponse, PdmDiscoverRequest, PdmDiscoverResponse, SSHKeyPairResponse, HeartbeatResponse, Conversation, ConversationSummary, ConversationCountResponse, CreateConversationRequest, SendMessageRequest, ChatMessage, AvailableModelsResponse, LoginRequest, LoginResponse, AuthStatus, User, LdapConfig, LdapDiscoverRequest, LdapDiscoverResponse, LdapBindDnLookupRequest, LdapBindDnLookupResponse, AnalyzeIndexRequest, IndexAnalysisResult, CheckRepoVisibilityRequest, RepoVisibilityResponse, FetchBranchesRequest, FetchBranchesResponse, McpRouteConfig, CreateMcpRouteRequest, UpdateMcpRouteRequest, McpRouteListResponse, HealthResponse, UserSpaceWorkspace, CreateUserSpaceWorkspaceRequest, DuplicateUserSpaceWorkspaceRequest, UpdateUserSpaceWorkspaceRequest, UpdateUserSpaceWorkspaceMembersRequest, WorkspaceAgentGrant, UpsertWorkspaceAgentGrantRequest, RevokeWorkspaceAgentGrantResponse, UserSpaceWorkspaceEnvVar, UpsertUserSpaceWorkspaceEnvVarRequest, DeleteUserSpaceWorkspaceEnvVarResponse, UpsertUserSpaceGlobalEnvVarRequest, DeleteUserSpaceGlobalEnvVarResponse, UserSpaceObjectStorageConfig, CreateUserSpaceObjectStorageBucketRequest, UpdateUserSpaceObjectStorageBucketRequest, DeleteUserSpaceObjectStorageBucketResponse, UserspaceMountSource, CreateUserspaceMountSourceRequest, UpdateUserspaceMountSourceRequest, BrowseUserspaceMountSourceRequest, BrowseCloudMountSourceRequest, CreateCloudMountSourceDirectoryRequest, DeleteUserspaceMountSourceResponse, CreateUserUserspaceMountSourceRequest, UpdateUserUserspaceMountSourceRequest, UserCloudOAuthAccount, CloudOAuthProviderStatus, CloudOAuthStartRequest, CloudOAuthStartResponse, CloudOAuthCallbackRequest, WorkspaceMount, MountableSource, BrowseWorkspaceMountSourceRequest, WorkspaceMountBrowseResponse, WorkspaceMountDirectoryEntry, CreateWorkspaceMountRequest, UpdateWorkspaceMountRequest, DeleteWorkspaceMountResponse, WorkspaceMountSyncPreviewRequest, WorkspaceMountSyncPreviewResponse, WorkspaceMountSyncRequest, WorkspaceMountSyncResponse, MountSourceAffectedWorkspacesResponse, UserSpaceFileInfo, UserSpaceFile, UpsertUserSpaceFileRequest, UserSpaceSnapshot, UserSpaceSnapshotDiffSummary, UserSpaceSnapshotFileDiff, UserSpaceSnapshotTimeline, CreateUserSpaceSnapshotRequest, UpdateUserSpaceSnapshotRequest, SwitchUserSpaceSnapshotBranchRequest, CreateUserSpaceSnapshotBranchRequest, PromoteUserSpaceSnapshotBranchRequest, RestoreUserSpaceSnapshotResponse, UserSpaceAvailableTool, PaginatedWorkspacesResponse, ExecuteComponentRequest, ExecuteComponentResponse, UserSpaceWorkspaceCreateTask, UserSpaceWorkspaceDeleteTask, UserSpaceWorkspaceDuplicateTask, UserSpaceWorkspaceArchiveExportRequest, UserSpaceWorkspaceArchiveExportTask, UserSpaceWorkspaceArchiveImportTask, UserSpaceWorkspaceArchiveExportListResponse, DeleteUserSpaceWorkspaceArchiveExportResponse, UserSpaceRuntimeRestartBatchTask, UserSpaceWorkspaceShareLink, UserSpaceWorkspaceShareLinkStatus, UserSpaceWorkspaceShareLinkListResponse, CreateWorkspaceShareLinkRequest, UpdateWorkspaceShareLinkRequest, WorkspaceShareSlugAvailabilityResponse, UpdateUserSpaceWorkspaceShareAccessRequest, ConversationMember, UpdateConversationMembersRequest, UpdateConversationToolsRequest, UserSpaceRuntimeSessionResponse, UserSpaceRuntimeStatusResponse, UserSpaceRuntimeActionResponse, UserSpaceCapabilityTokenResponse, UserSpaceBrowserAuthResponse, UserSpaceBrowserSurface, UserSpacePreviewLaunchRequest, UserSpacePreviewLaunchResponse, UserSpaceWorkspaceTabStateResponse, ProviderPromptDebugListResponse, ProviderPromptDebugRecord, CopilotAuthStatusResponse, CopilotDevicePollRequest, CopilotDevicePollResponse, CopilotDeviceStartRequest, CopilotDeviceStartResponse, LlmProviderWire, UserSpaceAcknowledgeChangedFilePathRequest, UserSpaceChangedFileState, UserSpacePreviewSettingsResponse, UserSpaceWorkspaceScmConnectionRequest, UserSpaceWorkspaceScmConnectionResponse, UserSpaceWorkspaceScmPreviewRequest, UserSpaceWorkspaceScmPreviewResponse, UserSpaceWorkspaceScmImportRequest, UserSpaceWorkspaceScmExportRequest, UserSpaceWorkspaceScmSyncResponse, UserSpaceWorkspaceScmSettingsRequest, UserSpaceWorkspaceSqliteImportTask, UserSpaceCollabPresenceResponse, ConversationShareLink, ConversationShareLinkStatus, ConversationShareLinkListResponse, CreateConversationShareLinkRequest, UpdateConversationShareLinkRequest, ConversationShareSlugAvailabilityResponse, UpdateConversationShareAccessRequest, SharedConversationResponse, PublicShareTargetResponse } from '@/types';

import type { AuthProviderConfig, UpdateAuthProviderConfigRequest, LocalUserCreateRequest, LocalUserUpdateRequest, AuthGroup, AuthGroupListResponse, AuthGroupUpsertRequest, SetUserGroupsRequest, LdapUserSearchRequest, LdapUserProfile, LdapUserImportResponse, LdapUserTypeaheadRequest, LdapUserTypeaheadResponse, UserDirectoryEntry } from '@/types';

const API_BASE = '/indexes';
const AUTH_BASE = '/auth';

type AuthExpiredListener = () => void;
const authExpiredListeners = new Set<AuthExpiredListener>();
let authExpiredNotified = false;

function notifyAuthExpired(): void {
  if (authExpiredNotified) {
    return;
  }
  authExpiredNotified = true;
  authExpiredListeners.forEach((listener) => {
    try {
      listener();
    } catch {
      // Ignore listener errors to ensure all callbacks run.
    }
  });
}

function resetAuthExpiredNotification(): void {
  authExpiredNotified = false;
}

export function onAuthExpired(listener: AuthExpiredListener): () => void {
  authExpiredListeners.add(listener);
  return () => {
    authExpiredListeners.delete(listener);
  };
}

function encodeFilePath(path: string): string {
  return path
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/');
}

function withWorkspaceQuery(url: string, workspaceId?: string): string {
  if (!workspaceId) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}workspace_id=${encodeURIComponent(workspaceId)}`;
}

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
  const headers = new Headers(options.headers ?? undefined);
  if (typeof window !== 'undefined' && window.location.origin) {
    headers.set('X-Ragtime-Browser-Origin', window.location.origin);
  }
  return fetch(url, {
    ...options,
    headers,
    credentials: 'include',
  });
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    if (response.status === 401) {
      notifyAuthExpired();
    }
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
  getConversationEventsUrl(conversationId: string, workspaceId?: string): string {
    return withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/events`, workspaceId);
  },

  // ===========================================================================
  // Authentication
  // ===========================================================================

  /**
   * Get authentication status
   */
  async getAuthStatus(): Promise<AuthStatus> {
    const response = await apiFetch(`${AUTH_BASE}/status`, { cache: 'no-store' });
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
    const result = await handleResponse<LoginResponse>(response);
    resetAuthExpiredNotification();
    return result;
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
    const user = await handleResponse<User>(response);
    resetAuthExpiredNotification();
    return user;
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

  async searchLdapUser(request: LdapUserSearchRequest): Promise<LdapUserProfile> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/search-user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LdapUserProfile>(response);
  },

  async searchLdapUsers(request: LdapUserTypeaheadRequest): Promise<LdapUserTypeaheadResponse> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/search-users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LdapUserTypeaheadResponse>(response);
  },

  async importLdapUser(request: LdapUserSearchRequest): Promise<LdapUserImportResponse> {
    const response = await apiFetch(`${AUTH_BASE}/ldap/import-user`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LdapUserImportResponse>(response);
  },

  async getAuthProviderConfig(): Promise<AuthProviderConfig> {
    const response = await apiFetch(`${AUTH_BASE}/provider/config`, {});
    return handleResponse<AuthProviderConfig>(response);
  },

  async updateAuthProviderConfig(request: UpdateAuthProviderConfigRequest): Promise<AuthProviderConfig> {
    const response = await apiFetch(`${AUTH_BASE}/provider/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<AuthProviderConfig>(response);
  },

  async createLocalUser(request: LocalUserCreateRequest): Promise<User> {
    const response = await apiFetch(`${AUTH_BASE}/local/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<User>(response);
  },

  async updateLocalUser(userId: string, request: LocalUserUpdateRequest): Promise<User> {
    const response = await apiFetch(`${AUTH_BASE}/local/users/${userId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<User>(response);
  },

  async listAuthGroups(): Promise<AuthGroup[]> {
    const response = await apiFetch(`${AUTH_BASE}/groups`, {});
    const data = await handleResponse<AuthGroupListResponse>(response);
    return data.groups;
  },

  async createAuthGroup(request: AuthGroupUpsertRequest): Promise<AuthGroup> {
    const response = await apiFetch(`${AUTH_BASE}/groups`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<AuthGroup>(response);
  },

  async updateAuthGroup(groupId: string, request: AuthGroupUpsertRequest): Promise<AuthGroup> {
    const response = await apiFetch(`${AUTH_BASE}/groups/${groupId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<AuthGroup>(response);
  },

  async deleteAuthGroup(groupId: string): Promise<void> {
    const response = await apiFetch(`${AUTH_BASE}/groups/${groupId}`, {
      method: 'DELETE',
    });
    await handleResponse(response);
  },

  async setUserGroups(userId: string, request: SetUserGroupsRequest): Promise<User> {
    const response = await apiFetch(`${AUTH_BASE}/users/${userId}/groups`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<User>(response);
  },

  /**
   * List all users
   */
  async listUsers(): Promise<User[]> {
    const response = await apiFetch(`${AUTH_BASE}/users`, {

    });
    const data = await handleResponse<{ users: User[] } | User[]>(response);
    return Array.isArray(data) ? data : (data as { users: User[] }).users ?? [];
  },

  /**
   * List all users for member-picker purposes (available to all authenticated users).
   */
  async listUsersDirectory(): Promise<UserDirectoryEntry[]> {
    const response = await apiFetch(`${AUTH_BASE}/users/directory`, {});
    const data = await handleResponse<{ users: UserDirectoryEntry[] } | UserDirectoryEntry[]>(response);
    return Array.isArray(data) ? data : (data as { users: UserDirectoryEntry[] }).users ?? [];
  },

  async getUsageSummary(days = 30): Promise<import('../types/api').UsageSummaryResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/summary?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageProviders(days = 30): Promise<import('../types/api').ProviderBreakdownResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/providers?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageDaily(days = 30): Promise<import('../types/api').DailyTrendResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/daily?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageUsersDaily(days = 30): Promise<import('../types/api').UserDailySeriesResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/users/daily?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageRange(): Promise<import('../types/api').UsageRangeResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/range`, {});
    return handleResponse(response);
  },

  async getUsageProviderDailyFailures(days = 30): Promise<import('../types/api').ProviderDailyFailuresResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/providers/daily-failures?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageApi(days = 30): Promise<import('../types/api').ApiUsageResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/api?days=${days}`, {});
    return handleResponse(response);
  },

  async getUsageMcp(days = 30): Promise<import('../types/api').McpUsageResponse> {
    const response = await apiFetch(`${AUTH_BASE}/usage/mcp?days=${days}`, {});
    return handleResponse(response);
  },

  async deleteUser(userId: string): Promise<void> {
    const response = await apiFetch(`${AUTH_BASE}/users/${userId}`, {
      method: 'DELETE',
    });
    await handleResponse(response);
  },

  async updateUserRole(userId: string, role: 'admin' | 'user'): Promise<void> {
    const response = await apiFetch(`${AUTH_BASE}/users/${userId}/role`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role }),
    });
    await handleResponse(response);
  },

  async resetUserRoleOverride(userId: string): Promise<{ success: boolean; role: 'admin' | 'user'; role_manually_set: boolean }> {
    const response = await apiFetch(`${AUTH_BASE}/users/${userId}/role/reset`, {
      method: 'POST',
    });
    return handleResponse(response);
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
   * Rename a git-based index
   */
  async renameIndex(name: string, newName: string): Promise<import('@/types').RenameIndexResponse> {
    const response = await apiFetch(`${API_BASE}/${encodeURIComponent(name)}/rename`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName }),
    });
    return handleResponse<import('@/types').RenameIndexResponse>(response);
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
   * Download a filesystem FAISS index as a zip file
   */
  async downloadFilesystemFaissIndex(indexName: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/filesystem/${encodeURIComponent(indexName)}/download`);
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
    const filename = filenameMatch ? filenameMatch[1] : `filesystem_${indexName}_index.zip`;

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

  // ===========================================================================
  // Health / System Status
  // ===========================================================================

  /**
   * Get health and memory status
   * Includes real-time memory usage and index loading progress
   */
  async getHealth(): Promise<HealthResponse> {
    const response = await apiFetch('/health');
    return handleResponse<HealthResponse>(response);
  },

  // ===========================================================================
  // Settings
  // ===========================================================================

  /**
   * Get application settings with configuration warnings
   */
  async getSettings(): Promise<GetSettingsResponse> {
    const response = await apiFetch(`${API_BASE}/settings`, { cache: 'no-store' });
    return handleResponse<GetSettingsResponse>(response);
  },

  /**
   * Get public preview sandbox settings for User Space iframes.
   */
  async getUserSpacePreviewSettings(): Promise<UserSpacePreviewSettingsResponse> {
    const response = await apiFetch(`${API_BASE}/settings/userspace-preview`);
    return handleResponse<UserSpacePreviewSettingsResponse>(response);
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
    // Backend returns {settings: AppSettings, embedding_warning: string | null}
    const result = await handleResponse<{ settings: AppSettings; embedding_warning: string | null }>(response);
    return result.settings;
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
   * Get vision-capable models from an Ollama server
   */
  async getOllamaVisionModels(request: OllamaVisionModelsRequest): Promise<OllamaVisionModelsResponse> {
    const response = await apiFetch(`${API_BASE}/ollama/vision-models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<OllamaVisionModelsResponse>(response);
  },

  /**
   * Get vision-capable OCR models from a provider
   */
  async getVisionModels(request: VisionModelsRequest): Promise<VisionModelsResponse> {
    const response = await apiFetch(`${API_BASE}/vision-models`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<VisionModelsResponse>(response);
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
   * Start GitHub Copilot OAuth device authorization flow.
   */
  async startCopilotDeviceFlow(request: CopilotDeviceStartRequest): Promise<CopilotDeviceStartResponse> {
    const response = await apiFetch(`${API_BASE}/github-copilot/device/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<CopilotDeviceStartResponse>(response);
  },

  /**
   * Poll GitHub Copilot OAuth device authorization status.
   */
  async pollCopilotDeviceFlow(request: CopilotDevicePollRequest): Promise<CopilotDevicePollResponse> {
    const response = await apiFetch(`${API_BASE}/github-copilot/device/poll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<CopilotDevicePollResponse>(response);
  },

  /**
   * Get current GitHub Copilot auth status.
   */
  async getCopilotAuthStatus(): Promise<CopilotAuthStatusResponse> {
    const response = await apiFetch(`${API_BASE}/github-copilot/auth/status`);
    return handleResponse<CopilotAuthStatusResponse>(response);
  },

  /**
   * Clear stored GitHub Copilot auth credentials.
   */
  async clearCopilotAuth(): Promise<{ success: boolean; message: string }> {
    const response = await apiFetch(`${API_BASE}/github-copilot/auth/clear`, {
      method: 'POST',
    });
    return handleResponse<{ success: boolean; message: string }>(response);
  },

  async loadLmstudioModel(request: LmStudioModelLoadRequest): Promise<LmStudioModelActionResponse> {
    const response = await apiFetch(`${API_BASE}/lmstudio/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LmStudioModelActionResponse>(response);
  },

  async unloadLmstudioModel(request: LmStudioModelUnloadRequest): Promise<LmStudioModelActionResponse> {
    const response = await apiFetch(`${API_BASE}/lmstudio/models/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<LmStudioModelActionResponse>(response);
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
   * Reorder tools by submitting their IDs in the desired display order
   */
  async reorderTools(request: ReorderToolsRequest): Promise<void> {
    const response = await apiFetch(`${API_BASE}/tools/reorder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    await handleResponse<{ message: string }>(response);
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

  // =========================================================================
  // Tool Group API
  // =========================================================================

  async listToolGroups(): Promise<ToolGroup[]> {
    const response = await apiFetch(`${API_BASE}/tool-groups`);
    return handleResponse<ToolGroup[]>(response);
  },

  async createToolGroup(request: CreateToolGroupRequest): Promise<ToolGroup> {
    const response = await apiFetch(`${API_BASE}/tool-groups`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ToolGroup>(response);
  },

  async updateToolGroup(groupId: string, request: UpdateToolGroupRequest): Promise<ToolGroup> {
    const response = await apiFetch(`${API_BASE}/tool-groups/${groupId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ToolGroup>(response);
  },

  async deleteToolGroup(groupId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/tool-groups/${groupId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(data.detail || 'Delete failed', response.status, data.detail);
    }
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
   * Discover available databases on a MySQL/MariaDB server
   */
  async discoverMysqlDatabases(request: MysqlDiscoverRequest): Promise<MysqlDiscoverResponse> {
    const response = await apiFetch(`${API_BASE}/tools/mysql/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<MysqlDiscoverResponse>(response);
  },

  /**
   * Discover available buckets on an InfluxDB server
   */
  async discoverInfluxdbBuckets(request: InfluxdbDiscoverRequest): Promise<InfluxdbDiscoverResponse> {
    const response = await apiFetch(`${API_BASE}/tools/influxdb/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<InfluxdbDiscoverResponse>(response);
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
  // MCP Routes API
  // =========================================================================

  /**
   * List all MCP route configurations
   */
  async listMcpRoutes(): Promise<McpRouteListResponse> {
    const response = await apiFetch('/mcp-routes');
    return handleResponse<McpRouteListResponse>(response);
  },

  /**
   * Get a specific MCP route configuration
   */
  async getMcpRoute(routeId: string): Promise<McpRouteConfig> {
    const response = await apiFetch(`/mcp-routes/${routeId}`);
    return handleResponse<McpRouteConfig>(response);
  },

  /**
   * Create a new MCP route configuration
   */
  async createMcpRoute(config: CreateMcpRouteRequest): Promise<McpRouteConfig> {
    const response = await apiFetch('/mcp-routes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse<McpRouteConfig>(response);
  },

  /**
   * Update an MCP route configuration
   */
  async updateMcpRoute(routeId: string, updates: UpdateMcpRouteRequest): Promise<McpRouteConfig> {
    const response = await apiFetch(`/mcp-routes/${routeId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return handleResponse<McpRouteConfig>(response);
  },

  /**
   * Delete an MCP route configuration
   */
  async deleteMcpRoute(routeId: string): Promise<void> {
    const response = await apiFetch(`/mcp-routes/${routeId}`, {
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
   * Toggle an MCP route's enabled status
   */
  async toggleMcpRoute(routeId: string, enabled: boolean): Promise<McpRouteConfig> {
    const response = await apiFetch(`/mcp-routes/${routeId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    return handleResponse<McpRouteConfig>(response);
  },

  // =========================================================================
  // MCP Default Route Filters API (LDAP group-based tool filtering)
  // =========================================================================

  /**
   * List all default route filters
   */
  async listMcpDefaultFilters(): Promise<import('@/types').McpDefaultRouteFilterListResponse> {
    const response = await apiFetch('/mcp-routes/default-filters');
    return handleResponse<import('@/types').McpDefaultRouteFilterListResponse>(response);
  },

  /**
   * Get a specific default route filter
   */
  async getMcpDefaultFilter(filterId: string): Promise<import('@/types').McpDefaultRouteFilter> {
    const response = await apiFetch(`/mcp-routes/default-filters/${filterId}`);
    return handleResponse<import('@/types').McpDefaultRouteFilter>(response);
  },

  /**
   * Create a new default route filter
   */
  async createMcpDefaultFilter(config: import('@/types').CreateMcpDefaultRouteFilterRequest): Promise<import('@/types').McpDefaultRouteFilter> {
    const response = await apiFetch('/mcp-routes/default-filters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return handleResponse<import('@/types').McpDefaultRouteFilter>(response);
  },

  /**
   * Update a default route filter
   */
  async updateMcpDefaultFilter(filterId: string, updates: import('@/types').UpdateMcpDefaultRouteFilterRequest): Promise<import('@/types').McpDefaultRouteFilter> {
    const response = await apiFetch(`/mcp-routes/default-filters/${filterId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return handleResponse<import('@/types').McpDefaultRouteFilter>(response);
  },

  /**
   * Delete a default route filter
   */
  async deleteMcpDefaultFilter(filterId: string): Promise<void> {
    const response = await apiFetch(`/mcp-routes/default-filters/${filterId}`, {
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
   * Toggle a default route filter's enabled status
   */
  async toggleMcpDefaultFilter(filterId: string, enabled: boolean): Promise<void> {
    const response = await apiFetch(`/mcp-routes/default-filters/${filterId}/toggle?enabled=${enabled}`, {
      method: 'POST',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Toggle failed',
        response.status,
        data.detail
      );
    }
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
   * Check container capabilities for mounting filesystems (SMB/NFS)
   * Returns whether the container has sufficient privileges for mount operations.
   */
  async checkContainerCapabilities(): Promise<import('@/types').ContainerCapabilitiesResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/capabilities`);
    return handleResponse<import('@/types').ContainerCapabilitiesResponse>(response);
  },

  /**
   * Browse a directory in the container filesystem
   */
  async browseFilesystem(path: string): Promise<import('@/types').BrowseResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/browse?path=${encodeURIComponent(path)}`);
    return handleResponse<import('@/types').BrowseResponse>(response);
  },

  /**
   * Browse a remote SSH filesystem
   */
  async browseSSHFilesystem(
    config: import('@/types').SSHShellConnectionConfig,
    path: string
  ): Promise<import('@/types').BrowseResponse> {
    const response = await apiFetch(`${API_BASE}/filesystem/ssh/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        host: config.host,
        user: config.user,
        port: config.port,
        password: config.password,
        key_path: config.key_path,
        key_content: config.key_content,
        key_passphrase: config.key_passphrase,
        path,
      }),
    });
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
  async triggerSchemaIndex(toolId: string, fullReindex?: boolean): Promise<import('@/types').SchemaIndexJob> {
    const response = await apiFetch(`${API_BASE}/tools/${toolId}/schema/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ full_reindex: fullReindex ?? false }),
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
   * List all conversations.
   *
   * Optional ``since`` / ``until`` are ISO-8601 timestamps that filter the
   * server response by ``updated_at``. ``since`` is inclusive, ``until`` is
   * exclusive — pass both to load only the "archive" window.
   */
  async listConversations(
    workspaceId?: string,
    options?: {
      since?: string | null;
      until?: string | null;
      limit?: number | null;
      cursorUpdatedAt?: string | null;
      cursorId?: string | null;
    },
  ): Promise<Conversation[]> {
    let url = withWorkspaceQuery(`${API_BASE}/conversations`, workspaceId);
    const extra: string[] = [];
    if (options?.since) extra.push(`since=${encodeURIComponent(options.since)}`);
    if (options?.until) extra.push(`until=${encodeURIComponent(options.until)}`);
    if (typeof options?.limit === 'number') extra.push(`limit=${encodeURIComponent(String(options.limit))}`);
    if (options?.cursorUpdatedAt) extra.push(`cursor_updated_at=${encodeURIComponent(options.cursorUpdatedAt)}`);
    if (options?.cursorId) extra.push(`cursor_id=${encodeURIComponent(options.cursorId)}`);
    if (extra.length) {
      url += url.includes('?') ? `&${extra.join('&')}` : `?${extra.join('&')}`;
    }
    const response = await apiFetch(url);
    return handleResponse<Conversation[]>(response);
  },

  async listConversationSummaries(
    workspaceId?: string,
    options?: { since?: string | null; until?: string | null },
  ): Promise<ConversationSummary[]> {
    let url = withWorkspaceQuery(`${API_BASE}/conversations/summaries`, workspaceId);
    const extra: string[] = [];
    if (options?.since) extra.push(`since=${encodeURIComponent(options.since)}`);
    if (options?.until) extra.push(`until=${encodeURIComponent(options.until)}`);
    if (extra.length) {
      url += url.includes('?') ? `&${extra.join('&')}` : `?${extra.join('&')}`;
    }
    const response = await apiFetch(url);
    return handleResponse<ConversationSummary[]>(response);
  },

  async countConversations(
    workspaceId?: string,
    options?: { since?: string | null; until?: string | null },
  ): Promise<ConversationCountResponse> {
    let url = withWorkspaceQuery(`${API_BASE}/conversations/count`, workspaceId);
    const extra: string[] = [];
    if (options?.since) extra.push(`since=${encodeURIComponent(options.since)}`);
    if (options?.until) extra.push(`until=${encodeURIComponent(options.until)}`);
    if (extra.length) {
      url += url.includes('?') ? `&${extra.join('&')}` : `?${extra.join('&')}`;
    }
    const response = await apiFetch(url);
    return handleResponse<ConversationCountResponse>(response);
  },

  /**
   * Get combined workspace conversation polling state.
   */
  async getWorkspaceConversationState(workspaceId: string): Promise<import('@/types').ConversationWorkspaceStateResponse> {
    const response = await apiFetch(`${API_BASE}/conversations/workspace/${workspaceId}/conversation-state`);
    return handleResponse<import('@/types').ConversationWorkspaceStateResponse>(response);
  },

  /**
   * Get live/interrupted summary for multiple workspaces in one request.
   */
  async getWorkspacesConversationStateSummary(workspaceIds: string[]): Promise<import('@/types').WorkspaceConversationStateSummaryItem[]> {
    const response = await apiFetch(`${API_BASE}/conversations/workspaces/state-summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_ids: workspaceIds }),
    });
    return handleResponse<import('@/types').WorkspaceConversationStateSummaryItem[]>(response);
  },

  /**
   * Get live/interrupted summary for multiple workspaces in one lightweight request.
   */
  async getWorkspacesConversationStateSummaryLite(workspaceIds: string[]): Promise<import('@/types').WorkspaceConversationStateSummaryItem[]> {
    const response = await apiFetch(`${API_BASE}/conversations/workspaces/state-summary-lite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workspace_ids: workspaceIds }),
    });
    return handleResponse<import('@/types').WorkspaceConversationStateSummaryItem[]>(response);
  },

  /**
   * Create a new conversation
   */
  async createConversation(request?: CreateConversationRequest, workspaceId?: string): Promise<Conversation> {
    const payload = { ...(request || {}) };
    if (workspaceId) {
      (payload as CreateConversationRequest).workspace_id = workspaceId;
    }
    const response = await apiFetch(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Get a specific conversation by ID
   */
  async getConversation(conversationId: string, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}`, workspaceId));
    return handleResponse<Conversation>(response);
  },

  /**
   * List DEBUG-mode provider prompt records for a conversation (admin only).
   */
  async getConversationProviderDebugPrompts(
    conversationId: string,
    messageIndex: number,
    workspaceId?: string,
    limit: number = 100,
  ): Promise<ProviderPromptDebugRecord[]> {
    const url = `${API_BASE}/conversations/${conversationId}/provider-debug-prompts?limit=${Math.max(1, Math.min(limit, 200))}&message_index=${messageIndex}`;
    const response = await apiFetch(
      withWorkspaceQuery(url, workspaceId)
    );
    const payload = await handleResponse<ProviderPromptDebugListResponse>(response);
    return payload.records || [];
  },

  /**
   * Delete a conversation
   */
  async deleteConversation(conversationId: string, workspaceId?: string): Promise<void> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}`, workspaceId), {
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
  async updateConversationTitle(conversationId: string, title: string, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/title`, workspaceId), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Update conversation model
   */
  async updateConversationModel(
    conversationId: string,
    model: string,
    workspaceId?: string,
    provider?: LlmProviderWire
  ): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/model`, workspaceId), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, provider }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Update a conversation's tool output mode
   */
  async updateConversationToolOutputMode(conversationId: string, toolOutputMode: import('@/types').ToolOutputMode, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/tool-output-mode`, workspaceId), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool_output_mode: toolOutputMode }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Send a message to a conversation (non-streaming)
   */
  async sendMessage(conversationId: string, request: SendMessageRequest, workspaceId?: string): Promise<{ message: ChatMessage; conversation: Conversation }> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/messages`, workspaceId), {
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
  async *sendMessageStream(conversationId: string, message: string, signal?: AbortSignal, workspaceId?: string): AsyncGenerator<import('@/types').StreamEvent, void, unknown> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/messages/stream`, workspaceId), {
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
                    input: tc.input,
                    connection: tc.connection,
                  }
                };
              } else if (tc.type === 'end') {
                yield {
                  type: 'tool_end',
                  toolCall: {
                    tool: tc.tool,
                    output: tc.output,
                    connection: tc.connection,
                  }
                };
              }
            }
            // Check for reasoning/thinking tokens
            else if (delta?.reasoning) {
              yield {
                type: 'reasoning',
                reasoning: delta.reasoning
              };
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
   * Truncate conversation messages to keep only the first N messages.
   * Used when editing/resending a message.
   */
  async truncateConversation(conversationId: string, keepCount: number, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/truncate?keep_count=${keepCount}`, workspaceId), {
      method: 'POST',
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Restore the workspace snapshot anchored to a chat message and rewind the
   * chat to the link's stored restore_message_count.
   */
  async restoreConversationMessageSnapshot(
    conversationId: string,
    messageId: string,
    workspaceId?: string,
  ): Promise<import('@/types').MessageSnapshotRestoreResponse> {
    const response = await apiFetch(
      withWorkspaceQuery(
        `${API_BASE}/conversations/${conversationId}/messages/${messageId}/restore-snapshot`,
        workspaceId,
      ),
      { method: 'POST' },
    );
    return handleResponse<import('@/types').MessageSnapshotRestoreResponse>(response);
  },

  /**
   * List branch points for a conversation (branches grouped by message index).
   */
  async getConversationBranchPoints(conversationId: string, workspaceId?: string): Promise<import('@/types').ConversationBranchPointInfo[]> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/branch-points`, workspaceId));
    return handleResponse<import('@/types').ConversationBranchPointInfo[]>(response);
  },

  /**
   * Create a conversation branch at a message edit point.
   */
  async createConversationBranch(conversationId: string, request: import('@/types').CreateConversationBranchRequest, workspaceId?: string): Promise<import('@/types').ConversationBranchSummary> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/branches`, workspaceId), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<import('@/types').ConversationBranchSummary>(response);
  },

  /**
   * Switch to a different conversation branch.
   */
  async switchConversationBranch(conversationId: string, branchId: string, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/branches/switch`, workspaceId), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ branch_id: branchId }),
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Release the active branch and return to the live path.
   * Stashes current downstream into the active branch so the user can
   * toggle between the restored state and the post-delete "Current" state.
   */
  async releaseConversationBranch(conversationId: string, workspaceId?: string): Promise<Conversation> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/branches/release`, workspaceId), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    return handleResponse<Conversation>(response);
  },

  /**
   * Delete a conversation branch.
   */
  async deleteConversationBranch(conversationId: string, branchId: string, workspaceId?: string): Promise<void> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/branches/${branchId}`, workspaceId), {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(data.detail || 'Delete failed', response.status, data.detail);
    }
  },

  /**
   * Get conversation members
   */
  async getConversationMembers(conversationId: string): Promise<ConversationMember[]> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/members`);
    return handleResponse<ConversationMember[]>(response);
  },

  /**
   * Update conversation members
   */
  async updateConversationMembers(conversationId: string, request: UpdateConversationMembersRequest): Promise<void> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/members`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Update failed',
        response.status,
        data.detail
      );
    }
  },

  /**
   * Get conversation tools
   */
  async getConversationTools(conversationId: string): Promise<{ tool_config_ids: string[]; tool_group_ids: string[]; disabled_builtin_tool_ids: string[] }> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/tools`);
    const data = await handleResponse<{ tool_config_ids: string[]; tool_group_ids?: string[]; disabled_builtin_tool_ids?: string[] }>(response);
    return {
      tool_config_ids: data.tool_config_ids || [],
      tool_group_ids: data.tool_group_ids || [],
      disabled_builtin_tool_ids: data.disabled_builtin_tool_ids || [],
    };
  },

  /**
   * Update conversation tools
   */
  async updateConversationTools(conversationId: string, request: UpdateConversationToolsRequest): Promise<void> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/tools`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Update failed',
        response.status,
        data.detail
      );
    }
  },

  // ===========================================================================
  // Background Task API
  // ===========================================================================

  /**
   * Send a message to be processed in the background.
   * Returns a task object that can be polled for status.
   */
  async sendMessageBackground(conversationId: string, message: string, workspaceId?: string): Promise<import('@/types').ChatTask> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/messages/background`, workspaceId), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    return handleResponse<import('@/types').ChatTask>(response);
  },

  async uploadConversationChatAttachment(
    conversationId: string,
    file: File,
    workspaceId?: string,
  ): Promise<import('@/types').ChatAttachmentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    const response = await apiFetch(
      withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/attachments`, workspaceId),
      {
        method: 'POST',
        body: formData,
      },
    );
    return handleResponse<import('@/types').ChatAttachmentUploadResponse>(response);
  },

  /**
   * Get the active task for a conversation, if any.
   */
  async getConversationActiveTask(conversationId: string, workspaceId?: string): Promise<import('@/types').ChatTask | null> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/task`, workspaceId));
    if (response.status === 204 || response.status === 404) {
      return null;
    }
    const data = await handleResponse<import('@/types').ChatTask | null>(response);
    return data;
  },

  /**
   * Get combined active/interrupted task state for a conversation.
   */
  async getConversationTaskState(conversationId: string, workspaceId?: string): Promise<import('@/types').ConversationTaskStateResponse> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/task-state`, workspaceId));
    return handleResponse<import('@/types').ConversationTaskStateResponse>(response);
  },

  async getWorkspaceChatState(workspaceId: string, selectedConversationId?: string | null): Promise<import('@/types').WorkspaceChatStateResponse> {
    const selectedQuery = selectedConversationId
      ? `?selected_conversation_id=${encodeURIComponent(selectedConversationId)}`
      : '';
    const response = await apiFetch(`${API_BASE}/conversations/workspace/${workspaceId}/chat-state${selectedQuery}`);
    return handleResponse<import('@/types').WorkspaceChatStateResponse>(response);
  },

  /**
   * Get the last interrupted task for a conversation, if any.
   * Used to show Continue button after server restart.
   */
  async getConversationInterruptedTask(conversationId: string, workspaceId?: string): Promise<import('@/types').ChatTask | null> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/conversations/${conversationId}/interrupted-task`, workspaceId));
    if (response.status === 204 || response.status === 404) {
      return null;
    }
    const data = await handleResponse<import('@/types').ChatTask | null>(response);
    return data;
  },

  /**
   * Get conversation IDs that have interrupted tasks in a workspace (batch).
   */
  async getWorkspaceInterruptedConversationIds(workspaceId: string): Promise<string[]> {
    const response = await apiFetch(`${API_BASE}/conversations/workspace/${workspaceId}/interrupted-conversation-ids`);
    return handleResponse<string[]>(response);
  },

  /**
   * Get a chat task by ID.
   * Use this to poll for task status and streaming state.
   * @param taskId The task ID to fetch
   * @param sinceVersion Optional version to enable delta polling - if provided and
   *                     version hasn't changed, streaming_state will be null to reduce data transfer
   */
  async getChatTask(taskId: string, sinceVersion?: number, workspaceId?: string): Promise<import('@/types').ChatTask> {
    const url = sinceVersion !== undefined
      ? `${API_BASE}/tasks/${taskId}?since_version=${sinceVersion}`
      : `${API_BASE}/tasks/${taskId}`;
    const response = await apiFetch(withWorkspaceQuery(url, workspaceId));
    return handleResponse<import('@/types').ChatTask>(response);
  },

  /**
   * Stream updates for a chat task via SSE.
   */
  async *streamChatTask(taskId: string, sinceVersion?: number, signal?: AbortSignal, workspaceId?: string): AsyncGenerator<any, void, unknown> {
    const url = sinceVersion !== undefined
      ? `${API_BASE}/tasks/${taskId}/stream?since_version=${sinceVersion}`
      : `${API_BASE}/tasks/${taskId}/stream`;

    const response = await apiFetch(withWorkspaceQuery(url, workspaceId), {
      method: 'GET',
      signal,
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || `Stream failed: ${response.status}`,
        response.status,
        data.detail
      );
    }

    const reader = response.body?.getReader();
    if (!reader) throw new ApiError('No response body', 500);

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim().startsWith('data: ')) {
          const data = line.trim().slice(6);
          try {
             yield JSON.parse(data);
          } catch { }
        }
      }
    }
  },

  /**
   * Cancel a running chat task.
   */
  async cancelChatTask(taskId: string, workspaceId?: string): Promise<import('@/types').ChatTask> {
    const response = await apiFetch(withWorkspaceQuery(`${API_BASE}/tasks/${taskId}/cancel`, workspaceId), {
      method: 'POST',
    });
    return handleResponse<import('@/types').ChatTask>(response);
  },

  // =========================================================================
  // User Space API
  // =========================================================================

  async listUserSpaceWorkspaces(offset = 0, limit = 50, includeAll = false): Promise<PaginatedWorkspacesResponse> {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (includeAll) params.set('include_all', 'true');
    const response = await apiFetch(`${API_BASE}/userspace/workspaces?${params}`);
    return handleResponse<PaginatedWorkspacesResponse>(response);
  },

  async listUserSpaceAvailableTools(): Promise<UserSpaceAvailableTool[]> {
    const response = await apiFetch(`${API_BASE}/userspace/tools`);
    return handleResponse<UserSpaceAvailableTool[]>(response);
  },

  async listUserSpaceToolGroups(): Promise<ToolGroup[]> {
    const response = await apiFetch(`${API_BASE}/userspace/tool-groups`);
    return handleResponse<ToolGroup[]>(response);
  },

  async createUserSpaceWorkspace(request: CreateUserSpaceWorkspaceRequest): Promise<UserSpaceWorkspace> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspace>(response);
  },

  async queueUserSpaceWorkspaceCreate(request: CreateUserSpaceWorkspaceRequest): Promise<UserSpaceWorkspaceCreateTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/create-task`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceCreateTask>(response);
  },

  async getUserSpaceWorkspaceCreateTask(taskId: string): Promise<UserSpaceWorkspaceCreateTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-create-tasks/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceWorkspaceCreateTask>(response);
  },

  async queueUserSpaceWorkspaceDuplicate(workspaceId: string, request: DuplicateUserSpaceWorkspaceRequest = {}): Promise<UserSpaceWorkspaceDuplicateTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${encodeURIComponent(workspaceId)}/duplicate-task`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceDuplicateTask>(response);
  },

  async getUserSpaceWorkspaceDuplicateTask(taskId: string): Promise<UserSpaceWorkspaceDuplicateTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-duplicate-tasks/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceWorkspaceDuplicateTask>(response);
  },

  async queueUserSpaceWorkspaceArchiveExport(
    workspaceId: string,
    request: UserSpaceWorkspaceArchiveExportRequest,
  ): Promise<UserSpaceWorkspaceArchiveExportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/archive/export-task`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceArchiveExportTask>(response);
  },

  async getUserSpaceWorkspaceArchiveExportTask(taskId: string): Promise<UserSpaceWorkspaceArchiveExportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-archive-export-tasks/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceWorkspaceArchiveExportTask>(response);
  },

  async downloadUserSpaceWorkspaceArchiveExportTask(taskId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-archive-export-tasks/${encodeURIComponent(taskId)}/download`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(
        data.detail || 'Archive download failed',
        response.status,
        data.detail,
      );
    }

    const contentDisposition = response.headers.get('content-disposition') || '';
    // Prefer RFC 5987 filename*=UTF-8''... when present; fall back to quoted or bare filename=.
    const extendedMatch = contentDisposition.match(/filename\*=(?:UTF-8'')?([^;]+)/i);
    const plainMatch = contentDisposition.match(/(?:^|;)\s*filename=("[^"]*"|[^;]+)/i);
    let filename = `workspace-export-${taskId}.zip`;
    if (extendedMatch?.[1]) {
      try {
        filename = decodeURIComponent(extendedMatch[1].trim());
      } catch {
        filename = extendedMatch[1].trim();
      }
    } else if (plainMatch?.[1]) {
      filename = plainMatch[1].trim().replace(/^"(.*)"$/, '$1');
    }
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

  async listUserSpaceWorkspaceArchiveExports(
    workspaceId: string,
  ): Promise<UserSpaceWorkspaceArchiveExportListResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${encodeURIComponent(workspaceId)}/archive-exports`);
    return handleResponse<UserSpaceWorkspaceArchiveExportListResponse>(response);
  },

  async deleteUserSpaceWorkspaceArchiveExportTask(
    taskId: string,
  ): Promise<DeleteUserSpaceWorkspaceArchiveExportResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-archive-export-tasks/${encodeURIComponent(taskId)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserSpaceWorkspaceArchiveExportResponse>(response);
  },

  async queueUserSpaceWorkspaceArchiveImport(
    workspaceId: string,
    formData: FormData,
  ): Promise<UserSpaceWorkspaceArchiveImportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/archive/import-task`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<UserSpaceWorkspaceArchiveImportTask>(response);
  },

  async getUserSpaceWorkspaceArchiveImportTask(taskId: string): Promise<UserSpaceWorkspaceArchiveImportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-archive-import-tasks/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceWorkspaceArchiveImportTask>(response);
  },

  async getUserSpaceWorkspace(workspaceId: string): Promise<UserSpaceWorkspace> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}`);
    return handleResponse<UserSpaceWorkspace>(response);
  },

  async getUserSpaceWorkspaceCollabPresence(workspaceId: string): Promise<UserSpaceCollabPresenceResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${encodeURIComponent(workspaceId)}/collab/presence`);
    return handleResponse<UserSpaceCollabPresenceResponse>(response);
  },

  async updateUserSpaceWorkspace(workspaceId: string, request: UpdateUserSpaceWorkspaceRequest): Promise<UserSpaceWorkspace> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspace>(response);
  },

  async queueUserSpaceWorkspaceDelete(workspaceId: string): Promise<UserSpaceWorkspaceDeleteTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/delete-task`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceWorkspaceDeleteTask>(response);
  },

  async getUserSpaceWorkspaceDeleteTask(taskId: string): Promise<UserSpaceWorkspaceDeleteTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-delete-tasks/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceWorkspaceDeleteTask>(response);
  },

  async getUserSpaceWorkspaceScm(workspaceId: string): Promise<UserSpaceWorkspaceScmConnectionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm`);
    return handleResponse<UserSpaceWorkspaceScmConnectionResponse>(response);
  },

  async updateUserSpaceWorkspaceScm(workspaceId: string, request: UserSpaceWorkspaceScmConnectionRequest): Promise<UserSpaceWorkspaceScmConnectionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmConnectionResponse>(response);
  },

  async checkUserSpaceWorkspaceScmRepoVisibility(workspaceId: string, request: CheckRepoVisibilityRequest): Promise<RepoVisibilityResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/check-visibility`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<RepoVisibilityResponse>(response);
  },

  async fetchUserSpaceWorkspaceScmBranches(workspaceId: string, request: FetchBranchesRequest): Promise<FetchBranchesResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/fetch-branches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<FetchBranchesResponse>(response);
  },

  async previewUserSpaceWorkspaceScmSync(workspaceId: string, request: UserSpaceWorkspaceScmPreviewRequest): Promise<UserSpaceWorkspaceScmPreviewResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/preview-sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmPreviewResponse>(response);
  },

  async previewUserSpaceWorkspaceScmImport(workspaceId: string, request: UserSpaceWorkspaceScmPreviewRequest): Promise<UserSpaceWorkspaceScmPreviewResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/preview-import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmPreviewResponse>(response);
  },

  async importUserSpaceWorkspaceFromScm(workspaceId: string, request: UserSpaceWorkspaceScmImportRequest): Promise<UserSpaceWorkspaceScmSyncResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmSyncResponse>(response);
  },

  async previewUserSpaceWorkspaceScmExport(workspaceId: string, request: UserSpaceWorkspaceScmPreviewRequest): Promise<UserSpaceWorkspaceScmPreviewResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/preview-export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmPreviewResponse>(response);
  },

  async exportUserSpaceWorkspaceToScm(workspaceId: string, request: UserSpaceWorkspaceScmExportRequest): Promise<UserSpaceWorkspaceScmSyncResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmSyncResponse>(response);
  },

  async updateUserSpaceWorkspaceScmSettings(workspaceId: string, request: UserSpaceWorkspaceScmSettingsRequest): Promise<UserSpaceWorkspaceScmConnectionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/scm/settings`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceScmConnectionResponse>(response);
  },

  async importSqlToWorkspaceSqlite(workspaceId: string, formData: FormData): Promise<UserSpaceWorkspaceSqliteImportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/sqlite-import`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<UserSpaceWorkspaceSqliteImportTask>(response);
  },

  async getUserSpaceWorkspaceSqliteImportTask(taskId: string): Promise<UserSpaceWorkspaceSqliteImportTask> {
    const response = await apiFetch(`${API_BASE}/userspace/workspace-sqlite-import-tasks/${taskId}`);
    return handleResponse<UserSpaceWorkspaceSqliteImportTask>(response);
  },

  async getLatestUserSpaceWorkspaceSqliteImportTask(workspaceId: string): Promise<UserSpaceWorkspaceSqliteImportTask | null> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/sqlite-import-task`);
    return handleResponse<UserSpaceWorkspaceSqliteImportTask | null>(response);
  },

  async updateUserSpaceWorkspaceMembers(workspaceId: string, request: UpdateUserSpaceWorkspaceMembersRequest): Promise<UserSpaceWorkspace> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/members`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspace>(response);
  },

  async listUserSpaceWorkspaceAgentGrants(workspaceId: string, direction: 'source' | 'target' = 'source'): Promise<WorkspaceAgentGrant[]> {
    const params = new URLSearchParams({ direction });
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/agent-grants?${params.toString()}`);
    return handleResponse<WorkspaceAgentGrant[]>(response);
  },

  async upsertUserSpaceWorkspaceAgentGrant(workspaceId: string, request: UpsertWorkspaceAgentGrantRequest): Promise<WorkspaceAgentGrant> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/agent-grants`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceAgentGrant>(response);
  },

  async revokeUserSpaceWorkspaceAgentGrant(workspaceId: string, targetWorkspaceId: string): Promise<RevokeWorkspaceAgentGrantResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/agent-grants/${encodeURIComponent(targetWorkspaceId)}`, {
      method: 'DELETE',
    });
    return handleResponse<RevokeWorkspaceAgentGrantResponse>(response);
  },

  async listUserSpaceWorkspaceEnvVars(workspaceId: string): Promise<UserSpaceWorkspaceEnvVar[]> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/env-vars`);
    return handleResponse<UserSpaceWorkspaceEnvVar[]>(response);
  },

  async upsertUserSpaceWorkspaceEnvVar(workspaceId: string, request: UpsertUserSpaceWorkspaceEnvVarRequest): Promise<UserSpaceWorkspaceEnvVar> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/env-vars`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceEnvVar>(response);
  },

  async deleteUserSpaceWorkspaceEnvVar(workspaceId: string, key: string): Promise<DeleteUserSpaceWorkspaceEnvVarResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/env-vars/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserSpaceWorkspaceEnvVarResponse>(response);
  },

  async listUserSpaceGlobalEnvVars(): Promise<UserSpaceWorkspaceEnvVar[]> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/env-vars`);
    return handleResponse<UserSpaceWorkspaceEnvVar[]>(response);
  },

  async upsertUserSpaceGlobalEnvVar(request: UpsertUserSpaceGlobalEnvVarRequest): Promise<UserSpaceWorkspaceEnvVar> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/env-vars`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceEnvVar>(response);
  },

  async deleteUserSpaceGlobalEnvVar(key: string): Promise<DeleteUserSpaceGlobalEnvVarResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/env-vars/${encodeURIComponent(key)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserSpaceGlobalEnvVarResponse>(response);
  },

  async getLatestUserSpaceRuntimeRestartTask(): Promise<UserSpaceRuntimeRestartBatchTask> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/runtime-restart-task`);
    return handleResponse<UserSpaceRuntimeRestartBatchTask>(response);
  },

  async queueUserSpaceRuntimeRestartTask(): Promise<UserSpaceRuntimeRestartBatchTask> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/runtime-restart-task`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceRuntimeRestartBatchTask>(response);
  },

  async getUserSpaceRuntimeRestartTask(taskId: string): Promise<UserSpaceRuntimeRestartBatchTask> {
    const response = await apiFetch(`${API_BASE}/userspace/admin/runtime-restart-task/${encodeURIComponent(taskId)}`);
    return handleResponse<UserSpaceRuntimeRestartBatchTask>(response);
  },

  async getUserSpaceObjectStorageConfig(workspaceId: string): Promise<UserSpaceObjectStorageConfig> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/object-storage`);
    return handleResponse<UserSpaceObjectStorageConfig>(response);
  },

  async createUserSpaceObjectStorageBucket(
    workspaceId: string,
    request: CreateUserSpaceObjectStorageBucketRequest,
  ): Promise<UserSpaceObjectStorageConfig> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/object-storage/buckets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceObjectStorageConfig>(response);
  },

  async updateUserSpaceObjectStorageBucket(
    workspaceId: string,
    bucketName: string,
    request: UpdateUserSpaceObjectStorageBucketRequest,
  ): Promise<UserSpaceObjectStorageConfig> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/object-storage/buckets/${encodeURIComponent(bucketName)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceObjectStorageConfig>(response);
  },

  async deleteUserSpaceObjectStorageBucket(
    workspaceId: string,
    bucketName: string,
  ): Promise<DeleteUserSpaceObjectStorageBucketResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/object-storage/buckets/${encodeURIComponent(bucketName)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserSpaceObjectStorageBucketResponse>(response);
  },

  async listUserspaceMountSources(): Promise<UserspaceMountSource[]> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources`);
    return handleResponse<UserspaceMountSource[]>(response);
  },

  async createUserspaceMountSource(request: CreateUserspaceMountSourceRequest): Promise<UserspaceMountSource> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserspaceMountSource>(response);
  },

  async updateUserspaceMountSource(
    mountSourceId: string,
    request: UpdateUserspaceMountSourceRequest,
  ): Promise<UserspaceMountSource> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources/${encodeURIComponent(mountSourceId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserspaceMountSource>(response);
  },

  async getMountSourceAffectedWorkspaces(mountSourceId: string): Promise<MountSourceAffectedWorkspacesResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources/${encodeURIComponent(mountSourceId)}/affected-workspaces`);
    return handleResponse<MountSourceAffectedWorkspacesResponse>(response);
  },

  async deleteUserspaceMountSource(mountSourceId: string): Promise<DeleteUserspaceMountSourceResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources/${encodeURIComponent(mountSourceId)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserspaceMountSourceResponse>(response);
  },

  async browseUserspaceMountSource(
    mountSourceId: string,
    request: BrowseUserspaceMountSourceRequest,
  ): Promise<WorkspaceMountBrowseResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources/${encodeURIComponent(mountSourceId)}/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountBrowseResponse>(response);
  },

  async browseToolConfig(
    toolConfigId: string,
    request: BrowseUserspaceMountSourceRequest,
  ): Promise<WorkspaceMountBrowseResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/tool-configs/${encodeURIComponent(toolConfigId)}/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountBrowseResponse>(response);
  },

  async createUserspaceMountSourceDirectory(
    mountSourceId: string,
    request: BrowseUserspaceMountSourceRequest,
  ): Promise<WorkspaceMountDirectoryEntry> {
    const response = await apiFetch(`${API_BASE}/userspace/mount-sources/${encodeURIComponent(mountSourceId)}/directory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountDirectoryEntry>(response);
  },

  async browseCloudMountSource(
    request: BrowseCloudMountSourceRequest,
  ): Promise<WorkspaceMountBrowseResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-mount-sources/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountBrowseResponse>(response);
  },

  async createCloudMountSourceDirectory(
    request: CreateCloudMountSourceDirectoryRequest,
  ): Promise<WorkspaceMountDirectoryEntry> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-mount-sources/directory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountDirectoryEntry>(response);
  },

  async listCloudOAuthProviders(): Promise<CloudOAuthProviderStatus[]> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-oauth/providers`);
    return handleResponse<CloudOAuthProviderStatus[]>(response);
  },

  async listUserCloudOAuthAccounts(): Promise<UserCloudOAuthAccount[]> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-oauth/accounts`);
    return handleResponse<UserCloudOAuthAccount[]>(response);
  },

  async startUserCloudOAuth(request: CloudOAuthStartRequest): Promise<CloudOAuthStartResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-oauth/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<CloudOAuthStartResponse>(response);
  },

  async completeUserCloudOAuth(request: CloudOAuthCallbackRequest): Promise<UserCloudOAuthAccount> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-oauth/callback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserCloudOAuthAccount>(response);
  },

  async disconnectUserCloudOAuth(accountId: string): Promise<DeleteUserspaceMountSourceResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/cloud-oauth/accounts/${encodeURIComponent(accountId)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserspaceMountSourceResponse>(response);
  },

  async listUserUserspaceMountSources(): Promise<UserspaceMountSource[]> {
    const response = await apiFetch(`${API_BASE}/userspace/user-mount-sources`);
    return handleResponse<UserspaceMountSource[]>(response);
  },

  async createUserUserspaceMountSource(request: CreateUserUserspaceMountSourceRequest): Promise<UserspaceMountSource> {
    const response = await apiFetch(`${API_BASE}/userspace/user-mount-sources`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserspaceMountSource>(response);
  },

  async updateUserUserspaceMountSource(
    mountSourceId: string,
    request: UpdateUserUserspaceMountSourceRequest,
  ): Promise<UserspaceMountSource> {
    const response = await apiFetch(`${API_BASE}/userspace/user-mount-sources/${encodeURIComponent(mountSourceId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserspaceMountSource>(response);
  },

  async deleteUserUserspaceMountSource(mountSourceId: string): Promise<DeleteUserspaceMountSourceResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/user-mount-sources/${encodeURIComponent(mountSourceId)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteUserspaceMountSourceResponse>(response);
  },

  async browseUserUserspaceMountSource(
    mountSourceId: string,
    request: BrowseUserspaceMountSourceRequest,
  ): Promise<WorkspaceMountBrowseResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/user-mount-sources/${encodeURIComponent(mountSourceId)}/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountBrowseResponse>(response);
  },

  // ── Workspace Mounts ────────────────────────────────────────────────

  async listWorkspaceMounts(workspaceId: string): Promise<WorkspaceMount[]> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts`);
    return handleResponse<WorkspaceMount[]>(response);
  },

  async listMountableSources(workspaceId: string): Promise<MountableSource[]> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mountable-sources`);
    return handleResponse<MountableSource[]>(response);
  },

  async browseWorkspaceMountSource(
    workspaceId: string,
    request: BrowseWorkspaceMountSourceRequest,
  ): Promise<WorkspaceMountBrowseResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts/browse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMountBrowseResponse>(response);
  },

  async createWorkspaceMount(workspaceId: string, request: CreateWorkspaceMountRequest): Promise<WorkspaceMount> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMount>(response);
  },

  async updateWorkspaceMount(
    workspaceId: string,
    mountId: string,
    request: UpdateWorkspaceMountRequest,
  ): Promise<WorkspaceMount> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts/${encodeURIComponent(mountId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<WorkspaceMount>(response);
  },

  async deleteWorkspaceMount(workspaceId: string, mountId: string): Promise<DeleteWorkspaceMountResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts/${encodeURIComponent(mountId)}`, {
      method: 'DELETE',
    });
    return handleResponse<DeleteWorkspaceMountResponse>(response);
  },

  async previewWorkspaceMountSync(
    workspaceId: string,
    mountId: string,
    request?: WorkspaceMountSyncPreviewRequest,
  ): Promise<WorkspaceMountSyncPreviewResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts/${encodeURIComponent(mountId)}/sync-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request ?? {}),
    });
    return handleResponse<WorkspaceMountSyncPreviewResponse>(response);
  },

  async syncWorkspaceMount(
    workspaceId: string,
    mountId: string,
    request?: WorkspaceMountSyncRequest,
  ): Promise<WorkspaceMountSyncResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/mounts/${encodeURIComponent(mountId)}/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request ?? {}),
    });
    return handleResponse<WorkspaceMountSyncResponse>(response);
  },

  async listUserSpaceFiles(
    workspaceId: string,
    options?: { includeDirs?: boolean }
  ): Promise<UserSpaceFileInfo[]> {
    const params = new URLSearchParams();
    if (options?.includeDirs) {
      params.set('include_dirs', 'true');
    }
    const query = params.toString();
    const querySuffix = query ? `?${query}` : '';
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/files${querySuffix}`);
    const data = await handleResponse<UserSpaceFileInfo[]>(response);
    return data;
  },

  async getUserSpaceFile(workspaceId: string, filePath: string): Promise<UserSpaceFile> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/files/${encodeFilePath(filePath)}`);
    return handleResponse<UserSpaceFile>(response);
  },

  async upsertUserSpaceFile(workspaceId: string, filePath: string, request: UpsertUserSpaceFileRequest): Promise<UserSpaceFile> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/files/${encodeFilePath(filePath)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceFile>(response);
  },

  async deleteUserSpaceFile(workspaceId: string, filePath: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/files/${encodeFilePath(filePath)}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new ApiError(data.detail || 'Delete failed', response.status, data.detail);
    }
  },

  async getUserSpaceChangedFileState(workspaceId: string): Promise<UserSpaceChangedFileState> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/changed-file-state`);
    return handleResponse<UserSpaceChangedFileState>(response);
  },

  async acknowledgeUserSpaceChangedFilePath(workspaceId: string, request: UserSpaceAcknowledgeChangedFilePathRequest): Promise<UserSpaceChangedFileState> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/changed-file-acknowledgements`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceChangedFileState>(response);
  },

  async clearUserSpaceChangedFileAcknowledgement(workspaceId: string, filePath: string): Promise<UserSpaceChangedFileState> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/changed-file-acknowledgements/${encodeFilePath(filePath)}`, {
      method: 'DELETE',
    });
    return handleResponse<UserSpaceChangedFileState>(response);
  },

  async clearUserSpaceChangedFileAcknowledgements(workspaceId: string): Promise<UserSpaceChangedFileState> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/changed-file-acknowledgements`, {
      method: 'DELETE',
    });
    return handleResponse<UserSpaceChangedFileState>(response);
  },

  async listUserSpaceSnapshots(workspaceId: string): Promise<UserSpaceSnapshot[]> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots`);
    return handleResponse<UserSpaceSnapshot[]>(response);
  },

  async createUserSpaceSnapshot(workspaceId: string, request: CreateUserSpaceSnapshotRequest): Promise<UserSpaceSnapshot> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceSnapshot>(response);
  },

  async getUserSpaceSnapshotTimeline(workspaceId: string): Promise<UserSpaceSnapshotTimeline> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/timeline`);
    return handleResponse<UserSpaceSnapshotTimeline>(response);
  },

  async getUserSpaceSnapshotDiffSummary(workspaceId: string, snapshotId: string): Promise<UserSpaceSnapshotDiffSummary> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/${snapshotId}/diff-summary`);
    return handleResponse<UserSpaceSnapshotDiffSummary>(response);
  },

  async getUserSpaceSnapshotFileDiff(workspaceId: string, snapshotId: string, filePath: string): Promise<UserSpaceSnapshotFileDiff> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/${snapshotId}/file-diff?file_path=${encodeURIComponent(filePath)}`);
    return handleResponse<UserSpaceSnapshotFileDiff>(response);
  },

  async updateUserSpaceSnapshot(workspaceId: string, snapshotId: string, request: UpdateUserSpaceSnapshotRequest): Promise<UserSpaceSnapshot> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/${snapshotId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceSnapshot>(response);
  },

  async restoreUserSpaceSnapshot(workspaceId: string, snapshotId: string): Promise<RestoreUserSpaceSnapshotResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/${snapshotId}/restore`, {
      method: 'POST',
    });
    return handleResponse<RestoreUserSpaceSnapshotResponse>(response);
  },

  async restorePreviousUserSpaceSnapshot(workspaceId: string): Promise<RestoreUserSpaceSnapshotResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/previous`, {
      method: 'POST',
    });
    return handleResponse<RestoreUserSpaceSnapshotResponse>(response);
  },

  async restoreNextUserSpaceSnapshot(workspaceId: string): Promise<RestoreUserSpaceSnapshotResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/next`, {
      method: 'POST',
    });
    return handleResponse<RestoreUserSpaceSnapshotResponse>(response);
  },

  async switchUserSpaceSnapshotBranch(workspaceId: string, request: SwitchUserSpaceSnapshotBranchRequest): Promise<UserSpaceSnapshotTimeline> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/switch-branch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceSnapshotTimeline>(response);
  },

  async createUserSpaceSnapshotBranch(workspaceId: string, request: CreateUserSpaceSnapshotBranchRequest): Promise<UserSpaceSnapshotTimeline> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/create-branch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceSnapshotTimeline>(response);
  },

  async promoteBranchToMain(workspaceId: string, request: PromoteUserSpaceSnapshotBranchRequest): Promise<UserSpaceSnapshotTimeline> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/promote-branch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceSnapshotTimeline>(response);
  },

  async deleteUserSpaceSnapshot(workspaceId: string, snapshotId: string): Promise<UserSpaceSnapshotTimeline> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/snapshots/${snapshotId}`, {
      method: 'DELETE',
    });
    return handleResponse<UserSpaceSnapshotTimeline>(response);
  },

  async executeWorkspaceComponent(workspaceId: string, request: ExecuteComponentRequest): Promise<ExecuteComponentResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/execute-component`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ExecuteComponentResponse>(response);
  },

  async execWorkspaceCommand(workspaceId: string, command: string, timeoutSeconds = 30, cwd?: string): Promise<Record<string, unknown>> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/exec`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, timeout_seconds: timeoutSeconds, cwd }),
    });
    return handleResponse<Record<string, unknown>>(response);
  },

  async retryTerminalToolCall(
    conversationId: string,
    request: import('@/types').RetryTerminalToolRequest,
    workspaceId?: string,
  ): Promise<import('@/types').RetryTerminalToolResponse> {
    const suffix = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : '';
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/retry-terminal-tool${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<import('@/types').RetryTerminalToolResponse>(response);
  },

  async retryVisualization(
    conversationId: string,
    request: import('@/types').RetryVisualizationRequest,
    workspaceId?: string,
  ): Promise<import('@/types').RetryVisualizationResponse> {
    const suffix = workspaceId ? `?workspace_id=${encodeURIComponent(workspaceId)}` : '';
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/retry-visualization${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<import('@/types').RetryVisualizationResponse>(response);
  },

  async listUserSpaceWorkspaceShareLinks(workspaceId: string): Promise<UserSpaceWorkspaceShareLinkListResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links`);
    return handleResponse<UserSpaceWorkspaceShareLinkListResponse>(response);
  },

  async revokeUserSpaceWorkspaceShareLink(workspaceId: string): Promise<UserSpaceWorkspaceShareLinkStatus> {
    const list = await this.listUserSpaceWorkspaceShareLinks(workspaceId);
    for (const link of list.links) {
      if (link.id) {
        await this.deleteUserSpaceWorkspaceShareLink(workspaceId, link.id);
      }
    }
    return this.getUserSpaceWorkspaceShareLinkStatus(workspaceId);
  },

  async createUserSpaceWorkspaceShareLink(
    workspaceId: string,
    requestOrRotate: CreateWorkspaceShareLinkRequest | boolean = {},
  ): Promise<UserSpaceWorkspaceShareLink> {
    const request: CreateWorkspaceShareLinkRequest =
      typeof requestOrRotate === 'boolean' ? {} : requestOrRotate;
    if (typeof requestOrRotate === 'boolean' && requestOrRotate) {
      // Legacy "rotateToken=true" callers expected revoke + new link.
      await this.revokeUserSpaceWorkspaceShareLink(workspaceId);
    }
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceShareLink>(response);
  },

  async getUserSpaceWorkspaceShareLinkStatus(workspaceId: string): Promise<UserSpaceWorkspaceShareLinkStatus> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-link`);
    return handleResponse<UserSpaceWorkspaceShareLinkStatus>(response);
  },

  async deleteUserSpaceWorkspaceShareLink(workspaceId: string, shareId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links/${shareId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      await handleResponse<unknown>(response);
    }
  },

  async updateUserSpaceWorkspaceShareLinkLabel(
    workspaceId: string,
    shareId: string,
    request: UpdateWorkspaceShareLinkRequest,
  ): Promise<UserSpaceWorkspaceShareLinkStatus> {
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links/${shareId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpaceWorkspaceShareLinkStatus>(response);
  },

  async updateUserSpaceWorkspaceShareSlug(
    workspaceId: string,
    shareIdOrSlug: string,
    slug?: string,
  ): Promise<UserSpaceWorkspaceShareLinkStatus> {
    let resolvedShareId: string;
    let resolvedSlug: string;
    if (slug === undefined) {
      const status = await this.getUserSpaceWorkspaceShareLinkStatus(workspaceId);
      if (!status.id) {
        throw new Error('No share link available to update');
      }
      resolvedShareId = status.id;
      resolvedSlug = shareIdOrSlug;
    } else {
      resolvedShareId = shareIdOrSlug;
      resolvedSlug = slug;
    }
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links/${resolvedShareId}/slug`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug: resolvedSlug }),
    });
    return handleResponse<UserSpaceWorkspaceShareLinkStatus>(response);
  },

  async checkUserSpaceWorkspaceShareSlugAvailability(
    workspaceId: string,
    slug: string,
    shareId?: string,
  ): Promise<WorkspaceShareSlugAvailabilityResponse> {
    const params = new URLSearchParams({ slug });
    if (shareId) params.set('share_id', shareId);
    const response = await apiFetch(
      `${API_BASE}/userspace/workspaces/${workspaceId}/share-links/availability?${params.toString()}`,
    );
    return handleResponse<WorkspaceShareSlugAvailabilityResponse>(response);
  },

  async updateUserSpaceWorkspaceShareAccess(
    workspaceId: string,
    shareIdOrRequest: string | UpdateUserSpaceWorkspaceShareAccessRequest,
    request?: UpdateUserSpaceWorkspaceShareAccessRequest,
  ): Promise<UserSpaceWorkspaceShareLinkStatus> {
    let resolvedShareId: string;
    let resolvedRequest: UpdateUserSpaceWorkspaceShareAccessRequest;
    if (typeof shareIdOrRequest === 'string') {
      resolvedShareId = shareIdOrRequest;
      resolvedRequest = request as UpdateUserSpaceWorkspaceShareAccessRequest;
    } else {
      const status = await this.getUserSpaceWorkspaceShareLinkStatus(workspaceId);
      if (!status.id) {
        throw new Error('No share link available to update');
      }
      resolvedShareId = status.id;
      resolvedRequest = shareIdOrRequest;
    }
    const response = await apiFetch(`${API_BASE}/userspace/workspaces/${workspaceId}/share-links/${resolvedShareId}/access`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(resolvedRequest),
    });
    return handleResponse<UserSpaceWorkspaceShareLinkStatus>(response);
  },

  async listConversationShareLinks(conversationId: string): Promise<ConversationShareLinkListResponse> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links`);
    return handleResponse<ConversationShareLinkListResponse>(response);
  },

  // Backward-compat helpers operating on the primary (oldest) share link.
  async getConversationShareLinkStatus(conversationId: string): Promise<ConversationShareLinkStatus> {
    const list = await this.listConversationShareLinks(conversationId);
    if (list.links.length > 0) {
      return list.links[0];
    }
    return {
      id: '',
      conversation_id: conversationId,
      has_share_link: false,
      owner_username: list.owner_username,
      share_slug: null,
      share_token: null,
      share_url: null,
      anonymous_share_url: null,
      created_at: null,
      share_access_mode: 'token',
      selected_user_ids: [],
      selected_ldap_groups: [],
      has_password: false,
      granted_role: 'viewer',
    };
  },

  async revokeConversationShareLink(conversationId: string): Promise<ConversationShareLinkStatus> {
    const list = await this.listConversationShareLinks(conversationId);
    for (const link of list.links) {
      if (link.id) {
        await this.deleteConversationShareLink(conversationId, link.id);
      }
    }
    return this.getConversationShareLinkStatus(conversationId);
  },

  async createConversationShareLink(
    conversationId: string,
    request: CreateConversationShareLinkRequest = {},
  ): Promise<ConversationShareLink> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ConversationShareLink>(response);
  },

  async deleteConversationShareLink(conversationId: string, shareId: string): Promise<void> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links/${shareId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      await handleResponse<unknown>(response);
    }
  },

  async updateConversationShareLinkMetadata(
    conversationId: string,
    shareId: string,
    request: UpdateConversationShareLinkRequest,
  ): Promise<ConversationShareLinkStatus> {
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links/${shareId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ConversationShareLinkStatus>(response);
  },

  async updateConversationShareSlug(
    conversationId: string,
    shareIdOrSlug: string,
    slug?: string,
  ): Promise<ConversationShareLinkStatus> {
    // Backward compatible: if called with two args, resolve primary share id.
    let resolvedShareId: string;
    let resolvedSlug: string;
    if (slug === undefined) {
      const status = await this.getConversationShareLinkStatus(conversationId);
      if (!status.id) {
        throw new Error('No share link available to update');
      }
      resolvedShareId = status.id;
      resolvedSlug = shareIdOrSlug;
    } else {
      resolvedShareId = shareIdOrSlug;
      resolvedSlug = slug;
    }
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links/${resolvedShareId}/slug`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug: resolvedSlug }),
    });
    return handleResponse<ConversationShareLinkStatus>(response);
  },

  async checkConversationShareSlugAvailability(
    conversationId: string,
    slug: string,
    shareId?: string,
  ): Promise<ConversationShareSlugAvailabilityResponse> {
    const params = new URLSearchParams({ slug });
    if (shareId) params.set('share_id', shareId);
    const response = await apiFetch(
      `${API_BASE}/conversations/${conversationId}/share-links/availability?${params.toString()}`,
    );
    return handleResponse<ConversationShareSlugAvailabilityResponse>(response);
  },

  async updateConversationShareAccess(
    conversationId: string,
    shareIdOrRequest: string | UpdateConversationShareAccessRequest,
    request?: UpdateConversationShareAccessRequest,
  ): Promise<ConversationShareLinkStatus> {
    let resolvedShareId: string;
    let resolvedRequest: UpdateConversationShareAccessRequest;
    if (typeof shareIdOrRequest === 'string') {
      resolvedShareId = shareIdOrRequest;
      resolvedRequest = request as UpdateConversationShareAccessRequest;
    } else {
      const status = await this.getConversationShareLinkStatus(conversationId);
      if (!status.id) {
        throw new Error('No share link available to update');
      }
      resolvedShareId = status.id;
      resolvedRequest = shareIdOrRequest;
    }
    const response = await apiFetch(`${API_BASE}/conversations/${conversationId}/share-links/${resolvedShareId}/access`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(resolvedRequest),
    });
    return handleResponse<ConversationShareLinkStatus>(response);
  },

  async resolvePublicShareTarget(shareToken: string): Promise<PublicShareTargetResponse> {
    const response = await apiFetch(`${API_BASE}/public-shares/${encodeURIComponent(shareToken)}/target`);
    return handleResponse<PublicShareTargetResponse>(response);
  },

  async resolvePublicShareTargetBySlug(ownerUsername: string, shareSlug: string): Promise<PublicShareTargetResponse> {
    const response = await apiFetch(`${API_BASE}/public-shares/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/target`);
    return handleResponse<PublicShareTargetResponse>(response);
  },

  async getSharedConversation(shareToken: string, password?: string): Promise<SharedConversationResponse> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(shareToken)}${suffix}`);
    return handleResponse<SharedConversationResponse>(response);
  },

  async getSharedConversationBySlug(ownerUsername: string, shareSlug: string, password?: string): Promise<SharedConversationResponse> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}${suffix}`);
    return handleResponse<SharedConversationResponse>(response);
  },

  getSharedConversationEventsUrl(shareToken: string, password?: string): string {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    return `${API_BASE}/shared-conversations/${encodeURIComponent(shareToken)}/events${suffix}`;
  },

  getSharedConversationEventsUrlBySlug(ownerUsername: string, shareSlug: string, password?: string): string {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    return `${API_BASE}/shared-conversations/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/events${suffix}`;
  },

  async joinSharedConversation(shareToken: string, password?: string): Promise<SharedConversationResponse> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(shareToken)}/join${suffix}`, {
      method: 'POST',
    });
    return handleResponse<SharedConversationResponse>(response);
  },

  async joinSharedConversationBySlug(ownerUsername: string, shareSlug: string, password?: string): Promise<SharedConversationResponse> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/join${suffix}`, {
      method: 'POST',
    });
    return handleResponse<SharedConversationResponse>(response);
  },

  async sendSharedConversationMessage(shareToken: string, request: SendMessageRequest, password?: string): Promise<{ message: ChatMessage; conversation: Conversation; task?: import('@/types').ChatTask | null }> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(shareToken)}/messages${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<{ message: ChatMessage; conversation: Conversation; task?: import('@/types').ChatTask | null }>(response);
  },

  async sendSharedConversationMessageBySlug(ownerUsername: string, shareSlug: string, request: SendMessageRequest, password?: string): Promise<{ message: ChatMessage; conversation: Conversation; task?: import('@/types').ChatTask | null }> {
    const suffix = password ? `?password=${encodeURIComponent(password)}` : '';
    const response = await apiFetch(`${API_BASE}/shared-conversations/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/messages${suffix}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<{ message: ChatMessage; conversation: Conversation; task?: import('@/types').ChatTask | null }>(response);
  },

  async executeUserSpaceSharedComponent(shareToken: string, request: ExecuteComponentRequest): Promise<ExecuteComponentResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/shared/${encodeURIComponent(shareToken)}/execute-component`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ExecuteComponentResponse>(response);
  },

  async executeUserSpaceSharedComponentBySlug(ownerUsername: string, shareSlug: string, request: ExecuteComponentRequest): Promise<ExecuteComponentResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/shared/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/execute-component`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<ExecuteComponentResponse>(response);
  },

  async getUserSpaceRuntimeSession(workspaceId: string): Promise<UserSpaceRuntimeSessionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/session`);
    return handleResponse<UserSpaceRuntimeSessionResponse>(response);
  },

  async startUserSpaceRuntimeSession(workspaceId: string): Promise<UserSpaceRuntimeActionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/session/start`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceRuntimeActionResponse>(response);
  },

  async stopUserSpaceRuntimeSession(workspaceId: string): Promise<UserSpaceRuntimeActionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/session/stop`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceRuntimeActionResponse>(response);
  },

  async getUserSpaceRuntimeDevserverStatus(workspaceId: string): Promise<UserSpaceRuntimeStatusResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/devserver/status`);
    return handleResponse<UserSpaceRuntimeStatusResponse>(response);
  },

  async getUserSpaceWorkspaceTabState(workspaceId: string, selectedConversationId?: string | null): Promise<UserSpaceWorkspaceTabStateResponse> {
    const selectedQuery = selectedConversationId
      ? `?selected_conversation_id=${encodeURIComponent(selectedConversationId)}`
      : '';
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/tab-state${selectedQuery}`);
    return handleResponse<UserSpaceWorkspaceTabStateResponse>(response);
  },

  async restartUserSpaceRuntimeDevserver(workspaceId: string): Promise<UserSpaceRuntimeActionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/devserver/restart`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceRuntimeActionResponse>(response);
  },

  async refreshUserSpaceRuntimeMount(workspaceId: string, mountId: string): Promise<UserSpaceRuntimeActionResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/mounts/${mountId}/refresh`, {
      method: 'POST',
    });
    return handleResponse<UserSpaceRuntimeActionResponse>(response);
  },

  /**
   * Subscribe to workspace change events via SSE.
    * Returns an EventSource that emits 'message' events with JSON payload
    * at minimum { generation: number } and optionally { event_type, runtime }.
   */
  subscribeWorkspaceEvents(workspaceId: string, afterGeneration: number = 0): EventSource {
    const url = `${API_BASE}/userspace/runtime/workspaces/${encodeURIComponent(workspaceId)}/events?after=${afterGeneration}`;
    return new EventSource(url, { withCredentials: true });
  },

  async issueUserSpaceCapabilityToken(workspaceId: string, capabilities: string[], sessionId?: string): Promise<UserSpaceCapabilityTokenResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/capability-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capabilities, session_id: sessionId }),
    });
    return handleResponse<UserSpaceCapabilityTokenResponse>(response);
  },

  async authorizeUserSpaceBrowserSurfaces(workspaceId: string, surfaces: UserSpaceBrowserSurface[]): Promise<UserSpaceBrowserAuthResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/browser-auth`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ surfaces }),
    });
    return handleResponse<UserSpaceBrowserAuthResponse>(response);
  },

  async launchUserSpacePreview(workspaceId: string, request: UserSpacePreviewLaunchRequest): Promise<UserSpacePreviewLaunchResponse> {
    const response = await apiFetch(`${API_BASE}/userspace/runtime/workspaces/${workspaceId}/preview-launch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpacePreviewLaunchResponse>(response);
  },

  async launchUserSpaceSharedPreview(shareToken: string, request: UserSpacePreviewLaunchRequest, password?: string): Promise<UserSpacePreviewLaunchResponse> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (password) {
      headers['X-UserSpace-Share-Password'] = password;
    }
    const response = await apiFetch(`${API_BASE}/userspace/shared/${encodeURIComponent(shareToken)}/preview-launch`, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpacePreviewLaunchResponse>(response);
  },

  async launchUserSpaceSharedPreviewBySlug(ownerUsername: string, shareSlug: string, request: UserSpacePreviewLaunchRequest, password?: string): Promise<UserSpacePreviewLaunchResponse> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (password) {
      headers['X-UserSpace-Share-Password'] = password;
    }
    const response = await apiFetch(`${API_BASE}/userspace/shared/${encodeURIComponent(ownerUsername)}/${encodeURIComponent(shareSlug)}/preview-launch`, {
      method: 'POST',
      headers,
      body: JSON.stringify(request),
    });
    return handleResponse<UserSpacePreviewLaunchResponse>(response);
  },

  getUserSpaceCollabWebSocketUrl(workspaceId: string, filePath: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const encodedPath = encodeFilePath(filePath);
    return `${protocol}//${window.location.host}${API_BASE}/userspace/collab/workspaces/${encodeURIComponent(workspaceId)}/files/${encodedPath}`;
  },

  getUserSpaceRuntimePtyWebSocketUrl(workspaceId: string): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}${API_BASE}/userspace/runtime/workspaces/${encodeURIComponent(workspaceId)}/pty`;
  },

  async createUserSpaceCollabFile(workspaceId: string, filePath: string, content: string = ''): Promise<{ success: boolean; workspace_id: string; file_path: string; version: number }> {
    const capability = await this.issueUserSpaceCapabilityToken(workspaceId, ['userspace.collab_mutate']);
    const response = await apiFetch(`${API_BASE}/userspace/collab/workspaces/${workspaceId}/files/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${capability.token}`,
      },
      body: JSON.stringify({ file_path: filePath, content }),
    });
    return handleResponse(response);
  },

  async renameUserSpaceCollabFile(workspaceId: string, oldPath: string, newPath: string): Promise<{ old_path: string; new_path: string; success: boolean }> {
    const capability = await this.issueUserSpaceCapabilityToken(workspaceId, ['userspace.collab_mutate']);
    const response = await apiFetch(`${API_BASE}/userspace/collab/workspaces/${workspaceId}/files/rename`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${capability.token}`,
      },
      body: JSON.stringify({ old_path: oldPath, new_path: newPath }),
    });
    return handleResponse(response);
  },

  async deleteUserSpaceCollabFile(workspaceId: string, filePath: string): Promise<{ file_path: string; success: boolean }> {
    const capability = await this.issueUserSpaceCapabilityToken(workspaceId, ['userspace.collab_mutate']);
    const response = await apiFetch(`${API_BASE}/userspace/collab/workspaces/${workspaceId}/files/delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${capability.token}`,
      },
      body: JSON.stringify({ file_path: filePath }),
    });
    return handleResponse(response);
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
