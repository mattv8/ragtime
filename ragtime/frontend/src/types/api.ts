/**
 * API Types for Ragtime Indexer
 * These types mirror the Python Pydantic models in app/indexer/models.py
 */

// =============================================================================
// Authentication Types
// =============================================================================

export type UserRole = 'user' | 'admin';
export type AuthProvider = 'ldap' | 'local';

export interface User {
  id: string;
  username: string;
  display_name: string | null;
  email: string | null;
  role: UserRole;
  auth_provider: AuthProvider;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  success: boolean;
  user_id?: string;
  username?: string;
  display_name?: string;
  email?: string;
  role: UserRole;
  error?: string;
}

export type AuthMethodAvailability = 'available' | 'unavailable' | 'not_configured';

export interface AuthMethodStatus {
  key: string;
  label: string;
  configured: boolean;
  available: boolean;
  status: AuthMethodAvailability;
  detail?: string | null;
}

export interface AuthStatus {
  authenticated: boolean;
  ldap_configured: boolean;
  local_admin_enabled: boolean;
  debug_mode: boolean;
  debug_username?: string;
  debug_password?: string;
  cookie_warning?: string;
  // Security status for UI banner
  api_key_configured: boolean;
  session_cookie_secure: boolean;
  allowed_origins_open: boolean;  // True if ALLOWED_ORIGINS=*
  auth_methods?: AuthMethodStatus[];
}

// =============================================================================
// MCP Route Configuration Types
// =============================================================================

export interface McpRouteConfig {
  id: string;
  name: string;
  route_path: string;
  description: string;
  enabled: boolean;
  require_auth: boolean;
  has_password: boolean;
  auth_password?: string;  // Decrypted password for admin display
  auth_method: 'password' | 'oauth2';
  allowed_ldap_group: string | null;
  include_knowledge_search: boolean;
  include_git_history: boolean;
  selected_document_indexes: string[];
  selected_filesystem_indexes: string[];
  selected_schema_indexes: string[];
  tool_config_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface CreateMcpRouteRequest {
  name: string;
  route_path: string;
  description?: string;
  require_auth?: boolean;
  auth_password?: string;
  auth_method?: 'password' | 'oauth2';
  allowed_ldap_group?: string;
  include_knowledge_search?: boolean;
  include_git_history?: boolean;
  selected_document_indexes?: string[];
  selected_filesystem_indexes?: string[];
  selected_schema_indexes?: string[];
  tool_config_ids?: string[];
}

export interface UpdateMcpRouteRequest {
  name?: string;
  description?: string;
  enabled?: boolean;
  require_auth?: boolean;
  auth_password?: string;
  clear_password?: boolean;
  auth_method?: 'password' | 'oauth2';
  allowed_ldap_group?: string;
  clear_allowed_ldap_group?: boolean;
  include_knowledge_search?: boolean;
  include_git_history?: boolean;
  selected_document_indexes?: string[];
  selected_filesystem_indexes?: string[];
  selected_schema_indexes?: string[];
  tool_config_ids?: string[];
}

export interface McpRouteListResponse {
  routes: McpRouteConfig[];
  count: number;
}

// =============================================================================
// MCP Default Route Filter Types (LDAP group-based tool filtering)
// =============================================================================

export interface McpDefaultRouteFilter {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  priority: number;
  ldap_group_dn: string;
  include_knowledge_search: boolean;
  include_git_history: boolean;
  selected_document_indexes: string[];
  selected_filesystem_indexes: string[];
  selected_schema_indexes: string[];
  tool_config_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface CreateMcpDefaultRouteFilterRequest {
  name: string;
  ldap_group_dn: string;
  description?: string;
  priority?: number;
  include_knowledge_search?: boolean;
  include_git_history?: boolean;
  selected_document_indexes?: string[];
  selected_filesystem_indexes?: string[];
  selected_schema_indexes?: string[];
  tool_config_ids?: string[];
}

export interface UpdateMcpDefaultRouteFilterRequest {
  name?: string;
  description?: string;
  enabled?: boolean;
  priority?: number;
  include_knowledge_search?: boolean;
  include_git_history?: boolean;
  selected_document_indexes?: string[];
  selected_filesystem_indexes?: string[];
  selected_schema_indexes?: string[];
  tool_config_ids?: string[];
}

export interface McpDefaultRouteFilterListResponse {
  filters: McpDefaultRouteFilter[];
  count: number;
}

// =============================================================================
// LDAP Configuration Types
// =============================================================================

export interface LdapConfig {
  server_url: string;
  bind_dn: string;
  allow_self_signed: boolean;
  base_dn: string;
  user_search_base: string;
  user_search_filter: string;
  admin_group_dn: string;
  user_group_dn: string;
  discovered_ous: string[];
  discovered_groups: { dn: string; name: string }[];
}

export interface LdapDiscoverRequest {
  server_url: string;
  bind_dn: string;
  bind_password: string;
  allow_self_signed?: boolean;
}

export interface LdapDiscoverResponse {
  success: boolean;
  base_dn?: string;
  user_ous: string[];
  groups: { dn: string; name: string }[];
  error?: string;
}

export interface LdapBindDnLookupRequest {
  server_url: string;
  username: string;
  password: string;
}

export interface LdapBindDnLookupResponse {
  success: boolean;
  bind_dn?: string;
  display_name?: string;
  error?: string;
}

// =============================================================================
// Index Types
// =============================================================================

export type IndexStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'interrupted';
export type OcrMode = 'disabled' | 'tesseract' | 'ollama';

export interface IndexConfig {
  name: string;
  description?: string;
  file_patterns: string[];
  exclude_patterns: string[];
  chunk_size: number;
  chunk_overlap: number;
  max_file_size_kb?: number;  // Max file size in KB (default 500)
  embedding_model?: string;
  vector_store_type?: VectorStoreType;  // Vector store backend: faiss (default) or pgvector
  ocr_mode?: OcrMode;  // OCR mode: disabled, tesseract, or ollama
  ocr_vision_model?: string;  // Ollama vision model for OCR
  git_clone_timeout_minutes?: number;  // Max time for git clone (default 5 min)
  git_history_depth?: number;  // 1=shallow (default), 0=full history
  reindex_interval_hours?: number;  // Hours between auto pull & re-index (0=manual)
}

export interface IndexJob {
  id: string;
  name: string;
  status: IndexStatus;
  progress_percent: number;
  total_files: number;
  processed_files: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  vector_store_type?: VectorStoreType;  // Vector store backend used
}

export interface IndexConfigSnapshot {
  file_patterns: string[];
  exclude_patterns: string[];
  chunk_size: number;
  chunk_overlap: number;
  max_file_size_kb: number;
  ocr_mode?: OcrMode;  // OCR mode: disabled, tesseract, or ollama
  ocr_vision_model?: string;  // Ollama vision model for OCR
  enable_ocr?: boolean;  // Legacy - for backward compatibility
  git_clone_timeout_minutes?: number;  // May be absent for older indexes
  git_history_depth?: number;  // 1=shallow (default), 0=full history
  reindex_interval_hours?: number;  // Hours between auto pull & re-index (0=manual)
}

export interface IndexInfo {
  name: string;  // Safe tool name (lowercase, alphanumeric with underscores)
  display_name: string | null;  // Human-readable name for UI display
  path: string;
  size_mb: number;
  document_count: number;
  chunk_count: number;  // Number of chunks/vectors for memory calculation
  description: string;
  enabled: boolean;
  search_weight: number;  // 0.0-10.0, default 1.0
  source_type: 'upload' | 'git';
  source: string | null;  // git URL or original filename
  git_branch: string | null;  // branch for git sources
  has_stored_token: boolean;  // True if a git token is stored for re-indexing
  config_snapshot: IndexConfigSnapshot | null;  // Configuration used for indexing
  created_at: string | null;
  last_modified: string | null;
  // Git history info
  git_repo_size_mb: number | null;  // Size of .git_repo directory (disk)
  has_git_history: boolean;  // True if .git_repo exists with history
  // Vector store backend
  vector_store_type?: VectorStoreType;  // faiss (default) or pgvector
}

export interface UpdateIndexConfigRequest {
  git_branch?: string;
  file_patterns?: string[];
  exclude_patterns?: string[];
  chunk_size?: number;
  chunk_overlap?: number;
  max_file_size_kb?: number;
  ocr_mode?: OcrMode;
  ocr_vision_model?: string;
  git_clone_timeout_minutes?: number;
  git_history_depth?: number;
  reindex_interval_hours?: number;
}

export interface RenameIndexRequest {
  new_name: string;
}

export interface RenameIndexResponse {
  old_name: string;
  new_name: string;  // Safe tool name
  display_name: string;  // Human-readable name
  message: string;
}

export interface CreateIndexRequest {
  name: string;
  git_url?: string;
  git_branch?: string;
  git_token?: string;
  config?: Partial<IndexConfig>;
}

// Index Analysis Types (pre-indexing estimation)
export interface FileTypeStats {
  extension: string;
  file_count: number;
  total_size_bytes: number;
  estimated_chunks: number;
  sample_files: string[];
}

export interface CommitHistorySample {
  depth: number;
  date: string;
  hash: string;
}

export interface DeleteUserspaceMountSourceResponse {
  success: boolean;
  mount_source_id: string;
}

export interface CommitHistoryInfo {
  total_commits: number;
  samples: CommitHistorySample[];
  oldest_date: string | null;
  newest_date: string | null;
}

export interface MemoryDimensionBreakdown {
  dimension: number;
  steady_memory_mb: number;
  peak_memory_mb: number;
  examples: string[];
}

export interface MemoryEstimate {
  embedding_dimension: number;
  steady_memory_mb: number;
  peak_memory_mb: number;
  dimension_breakdown?: MemoryDimensionBreakdown[];
}

export interface IndexAnalysisResult {
  total_files: number;
  total_size_bytes: number;
  total_size_mb: number;
  estimated_chunks: number;
  estimated_index_size_mb: number;
  memory_estimate?: MemoryEstimate;
  total_memory_with_existing_mb?: number;
  file_type_stats: FileTypeStats[];
  suggested_exclusions: string[];
  matched_file_patterns: string[];
  warnings: string[];
  chunk_size: number;
  chunk_overlap: number;
  commit_history?: CommitHistoryInfo;
}

export interface AnalyzeIndexRequest {
  git_url: string;
  git_branch: string;
  git_token?: string;
  file_patterns: string[];
  exclude_patterns: string[];
  chunk_size: number;
  chunk_overlap: number;
  max_file_size_kb?: number;
  ocr_mode?: OcrMode;
  ocr_vision_model?: string;
}

// =============================================================================
// Git Repository Visibility
// =============================================================================

export type RepoVisibility = 'public' | 'private' | 'not_found' | 'error';

export interface CheckRepoVisibilityRequest {
  git_url: string;
  index_name?: string;
}

export interface RepoVisibilityResponse {
  visibility: RepoVisibility;
  has_stored_token: boolean;
  needs_token: boolean;
  message: string;
}

export interface FetchBranchesRequest {
  git_url: string;
  git_token?: string;
  index_name?: string;
}

export interface FetchBranchesResponse {
  branches: string[];
  error: string | null;
  needs_token: boolean;
}

export interface UploadFormData {
  name: string;
  file: File;
  file_patterns: string;
  exclude_patterns: string;
  chunk_size: number;
  chunk_overlap: number;
}

export interface GitFormData {
  name: string;
  git_url: string;
  git_branch: string;
  file_patterns: string;
  exclude_patterns: string;
}

// Application Settings
export interface ConfigurationWarning {
  level: 'info' | 'warning' | 'error';
  category: string;
  message: string;
  recommendation?: string | null;
}

export interface GetSettingsResponse {
  settings: AppSettings;
  configuration_warnings: ConfigurationWarning[];
}

export interface UserSpacePreviewSettingsResponse {
  userspace_preview_sandbox_flags: string[];
  userspace_preview_sandbox_default_flags: string[];
  userspace_preview_sandbox_flag_options: UserSpacePreviewSandboxFlagOption[];
}

export interface UserSpacePreviewSandboxFlagOption {
  value: string;
  label: string;
  description: string;
}

// Canonical providers used by current UI flows.
export type LlmProvider = 'openai' | 'anthropic' | 'ollama' | 'github_copilot';
// Legacy wire compatibility for older persisted/provider values.
export type LlmProviderWire = LlmProvider | 'github_models';

export interface AppSettings {
  id: string;
  // Server branding
  server_name: string;
  // Embedding Configuration (for FAISS indexing)
  embedding_provider: 'ollama' | 'openai';
  embedding_model: string;
  embedding_dimensions?: number | null;
  // Ollama connection settings for embeddings (separate fields)
  ollama_protocol: 'http' | 'https';
  ollama_host: string;
  ollama_port: number;
  ollama_base_url: string;
  // LLM Configuration (for chat/RAG responses)
  llm_provider: LlmProviderWire;
  llm_model: string;
  llm_max_tokens?: number;
  image_payload_max_width?: number;
  image_payload_max_height?: number;
  image_payload_max_pixels?: number;
  image_payload_max_bytes?: number;
  // Ollama LLM connection settings (separate from embedding Ollama)
  llm_ollama_protocol: 'http' | 'https';
  llm_ollama_host: string;
  llm_ollama_port: number;
  llm_ollama_base_url: string;
  openai_api_key: string;
  anthropic_api_key: string;
  github_models_api_token: string;
  github_copilot_access_token: string;
  github_copilot_refresh_token: string;
  github_copilot_token_expires_at?: string | null;
  github_copilot_enterprise_url?: string | null;
  github_copilot_base_url: string;
  include_copilot_third_party_models: boolean;
  has_github_copilot_auth: boolean;
  allowed_chat_models: string[];
  default_chat_model?: string | null;
  // OpenAPI-compatible endpoint model configuration
  allowed_openapi_models: string[];
  openapi_sync_chat_models: boolean;
  max_iterations: number;
  // Token optimization settings
  max_tool_output_chars: number;
  scratchpad_window_size: number;
  // Search Configuration
  search_results_k: number;
  aggregate_search: boolean;
  // Advanced Search Settings
  search_use_mmr: boolean;
  search_mmr_lambda: number;
  context_token_budget: number;
  chunking_use_tokens: boolean;
  ivfflat_lists: number;
  // Embedding dimension tracking (pgvector)
  embedding_dimension?: number | null;
  embedding_config_hash?: string | null;
  // OCR Configuration
  default_ocr_mode: 'disabled' | 'tesseract' | 'ollama';
  default_ocr_vision_model?: string | null;
  ocr_concurrency_limit: number;
  ollama_embedding_timeout_seconds: number;
  // Performance / Memory Configuration
  sequential_index_loading: boolean;
  // API Tool Output Configuration
  tool_output_mode: ToolOutputMode;
  // MCP Configuration
  mcp_enabled: boolean;
  mcp_default_route_auth: boolean;
  mcp_default_route_auth_method: 'password' | 'oauth2';
  mcp_default_route_allowed_group: string | null;
  has_mcp_default_password: boolean;
  mcp_default_route_password?: string;
  // Tool Configuration
  enabled_tools: string[];
  odoo_container: string;
  postgres_container: string;
  postgres_host: string;
  postgres_port: number;
  postgres_user: string;
  postgres_password: string;
  postgres_database: string;
  max_query_results: number;
  query_timeout: number;
  enable_write_ops: boolean;
  snapshot_retention_days: number;
  userspace_preview_sandbox_flags: string[];
  updated_at: string | null;
}

export interface UpdateSettingsRequest {
  // Server branding
  server_name?: string;
  // Embedding settings
  embedding_provider?: 'ollama' | 'openai';
  embedding_model?: string;
  embedding_dimensions?: number | null;
  ollama_protocol?: 'http' | 'https';
  ollama_host?: string;
  ollama_port?: number;
  ollama_base_url?: string;
  // LLM settings
  llm_provider?: LlmProviderWire;
  llm_model?: string;
  llm_max_tokens?: number;
  image_payload_max_width?: number;
  image_payload_max_height?: number;
  image_payload_max_pixels?: number;
  image_payload_max_bytes?: number;
  // LLM Ollama connection settings
  llm_ollama_protocol?: 'http' | 'https';
  llm_ollama_host?: string;
  llm_ollama_port?: number;
  llm_ollama_base_url?: string;
  openai_api_key?: string;
  anthropic_api_key?: string;
  github_models_api_token?: string;
  github_copilot_access_token?: string;
  github_copilot_refresh_token?: string;
  github_copilot_token_expires_at?: string | null;
  github_copilot_enterprise_url?: string | null;
  github_copilot_base_url?: string;
  include_copilot_third_party_models?: boolean;
  allowed_chat_models?: string[];
  default_chat_model?: string | null;
  allowed_openapi_models?: string[];
  openapi_sync_chat_models?: boolean;
  max_iterations?: number;
  // Token optimization settings
  max_tool_output_chars?: number;
  scratchpad_window_size?: number;
  // Search settings
  search_results_k?: number;
  aggregate_search?: boolean;
  // Advanced Search Settings
  search_use_mmr?: boolean;
  search_mmr_lambda?: number;
  context_token_budget?: number;
  chunking_use_tokens?: boolean;
  ivfflat_lists?: number;
  // Performance / Memory settings
  sequential_index_loading?: boolean;
  // OCR settings
  default_ocr_mode?: 'disabled' | 'tesseract' | 'ollama';
  default_ocr_vision_model?: string | null;
  ocr_concurrency_limit?: number;
  ollama_embedding_timeout_seconds?: number;
  // API Tool Output settings
  tool_output_mode?: ToolOutputMode;
  // MCP settings
  mcp_enabled?: boolean;
  mcp_default_route_auth?: boolean;
  mcp_default_route_auth_method?: 'password' | 'oauth2';
  mcp_default_route_allowed_group?: string | null;
  mcp_default_route_password?: string;
  // Tool settings
  enabled_tools?: string[];
  odoo_container?: string;
  postgres_container?: string;
  postgres_host?: string;
  postgres_port?: number;
  postgres_user?: string;
  postgres_password?: string;
  postgres_database?: string;
  max_query_results?: number;
  query_timeout?: number;
  enable_write_ops?: boolean;
  snapshot_retention_days?: number;
  userspace_preview_sandbox_flags?: string[];
}

// Ollama Connection Testing
export interface OllamaTestRequest {
  protocol: 'http' | 'https';
  host: string;
  port: number;
  /** If true, only return models capable of generating embeddings */
  embeddings_only?: boolean;
}

export interface OllamaModel {
  name: string;
  modified_at?: string;
  size?: number;
  dimensions?: number;
  is_embedding_model?: boolean;
}

export interface OllamaTestResponse {
  success: boolean;
  message: string;
  models: OllamaModel[];
  base_url: string;
}

// Ollama Vision Models
export interface OllamaVisionModel {
  name: string;
  modified_at?: string;
  size?: number;
  family?: string;
  parameter_size?: string;
  capabilities?: string[];
}

export interface OllamaVisionModelsRequest {
  protocol: 'http' | 'https';
  host: string;
  port: number;
}

export interface OllamaVisionModelsResponse {
  success: boolean;
  message: string;
  models: OllamaVisionModel[];
  base_url: string;
}

// LLM Provider Model Fetching
export interface LLMModelsRequest {
  provider: Extract<LlmProvider, 'openai' | 'anthropic' | 'github_copilot'>;
  api_key?: string;
  auth_mode?: 'oauth' | 'pat';
  include_directory_models?: boolean;
  include_anthropic_models?: boolean;
  include_google_models?: boolean;
}

export interface LLMModel {
  id: string;
  name: string;
  created?: number;
  group?: string;
  is_latest?: boolean;
  max_output_tokens?: number;
  context_limit?: number;
  capabilities?: string[];
  supported_endpoints?: string[];
  reasoning_supported?: boolean;
  thinking_budget_supported?: boolean;
  effort_levels?: string[];
}

export interface LLMModelsResponse {
  success: boolean;
  message: string;
  models: LLMModel[];
  default_model?: string;
}

export interface CopilotDeviceStartRequest {
  deployment_type?: 'github.com' | 'enterprise';
  enterprise_url?: string;
}

export interface CopilotDeviceStartResponse {
  success: boolean;
  request_id: string;
  verification_uri: string;
  verification_uri_complete?: string | null;
  user_code: string;
  interval: number;
  expires_in: number;
  deployment_type: 'github.com' | 'enterprise';
  enterprise_url?: string | null;
}

export interface CopilotDevicePollRequest {
  request_id: string;
}

export interface CopilotDevicePollResponse {
  success: boolean;
  status: 'pending' | 'connected' | 'expired' | 'failed';
  message: string;
  retry_after_seconds?: number;
}

export interface CopilotAuthStatusResponse {
  connected: boolean;
  deployment_type: 'github.com' | 'enterprise';
  enterprise_url?: string | null;
  base_url: string;
  token_expires_at?: string | null;
}

// Embedding Provider Model Fetching
export interface EmbeddingModelsRequest {
  provider: 'openai';
  api_key: string;
}

export interface EmbeddingModel {
  id: string;
  name: string;
  dimensions?: number;
}

export interface EmbeddingModelsResponse {
  success: boolean;
  message: string;
  models: EmbeddingModel[];
  default_model?: string;
}

// Tool Configuration Types
export type ToolType = 'postgres' | 'mysql' | 'mssql' | 'influxdb' | 'odoo_shell' | 'ssh_shell' | 'filesystem_indexer' | 'solidworks_pdm';

// Mount type for filesystem indexer
export type FilesystemMountType = 'docker_volume' | 'smb' | 'nfs' | 'local';

// Vector store type for indexes (document and filesystem)
export type VectorStoreType = 'pgvector' | 'faiss';

// SSH Tunnel configuration (shared across database tools)
export interface SSHTunnelConfig {
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
  ssh_tunnel_public_key?: string;
}

export interface PostgresConnectionConfig extends SSHTunnelConfig {
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  database?: string;
  container?: string;
  docker_network?: string;
  // Schema indexing options
  schema_index_enabled?: boolean;
  schema_index_interval_hours?: number;
  last_schema_indexed_at?: string | null;
  schema_hash?: string | null;
}

export interface MssqlConnectionConfig extends SSHTunnelConfig {
  host: string;
  port?: number;
  user: string;
  password: string;
  database: string;
  // Schema indexing options
  schema_index_enabled?: boolean;
  schema_index_interval_hours?: number;
  last_schema_indexed_at?: string | null;
  schema_hash?: string | null;
}

export interface MysqlConnectionConfig extends SSHTunnelConfig {
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  database?: string;
  container?: string;
  docker_network?: string;
  // Schema indexing options
  schema_index_enabled?: boolean;
  schema_index_interval_hours?: number;
  last_schema_indexed_at?: string | null;
  schema_hash?: string | null;
}

export interface InfluxdbConnectionConfig extends SSHTunnelConfig {
  host?: string;
  port?: number;
  use_https?: boolean;
  token?: string;
  org?: string;
  bucket?: string;
}

export interface OdooShellConnectionConfig {
  mode?: 'docker' | 'ssh';
  // Docker mode
  container?: string;
  docker_network?: string;
  // SSH mode
  ssh_host?: string;
  ssh_port?: number;
  ssh_user?: string;
  ssh_key_path?: string;
  ssh_key_content?: string;
  ssh_public_key?: string;
  ssh_key_passphrase?: string;
  ssh_password?: string;
  // Common Odoo fields
  database?: string;
  odoo_bin_path?: string;
  config_path?: string;
  working_directory?: string;
  run_as_user?: string;
}

export interface SSHShellConnectionConfig {
  host: string;
  port?: number;
  user: string;
  key_path?: string;
  key_content?: string;
  public_key?: string;
  key_passphrase?: string;
  password?: string;
  command_prefix?: string;
  working_directory?: string;
}

export interface FilesystemConnectionConfig {
  // Mount configuration
  mount_type: FilesystemMountType;
  base_path: string;

  // Docker volume settings
  volume_name?: string;

  // SMB settings
  smb_host?: string;
  smb_share?: string;
  smb_user?: string;
  smb_password?: string;
  smb_domain?: string;

  // NFS settings
  nfs_host?: string;
  nfs_export?: string;
  nfs_options?: string;

  // Indexing configuration
  index_name: string;
  file_patterns?: string[];
  exclude_patterns?: string[];
  recursive?: boolean;
  chunk_size?: number;
  chunk_overlap?: number;

  // Vector store backend selection
  vector_store_type?: VectorStoreType;

  // Safety limits
  max_file_size_mb?: number;
  max_total_files?: number;
  ocr_mode?: OcrMode;  // OCR mode: disabled, tesseract, or ollama
  ocr_vision_model?: string;  // Ollama vision model for OCR

  // Re-indexing schedule
  reindex_interval_hours?: number;
  last_indexed_at?: string | null;
}

export interface SolidworksPdmConnectionConfig extends SSHTunnelConfig {
  // MSSQL Connection (same as MssqlConnectionConfig)
  host: string;
  port?: number;
  user: string;
  password: string;
  database: string;

  // Document filtering
  file_extensions?: string[];
  exclude_deleted?: boolean;

  // Metadata extraction
  variable_names?: string[];
  include_bom?: boolean;
  include_folder_path?: boolean;
  include_configurations?: boolean;

  // Indexing options
  max_documents?: number | null;

  // Last indexed info
  last_indexed_at?: string | null;
}

export type ConnectionConfig = PostgresConnectionConfig | MysqlConnectionConfig | MssqlConnectionConfig | InfluxdbConnectionConfig | OdooShellConnectionConfig | SSHShellConnectionConfig | FilesystemConnectionConfig | SolidworksPdmConnectionConfig;
export type UserspaceMountSourceType = 'ssh' | 'filesystem';
export type UserspaceMountBackend = 'ssh' | 'docker_volume' | 'smb' | 'nfs' | 'local';

export interface ToolConfig {
  id: string;
  name: string;
  tool_type: ToolType;
  enabled: boolean;
  description: string;
  connection_config: ConnectionConfig;
  max_results: number;
  timeout: number;
  timeout_max_seconds: number;
  allow_write: boolean;
  group_id?: string | null;
  group_name?: string | null;
  disabled_reason?: string;
  last_test_at: string | null;
  last_test_result: boolean | null;
  last_test_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ToolGroup {
  id: string;
  name: string;
  description: string;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

export interface CreateToolGroupRequest {
  name: string;
  description?: string;
  sort_order?: number;
}

export interface UpdateToolGroupRequest {
  name?: string;
  description?: string;
  sort_order?: number;
}

export interface CreateToolConfigRequest {
  name: string;
  tool_type: ToolType;
  description?: string;
  connection_config: ConnectionConfig;
  max_results?: number;
  timeout?: number;
  timeout_max_seconds?: number;
  allow_write?: boolean;
  group_id?: string | null;
  skip_indexing?: boolean;
}

export interface UpdateToolConfigRequest {
  name?: string;
  enabled?: boolean;
  description?: string;
  connection_config?: ConnectionConfig;
  max_results?: number;
  timeout?: number;
  timeout_max_seconds?: number;
  allow_write?: boolean;
  group_id?: string | null;
}

export interface ToolTestRequest {
  tool_type: ToolType;
  connection_config: ConnectionConfig;
}

export interface ToolTestResponse {
  success: boolean;
  message: string;
  details?: Record<string, unknown>;
}

export interface DatabaseDiscoverOption {
  name: string;
  accessible: boolean;
  access_error?: string;
}

// PostgreSQL Database Discovery
export interface PostgresDiscoverRequest {
  host: string;
  port: number;
  user: string;
  password: string;
  // SSH tunnel fields
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
}

export interface PostgresDiscoverResponse {
  success: boolean;
  databases: string[];
  database_options?: DatabaseDiscoverOption[];
  error?: string;
}

// MSSQL Database Discovery
export interface MssqlDiscoverRequest {
  host: string;
  port: number;
  user: string;
  password: string;
  // SSH tunnel fields
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
}

export interface MssqlDiscoverResponse {
  success: boolean;
  databases: string[];
  database_options?: DatabaseDiscoverOption[];
  error?: string;
}

// MySQL Database Discovery
export interface MysqlDiscoverRequest {
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  container?: string;
  docker_network?: string;
  // SSH tunnel fields
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
}

export interface MysqlDiscoverResponse {
  success: boolean;
  databases: string[];
  database_options?: DatabaseDiscoverOption[];
  error?: string;
}

// InfluxDB Bucket Discovery
export interface InfluxdbDiscoverRequest {
  host: string;
  port?: number;
  use_https?: boolean;
  token: string;
  org: string;
  // SSH tunnel fields
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
}

export interface InfluxdbDiscoverResponse {
  success: boolean;
  buckets: string[];
  database_options?: DatabaseDiscoverOption[];
  error?: string;
}

// PDM Schema Discovery
export interface PdmDiscoverRequest {
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
  // SSH tunnel fields
  ssh_tunnel_enabled?: boolean;
  ssh_tunnel_host?: string;
  ssh_tunnel_port?: number;
  ssh_tunnel_user?: string;
  ssh_tunnel_password?: string;
  ssh_tunnel_key_path?: string;
  ssh_tunnel_key_content?: string;
  ssh_tunnel_key_passphrase?: string;
}

export interface PdmDiscoverResponse {
  success: boolean;
  file_extensions: string[];
  variable_names: string[];
  document_count: number;
  error?: string;
}

// SSH Keypair Generation
export interface SSHKeyPairResponse {
  private_key: string;
  public_key: string;
  fingerprint: string;
}

// Heartbeat Status Types
export interface HeartbeatStatus {
  tool_id: string;
  alive: boolean;
  latency_ms: number | null;
  error: string | null;
  checked_at: string;
}

export interface HeartbeatResponse {
  statuses: Record<string, HeartbeatStatus>;
}

// Tool type metadata for the wizard UI
export const TOOL_TYPE_INFO: Record<ToolType, { name: string; description: string; icon: string; recommended?: boolean }> = {
  postgres: {
    name: 'PostgreSQL Database',
    description: 'Connect to a PostgreSQL database to execute SQL queries',
    icon: 'database',
  },
  mysql: {
    name: 'MySQL / MariaDB',
    description: 'Connect to a MySQL or MariaDB database to execute SQL queries',
    icon: 'database',
  },
  mssql: {
    name: 'MSSQL / SQL Server',
    description: 'Connect to Microsoft SQL Server or Azure SQL database',
    icon: 'database',
  },
  influxdb: {
    name: 'InfluxDB (Flux)',
    description: 'Connect to InfluxDB 2.x to execute Flux queries',
    icon: 'database',
  },
  odoo_shell: {
    name: 'Odoo Shell',
    description: 'Connect to an Odoo instance via Docker to run ORM commands',
    icon: 'terminal',
  },
  ssh_shell: {
    name: 'SSH Shell',
    description: 'Connect to a remote server via SSH to run shell commands',
    icon: 'server',
  },
  filesystem_indexer: {
    name: 'Filesystem Indexer',
    description: 'Index files from a filesystem path (Docker volume, SMB, NFS, or local) for semantic search',
    icon: 'folder',
  },
  solidworks_pdm: {
    name: 'SolidWorks PDM',
    description: 'Index SolidWorks PDM documents with metadata for semantic search',
    icon: 'folder',
  },
};

// Filesystem mount type metadata for the wizard UI
// Note: 'local' type exists in backend but hidden from UI since Docker Volume handles it
export const MOUNT_TYPE_INFO: Record<string, { name: string; description: string; recommended?: boolean }> = {
  docker_volume: {
    name: 'Docker Volume',
    description: 'Browse files from mounted Docker volumes',
  },
  smb: {
    name: 'SMB/CIFS Share',
    description: 'Windows network share (requires credentials)',
  },
  nfs: {
    name: 'NFS Mount',
    description: 'Unix/Linux network filesystem mount',
  },
};

// Filesystem indexer types
export type FilesystemIndexStatus = 'pending' | 'indexing' | 'completed' | 'failed' | 'cancelled';

export interface FilesystemIndexJob {
  id: string;
  tool_config_id: string;
  status: FilesystemIndexStatus;
  index_name: string;
  progress_percent: number;
  total_files: number;
  processed_files: number;
  skipped_files: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  cancel_requested: boolean;
  // Collection phase progress (for slow network filesystems)
  files_scanned: number;
  dirs_scanned: number;
  current_directory: string | null;
  // Timing
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface FilesystemTestResponse {
  success: boolean;
  message: string;
  file_count?: number;
  sample_files?: string[];
}

export interface FilesystemIndexStats {
  index_name: string;
  embedding_count: number;
  file_count: number;
  last_indexed: string | null;
  // Memory usage (FAISS RAM estimate)
  chunk_count?: number;
  estimated_memory_mb?: number;
  // pgvector disk storage size
  pgvector_size_mb?: number;
  // Vector store info
  vector_store_type?: 'pgvector' | 'faiss' | 'auto';
  pgvector_count?: number;
  faiss_count?: number;
}

export interface TriggerFilesystemIndexRequest {
  full_reindex?: boolean;
}

// Filesystem Analysis Types
export type FilesystemAnalysisStatus = 'pending' | 'scanning' | 'analyzing' | 'completed' | 'failed';

export interface FileTypeStats {
  extension: string;
  file_count: number;
  total_size_bytes: number;
  estimated_chunks: number;
  sample_files: string[];
}

export interface FilesystemAnalysisResult {
  total_files: number;
  total_size_bytes: number;
  total_size_mb: number;
  estimated_chunks: number;
  estimated_index_size_mb: number;
  file_type_stats: FileTypeStats[];
  suggested_exclusions: string[];
  warnings: string[];
  chunk_size: number;
  chunk_overlap: number;
  analysis_duration_seconds: number;
  directories_scanned: number;
}

export interface FilesystemAnalysisJob {
  id: string;
  tool_config_id: string;
  status: FilesystemAnalysisStatus;
  progress_percent: number;
  files_scanned: number;
  dirs_scanned: number;
  total_dirs_to_scan: number;
  current_directory: string;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  result: FilesystemAnalysisResult | null;
}

// Mount discovery types
export interface MountInfo {
  container_path: string;
  host_path: string;
  read_only: boolean;
  mount_type: string;
}

export interface DirectoryEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size?: number;
}

export interface BrowseResponse {
  path: string;
  entries: DirectoryEntry[];
  error?: string;
}

export interface MountDiscoveryResponse {
  mounts: MountInfo[];
  suggested_paths: string[];
  docker_compose_example: string;
}

// Container Capabilities Response
export interface ContainerCapabilitiesResponse {
  /** Whether the container is running in privileged mode */
  privileged: boolean;
  /** Whether the container has CAP_SYS_ADMIN capability */
  has_sys_admin: boolean;
  /** Whether the container can perform mount operations (SMB/NFS) */
  can_mount: boolean;
  /** Human-readable explanation of the capabilities status */
  message: string;
}

// NFS Discovery Types
export interface NFSExport {
  export_path: string;
  allowed_hosts: string;
}

export interface NFSDiscoveryResponse {
  success: boolean;
  exports: NFSExport[];
  error?: string;
}

// SMB Discovery Types
export interface SMBShare {
  name: string;
  share_type: string;
  comment: string;
}

export interface SMBDiscoveryResponse {
  success: boolean;
  shares: SMBShare[];
  error?: string;
}

// =============================================================================
// Schema Indexer Types (SQL Database Schema Indexing)
// =============================================================================

export type SchemaIndexStatus = 'pending' | 'indexing' | 'completed' | 'failed' | 'cancelled';

export interface SchemaIndexJob {
  id: string;
  tool_config_id: string;
  status: SchemaIndexStatus;
  index_name: string;
  progress_percent: number;
  total_tables: number;
  processed_tables: number;
  introspected_tables: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  cancel_requested: boolean;
  status_detail: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface SchemaIndexStatusResponse {
  enabled: boolean;
  last_indexed: string | null;
  schema_hash: string | null;
  interval_hours: number;
  embedding_count: number;
  current_job: SchemaIndexJob | null;
}

export interface SchemaIndexStats {
  embedding_count: number;
  table_count: number;
  last_indexed: string | null;
  schema_hash: string | null;
  // Memory estimation for schema index (pgvector storage, not process RAM)
  embedding_dimension?: number;
  estimated_memory_mb?: number;
}

// =============================================================================
// SolidWorks PDM Indexer Types
// =============================================================================

export type PdmIndexStatus = 'pending' | 'indexing' | 'completed' | 'failed' | 'cancelled';

export interface PdmIndexJob {
  id: string;
  tool_config_id: string;
  status: PdmIndexStatus;
  index_name: string;
  progress_percent: number;
  total_documents: number;
  processed_documents: number;
  skipped_documents: number;
  total_chunks: number;
  processed_chunks: number;
  error_message: string | null;
  cancel_requested: boolean;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface PdmIndexStatusResponse {
  enabled: boolean;
  last_indexed: string | null;
  document_count: number;
  embedding_count: number;
  current_job: PdmIndexJob | null;
}

export interface PdmIndexStats {
  document_count: number;
  embedding_count: number;
  last_indexed: string | null;
}

// =============================================================================
// Chat Types
// =============================================================================

export interface ToolCallRecord {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
  connection?: ToolConnectionRef;
}

export interface ToolConnectionRef {
  tool_config_id: string;
  tool_config_name?: string;
  tool_type?: string;
}

export interface ContentEvent {
  type: 'content';
  content: string;
}

export interface ToolCallEvent {
  type: 'tool';
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
  connection?: ToolConnectionRef;
}

export interface ReasoningEvent {
  type: 'reasoning';
  content: string;
}

export type MessageEvent = ContentEvent | ToolCallEvent | ReasoningEvent;

// Multimodal content types
export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageUrl {
  url: string;
  detail?: 'auto' | 'low' | 'high';
}

export interface ImageContent {
  type: 'image_url';
  image_url: ImageUrl;
}

export interface FileContent {
  type: 'file';
  file_path: string;
  filename: string;
  mime_type?: string;
}

export type ContentPart = TextContent | ImageContent | FileContent;

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | ContentPart[];  // Support both simple string and multimodal array
  timestamp: string;
  tool_calls?: ToolCallRecord[];  // Deprecated, for backward compatibility
  events?: MessageEvent[];        // Preferred: chronological events
}

// Streaming event types
export type StreamEventType = 'content' | 'tool_start' | 'tool_end' | 'reasoning' | 'error';

export interface ToolCallInfo {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
  connection?: ToolConnectionRef;
}

export interface StreamEvent {
  type: StreamEventType;
  content?: string;         // For 'content' events
  toolCall?: ToolCallInfo;  // For 'tool_start' and 'tool_end' events
  reasoning?: string;       // For 'reasoning' events (thinking/reasoning content)
  error?: string;          // For 'error' events
}

export type ToolOutputMode = 'default' | 'show' | 'hide' | 'auto';

export interface Conversation {
  id: string;
  title: string;
  model: string;
  user_id?: string;
  workspace_id?: string | null;
  workspaceId?: string | null;
  username?: string;
  display_name?: string;
  messages: ChatMessage[];
  total_tokens: number;
  active_task_id: string | null;  // ID of currently running background task
  tool_output_mode: ToolOutputMode;  // Per-conversation tool output preference
  created_at: string;
  updated_at: string;
}

export interface ConversationWorkspaceStateResponse {
  conversations: Conversation[];
  interrupted_conversation_ids: string[];
}

export interface WorkspaceConversationStateSummaryRequest {
  workspace_ids: string[];
}

export interface WorkspaceConversationStateSummaryItem {
  workspace_id: string;
  has_live_task: boolean;
  has_interrupted_task: boolean;
}

export interface WorkspaceChatStateResponse {
  conversations: Conversation[];
  interrupted_conversation_ids: string[];
  selected_conversation_id: string | null;
  active_task: ChatTask | null;
  interrupted_task: ChatTask | null;
}

export interface CreateConversationRequest {
  title?: string;
  model?: string;
  workspace_id?: string;
}

export interface SendMessageRequest {
  message: string;
  stream?: boolean;
}

export interface SendMessageResponse {
  message: ChatMessage;
  conversation: Conversation;
}

export interface ConversationMember {
  user_id: string;
  role: WorkspaceRole;  // Reuse existing WorkspaceRole type
}

export interface UpdateConversationMembersRequest {
  members: ConversationMember[];
}

export interface UpdateConversationToolsRequest {
  tool_config_ids: string[];
  tool_group_ids?: string[];
}

// =============================================================================
// User Space Types
// =============================================================================

export type WorkspaceRole = 'owner' | 'editor' | 'viewer';
export type UserSpaceArtifactType = 'module_ts';
export type SqlitePersistenceMode = 'include' | 'exclude';
export type UserSpaceWorkspaceDeletionPhase = 'queued' | 'deleting' | 'refreshing' | 'failed';

export interface UserSpaceWorkspaceMember {
  user_id: string;
  role: WorkspaceRole;
}

export type UserSpaceWorkspaceScmProvider = 'github' | 'gitlab' | 'generic';
export type UserSpaceWorkspaceScmDirection = 'import' | 'export';
export type UserSpaceWorkspaceScmPreviewState = 'missing_remote' | 'missing_branch' | 'up_to_date' | 'safe' | 'destructive';

export interface UserSpaceWorkspaceScmStatus {
  connected: boolean;
  git_url?: string | null;
  git_branch?: string | null;
  provider?: UserSpaceWorkspaceScmProvider | null;
  repo_visibility?: RepoVisibility | null;
  has_stored_token: boolean;
  connected_at?: string | null;
  last_sync_at?: string | null;
  last_sync_direction?: UserSpaceWorkspaceScmDirection | null;
  last_sync_status?: string | null;
  last_sync_message?: string | null;
  last_remote_commit_hash?: string | null;
  last_synced_snapshot_id?: string | null;
}

export interface UserSpaceWorkspace {
  id: string;
  name: string;
  description?: string | null;
  sqlite_persistence_mode: SqlitePersistenceMode;
  owner_user_id: string;
  owner_username?: string | null;
  owner_display_name?: string | null;
  selected_tool_ids: string[];
  selected_tool_group_ids: string[];
  conversation_ids: string[];
  members: UserSpaceWorkspaceMember[];
  scm?: UserSpaceWorkspaceScmStatus | null;
  created_at: string;
  updated_at: string;
}

export interface UserSpaceWorkspaceScmConnectionRequest {
  git_url: string;
  git_branch?: string;
  git_token?: string;
  repo_visibility?: RepoVisibility;
}

export interface UserSpaceWorkspaceScmConnectionResponse {
  workspace_id: string;
  scm: UserSpaceWorkspaceScmStatus;
}

export interface UserSpaceWorkspaceScmPreviewRequest {
  git_url?: string;
  git_branch?: string;
  git_token?: string;
  create_repo_if_missing?: boolean;
  create_repo_private?: boolean;
  create_repo_description?: string;
}

export interface UserSpaceWorkspaceScmPreviewResponse {
  workspace_id: string;
  direction: UserSpaceWorkspaceScmDirection;
  state: UserSpaceWorkspaceScmPreviewState;
  summary: string;
  git_url: string;
  git_branch: string;
  provider: UserSpaceWorkspaceScmProvider;
  repo_visibility?: RepoVisibility | null;
  local_changed: boolean;
  remote_changed: boolean;
  local_has_uncommitted_changes: boolean;
  will_overwrite_local: boolean;
  will_overwrite_remote: boolean;
  can_proceed_without_force: boolean;
  local_commit_hash?: string | null;
  remote_commit_hash?: string | null;
  current_snapshot_id?: string | null;
  changed_files_sample: string[];
  preview_token?: string | null;
  preview_expires_at?: string | null;
}

export interface UserSpaceWorkspaceScmImportRequest extends UserSpaceWorkspaceScmPreviewRequest {
  overwrite_preview_token?: string;
}

export interface UserSpaceWorkspaceScmExportRequest extends UserSpaceWorkspaceScmPreviewRequest {
  overwrite_preview_token?: string;
}

export interface UserSpaceWorkspaceScmSyncResponse {
  workspace_id: string;
  direction: UserSpaceWorkspaceScmDirection;
  state: string;
  summary: string;
  scm: UserSpaceWorkspaceScmStatus;
  snapshot?: UserSpaceSnapshot | null;
  remote_commit_hash?: string | null;
  suggested_setup_prompt?: string | null;
}

export interface UserSpaceAvailableTool {
  id: string;
  name: string;
  tool_type: string;
  description?: string | null;
  group_id?: string | null;
  group_name?: string | null;
}

export interface CreateUserSpaceWorkspaceRequest {
  name?: string;
  description?: string;
  sqlite_persistence_mode?: SqlitePersistenceMode;
  selected_tool_ids?: string[];
  selected_tool_group_ids?: string[];
}

export interface UpdateUserSpaceWorkspaceRequest {
  name?: string;
  description?: string;
  sqlite_persistence_mode?: SqlitePersistenceMode;
  selected_tool_ids?: string[];
  selected_tool_group_ids?: string[];
  owner_user_id?: string;
}

export interface UpdateUserSpaceWorkspaceMembersRequest {
  members: UserSpaceWorkspaceMember[];
}

export interface UserSpaceWorkspaceEnvVar {
  key: string;
  has_value: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface UpsertUserSpaceWorkspaceEnvVarRequest {
  key: string;
  value?: string;
  new_key?: string;
  description?: string;
}

export interface DeleteUserSpaceWorkspaceEnvVarResponse {
  success: boolean;
  key: string;
}

// ── Workspace Mounts ──────────────────────────────────────────────────

export type MountSyncStatus = 'pending' | 'synced' | 'error';
export type WorkspaceMountSyncMode = 'merge' | 'source_authoritative' | 'target_authoritative';

export interface UserspaceMountSource {
  id: string;
  name: string;
  description: string | null;
  enabled: boolean;
  source_type: UserspaceMountSourceType;
  mount_backend: UserspaceMountBackend;
  tool_config_id: string | null;
  tool_name: string | null;
  connection_config: ConnectionConfig;
  approved_paths: string[];
  sync_interval_seconds: number | null;
  usage_count: number;
  created_at: string;
  updated_at: string;
}

export interface CreateUserspaceMountSourceRequest {
  name: string;
  description?: string | null;
  enabled?: boolean;
  tool_config_id?: string;
  source_type?: UserspaceMountSourceType;
  connection_config?: ConnectionConfig;
  approved_paths?: string[];
  sync_interval_seconds?: number | null;
}

export interface UpdateUserspaceMountSourceRequest {
  name?: string;
  description?: string | null;
  enabled?: boolean;
  connection_config?: ConnectionConfig;
  approved_paths?: string[];
  sync_interval_seconds?: number | null;
}

export interface MountSourceAffectedWorkspace {
  workspace_id: string;
  workspace_name: string;
  owner_user_id: string;
  mount_count: number;
}

export interface MountSourceAffectedWorkspacesResponse {
  mount_source_id: string;
  mount_source_name: string;
  source_type: UserspaceMountSourceType;
  total_mounts: number;
  workspaces: MountSourceAffectedWorkspace[];
}

export interface WorkspaceMount {
  id: string;
  workspace_id: string;
  mount_source_id: string;
  source_path: string;
  target_path: string;
  description: string | null;
  enabled: boolean;
  sync_mode: WorkspaceMountSyncMode;
  sync_status: MountSyncStatus;
  sync_backend: string | null;
  sync_notice: string | null;
  last_sync_at: string | null;
  last_sync_error: string | null;
  auto_sync_enabled: boolean;
  source_name: string | null;
  source_type: UserspaceMountSourceType | null;
  mount_backend: UserspaceMountBackend | null;
  source_available: boolean;
  created_at: string;
  updated_at: string;
}

export interface MountableSource {
  mount_source_id: string;
  source_name: string;
  source_type: UserspaceMountSourceType;
  mount_backend: UserspaceMountBackend;
  source_path: string;
}

export interface BrowseUserspaceMountSourceRequest {
  path: string;
}

export interface BrowseWorkspaceMountSourceRequest {
  mount_source_id: string;
  root_source_path: string;
  path: string;
}

export interface CreateWorkspaceMountRequest {
  mount_source_id: string;
  source_path: string;
  target_path: string;
  source_directory_to_create?: string | null;
  target_directory_to_create?: string | null;
  auto_sync_enabled?: boolean;
  sync_mode?: WorkspaceMountSyncMode;
  description?: string | null;
}

export interface UpdateWorkspaceMountRequest {
  target_path?: string;
  description?: string | null;
  enabled?: boolean;
  auto_sync_enabled?: boolean;
  sync_mode?: WorkspaceMountSyncMode;
  destructive_auto_sync_preview_token?: string;
}

export interface DeleteWorkspaceMountResponse {
  success: boolean;
  mount_id: string;
}

export interface WorkspaceMountDirectoryEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size?: number;
}

export interface WorkspaceMountBrowseResponse {
  path: string;
  entries: WorkspaceMountDirectoryEntry[];
  error?: string;
}

export interface WorkspaceMountSyncResponse {
  mount_id: string;
  sync_mode: WorkspaceMountSyncMode;
  sync_status: MountSyncStatus;
  files_synced: number;
  sync_backend: string | null;
  sync_notice: string | null;
  last_sync_error: string | null;
}

export interface WorkspaceMountSyncRequest {
  preview_token?: string;
}

export interface WorkspaceMountSyncPreviewRequest {
  sync_mode?: WorkspaceMountSyncMode;
}

export interface WorkspaceMountSyncPreviewResponse {
  mount_id: string;
  sync_mode: WorkspaceMountSyncMode;
  sync_backend: string | null;
  sync_notice: string | null;
  requires_confirmation: boolean;
  preview_token: string;
  preview_expires_at: string;
  delete_from_source_count: number;
  delete_from_target_count: number;
  delete_from_source_paths: string[];
  delete_from_target_paths: string[];
  sample_limit: number;
  last_sync_error: string | null;
}

export interface UserSpaceFileInfo {
  path: string;
  size_bytes: number;
  updated_at: string;
  entry_type?: 'file' | 'directory';
}

export interface UserSpaceAcknowledgeChangedFilePathRequest {
  path: string;
}

export interface UserSpaceChangedFileState {
  workspace_id: string;
  generation: number;
  changed_file_paths: string[];
  acknowledged_changed_file_paths: string[];
}

export interface UserSpaceLiveDataConnection {
  component_kind: 'tool_config';
  component_id: string;
  request: Record<string, unknown> | string;
  component_name?: string | null;
  component_type?: string | null;
  refresh_interval_seconds?: number | null;
}

export interface UserSpaceLiveDataCheck {
  component_id: string;
  connection_check_passed: boolean;
  transformation_check_passed: boolean;
  input_row_count?: number | null;
  output_row_count?: number | null;
  note?: string | null;
}

export interface UserSpaceFile {
  path: string;
  content: string;
  artifact_type?: UserSpaceArtifactType | null;
  live_data_connections?: UserSpaceLiveDataConnection[] | null;
  live_data_checks?: UserSpaceLiveDataCheck[] | null;
  updated_at: string;
}

export interface UpsertUserSpaceFileRequest {
  content: string;
  artifact_type?: UserSpaceArtifactType;
  live_data_requested?: boolean;
  live_data_connections?: UserSpaceLiveDataConnection[];
  live_data_checks?: UserSpaceLiveDataCheck[];
}

export interface UserSpaceSnapshot {
  id: string;
  workspace_id: string;
  branch_id: string;
  branch_name: string;
  parent_snapshot_id?: string | null;
  is_current: boolean;
  can_rename: boolean;
  git_commit_hash?: string | null;
  remote_commit_hash?: string | null; // When set, snapshot is backed by remote commit (source of truth)
  message?: string | null;
  created_at: string;
  file_count: number;
}

export type UserSpaceSnapshotDiffStatus = 'A' | 'D' | 'M' | 'R';

export interface UserSpaceSnapshotDiffFileSummary {
  path: string;
  status: UserSpaceSnapshotDiffStatus;
  old_path?: string | null;
  additions: number;
  deletions: number;
  is_binary: boolean;
}

export interface UserSpaceSnapshotDiffSummary {
  workspace_id: string;
  snapshot_id: string;
  snapshot_commit_hash?: string | null;
  files: UserSpaceSnapshotDiffFileSummary[];
  is_snapshot_own_diff?: boolean;
}

export interface UserSpaceSnapshotFileDiff {
  workspace_id: string;
  snapshot_id: string;
  path: string;
  status: UserSpaceSnapshotDiffStatus;
  old_path?: string | null;
  before_path?: string | null;
  after_path?: string | null;
  before_content: string;
  after_content: string;
  additions: number;
  deletions: number;
  is_binary: boolean;
  is_deleted_in_current: boolean;
  is_untracked_in_current: boolean;
  is_snapshot_own_diff?: boolean;
  is_truncated?: boolean;
  message?: string | null;
}

export interface UserSpaceSnapshotBranch {
  id: string;
  workspace_id: string;
  name: string;
  git_ref_name: string;
  base_snapshot_id?: string | null;
  branched_from_snapshot_id?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface UserSpaceSnapshotTimeline {
  workspace_id: string;
  current_snapshot_id?: string | null;
  current_branch_id?: string | null;
  has_previous: boolean;
  has_next: boolean;
  snapshots: UserSpaceSnapshot[];
  branches: UserSpaceSnapshotBranch[];
}

export interface CreateUserSpaceSnapshotRequest {
  message?: string;
}

export interface UpdateUserSpaceSnapshotRequest {
  message: string;
}

export interface SwitchUserSpaceSnapshotBranchRequest {
  branch_id: string;
}

export interface CreateUserSpaceSnapshotBranchRequest {
  name?: string | null;
}

export interface RestoreUserSpaceSnapshotResponse {
  restored_snapshot_id: string;
  file_count: number;
  current_branch_id?: string | null;
  has_previous: boolean;
  has_next: boolean;
}

export interface PaginatedWorkspacesResponse {
  items: UserSpaceWorkspace[];
  total: number;
  offset: number;
  limit: number;
}

export interface ExecuteComponentRequest {
  component_id: string;
  request: Record<string, unknown> | string;
}

export interface ExecuteComponentResponse {
  component_id: string;
  rows: Record<string, unknown>[];
  columns: string[];
  row_count: number;
  error?: string | null;
}

export interface UserSpaceWorkspaceShareLink {
  workspace_id: string;
  share_token: string;
  owner_username: string;
  share_slug: string;
  share_url: string;
}

export interface UserSpaceWorkspaceShareLinkStatus {
  workspace_id: string;
  has_share_link: boolean;
  owner_username: string;
  share_slug: string | null;
  share_token: string | null;
  share_url: string | null;
  created_at: string | null;
  share_access_mode: UserSpaceShareAccessMode;
  selected_user_ids: string[];
  selected_ldap_groups: string[];
  has_password: boolean;
}

export type UserSpaceShareAccessMode =
  | 'token'
  | 'password'
  | 'authenticated_users'
  | 'selected_users'
  | 'ldap_groups';

export interface UpdateUserSpaceWorkspaceShareAccessRequest {
  share_access_mode: UserSpaceShareAccessMode;
  password?: string | null;
  selected_user_ids?: string[];
  selected_ldap_groups?: string[];
}

export interface WorkspaceShareSlugAvailabilityResponse {
  slug: string;
  available: boolean;
}

export interface UserSpaceSharedPreviewResponse {
  workspace_id: string;
  workspace_name: string;
  entry_path: string;
  workspace_files: Record<string, string>;
  live_data_connections?: UserSpaceLiveDataConnection[] | null;
}

export type UserSpaceRuntimeSessionState = 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
export type UserSpaceRuntimeOperationPhase =
  | 'queued'
  | 'provisioning'
  | 'bootstrapping'
  | 'deps_install'
  | 'launching'
  | 'probing'
  | 'ready'
  | 'failed'
  | 'stopped';

export interface UserSpaceRuntimeSession {
  id: string;
  workspace_id: string;
  leased_by_user_id: string;
  state: UserSpaceRuntimeSessionState;
  runtime_provider: string;
  provider_session_id?: string | null;
  preview_internal_url?: string | null;
  created_at: string;
  updated_at: string;
  last_heartbeat_at?: string | null;
  idle_expires_at?: string | null;
  ttl_expires_at?: string | null;
  last_error?: string | null;
}

export interface UserSpaceRuntimeSessionResponse {
  workspace_id: string;
  session?: UserSpaceRuntimeSession | null;
}

export interface UserSpaceRuntimeStatusResponse {
  workspace_id: string;
  session_state: UserSpaceRuntimeSessionState;
  session_id?: string | null;
  devserver_running: boolean;
  devserver_port: number;
  runtime_capabilities?: Record<string, unknown> | null;
  runtime_has_cap_sys_admin?: boolean | null;
  preview_url?: string | null;
  last_error?: string | null;
  runtime_operation_id?: string | null;
  runtime_operation_phase?: UserSpaceRuntimeOperationPhase | null;
  runtime_operation_started_at?: string | null;
  runtime_operation_updated_at?: string | null;
}

export interface UserSpaceWorkspaceTabStateResponse {
  workspace_id: string;
  runtime_status: UserSpaceRuntimeStatusResponse;
  chat_state: WorkspaceChatStateResponse;
}

export interface UserSpaceRuntimeActionResponse {
  workspace_id: string;
  session_id: string;
  state: UserSpaceRuntimeSessionState;
  success: boolean;
  runtime_operation_id?: string | null;
  runtime_operation_phase?: UserSpaceRuntimeOperationPhase | null;
  runtime_operation_started_at?: string | null;
  runtime_operation_updated_at?: string | null;
}

export interface UserSpaceCapabilityTokenResponse {
  token: string;
  expires_at: string;
  workspace_id: string;
  session_id?: string | null;
  capabilities: string[];
}

export interface UserSpaceCollabSnapshotMessage {
  type: 'snapshot';
  workspace_id: string;
  file_path: string;
  version: number;
  content: string;
  read_only: boolean;
}

export interface UserSpaceCollabUpdateMessage {
  type: 'update';
  workspace_id: string;
  file_path: string;
  version: number;
  content: string;
}

export interface UserSpaceCollabAckMessage {
  type: 'ack';
  workspace_id: string;
  file_path: string;
  version: number;
}

export interface UserSpaceCollabErrorMessage {
  type: 'error';
  message: string;
}

export interface UserSpaceCollabPresenceMessage {
  type: 'presence';
  workspace_id: string;
  file_path: string;
  users: Array<{
    user_id: string;
    cursor?: unknown;
    selection?: unknown;
    updated_at?: string;
  }>;
}

export interface UserSpaceCollabFileCreatedMessage {
  type: 'file_created';
  workspace_id: string;
  file_path: string;
  version: number;
}

export interface UserSpaceCollabFileRenamedMessage {
  type: 'file_renamed';
  workspace_id: string;
  old_path: string;
  new_path: string;
}

export type UserSpaceCollabMessage =
  | UserSpaceCollabSnapshotMessage
  | UserSpaceCollabUpdateMessage
  | UserSpaceCollabAckMessage
  | UserSpaceCollabErrorMessage
  | UserSpaceCollabPresenceMessage
  | UserSpaceCollabFileCreatedMessage
  | UserSpaceCollabFileRenamedMessage;

// Retry visualization request/response
export interface RetryVisualizationRequest {
  tool_type: 'datatable' | 'chart';
  source_data: {
    columns?: string[];
    rows?: unknown[][];
    labels?: string[];
    datasets?: unknown[];
    chart_type?: string;
  };
  title?: string;
}

export interface RetryVisualizationResponse {
  success: boolean;
  output?: string;
  error?: string;
}

// Available model for chat
export interface AvailableModel {
  id: string;
  name: string;
  provider: LlmProviderWire;
  context_limit: number;  // Max context window tokens
  max_output_tokens?: number;  // Max output tokens for this model
  group?: string;  // Optional group for UI organization
  is_latest?: boolean; // Whether this model is considered the latest in its group
  capabilities?: string[];
  supported_endpoints?: string[];
  reasoning_supported?: boolean;
  thinking_budget_supported?: boolean;
  effort_levels?: string[];
}

export interface ProviderModelState {
  provider: LlmProviderWire;
  configured: boolean;
  connected: boolean;
  loading: boolean;
  available: boolean;
  error?: string | null;
}

// Response with all available models
export interface AvailableModelsResponse {
  models: AvailableModel[];
  default_model: string | null;
  automatic_default_model?: string | null;
  current_model: string | null;
  allowed_models: string[];  // List of allowed model IDs (for settings UI)
  allowed_openapi_models: string[];  // Separately curated OpenAPI model list
  models_loading?: boolean;
  copilot_refresh_in_progress?: boolean;
  provider_states?: ProviderModelState[];
}

// =============================================================================
// Background Chat Task Types
// =============================================================================
// Health Check / Memory Monitoring Types
// =============================================================================

export interface MemoryStats {
  rss_mb: number;  // Resident Set Size (actual RAM used)
  vms_mb: number;  // Virtual Memory Size
  percent: number;  // Percentage of total system RAM
  available_mb: number;  // Available system RAM
  total_mb: number;  // Total system RAM
}

export interface IndexLoadingDetail {
  name: string;
  status: 'pending' | 'loading' | 'loaded' | 'error';
  type?: 'document' | 'filesystem_faiss' | null;
  size_mb?: number | null;
  chunk_count?: number | null;
  load_time_seconds?: number | null;
  error?: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  indexes_loaded: string[];
  model: string;
  llm_provider: string;
  indexes_ready?: boolean;
  indexes_loading?: boolean;
  indexes_total?: number;
  indexes_loaded_count?: number;
  memory?: MemoryStats;
  index_details?: IndexLoadingDetail[];
  sequential_loading?: boolean;
  loading_index?: string | null;
}

// =============================================================================
// Chat Task Types
// =============================================================================

export type ChatTaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';

export interface ChatTaskStreamingState {
  content: string;
  events: MessageEvent[];
  tool_calls: ToolCallRecord[];
  hit_max_iterations?: boolean;
  version: number;  // Increments on each update for efficient polling
  content_length: number;  // Quick change detection
}

export interface ChatTask {
  id: string;
  conversation_id: string;
  status: ChatTaskStatus;
  user_message: string;
  streaming_state: ChatTaskStreamingState | null;
  response_content: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  last_update_at: string;
}

export interface ConversationTaskStateResponse {
  active_task: ChatTask | null;
  interrupted_task: ChatTask | null;
}

export interface ProviderPromptDebugRecord {
  id: string;
  conversation_id: string;
  chat_task_id?: string | null;
  user_id?: string | null;
  provider: 'openai' | 'anthropic' | 'ollama' | string;
  model: string;
  mode: 'chat' | 'userspace' | string;
  request_kind: 'agent_executor' | 'direct_llm' | string;
  rendered_system_prompt: string;
  rendered_user_input: string;
  rendered_provider_messages: Record<string, unknown>[];
  rendered_chat_history: Record<string, unknown>[];
  tool_scope_prompt: string;
  prompt_additions: string;
  turn_reminders: string;
  debug_metadata?: Record<string, unknown> | null;
  prompt_token_count?: number | null;
  message_index?: number | null;
  created_at: string;
}

export interface ProviderPromptDebugListResponse {
  records: ProviderPromptDebugRecord[];
}
