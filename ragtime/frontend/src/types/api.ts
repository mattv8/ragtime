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

export interface AuthStatus {
  authenticated: boolean;
  ldap_configured: boolean;
  local_admin_enabled: boolean;
}

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

export type IndexStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface IndexConfig {
  name: string;
  description?: string;
  file_patterns: string[];
  exclude_patterns: string[];
  chunk_size: number;
  chunk_overlap: number;
  embedding_model?: string;
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
}

export interface IndexInfo {
  name: string;
  path: string;
  size_mb: number;
  document_count: number;
  description: string;
  enabled: boolean;
  created_at: string | null;
  last_modified: string | null;
}

export interface CreateIndexRequest {
  name: string;
  git_url?: string;
  git_branch?: string;
  config?: Partial<IndexConfig>;
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
export interface AppSettings {
  id: string;
  // Embedding Configuration (for FAISS indexing)
  embedding_provider: 'ollama' | 'openai';
  embedding_model: string;
  // Ollama connection settings (separate fields)
  ollama_protocol: 'http' | 'https';
  ollama_host: string;
  ollama_port: number;
  ollama_base_url: string;
  // LLM Configuration (for chat/RAG responses)
  llm_provider: 'openai' | 'anthropic';
  llm_model: string;
  openai_api_key: string;
  anthropic_api_key: string;
  allowed_chat_models: string[];
  max_iterations: number;
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
  updated_at: string | null;
}

export interface UpdateSettingsRequest {
  // Embedding settings
  embedding_provider?: 'ollama' | 'openai';
  embedding_model?: string;
  ollama_protocol?: 'http' | 'https';
  ollama_host?: string;
  ollama_port?: number;
  ollama_base_url?: string;
  // LLM settings
  llm_provider?: 'openai' | 'anthropic';
  llm_model?: string;
  openai_api_key?: string;
  anthropic_api_key?: string;
  allowed_chat_models?: string[];
  max_iterations?: number;
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
}

// Ollama Connection Testing
export interface OllamaTestRequest {
  protocol: 'http' | 'https';
  host: string;
  port: number;
}

export interface OllamaModel {
  name: string;
  modified_at?: string;
  size?: number;
}

export interface OllamaTestResponse {
  success: boolean;
  message: string;
  models: OllamaModel[];
  base_url: string;
}

// LLM Provider Model Fetching
export interface LLMModelsRequest {
  provider: 'openai' | 'anthropic';
  api_key: string;
}

export interface LLMModel {
  id: string;
  name: string;
  created?: number;
}

export interface LLMModelsResponse {
  success: boolean;
  message: string;
  models: LLMModel[];
  default_model?: string;
}

// Tool Configuration Types
export type ToolType = 'postgres' | 'odoo_shell' | 'ssh_shell';

export interface PostgresConnectionConfig {
  host?: string;
  port?: number;
  user?: string;
  password?: string;
  database?: string;
  container?: string;
  docker_network?: string;
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
}

export type ConnectionConfig = PostgresConnectionConfig | OdooShellConnectionConfig | SSHShellConnectionConfig;

export interface ToolConfig {
  id: string;
  name: string;
  tool_type: ToolType;
  enabled: boolean;
  description: string;
  connection_config: ConnectionConfig;
  max_results: number;
  timeout: number;
  allow_write: boolean;
  last_test_at: string | null;
  last_test_result: boolean | null;
  last_test_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateToolConfigRequest {
  name: string;
  tool_type: ToolType;
  description?: string;
  connection_config: ConnectionConfig;
  max_results?: number;
  timeout?: number;
  allow_write?: boolean;
}

export interface UpdateToolConfigRequest {
  name?: string;
  enabled?: boolean;
  description?: string;
  connection_config?: ConnectionConfig;
  max_results?: number;
  timeout?: number;
  allow_write?: boolean;
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

// PostgreSQL Database Discovery
export interface PostgresDiscoverRequest {
  host: string;
  port: number;
  user: string;
  password: string;
}

export interface PostgresDiscoverResponse {
  success: boolean;
  databases: string[];
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
export const TOOL_TYPE_INFO: Record<ToolType, { name: string; description: string; icon: string }> = {
  postgres: {
    name: 'PostgreSQL Database',
    description: 'Connect to a PostgreSQL database to execute SQL queries',
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
};

// =============================================================================
// Chat Types
// =============================================================================

export interface ToolCallRecord {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
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
}

export type MessageEvent = ContentEvent | ToolCallEvent;

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  tool_calls?: ToolCallRecord[];  // Deprecated, for backward compatibility
  events?: MessageEvent[];        // Preferred: chronological events
}

// Streaming event types
export type StreamEventType = 'content' | 'tool_start' | 'tool_end' | 'error';

export interface ToolCallInfo {
  tool: string;
  input?: Record<string, unknown>;
  output?: string;
}

export interface StreamEvent {
  type: StreamEventType;
  content?: string;         // For 'content' events
  toolCall?: ToolCallInfo;  // For 'tool_start' and 'tool_end' events
  error?: string;          // For 'error' events
}

export interface Conversation {
  id: string;
  title: string;
  model: string;
  user_id?: string;
  username?: string;
  display_name?: string;
  messages: ChatMessage[];
  total_tokens: number;
  active_task_id: string | null;  // ID of currently running background task
  created_at: string;
  updated_at: string;
}

export interface CreateConversationRequest {
  title?: string;
  model?: string;
}

export interface SendMessageRequest {
  message: string;
  stream?: boolean;
}

export interface SendMessageResponse {
  message: ChatMessage;
  conversation: Conversation;
}

// Available model for chat
export interface AvailableModel {
  id: string;
  name: string;
  provider: 'openai' | 'anthropic';
}

// Response with all available models
export interface AvailableModelsResponse {
  models: AvailableModel[];
  default_model: string | null;
  current_model: string | null;
  allowed_models: string[];  // List of allowed model IDs (for settings UI)
}

// =============================================================================
// Background Chat Task Types
// =============================================================================

export type ChatTaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface ChatTaskStreamingState {
  content: string;
  events: MessageEvent[];
  tool_calls: ToolCallRecord[];
  hit_max_iterations?: boolean;
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

// Model context limits (approximate token counts)
export const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  'gpt-4-turbo': 128000,
  'gpt-4': 8192,
  'gpt-4-32k': 32768,
  'gpt-3.5-turbo': 16385,
  'gpt-3.5-turbo-16k': 16385,
  'claude-3-opus-20240229': 200000,
  'claude-3-sonnet-20240229': 200000,
  'claude-3-haiku-20240307': 200000,
  'claude-3-5-sonnet-20241022': 200000,
  // Ollama models (typical defaults)
  'llama2': 4096,
  'llama3': 8192,
  'llama3.1': 128000,
  'mistral': 8192,
  'mixtral': 32768,
  'codellama': 16384,
  'qwen2.5': 32768,
};
