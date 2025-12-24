/**
 * API Types for Ragtime Indexer
 * These types mirror the Python Pydantic models in app/indexer/models.py
 */

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
