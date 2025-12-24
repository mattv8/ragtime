"""
Indexer data models and schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class IndexStatus(str, Enum):
    """Status of an indexing job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexConfig(BaseModel):
    """Configuration for creating an index."""
    name: str = Field(description="Name for the index (used as directory name)")
    description: str = Field(
        default="",
        description="Description of what this index contains (shown to LLM for context)"
    )
    file_patterns: List[str] = Field(
        default=["**/*.py", "**/*.md", "**/*.rst", "**/*.txt", "**/*.xml"],
        description="Glob patterns for files to include"
    )
    exclude_patterns: List[str] = Field(
        default=["**/node_modules/**", "**/__pycache__/**", "**/venv/**", "**/.git/**"],
        description="Glob patterns for files/dirs to exclude"
    )
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    embedding_model: str = Field(default="text-embedding-3-small")


class IndexJob(BaseModel):
    """An indexing job."""
    id: str
    name: str
    status: IndexStatus = IndexStatus.PENDING
    config: IndexConfig
    source_type: str = Field(description="'upload' or 'git'")
    source_path: Optional[str] = None
    git_url: Optional[str] = None
    git_branch: str = "main"

    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100


class IndexInfo(BaseModel):
    """Information about an existing index."""
    name: str
    path: str
    size_mb: float
    document_count: int
    description: str = ""
    enabled: bool = True
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class CreateIndexRequest(BaseModel):
    """Request to create a new index."""
    name: str = Field(description="Name for the index")
    git_url: Optional[str] = Field(default=None, description="Git repository URL to clone")
    git_branch: str = Field(default="main", description="Git branch to use")
    config: Optional[IndexConfig] = Field(default=None, description="Indexing configuration")


class IndexJobResponse(BaseModel):
    """Response for index job status."""
    id: str
    name: str
    status: IndexStatus
    progress_percent: float
    total_files: int
    processed_files: int
    total_chunks: int
    processed_chunks: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# -----------------------------------------------------------------------------
# Application Settings
# -----------------------------------------------------------------------------

class AppSettings(BaseModel):
    """Application settings stored in database."""
    id: str = "default"

    # Embedding Configuration (for FAISS indexing)
    embedding_provider: str = Field(
        default="ollama",
        description="Embedding provider: 'ollama' or 'openai'"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name (e.g., 'nomic-embed-text' for Ollama, 'text-embedding-3-small' for OpenAI)"
    )
    # Ollama connection settings (separate fields for UI)
    ollama_protocol: str = Field(
        default="http",
        description="Ollama server protocol: 'http' or 'https'"
    )
    ollama_host: str = Field(
        default="localhost",
        description="Ollama server hostname or IP address"
    )
    ollama_port: int = Field(
        default=11434,
        ge=1,
        le=65535,
        description="Ollama server port"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL (computed from protocol/host/port)"
    )

    # LLM Configuration (for chat/RAG responses)
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' or 'anthropic'"
    )
    llm_model: str = Field(
        default="gpt-4-turbo",
        description="LLM model name (e.g., 'gpt-4-turbo' for OpenAI, 'claude-3-sonnet-20240229' for Anthropic)"
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (used for LLM and optionally embeddings)"
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (used when llm_provider is 'anthropic')"
    )

    # Legacy tool configuration (deprecated - use ToolConfig)
    enabled_tools: List[str] = Field(
        default=[],
        description="Legacy: List of enabled tool names (deprecated)"
    )

    # Legacy Odoo tool settings
    odoo_container: str = Field(
        default="odoo-server",
        description="Legacy: Docker container name for Odoo server"
    )

    # Legacy Postgres tool settings (target DB, not ragtime-db)
    postgres_container: str = Field(
        default="odoo-postgres",
        description="Legacy: Docker container name for target PostgreSQL"
    )
    postgres_host: str = Field(
        default="localhost",
        description="Legacy: PostgreSQL host for tool queries"
    )
    postgres_port: int = Field(
        default=5432,
        description="Legacy: PostgreSQL port for tool queries"
    )
    postgres_user: str = Field(
        default="odoo",
        description="Legacy: PostgreSQL user for tool queries"
    )
    postgres_password: str = Field(
        default="",
        description="Legacy: PostgreSQL password for tool queries"
    )
    postgres_database: str = Field(
        default="odoo",
        description="Legacy: PostgreSQL database for tool queries"
    )

    # Query limits
    max_query_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows returned by queries"
    )
    query_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Query timeout in seconds"
    )

    # Security
    enable_write_ops: bool = Field(
        default=False,
        description="Allow write operations (INSERT/UPDATE/DELETE)"
    )

    updated_at: Optional[datetime] = None


class UpdateSettingsRequest(BaseModel):
    """Request to update application settings."""
    # Embedding settings
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    ollama_protocol: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_port: Optional[int] = Field(default=None, ge=1, le=65535)
    ollama_base_url: Optional[str] = None
    # LLM settings
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    # Legacy tool settings (for backward compatibility)
    enabled_tools: Optional[List[str]] = None
    odoo_container: Optional[str] = None
    postgres_container: Optional[str] = None
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None
    postgres_database: Optional[str] = None
    max_query_results: Optional[int] = Field(default=None, ge=1, le=1000)
    query_timeout: Optional[int] = Field(default=None, ge=1, le=300)
    enable_write_ops: Optional[bool] = None


# -----------------------------------------------------------------------------
# Tool Configuration Models
# -----------------------------------------------------------------------------

class ToolType(str, Enum):
    """Type of tool connection."""
    POSTGRES = "postgres"
    ODOO_SHELL = "odoo_shell"
    SSH_SHELL = "ssh_shell"


class PostgresConnectionConfig(BaseModel):
    """Connection configuration for PostgreSQL tool."""
    host: str = Field(default="", description="PostgreSQL host (leave empty for Docker)")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    user: str = Field(default="", description="PostgreSQL username")
    password: str = Field(default="", description="PostgreSQL password")
    database: str = Field(default="", description="Database name")
    container: str = Field(default="", description="Docker container name (alternative to host)")
    docker_network: str = Field(
        default="",
        description="Docker network to connect ragtime to the PostgreSQL container's network"
    )


class OdooShellConnectionConfig(BaseModel):
    """Connection configuration for Odoo shell tool."""
    # Connection mode: 'docker' or 'ssh'
    mode: str = Field(
        default="docker",
        description="Connection mode: 'docker' for container or 'ssh' for remote server"
    )
    # Docker mode fields
    container: str = Field(default="", description="Docker container name running Odoo")
    docker_network: str = Field(
        default="",
        description="Docker network to connect ragtime to the Odoo container's network"
    )
    # SSH mode fields
    ssh_host: str = Field(default="", description="SSH host for remote Odoo server")
    ssh_port: int = Field(default=22, ge=1, le=65535, description="SSH port")
    ssh_user: str = Field(default="", description="SSH username")
    ssh_key_path: str = Field(default="", description="Path to SSH private key")
    ssh_password: str = Field(default="", description="SSH password (if not using key)")
    # Odoo-specific fields (used in both modes)
    database: str = Field(default="odoo", description="Odoo database name")
    odoo_bin_path: str = Field(
        default="",
        description="Path to odoo-bin or odoo command (e.g., '/var/odoo/venv/bin/python3 /var/odoo/src/odoo-bin')"
    )
    config_path: str = Field(
        default="",
        description="Path to odoo.conf (leave empty for defaults)"
    )
    working_directory: str = Field(
        default="",
        description="Working directory to cd into before running Odoo shell"
    )
    run_as_user: str = Field(
        default="",
        description="User to run Odoo shell as (e.g., 'odoo' for sudo -u odoo)"
    )


class SSHShellConnectionConfig(BaseModel):
    """Connection configuration for SSH shell tool."""
    host: str = Field(description="SSH host")
    port: int = Field(default=22, ge=1, le=65535, description="SSH port")
    user: str = Field(description="SSH username")
    key_path: Optional[str] = Field(default=None, description="Path to SSH private key")
    password: Optional[str] = Field(default=None, description="SSH password (if not using key)")
    command_prefix: str = Field(
        default="",
        description="Command prefix (e.g., 'cd /app && ' or 'source venv/bin/activate && ')"
    )


class ToolConfig(BaseModel):
    """Configuration for a tool instance."""
    id: Optional[str] = None
    name: str = Field(description="User-friendly name for this tool instance")
    tool_type: ToolType = Field(description="Type of tool")
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    description: str = Field(
        default="",
        description="Description for RAG context - this is presented to the model to explain what this tool is for"
    )
    connection_config: dict = Field(
        description="Connection configuration (structure depends on tool_type)"
    )
    max_results: int = Field(default=100, ge=1, le=1000, description="Maximum results per query")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    allow_write: bool = Field(default=False, description="Allow write operations")

    # Test results
    last_test_at: Optional[datetime] = None
    last_test_result: Optional[bool] = None
    last_test_error: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CreateToolConfigRequest(BaseModel):
    """Request to create a new tool configuration."""
    name: str = Field(description="User-friendly name for this tool instance")
    tool_type: ToolType = Field(description="Type of tool")
    description: str = Field(
        default="",
        description="Description for RAG context"
    )
    connection_config: dict = Field(description="Connection configuration")
    max_results: int = Field(default=100, ge=1, le=1000)
    timeout: int = Field(default=30, ge=1, le=300)
    allow_write: bool = Field(default=False)


class UpdateToolConfigRequest(BaseModel):
    """Request to update an existing tool configuration."""
    name: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    connection_config: Optional[dict] = None
    max_results: Optional[int] = Field(default=None, ge=1, le=1000)
    timeout: Optional[int] = Field(default=None, ge=1, le=300)
    allow_write: Optional[bool] = None


class ToolTestRequest(BaseModel):
    """Request to test a tool connection."""
    tool_type: ToolType = Field(description="Type of tool to test")
    connection_config: dict = Field(description="Connection configuration to test")
