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
    embedding_dimensions: Optional[int] = Field(
        default=None,
        ge=256,
        le=3072,
        description="Target embedding dimensions for OpenAI text-embedding-3-* models (256-3072). Leave empty for model default. Use <=2000 for pgvector indexed search."
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
    allowed_chat_models: List[str] = Field(
        default=[],
        description="List of allowed model IDs for chat. Empty = all models allowed."
    )

    # Agent behavior
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent iterations before stopping tool calls"
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

    # Embedding dimension tracking
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of embeddings in filesystem_embeddings table (set on first index)"
    )
    embedding_config_hash: Optional[str] = Field(
        default=None,
        description="Hash of embedding provider+model to detect configuration changes"
    )

    updated_at: Optional[datetime] = None

    def get_embedding_config_hash(self) -> str:
        """Generate a hash for current embedding provider+model+dimensions configuration."""
        dims = self.embedding_dimensions or "default"
        return f"{self.embedding_provider}:{self.embedding_model}:{dims}"

    def has_embedding_config_changed(self) -> bool:
        """Check if the embedding configuration has changed from what was indexed."""
        if self.embedding_config_hash is None:
            return False  # No previous config, nothing has changed
        return self.get_embedding_config_hash() != self.embedding_config_hash


class EmbeddingStatus(BaseModel):
    """Status of embedding configuration compatibility."""
    current_provider: str
    current_model: str
    current_config_hash: str
    stored_config_hash: Optional[str] = None
    stored_dimension: Optional[int] = None
    has_mismatch: bool = False
    requires_reindex: bool = False
    message: str = ""


class UpdateSettingsRequest(BaseModel):
    """Request to update application settings."""
    # Embedding settings
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = Field(default=None, ge=256, le=3072)
    ollama_protocol: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_port: Optional[int] = Field(default=None, ge=1, le=65535)
    ollama_base_url: Optional[str] = None
    # LLM settings
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    allowed_chat_models: Optional[List[str]] = None
    max_iterations: Optional[int] = Field(default=None, ge=1, le=50)
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
    FILESYSTEM_INDEXER = "filesystem_indexer"


class FilesystemMountType(str, Enum):
    """Mount type for filesystem indexer."""
    DOCKER_VOLUME = "docker_volume"  # Preferred - fastest, requires container config
    SMB = "smb"                       # SMB/CIFS network share
    NFS = "nfs"                       # NFS network mount
    LOCAL = "local"                   # Direct local path


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
    ssh_key_path: str = Field(default="", description="Path to SSH private key (legacy)")
    ssh_key_content: str = Field(default="", description="SSH private key content (preferred)")
    ssh_public_key: str = Field(default="", description="SSH public key (for reference/copying)")
    ssh_key_passphrase: str = Field(default="", description="Passphrase for encrypted SSH key")
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
    key_path: Optional[str] = Field(default=None, description="Path to SSH private key (legacy)")
    key_content: Optional[str] = Field(default=None, description="SSH private key content (preferred)")
    public_key: Optional[str] = Field(default=None, description="SSH public key (for reference/copying)")
    key_passphrase: Optional[str] = Field(default=None, description="Passphrase for encrypted SSH key")
    password: Optional[str] = Field(default=None, description="SSH password (if not using key)")
    command_prefix: str = Field(
        default="",
        description="Command prefix (e.g., 'cd /app && ' or 'source venv/bin/activate && ')"
    )


class FilesystemConnectionConfig(BaseModel):
    """Connection configuration for filesystem indexer tool."""
    # Mount configuration
    mount_type: FilesystemMountType = Field(
        default=FilesystemMountType.DOCKER_VOLUME,
        description="How to access the filesystem (docker_volume recommended)"
    )
    base_path: str = Field(
        description="Base path to index (e.g., /mnt/data for docker volume, //server/share for SMB)"
    )

    # Docker volume settings (when mount_type is docker_volume)
    volume_name: str = Field(
        default="",
        description="Docker volume name (for display purposes - actual mount handled in docker-compose)"
    )

    # SMB settings (when mount_type is smb)
    smb_host: str = Field(default="", description="SMB server hostname/IP")
    smb_share: str = Field(default="", description="SMB share name")
    smb_user: str = Field(default="", description="SMB username")
    smb_password: str = Field(default="", description="SMB password")
    smb_domain: str = Field(default="", description="SMB domain (optional)")

    # NFS settings (when mount_type is nfs)
    nfs_host: str = Field(default="", description="NFS server hostname/IP")
    nfs_export: str = Field(default="", description="NFS export path")
    nfs_options: str = Field(default="ro,noatime", description="NFS mount options")

    # Indexing configuration
    index_name: str = Field(description="Name for this index (used in embeddings table)")
    file_patterns: List[str] = Field(
        default=["**/*.txt", "**/*.md", "**/*.pdf", "**/*.docx", "**/*.py", "**/*.json"],
        description="Glob patterns for files to include"
    )
    exclude_patterns: List[str] = Field(
        default=["**/node_modules/**", "**/__pycache__/**", "**/venv/**", "**/.git/**", "**/.*"],
        description="Glob patterns for files/dirs to exclude"
    )
    recursive: bool = Field(default=True, description="Recursively index subdirectories")
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")

    # Safety limits
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum file size to index (MB)"
    )
    max_total_files: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum total files to index"
    )
    allowed_extensions: List[str] = Field(
        default=[".txt", ".md", ".pdf", ".docx", ".doc", ".py", ".js", ".ts", ".json", ".xml", ".html", ".csv", ".rst"],
        description="Allowed file extensions (security filter)"
    )

    # Re-indexing schedule
    reindex_interval_hours: int = Field(
        default=24,
        ge=0,
        le=8760,  # Max 1 year
        description="Hours between automatic re-indexing (0 = manual only)"
    )
    last_indexed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last completed index"
    )


class FilesystemIndexStatus(str, Enum):
    """Status of a filesystem indexing job."""
    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FilesystemIndexJob(BaseModel):
    """A filesystem indexing job."""
    id: str
    tool_config_id: str
    status: FilesystemIndexStatus = FilesystemIndexStatus.PENDING
    index_name: str

    # Progress tracking
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0  # Unchanged files (incremental)
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_files == 0:
            return 0.0
        return ((self.processed_files + self.skipped_files) / self.total_files) * 100


class FilesystemFileMetadata(BaseModel):
    """Metadata for a file in the filesystem index."""
    id: Optional[str] = None
    index_name: str
    file_path: str  # Relative path within the mount
    file_hash: str  # SHA-256 for change detection
    file_size: int
    mime_type: Optional[str] = None
    chunk_count: int = 0
    last_indexed: datetime = Field(default_factory=datetime.utcnow)


class FilesystemIndexJobResponse(BaseModel):
    """Response for filesystem index job status."""
    id: str
    tool_config_id: str
    status: FilesystemIndexStatus
    index_name: str
    progress_percent: float
    total_files: int
    processed_files: int
    skipped_files: int
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TriggerFilesystemIndexRequest(BaseModel):
    """Request to trigger filesystem indexing for a tool config."""
    full_reindex: bool = Field(
        default=False,
        description="If true, re-index all files regardless of change detection"
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


class PostgresDiscoverRequest(BaseModel):
    """Request to discover databases on a PostgreSQL server."""
    host: str = Field(description="PostgreSQL server hostname or IP")
    port: int = Field(default=5432, description="PostgreSQL server port")
    user: str = Field(description="PostgreSQL username")
    password: str = Field(description="PostgreSQL password")


class PostgresDiscoverResponse(BaseModel):
    """Response from PostgreSQL database discovery."""
    success: bool = Field(description="Whether discovery succeeded")
    databases: List[str] = Field(default_factory=list, description="List of discovered database names")
    error: Optional[str] = Field(default=None, description="Error message if discovery failed")


# -----------------------------------------------------------------------------
# Chat Conversation Models
# -----------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    """Record of a tool call made during message generation."""
    tool: str = Field(description="Name of the tool called")
    input: Optional[dict] = Field(default=None, description="Tool input/query")
    output: Optional[str] = Field(default=None, description="Tool output/result")


class ContentEvent(BaseModel):
    """Content chunk in chronological message events."""
    type: str = Field(default="content", description="Event type")
    content: str = Field(description="Text content")


class ToolCallEvent(BaseModel):
    """Tool call in chronological message events."""
    type: str = Field(default="tool", description="Event type")
    tool: str = Field(description="Name of the tool called")
    input: Optional[dict] = Field(default=None, description="Tool input/query")
    output: Optional[str] = Field(default=None, description="Tool output/result")


class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tool_calls: Optional[List[ToolCallRecord]] = Field(default=None, description="Tool calls made during this message (deprecated, use events)")
    events: Optional[List[dict]] = Field(default=None, description="Chronological events (content and tool calls interleaved)")


class Conversation(BaseModel):
    """A chat conversation with the RAG assistant."""
    id: str
    title: str = Field(default="New Chat")
    model: str = Field(default="gpt-4-turbo")
    user_id: Optional[str] = Field(default=None, description="ID of the conversation owner")
    username: Optional[str] = Field(default=None, description="Username of the conversation owner")
    display_name: Optional[str] = Field(default=None, description="Display name of the conversation owner")
    messages: List[ChatMessage] = Field(default_factory=list)
    total_tokens: int = Field(default=0)
    active_task_id: Optional[str] = Field(default=None, description="ID of currently running background task")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationResponse(BaseModel):
    """Response for conversation data."""
    id: str
    title: str
    model: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    messages: List[ChatMessage]
    total_tokens: int
    active_task_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    title: Optional[str] = Field(default=None, description="Optional title for the conversation")
    model: Optional[str] = Field(default=None, description="Optional model override")


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""
    message: str = Field(description="The message content to send")
    stream: bool = Field(default=False, description="Whether to stream the response")
    background: bool = Field(default=False, description="Whether to run in background mode")


# =============================================================================
# Background Chat Task Models
# =============================================================================

class ChatTaskStatus(str, Enum):
    """Status of a background chat task."""
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ChatTaskStreamingState(BaseModel):
    """Streaming state for a background chat task."""
    content: str = Field(default="", description="Accumulated response content")
    events: List[dict] = Field(default_factory=list, description="Chronological events")
    tool_calls: List[dict] = Field(default_factory=list, description="Tool calls made")
    hit_max_iterations: bool = Field(default=False, description="Whether max iterations was reached")
    version: int = Field(default=0, description="Increments on each update for efficient polling")
    content_length: int = Field(default=0, description="Length of content for quick change detection")


class ChatTask(BaseModel):
    """A background chat task for async message processing."""
    id: str
    conversation_id: str
    status: ChatTaskStatus = Field(default=ChatTaskStatus.pending)
    user_message: str
    streaming_state: Optional[ChatTaskStreamingState] = None
    response_content: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_update_at: datetime = Field(default_factory=datetime.utcnow)


class ChatTaskResponse(BaseModel):
    """Response for a chat task."""
    id: str
    conversation_id: str
    status: ChatTaskStatus
    user_message: str
    streaming_state: Optional[ChatTaskStreamingState] = None
    response_content: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_update_at: datetime
