"""
Indexer data models and schemas.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

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
        description="Description of what this index contains (shown to LLM for context)",
    )
    file_patterns: List[str] = Field(
        default=["**/*"],
        description="Glob patterns for files to include (default: **/* matches all files)",
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns for files/dirs to exclude (e.g. **/node_modules/**, **/__pycache__/**)",
    )
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_file_size_kb: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum file size in KB to include (files larger are skipped). Default 500KB.",
    )
    embedding_model: str = Field(default="text-embedding-3-small")
    enable_ocr: bool = Field(
        default=False,
        description="Enable OCR to extract text from images (slower but captures text in screenshots, scanned docs, etc.)",
    )
    git_clone_timeout_minutes: int = Field(
        default=5,
        ge=1,
        le=480,
        description="Maximum time in minutes to wait for git clone to complete. Default 5 minutes (shallow clone).",
    )


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
    git_token: Optional[str] = Field(
        default=None, exclude=True
    )  # Never persisted/returned

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


class IndexConfigSnapshot(BaseModel):
    """Snapshot of index configuration settings used during indexing."""

    file_patterns: List[str] = Field(default=["**/*"])
    exclude_patterns: List[str] = Field(default_factory=list)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_file_size_kb: int = Field(default=500)
    enable_ocr: bool = Field(default=False)
    git_clone_timeout_minutes: int = Field(default=5)


class IndexInfo(BaseModel):
    """Information about an existing index."""

    name: str
    path: str
    size_mb: float
    document_count: int
    description: str = ""
    enabled: bool = True
    search_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Search weight for result prioritization. Higher values make this index more prominent in aggregated results. Default 1.0 = equal weighting.",
    )
    source_type: str = "upload"  # 'upload' or 'git'
    source: Optional[str] = None  # git URL or original filename
    git_branch: Optional[str] = None  # branch for git sources
    has_stored_token: bool = False  # True if a git token is stored for re-indexing
    config_snapshot: Optional[IndexConfigSnapshot] = (
        None  # Configuration used for indexing
    )
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class CreateIndexRequest(BaseModel):
    """Request to create a new index."""

    name: str = Field(description="Name for the index")
    git_url: Optional[str] = Field(
        default=None, description="Git repository URL to clone"
    )
    git_branch: str = Field(default="main", description="Git branch to use")
    git_token: Optional[str] = Field(
        default=None,
        description="GitHub Personal Access Token for private repos (stored encrypted for re-indexing)",
    )
    config: Optional[IndexConfig] = Field(
        default=None, description="Indexing configuration"
    )


class FileTypeStats(BaseModel):
    """Statistics for a file extension."""

    extension: str = Field(description="File extension (e.g., '.py', '.js')")
    file_count: int = Field(description="Number of files with this extension")
    total_size_bytes: int = Field(description="Total size of files in bytes")
    estimated_chunks: int = Field(
        description="Estimated number of chunks after splitting"
    )
    sample_files: List[str] = Field(
        default_factory=list, description="Sample file paths (up to 5)"
    )


class IndexAnalysisResult(BaseModel):
    """Results from pre-indexing analysis."""

    # Overall stats
    total_files: int = Field(description="Total number of files found")
    total_size_bytes: int = Field(description="Total size in bytes")
    total_size_mb: float = Field(description="Total size in megabytes")
    estimated_chunks: int = Field(
        description="Estimated total chunks (based on chunk_size)"
    )
    estimated_index_size_mb: float = Field(
        description="Estimated FAISS index size in MB"
    )

    # Breakdown by file type
    file_type_stats: List[FileTypeStats] = Field(
        default_factory=list, description="Stats broken down by file extension"
    )

    # Suggested exclusions (patterns that would significantly reduce size)
    suggested_exclusions: List[str] = Field(
        default_factory=list,
        description="Suggested glob patterns to exclude (e.g., minified files, binaries)",
    )

    # Currently matched files (using current patterns)
    matched_file_patterns: List[str] = Field(
        default_factory=list, description="File patterns that matched files"
    )

    # Warnings/recommendations
    warnings: List[str] = Field(
        default_factory=list, description="Warnings about potential issues"
    )

    # Config used for analysis
    chunk_size: int = Field(description="Chunk size used for estimation")
    chunk_overlap: int = Field(description="Chunk overlap used for estimation")


class AnalyzeIndexRequest(BaseModel):
    """Request to analyze a git repository before indexing."""

    git_url: str = Field(description="Git repository URL to analyze")
    git_branch: str = Field(default="main", description="Git branch to analyze")
    git_token: Optional[str] = Field(
        default=None, description="Token for private repos"
    )
    file_patterns: List[str] = Field(
        default=["**/*"],
        description="Glob patterns for files to include (default: **/* matches all files)",
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns for files/dirs to exclude (e.g. **/node_modules/**, **/__pycache__/**)",
    )
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_file_size_kb: int = Field(
        default=500,
        ge=10,
        le=10000,
        description="Maximum file size in KB to include",
    )
    enable_ocr: bool = Field(
        default=False,
        description="Enable OCR to extract text from images",
    )


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
# Git Repository Models
# -----------------------------------------------------------------------------


class CheckRepoVisibilityRequest(BaseModel):
    """Request to check repository visibility."""

    git_url: str = Field(description="Git repository URL to check")
    index_name: Optional[str] = Field(
        default=None,
        description="Index name to check for stored token (if re-indexing)",
    )


class RepoVisibilityResponse(BaseModel):
    """Response for repository visibility check."""

    visibility: str = Field(description="'public', 'private', 'not_found', or 'error'")
    has_stored_token: bool = Field(
        default=False, description="Whether a valid token is stored for this repo"
    )
    needs_token: bool = Field(
        default=False,
        description="Whether user needs to provide a token for access",
    )
    message: str = Field(default="", description="Human-readable status message")


class FetchBranchesRequest(BaseModel):
    """Request to fetch branches from a Git repository."""

    git_url: str = Field(description="Git repository URL")
    git_token: Optional[str] = Field(
        default=None, description="Token for private repos (optional)"
    )
    index_name: Optional[str] = Field(
        default=None,
        description="Index name to use stored token from (optional)",
    )


class FetchBranchesResponse(BaseModel):
    """Response for branch listing."""

    branches: List[str] = Field(
        default_factory=list, description="List of branch names"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    needs_token: bool = Field(
        default=False, description="Whether authentication is required"
    )


# -----------------------------------------------------------------------------
# Application Settings
# -----------------------------------------------------------------------------


class AppSettings(BaseModel):
    """Application settings stored in database."""

    id: str = "default"

    # Server branding
    server_name: str = Field(
        default="Ragtime",
        description="Server display name (shown in UI and API model name)",
    )

    # Embedding Configuration (for FAISS indexing)
    embedding_provider: str = Field(
        default="ollama", description="Embedding provider: 'ollama' or 'openai'"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model name (e.g., 'nomic-embed-text' for Ollama, 'text-embedding-3-small' for OpenAI)",
    )
    embedding_dimensions: Optional[int] = Field(
        default=None,
        ge=256,
        le=3072,
        description="Target embedding dimensions for OpenAI text-embedding-3-* models (256-3072). Leave empty for model default. Use <=2000 for pgvector indexed search.",
    )
    # Ollama connection settings (separate fields for UI)
    ollama_protocol: str = Field(
        default="http", description="Ollama server protocol: 'http' or 'https'"
    )
    ollama_host: str = Field(
        default="localhost", description="Ollama server hostname or IP address"
    )
    ollama_port: int = Field(
        default=11434, ge=1, le=65535, description="Ollama server port"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL (computed from protocol/host/port)",
    )

    # LLM Configuration (for chat/RAG responses)
    llm_provider: str = Field(
        default="openai", description="LLM provider: 'openai' or 'anthropic'"
    )
    llm_model: str = Field(
        default="gpt-4-turbo",
        description="LLM model name (e.g., 'gpt-4-turbo' for OpenAI, 'claude-3-sonnet-20240229' for Anthropic)",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (used for LLM and optionally embeddings)",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key (used when llm_provider is 'anthropic')",
    )
    allowed_chat_models: List[str] = Field(
        default=[],
        description="List of allowed model IDs for chat. Empty = all models allowed.",
    )

    # Agent behavior
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum agent iterations before stopping tool calls",
    )

    # Search configuration
    search_results_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results returned per vector search (k). Higher values provide more context but increase response time and token usage.",
    )
    aggregate_search: bool = Field(
        default=True,
        description="If True, provide a single search_knowledge tool that searches all indexes. If False, create separate search_<index_name> tools for granular control.",
    )

    # Legacy tool configuration (deprecated - use ToolConfig)
    enabled_tools: List[str] = Field(
        default=[], description="Legacy: List of enabled tool names (deprecated)"
    )

    # Legacy Odoo tool settings
    odoo_container: str = Field(
        default="odoo-server",
        description="Legacy: Docker container name for Odoo server",
    )

    # Legacy Postgres tool settings (target DB, not ragtime-db)
    postgres_container: str = Field(
        default="odoo-postgres",
        description="Legacy: Docker container name for target PostgreSQL",
    )
    postgres_host: str = Field(
        default="localhost", description="Legacy: PostgreSQL host for tool queries"
    )
    postgres_port: int = Field(
        default=5432, description="Legacy: PostgreSQL port for tool queries"
    )
    postgres_user: str = Field(
        default="odoo", description="Legacy: PostgreSQL user for tool queries"
    )
    postgres_password: str = Field(
        default="", description="Legacy: PostgreSQL password for tool queries"
    )
    postgres_database: str = Field(
        default="odoo", description="Legacy: PostgreSQL database for tool queries"
    )

    # Query limits
    max_query_results: int = Field(
        default=100, ge=1, le=1000, description="Maximum rows returned by queries"
    )
    query_timeout: int = Field(
        default=30, ge=1, le=300, description="Query timeout in seconds"
    )

    # Security
    enable_write_ops: bool = Field(
        default=False, description="Allow write operations (INSERT/UPDATE/DELETE)"
    )

    # Embedding dimension tracking
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of embeddings in filesystem_embeddings table (set on first index)",
    )
    embedding_config_hash: Optional[str] = Field(
        default=None,
        description="Hash of embedding provider+model to detect configuration changes",
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

    # Server branding
    server_name: Optional[str] = None
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
    # Search configuration
    search_results_k: Optional[int] = Field(default=None, ge=1, le=100)
    aggregate_search: Optional[bool] = None


# -----------------------------------------------------------------------------
# Tool Configuration Models
# -----------------------------------------------------------------------------


class ToolType(str, Enum):
    """Type of tool connection."""

    POSTGRES = "postgres"
    MSSQL = "mssql"
    ODOO_SHELL = "odoo_shell"
    SSH_SHELL = "ssh_shell"
    FILESYSTEM_INDEXER = "filesystem_indexer"
    SOLIDWORKS_PDM = "solidworks_pdm"


class FilesystemMountType(str, Enum):
    """Mount type for filesystem indexer."""

    DOCKER_VOLUME = "docker_volume"  # Preferred - fastest, requires container config
    SMB = "smb"  # SMB/CIFS network share
    NFS = "nfs"  # NFS network mount
    LOCAL = "local"  # Direct local path


class PostgresConnectionConfig(BaseModel):
    """Connection configuration for PostgreSQL tool."""

    host: str = Field(
        default="", description="PostgreSQL host (leave empty for Docker)"
    )
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    user: str = Field(default="", description="PostgreSQL username")
    password: str = Field(default="", description="PostgreSQL password")
    database: str = Field(default="", description="Database name")
    container: str = Field(
        default="", description="Docker container name (alternative to host)"
    )
    docker_network: str = Field(
        default="",
        description="Docker network to connect ragtime to the PostgreSQL container's network",
    )


class MssqlConnectionConfig(BaseModel):
    """Connection configuration for MSSQL/SQL Server tool."""

    host: str = Field(description="MSSQL server hostname or IP address")
    port: int = Field(default=1433, ge=1, le=65535, description="MSSQL port")
    user: str = Field(description="MSSQL username (e.g., 'sa' or 'domain\\user')")
    password: str = Field(description="MSSQL password")
    database: str = Field(description="Database name to connect to")


class OdooShellConnectionConfig(BaseModel):
    """Connection configuration for Odoo shell tool."""

    # Connection mode: 'docker' or 'ssh'
    mode: str = Field(
        default="docker",
        description="Connection mode: 'docker' for container or 'ssh' for remote server",
    )
    # Docker mode fields
    container: str = Field(default="", description="Docker container name running Odoo")
    docker_network: str = Field(
        default="",
        description="Docker network to connect ragtime to the Odoo container's network",
    )
    # SSH mode fields
    ssh_host: str = Field(default="", description="SSH host for remote Odoo server")
    ssh_port: int = Field(default=22, ge=1, le=65535, description="SSH port")
    ssh_user: str = Field(default="", description="SSH username")
    ssh_key_path: str = Field(
        default="", description="Path to SSH private key (legacy)"
    )
    ssh_key_content: str = Field(
        default="", description="SSH private key content (preferred)"
    )
    ssh_public_key: str = Field(
        default="", description="SSH public key (for reference/copying)"
    )
    ssh_key_passphrase: str = Field(
        default="", description="Passphrase for encrypted SSH key"
    )
    ssh_password: str = Field(default="", description="SSH password (if not using key)")
    # Odoo-specific fields (used in both modes)
    database: str = Field(default="odoo", description="Odoo database name")
    odoo_bin_path: str = Field(
        default="",
        description="Path to odoo-bin or odoo command (e.g., '/var/odoo/venv/bin/python3 /var/odoo/src/odoo-bin')",
    )
    config_path: str = Field(
        default="", description="Path to odoo.conf (leave empty for defaults)"
    )
    working_directory: str = Field(
        default="", description="Working directory to cd into before running Odoo shell"
    )
    run_as_user: str = Field(
        default="",
        description="User to run Odoo shell as (e.g., 'odoo' for sudo -u odoo)",
    )


class SSHShellConnectionConfig(BaseModel):
    """Connection configuration for SSH shell tool."""

    host: str = Field(description="SSH host")
    port: int = Field(default=22, ge=1, le=65535, description="SSH port")
    user: str = Field(description="SSH username")
    key_path: Optional[str] = Field(
        default=None, description="Path to SSH private key (legacy)"
    )
    key_content: Optional[str] = Field(
        default=None, description="SSH private key content (preferred)"
    )
    public_key: Optional[str] = Field(
        default=None, description="SSH public key (for reference/copying)"
    )
    key_passphrase: Optional[str] = Field(
        default=None, description="Passphrase for encrypted SSH key"
    )
    password: Optional[str] = Field(
        default=None, description="SSH password (if not using key)"
    )
    command_prefix: str = Field(
        default="",
        description="Command prefix (e.g., 'cd /app && ' or 'source venv/bin/activate && ')",
    )


class FilesystemConnectionConfig(BaseModel):
    """Connection configuration for filesystem indexer tool."""

    # Mount configuration
    mount_type: FilesystemMountType = Field(
        default=FilesystemMountType.DOCKER_VOLUME,
        description="How to access the filesystem (docker_volume recommended)",
    )
    base_path: str = Field(
        description="Base path to index (e.g., /mnt/data for docker volume, //server/share for SMB)"
    )

    # Docker volume settings (when mount_type is docker_volume)
    volume_name: str = Field(
        default="",
        description="Docker volume name (for display purposes - actual mount handled in docker-compose)",
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
    index_name: str = Field(
        description="Name for this index (used in embeddings table)"
    )
    file_patterns: List[str] = Field(
        default=["**/*"],
        description="Glob patterns for files to include (default: **/* matches all files)",
    )
    exclude_patterns: List[str] = Field(
        default=[
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.git/**",
        ],
        description="Glob patterns for files/dirs to exclude",
    )
    recursive: bool = Field(
        default=True, description="Recursively index subdirectories"
    )
    chunk_size: int = Field(
        default=1000, ge=100, le=4000, description="Chunk size for text splitting"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks"
    )

    # Safety limits
    max_file_size_mb: int = Field(
        default=10, ge=1, le=100, description="Maximum file size to index (MB)"
    )
    max_total_files: int = Field(
        default=10000, ge=1, le=100000, description="Maximum total files to index"
    )
    enable_ocr: bool = Field(
        default=False,
        description="Enable OCR to extract text from images (slower but captures text in screenshots, scanned docs, etc.)",
    )

    # Re-indexing schedule
    reindex_interval_hours: int = Field(
        default=24,
        ge=0,
        le=8760,  # Max 1 year
        description="Hours between automatic re-indexing (0 = manual only)",
    )
    last_indexed_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last completed index"
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
    cancel_requested: bool = False

    # File collection progress (for slow network filesystems)
    files_scanned: int = 0  # Files found during collection phase
    dirs_scanned: int = 0  # Directories scanned during collection
    current_directory: Optional[str] = None  # Current directory being scanned

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
    cancel_requested: bool = False
    # Collection phase progress (for slow network filesystems)
    files_scanned: int = 0
    dirs_scanned: int = 0
    current_directory: Optional[str] = None
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TriggerFilesystemIndexRequest(BaseModel):
    """Request to trigger filesystem indexing for a tool config."""

    full_reindex: bool = Field(
        default=False,
        description="If true, re-index all files regardless of change detection",
    )


class FilesystemAnalysisStatus(str, Enum):
    """Status of a filesystem analysis job."""

    PENDING = "pending"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class FilesystemAnalysisJob(BaseModel):
    """A filesystem analysis job for progress tracking."""

    id: str
    tool_config_id: str
    status: FilesystemAnalysisStatus = FilesystemAnalysisStatus.PENDING

    # Progress tracking
    files_scanned: int = 0
    total_dirs_to_scan: int = 0
    dirs_scanned: int = 0
    current_directory: str = ""
    error_message: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_dirs_to_scan == 0:
            return 0.0
        return (self.dirs_scanned / self.total_dirs_to_scan) * 100


class FilesystemAnalysisResult(BaseModel):
    """Results from pre-indexing filesystem analysis."""

    # Overall stats
    total_files: int = Field(description="Total number of files found")
    total_size_bytes: int = Field(description="Total size in bytes")
    total_size_mb: float = Field(description="Total size in megabytes")
    estimated_chunks: int = Field(
        description="Estimated total chunks (based on chunk_size)"
    )
    estimated_index_size_mb: float = Field(
        description="Estimated pgvector index size in MB"
    )

    # Breakdown by file type
    file_type_stats: List[FileTypeStats] = Field(
        default_factory=list, description="Stats broken down by file extension"
    )

    # Suggested exclusions
    suggested_exclusions: List[str] = Field(
        default_factory=list,
        description="Suggested glob patterns to add to exclude_patterns",
    )

    # Warnings/recommendations
    warnings: List[str] = Field(
        default_factory=list, description="Warnings about potential issues"
    )

    # Config used for analysis
    chunk_size: int = Field(description="Chunk size used for estimation")
    chunk_overlap: int = Field(description="Chunk overlap used for estimation")

    # Analysis metadata
    analysis_duration_seconds: float = Field(
        default=0.0, description="Time taken to complete analysis"
    )
    directories_scanned: int = Field(
        default=0, description="Number of directories scanned"
    )


class FilesystemAnalysisJobResponse(BaseModel):
    """Response for filesystem analysis job status (polling)."""

    id: str
    tool_config_id: str
    status: FilesystemAnalysisStatus
    progress_percent: float
    files_scanned: int
    dirs_scanned: int
    total_dirs_to_scan: int
    current_directory: str = ""
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[FilesystemAnalysisResult] = None


class ToolConfig(BaseModel):
    """Configuration for a tool instance."""

    id: Optional[str] = None
    name: str = Field(description="User-friendly name for this tool instance")
    tool_type: ToolType = Field(description="Type of tool")
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    description: str = Field(
        default="",
        description="Description for RAG context - this is presented to the model to explain what this tool is for",
    )
    connection_config: dict = Field(
        description="Connection configuration (structure depends on tool_type)"
    )
    max_results: int = Field(
        default=100, ge=1, le=1000, description="Maximum results per query"
    )
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
    description: str = Field(default="", description="Description for RAG context")
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
    databases: List[str] = Field(
        default_factory=list, description="List of discovered database names"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if discovery failed"
    )


class MssqlDiscoverRequest(BaseModel):
    """Request to discover databases on an MSSQL server."""

    host: str = Field(description="SQL Server hostname or IP")
    port: int = Field(default=1433, description="SQL Server port")
    user: str = Field(description="SQL Server username")
    password: str = Field(description="SQL Server password")


class MssqlDiscoverResponse(BaseModel):
    """Response from MSSQL database discovery."""

    success: bool = Field(description="Whether discovery succeeded")
    databases: List[str] = Field(
        default_factory=list, description="List of discovered database names"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if discovery failed"
    )


class PdmDiscoverRequest(BaseModel):
    """Request to discover PDM schema metadata."""

    host: str = Field(description="SQL Server hostname or IP")
    port: int = Field(default=1433, description="SQL Server port")
    user: str = Field(description="SQL Server username")
    password: str = Field(description="SQL Server password")
    database: str = Field(description="PDM database name")


class PdmDiscoverResponse(BaseModel):
    """Response from PDM schema discovery."""

    success: bool = Field(description="Whether discovery succeeded")
    file_extensions: List[str] = Field(
        default_factory=list,
        description="List of file extensions found in the Documents table",
    )
    variable_names: List[str] = Field(
        default_factory=list,
        description="List of variable names from the Variable table",
    )
    document_count: int = Field(
        default=0, description="Total number of documents in the vault"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if discovery failed"
    )


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
    tool_calls: Optional[List[ToolCallRecord]] = Field(
        default=None,
        description="Tool calls made during this message (deprecated, use events)",
    )
    events: Optional[List[dict]] = Field(
        default=None,
        description="Chronological events (content and tool calls interleaved)",
    )


class Conversation(BaseModel):
    """A chat conversation with the RAG assistant."""

    id: str
    title: str = Field(default="New Chat")
    model: str = Field(default="gpt-4-turbo")
    user_id: Optional[str] = Field(
        default=None, description="ID of the conversation owner"
    )
    username: Optional[str] = Field(
        default=None, description="Username of the conversation owner"
    )
    display_name: Optional[str] = Field(
        default=None, description="Display name of the conversation owner"
    )
    messages: List[ChatMessage] = Field(default_factory=list)
    total_tokens: int = Field(default=0)
    active_task_id: Optional[str] = Field(
        default=None, description="ID of currently running background task"
    )
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

    title: Optional[str] = Field(
        default=None, description="Optional title for the conversation"
    )
    model: Optional[str] = Field(default=None, description="Optional model override")


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""

    message: str = Field(description="The message content to send")
    stream: bool = Field(default=False, description="Whether to stream the response")
    background: bool = Field(
        default=False, description="Whether to run in background mode"
    )


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
    hit_max_iterations: bool = Field(
        default=False, description="Whether max iterations was reached"
    )
    version: int = Field(
        default=0, description="Increments on each update for efficient polling"
    )
    content_length: int = Field(
        default=0, description="Length of content for quick change detection"
    )


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


# =============================================================================
# Schema Indexer Models (for SQL database tools)
# =============================================================================


class SchemaIndexStatus(str, Enum):
    """Status of a schema indexing job."""

    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchemaIndexConfig(BaseModel):
    """Configuration for schema indexing on SQL database tools.

    This is stored as part of the connection_config JSON for postgres/mssql tools.
    """

    schema_index_enabled: bool = Field(
        default=False,
        description="Whether to index the database schema for AI-assisted querying",
    )
    schema_index_interval_hours: int = Field(
        default=24,
        ge=0,
        le=8760,  # Max 1 year
        description="Hours between automatic schema re-indexing (0 = manual only)",
    )
    last_schema_indexed_at: Optional[datetime] = Field(
        default=None, description="Timestamp of last completed schema index"
    )
    schema_hash: Optional[str] = Field(
        default=None,
        description="Hash of schema content for change detection",
    )


class SchemaIndexJob(BaseModel):
    """A schema indexing job for SQL database tools."""

    id: str
    tool_config_id: str
    status: SchemaIndexStatus = SchemaIndexStatus.PENDING
    index_name: str

    # Progress tracking
    total_tables: int = 0
    processed_tables: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    cancel_requested: bool = False

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_tables == 0:
            return 0.0
        return (self.processed_tables / self.total_tables) * 100


class SchemaIndexJobResponse(BaseModel):
    """Response for schema index job status."""

    id: str
    tool_config_id: str
    status: SchemaIndexStatus
    index_name: str
    progress_percent: float
    total_tables: int
    processed_tables: int
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None
    cancel_requested: bool = False
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TriggerSchemaIndexRequest(BaseModel):
    """Request to trigger schema indexing for a tool config."""

    full_reindex: bool = Field(
        default=False,
        description="If true, re-index all tables regardless of change detection",
    )


class TableSchemaInfo(BaseModel):
    """Information about a single table's schema."""

    table_schema: str = Field(
        description="Database schema name (e.g., 'public', 'dbo')"
    )
    table_name: str = Field(description="Table name")
    full_name: str = Field(description="Fully qualified name (schema.table)")
    table_type: str = Field(default="TABLE", description="TABLE or VIEW")
    columns: List[dict] = Field(
        default_factory=list,
        description="List of column definitions with name, type, nullable, default",
    )
    primary_key: List[str] = Field(
        default_factory=list, description="List of primary key column names"
    )
    foreign_keys: List[dict] = Field(
        default_factory=list,
        description="List of foreign key relationships",
    )
    indexes: List[dict] = Field(
        default_factory=list, description="List of index definitions"
    )
    row_count_estimate: Optional[int] = Field(
        default=None, description="Estimated row count (if available)"
    )

    def to_embedding_text(self) -> str:
        """Convert table schema to text suitable for embedding.

        Format designed for semantic search - includes all relevant details
        in a structured but natural language format.
        """
        lines = []

        # Header with table type
        table_type = self.table_type.upper()
        lines.append(f"# {table_type}: {self.full_name}")
        if self.row_count_estimate:
            lines.append(f"Estimated rows: ~{self.row_count_estimate:,}")

        # Columns section
        lines.append("\n## Columns:")
        for col in self.columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            default = col.get("default", "")

            col_line = f"  - {col_name}: {col_type} {nullable}"
            if default:
                col_line += f" DEFAULT {default}"
            if col_name in self.primary_key:
                col_line += " [PRIMARY KEY]"
            lines.append(col_line)

        # Primary key section
        if self.primary_key:
            lines.append(f"\n## Primary Key: ({', '.join(self.primary_key)})")

        # Foreign keys section
        if self.foreign_keys:
            lines.append("\n## Foreign Keys:")
            for fk in self.foreign_keys:
                fk_name = fk.get("name", "")
                fk_columns = fk.get("columns", [])
                ref_table = fk.get("references_table", "")
                ref_columns = fk.get("references_columns", [])
                lines.append(
                    f"  - {fk_name}: ({', '.join(fk_columns)}) -> "
                    f"{ref_table}({', '.join(ref_columns)})"
                )

        # Indexes section
        if self.indexes:
            lines.append("\n## Indexes:")
            for idx in self.indexes:
                idx_name = idx.get("name", "")
                idx_columns = idx.get("columns", [])
                is_unique = "UNIQUE " if idx.get("unique", False) else ""
                lines.append(f"  - {is_unique}{idx_name}: ({', '.join(idx_columns)})")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# SolidWorks PDM Indexer Models
# -----------------------------------------------------------------------------


class PdmIndexStatus(str, Enum):
    """Status of a PDM indexing job."""

    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PdmIndexJob(BaseModel):
    """A PDM metadata indexing job for SolidWorks PDM tools."""

    id: str
    tool_config_id: str
    status: PdmIndexStatus = PdmIndexStatus.PENDING
    index_name: str
    current_step: Optional[str] = None  # Current processing step for UI display

    # Progress tracking
    total_documents: int = 0
    processed_documents: int = 0
    skipped_documents: int = 0  # Unchanged documents
    extracted_documents: int = 0  # Documents extracted from PDM
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    cancel_requested: bool = False

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def progress_percent(self) -> float:
        if self.total_documents == 0:
            return 0.0
        return (
            (self.processed_documents + self.skipped_documents) / self.total_documents
        ) * 100


class PdmIndexJobResponse(BaseModel):
    """Response for PDM index job status."""

    id: str
    tool_config_id: str
    status: PdmIndexStatus
    index_name: str
    current_step: Optional[str] = None
    progress_percent: float
    total_documents: int
    processed_documents: int
    skipped_documents: int
    extracted_documents: int = 0
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None
    cancel_requested: bool = False
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TriggerPdmIndexRequest(BaseModel):
    """Request to trigger PDM indexing for a tool config."""

    full_reindex: bool = Field(
        default=False,
        description="If true, re-index all documents regardless of change detection",
    )


class PdmDocumentInfo(BaseModel):
    """Information about a single PDM document."""

    document_id: int = Field(description="PDM DocumentID from SQL Server")
    filename: str = Field(description="Document filename (e.g., SW-24779.SLDPRT)")
    document_type: str = Field(
        description="File extension type (SLDPRT, SLDASM, SLDDRW)"
    )
    folder_path: Optional[str] = Field(
        default=None, description="Folder path in PDM vault"
    )
    revision_no: int = Field(default=1, description="Current revision number")

    # Variables from PDM
    part_number: Optional[str] = Field(default=None, description="Part number variable")
    description: Optional[str] = Field(default=None, description="Description variable")
    material: Optional[str] = Field(default=None, description="Material variable")
    author: Optional[str] = Field(default=None, description="Author/creator")
    stocked_status: Optional[str] = Field(
        default=None, description="Stocked status (STOCKED, BUILT, etc.)"
    )

    # Additional variables as dict
    variables: dict = Field(
        default_factory=dict, description="All extracted PDM variables"
    )

    # BOM data (for assemblies)
    bom_components: List[dict] = Field(
        default_factory=list, description="List of BOM child components"
    )

    # Configurations
    configurations: List[dict] = Field(
        default_factory=list,
        description="List of configurations with their part numbers",
    )

    def to_embedding_text(self) -> str:
        """Convert PDM document to text suitable for embedding.

        Format designed for semantic search - includes all relevant details
        in a structured but natural language format.
        """
        lines = []

        # Header with document type
        doc_type_map = {
            "SLDPRT": "PART",
            "SLDASM": "ASSEMBLY",
            "SLDDRW": "DRAWING",
        }
        display_type = doc_type_map.get(self.document_type.upper(), self.document_type)
        lines.append(f"# {display_type}: {self.filename}")

        # Identification section
        lines.append("\n## Identification")
        if self.part_number:
            lines.append(f"- Part Number: {self.part_number}")
        lines.append(f"- Filename: {self.filename}")
        if self.folder_path:
            lines.append(f"- Folder: {self.folder_path}")
        lines.append(f"- Revision: {self.revision_no}")

        # Properties section
        if self.description or self.material or self.stocked_status or self.author:
            lines.append("\n## Properties")
            if self.description:
                lines.append(f"- Description: {self.description}")
            if self.material:
                lines.append(f"- Material: {self.material}")
            if self.stocked_status:
                lines.append(f"- Stocked Status: {self.stocked_status}")
            if self.author:
                lines.append(f"- Author: {self.author}")

        # Additional variables
        extra_vars = {
            k: v
            for k, v in self.variables.items()
            if k
            not in (
                "Part Number",
                "Description",
                "Material",
                "Stocked Status",
                "Author",
            )
            and v
        }
        if extra_vars:
            lines.append("\n## Additional Properties")
            for name, value in sorted(extra_vars.items()):
                lines.append(f"- {name}: {value}")

        # Configurations
        if self.configurations:
            lines.append("\n## Configurations")
            for config in self.configurations:
                config_name = config.get("name", "Default")
                config_pn = config.get("part_number", "")
                config_desc = config.get("description", "")
                config_line = f"- {config_name}"
                if config_pn:
                    config_line += f": Part Number {config_pn}"
                if config_desc:
                    config_line += f', Description "{config_desc}"'
                lines.append(config_line)

        # BOM Components (for assemblies)
        if self.bom_components:
            lines.append("\n## BOM Components")
            for i, comp in enumerate(self.bom_components[:50], 1):  # Limit to 50
                comp_name = comp.get("filename", "")
                comp_pn = comp.get("part_number", "")
                comp_qty = comp.get("quantity", 1)
                comp_config = comp.get("configuration", "")
                comp_line = f"{i}. {comp_name}"
                if comp_config:
                    comp_line += f" (Config: {comp_config})"
                if comp_pn:
                    comp_line += f" - PN: {comp_pn}"
                comp_line += f" - Qty: {comp_qty}"
                lines.append(comp_line)
            if len(self.bom_components) > 50:
                lines.append(f"... and {len(self.bom_components) - 50} more components")

        return "\n".join(lines)

    def compute_metadata_hash(self) -> str:
        """Compute a hash of the document metadata for change detection."""
        import hashlib
        import json

        data = {
            "filename": self.filename,
            "revision_no": self.revision_no,
            "variables": self.variables,
            "configurations": self.configurations,
            "bom_components": self.bom_components,
        }
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


class SolidworksPdmConnectionConfig(BaseModel):
    """Connection configuration for SolidWorks PDM indexer tool."""

    # MSSQL Connection (same as MssqlConnectionConfig)
    host: str = Field(description="PDM SQL Server hostname or IP")
    port: int = Field(default=1433, ge=1, le=65535, description="SQL Server port")
    user: str = Field(description="SQL Server username (readonly recommended)")
    password: str = Field(default="", description="SQL Server password")
    database: str = Field(description="PDM database name (e.g., 'HAM-PDM')")

    # Document filtering
    file_extensions: List[str] = Field(
        default=["SLDPRT", "SLDASM", "SLDDRW"],
        description="File extensions to index (without dot)",
    )
    exclude_deleted: bool = Field(default=True, description="Exclude deleted documents")

    # Metadata extraction
    variable_names: List[str] = Field(
        default=[
            "Part Number",
            "Description",
            "Material",
            "Author",
            "Stocked Status",
            "Finish",
            "Weight",
            "Cost",
        ],
        description="PDM variable names to extract and index",
    )
    include_bom: bool = Field(
        default=True, description="Include BOM relationships for assemblies"
    )
    include_folder_path: bool = Field(
        default=True, description="Include folder path in indexed content"
    )
    include_configurations: bool = Field(
        default=True, description="Include configuration data"
    )

    # Indexing options
    max_documents: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of documents to index (for testing, None = all)",
    )


class PdmIndexStatusResponse(BaseModel):
    """Response for PDM index status."""

    enabled: bool = Field(description="Whether PDM indexing is configured")
    last_indexed: Optional[str] = Field(
        default=None, description="Timestamp of last successful index"
    )
    document_count: int = Field(default=0, description="Number of indexed documents")
    embedding_count: int = Field(default=0, description="Number of embeddings stored")
    current_job: Optional[PdmIndexJobResponse] = Field(
        default=None, description="Currently active indexing job"
    )
