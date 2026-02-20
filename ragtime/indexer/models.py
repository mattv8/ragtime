"""
Indexer data models and schemas.
"""


import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ragtime.core.embedding_models import (
    get_embedding_models,
    get_model_dimensions_sync,
)

class IndexStatus(str, Enum):
    """Status of an indexing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolOutputMode(str, Enum):
    """Tool output mode for conversations."""

    DEFAULT = "default"  # Use global setting
    SHOW = "show"  # Always show tool output
    HIDE = "hide"  # Always hide tool output
    AUTO = "auto"  # Let AI decide based on relevance


class OcrMode(str, Enum):
    """OCR mode for extracting text from images."""

    DISABLED = "disabled"  # No OCR - skip image files
    TESSERACT = "tesseract"  # Traditional OCR with Tesseract (fast, basic)
    OLLAMA = (
        "ollama"  # Semantic OCR with Ollama vision model (slower, better understanding)
    )


class VectorStoreType(str, Enum):
    """Vector store backend for indexes (both document and filesystem)."""

    PGVECTOR = "pgvector"  # PostgreSQL pgvector - persistent, scalable
    FAISS = "faiss"  # FAISS - in-memory, loaded at startup


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
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="Vector store backend: 'faiss' (in-memory, default for document indexes) or 'pgvector' (PostgreSQL, persistent)",
    )
    ocr_mode: OcrMode = Field(
        default=OcrMode.DISABLED,
        description="OCR mode: 'disabled' (skip images), 'tesseract' (fast traditional OCR), or 'ollama' (semantic OCR with vision model)",
    )
    ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Ollama vision model for OCR (e.g., 'granite3.2-vision:2b'). Required when ocr_mode is 'ollama'.",
    )
    git_clone_timeout_minutes: int = Field(
        default=5,
        ge=1,
        le=480,
        description="Maximum time in minutes to wait for git clone to complete. Default 5 minutes (shallow clone).",
    )
    git_history_depth: int = Field(
        default=1,
        ge=0,
        description="Git clone depth. 1=latest commit only (fastest), 0=full history (slowest). Values >1 specify number of commits.",
    )
    reindex_interval_hours: int = Field(
        default=0,
        ge=0,
        le=8760,  # Max 1 year
        description="Hours between automatic pull & re-index (0 = manual only). Only applies to git-based indexes.",
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
        """Calculate overall progress: 30% for file loading, 70% for embedding."""
        file_progress = (
            (self.processed_files / self.total_files) * 30
            if self.total_files > 0
            else 0.0
        )
        chunk_progress = (
            (self.processed_chunks / self.total_chunks) * 70
            if self.total_chunks > 0
            else 0.0
        )
        return min(file_progress + chunk_progress, 99.0)  # Cap at 99 until completed


class IndexConfigSnapshot(BaseModel):
    """Snapshot of index configuration settings used during indexing."""

    file_patterns: List[str] = Field(default=["**/*"])
    exclude_patterns: List[str] = Field(default_factory=list)
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    max_file_size_kb: int = Field(default=500)
    ocr_mode: OcrMode = Field(default=OcrMode.DISABLED)
    ocr_vision_model: Optional[str] = Field(default=None)
    git_clone_timeout_minutes: int = Field(default=5)
    git_history_depth: int = Field(default=1)
    reindex_interval_hours: int = Field(default=0)


class IndexInfo(BaseModel):
    """Information about an existing index."""

    name: str  # Safe tool name (lowercase, alphanumeric with underscores)
    display_name: Optional[str] = None  # Human-readable name for UI display
    path: str
    size_mb: float
    document_count: int
    chunk_count: int = 0  # Number of chunks/vectors for memory calculation
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
    # Git history info (for git-sourced indexes)
    git_repo_size_mb: Optional[float] = None  # Size of .git_repo directory (disk)
    has_git_history: bool = False  # True if .git_repo exists with history
    # Vector store backend
    vector_store_type: VectorStoreType = VectorStoreType.FAISS


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


class CommitHistorySample(BaseModel):
    """A sample point from commit history for depth-to-date interpolation."""

    depth: int = Field(description="Commit depth (0 = most recent)")
    date: str = Field(description="ISO 8601 date of the commit")
    hash: str = Field(description="Short commit hash (7 chars)")


class CommitHistoryInfo(BaseModel):
    """Commit history metadata for estimating depth-to-date mapping."""

    total_commits: int = Field(description="Total number of commits in the repo")
    samples: List[CommitHistorySample] = Field(
        default_factory=list,
        description="Sample commits at various depths for interpolation",
    )
    oldest_date: Optional[str] = Field(
        default=None, description="Date of the oldest commit"
    )
    newest_date: Optional[str] = Field(
        default=None, description="Date of the newest commit"
    )


class MemoryEstimate(BaseModel):
    """Memory estimates for an index."""

    embedding_dimension: int = Field(description="Embedding vector dimension")
    steady_memory_mb: float = Field(
        description="Estimated RAM after loading (steady-state)"
    )
    peak_memory_mb: float = Field(description="Estimated peak RAM during loading")
    # Per-index estimates for comparison table
    dimension_breakdown: Optional[List[dict]] = Field(
        default=None,
        description="Memory estimates at different embedding dimensions",
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

    # Memory estimates
    memory_estimate: Optional[MemoryEstimate] = Field(
        default=None,
        description="RAM requirements for this index",
    )
    total_memory_with_existing_mb: Optional[float] = Field(
        default=None,
        description="Total RAM needed including all existing indexes",
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

    # Git commit history info (for depth-to-date interpolation)
    commit_history: Optional[CommitHistoryInfo] = Field(
        default=None,
        description="Commit history samples for estimating date range at different depths",
    )


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
    ocr_mode: OcrMode = Field(
        default=OcrMode.DISABLED,
        description="OCR mode: 'disabled' (skip images), 'tesseract' (fast traditional OCR), or 'ollama' (semantic OCR with vision model)",
    )
    ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Ollama vision model for OCR (e.g., 'qwen3-vl:latest'). Required when ocr_mode is 'ollama'.",
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
    vector_store_type: VectorStoreType = VectorStoreType.FAISS


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
        description="Ollama server URL for embeddings (computed from protocol/host/port)",
    )

    # LLM Configuration (for chat/RAG responses)
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai', 'anthropic', or 'ollama'",
    )
    llm_model: str = Field(
        default="gpt-4-turbo",
        description="LLM model name (e.g., 'gpt-4-turbo' for OpenAI, 'claude-3-sonnet-20240229' for Anthropic, 'llama3' for Ollama)",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of tokens to generate in LLM response",
    )
    # LLM Ollama connection settings (separate from embedding Ollama)
    llm_ollama_protocol: str = Field(
        default="http", description="Ollama LLM server protocol: 'http' or 'https'"
    )
    llm_ollama_host: str = Field(
        default="localhost",
        description="Ollama LLM server hostname or IP address",
    )
    llm_ollama_port: int = Field(
        default=11434, ge=1, le=65535, description="Ollama LLM server port"
    )
    llm_ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama LLM server URL (computed from protocol/host/port)",
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
        le=100,
        description="Maximum agent iterations before stopping tool calls",
    )

    # Token optimization settings
    max_tool_output_chars: int = Field(
        default=5000,
        ge=0,
        le=100000,
        description="Maximum characters per tool output before truncation. 0=unlimited. Reduces quadratic token growth in multi-tool loops.",
    )
    scratchpad_window_size: int = Field(
        default=6,
        ge=0,
        le=100,
        description="Keep last N tool calls in full detail; older steps are summarized. 0=keep all (no compression).",
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
    search_use_mmr: bool = Field(
        default=True,
        description="Use Max Marginal Relevance for result diversification. Reduces near-duplicate results by balancing relevance with diversity.",
    )
    search_mmr_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MMR diversity/relevance tradeoff. 0=maximum diversity (most varied results), 1=maximum relevance (closest matches). Recommended: 0.5-0.7.",
    )
    context_token_budget: int = Field(
        default=4000,
        ge=0,
        le=32000,
        description="Maximum tokens for retrieved context sent to LLM. 0=unlimited. Prevents context overflow for models with smaller context windows.",
    )
    chunking_use_tokens: bool = Field(
        default=True,
        description="Use token-based chunking instead of character-based. More accurate chunk sizes aligned with model tokenization.",
    )

    # pgvector configuration
    ivfflat_lists: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="IVFFlat index lists parameter. Higher values: slower build but faster queries for large datasets. Scale with sqrt(num_embeddings).",
    )

    # Performance configuration
    sequential_index_loading: bool = Field(
        default=False,
        description="If True, load FAISS indexes one at a time (smallest first) to reduce peak memory. If False (default), load all indexes in parallel for faster startup.",
    )

    # API Tool Output Configuration
    tool_output_mode: ToolOutputMode = Field(
        default=ToolOutputMode.DEFAULT,
        description="Tool output visibility mode: 'default'/'show' (always show), 'hide' (suppress), 'auto' (AI decides). Does not affect MCP or internal chat UI.",
    )

    # MCP Configuration
    mcp_enabled: bool = Field(
        default=False,
        description="If True, enable the MCP (Model Context Protocol) server endpoints.",
    )
    mcp_default_route_auth: bool = Field(
        default=False,
        description="If True, require Bearer token authentication for the default /mcp route.",
    )
    mcp_default_route_auth_method: str = Field(
        default="password",
        description="Authentication method for default MCP route: 'password' for Bearer token, 'oauth2' for LDAP OAuth2 flow.",
    )
    mcp_default_route_password: Optional[str] = Field(
        default=None,
        description="Password for the default /mcp route (decrypted for display). Use this as Bearer token for MCP clients.",
    )
    mcp_default_route_allowed_group: Optional[str] = Field(
        default=None,
        description="LDAP group DN required for OAuth2 access. If empty, any authenticated LDAP user can access.",
    )
    has_mcp_default_password: bool = Field(
        default=False,
        description="Indicates if a password is set for the default MCP route (computed, never stored)",
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

    # OCR Configuration (global defaults for new indexes)
    default_ocr_mode: OcrMode = Field(
        default=OcrMode.DISABLED,
        description="Default OCR mode for new indexes: 'disabled' (skip images), 'tesseract' (fast), 'ollama' (semantic)",
    )
    default_ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Default Ollama vision model for OCR (e.g., 'granite3.2-vision:2b'). Required when ocr_mode is 'ollama'.",
    )
    ocr_concurrency_limit: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max concurrent Ollama vision OCR requests. Higher values use more VRAM.",
    )

    # User Space Snapshot Retention
    snapshot_retention_days: int = Field(
        default=0,
        ge=0,
        description="Snapshot retention in days (0 = unlimited). Snapshots older than this are hidden and cannot be restored.",
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

    async def get_configuration_warnings(
        self, chunk_count: int = 0
    ) -> List["ConfigurationWarning"]:
        """
        Generate warnings about potentially suboptimal configuration.

        Uses LiteLLM model data for intelligent dimension/model detection.

        Args:
            chunk_count: Total number of chunks across all indexes (for ivfflat tuning)

        Returns:
            List of configuration warnings
        """
        warnings: List["ConfigurationWarning"] = []

        # Fetch LiteLLM model data for intelligent checks
        embedding_models = await get_embedding_models()

        # pgvector's IVFFlat maximum dimension limit
        PGVECTOR_IVFFLAT_MAX_DIM = 2000

        # Check actual embedding dimension vs pgvector index limit
        # This is the authoritative check - we know the actual dimension in use
        if (
            self.embedding_dimension
            and self.embedding_dimension > PGVECTOR_IVFFLAT_MAX_DIM
        ):
            warnings.append(
                ConfigurationWarning(
                    level="warning",
                    category="embedding",
                    message=f"Embedding dimension ({self.embedding_dimension}) exceeds "
                    f"pgvector's IVFFlat index limit ({PGVECTOR_IVFFLAT_MAX_DIM}). Using exact search.",
                    recommendation="Consider an embedding model with <=2000 dimensions for faster "
                    "indexed queries, especially for large datasets (>10k chunks).",
                )
            )
        elif not self.embedding_dimension:
            # No embeddings yet - check if the selected model will exceed the limit
            # This provides a heads-up before they start indexing
            model_dims = get_model_dimensions_sync(
                self.embedding_model or "", embedding_models
            )
            if model_dims and model_dims > PGVECTOR_IVFFLAT_MAX_DIM:
                warnings.append(
                    ConfigurationWarning(
                        level="info",
                        category="embedding",
                        message=f"{self.embedding_model} uses {model_dims} dimensions.",
                        recommendation=f"This exceeds pgvector's {PGVECTOR_IVFFLAT_MAX_DIM}-dim "
                        f"IVFFlat limit. Consider a smaller model for large datasets.",
                    )
                )

        # Check ivfflat_lists vs chunk count
        if chunk_count > 0 and self.ivfflat_lists:
            import math

            optimal_lists = int(math.sqrt(chunk_count))
            if self.ivfflat_lists < optimal_lists * 0.5 and chunk_count > 1000:
                warnings.append(
                    ConfigurationWarning(
                        level="info",
                        category="performance",
                        message=f"ivfflat_lists ({self.ivfflat_lists}) may be suboptimal "
                        f"for {chunk_count} chunks.",
                        recommendation=f"Consider increasing to ~{optimal_lists} for "
                        f"faster queries on large datasets.",
                    )
                )
            elif self.ivfflat_lists > optimal_lists * 3 and chunk_count < 10000:
                warnings.append(
                    ConfigurationWarning(
                        level="info",
                        category="performance",
                        message=f"ivfflat_lists ({self.ivfflat_lists}) is high for "
                        f"{chunk_count} chunks.",
                        recommendation=f"Consider reducing to ~{optimal_lists} for faster "
                        f"index building.",
                    )
                )

        # Check if MMR is disabled
        if not self.search_use_mmr:
            warnings.append(
                ConfigurationWarning(
                    level="info",
                    category="retrieval",
                    message="MMR (Max Marginal Relevance) is disabled.",
                    recommendation="Enable MMR (search_use_mmr=true) for more diverse "
                    "search results and better answer quality.",
                )
            )

        return warnings


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


class ConfigurationWarning(BaseModel):
    """A warning about potentially suboptimal configuration."""

    level: Literal["info", "warning", "error"] = Field(
        description="Severity level of the warning"
    )
    category: str = Field(description="Category: embedding, indexing, performance")
    message: str = Field(description="Human-readable warning message")
    recommendation: Optional[str] = Field(
        default=None, description="Suggested action to resolve"
    )


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
    llm_max_tokens: Optional[int] = Field(default=None, ge=1)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    allowed_chat_models: Optional[List[str]] = None
    max_iterations: Optional[int] = Field(default=None, ge=1, le=100)
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
    # Retrieval optimization
    search_use_mmr: Optional[bool] = Field(
        default=None,
        description="Use MMR (Maximal Marginal Relevance) for result diversity.",
    )
    search_mmr_lambda: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="MMR diversity vs relevance tradeoff (0=max diversity, 1=max relevance).",
    )
    context_token_budget: Optional[int] = Field(
        default=None,
        ge=500,
        le=32000,
        description="Max tokens for retrieved context sent to LLM.",
    )
    chunking_use_tokens: Optional[bool] = Field(
        default=None,
        description="Use token-based chunking instead of character-based.",
    )
    ivfflat_lists: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Number of IVFFlat index lists for pgvector (tune for dataset size).",
    )
    # Performance / Memory configuration
    sequential_index_loading: Optional[bool] = None
    # API Tool Output configuration
    tool_output_mode: Optional[str] = Field(
        default=None,
        description="Tool output mode: 'default', 'show', 'hide', or 'auto'.",
    )
    # MCP configuration
    mcp_enabled: Optional[bool] = None
    mcp_default_route_auth: Optional[bool] = None
    mcp_default_route_auth_method: Optional[str] = Field(
        default=None,
        description="Authentication method for default MCP route: 'password' or 'oauth2'.",
    )
    mcp_default_route_password: Optional[str] = Field(
        default=None,
        description="Password for the default /mcp route. Set to empty string to clear.",
    )
    mcp_default_route_allowed_group: Optional[str] = Field(
        default=None,
        description="LDAP group DN required for OAuth2 access. Set to empty string to clear.",
    )
    # OCR configuration
    default_ocr_mode: Optional[str] = Field(
        default=None,
        description="Default OCR mode: 'disabled', 'tesseract', or 'ollama'.",
    )
    default_ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Default Ollama vision model for OCR (e.g., 'granite3.2-vision:2b').",
    )
    ocr_concurrency_limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Max concurrent Ollama vision OCR requests.",
    )
    # User Space
    snapshot_retention_days: Optional[int] = Field(
        default=None,
        ge=0,
        description="Snapshot retention in days (0 = unlimited).",
    )


# -----------------------------------------------------------------------------
# Tool Configuration Models
# -----------------------------------------------------------------------------


class ToolType(str, Enum):
    """Type of tool connection."""

    POSTGRES = "postgres"
    MSSQL = "mssql"
    MYSQL = "mysql"
    ODOO_SHELL = "odoo_shell"
    SSH_SHELL = "ssh_shell"
    FILESYSTEM_INDEXER = "filesystem_indexer"
    SOLIDWORKS_PDM = "solidworks_pdm"


# Tool types that support schema indexing (use for ToolType enum comparisons)
SCHEMA_INDEXER_CAPABLE_TOOL_TYPES = frozenset(
    {ToolType.POSTGRES, ToolType.MSSQL, ToolType.MYSQL}
)


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
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Use SSH tunnel for database connection"
    )
    ssh_tunnel_host: str = Field(
        default="", description="SSH server hostname for tunnel"
    )
    ssh_tunnel_port: int = Field(
        default=22, ge=1, le=65535, description="SSH server port"
    )
    ssh_tunnel_user: str = Field(default="", description="SSH username")
    ssh_tunnel_password: str = Field(
        default="", description="SSH password (if not using key)"
    )
    ssh_tunnel_key_path: str = Field(default="", description="Path to SSH private key")
    ssh_tunnel_key_content: str = Field(
        default="", description="SSH private key content (preferred over path)"
    )
    ssh_tunnel_key_passphrase: str = Field(
        default="", description="Passphrase for encrypted SSH key"
    )
    ssh_tunnel_public_key: str = Field(
        default="", description="SSH public key for generated keypairs"
    )


class MssqlConnectionConfig(BaseModel):
    """Connection configuration for MSSQL/SQL Server tool."""

    host: str = Field(description="MSSQL server hostname or IP address")
    port: int = Field(default=1433, ge=1, le=65535, description="MSSQL port")
    user: str = Field(description="MSSQL username (e.g., 'sa' or 'domain\\user')")
    password: str = Field(description="MSSQL password")
    database: str = Field(description="Database name to connect to")
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Use SSH tunnel for database connection"
    )
    ssh_tunnel_host: str = Field(
        default="", description="SSH server hostname for tunnel"
    )
    ssh_tunnel_port: int = Field(
        default=22, ge=1, le=65535, description="SSH server port"
    )
    ssh_tunnel_user: str = Field(default="", description="SSH username")
    ssh_tunnel_password: str = Field(
        default="", description="SSH password (if not using key)"
    )
    ssh_tunnel_key_path: str = Field(default="", description="Path to SSH private key")
    ssh_tunnel_key_content: str = Field(
        default="", description="SSH private key content (preferred over path)"
    )
    ssh_tunnel_key_passphrase: str = Field(
        default="", description="Passphrase for encrypted SSH key"
    )
    ssh_tunnel_public_key: str = Field(
        default="", description="SSH public key for generated keypairs"
    )


class MysqlConnectionConfig(BaseModel):
    """Connection configuration for MySQL/MariaDB tool."""

    host: str = Field(
        default="", description="MySQL/MariaDB server hostname or IP address"
    )
    port: int = Field(default=3306, ge=1, le=65535, description="MySQL port")
    user: str = Field(default="", description="MySQL username")
    password: str = Field(default="", description="MySQL password")
    database: str = Field(default="", description="Database name to connect to")
    # Docker container mode (alternative to direct connection)
    container: str = Field(
        default="", description="Docker container name (alternative to host)"
    )
    docker_network: str = Field(
        default="",
        description="Docker network to connect ragtime to the MySQL container's network",
    )
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Use SSH tunnel for database connection"
    )
    ssh_tunnel_host: str = Field(
        default="", description="SSH server hostname for tunnel"
    )
    ssh_tunnel_port: int = Field(
        default=22, ge=1, le=65535, description="SSH server port"
    )
    ssh_tunnel_user: str = Field(default="", description="SSH username")
    ssh_tunnel_password: str = Field(
        default="", description="SSH password (if not using key)"
    )
    ssh_tunnel_key_path: str = Field(default="", description="Path to SSH private key")
    ssh_tunnel_key_content: str = Field(
        default="", description="SSH private key content (preferred over path)"
    )
    ssh_tunnel_key_passphrase: str = Field(
        default="", description="Passphrase for encrypted SSH key"
    )
    ssh_tunnel_public_key: str = Field(
        default="", description="SSH public key for generated keypairs"
    )
    # Schema indexing options
    schema_index_enabled: bool = Field(
        default=False, description="Enable automatic schema indexing"
    )
    schema_index_interval_hours: int = Field(
        default=24, ge=1, le=168, description="Re-index schema every N hours"
    )
    last_schema_indexed_at: Optional[str] = Field(
        default=None, description="Timestamp of last schema indexing"
    )
    schema_hash: Optional[str] = Field(
        default=None, description="Hash of last indexed schema"
    )


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

    # Vector store backend selection
    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.PGVECTOR,
        description="Vector store backend: 'pgvector' (PostgreSQL, persistent) or 'faiss' (in-memory, loaded at startup)",
    )

    # Safety limits
    max_file_size_mb: int = Field(
        default=10, ge=1, le=100, description="Maximum file size to index (MB)"
    )
    max_total_files: int = Field(
        default=10000, ge=1, le=100000, description="Maximum total files to index"
    )
    ocr_mode: OcrMode = Field(
        default=OcrMode.DISABLED,
        description="OCR mode: 'disabled' (skip images), 'tesseract' (fast traditional OCR), or 'ollama' (semantic OCR with vision model)",
    )
    ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Ollama vision model for OCR (e.g., 'qwen3-vl:latest'). Required when ocr_mode is 'ollama'.",
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
    timeout: int = Field(
        default=30,
        ge=0,
        le=86400,
        description="Default timeout in seconds (0 = no timeout)",
    )
    timeout_max_seconds: int = Field(
        default=300,
        ge=0,
        le=86400,
        description="Maximum timeout user/agent can choose (0 = unlimited)",
    )
    allow_write: bool = Field(default=False, description="Allow write operations")

    # Transient runtime status (not persisted)
    disabled_reason: Optional[str] = Field(
        default=None, description="Reason why the tool is disabled (runtime check)"
    )

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
    timeout: int = Field(default=30, ge=0, le=86400)
    timeout_max_seconds: int = Field(default=300, ge=0, le=86400)
    allow_write: bool = Field(default=False)


class UpdateToolConfigRequest(BaseModel):
    """Request to update an existing tool configuration."""

    name: Optional[str] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None
    connection_config: Optional[dict] = None
    max_results: Optional[int] = Field(default=None, ge=1, le=1000)
    timeout: Optional[int] = Field(default=None, ge=0, le=86400)
    timeout_max_seconds: Optional[int] = Field(default=None, ge=0, le=86400)
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
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Whether to use SSH tunnel"
    )
    ssh_tunnel_host: Optional[str] = Field(
        default=None, description="SSH server hostname"
    )
    ssh_tunnel_port: int = Field(default=22, description="SSH server port")
    ssh_tunnel_user: Optional[str] = Field(default=None, description="SSH username")
    ssh_tunnel_password: Optional[str] = Field(default=None, description="SSH password")
    ssh_tunnel_key_path: Optional[str] = Field(
        default=None, description="Path to SSH private key"
    )
    ssh_tunnel_key_content: Optional[str] = Field(
        default=None, description="SSH private key content"
    )
    ssh_tunnel_key_passphrase: Optional[str] = Field(
        default=None, description="SSH key passphrase"
    )


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
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Whether to use SSH tunnel"
    )
    ssh_tunnel_host: Optional[str] = Field(
        default=None, description="SSH server hostname"
    )
    ssh_tunnel_port: int = Field(default=22, description="SSH server port")
    ssh_tunnel_user: Optional[str] = Field(default=None, description="SSH username")
    ssh_tunnel_password: Optional[str] = Field(default=None, description="SSH password")
    ssh_tunnel_key_path: Optional[str] = Field(
        default=None, description="Path to SSH private key"
    )
    ssh_tunnel_key_content: Optional[str] = Field(
        default=None, description="SSH private key content"
    )
    ssh_tunnel_key_passphrase: Optional[str] = Field(
        default=None, description="SSH key passphrase"
    )


class MssqlDiscoverResponse(BaseModel):
    """Response from MSSQL database discovery."""

    success: bool = Field(description="Whether discovery succeeded")
    databases: List[str] = Field(
        default_factory=list, description="List of discovered database names"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if discovery failed"
    )


class MysqlDiscoverRequest(BaseModel):
    """Request to discover databases on a MySQL/MariaDB server."""

    host: Optional[str] = Field(
        default=None, description="MySQL/MariaDB hostname or IP (for direct mode)"
    )
    port: int = Field(default=3306, description="MySQL/MariaDB port")
    user: Optional[str] = Field(
        default=None, description="MySQL username (for direct mode)"
    )
    password: Optional[str] = Field(
        default=None, description="MySQL password (for direct mode)"
    )
    database: Optional[str] = Field(
        default=None, description="Default database to check if discovery fails"
    )
    container: Optional[str] = Field(
        default=None, description="Docker container name (for container mode)"
    )
    docker_network: Optional[str] = Field(
        default=None, description="Docker network name (for container mode)"
    )
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Whether to use SSH tunnel"
    )
    ssh_tunnel_host: Optional[str] = Field(
        default=None, description="SSH server hostname"
    )
    ssh_tunnel_port: int = Field(default=22, description="SSH server port")
    ssh_tunnel_user: Optional[str] = Field(default=None, description="SSH username")
    ssh_tunnel_password: Optional[str] = Field(default=None, description="SSH password")
    ssh_tunnel_key_path: Optional[str] = Field(
        default=None, description="Path to SSH private key"
    )
    ssh_tunnel_key_content: Optional[str] = Field(
        default=None, description="SSH private key content"
    )
    ssh_tunnel_key_passphrase: Optional[str] = Field(
        default=None, description="SSH key passphrase"
    )


class MysqlDiscoverResponse(BaseModel):
    """Response from MySQL database discovery."""

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
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Whether to use SSH tunnel"
    )
    ssh_tunnel_host: Optional[str] = Field(
        default=None, description="SSH server hostname"
    )
    ssh_tunnel_port: int = Field(default=22, description="SSH server port")
    ssh_tunnel_user: Optional[str] = Field(default=None, description="SSH username")
    ssh_tunnel_password: Optional[str] = Field(default=None, description="SSH password")
    ssh_tunnel_key_path: Optional[str] = Field(
        default=None, description="Path to SSH private key"
    )
    ssh_tunnel_key_content: Optional[str] = Field(
        default=None, description="SSH private key content"
    )
    ssh_tunnel_key_passphrase: Optional[str] = Field(
        default=None, description="SSH key passphrase"
    )


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
    connection: Optional[dict] = Field(
        default=None,
        description="Optional tool connection reference metadata (tool_config_id/name/type)",
    )


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
    connection: Optional[dict] = Field(
        default=None,
        description="Optional tool connection reference metadata (tool_config_id/name/type)",
    )


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
    workspace_id: Optional[str] = Field(
        default=None, description="Optional User Space workspace ID"
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
    tool_output_mode: ToolOutputMode = Field(
        default=ToolOutputMode.DEFAULT,
        description="Per-conversation tool output preference: default (use global), show (always), hide (always), auto (AI decides)",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationResponse(BaseModel):
    """Response for conversation data."""

    id: str
    title: str
    model: str
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    username: Optional[str] = None
    display_name: Optional[str] = None
    messages: List[ChatMessage]
    total_tokens: int
    active_task_id: Optional[str] = None
    tool_output_mode: ToolOutputMode = Field(default=ToolOutputMode.DEFAULT)
    created_at: datetime
    updated_at: datetime


class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""

    title: Optional[str] = Field(
        default=None, description="Optional title for the conversation"
    )
    model: Optional[str] = Field(default=None, description="Optional model override")
    workspace_id: Optional[str] = Field(
        default=None,
        description="Optional User Space workspace ID for workspace-scoped conversations",
    )


class SendMessageRequest(BaseModel):
    """Request to send a message in a conversation."""

    message: str = Field(description="The message content to send")
    stream: bool = Field(default=False, description="Whether to stream the response")
    background: bool = Field(
        default=False, description="Whether to run in background mode"
    )


class RetryVisualizationRequest(BaseModel):
    """Request to retry a failed visualization tool call."""

    tool_type: Literal["datatable", "chart"] = Field(
        description="Type of visualization to create: 'datatable' or 'chart'"
    )
    source_data: dict = Field(
        description="Source data for the visualization. For datatable: {columns: [], rows: []}. For chart: raw data."
    )
    title: str | None = Field(
        default=None, description="Optional title override for the visualization"
    )


class RetryVisualizationResponse(BaseModel):
    """Response from a retry visualization request."""

    success: bool = Field(description="Whether the retry was successful")
    output: str | None = Field(
        default=None, description="The visualization output JSON on success"
    )
    error: str | None = Field(default=None, description="Error message on failure")


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
    interrupted = "interrupted"  # Task was running when server restarted


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
    introspected_tables: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    cancel_requested: bool = False
    status_detail: Optional[str] = None  # Runtime-only phase detail (not persisted)

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
    introspected_tables: int = 0
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None
    cancel_requested: bool = False
    status_detail: Optional[str] = None
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
    table_comment: Optional[str] = Field(
        default=None, description="Table/view description or comment"
    )
    columns: List[dict] = Field(
        default_factory=list,
        description="List of column definitions with name, type, nullable, default, comment",
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
    check_constraints: List[dict] = Field(
        default_factory=list,
        description="List of check constraints with name and definition",
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
        if self.table_comment:
            lines.append(f"Description: {self.table_comment}")
        if self.row_count_estimate:
            lines.append(f"Estimated rows: ~{self.row_count_estimate:,}")

        # Columns section
        lines.append("\n## Columns:")
        for col in self.columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            # Normalize nullable: handles bool, "YES"/"NO", "t"/"f" from various DBs
            raw_nullable = col.get("nullable", True)
            if isinstance(raw_nullable, str):
                is_nullable = raw_nullable.lower() in ("yes", "true", "t", "1")
            else:
                is_nullable = bool(raw_nullable)
            nullable = "NULL" if is_nullable else "NOT NULL"
            default = col.get("default", "")

            col_line = f"  - {col_name}: {col_type} {nullable}"
            if default:
                col_line += f" DEFAULT {default}"
            if col_name in self.primary_key:
                col_line += " [PRIMARY KEY]"
            # Include column comment/description if present
            col_comment = col.get("comment") or col.get("description")
            if col_comment:
                col_line += f" -- {col_comment}"
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

        # Check constraints section
        if self.check_constraints:
            lines.append("\n## Check Constraints:")
            for cc in self.check_constraints:
                cc_name = cc.get("name", "")
                cc_def = cc.get("definition", "")
                lines.append(f"  - {cc_name}: {cc_def}")

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

        # Configurations - only include if they have meaningful data
        # Skip configs that only have a name (like "@" or "Default") with no PartNumber/Description
        if self.configurations:
            config_lines = []
            for config in self.configurations:
                config_name = config.get("name", "Default")
                config_pn = config.get("part_number", "")
                config_desc = config.get("description", "")
                # Only include configurations with actual part number or description data
                if config_pn or config_desc:
                    config_line = f"- {config_name}"
                    if config_pn:
                        config_line += f": Part Number {config_pn}"
                    if config_desc:
                        config_line += f', Description "{config_desc}"'
                    config_lines.append(config_line)
            if config_lines:
                lines.append("\n## Configurations")
                lines.extend(config_lines)

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
    # SSH tunnel configuration
    ssh_tunnel_enabled: bool = Field(
        default=False, description="Use SSH tunnel for database connection"
    )
    ssh_tunnel_host: str = Field(
        default="", description="SSH server hostname for tunnel"
    )
    ssh_tunnel_port: int = Field(
        default=22, ge=1, le=65535, description="SSH server port"
    )
    ssh_tunnel_user: str = Field(default="", description="SSH username")
    ssh_tunnel_password: str = Field(
        default="", description="SSH password (if not using key)"
    )
    ssh_tunnel_key_path: str = Field(default="", description="Path to SSH private key")
    ssh_tunnel_key_content: str = Field(
        default="", description="SSH private key content (preferred over path)"
    )
    ssh_tunnel_key_passphrase: str = Field(
        default="", description="Passphrase for encrypted SSH key"
    )
    ssh_tunnel_public_key: str = Field(
        default="", description="SSH public key for generated keypairs"
    )

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
