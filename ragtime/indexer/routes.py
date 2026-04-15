"""
Indexer API routes.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional

import httpx
from fastapi import (APIRouter, Body, Depends, File, Form, HTTPException,
                     Query, UploadFile)
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from prisma import Prisma
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.container_capabilities import get_container_capabilities
from ragtime.core.copilot_api import (COPILOT_DEFAULT_BASE_URL,
                                      build_copilot_headers)
from ragtime.core.copilot_auth import (ensure_copilot_token_fresh,
                                       exchange_github_token_for_copilot_token,
                                       is_copilot_token_refresh_in_progress)
from ragtime.core.embedding_models import (OPENAI_EMBEDDING_PRIORITY,
                                           get_embedding_models)
from ragtime.core.encryption import decrypt_secret
from ragtime.core.event_bus import task_event_bus
from ragtime.core.git import check_repo_visibility as git_check_visibility
from ragtime.core.git import fetch_branches as git_fetch_branches
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import (MODEL_FAMILY_PATTERNS,
                                       get_context_limit, get_output_limit,
                                       register_model_reasoning_capabilities,
                                       register_model_supported_endpoints,
                                       requires_responses_api,
                                       supports_function_calling,
                                       supports_responses_api,
                                       update_model_function_calling,
                                       update_model_limit,
                                       update_model_output_limit)
from ragtime.core.ollama import (extract_capabilities,
                                 extract_effective_context_length,
                                 get_model_details, is_embedding_capable,
                                 is_reachable)
from ragtime.core.ollama import list_models
from ragtime.core.ollama import list_models as ollama_list_models
from ragtime.core.security import get_current_user, require_admin
from ragtime.core.sql_utils import (MssqlConnectionError, MysqlConnectionError,
                                    mssql_connect, mysql_connect,
                                    normalize_mssql_error_message)
from ragtime.core.ssh import (SSHConfig, SSHTunnel, build_ssh_tunnel_config,
                              execute_ssh_command, ssh_tunnel_config_from_dict,
                              test_ssh_connection)
from ragtime.core.tokenization import count_tokens
from ragtime.core.userspace_preview_sandbox import (
    USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS,
    USERSPACE_PREVIEW_SANDBOX_FLAG_OPTIONS)
from ragtime.core.validation import require_valid_embedding_provider
from ragtime.core.vision_models import list_vision_models
from ragtime.indexer.background_tasks import (
    background_task_service, rebuild_tool_messages_from_events)
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.models import (AnalyzeIndexRequest, AppSettings,
                                    ChatMessage, ChatTaskResponse,
                                    ChatTaskStatus, CheckRepoVisibilityRequest,
                                    ConfigurationWarning, Conversation,
                                    ConversationResponse,
                                    CreateConversationRequest,
                                    CreateIndexRequest,
                                    CreateToolConfigRequest,
                                    CreateToolGroupRequest,
                                    DatabaseDiscoverOption, EmbeddingStatus,
                                    FetchBranchesRequest,
                                    FetchBranchesResponse,
                                    FilesystemAnalysisJobResponse,
                                    FilesystemConnectionConfig,
                                    FilesystemIndexJobResponse,
                                    IndexAnalysisResult, IndexConfig,
                                    IndexInfo, IndexJobResponse, IndexStatus,
                                    InfluxdbDiscoverRequest,
                                    InfluxdbDiscoverResponse,
                                    MssqlDiscoverRequest,
                                    MssqlDiscoverResponse,
                                    MysqlDiscoverRequest,
                                    MysqlDiscoverResponse, OcrMode,
                                    PdmDiscoverRequest, PdmDiscoverResponse,
                                    PdmIndexJobResponse,
                                    PostgresDiscoverRequest,
                                    PostgresDiscoverResponse,
                                    ProviderPromptDebugListResponse,
                                    ProviderPromptDebugRecord,
                                    RepoVisibilityResponse,
                                    RetryVisualizationRequest,
                                    RetryVisualizationResponse,
                                    SchemaIndexJobResponse, SendMessageRequest,
                                    ToolConfig, ToolGroup, ToolTestRequest,
                                    ToolType, TriggerFilesystemIndexRequest,
                                    TriggerPdmIndexRequest,
                                    TriggerSchemaIndexRequest,
                                    UpdateSettingsRequest,
                                    UpdateToolConfigRequest,
                                    UpdateToolGroupRequest,
                                    UserSpacePreviewSettingsResponse,
                                    VectorStoreType,
                                    WorkspaceChatStateResponse)
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import (SCHEMA_INDEXER_CAPABLE_TYPES,
                                            schema_indexer)
from ragtime.indexer.service import indexer
from ragtime.indexer.title_generation import schedule_title_generation
from ragtime.indexer.tool_health import get_heartbeat_timeout_seconds
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_backends import FAISS_INDEX_BASE_PATH
from ragtime.indexer.vector_utils import ensure_pgvector_extension
from ragtime.indexer.workspace_state import build_workspace_chat_state
from ragtime.mcp.server import notify_tools_changed
from ragtime.rag import rag
from ragtime.tools.chart import create_chart
from ragtime.tools.datatable import create_datatable
from ragtime.tools.influxdb import test_influxdb_connection
from ragtime.tools.mssql import test_mssql_connection
from ragtime.tools.mysql import test_mysql_connection
from ragtime.userspace.service import userspace_service

if TYPE_CHECKING:
    from prisma.models import User

logger = get_logger(__name__)

# GitHub Copilot OAuth device flow
# We must use VSCode's Client ID to get user subscribed models
GITHUB_COPILOT_CLIENT_ID = os.getenv("GITHUB_COPILOT_CLIENT_ID", "Iv1.b507a08c87ecfe98")
GITHUB_MODELS_CATALOG_URL = "https://models.github.ai/catalog/models"
MODELS_DEV_API_URL = "https://models.dev/api.json"
OAUTH_POLLING_SAFETY_MARGIN_SECONDS = 3

# In-memory pending OAuth device requests (request_id -> request state)
_copilot_device_requests: dict[str, dict[str, Any]] = {}

router = APIRouter(prefix="/indexes", tags=["Indexer"])

# Path to static files - React build lives under frontend/dist
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"

# Check if running in development mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Cache for SSH credential validation results (tool_id -> (timestamp, result))
# This allows the heartbeat to return cached deep-check results
_ssh_credential_cache: dict[str, tuple[float, ToolTestResponse]] = {}
_SSH_CREDENTIAL_CACHE_TTL = 15.0  # seconds

# Short-lived cache for external model discovery endpoints used by settings/chat
# model selectors. This avoids repeated high-latency calls during UI refreshes.
_MODEL_DISCOVERY_CACHE_TTL_SECONDS = float(
    os.getenv("MODEL_DISCOVERY_CACHE_TTL_SECONDS", "30")
)
_model_discovery_cache: dict[str, tuple[float, "LLMModelsResponse"]] = {}
_model_discovery_inflight: dict[str, asyncio.Task["LLMModelsResponse"]] = {}
_model_discovery_lock = asyncio.Lock()


def _resolve_github_auth_mode(
    settings: AppSettings,
    requested_mode: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve GitHub auth mode and enforce PAT/OAuth exclusivity."""
    pat_token = (settings.github_models_api_token or "").strip()
    oauth_access = (settings.github_copilot_access_token or "").strip()
    oauth_refresh = (settings.github_copilot_refresh_token or "").strip()
    has_pat = bool(pat_token)
    has_oauth = bool(oauth_access or oauth_refresh)

    mode = (requested_mode or "").strip().lower()
    if mode:
        if mode not in {"oauth", "pat"}:
            return None, "Invalid auth_mode. Supported values: oauth, pat."
        return mode, None

    if has_pat and has_oauth:
        return (
            None,
            "Both GitHub PAT and Copilot OAuth credentials are configured. "
            "Choose one authentication mode and clear the other.",
        )

    return ("pat" if has_pat else "oauth"), None


def _copilot_token_fingerprint(token: str) -> str:
    """Build a non-sensitive token fingerprint for cache keying."""
    token = token.strip()
    return f"len={len(token)}:tail={token[-8:] if token else ''}"


async def _get_or_fetch_model_discovery(
    cache_key: str,
    fetcher: Callable[[], Awaitable["LLMModelsResponse"]],
    *,
    force_refresh: bool = False,
) -> "LLMModelsResponse":
    """Return cached model discovery result or dedupe concurrent refreshes."""
    now = time.monotonic()

    async with _model_discovery_lock:
        cached = None if force_refresh else _model_discovery_cache.get(cache_key)
        if cached and (now - cached[0]) < _MODEL_DISCOVERY_CACHE_TTL_SECONDS:
            return cached[1].model_copy(deep=True)

        inflight = _model_discovery_inflight.get(cache_key)
        if inflight is None:
            inflight = asyncio.create_task(fetcher())
            _model_discovery_inflight[cache_key] = inflight

    try:
        result = await inflight
    finally:
        async with _model_discovery_lock:
            current = _model_discovery_inflight.get(cache_key)
            if current is inflight:
                _model_discovery_inflight.pop(cache_key, None)

    if result.success:
        async with _model_discovery_lock:
            _model_discovery_cache[cache_key] = (
                time.monotonic(),
                result.model_copy(deep=True),
            )

    return result


@router.get("", response_model=List[IndexInfo])
async def list_indexes(_user: User = Depends(require_admin)):
    """List all available FAISS indexes. Admin only."""
    return await indexer.list_indexes()


@router.post("/analyze", response_model=IndexAnalysisResult)
async def analyze_repository(
    request: AnalyzeIndexRequest,
    _user: User = Depends(require_admin),
):
    """
    Analyze a git repository before indexing.

    Performs a shallow clone and scans files to provide:
    - Total file count, size, and estimated chunks
    - Breakdown by file extension
    - Suggested exclusion patterns
    - Warnings about potential issues (large files, binaries, etc.)

    Use this to preview index size and tune configuration before creating an index.
    """
    try:
        return await indexer.analyze_git_repository(request)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e


@router.post("/upload/analyze", response_model=IndexAnalysisResult)
async def analyze_upload(
    file: UploadFile = File(
        ...,
        description="Archive file to analyze (.zip, .tar, .tar.gz, .tar.bz2)",
    ),
    file_patterns: str = Form(
        default="",
        description="Comma-separated glob patterns for files to include (e.g. **/*.py, **/*.md)",
    ),
    exclude_patterns: str = Form(
        default="",
        description="Comma-separated glob patterns to exclude (e.g. **/node_modules/**, **/__pycache__/**)",
    ),
    chunk_size: int = Form(default=1000, ge=100, le=4000),
    chunk_overlap: int = Form(default=200, ge=0, le=1000),
    max_file_size_kb: int = Form(default=500, ge=10, le=10000),
    ocr_mode: str = Form(
        default="disabled",
        description="OCR mode: 'disabled', 'tesseract', or 'ollama'",
    ),
    ocr_vision_model: Optional[str] = Form(
        default=None,
        description="Ollama vision model for OCR (e.g., 'qwen3-vl:latest')",
    ),
    _user: User = Depends(require_admin),
):
    """
    Analyze an uploaded archive before indexing.

    Extracts and scans files to provide:
    - Total file count, size, and estimated chunks
    - Breakdown by file extension
    - Suggested exclusion patterns
    - Warnings about potential issues (large files, binaries, etc.)

    Use this to preview index size and tune configuration before creating an index.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_ARCHIVE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_ARCHIVE_EXTENSIONS)}",
        )

    try:
        # Parse patterns, defaulting file_patterns to **/* if empty
        parsed_file_patterns = [
            p.strip() for p in file_patterns.split(",") if p.strip()
        ]
        if not parsed_file_patterns:
            parsed_file_patterns = ["**/*"]
        parsed_exclude_patterns = [
            p.strip() for p in exclude_patterns.split(",") if p.strip()
        ]

        return await indexer.analyze_upload(
            file=file.file,
            filename=file.filename,
            file_patterns=parsed_file_patterns,
            exclude_patterns=parsed_exclude_patterns,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_file_size_kb=max_file_size_kb,
            ocr_mode=ocr_mode,
            ocr_vision_model=ocr_vision_model,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Upload analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e


@router.post("/check-visibility", response_model=RepoVisibilityResponse)
async def check_repo_visibility(
    request: CheckRepoVisibilityRequest,
    _user: User = Depends(require_admin),
):
    """
    Check if a Git repository is publicly accessible.

    This is used to determine whether a token is needed for re-indexing.
    If an index_name is provided, checks if a stored token exists and is valid.

    Returns:
    - visibility: 'public', 'private', 'not_found', or 'error'
    - has_stored_token: True if a valid token is stored
    - needs_token: True if user needs to provide a token
    - message: Human-readable status message
    """
    # Get stored token if index_name provided
    stored_token = None
    if request.index_name:
        try:
            metadata = await repository.get_index_metadata(request.index_name)
            encrypted_token = getattr(metadata, "gitToken", None) if metadata else None
            stored_token = decrypt_secret(encrypted_token) if encrypted_token else None
        except Exception:
            pass  # No stored token available

    result = await git_check_visibility(
        url=request.git_url,
        stored_token=stored_token,
    )

    return RepoVisibilityResponse(
        visibility=result.visibility.value,
        has_stored_token=result.has_stored_token,
        needs_token=result.needs_token,
        message=result.message,
    )


@router.post("/fetch-branches", response_model=FetchBranchesResponse)
async def fetch_branches(
    request: FetchBranchesRequest,
    _user: User = Depends(require_admin),
):
    """
    Fetch branches from a Git repository.

    Uses stored token from an existing index if available, otherwise uses provided token.
    """
    # Try to get stored token from existing index if index_name provided
    token = request.git_token
    if not token and request.index_name:
        try:
            metadata = await repository.get_index_metadata(request.index_name)
            encrypted_token = getattr(metadata, "gitToken", None) if metadata else None
            token = decrypt_secret(encrypted_token) if encrypted_token else None
        except Exception:
            pass  # No stored token available

    branches, error = await git_fetch_branches(
        url=request.git_url,
        token=token,
    )

    if error:
        needs_token = "private" in error.lower() or "token" in error.lower()
        return FetchBranchesResponse(
            branches=[],
            error=error,
            needs_token=needs_token,
        )

    return FetchBranchesResponse(
        branches=branches,
        error=None,
        needs_token=False,
    )


@router.get("/jobs", response_model=List[IndexJobResponse])
async def list_jobs(_user: User = Depends(require_admin)):
    """List all indexing jobs. Admin only."""
    jobs = await indexer.list_jobs()
    return [
        IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}", response_model=IndexJobResponse)
async def get_job(job_id: str, _user: User = Depends(require_admin)):
    """Get status of an indexing job. Admin only."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return IndexJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        progress_percent=job.progress_percent,
        total_files=job.total_files,
        processed_files=job.processed_files,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, _user: User = Depends(require_admin)):
    """Cancel a pending or processing job. Admin only."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [IndexStatus.PENDING, IndexStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status.value}'",
        )

    await indexer.cancel_job(job_id)
    return {"message": f"Job '{job_id}' cancelled successfully"}


# Supported archive extensions
SUPPORTED_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


@router.post("/upload", response_model=IndexJobResponse)
async def upload_and_index(
    file: UploadFile = File(
        ...,
        description="Archive file containing source code (.zip, .tar, .tar.gz, .tar.bz2)",
    ),
    name: str = Form(..., description="Name for the index"),
    description: str = Form(
        default="",
        description="Description for AI context - helps the model understand what this index contains",
    ),
    file_patterns: str = Form(
        default="",
        description="Comma-separated glob patterns for files to include (e.g. **/*.py, **/*.md)",
    ),
    exclude_patterns: str = Form(
        default="",
        description="Comma-separated glob patterns to exclude (e.g. **/node_modules/**, **/__pycache__/**)",
    ),
    chunk_size: int = Form(default=1000, ge=100, le=4000),
    chunk_overlap: int = Form(default=200, ge=0, le=1000),
    ocr_mode: str = Form(
        default="disabled",
        description="OCR mode: 'disabled', 'tesseract', or 'ollama'",
    ),
    ocr_vision_model: Optional[str] = Form(
        default=None,
        description="Ollama vision model for OCR (e.g., 'qwen3-vl:latest')",
    ),
    vector_store_type: str = Form(
        default="faiss",
        description="Vector store backend: 'faiss' (in-memory, default) or 'pgvector' (PostgreSQL)",
    ),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Upload an archive file and create a vector index from it. Admin only.

    Supported formats: .zip, .tar, .tar.gz, .tar.bz2
    The archive should contain source code files. Large codebases are supported.
    Processing happens in the background - check /jobs/{id} for status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_ARCHIVE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_ARCHIVE_EXTENSIONS)}",
        )

    # Parse patterns, defaulting file_patterns to **/* if empty
    parsed_file_patterns = [p.strip() for p in file_patterns.split(",") if p.strip()]
    if not parsed_file_patterns:
        parsed_file_patterns = ["**/*"]
    parsed_exclude_patterns = [
        p.strip() for p in exclude_patterns.split(",") if p.strip()
    ]

    config = IndexConfig(
        name=name,
        description=description,
        file_patterns=parsed_file_patterns,
        exclude_patterns=parsed_exclude_patterns,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_mode=OcrMode(ocr_mode),
        ocr_vision_model=ocr_vision_model,
        vector_store_type=VectorStoreType(vector_store_type),
    )

    try:
        job = await indexer.create_index_from_upload(file.file, file.filename, config)

        return IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            vector_store_type=job.config.vector_store_type,
        )
    except Exception as e:
        logger.exception("Failed to start indexing job")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/git", response_model=IndexJobResponse)
async def index_from_git(
    request: CreateIndexRequest,
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Create a vector index from a git repository. Admin only.

    Clones the repository and indexes the source files.
    Processing happens in the background - check /jobs/{id} for status.
    """
    if not request.git_url:
        raise HTTPException(status_code=400, detail="git_url is required")

    config = request.config or IndexConfig(name=request.name)
    config.name = request.name  # Ensure name is set

    try:
        job = await indexer.create_index_from_git(
            git_url=request.git_url,
            branch=request.git_branch,
            config=config,
            git_token=request.git_token,
        )

        return IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            vector_store_type=job.config.vector_store_type,
        )
    except Exception as e:
        logger.exception("Failed to start git indexing job")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ReindexGitRequest(BaseModel):
    """Request to re-index from git (pull latest changes)."""

    git_token: Optional[str] = Field(
        default=None,
        description="GitHub/GitLab Personal Access Token for private repos (uses stored token if not provided)",
    )


@router.post("/{name}/reindex", response_model=IndexJobResponse)
async def reindex_from_git(
    name: str,
    request: ReindexGitRequest = Body(default=ReindexGitRequest()),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Re-index an existing git-based index by pulling latest changes. Admin only.

    This endpoint fetches the latest changes from the git repository and
    re-creates the index using its configured vector store backend (FAISS or pgvector).
    Only works for indexes created from git repos.
    For private repos, uses the stored token unless a new one is provided.
    """
    # Get existing index metadata
    metadata = await repository.get_index_metadata(name)
    if not metadata:
        raise HTTPException(status_code=404, detail="Index not found")

    if metadata.sourceType != "git":
        raise HTTPException(
            status_code=400,
            detail="Re-indexing only supported for git-based indexes. Upload-based indexes must be re-uploaded.",
        )

    if not metadata.source:
        raise HTTPException(
            status_code=400,
            detail="Git URL not found in index metadata. Cannot re-index.",
        )

    # Use provided token, or fall back to stored token (decrypt if encrypted)
    encrypted_token = getattr(metadata, "gitToken", None)
    stored_token = decrypt_secret(encrypted_token) if encrypted_token else None
    git_token = request.git_token or stored_token

    # Get config from snapshot or use defaults
    config_snapshot = getattr(metadata, "configSnapshot", None)
    config_data: dict[str, Any] = (
        config_snapshot if isinstance(config_snapshot, dict) else {}
    )

    # Handle legacy enable_ocr field and convert to ocr_mode
    ocr_mode_str = config_data.get("ocr_mode", "disabled")
    if ocr_mode_str == "disabled" and config_data.get("enable_ocr", False):
        ocr_mode_str = "tesseract"  # Legacy compatibility

    config = IndexConfig(
        name=name,
        description=metadata.description or "",
        file_patterns=config_data.get("file_patterns", ["**/*"]),
        exclude_patterns=config_data.get(
            "exclude_patterns", ["**/test/**", "**/tests/**", "**/__pycache__/**"]
        ),
        chunk_size=config_data.get("chunk_size", 1000),
        chunk_overlap=config_data.get("chunk_overlap", 200),
        max_file_size_kb=config_data.get("max_file_size_kb", 500),
        ocr_mode=OcrMode(ocr_mode_str),
        ocr_vision_model=config_data.get("ocr_vision_model"),
        git_clone_timeout_minutes=config_data.get("git_clone_timeout_minutes", 5),
        git_history_depth=config_data.get("git_history_depth", 1),
        reindex_interval_hours=config_data.get("reindex_interval_hours", 0),
    )

    try:
        job = await indexer.create_index_from_git(
            git_url=metadata.source,
            branch=(getattr(metadata, "gitBranch", None) or "main"),
            config=config,
            git_token=git_token,
        )

        return IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
    except Exception as e:
        logger.exception("Failed to start re-indexing job")
        raise HTTPException(status_code=500, detail=str(e)) from e


class RetryJobRequest(BaseModel):
    """Request to retry a failed job."""

    git_token: Optional[str] = Field(
        default=None,
        description="GitHub/GitLab Personal Access Token (uses stored token if not provided)",
    )


@router.post("/jobs/{job_id}/retry", response_model=IndexJobResponse)
async def retry_failed_job(
    job_id: str,
    request: RetryJobRequest = Body(default=RetryJobRequest()),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Retry a failed or stuck indexing job. Admin only.

    This creates a new job using the same settings as the failed job.
    For private repos, uses the stored token unless a new one is provided.
    """
    # Get the failed job
    failed_job = await repository.get_job(job_id)
    if not failed_job:
        raise HTTPException(status_code=404, detail="Job not found")

    if failed_job.status not in [IndexStatus.FAILED, IndexStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status '{failed_job.status}'. Only failed or stuck jobs can be retried.",
        )

    # Use provided token, or fall back to stored token
    git_token = request.git_token or failed_job.git_token

    if failed_job.source_type == "git":
        if not failed_job.git_url:
            raise HTTPException(
                status_code=400, detail="Git URL not found in job. Cannot retry."
            )

        try:
            new_job = await indexer.create_index_from_git(
                git_url=failed_job.git_url,
                branch=failed_job.git_branch,
                config=failed_job.config,
                git_token=git_token,
            )

            return IndexJobResponse(
                id=new_job.id,
                name=new_job.name,
                status=new_job.status,
                progress_percent=new_job.progress_percent,
                total_files=new_job.total_files,
                processed_files=new_job.processed_files,
                total_chunks=new_job.total_chunks,
                error_message=new_job.error_message,
                created_at=new_job.created_at,
                started_at=new_job.started_at,
                completed_at=new_job.completed_at,
            )
        except Exception as e:
            logger.exception("Failed to retry indexing job")
            raise HTTPException(status_code=500, detail=str(e)) from e
    elif failed_job.source_type == "upload":
        try:
            new_job = await indexer.retry_upload_job(failed_job)

            return IndexJobResponse(
                id=new_job.id,
                name=new_job.name,
                status=new_job.status,
                progress_percent=new_job.progress_percent,
                total_files=new_job.total_files,
                processed_files=new_job.processed_files,
                total_chunks=new_job.total_chunks,
                error_message=new_job.error_message,
                created_at=new_job.created_at,
                started_at=new_job.started_at,
                completed_at=new_job.completed_at,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Failed to retry upload job")
            raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source type: {failed_job.source_type}",
        )


@router.delete("/{name}")
async def delete_index(name: str, _user: User = Depends(require_admin)):
    """Delete an index by name. Admin only."""
    # Delete files and metadata
    if await indexer.delete_index(name):
        # Remove from memory
        rag.unload_index(name)

        # Invalidate caches
        invalidate_settings_cache()

        # Rebuild agent with updated tools (without full reinit to avoid interrupting background tasks)
        try:
            await rag.rebuild_agent()
        except Exception as e:
            logger.warning(f"Failed to rebuild agent after delete: {e}")

        # Notify MCP clients about tool changes
        try:
            notify_tools_changed()
        except Exception as e:
            logger.warning(f"Failed to notify MCP about delete: {e}")

        return {"message": f"Index '{name}' deleted successfully"}

    raise HTTPException(status_code=404, detail="Index not found")


class ToggleIndexRequest(BaseModel):
    """Request to toggle index enabled status."""

    enabled: bool = Field(description="Whether the index is enabled for RAG context")


class UpdateIndexDescriptionRequest(BaseModel):
    """Request to update index description."""

    description: str = Field(description="Description for AI context")


class UpdateIndexWeightRequest(BaseModel):
    """Request to update index search weight."""

    weight: float = Field(
        ge=0.0,
        le=10.0,
        description="Search weight (0.0-10.0). Higher values make this index more prominent in results.",
    )


@router.patch("/{name}/toggle")
async def toggle_index(
    name: str, request: ToggleIndexRequest, _user: User = Depends(require_admin)
):
    """Toggle an index's enabled status for RAG context. Admin only.

    When enabling an index, this will reinitialize RAG components to attempt
    loading the index (useful for retrying failed loads).
    """
    success = await repository.set_index_enabled(name, request.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")

    # When disabling, unload the index from memory to free RAM
    if not request.enabled:
        rag.unload_index(name)

        # Rebuild agent without the disabled index
        try:
            await rag.rebuild_agent()
        except Exception as e:
            logger.warning(f"Failed to rebuild agent after disabling index {name}: {e}")

    # Reinitialize RAG when enabling to attempt loading the index
    # This allows retrying failed loads when user toggles an errored index back on
    if request.enabled:
        try:
            await rag.initialize()
        except Exception as e:
            logger.warning(
                f"Failed to reinitialize RAG after enabling index {name}: {e}"
            )

    return {
        "message": f"Index '{name}' {'enabled' if request.enabled else 'disabled'}",
        "enabled": request.enabled,
    }


@router.patch("/{name}/description")
async def update_index_description(
    name: str,
    request: UpdateIndexDescriptionRequest,
    _user: User = Depends(require_admin),
):
    """Update an index's description for AI context. Admin only."""
    success = await repository.update_index_description(name, request.description)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return {
        "message": f"Description updated for '{name}'",
        "description": request.description,
    }


@router.patch("/{name}/weight")
async def update_index_weight(
    name: str,
    request: UpdateIndexWeightRequest,
    _user: User = Depends(require_admin),
):
    """Update an index's search weight. Admin only.

    Weight affects result prioritization when aggregate_search is enabled:
    - 1.0 = default/equal weighting
    - >1.0 = higher priority (more results from this index shown first)
    - <1.0 = lower priority
    - 0.0 = effectively excluded from weighted results (still searched, but deprioritized)
    """
    success = await repository.update_index_weight(name, request.weight)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return {
        "message": f"Search weight updated for '{name}'",
        "weight": request.weight,
    }


class UpdateIndexConfigRequest(BaseModel):
    """Request to update index configuration for next re-index."""

    git_branch: Optional[str] = Field(
        default=None, description="Git branch to use for re-indexing"
    )
    file_patterns: Optional[List[str]] = Field(
        default=None, description="Glob patterns for files to include"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None, description="Glob patterns for files to exclude"
    )
    chunk_size: Optional[int] = Field(
        default=None, ge=100, le=4000, description="Chunk size for text splitting"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, ge=0, le=1000, description="Chunk overlap for text splitting"
    )
    max_file_size_kb: Optional[int] = Field(
        default=None, ge=10, le=10000, description="Maximum file size in KB"
    )
    ocr_mode: Optional[str] = Field(
        default=None, description="OCR mode: 'disabled', 'tesseract', or 'ollama'"
    )
    ocr_vision_model: Optional[str] = Field(
        default=None,
        description="Ollama vision model for OCR (e.g., 'qwen3-vl:latest')",
    )
    git_clone_timeout_minutes: Optional[int] = Field(
        default=None,
        ge=1,
        le=480,
        description="Git clone timeout in minutes (shallow clone)",
    )
    git_history_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Git history depth. 0=full history, 1=shallow (latest only), N=last N commits",
    )
    reindex_interval_hours: Optional[int] = Field(
        default=None,
        ge=0,
        le=8760,
        description="Hours between automatic pull & re-index (0 = manual only)",
    )


@router.patch("/{name}/config")
async def update_index_config(
    name: str,
    request: UpdateIndexConfigRequest,
    _user: User = Depends(require_admin),
):
    """Update an index's configuration for the next re-index. Admin only.

    This updates the stored config snapshot that will be used when
    "Pull & Re-index" is triggered. Only works for git-based indexes.
    """
    # Get existing metadata
    metadata = await repository.get_index_metadata(name)
    if not metadata:
        raise HTTPException(status_code=404, detail="Index not found")

    if metadata.sourceType != "git":
        raise HTTPException(
            status_code=400,
            detail="Config updates only supported for git-based indexes",
        )

    # Merge with existing config snapshot
    existing_snapshot = getattr(metadata, "configSnapshot", None)
    existing_config: dict[str, Any] = (
        existing_snapshot if isinstance(existing_snapshot, dict) else {}
    )
    new_config = {}

    # Copy existing values
    for key in [
        "file_patterns",
        "exclude_patterns",
        "chunk_size",
        "chunk_overlap",
        "max_file_size_kb",
        "ocr_mode",
        "ocr_vision_model",
        "git_clone_timeout_minutes",
        "git_history_depth",
        "reindex_interval_hours",
    ]:
        if key in existing_config:
            new_config[key] = existing_config[key]

    # Apply updates
    if request.file_patterns is not None:
        new_config["file_patterns"] = request.file_patterns
    if request.exclude_patterns is not None:
        new_config["exclude_patterns"] = request.exclude_patterns
    if request.chunk_size is not None:
        new_config["chunk_size"] = request.chunk_size
    if request.chunk_overlap is not None:
        new_config["chunk_overlap"] = request.chunk_overlap
    if request.max_file_size_kb is not None:
        new_config["max_file_size_kb"] = request.max_file_size_kb
    if request.ocr_mode is not None:
        new_config["ocr_mode"] = request.ocr_mode
    if request.ocr_vision_model is not None:
        new_config["ocr_vision_model"] = request.ocr_vision_model
    if request.git_clone_timeout_minutes is not None:
        new_config["git_clone_timeout_minutes"] = request.git_clone_timeout_minutes
    if request.git_history_depth is not None:
        new_config["git_history_depth"] = request.git_history_depth
    if request.reindex_interval_hours is not None:
        new_config["reindex_interval_hours"] = request.reindex_interval_hours

    success = await repository.update_index_config(
        name=name,
        git_branch=request.git_branch,
        config_snapshot=new_config if new_config else None,
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update config")

    return {
        "message": f"Config updated for '{name}'",
        "git_branch": request.git_branch or getattr(metadata, "gitBranch", None),
        "config_snapshot": new_config,
    }


class RenameIndexRequest(BaseModel):
    """Request to rename an index."""

    new_name: str = Field(
        min_length=1,
        max_length=100,
        description="New name for the index (will be converted to safe identifier)",
    )


class RenameIndexResponse(BaseModel):
    """Response from renaming an index."""

    old_name: str
    new_name: str  # Safe tool name (lowercase, alphanumeric with underscores)
    display_name: str  # Human-readable name (original user input)
    message: str


@router.patch("/{name}/rename", response_model=RenameIndexResponse)
async def rename_index(
    name: str,
    request: RenameIndexRequest,
    _user: User = Depends(require_admin),
):
    """Rename a git-based index. Admin only.

    This renames both the database record and the FAISS index directory on disk.
    The MCP server will automatically pick up the new name.

    The user-provided name is automatically converted to a safe identifier using
    safe_tool_name (lowercase, alphanumeric with underscores).
    """
    # Convert user input to safe tool name
    raw_name = request.new_name.strip()
    if not raw_name:
        raise HTTPException(status_code=400, detail="New name cannot be empty")

    new_name = safe_tool_name(raw_name)
    if not new_name:
        raise HTTPException(
            status_code=400,
            detail="Name must contain at least one alphanumeric character",
        )

    # Check if names are the same
    if name == new_name:
        return RenameIndexResponse(
            old_name=name,
            new_name=new_name,
            display_name=request.new_name.strip(),
            message="Name unchanged",
        )

    # Get existing metadata
    metadata = await repository.get_index_metadata(name)
    if not metadata:
        raise HTTPException(status_code=404, detail="Index not found")

    # Check for active indexing jobs - block rename to prevent duplicate indexes
    active_job = await repository.get_active_job_for_index(name)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot rename index while an indexing job is in progress. "
                f"Job '{active_job.id}' (status: {active_job.status.value}) would create "
                f"a duplicate index with the old name '{name}' when it completes. "
                f"Please wait for the job to finish or cancel it first."
            ),
        )

    # Check if new name already exists
    existing = await repository.get_index_metadata(new_name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"An index named '{new_name}' already exists",
        )

    # Rename the FAISS directory on disk first
    old_path = indexer.index_base_path / name
    new_path = indexer.index_base_path / new_name

    if old_path.exists():
        try:
            await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
            logger.info(f"Renamed index directory: {old_path} -> {new_path}")
        except Exception as e:
            logger.exception(f"Failed to rename index directory: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to rename index directory: {e}",
            ) from e
    else:
        logger.warning(f"Index directory not found: {old_path}")

    # Now rename in database (pass the original user input as display_name)
    success = await repository.rename_index(name, new_name, display_name=raw_name)
    if not success:
        # Try to rollback the directory rename
        if new_path.exists():
            try:
                await asyncio.to_thread(shutil.move, str(new_path), str(old_path))
            except Exception:
                logger.error("Failed to rollback directory rename")
        raise HTTPException(
            status_code=500, detail="Failed to rename index in database"
        )

    # Invalidate caches so MCP and RAG pick up the new name
    invalidate_settings_cache()

    # Rename the index in memory (update dict key, reuse loaded FAISS data)
    # This avoids reloading from disk and prevents interrupting background tasks
    rag.rename_index(name, new_name)

    # Rebuild agent with updated tools (without full reinit to avoid interrupting background tasks)
    try:
        await rag.rebuild_agent()
    except Exception as e:
        logger.warning(f"Failed to rebuild agent after rename: {e}")

    # Notify MCP clients about tool changes
    try:
        notify_tools_changed()  # This is a sync function that schedules an async notify
    except Exception as e:
        logger.warning(f"Failed to notify MCP about rename: {e}")

    logger.info(
        f"Successfully renamed index '{name}' to '{new_name}' (display: '{raw_name}')"
    )

    return RenameIndexResponse(
        old_name=name,
        new_name=new_name,
        display_name=raw_name,
        message=f"Index renamed from '{name}' to '{new_name}'",
    )


@router.get("/{name}/download")
async def download_index(name: str, _user: User = Depends(require_admin)):
    """Download FAISS index files as a zip archive. Admin only.

    Returns a zip file containing the index.faiss and index.pkl files.
    """
    # Validate index name (prevent path traversal)
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid index name")

    # Get index path from indexer service
    index_path = indexer.index_base_path / name

    # Ensure the resolved path is within index_base_path (defense in depth)
    try:
        index_path = index_path.resolve()
        if not str(index_path).startswith(str(indexer.index_base_path.resolve())):
            raise HTTPException(status_code=400, detail="Invalid index path")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid index name") from exc

    # Validate the index exists
    if not index_path.exists():
        raise HTTPException(status_code=404, detail=f"Index '{name}' not found")

    # Check that required files exist
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Index files incomplete - index.faiss or index.pkl not found",
        )

    # Create zip file in memory (can be large, run in thread)
    def _create_zip() -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(faiss_file, arcname=f"{name}/index.faiss")
            zf.write(pkl_file, arcname=f"{name}/index.pkl")
        buf.seek(0)
        return buf

    zip_buffer = await asyncio.to_thread(_create_zip)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={name}_index.zip"},
    )


@router.get("/filesystem/{name}/download")
async def download_filesystem_faiss_index(
    name: str, _user: User = Depends(require_admin)
):
    """Download filesystem FAISS index files as a zip archive. Admin only.

    Returns a zip file containing the index.faiss and index.pkl files for a
    filesystem index that uses FAISS as its vector store.
    """
    # Validate index name (prevent path traversal)
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid index name")

    # Get index path
    index_path = FAISS_INDEX_BASE_PATH / name

    # Ensure the resolved path is within FAISS_INDEX_BASE_PATH (defense in depth)
    try:
        index_path = index_path.resolve()
        if not str(index_path).startswith(str(FAISS_INDEX_BASE_PATH.resolve())):
            raise HTTPException(status_code=400, detail="Invalid index path")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid index name") from exc

    # Validate the index exists
    if not index_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Filesystem FAISS index '{name}' not found"
        )

    # Check that required files exist
    faiss_file = index_path / "index.faiss"
    pkl_file = index_path / "index.pkl"

    if not faiss_file.exists() or not pkl_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Index files incomplete - index.faiss or index.pkl not found",
        )

    # Create zip file in memory (can be large, run in thread)
    def _create_zip() -> io.BytesIO:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(faiss_file, arcname=f"{name}/index.faiss")
            zf.write(pkl_file, arcname=f"{name}/index.pkl")
        buf.seek(0)
        return buf

    zip_buffer = await asyncio.to_thread(_create_zip)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=filesystem_{name}_index.zip"
        },
    )


# -----------------------------------------------------------------------------
# Settings Endpoints (Admin only)
# -----------------------------------------------------------------------------


class GetSettingsResponse(BaseModel):
    """Response from get settings including configuration warnings."""

    settings: AppSettings
    configuration_warnings: List[ConfigurationWarning] = Field(
        default_factory=list,
        description="Warnings about potentially suboptimal configuration",
    )


@router.get("/settings", response_model=GetSettingsResponse, tags=["Settings"])
async def get_settings(_user: User = Depends(require_admin)):
    """Get current application settings with configuration warnings. Admin only."""
    settings = await repository.get_settings()

    # Get total chunk count for ivfflat tuning warnings
    chunk_count = 0
    try:
        indexes = await repository.list_index_metadata()
        chunk_count = sum(idx.chunk_count or 0 for idx in indexes)
    except Exception:
        pass

    warnings = await settings.get_configuration_warnings(chunk_count=chunk_count)

    return GetSettingsResponse(settings=settings, configuration_warnings=warnings)


@router.get(
    "/settings/userspace-preview",
    response_model=UserSpacePreviewSettingsResponse,
    tags=["Settings"],
)
async def get_userspace_preview_settings():
    """Get public User Space preview sandbox settings."""

    settings = await repository.get_settings()
    return UserSpacePreviewSettingsResponse(
        userspace_preview_sandbox_flags=settings.userspace_preview_sandbox_flags,
        userspace_preview_sandbox_default_flags=list(
            USERSPACE_PREVIEW_SANDBOX_DEFAULT_FLAGS
        ),
        userspace_preview_sandbox_flag_options=[
            dict(option) for option in USERSPACE_PREVIEW_SANDBOX_FLAG_OPTIONS
        ],
    )


class UpdateSettingsResponse(BaseModel):
    """Response from settings update including embedding warnings."""

    settings: AppSettings
    embedding_warning: Optional[str] = None


@router.put("/settings", response_model=UpdateSettingsResponse, tags=["Settings"])
async def update_settings(
    request: UpdateSettingsRequest, _user: User = Depends(require_admin)
):
    """Update application settings. Admin only.

    Returns a warning if the embedding provider or model was changed and
    existing filesystem indexes will need a full re-index.
    """
    updates = request.model_dump(exclude_unset=True)

    if "server_name" in updates:
        normalized_server_name = str(updates.get("server_name") or "").strip()
        if not normalized_server_name:
            raise HTTPException(status_code=400, detail="Server name cannot be empty")
        updates["server_name"] = normalized_server_name

    if "default_chat_model" in updates:
        default_chat_model = updates.get("default_chat_model")
        if default_chat_model is None:
            updates["default_chat_model"] = None
        else:
            normalized_default_chat_model = str(default_chat_model).strip()
            updates["default_chat_model"] = normalized_default_chat_model or None

    # Enforce mutually-exclusive GitHub auth modes (PAT vs Copilot OAuth).
    current_settings = await repository.get_settings()
    pat_candidate = str(
        updates.get(
            "github_models_api_token",
            current_settings.github_models_api_token,
        )
        or ""
    ).strip()
    oauth_access_candidate = str(
        updates.get(
            "github_copilot_access_token",
            current_settings.github_copilot_access_token,
        )
        or ""
    ).strip()
    oauth_refresh_candidate = str(
        updates.get(
            "github_copilot_refresh_token",
            current_settings.github_copilot_refresh_token,
        )
        or ""
    ).strip()
    if pat_candidate and (oauth_access_candidate or oauth_refresh_candidate):
        raise HTTPException(
            status_code=400,
            detail=(
                "GitHub authentication must use exactly one mode: PAT or OAuth. "
                "Clear one credential set before saving."
            ),
        )

    # Check if embedding config is changing
    old_config_hash = current_settings.get_embedding_config_hash()

    result = await repository.update_settings(updates)

    # Check if embedding config changed
    new_config_hash = result.get_embedding_config_hash()
    embedding_warning = None

    if old_config_hash != new_config_hash and result.embedding_config_hash is not None:
        # Config changed and there are existing embeddings
        if result.embedding_config_hash != new_config_hash:
            embedding_warning = (
                f"Embedding configuration changed from '{result.embedding_config_hash}' "
                f"to '{new_config_hash}'. Existing filesystem indexes are incompatible "
                "with the new configuration. You must perform a full re-index of all "
                "filesystem indexes for search to work correctly."
            )
            logger.warning(embedding_warning)

    # Invalidate cache and reinitialize RAG agent to pick up LLM/embedding changes
    invalidate_settings_cache()
    await rag.initialize()

    # Reset OCR semaphore if concurrency limit was changed
    if "ocr_concurrency_limit" in updates:
        from ragtime.core.ollama_concurrency import reset_ollama_semaphore

        reset_ollama_semaphore()
        logger.info(
            f"OCR concurrency limit changed to {updates['ocr_concurrency_limit']}, "
            "semaphore will reinitialize on next use"
        )

    # Notify MCP clients that tools may have changed (e.g., aggregate_search toggle)
    notify_tools_changed()

    return UpdateSettingsResponse(settings=result, embedding_warning=embedding_warning)


@router.get(
    "/settings/embedding-status", response_model=EmbeddingStatus, tags=["Settings"]
)
async def get_embedding_status(_user: User = Depends(require_admin)):
    """
    Get embedding configuration status.

    Returns information about the current embedding provider/model and whether
    it's compatible with existing filesystem indexes.
    """
    settings = await repository.get_settings()

    current_hash = settings.get_embedding_config_hash()
    has_mismatch = settings.has_embedding_config_changed()
    requires_reindex = has_mismatch and settings.embedding_config_hash is not None

    if requires_reindex:
        message = (
            f"Embedding configuration mismatch: indexes use '{settings.embedding_config_hash}' "
            f"but current config is '{current_hash}'. Full re-index required."
        )
    elif settings.embedding_config_hash is None:
        message = "No filesystem indexes exist yet. First index will set the embedding configuration."
    else:
        message = f"Embedding configuration matches existing indexes ({current_hash})."

    return EmbeddingStatus(
        current_provider=settings.embedding_provider,
        current_model=settings.embedding_model,
        current_config_hash=current_hash,
        stored_config_hash=settings.embedding_config_hash,
        stored_dimension=settings.embedding_dimension,
        has_mismatch=has_mismatch,
        requires_reindex=requires_reindex,
        message=message,
    )


# -----------------------------------------------------------------------------
# Tool Configuration Endpoints (Admin only)
# -----------------------------------------------------------------------------


class SSHKeyPairResponse(BaseModel):
    """Response containing generated SSH keypair."""

    private_key: str
    public_key: str
    fingerprint: str


class ToolTestResponse(BaseModel):
    """Response from a tool connection test."""

    success: bool
    message: str
    details: Optional[dict] = None


@router.get("/tools", response_model=List[ToolConfig], tags=["Tools"])
async def list_tool_configs(
    enabled_only: bool = False, _user: User = Depends(require_admin)
):
    """List all tool configurations. Admin only."""
    return await repository.list_tool_configs(enabled_only=enabled_only)


# =========================================================================
# Tool Groups
# =========================================================================


@router.get("/tool-groups", response_model=List[ToolGroup], tags=["Tool Groups"])
async def list_tool_groups(_user: User = Depends(require_admin)):
    """List all tool groups. Admin only."""
    return await repository.list_tool_groups()


@router.post("/tool-groups", response_model=ToolGroup, tags=["Tool Groups"])
async def create_tool_group(
    request: CreateToolGroupRequest, _user: User = Depends(require_admin)
):
    """Create a tool group. Admin only."""
    return await repository.create_tool_group(
        name=request.name,
        description=request.description,
        sort_order=request.sort_order,
    )


@router.put("/tool-groups/{group_id}", response_model=ToolGroup, tags=["Tool Groups"])
async def update_tool_group(
    group_id: str,
    request: UpdateToolGroupRequest,
    _user: User = Depends(require_admin),
):
    """Update a tool group. Admin only."""
    updates = request.model_dump(exclude_unset=True)
    result = await repository.update_tool_group(group_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Tool group not found")
    return result


@router.delete("/tool-groups/{group_id}", tags=["Tool Groups"])
async def delete_tool_group(group_id: str, _user: User = Depends(require_admin)):
    """Delete a tool group. Tools in it become ungrouped. Admin only."""
    success = await repository.delete_tool_group(group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool group not found")
    return {"message": "Tool group deleted"}


@router.post("/tools", response_model=ToolConfig, tags=["Tools"])
async def create_tool_config(
    request: CreateToolConfigRequest, _user: User = Depends(require_admin)
):
    """Create a new tool configuration. Admin only."""
    connection_config = request.connection_config.copy()

    # Sanitize integer fields in connection_config
    # This addresses an issue where the frontend might send them as strings (storing "24" in JSON)
    # causing runtime type errors when consumed by backend services
    int_fields = [
        "port",
        "ssh_tunnel_port",
        "ssh_port",
        "chunk_size",
        "chunk_overlap",
        "max_file_size_mb",
        "max_total_files",
        "schema_index_interval_hours",
        "reindex_interval_hours",
    ]

    for field in int_fields:
        if field in connection_config:
            try:
                # Convert to int if possible associated with numeric fields
                connection_config[field] = int(connection_config[field])
            except (ValueError, TypeError):
                pass  # Keep original value if conversion fails

    # For filesystem indexers, ensure index_name is sanitized for safe filesystem/DB usage
    if request.tool_type == ToolType.FILESYSTEM_INDEXER:
        # Sanitize the index_name to prevent issues with spaces/special chars
        raw_index_name = connection_config.get("index_name", request.name)
        sanitized_index_name = safe_tool_name(raw_index_name or request.name)
        connection_config["index_name"] = sanitized_index_name

        vector_store = connection_config.get("vector_store_type", "pgvector")
        if vector_store == "faiss":
            # Check if a document index with this name already exists
            from ragtime.indexer.vector_backends import FAISS_INDEX_BASE_PATH

            index_path = FAISS_INDEX_BASE_PATH / sanitized_index_name
            if index_path.exists():
                raise HTTPException(
                    status_code=409,
                    detail=f"An index named '{sanitized_index_name}' already exists",
                )

    config = ToolConfig(
        name=request.name,
        tool_type=request.tool_type,
        description=request.description,
        connection_config=connection_config,
        max_results=request.max_results,
        timeout=request.timeout,
        timeout_max_seconds=request.timeout_max_seconds,
        allow_write=request.allow_write,
        group_id=request.group_id,
    )
    result = await repository.create_tool_config(config)

    # Reinitialize RAG agent to make the new tool available immediately
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed
    notify_tools_changed()

    # Auto-start indexing for filesystem_indexer tools (unless skip_indexing requested)
    if request.tool_type == ToolType.FILESYSTEM_INDEXER and not request.skip_indexing:
        try:
            # Use the sanitized connection_config
            fs_config = FilesystemConnectionConfig(**connection_config)
            await filesystem_indexer.trigger_index(
                tool_config_id=result.id,
                config=fs_config,
                full_reindex=True,
            )
            logger.info(f"Auto-started indexing for new filesystem tool: {result.name}")
        except Exception as e:
            # Log error but don't fail the tool creation
            logger.warning(f"Failed to auto-start indexing for {result.name}: {e}")

    return result


# Heartbeat models - must be defined before the route
class HeartbeatStatus(BaseModel):
    """Heartbeat status for a single tool."""

    tool_id: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: str


class HeartbeatResponse(BaseModel):
    """Response from batch heartbeat check."""

    statuses: dict[str, HeartbeatStatus]


@router.get("/tools/heartbeat", response_model=HeartbeatResponse, tags=["Tools"])
async def check_tool_heartbeats(_user: User = Depends(require_admin)):
    """
    Check connection heartbeat for all enabled tools. Admin only.
    Returns quick connectivity status without updating database test results.
    Designed for frequent polling (every 10-30 seconds).
    """
    tools = await repository.list_tool_configs(enabled_only=True)
    statuses: dict[str, HeartbeatStatus] = {}

    async def check_single_tool(tool) -> HeartbeatStatus:
        """Check heartbeat for a single tool with timeout."""
        start_time = asyncio.get_event_loop().time()
        checked_at = datetime.now(timezone.utc).isoformat()
        heartbeat_timeout = get_heartbeat_timeout_seconds(tool.connection_config)

        try:
            # Quick ping-style check with short timeout
            result = await asyncio.wait_for(
                _heartbeat_check(tool.tool_type, tool.connection_config),
                timeout=heartbeat_timeout,
            )
            latency = (asyncio.get_event_loop().time() - start_time) * 1000

            return HeartbeatStatus(
                tool_id=tool.id,
                alive=result.success,
                latency_ms=round(latency, 1) if result.success else None,
                error=result.message if not result.success else None,
                checked_at=checked_at,
            )
        except asyncio.TimeoutError:
            return HeartbeatStatus(
                tool_id=tool.id,
                alive=False,
                latency_ms=None,
                error=f"Heartbeat timeout ({int(heartbeat_timeout)}s)",
                checked_at=checked_at,
            )
        except Exception as e:
            return HeartbeatStatus(
                tool_id=tool.id,
                alive=False,
                latency_ms=None,
                error=str(e),
                checked_at=checked_at,
            )

    # Check all tools concurrently
    results = await asyncio.gather(
        *[check_single_tool(tool) for tool in tools], return_exceptions=True
    )

    for result in results:
        if isinstance(result, HeartbeatStatus):
            statuses[result.tool_id] = result

    return HeartbeatResponse(statuses=statuses)


@router.get("/tools/{tool_id}", response_model=ToolConfig, tags=["Tools"])
async def get_tool_config(tool_id: str, _user: User = Depends(require_admin)):
    """Get a specific tool configuration. Admin only."""
    config = await repository.get_tool_config(tool_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return config


@router.put("/tools/{tool_id}", response_model=ToolConfig, tags=["Tools"])
async def update_tool_config(
    tool_id: str, request: UpdateToolConfigRequest, _user: User = Depends(require_admin)
):
    """Update an existing tool configuration. Admin only.

    If the name is changed, this also updates all associated index names in
    embedding tables (schema_embeddings, pdm_embeddings, filesystem_embeddings, etc.)
    to maintain consistency.
    """
    updates = request.model_dump(exclude_unset=True)

    # Sanitize connection_config if present
    if "connection_config" in updates and isinstance(
        updates["connection_config"], dict
    ):
        cc = updates["connection_config"]
        # Sanitize integer fields in connection_config
        # This addresses an issue where the frontend might send them as strings (storing "24" in JSON)
        # causing runtime type errors when consumed by backend services
        int_fields = [
            "port",
            "ssh_tunnel_port",
            "ssh_port",
            "chunk_size",
            "chunk_overlap",
            "max_file_size_mb",
            "max_total_files",
            "schema_index_interval_hours",
            "reindex_interval_hours",
        ]

        for field in int_fields:
            if field in cc:
                try:
                    cc[field] = int(cc[field])
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
        updates["connection_config"] = cc

    # Capture the current config to detect changes (e.g., schema indexing enablement)
    original_config = await repository.get_tool_config(tool_id)
    if original_config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Check if name is being changed - if so, use rename_tool_config for consistency
    if "name" in updates and updates["name"] is not None:
        new_name = updates.pop("name")
        if new_name != original_config.name:
            # For filesystem_indexer with FAISS, get the old index name before rename
            old_index_name = None
            new_index_name = None
            is_faiss = False
            if original_config.tool_type == ToolType.FILESYSTEM_INDEXER:
                old_index_name = (original_config.connection_config or {}).get(
                    "index_name"
                )
                vector_store = (original_config.connection_config or {}).get(
                    "vector_store_type", "pgvector"
                )
                is_faiss = vector_store == "faiss"
                new_index_name = safe_tool_name(new_name)

                # Check for name conflicts with existing indexes
                if is_faiss and new_index_name != old_index_name:
                    from ragtime.indexer.vector_backends import \
                        FAISS_INDEX_BASE_PATH

                    new_path = FAISS_INDEX_BASE_PATH / new_index_name
                    if new_path.exists():
                        raise HTTPException(
                            status_code=409,
                            detail=f"An index named '{new_index_name}' already exists",
                        )

            # Use rename_tool_config for comprehensive name updates
            config, update_counts = await repository.rename_tool_config(
                tool_id, new_name
            )
            if config is None:
                raise HTTPException(
                    status_code=500, detail="Failed to rename tool configuration"
                )

            # For FAISS filesystem indexes, rename using the backend
            if is_faiss and old_index_name and new_index_name:
                if old_index_name != new_index_name:
                    from ragtime.indexer.vector_backends import \
                        get_faiss_backend

                    faiss_backend = get_faiss_backend()
                    success = await faiss_backend.rename_index(
                        old_index_name, new_index_name
                    )
                    if success:
                        # Also update RAG's index tracking
                        rag.rename_index(old_index_name, new_index_name)
                    else:
                        logger.warning(
                            f"FAISS index rename failed: {old_index_name} -> {new_index_name}"
                        )

            # Log any embedding updates
            total_updates = sum(update_counts.values())
            if total_updates > 0:
                logger.info(
                    f"Tool rename updated {total_updates} embedding records: {update_counts}"
                )

            # Apply any remaining updates
            if updates:
                config = await repository.update_tool_config(tool_id, updates)
                if config is None:
                    raise HTTPException(
                        status_code=404, detail="Tool configuration not found"
                    )
        else:
            # Name unchanged, just apply other updates
            config = await repository.update_tool_config(tool_id, updates)
            if config is None:
                raise HTTPException(
                    status_code=404, detail="Tool configuration not found"
                )
    else:
        config = await repository.update_tool_config(tool_id, updates)
        if config is None:
            raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Reinitialize RAG agent to pick up the tool changes
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed (e.g., renamed)
    notify_tools_changed()

    # Auto-trigger schema indexing when it is newly enabled on supported tools
    try:
        prev_enabled = bool(
            (original_config.connection_config or {}).get("schema_index_enabled", False)
        )
        new_enabled = bool(
            (config.connection_config or {}).get("schema_index_enabled", False)
        )
        if (
            config.tool_type.value in SCHEMA_INDEXER_CAPABLE_TYPES
            and new_enabled
            and not prev_enabled
        ):
            # Ensure pgvector is available; log instead of failing the update
            if await ensure_pgvector_extension(logger_override=logger):
                await schema_indexer.trigger_index(
                    tool_config_id=tool_id,
                    tool_type=config.tool_type.value,
                    connection_config=config.connection_config or {},
                    full_reindex=True,
                    tool_name=safe_tool_name(config.name) or None,
                )
                logger.info(
                    "Auto-triggered schema indexing after enabling on tool %s",
                    tool_id,
                )
            else:
                logger.warning(
                    "pgvector extension unavailable; skipping auto schema index for tool %s",
                    tool_id,
                )
    except Exception:
        logger.exception("Failed to auto-trigger schema indexing after update")

    return config


@router.delete("/tools/{tool_id}", tags=["Tools"])
async def delete_tool_config(tool_id: str, _user: User = Depends(require_admin)):
    """
    Delete a tool configuration. Admin only.

    Cleans up associated data:
    - For filesystem indexer tools: embeddings, file metadata, and unmounts shares
    - For postgres/mssql tools: schema embeddings
    - For SolidWorks PDM tools: PDM embeddings and document metadata
    - For Odoo tools: disconnects from Docker network if no other tools use it
    """
    # Get the tool config before deleting to check for cleanup
    tool = await repository.get_tool_config(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    tool_name = safe_tool_name(tool.name)

    # Check if this is an Odoo tool with a docker network
    docker_network = None
    if tool.tool_type == "odoo_shell" and tool.connection_config:
        docker_network = tool.connection_config.get("docker_network")

    # Cleanup embeddings BEFORE deleting tool config (while we still have the config)
    if tool.tool_type == "filesystem_indexer":
        try:
            # Get index name and vector store type from connection config
            index_name = (
                tool.connection_config.get("index_name")
                if tool.connection_config
                else None
            )
            vector_store_type_str = (
                tool.connection_config.get("vector_store_type")
                if tool.connection_config
                else None
            )
            vector_store_type = (
                VectorStoreType(vector_store_type_str)
                if vector_store_type_str
                else None
            )
            if index_name:
                deleted = await filesystem_indexer.delete_index(
                    index_name, vector_store_type=vector_store_type
                )
                logger.info(
                    f"Cleaned up {deleted} filesystem embeddings for tool {tool_id}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to cleanup filesystem embeddings for tool {tool_id}: {e}"
            )

    elif tool.tool_type in SCHEMA_INDEXER_CAPABLE_TYPES:
        try:
            success, msg = await schema_indexer.delete_index(tool_id, tool_name)
            if success:
                logger.info(f"Cleaned up schema embeddings for tool {tool_id}: {msg}")
        except Exception as e:
            logger.warning(
                f"Failed to cleanup schema embeddings for tool {tool_id}: {e}"
            )

    elif tool.tool_type == "solidworks_pdm":
        try:
            success, msg = await pdm_indexer.delete_index(tool_id)
            if success:
                logger.info(f"Cleaned up PDM embeddings for tool {tool_id}: {msg}")
        except Exception as e:
            logger.warning(f"Failed to cleanup PDM embeddings for tool {tool_id}: {e}")

    # Delete the tool (cascade deletes jobs)
    success = await repository.delete_tool_config(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Cleanup: unmount filesystem for filesystem indexer tools
    if tool.tool_type == "filesystem_indexer":
        try:
            await filesystem_indexer.unmount_tool(tool_id)
        except Exception as e:
            logger.warning(f"Failed to unmount filesystem for tool {tool_id}: {e}")

    # Cleanup: disconnect from network if no other tools need it
    if docker_network:
        # Check if any other odoo_shell tools use this network
        all_tools = await repository.list_tool_configs()
        network_still_needed = any(
            t.tool_type == "odoo_shell"
            and t.connection_config.get("docker_network") == docker_network
            and t.id != tool_id
            for t in all_tools
        )

        if not network_still_needed:
            # Disconnect from the network
            try:
                await disconnect_from_network(docker_network)
                logger.info(
                    f"Disconnected from network '{docker_network}' after tool deletion"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to disconnect from network '{docker_network}': {e}"
                )

    # Reinitialize RAG agent to remove the deleted tool
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed
    notify_tools_changed()

    return {"message": "Tool configuration deleted"}


@router.post("/tools/{tool_id}/toggle", tags=["Tools"])
async def toggle_tool_config(
    tool_id: str, enabled: bool, _user: User = Depends(require_admin)
):
    """Toggle a tool's enabled status. Admin only."""
    config = await repository.update_tool_config(tool_id, {"enabled": enabled})
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Invalidate cache and reinitialize RAG agent to pick up the change
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed
    notify_tools_changed()

    return config


@router.post("/tools/test", response_model=ToolTestResponse, tags=["Tools"])
async def test_tool_connection(
    request: ToolTestRequest, _user: User = Depends(require_admin)
):
    """
    Test a tool connection without saving. Admin only.
    Used during the wizard to validate connection settings.
    """
    tool_type = request.tool_type
    config = request.connection_config

    if tool_type == ToolType.POSTGRES:
        return await _test_postgres_connection(config)
    elif tool_type == ToolType.MSSQL:
        return await _test_mssql_connection(config)
    elif tool_type == ToolType.MYSQL:
        return await _test_mysql_connection(config)
    elif tool_type == ToolType.INFLUXDB:
        return await _test_influxdb_connection(config)
    elif tool_type == ToolType.ODOO_SHELL:
        return await _test_odoo_connection(config)
    elif tool_type == ToolType.SSH_SHELL:
        return await _test_ssh_connection(config)
    elif tool_type == ToolType.FILESYSTEM_INDEXER:
        return await _test_filesystem_connection(config)
    elif tool_type == ToolType.SOLIDWORKS_PDM:
        return await _test_pdm_connection(config)
    else:
        return ToolTestResponse(
            success=False, message=f"Unknown tool type: {tool_type}"
        )


async def _test_filesystem_connection(config: dict) -> ToolTestResponse:
    """Test filesystem path accessibility."""
    try:
        fs_config = FilesystemConnectionConfig(**config)
        result = await filesystem_indexer.validate_path_access(fs_config)

        if result["success"]:
            return ToolTestResponse(
                success=True,
                message=result["message"],
                details={
                    "file_count": result.get("file_count", 0),
                    "sample_files": result.get("sample_files", []),
                },
            )
        else:
            return ToolTestResponse(success=False, message=result["message"])
    except Exception as e:
        return ToolTestResponse(success=False, message=f"Configuration error: {str(e)}")


async def _test_pdm_connection(config: dict) -> ToolTestResponse:
    """Test SolidWorks PDM database connection."""

    host = config.get("host", "")
    port = config.get("port", 1433)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")

    if not all([host, user, password, database]):
        return ToolTestResponse(
            success=False, message="Missing required connection parameters"
        )

    def test_connection() -> tuple[bool, str, dict | None]:
        try:
            with mssql_connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                login_timeout=10,
                timeout=10,
            ) as conn:
                cursor = conn.cursor()

                # Test query to count SolidWorks documents
                cursor.execute(
                    """
                    SELECT COUNT(*) as doc_count FROM Documents
                    WHERE Filename LIKE '%.SLDPRT' OR Filename LIKE '%.SLDASM' OR Filename LIKE '%.SLDDRW'
                """
                )
                result = cursor.fetchone()
                doc_count = 0
                if isinstance(result, dict):
                    doc_count = int(result.get("doc_count", 0))
                elif result:
                    doc_count = int(result[0])

                return (
                    True,
                    f"Connected successfully. Found {doc_count} SolidWorks documents.",
                    {"document_count": doc_count},
                )

        except MssqlConnectionError as e:
            return False, str(e), None
        except Exception as e:
            return False, f"Connection error: {str(e)}", None

    try:
        success, message, details = await asyncio.to_thread(test_connection)
        return ToolTestResponse(success=success, message=message, details=details)
    except asyncio.TimeoutError:
        return ToolTestResponse(success=False, message="Connection timed out")
    except Exception as e:
        return ToolTestResponse(success=False, message=f"Error: {str(e)}")


@router.post(
    "/tools/postgres/discover", response_model=PostgresDiscoverResponse, tags=["Tools"]
)
async def discover_postgres_databases(
    request: PostgresDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover available databases on a PostgreSQL server. Admin only.
    Connects to the server and lists all accessible databases.
    Supports SSH tunnel for remote database access.
    """
    tunnel = None
    try:
        # Determine host and port (may be overridden by SSH tunnel)
        connect_host = request.host
        connect_port = request.port

        # Set up SSH tunnel if enabled
        if request.ssh_tunnel_enabled:
            tunnel_config_dict = {
                "host": request.host,
                "port": request.port,
                "ssh_tunnel_host": request.ssh_tunnel_host,
                "ssh_tunnel_port": request.ssh_tunnel_port,
                "ssh_tunnel_user": request.ssh_tunnel_user,
                "ssh_tunnel_password": request.ssh_tunnel_password,
                "ssh_tunnel_key_path": request.ssh_tunnel_key_path,
                "ssh_tunnel_key_content": request.ssh_tunnel_key_content,
                "ssh_tunnel_key_passphrase": request.ssh_tunnel_key_passphrase,
            }
            tunnel_config = ssh_tunnel_config_from_dict(
                tunnel_config_dict, default_remote_port=5432
            )
            tunnel = SSHTunnel(tunnel_config)
            tunnel.start()
            # Connect through the tunnel's local port
            connect_host = "127.0.0.1"
            connect_port = tunnel.local_port
            logger.info(
                f"SSH tunnel established for PostgreSQL discovery: localhost:{connect_port}"
            )

        # Connect to 'postgres' system database to list all databases
        cmd = [
            "psql",
            "-h",
            connect_host,
            "-p",
            str(connect_port),
            "-U",
            request.user,
            "-d",
            "postgres",  # Connect to system database
            "-t",  # Tuples only (no headers/footers)
            "-A",  # Unaligned output
            "-F",
            "\t",  # Tab-separated output for reliable parsing
            "-c",
            (
                "SELECT datname, datallowconn, "
                "has_database_privilege(current_user, datname, 'CONNECT') "
                "FROM pg_database "
                "WHERE datistemplate = false "
                "ORDER BY datname;"
            ),
        ]
        env = {"PGPASSWORD": request.password}

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            ),
            timeout=15.0,
        )
        _stdout, stderr = await process.communicate()

        if process.returncode == 0:
            # Parse database rows and compute per-db access.
            output = _stdout.decode("utf-8", errors="replace").strip()
            database_options: list[DatabaseDiscoverOption] = []
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if not parts:
                    continue

                name = parts[0].strip()
                if not name:
                    continue

                allow_conn = True
                has_connect = True
                if len(parts) > 1:
                    allow_conn = parts[1].strip().lower() in {"t", "true", "1"}
                if len(parts) > 2:
                    has_connect = parts[2].strip().lower() in {"t", "true", "1"}

                accessible = allow_conn and has_connect
                if not allow_conn:
                    access_error = "Database is configured to disallow new connections"
                elif not has_connect:
                    access_error = "User lacks CONNECT privilege on this database"
                else:
                    access_error = None

                database_options.append(
                    DatabaseDiscoverOption(
                        name=name,
                        accessible=accessible,
                        access_error=access_error,
                    )
                )

            databases = [db.name for db in database_options if db.accessible]
            return PostgresDiscoverResponse(
                success=True,
                databases=databases,
                database_options=database_options,
            )
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return PostgresDiscoverResponse(
                success=False,
                databases=[],
                database_options=[],
                error=f"Connection failed: {error}",
            )

    except asyncio.TimeoutError:
        return PostgresDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error="Connection timed out after 15 seconds",
        )
    except FileNotFoundError:
        return PostgresDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error="psql command not found",
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return PostgresDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error=f"Discovery failed: {error_msg}",
        )
    finally:
        if tunnel:
            tunnel.stop()


@router.post(
    "/tools/mssql/discover", response_model=MssqlDiscoverResponse, tags=["Tools"]
)
async def discover_mssql_databases(
    request: MssqlDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover available databases on an MSSQL server. Admin only.
    Connects to the server and lists all accessible databases.
    Supports SSH tunnel for remote database access.
    """
    tunnel = None

    def discover_databases(
        connect_host: str, connect_port: int
    ) -> tuple[bool, list[DatabaseDiscoverOption], str | None]:
        try:
            with mssql_connect(
                host=connect_host,
                port=connect_port,
                user=request.user,
                password=request.password,
                database="master",  # Connect to master to list databases
                login_timeout=10,
                timeout=15,
            ) as conn:
                cursor = conn.cursor()

                # Discover databases and whether this login can access each one.
                cursor.execute(
                    """
                    SELECT
                        name,
                        CASE WHEN HAS_DBACCESS(name) = 1 THEN 1 ELSE 0 END AS has_access
                    FROM sys.databases
                    WHERE database_id > 4  -- Exclude system databases
                    ORDER BY name
                    """
                )
                rows = cursor.fetchall() or []
                database_options: list[DatabaseDiscoverOption] = []
                for row in rows:
                    if isinstance(row, dict):
                        name = row.get("name")
                        has_access = row.get("has_access")
                    elif row and len(row) > 0:
                        name = row[0]
                        has_access = row[1] if len(row) > 1 else 1
                    else:
                        name = None
                        has_access = 0
                    if name:
                        accessible = int(has_access or 0) == 1
                        database_options.append(
                            DatabaseDiscoverOption(
                                name=str(name),
                                accessible=accessible,
                                access_error=(
                                    "Login does not have access to this database"
                                    if not accessible
                                    else None
                                ),
                            )
                        )
                return True, database_options, None

        except MssqlConnectionError as e:
            return (
                False,
                [],
                normalize_mssql_error_message(str(e), database="master"),
            )
        except Exception as e:
            return (
                False,
                [],
                normalize_mssql_error_message(str(e), database="master"),
            )

    try:
        # Determine host and port (may be overridden by SSH tunnel)
        connect_host = request.host
        connect_port = request.port

        # Set up SSH tunnel if enabled
        if request.ssh_tunnel_enabled:
            tunnel_config_dict = {
                "host": request.host,
                "port": request.port,
                "ssh_tunnel_host": request.ssh_tunnel_host,
                "ssh_tunnel_port": request.ssh_tunnel_port,
                "ssh_tunnel_user": request.ssh_tunnel_user,
                "ssh_tunnel_password": request.ssh_tunnel_password,
                "ssh_tunnel_key_path": request.ssh_tunnel_key_path,
                "ssh_tunnel_key_content": request.ssh_tunnel_key_content,
                "ssh_tunnel_key_passphrase": request.ssh_tunnel_key_passphrase,
            }
            tunnel_config = ssh_tunnel_config_from_dict(
                tunnel_config_dict, default_remote_port=1433
            )
            tunnel = SSHTunnel(tunnel_config)
            tunnel.start()
            connect_host = "127.0.0.1"
            connect_port = tunnel.local_port
            logger.info(
                f"SSH tunnel established for MSSQL discovery: localhost:{connect_port}"
            )

        success, database_options, error = await asyncio.to_thread(
            discover_databases, connect_host, connect_port
        )
        accessible_databases = [d.name for d in database_options if d.accessible]
        return MssqlDiscoverResponse(
            success=success,
            databases=accessible_databases,
            database_options=database_options,
            error=error,
        )
    except asyncio.TimeoutError:
        return MssqlDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error="Connection timed out",
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return MssqlDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error=f"Discovery failed: {error_msg}",
        )
    finally:
        if tunnel:
            tunnel.stop()


@router.post(
    "/tools/mysql/discover", response_model=MysqlDiscoverResponse, tags=["Tools"]
)
async def discover_mysql_databases(
    request: MysqlDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover available databases on a MySQL/MariaDB server. Admin only.
    Connects to the server and lists all accessible databases.
    Supports direct connections, Docker containers, and SSH tunnels.
    """
    tunnel = None

    def discover_databases(
        connect_host: str, connect_port: int
    ) -> tuple[bool, list[DatabaseDiscoverOption], str | None]:
        try:
            direct_user = request.user or ""
            direct_password = request.password or ""

            def mysql_access_error(err: Exception | str) -> str:
                msg = str(err).strip()
                lower = msg.lower()
                if "access denied" in lower:
                    return "Login does not have access to this database"
                if "unknown database" in lower:
                    return "Database not found"
                return msg or "Connection failed"

            # Handle Docker container mode
            if request.container:
                container = request.container

                # Get credentials from container environment
                def get_env_var(var_name: str) -> str | None:
                    try:
                        result = subprocess.run(
                            ["docker", "exec", container, "printenv", var_name],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            check=False,
                        )
                        return result.stdout.strip() if result.returncode == 0 else None
                    except Exception:
                        return None

                # Try MYSQL_USER, fallback to root
                user = get_env_var("MYSQL_USER") or "root"
                # Try MYSQL_PASSWORD for regular user, MYSQL_ROOT_PASSWORD for root
                if user == "root":
                    password = get_env_var("MYSQL_ROOT_PASSWORD") or ""
                else:
                    password = get_env_var("MYSQL_PASSWORD") or ""

                # Execute mysql command inside container to list databases
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        container,
                        "mysql",
                        f"-u{user}",
                        f"-p{password}",
                        "-N",
                        "-e",
                        "SHOW DATABASES",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                if result.returncode != 0:
                    return False, [], f"MySQL command failed: {result.stderr}"

                # Parse database list, excluding system databases
                system_dbs = {
                    "information_schema",
                    "mysql",
                    "performance_schema",
                    "sys",
                }
                databases = [
                    db.strip()
                    for db in result.stdout.strip().split("\n")
                    if db.strip() and db.strip().lower() not in system_dbs
                ]
                container_database_options: list[DatabaseDiscoverOption] = []
                for db in databases:
                    check_result = subprocess.run(
                        [
                            "docker",
                            "exec",
                            container,
                            "mysql",
                            f"-u{user}",
                            f"-p{password}",
                            "-D",
                            db,
                            "-N",
                            "-e",
                            "SELECT 1",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    if check_result.returncode == 0:
                        container_database_options.append(
                            DatabaseDiscoverOption(name=db, accessible=True)
                        )
                    else:
                        err_text = (
                            check_result.stderr or check_result.stdout or ""
                        ).strip()
                        container_database_options.append(
                            DatabaseDiscoverOption(
                                name=db,
                                accessible=False,
                                access_error=mysql_access_error(err_text),
                            )
                        )

                return True, container_database_options, None

            # Direct connection mode (or through SSH tunnel)
            try:
                with mysql_connect(
                    host=connect_host,
                    port=connect_port,
                    user=direct_user,
                    password=direct_password,
                    database="information_schema",  # Connect to information_schema to list databases
                    connect_timeout=15,
                ) as conn:
                    cursor = conn.cursor()

                    # Query for non-system databases
                    cursor.execute(
                        """
                        SELECT SCHEMA_NAME FROM information_schema.SCHEMATA
                        WHERE SCHEMA_NAME NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                        ORDER BY SCHEMA_NAME
                        """
                    )
                    rows = cursor.fetchall() or []
                    db_list: list[str] = []
                    for row in rows:
                        if not row:
                            continue

                        name = None
                        if isinstance(row, dict):
                            name = row.get("SCHEMA_NAME")
                            if name is None and row:
                                # Fallback to first value in dict
                                name = next(iter(row.values()), None)
                        elif isinstance(row, (list, tuple)):
                            if len(row) > 0:
                                name = row[0]
                        else:
                            # Fallback: try attribute or str
                            try:
                                name = row[0]  # type: ignore[index]
                            except Exception:
                                name = str(row)

                        if name:
                            db_list.append(str(name))

                    direct_database_options: list[DatabaseDiscoverOption] = []
                    for db_name in db_list:
                        try:
                            with mysql_connect(
                                host=connect_host,
                                port=connect_port,
                                user=direct_user,
                                password=direct_password,
                                database=db_name,
                                connect_timeout=10,
                            ):
                                pass
                            direct_database_options.append(
                                DatabaseDiscoverOption(name=db_name, accessible=True)
                            )
                        except Exception as check_error:
                            direct_database_options.append(
                                DatabaseDiscoverOption(
                                    name=db_name,
                                    accessible=False,
                                    access_error=mysql_access_error(check_error),
                                )
                            )

                    return True, direct_database_options, None
            except Exception as e:
                # Fallback: if global discovery fails, try connecting to the specific database if provided
                if request.database:
                    try:
                        with mysql_connect(
                            host=connect_host,
                            port=connect_port,
                            user=direct_user,
                            password=direct_password,
                            database=request.database,
                            connect_timeout=10,
                        ):
                            pass
                        return (
                            True,
                            [
                                DatabaseDiscoverOption(
                                    name=request.database,
                                    accessible=True,
                                )
                            ],
                            None,
                        )
                    except Exception as fallback_error:
                        # Prefer the fallback error message if it is more descriptive
                        if str(fallback_error):
                            e = fallback_error

                # Log the discovery failure for visibility in container logs
                logger.error(
                    "MySQL discovery failed",
                    exc_info=e,
                    extra={
                        "host": connect_host,
                        "port": connect_port,
                        "user": request.user,
                        "database": request.database,
                        "ssh_tunnel": bool(request.ssh_tunnel_enabled),
                    },
                )

                error_msg = str(e) or "Unknown error"
                if getattr(e, "args", None):
                    arg_parts = [str(a) for a in e.args if str(a)]
                    if arg_parts:
                        error_msg = " | ".join(arg_parts)

                return False, [], f"Connection error: {error_msg}"

        except MysqlConnectionError as e:
            return False, [], str(e)
        except Exception as e:
            error_msg = str(e) or "Unknown error"
            if getattr(e, "args", None):
                arg_parts = [str(a) for a in e.args if str(a)]
                if arg_parts:
                    error_msg = " | ".join(arg_parts)
            # Log the outer exception to help diagnose opaque UI errors
            logger.error(
                "MySQL discovery encountered an unexpected error",
                exc_info=e,
                extra={
                    "host": connect_host,
                    "port": connect_port,
                    "user": request.user,
                    "database": request.database,
                    "ssh_tunnel": bool(request.ssh_tunnel_enabled),
                },
            )
            return False, [], f"Connection error: {error_msg}"

    try:
        # Determine host and port (may be overridden by SSH tunnel)
        connect_host = request.host or "127.0.0.1"
        connect_port = request.port

        # Set up SSH tunnel if enabled
        if request.ssh_tunnel_enabled:
            tunnel_config_dict = {
                "host": request.host or "127.0.0.1",
                "port": request.port,
                "ssh_tunnel_host": request.ssh_tunnel_host,
                "ssh_tunnel_port": request.ssh_tunnel_port,
                "ssh_tunnel_user": request.ssh_tunnel_user,
                "ssh_tunnel_password": request.ssh_tunnel_password,
                "ssh_tunnel_key_path": request.ssh_tunnel_key_path,
                "ssh_tunnel_key_content": request.ssh_tunnel_key_content,
                "ssh_tunnel_key_passphrase": request.ssh_tunnel_key_passphrase,
            }
            tunnel_config = ssh_tunnel_config_from_dict(
                tunnel_config_dict, default_remote_port=3306
            )
            tunnel = SSHTunnel(tunnel_config)
            tunnel.start()
            connect_host = "127.0.0.1"
            connect_port = tunnel.local_port
            logger.info(
                f"SSH tunnel established for MySQL discovery: localhost:{connect_port}"
            )

        success, database_options, error = await asyncio.to_thread(
            discover_databases, connect_host, connect_port
        )
        databases = [db.name for db in database_options if db.accessible]
        return MysqlDiscoverResponse(
            success=success,
            databases=databases,
            database_options=database_options,
            error=error,
        )
    except asyncio.TimeoutError:
        return MysqlDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error="Connection timed out",
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return MysqlDiscoverResponse(
            success=False,
            databases=[],
            database_options=[],
            error=f"Discovery failed: {error_msg}",
        )
    finally:
        if tunnel:
            tunnel.stop()


@router.post(
    "/tools/influxdb/discover",
    response_model=InfluxdbDiscoverResponse,
    tags=["Tools"],
)
async def discover_influxdb_buckets(
    request: InfluxdbDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover available buckets on an InfluxDB server. Admin only.
    Supports direct connections and SSH tunnels.
    """
    tunnel = None

    def discover_buckets(effective_url: str) -> tuple[bool, list[str], str | None]:
        try:
            from influxdb_client import \
                InfluxDBClient  # type: ignore[import-untyped]
        except ImportError:
            return False, [], "influxdb-client package not installed"

        if not request.org:
            return (
                False,
                [],
                "Organization is required to discover buckets. Enter the org name and try again.",
            )

        client = None
        try:
            client = InfluxDBClient(
                url=effective_url,
                token=request.token,
                org=request.org,
                timeout=15000,
            )
            # Prefer the Flux buckets() query over the management API
            # because read-only tokens typically cannot call the management
            # endpoint but CAN run Flux queries.
            try:
                tables = client.query_api().query("buckets()", org=request.org)
                buckets = sorted(
                    {
                        r.values.get("name", "")
                        for t in tables
                        for r in t.records
                        if r.values.get("name")
                    }
                )
                return True, buckets, None
            except Exception:
                pass

            # Fallback: management API (requires broader permissions)
            buckets_response = client.buckets_api().find_buckets(org=request.org)
            buckets = sorted(
                {b.name for b in (buckets_response.buckets or []) if b.name}
            )
            return True, buckets, None

        except Exception as e:
            err = str(e)
            if "401" in err or "unauthorized" in err.lower():
                return (
                    False,
                    [],
                    "Token authentication failed — check the token and organization name",
                )
            if "not found" in err.lower() or "organization" in err.lower():
                return (
                    False,
                    [],
                    f"Organization '{request.org}' not found on this InfluxDB server",
                )
            return False, [], err
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    try:
        if not request.host:
            return InfluxdbDiscoverResponse(
                success=False,
                buckets=[],
                database_options=[],
                error="Host is required",
            )
        if not request.token:
            return InfluxdbDiscoverResponse(
                success=False,
                buckets=[],
                database_options=[],
                error="Token is required",
            )
        if not request.org:
            return InfluxdbDiscoverResponse(
                success=False,
                buckets=[],
                database_options=[],
                error="Organization is required",
            )

        host = request.host
        port = request.port
        scheme = "https" if request.use_https else "http"
        effective_url = f"{scheme}://{host}:{port}"

        if request.ssh_tunnel_enabled:
            tunnel_config_dict = {
                "host": host,
                "port": port,
                "ssh_tunnel_host": request.ssh_tunnel_host,
                "ssh_tunnel_port": request.ssh_tunnel_port,
                "ssh_tunnel_user": request.ssh_tunnel_user,
                "ssh_tunnel_password": request.ssh_tunnel_password,
                "ssh_tunnel_key_path": request.ssh_tunnel_key_path,
                "ssh_tunnel_key_content": request.ssh_tunnel_key_content,
                "ssh_tunnel_key_passphrase": request.ssh_tunnel_key_passphrase,
            }
            tunnel_config = ssh_tunnel_config_from_dict(
                tunnel_config_dict, default_remote_port=port
            )
            tunnel = SSHTunnel(tunnel_config)
            tunnel.start()
            effective_url = f"{scheme}://127.0.0.1:{tunnel.local_port}"
            logger.info(
                f"SSH tunnel established for InfluxDB discovery: localhost:{tunnel.local_port}"
            )

        success, buckets, error = await asyncio.to_thread(
            discover_buckets, effective_url
        )
        options = [
            DatabaseDiscoverOption(name=name, accessible=True, access_error=None)
            for name in buckets
        ]

        return InfluxdbDiscoverResponse(
            success=success,
            buckets=buckets,
            database_options=options,
            error=error,
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return InfluxdbDiscoverResponse(
            success=False,
            buckets=[],
            database_options=[],
            error=f"Discovery failed: {error_msg}",
        )
    finally:
        if tunnel:
            tunnel.stop()


@router.post("/tools/pdm/discover", response_model=PdmDiscoverResponse, tags=["Tools"])
async def discover_pdm_schema(
    request: PdmDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover PDM schema metadata from a SolidWorks PDM database. Admin only.
    Returns available file extensions and variable names.
    Supports SSH tunnel connections for remote database access.
    """
    tunnel = None
    try:
        # Set up connection parameters
        connect_host = request.host
        connect_port = request.port

        # Start SSH tunnel if enabled
        if request.ssh_tunnel_enabled:
            tunnel_config_dict = {
                "host": request.host,  # Remote endpoint from SSH server's perspective
                "port": request.port,
                "ssh_tunnel_host": request.ssh_tunnel_host,
                "ssh_tunnel_port": request.ssh_tunnel_port,
                "ssh_tunnel_user": request.ssh_tunnel_user,
                "ssh_tunnel_password": request.ssh_tunnel_password,
                "ssh_tunnel_key_path": request.ssh_tunnel_key_path,
                "ssh_tunnel_key_content": request.ssh_tunnel_key_content,
                "ssh_tunnel_key_passphrase": request.ssh_tunnel_key_passphrase,
            }
            tunnel_config = ssh_tunnel_config_from_dict(
                tunnel_config_dict,
                default_remote_port=1433,  # MSSQL default port
            )
            tunnel = SSHTunnel(tunnel_config)
            tunnel.start()
            connect_host = "127.0.0.1"
            connect_port = tunnel.local_port

        def discover_schema(
            host: str, port: int
        ) -> tuple[bool, list[str], list[str], int, str | None]:
            """Returns (success, file_extensions, variable_names, doc_count, error)."""
            try:
                with mssql_connect(
                    host=host,
                    port=port,
                    user=request.user,
                    password=request.password,
                    database=request.database,
                    login_timeout=10,
                    timeout=30,
                ) as conn:
                    cursor = conn.cursor()

                    # Get distinct file extensions from Documents table
                    # Use RIGHT() with CHARINDEX on reversed string to get extension from end
                    # Filter out configuration markers containing angle brackets
                    cursor.execute(
                        """
                        SELECT DISTINCT
                            UPPER(RIGHT(Filename, CHARINDEX('.', REVERSE(Filename)))) AS Extension
                        FROM Documents
                        WHERE Filename LIKE '%.%'
                            AND Deleted = 0
                            AND CHARINDEX('.', REVERSE(Filename)) > 1
                            AND CHARINDEX('.', REVERSE(Filename)) <= 10
                            AND Filename NOT LIKE '%<%'
                        ORDER BY Extension
                        """
                    )
                    ext_rows = cursor.fetchall() or []
                    extensions = []
                    for row in ext_rows:
                        ext_value = None
                        if isinstance(row, dict):
                            ext_value = row.get("Extension")
                        elif row and len(row) > 0:
                            ext_value = row[0]
                        if ext_value:
                            extensions.append(str(ext_value))

                    # Get all variable names from Variable table
                    # Filter out GUID-like names (patterns like {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx})
                    cursor.execute(
                        """
                        SELECT VariableName FROM Variable
                        WHERE VariableName NOT LIKE '{________-____-____-____-____________}'
                            AND VariableName NOT LIKE '{%-%-%-%-%}'
                        ORDER BY VariableName
                        """
                    )
                    var_rows = cursor.fetchall() or []
                    variables = []
                    for row in var_rows:
                        var_value = None
                        if isinstance(row, dict):
                            var_value = row.get("VariableName")
                        elif row and len(row) > 0:
                            var_value = row[0]
                        if var_value:
                            variables.append(str(var_value))

                    # Get total document count
                    cursor.execute(
                        "SELECT COUNT(*) as cnt FROM Documents WHERE Deleted = 0"
                    )
                    result = cursor.fetchone()
                    if isinstance(result, dict):
                        doc_count = int(result.get("cnt", 0))
                    elif result:
                        doc_count = int(result[0])
                    else:
                        doc_count = 0

                    return True, extensions, variables, doc_count, None

            except MssqlConnectionError as e:
                return False, [], [], 0, str(e)
            except Exception as e:
                return False, [], [], 0, f"Connection error: {str(e)}"

        success, extensions, variables, doc_count, error = await asyncio.to_thread(
            discover_schema, connect_host, connect_port
        )
        return PdmDiscoverResponse(
            success=success,
            file_extensions=extensions,
            variable_names=variables,
            document_count=doc_count,
            error=error,
        )
    except asyncio.TimeoutError:
        return PdmDiscoverResponse(success=False, error="Connection timed out")
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return PdmDiscoverResponse(
            success=False, error=f"Discovery failed: {error_msg}"
        )
    finally:
        if tunnel:
            tunnel.stop()


@router.post(
    "/tools/ssh/generate-keypair", response_model=SSHKeyPairResponse, tags=["Tools"]
)
async def generate_ssh_keypair(
    comment: str = "ragtime", passphrase: str = "", _user: User = Depends(require_admin)
):
    """
    Generate a new SSH keypair for use with remote connections. Admin only.
    Returns private key, public key, and fingerprint.
    The private key should be stored in the tool's connection config.
    The public key should be added to the remote server's authorized_keys.

    Args:
        comment: Comment for the key (appears in public key)
        passphrase: Optional passphrase to encrypt the private key
    """

    def _generate_keypair() -> tuple[str, str, str]:
        """Generate keypair in thread to avoid blocking event loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "id_rsa")

            # Generate RSA keypair using ssh-keygen
            process = subprocess.run(
                [
                    "ssh-keygen",
                    "-t",
                    "rsa",
                    "-b",
                    "4096",
                    "-f",
                    key_path,
                    "-N",
                    passphrase,
                    "-C",
                    comment,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            if process.returncode != 0:
                raise RuntimeError(f"Failed to generate keypair: {process.stderr}")

            # Read the generated keys
            with open(key_path, "r", encoding="utf-8") as f:
                private_key = f.read()
            with open(f"{key_path}.pub", "r", encoding="utf-8") as f:
                public_key = f.read().strip()

            # Get fingerprint
            fp_process = subprocess.run(
                ["ssh-keygen", "-lf", key_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            fingerprint = (
                fp_process.stdout.strip() if fp_process.returncode == 0 else "unknown"
            )

            return private_key, public_key, fingerprint

    try:
        # Run keypair generation in thread to avoid blocking event loop
        private_key, public_key, fingerprint = await asyncio.to_thread(
            _generate_keypair
        )
        return SSHKeyPairResponse(
            private_key=private_key, public_key=public_key, fingerprint=fingerprint
        )

    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail="Key generation timed out") from e
    except Exception as e:
        logger.exception("Failed to generate SSH keypair")
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Filesystem Indexer Routes (must be before {tool_id} routes)
# =============================================================================


class FilesystemTestResponse(BaseModel):
    """Response from filesystem path test."""

    success: bool
    message: str
    file_count: Optional[int] = None
    sample_files: Optional[List[str]] = None


@router.post(
    "/tools/filesystem/test",
    response_model=FilesystemTestResponse,
    tags=["Filesystem Indexer"],
)
async def test_filesystem_path(
    config: FilesystemConnectionConfig, _user: User = Depends(require_admin)
):
    """
    Test filesystem path accessibility. Admin only.

    Validates that the configured path is accessible and counts matching files.
    Used during the wizard to validate configuration.
    """
    result = await filesystem_indexer.validate_path_access(config)
    return FilesystemTestResponse(
        success=result["success"],
        message=result["message"],
        file_count=result.get("file_count"),
        sample_files=result.get("sample_files"),
    )


@router.post("/tools/{tool_id}/test", response_model=ToolTestResponse, tags=["Tools"])
async def test_saved_tool_connection(
    tool_id: str, _user: User = Depends(require_admin)
):
    """Test the connection for a saved tool configuration."""
    config = await repository.get_tool_config(tool_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Test the connection
    result = await test_tool_connection(
        ToolTestRequest(
            tool_type=config.tool_type, connection_config=config.connection_config
        )
    )

    # Update test results in database
    await repository.update_tool_test_result(
        tool_id,
        success=result.success,
        error=result.message if not result.success else None,
    )

    return result


async def _heartbeat_check(tool_type, config: dict) -> ToolTestResponse:
    """
    Quick heartbeat check for a tool connection.
    Uses minimal queries/commands to verify connectivity.
    """
    # Normalize tool_type to string value (handle enum or string)
    if hasattr(tool_type, "value"):
        tool_type_str = tool_type.value
    else:
        tool_type_str = str(tool_type)

    if tool_type_str == "postgres":
        return await _heartbeat_postgres(config)
    elif tool_type_str == "mysql":
        return await _heartbeat_mysql(config)
    elif tool_type_str == "mssql":
        return await _heartbeat_mssql(config)
    elif tool_type_str == "influxdb":
        return await _heartbeat_influxdb(config)
    elif tool_type_str == "odoo_shell":
        return await _heartbeat_odoo(config)
    elif tool_type_str == "ssh_shell":
        return await _heartbeat_ssh(config)
    elif tool_type_str == "filesystem_indexer":
        return await _heartbeat_filesystem(config)
    elif tool_type_str == "solidworks_pdm":
        return await _heartbeat_pdm(config)
    else:
        return ToolTestResponse(
            success=False, message=f"Unknown tool type: {tool_type_str}"
        )


def _coerce_int(value, default: int) -> int:
    """Best-effort int conversion for heartbeat inputs."""
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            return int(stripped)
        return int(value)
    except (TypeError, ValueError):
        return default


def _start_ssh_tunnel_if_enabled(
    config: dict, host: str, port: int
) -> tuple[Optional["SSHTunnel"], str, int]:
    """Start SSH tunnel if enabled and return (tunnel, host, port)."""
    tunnel_cfg_dict = build_ssh_tunnel_config(config, host, port)
    if not tunnel_cfg_dict:
        return None, host, port

    tunnel_config = ssh_tunnel_config_from_dict(
        tunnel_cfg_dict, default_remote_port=port
    )
    tunnel = SSHTunnel(tunnel_config)
    tunnel.start()
    return tunnel, "127.0.0.1", tunnel.local_port


async def _heartbeat_postgres(config: dict) -> ToolTestResponse:
    """Quick PostgreSQL heartbeat check."""
    host = config.get("host", "")
    port = _coerce_int(config.get("port", 5432), 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")

    tunnel: Optional["SSHTunnel"] = None

    try:
        if host:
            tunnel, connect_host, connect_port = _start_ssh_tunnel_if_enabled(
                config, host, port
            )

            cmd = [
                "psql",
                "-h",
                connect_host,
                "-p",
                str(connect_port),
                "-U",
                user,
                "-d",
                database,
                "-c",
                "SELECT 1;",
            ]
            env = {"PGPASSWORD": password}
        elif container:
            cmd = [
                "docker",
                "exec",
                "-i",
                container,
                "bash",
                "-c",
                'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"',
            ]
            env = None
        else:
            return ToolTestResponse(success=False, message="No connection configured")

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        _stdout, stderr = await process.communicate()

        return ToolTestResponse(
            success=process.returncode == 0,
            message=(
                "OK"
                if process.returncode == 0
                else stderr.decode("utf-8", errors="replace").strip()[:100]
            ),
        )
    except Exception as e:
        return ToolTestResponse(success=False, message=str(e)[:100])
    finally:
        if tunnel:
            tunnel.stop()


async def _heartbeat_mysql(config: dict) -> ToolTestResponse:
    """Quick MySQL/MariaDB heartbeat check."""
    host = config.get("host", "")
    port = _coerce_int(config.get("port", 3306), 3306)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")
    docker_network = config.get("docker_network", "")

    if not host and not container:
        return ToolTestResponse(success=False, message="MySQL not configured")

    ssh_tunnel_config = build_ssh_tunnel_config(config, host, port)

    success, message, _ = await test_mysql_connection(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        container=container,
        docker_network=docker_network,
        ssh_tunnel_config=ssh_tunnel_config,
        timeout=5,
    )

    return ToolTestResponse(
        success=success,
        message=message[:100] if message else "OK",
    )


async def _heartbeat_mssql(config: dict) -> ToolTestResponse:
    """Quick MSSQL heartbeat check."""
    host = config.get("host", "")
    port = _coerce_int(config.get("port", 1433), 1433)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")

    if not host or not user or not database:
        return ToolTestResponse(success=False, message="MSSQL not configured")

    ssh_tunnel_config = build_ssh_tunnel_config(config, host, port)

    success, message, _ = await test_mssql_connection(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        ssh_tunnel_config=ssh_tunnel_config,
        timeout=5,
    )

    return ToolTestResponse(
        success=success,
        message="OK" if success else message[:100],
    )


async def _heartbeat_influxdb(config: dict) -> ToolTestResponse:
    """Quick InfluxDB heartbeat check."""
    host = config.get("host", "")
    port = _coerce_int(config.get("port", 8086), 8086)
    use_https = bool(config.get("use_https", False))
    token = config.get("token", "")
    org = config.get("org", "")
    bucket = config.get("bucket", "")

    if not host or not token or not org:
        return ToolTestResponse(success=False, message="InfluxDB not configured")

    ssh_tunnel_config = build_ssh_tunnel_config(config, host, port)

    success, message, _ = await test_influxdb_connection(
        host=host,
        port=port,
        use_https=use_https,
        token=token,
        org=org,
        bucket=bucket,
        timeout=5,
        ssh_tunnel_config=ssh_tunnel_config,
    )

    return ToolTestResponse(
        success=success,
        message="OK" if success else message[:100],
    )


async def _heartbeat_odoo(config: dict) -> ToolTestResponse:
    """Quick Odoo container/SSH heartbeat check."""
    mode = config.get("mode", "docker")

    if mode == "ssh":
        # SSH mode - use socket-level check (non-blocking, avoids paramiko thread issues)
        ssh_host = config.get("ssh_host", "")
        ssh_port = config.get("ssh_port", 22)
        ssh_user = config.get("ssh_user", "")

        if not ssh_host or not ssh_user:
            return ToolTestResponse(success=False, message="SSH not configured")

        try:
            # Use asyncio socket to check if SSH port is reachable
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ssh_host, ssh_port), timeout=3.0
            )

            # Read SSH banner to verify it's actually an SSH server
            banner = await asyncio.wait_for(reader.readline(), timeout=2.0)
            writer.close()
            await writer.wait_closed()

            if banner and b"SSH" in banner:
                return ToolTestResponse(success=True, message="OK")
            else:
                return ToolTestResponse(
                    success=False,
                    message=f"Not an SSH server: {banner.decode('utf-8', errors='replace')[:50]}",
                )

        except asyncio.TimeoutError:
            return ToolTestResponse(success=False, message="Connection timeout")
        except ConnectionRefusedError:
            return ToolTestResponse(success=False, message="Connection refused")
        except OSError as e:
            return ToolTestResponse(success=False, message=str(e)[:100])
    else:
        # Docker mode - check container is running
        container = config.get("container", "")
        if not container:
            return ToolTestResponse(success=False, message="No container configured")

        cmd = ["docker", "exec", "-i", container, "echo", "OK"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            _stdout, stderr = await process.communicate()

            return ToolTestResponse(
                success=process.returncode == 0,
                message=(
                    "OK"
                    if process.returncode == 0
                    else stderr.decode("utf-8", errors="replace").strip()[:100]
                ),
            )
        except Exception as e:
            return ToolTestResponse(success=False, message=str(e)[:100])


def _get_ssh_cache_key(config: dict) -> str:
    """Generate a cache key from SSH config (host:port:user)."""
    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    return f"{host}:{port}:{user}"


async def _deep_ssh_credential_check(config: dict) -> ToolTestResponse:
    """
    Perform a full SSH credential validation using Paramiko.

    This is slower but verifies that credentials are correct.
    Results are cached for 15 seconds to avoid repeated checks.
    """
    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    key_path = config.get("key_path")
    key_content = config.get("key_content")
    key_passphrase = config.get("key_passphrase")
    password = config.get("password")

    if not host or not user:
        return ToolTestResponse(success=False, message="SSH not configured")

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        password=password if password else None,
        key_path=key_path if key_path else None,
        key_content=key_content if key_content else None,
        key_passphrase=key_passphrase if key_passphrase else None,
        timeout=10,  # Shorter timeout for heartbeat check
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: test_ssh_connection(ssh_config)
        )

        if result.success:
            return ToolTestResponse(success=True, message="OK")
        else:
            error_msg = result.stderr or result.stdout or "Authentication failed"
            # Truncate for heartbeat display
            return ToolTestResponse(
                success=False,
                message=error_msg[:80] if len(error_msg) > 80 else error_msg,
            )

    except Exception as e:
        return ToolTestResponse(success=False, message=f"SSH error: {str(e)[:60]}")


async def _heartbeat_ssh(config: dict) -> ToolTestResponse:
    """
    SSH heartbeat check with credential validation.

    Uses cached credential check results (refreshed every 15 seconds) to avoid
    blocking the UI while still validating that credentials are correct.
    Falls back to quick port check if cache is being refreshed.
    """
    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")

    if not host or not user:
        return ToolTestResponse(success=False, message="SSH not configured")

    cache_key = _get_ssh_cache_key(config)
    current_time = time.time()

    # Check if we have a valid cached result
    if cache_key in _ssh_credential_cache:
        cached_time, cached_result = _ssh_credential_cache[cache_key]
        cache_age = current_time - cached_time

        if cache_age < _SSH_CREDENTIAL_CACHE_TTL:
            # Return cached result
            return cached_result

    # Cache expired or not present - do a quick port check first
    # to provide fast feedback, then trigger background credential refresh
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=2.0
        )
        banner = await asyncio.wait_for(reader.readline(), timeout=1.0)
        writer.close()
        await writer.wait_closed()

        if not (banner and b"SSH" in banner):
            result = ToolTestResponse(
                success=False,
                message=f"Not an SSH server: {banner.decode('utf-8', errors='replace')[:50]}",
            )
            _ssh_credential_cache[cache_key] = (current_time, result)
            return result

    except asyncio.TimeoutError:
        result = ToolTestResponse(success=False, message="Connection timeout")
        _ssh_credential_cache[cache_key] = (current_time, result)
        return result
    except ConnectionRefusedError:
        result = ToolTestResponse(success=False, message="Connection refused")
        _ssh_credential_cache[cache_key] = (current_time, result)
        return result
    except OSError as e:
        result = ToolTestResponse(success=False, message=str(e)[:100])
        _ssh_credential_cache[cache_key] = (current_time, result)
        return result

    # Port is reachable - now do a full credential check
    # This runs inline but is cached, so subsequent calls return quickly
    try:
        result = await asyncio.wait_for(
            _deep_ssh_credential_check(config), timeout=12.0
        )
        _ssh_credential_cache[cache_key] = (current_time, result)
        return result
    except asyncio.TimeoutError:
        result = ToolTestResponse(success=False, message="Credential check timeout")
        _ssh_credential_cache[cache_key] = (current_time, result)
        return result


async def _heartbeat_filesystem(config: dict) -> ToolTestResponse:
    """Quick filesystem indexer heartbeat check - verify base path is accessible."""
    base_path = config.get("base_path", "")

    if not base_path:
        return ToolTestResponse(success=False, message="Filesystem path not configured")

    def _check_fs() -> ToolTestResponse:
        path = Path(base_path)
        if not path.exists():
            return ToolTestResponse(
                success=False, message=f"Path does not exist: {base_path}"
            )
        if not path.is_dir():
            return ToolTestResponse(
                success=False, message=f"Path is not a directory: {base_path}"
            )
        try:
            entries = list(path.iterdir())[:5]
            return ToolTestResponse(
                success=True, message=f"OK - {len(entries)} entries accessible"
            )
        except PermissionError:
            return ToolTestResponse(
                success=False, message=f"Permission denied: {base_path}"
            )
        except Exception as e:
            return ToolTestResponse(success=False, message=str(e)[:100])

    return await asyncio.to_thread(_check_fs)


async def _heartbeat_pdm(config: dict) -> ToolTestResponse:
    """Quick SolidWorks PDM database heartbeat check."""

    host = config.get("host", "")
    port = _coerce_int(config.get("port", 1433), 1433)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")

    if not all([host, user, password, database]):
        return ToolTestResponse(success=False, message="PDM connection not configured")

    tunnel: Optional["SSHTunnel"] = None
    connect_host = host
    connect_port = port

    tunnel, connect_host, connect_port = _start_ssh_tunnel_if_enabled(
        config, host, port
    )

    def check_connection() -> tuple[bool, str]:
        try:
            with mssql_connect(
                host=connect_host,
                port=connect_port,
                user=user,
                password=password,
                database=database,
                login_timeout=5,
                timeout=5,
                as_dict=False,
            ) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True, "OK"
        except MssqlConnectionError as e:
            return False, str(e)[:80]
        except Exception as e:
            return False, str(e)[:80]

    try:
        success, message = await asyncio.wait_for(
            asyncio.to_thread(check_connection), timeout=6
        )
        return ToolTestResponse(success=success, message=message)
    except asyncio.TimeoutError:
        return ToolTestResponse(success=False, message="Connection timed out")
    except Exception as e:
        return ToolTestResponse(success=False, message=str(e)[:80])
    finally:
        if tunnel:
            tunnel.stop()


async def _test_postgres_connection(config: dict) -> ToolTestResponse:
    """Test PostgreSQL connection. Supports direct, Docker container, and SSH tunnel modes."""
    host = config.get("host", "")
    port = config.get("port", 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")
    ssh_tunnel_enabled = config.get("ssh_tunnel_enabled", False)

    try:
        if ssh_tunnel_enabled:
            # SSH tunnel mode - requires psycopg2
            ssh_tunnel_host = config.get("ssh_tunnel_host", "")
            ssh_tunnel_user = config.get("ssh_tunnel_user", "")

            if not ssh_tunnel_host:
                return ToolTestResponse(
                    success=False, message="SSH tunnel host is required"
                )
            if not ssh_tunnel_user:
                return ToolTestResponse(
                    success=False, message="SSH tunnel user is required"
                )
            if not user or not password:
                return ToolTestResponse(
                    success=False, message="Database username and password are required"
                )
            if not database:
                return ToolTestResponse(success=False, message="Database is required")

            # Build SSH tunnel config - host/port represent the remote endpoint
            ssh_tunnel_config_dict = {
                "ssh_tunnel_enabled": True,
                "ssh_tunnel_host": ssh_tunnel_host,
                "ssh_tunnel_port": config.get("ssh_tunnel_port", 22),
                "ssh_tunnel_user": ssh_tunnel_user,
                "ssh_tunnel_password": config.get("ssh_tunnel_password", ""),
                "ssh_tunnel_key_path": config.get("ssh_tunnel_key_path", ""),
                "ssh_tunnel_key_content": config.get("ssh_tunnel_key_content", ""),
                "ssh_tunnel_key_passphrase": config.get(
                    "ssh_tunnel_key_passphrase", ""
                ),
                "host": host
                or "127.0.0.1",  # Remote endpoint from SSH server's perspective
                "port": port,
            }

            def do_tunnel_test() -> ToolTestResponse:
                try:
                    import psycopg2  # type: ignore[import-untyped]
                except ImportError:
                    return ToolTestResponse(
                        success=False,
                        message="psycopg2 package not installed. Install with: pip install psycopg2-binary",
                    )

                tunnel = None
                conn = None
                try:
                    tunnel_cfg = ssh_tunnel_config_from_dict(
                        ssh_tunnel_config_dict, default_remote_port=5432
                    )
                    if not tunnel_cfg:
                        return ToolTestResponse(
                            success=False, message="Invalid SSH tunnel configuration"
                        )

                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()

                    conn = psycopg2.connect(
                        host="127.0.0.1",
                        port=local_port,
                        user=user,
                        password=password,
                        dbname=database,
                        connect_timeout=30,
                    )
                    cursor = conn.cursor()
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()
                    cursor.close()

                    return ToolTestResponse(
                        success=True,
                        message="PostgreSQL connection successful (via SSH tunnel)",
                        details={
                            "mode": "ssh_tunnel",
                            "ssh_host": ssh_tunnel_host,
                            "database": database,
                            "version": (
                                version[0].split(",")[0] if version else "Unknown"
                            ),
                        },
                    )
                except Exception as e:
                    error_str = str(e)
                    if "password authentication failed" in error_str:
                        return ToolTestResponse(
                            success=False,
                            message="Authentication failed - check username and password",
                        )
                    if "does not exist" in error_str:
                        return ToolTestResponse(
                            success=False,
                            message=f"Database '{database}' does not exist",
                        )
                    return ToolTestResponse(
                        success=False, message=f"Connection failed: {error_str}"
                    )
                finally:
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    if tunnel:
                        try:
                            tunnel.stop()
                        except Exception:
                            pass

            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, do_tunnel_test),
                timeout=35.0,
            )

        elif host:
            # Direct connection test
            cmd = [
                "psql",
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                user,
                "-d",
                database,
                "-c",
                "SELECT 1;",
            ]
            env = {"PGPASSWORD": password}
        elif container:
            # Docker container test
            cmd = [
                "docker",
                "exec",
                "-i",
                container,
                "bash",
                "-c",
                'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"',
            ]
            env = None
        else:
            return ToolTestResponse(
                success=False,
                message="Either host, container, or SSH tunnel must be specified",
            )

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            ),
            timeout=10.0,
        )
        _stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return ToolTestResponse(
                success=True,
                message="PostgreSQL connection successful",
                details={"host": host or container, "database": database},
            )
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return ToolTestResponse(
                success=False, message=f"Connection failed: {error}"
            )

    except asyncio.TimeoutError:
        return ToolTestResponse(
            success=False, message="Connection timed out after 10 seconds"
        )
    except FileNotFoundError:
        return ToolTestResponse(
            success=False, message="Required command (psql or docker) not found"
        )
    except Exception as e:
        return ToolTestResponse(
            success=False, message=f"Connection test failed: {str(e)}"
        )


async def _test_mssql_connection(config: dict) -> ToolTestResponse:
    """Test MSSQL/SQL Server connection. Supports direct and SSH tunnel modes."""
    ssh_tunnel_enabled = config.get("ssh_tunnel_enabled", False)

    if ssh_tunnel_enabled:
        # SSH tunnel mode
        ssh_tunnel_host = config.get("ssh_tunnel_host", "")
        ssh_tunnel_user = config.get("ssh_tunnel_user", "")
        user = config.get("user", "")
        password = config.get("password", "")
        database = config.get("database", "")
        host = config.get("host", "127.0.0.1")  # Remote endpoint from SSH server
        port = config.get("port", 1433)

        if not ssh_tunnel_host:
            return ToolTestResponse(
                success=False, message="SSH tunnel host is required"
            )
        if not ssh_tunnel_user:
            return ToolTestResponse(
                success=False, message="SSH tunnel user is required"
            )
        if not user or not password:
            return ToolTestResponse(
                success=False, message="Database username and password are required"
            )
        if not database:
            return ToolTestResponse(success=False, message="Database is required")

        # Build SSH tunnel config - host/port represent the remote endpoint
        ssh_tunnel_config = {
            "ssh_tunnel_enabled": True,
            "ssh_tunnel_host": ssh_tunnel_host,
            "ssh_tunnel_port": config.get("ssh_tunnel_port", 22),
            "ssh_tunnel_user": ssh_tunnel_user,
            "ssh_tunnel_password": config.get("ssh_tunnel_password", ""),
            "ssh_tunnel_key_path": config.get("ssh_tunnel_key_path", ""),
            "ssh_tunnel_key_content": config.get("ssh_tunnel_key_content", ""),
            "ssh_tunnel_key_passphrase": config.get("ssh_tunnel_key_passphrase", ""),
            "host": host,  # Remote endpoint from SSH server's perspective
            "port": port,
        }

        success, message, details = await test_mssql_connection(
            user=user,
            password=password,
            database=database,
            timeout=30,
            ssh_tunnel_config=ssh_tunnel_config,
        )
    else:
        # Direct connection mode
        host = config.get("host", "")
        port = config.get("port", 1433)
        user = config.get("user", "")
        password = config.get("password", "")
        database = config.get("database", "")

        if not host:
            return ToolTestResponse(success=False, message="Host is required")
        if not user or not password:
            return ToolTestResponse(
                success=False, message="Username and password are required"
            )
        if not database:
            return ToolTestResponse(success=False, message="Database is required")

        success, message, details = await test_mssql_connection(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=10,
        )

    return ToolTestResponse(success=success, message=message, details=details)


async def _test_mysql_connection(config: dict) -> ToolTestResponse:
    """Test MySQL/MariaDB connection. Supports direct, Docker container, and SSH tunnel modes."""
    container = config.get("container", "")
    ssh_tunnel_enabled = config.get("ssh_tunnel_enabled", False)

    if container:
        # Docker container mode
        docker_network = config.get("docker_network", "")
        database = config.get("database", "")

        if not database:
            return ToolTestResponse(success=False, message="Database is required")

        success, message, details = await test_mysql_connection(
            container=container,
            docker_network=docker_network,
            database=database,
            timeout=30,
        )
    elif ssh_tunnel_enabled:
        # SSH tunnel mode
        ssh_tunnel_host = config.get("ssh_tunnel_host", "")
        ssh_tunnel_user = config.get("ssh_tunnel_user", "")
        user = config.get("user", "")
        password = config.get("password", "")
        database = config.get("database", "")
        host = config.get("host", "127.0.0.1")  # Remote endpoint from SSH server
        port = config.get("port", 3306)

        if not ssh_tunnel_host:
            return ToolTestResponse(
                success=False, message="SSH tunnel host is required"
            )
        if not ssh_tunnel_user:
            return ToolTestResponse(
                success=False, message="SSH tunnel user is required"
            )
        if not user or not password:
            return ToolTestResponse(
                success=False, message="Database username and password are required"
            )
        if not database:
            return ToolTestResponse(success=False, message="Database is required")

        # Build SSH tunnel config - host/port represent the remote endpoint
        ssh_tunnel_config = {
            "ssh_tunnel_enabled": True,
            "ssh_tunnel_host": ssh_tunnel_host,
            "ssh_tunnel_port": config.get("ssh_tunnel_port", 22),
            "ssh_tunnel_user": ssh_tunnel_user,
            "ssh_tunnel_password": config.get("ssh_tunnel_password", ""),
            "ssh_tunnel_key_path": config.get("ssh_tunnel_key_path", ""),
            "ssh_tunnel_key_content": config.get("ssh_tunnel_key_content", ""),
            "ssh_tunnel_key_passphrase": config.get("ssh_tunnel_key_passphrase", ""),
            "host": host,  # Remote endpoint from SSH server's perspective
            "port": port,
        }

        success, message, details = await test_mysql_connection(
            user=user,
            password=password,
            database=database,
            timeout=30,
            ssh_tunnel_config=ssh_tunnel_config,
        )
    else:
        # Direct connection mode
        host = config.get("host", "")
        port = config.get("port", 3306)
        user = config.get("user", "")
        password = config.get("password", "")
        database = config.get("database", "")

        if not host:
            return ToolTestResponse(success=False, message="Host is required")
        if not user or not password:
            return ToolTestResponse(
                success=False, message="Username and password are required"
            )
        if not database:
            return ToolTestResponse(success=False, message="Database is required")

        success, message, details = await test_mysql_connection(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=10,
        )

    return ToolTestResponse(success=success, message=message, details=details)


async def _test_influxdb_connection(config: dict) -> ToolTestResponse:
    """Test InfluxDB connection. Supports direct and SSH tunnel modes."""
    host = config.get("host", "")
    port = _coerce_int(config.get("port", 8086), 8086)
    use_https = bool(config.get("use_https", False))
    token = config.get("token", "")
    org = config.get("org", "")
    bucket = config.get("bucket", "")
    ssh_tunnel_enabled = config.get("ssh_tunnel_enabled", False)

    if not host:
        return ToolTestResponse(success=False, message="Host is required")
    if not token:
        return ToolTestResponse(success=False, message="Token is required")
    if not org:
        return ToolTestResponse(success=False, message="Organization is required")

    ssh_tunnel_config = None
    if ssh_tunnel_enabled:
        ssh_tunnel_host = config.get("ssh_tunnel_host", "")
        ssh_tunnel_user = config.get("ssh_tunnel_user", "")
        if not ssh_tunnel_host:
            return ToolTestResponse(
                success=False, message="SSH tunnel host is required"
            )
        if not ssh_tunnel_user:
            return ToolTestResponse(
                success=False, message="SSH tunnel user is required"
            )

        ssh_tunnel_config = {
            "ssh_tunnel_enabled": True,
            "ssh_tunnel_host": ssh_tunnel_host,
            "ssh_tunnel_port": _coerce_int(config.get("ssh_tunnel_port", 22), 22),
            "ssh_tunnel_user": ssh_tunnel_user,
            "ssh_tunnel_password": config.get("ssh_tunnel_password", ""),
            "ssh_tunnel_key_path": config.get("ssh_tunnel_key_path", ""),
            "ssh_tunnel_key_content": config.get("ssh_tunnel_key_content", ""),
            "ssh_tunnel_key_passphrase": config.get("ssh_tunnel_key_passphrase", ""),
            "host": host,
            "port": port,
        }

    success, message, details = await test_influxdb_connection(
        host=host,
        port=port,
        use_https=use_https,
        token=token,
        org=org,
        bucket=bucket,
        timeout=10,
        ssh_tunnel_config=ssh_tunnel_config,
    )

    return ToolTestResponse(success=success, message=message, details=details)


async def _test_odoo_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection (Docker or SSH mode)."""
    mode = config.get("mode", "docker")

    if mode == "ssh":
        return await _test_odoo_ssh_connection(config)
    else:
        return await _test_odoo_docker_connection(config)


async def _test_odoo_ssh_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection via SSH using Paramiko."""

    ssh_host = config.get("ssh_host", "")
    ssh_port = config.get("ssh_port", 22)
    ssh_user = config.get("ssh_user", "")
    ssh_key_path = config.get("ssh_key_path", "")
    ssh_key_content = config.get("ssh_key_content", "")
    ssh_key_passphrase = config.get("ssh_key_passphrase", "")
    ssh_password = config.get("ssh_password", "")
    database = config.get("database", "odoo")
    config_path = config.get("config_path", "")
    odoo_bin_path = config.get("odoo_bin_path", "odoo-bin")
    working_directory = config.get("working_directory", "")
    run_as_user = config.get("run_as_user", "")

    if not ssh_host or not ssh_user:
        return ToolTestResponse(success=False, message="SSH host and user are required")

    # Build SSH config
    ssh_config = SSHConfig(
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        password=ssh_password if ssh_password else None,
        key_path=ssh_key_path if ssh_key_path else None,
        key_content=ssh_key_content if ssh_key_content else None,
        key_passphrase=ssh_key_passphrase if ssh_key_passphrase else None,
        timeout=30,
    )

    try:
        # First test basic SSH connectivity
        loop = asyncio.get_event_loop()
        ssh_result = await loop.run_in_executor(
            None, lambda: test_ssh_connection(ssh_config)
        )

        if not ssh_result.success:
            return ToolTestResponse(
                success=False,
                message=f"SSH connection failed: {ssh_result.stderr or ssh_result.stdout}",
            )

        # Build Odoo shell command
        odoo_cmd = f"{odoo_bin_path} shell --no-http -d {database}"
        if config_path:
            odoo_cmd = f"{odoo_cmd} -c {config_path}"
        if run_as_user:
            odoo_cmd = f"sudo -u {run_as_user} {odoo_cmd}"
        if working_directory:
            odoo_cmd = f"cd {working_directory} && {odoo_cmd}"

        # Test Odoo shell with heredoc
        test_input = "print('ODOO_TEST_SUCCESS')\nexit()\n"
        full_command = f"{odoo_cmd} <<'ODOO_EOF'\n{test_input}ODOO_EOF"

        odoo_result = await loop.run_in_executor(
            None, lambda: execute_ssh_command(ssh_config, full_command)
        )

        if "ODOO_TEST_SUCCESS" in odoo_result.output:
            return ToolTestResponse(
                success=True,
                message="Odoo shell accessible via SSH",
                details={"host": ssh_host, "database": database, "mode": "ssh"},
            )
        else:
            output_snippet = (
                odoo_result.output[-500:]
                if len(odoo_result.output) > 500
                else odoo_result.output
            )
            return ToolTestResponse(
                success=False,
                message="Odoo shell test failed via SSH",
                details={"output_tail": output_snippet},
            )

    except Exception as e:
        return ToolTestResponse(success=False, message=f"SSH test failed: {str(e)}")


async def _test_odoo_docker_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection via Docker."""
    container = config.get("container", "")
    database = config.get("database", "odoo")
    config_path = config.get("config_path", "")
    docker_network = config.get("docker_network", "")

    if not container:
        return ToolTestResponse(success=False, message="Container name is required")

    try:
        # Test if container is running
        cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        stdout, _stderr = await process.communicate()

        if process.returncode != 0:
            return ToolTestResponse(
                success=False,
                message=f"Container '{container}' not found or not accessible",
            )

        is_running = stdout.decode().strip().lower() == "true"
        if not is_running:
            return ToolTestResponse(
                success=False, message=f"Container '{container}' is not running"
            )

        # Get Odoo version - try multiple approaches
        version = "unknown"

        # Try direct odoo --version (works for standard Odoo)
        cmd = ["docker", "exec", "-i", container, "odoo", "--version"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        stdout, _stderr = await process.communicate()

        if process.returncode == 0:
            version = stdout.decode().strip()
        else:
            # Try alternative: check if odoo shell command exists
            cmd = ["docker", "exec", "-i", container, "which", "odoo"]
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ),
                timeout=5.0,
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                return ToolTestResponse(
                    success=False, message="Odoo command not found in container"
                )
            # odoo exists, but --version not supported (custom wrapper)
            version = "detected (custom wrapper)"

        # Test shell execution with stdin pipe (standard approach)
        cmd = [
            "docker",
            "exec",
            "-i",
            container,
            "odoo",
            "shell",
            "--no-http",
            "-d",
            database,
        ]
        if config_path:
            cmd.extend(["-c", config_path])

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ),
            timeout=120.0,  # Shell initialization can take 10-60+ seconds
        )

        test_input = "print('ODOO_TEST_SUCCESS')\nexit()\n"
        stdout, _ = await process.communicate(input=test_input.encode())
        output = stdout.decode()

        if "ODOO_TEST_SUCCESS" in output:
            return ToolTestResponse(
                success=True,
                message=f"Odoo shell accessible: {version}",
                details={
                    "container": container,
                    "database": database,
                    "version": version,
                    "docker_network": docker_network,
                    "mode": "docker",
                },
            )
        else:
            # Check for common errors in output
            if "database" in output.lower() and "not exist" in output.lower():
                return ToolTestResponse(
                    success=False,
                    message=f"Database '{database}' does not exist in container",
                )
            # Include some output context for debugging
            output_snippet = output[-500:] if len(output) > 500 else output
            return ToolTestResponse(
                success=False,
                message="Odoo shell test failed - could not verify shell access",
                details={"output_tail": output_snippet},
            )

    except asyncio.TimeoutError:
        return ToolTestResponse(
            success=False,
            message="Connection timed out (shell initialization may need >2 minutes)",
        )
    except FileNotFoundError:
        return ToolTestResponse(success=False, message="Docker command not found")
    except Exception as e:
        return ToolTestResponse(
            success=False, message=f"Connection test failed: {str(e)}"
        )


async def _test_ssh_connection(config: dict) -> ToolTestResponse:
    """Test SSH shell connection using Paramiko."""

    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    key_path = config.get("key_path")
    key_content = config.get("key_content")
    key_passphrase = config.get("key_passphrase")
    password = config.get("password")

    if not host or not user:
        return ToolTestResponse(success=False, message="Host and user are required")

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        password=password if password else None,
        key_path=key_path if key_path else None,
        key_content=key_content if key_content else None,
        key_passphrase=key_passphrase if key_passphrase else None,
        timeout=15,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: test_ssh_connection(ssh_config)
        )

        if result.success:
            return ToolTestResponse(
                success=True,
                message="SSH connection successful",
                details={"host": host, "user": user, "port": port},
            )
        else:
            return ToolTestResponse(
                success=False,
                message=f"SSH connection failed: {result.stderr or result.stdout}",
            )

    except Exception as e:
        return ToolTestResponse(success=False, message=f"SSH test failed: {str(e)}")


# -----------------------------------------------------------------------------
# Docker Discovery (Networks and Containers)
# -----------------------------------------------------------------------------


class DockerNetwork(BaseModel):
    """Information about a Docker network."""

    name: str
    driver: str
    scope: str
    containers: List[str] = []


class DockerContainer(BaseModel):
    """Information about a Docker container."""

    name: str
    image: str
    status: str
    networks: List[str] = []
    has_odoo: bool = False


class DockerDiscoveryResponse(BaseModel):
    """Response from Docker discovery."""

    success: bool
    message: str
    networks: List[DockerNetwork] = []
    containers: List[DockerContainer] = []
    current_network: Optional[str] = None
    current_container: Optional[str] = None


@router.get("/docker/discover", response_model=DockerDiscoveryResponse, tags=["Tools"])
async def discover_docker_resources(_user: User = Depends(require_admin)):
    """
    Discover Docker networks and containers for tool configuration.

    Returns available networks, running containers, and which containers have Odoo.
    Also detects the current ragtime container's network.
    """
    networks = []
    containers = []
    current_network = None
    current_container = None

    try:
        # Get our own container name by querying Docker
        hostname = os.environ.get("HOSTNAME", "")
        if hostname:
            # The hostname in Docker is typically the container ID
            # Get the actual container name from the ID
            cmd = ["docker", "inspect", "-f", "{{.Name}}", hostname]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                # Remove leading slash from container name
                current_container = stdout.decode().strip().lstrip("/")

        # Get networks
        cmd = ["docker", "network", "ls", "--format", "{{json .}}"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    try:
                        net = json.loads(line)
                        # Skip default networks
                        if net.get("Name") not in ["bridge", "host", "none"]:
                            networks.append(
                                DockerNetwork(
                                    name=net.get("Name", ""),
                                    driver=net.get("Driver", ""),
                                    scope=net.get("Scope", ""),
                                )
                            )
                    except json.JSONDecodeError:
                        continue

        # Get running containers
        cmd = ["docker", "ps", "--format", "{{json .}}"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    try:
                        cont = json.loads(line)
                        container_name = cont.get("Names", "")

                        # Get networks for this container
                        net_cmd = [
                            "docker",
                            "inspect",
                            "-f",
                            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
                            container_name,
                        ]
                        net_process = await asyncio.create_subprocess_exec(
                            *net_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        net_stdout, _ = await net_process.communicate()
                        container_networks = (
                            net_stdout.decode().strip().split()
                            if net_process.returncode == 0
                            else []
                        )

                        # Check if this is the ragtime container
                        if current_container and container_name == current_container:
                            current_network = (
                                container_networks[0] if container_networks else None
                            )

                        # Check if container has Odoo (simple version check)
                        has_odoo = False
                        if (
                            "odoo" in cont.get("Image", "").lower()
                            or "odoo" in container_name.lower()
                        ):
                            try:
                                ver_cmd = [
                                    "docker",
                                    "exec",
                                    "-i",
                                    container_name,
                                    "odoo",
                                    "--version",
                                ]
                                ver_process = await asyncio.wait_for(
                                    asyncio.create_subprocess_exec(
                                        *ver_cmd,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                    ),
                                    timeout=5.0,
                                )
                                await ver_process.communicate()
                                has_odoo = ver_process.returncode == 0
                            except Exception:
                                has_odoo = True  # Assume it's Odoo based on name

                        containers.append(
                            DockerContainer(
                                name=container_name,
                                image=cont.get("Image", ""),
                                status=cont.get("Status", ""),
                                networks=container_networks,
                                has_odoo=has_odoo,
                            )
                        )
                    except json.JSONDecodeError:
                        continue

        # Add container names to networks
        for network in networks:
            network.containers = [
                c.name for c in containers if network.name in c.networks
            ]

        return DockerDiscoveryResponse(
            success=True,
            message=f"Found {len(networks)} networks and {len(containers)} containers",
            networks=networks,
            containers=containers,
            current_network=current_network,
            current_container=current_container,
        )

    except asyncio.TimeoutError:
        return DockerDiscoveryResponse(
            success=False, message="Docker discovery timed out"
        )
    except FileNotFoundError:
        return DockerDiscoveryResponse(
            success=False, message="Docker command not found"
        )
    except Exception as e:
        return DockerDiscoveryResponse(
            success=False, message=f"Docker discovery failed: {str(e)}"
        )


@router.post("/docker/connect-network", tags=["Tools"])
async def connect_to_network(network_name: str, _user: User = Depends(require_admin)):
    """
    Connect the ragtime container to a Docker network.

    This enables container-to-container communication with services on that network.
    """
    # Get current container name
    container_name = os.environ.get("HOSTNAME", "ragtime-dev")

    try:
        # Check if already connected
        cmd = [
            "docker",
            "inspect",
            "-f",
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
            container_name,
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, _ = await process.communicate()

        current_networks = (
            stdout.decode().strip().split() if process.returncode == 0 else []
        )

        if network_name in current_networks:
            return {
                "success": True,
                "message": f"Already connected to network '{network_name}'",
            }

        # Connect to the network
        cmd = ["docker", "network", "connect", network_name, container_name]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            return {
                "success": True,
                "message": f"Connected to network '{network_name}'",
            }
        else:
            error = stderr.decode().strip()
            return {"success": False, "message": f"Failed to connect: {error}"}

    except asyncio.TimeoutError:
        return {"success": False, "message": "Network connection timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/docker/disconnect-network", tags=["Tools"])
async def disconnect_from_network(
    network_name: str, _user: User = Depends(require_admin)
):
    """
    Disconnect the ragtime container from a Docker network.

    Used for cleanup when removing tools that required network access.
    """
    # Get current container name
    container_name = os.environ.get("HOSTNAME", "ragtime-dev")

    # Don't disconnect from default networks
    protected_networks = ["ragtime_default", "bridge", "host", "none"]
    if network_name in protected_networks:
        return {
            "success": True,
            "message": f"Network '{network_name}' is protected, skipping disconnect",
        }

    try:
        # Check if connected
        cmd = [
            "docker",
            "inspect",
            "-f",
            "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}",
            container_name,
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, _ = await process.communicate()

        current_networks = (
            stdout.decode().strip().split() if process.returncode == 0 else []
        )

        if network_name not in current_networks:
            return {
                "success": True,
                "message": f"Not connected to network '{network_name}'",
            }

        # Disconnect from the network
        cmd = ["docker", "network", "disconnect", network_name, container_name]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ),
            timeout=10.0,
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            return {
                "success": True,
                "message": f"Disconnected from network '{network_name}'",
            }
        else:
            error = stderr.decode().strip()
            return {"success": False, "message": f"Failed to disconnect: {error}"}

    except asyncio.TimeoutError:
        return {"success": False, "message": "Network disconnection timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# -----------------------------------------------------------------------------
# Container Capabilities Detection
# -----------------------------------------------------------------------------


class ContainerCapabilitiesResponse(BaseModel):
    """Response from container capabilities check."""

    privileged: bool = Field(
        description="Whether the container is running in privileged mode"
    )
    has_sys_admin: bool = Field(
        description="Whether the container has CAP_SYS_ADMIN capability"
    )
    can_mount: bool = Field(
        description="Whether the container can perform mount operations (SMB/NFS)"
    )
    message: str = Field(
        description="Human-readable explanation of the capabilities status"
    )


def _check_container_capabilities() -> ContainerCapabilitiesResponse:
    """
    Check if the container has sufficient privileges for mounting filesystems.

    This checks for either privileged mode or CAP_SYS_ADMIN capability by:
    1. Reading /proc/self/status for capability flags
    2. Checking if we can read the raw effective capabilities bitmask

    CAP_SYS_ADMIN (bit 21) is required for mount operations.
    In privileged mode, all capabilities are granted.
    """
    caps = get_container_capabilities()
    privileged = caps.privileged
    has_sys_admin = caps.has_sys_admin

    can_mount = privileged or has_sys_admin

    if privileged:
        message = (
            "Container is running in privileged mode. SMB/NFS mounts are available."
        )
    elif has_sys_admin:
        message = (
            "Container has CAP_SYS_ADMIN capability. SMB/NFS mounts are available."
        )
    else:
        message = (
            "Container lacks mount privileges. To enable SMB/NFS mounting, "
            "uncomment 'privileged: true' and 'cap_add: SYS_ADMIN' in docker-compose.yml "
            "and restart the container."
        )

    return ContainerCapabilitiesResponse(
        privileged=privileged,
        has_sys_admin=has_sys_admin,
        can_mount=can_mount,
        message=message,
    )


@router.get(
    "/filesystem/capabilities",
    response_model=ContainerCapabilitiesResponse,
    tags=["Filesystem Indexer"],
)
async def check_container_capabilities(_user: User = Depends(require_admin)):
    """
    Check if the container has sufficient privileges for mounting SMB/NFS filesystems.

    Returns whether the container is running in privileged mode or has CAP_SYS_ADMIN,
    which are required for mounting network filesystems inside the container.

    This endpoint helps the UI determine whether to show SMB/NFS options in the
    filesystem indexer wizard.
    """
    return _check_container_capabilities()


# -----------------------------------------------------------------------------
# Host Filesystem Browser (for volume mount discovery)
# -----------------------------------------------------------------------------


class MountInfo(BaseModel):
    """Information about a mounted volume."""

    container_path: str
    host_path: str
    read_only: bool
    mount_type: str  # bind, volume, or tmpfs


class DirectoryEntry(BaseModel):
    """Entry in a directory listing."""

    name: str
    path: str
    is_dir: bool
    size: Optional[int] = None


class BrowseResponse(BaseModel):
    """Response from directory browsing."""

    path: str
    entries: List[DirectoryEntry]
    error: Optional[str] = None


class SSHBrowseRequest(BaseModel):
    """Request to browse remote SSH filesystem."""

    host: str
    user: str
    port: int = 22
    password: Optional[str] = None
    key_path: Optional[str] = None
    key_content: Optional[str] = None
    key_passphrase: Optional[str] = None
    path: str = "/"


class MountDiscoveryResponse(BaseModel):
    """Response from mount discovery."""

    mounts: List[MountInfo]
    suggested_paths: List[str]
    docker_compose_example: str


@router.get(
    "/filesystem/mounts",
    response_model=MountDiscoveryResponse,
    tags=["Filesystem Indexer"],
)
async def discover_mounts(_user: User = Depends(require_admin)):
    """
    Discover current volume mounts in the container.

    Returns mounted paths and example docker-compose configuration for adding new mounts.
    """
    mounts = []
    hostname = os.environ.get("HOSTNAME", "")

    try:
        # Get mount information from Docker inspect
        if hostname:
            cmd = ["docker", "inspect", "-f", "{{json .Mounts}}", hostname]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _stderr = await process.communicate()

            if process.returncode == 0:
                import json

                mount_data = json.loads(stdout.decode().strip())
                for m in mount_data:
                    mounts.append(
                        MountInfo(
                            container_path=m.get("Destination", ""),
                            host_path=m.get("Source", ""),
                            read_only=m.get("RW", True) is False,
                            mount_type=m.get("Type", "bind"),
                        )
                    )
    except Exception as e:
        logger.warning(f"Failed to inspect mounts: {e}")

    # Also check /proc/mounts as fallback
    if not mounts:
        try:
            with open("/proc/mounts", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and parts[1].startswith("/mnt"):
                        mounts.append(
                            MountInfo(
                                container_path=parts[1],
                                host_path=parts[0],
                                read_only="ro" in parts[3],
                                mount_type=(
                                    "bind" if parts[0].startswith("/") else "unknown"
                                ),
                            )
                        )
        except Exception:
            pass

    # Suggested paths to mount (common document locations)
    suggested_paths = [
        "/mnt/documents",
        "/mnt/data",
        "/mnt/files",
        "/mnt/shared",
    ]

    # Docker-compose example
    docker_compose_example = """# Add this to your docker-compose.yml under the ragtime service volumes:
volumes:
  # Example: Mount your Documents folder
  - /path/to/your/documents:/mnt/documents:ro

  # Windows example (WSL path)
  - /mnt/c/Users/YourName/Documents:/mnt/documents:ro

  # Linux/Mac example
  - ~/Documents:/mnt/documents:ro

# Then restart the container:
# docker compose -f docker/docker-compose.dev.yml restart ragtime"""

    return MountDiscoveryResponse(
        mounts=mounts,
        suggested_paths=suggested_paths,
        docker_compose_example=docker_compose_example,
    )


@router.post(
    "/filesystem/ssh/browse", response_model=BrowseResponse, tags=["Filesystem Indexer"]
)
async def browse_ssh_filesystem(
    request: SSHBrowseRequest, _: User = Depends(require_admin)
):
    """
    Browse a remote directory path via SSH.
    """
    config = SSHConfig(
        host=request.host,
        user=request.user,
        port=request.port,
        password=request.password,
        key_path=request.key_path,
        key_content=request.key_content,
        key_passphrase=request.key_passphrase,
    )

    # Use ls -p1a to list files (append / to dirs, one per line, all files)
    # ls -p appends / indicator to directories
    # -1 forces one entry per line
    # -a includes hidden files
    path = request.path
    # Quote the path to handle spaces and special chars
    quoted_path = shlex.quote(path)

    # We list the contents OF the directory, not the directory itself if it is one
    # If path is a file, ls might return it.
    # To ensure we are listing directory content, we assume path is a dir.

    cmd = f"ls -p1a {quoted_path}"

    # Execute SSH command
    try:
        # We need to make sure we don't hang if it prompts for anything (SSHConfig handles timeouts)
        result = await asyncio.to_thread(execute_ssh_command, config, cmd)
    except Exception as e:
        return BrowseResponse(
            path=path,
            entries=[],
            error=str(e),
        )

    if not result.success:
        # Check if it's "No such file or directory" or "Not a directory"
        error_msg = result.stderr or result.stdout or "Failed to list directory"
        return BrowseResponse(
            path=path,
            entries=[],
            error=error_msg,
        )

    entries = []
    # Parse output
    for line in result.stdout.splitlines():
        name = line.strip()
        # Skip current/parent dir markers from -a
        if name in (".", "./", "..", "../"):
            continue

        if not name:
            continue

        is_dir = name.endswith("/")

        # Only show directories
        if not is_dir:
            continue

        clean_name = name.rstrip("/")

        # Construct full path safely
        # We assume unix remote
        if path.endswith("/"):
            full_path = f"{path}{clean_name}"
        else:
            full_path = f"{path}/{clean_name}"

        entries.append(
            DirectoryEntry(name=clean_name, path=full_path, is_dir=is_dir, size=None)
        )

    # Sort: directories first, then files, ignoring case
    entries.sort(key=lambda x: (not x.is_dir, x.name.lower()))

    return BrowseResponse(path=path, entries=entries)


@router.get(
    "/filesystem/browse", response_model=BrowseResponse, tags=["Filesystem Indexer"]
)
async def browse_filesystem(path: str = "/mnt", _: User = Depends(require_admin)):
    """
    Browse a directory path in the container.

    Used to explore mounted volumes and select paths for indexing.
    Only allows browsing under /mnt or other mounted paths for security.
    """
    # Security: only allow browsing under /mnt or specific safe prefixes
    allowed_prefixes = ["/mnt", "/data", "/shared"]
    path_obj = Path(path).resolve()

    if not any(str(path_obj).startswith(prefix) for prefix in allowed_prefixes):
        return BrowseResponse(
            path=path,
            entries=[],
            error=f"Access denied. Only paths under {', '.join(allowed_prefixes)} are browsable.",
        )

    def _browse() -> BrowseResponse:
        if not path_obj.exists():
            return BrowseResponse(
                path=path,
                entries=[],
                error=f"Path does not exist: {path}",
            )

        if not path_obj.is_dir():
            return BrowseResponse(
                path=path,
                entries=[],
                error=f"Not a directory: {path}",
            )

        try:
            entries = []
            for entry in sorted(
                path_obj.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
            ):
                try:
                    entries.append(
                        DirectoryEntry(
                            name=entry.name,
                            path=str(entry),
                            is_dir=entry.is_dir(),
                            size=entry.stat().st_size if entry.is_file() else None,
                        )
                    )
                except (PermissionError, OSError):
                    continue

            return BrowseResponse(path=str(path_obj), entries=entries)

        except PermissionError:
            return BrowseResponse(
                path=path,
                entries=[],
                error=f"Permission denied: {path}",
            )
        except Exception as e:
            return BrowseResponse(
                path=path,
                entries=[],
                error=str(e),
            )

    return await asyncio.to_thread(_browse)


# -----------------------------------------------------------------------------
# NFS/SMB Discovery Endpoints (Admin only)
# -----------------------------------------------------------------------------


class NFSExport(BaseModel):
    """NFS export entry."""

    export_path: str
    allowed_hosts: str = ""


class NFSDiscoveryResponse(BaseModel):
    """Response from NFS export discovery."""

    success: bool
    exports: list[NFSExport] = []
    error: str | None = None


class SMBShare(BaseModel):
    """SMB share entry."""

    name: str
    share_type: str = "Disk"
    comment: str = ""


class SMBDiscoveryResponse(BaseModel):
    """Response from SMB share discovery."""

    success: bool
    shares: list[SMBShare] = []
    error: str | None = None


@router.get(
    "/filesystem/nfs/discover",
    response_model=NFSDiscoveryResponse,
    tags=["Filesystem Indexer"],
)
async def discover_nfs_exports(
    host: str,
    _: User = Depends(require_admin),
):
    """
    Discover NFS exports from a remote server.

    Uses showmount to list available exports on the target NFS server.
    """
    if not host:
        return NFSDiscoveryResponse(success=False, error="Host is required")

    try:
        # Use showmount to list exports
        cmd = ["showmount", "-e", host, "--no-headers"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
            timeout=10.0,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return NFSDiscoveryResponse(
                success=False, error=f"Connection timeout to {host}"
            )

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            if "command not found" in error_msg.lower():
                return NFSDiscoveryResponse(
                    success=False,
                    error="NFS tools not installed. Install nfs-common package.",
                )
            return NFSDiscoveryResponse(
                success=False, error=error_msg or "Failed to connect"
            )

        exports = []
        for line in stdout.decode("utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            if parts:
                export_path = parts[0]
                allowed_hosts = " ".join(parts[1:]) if len(parts) > 1 else "*"
                exports.append(
                    NFSExport(export_path=export_path, allowed_hosts=allowed_hosts)
                )

        return NFSDiscoveryResponse(success=True, exports=exports)

    except asyncio.TimeoutError:
        return NFSDiscoveryResponse(
            success=False, error=f"Connection timeout to {host}"
        )
    except Exception as e:
        return NFSDiscoveryResponse(success=False, error=str(e))


@router.get(
    "/filesystem/smb/discover",
    response_model=SMBDiscoveryResponse,
    tags=["Filesystem Indexer"],
)
async def discover_smb_shares(
    host: str,
    user: str = "",
    password: str = "",
    domain: str = "",
    _: User = Depends(require_admin),
):
    """
    Discover SMB/CIFS shares from a remote server using pysmb.
    """
    if not host:
        return SMBDiscoveryResponse(success=False, error="Host is required")

    def _discover_smb() -> SMBDiscoveryResponse:
        try:
            smb_module = importlib.import_module(
                "smb.SMBConnection"
            )  # pyright: ignore[reportMissingImports]
            SMBConnection = getattr(smb_module, "SMBConnection")

            # Resolve hostname to IP
            try:
                server_ip = socket.gethostbyname(host)
            except socket.gaierror:
                return SMBDiscoveryResponse(
                    success=False, error=f"Cannot resolve hostname: {host}"
                )

            # Create SMB connection
            conn = SMBConnection(
                user or "guest",
                password or "",
                "ragtime",  # client machine name
                host,  # server name
                domain=domain or "",
                use_ntlm_v2=True,
                is_direct_tcp=True,
            )

            try:
                connected = conn.connect(server_ip, 445, timeout=10)
                if not connected:
                    return SMBDiscoveryResponse(
                        success=False, error="Failed to connect to SMB server"
                    )
            except Exception as e:
                return SMBDiscoveryResponse(
                    success=False, error=f"Connection failed: {e}"
                )

            try:
                share_list = conn.listShares()
                shares = []
                for s in share_list:
                    # Skip system shares (ending with $)
                    if s.name.endswith("$"):
                        continue
                    shares.append(
                        SMBShare(
                            name=s.name,
                            share_type="Disk" if s.type == 0 else "Other",
                            comment=s.comments or "",
                        )
                    )
                return SMBDiscoveryResponse(success=True, shares=shares)
            except Exception as e:
                return SMBDiscoveryResponse(success=False, error=str(e))
            finally:
                conn.close()

        except ImportError:
            return SMBDiscoveryResponse(success=False, error="pysmb not installed")
        except Exception as e:
            return SMBDiscoveryResponse(success=False, error=str(e))

    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _discover_smb),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        return SMBDiscoveryResponse(
            success=False, error=f"Connection timeout to {host}"
        )


@router.get(
    "/filesystem/nfs/browse",
    response_model=BrowseResponse,
    tags=["Filesystem Indexer"],
)
async def browse_nfs_export(
    host: str,
    export_path: str,
    path: str = "",
    _: User = Depends(require_admin),
):
    """
    Browse an NFS export using nfs-ls (userspace, no mount required).
    """
    if not host or not export_path:
        return BrowseResponse(
            path=path, entries=[], error="Host and export path are required"
        )

    try:
        # Build NFS URL: nfs://host/export/path
        browse_path = path.lstrip("/") if path else ""
        nfs_url = f"nfs://{host}{export_path}"
        if browse_path:
            nfs_url = f"{nfs_url}/{browse_path}"

        # Use userspace nfs-ls (libnfs-utils); -l is not supported, so rely on trailing slash for dirs
        cmd = ["nfs-ls", nfs_url]

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ),
            timeout=15.0,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return BrowseResponse(
                path=path, entries=[], error=f"Timeout connecting to {host}"
            )

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            if "command not found" in error_msg.lower():
                return BrowseResponse(
                    path=path,
                    entries=[],
                    error="nfs-ls not installed. Install libnfs-utils package.",
                )
            if "usage:" in error_msg.lower():
                return BrowseResponse(
                    path=path,
                    entries=[],
                    error="Browse failed: nfs-ls usage error (ensure export/path exists)",
                )
            if (
                "mnt3err_acces" in error_msg.lower()
                or "permission denied" in error_msg.lower()
            ):
                return BrowseResponse(
                    path=path,
                    entries=[],
                    error="Access denied to NFS export. Allow this client in the export or adjust permissions.",
                )
            if "no such file" in error_msg.lower():
                return BrowseResponse(
                    path=path,
                    entries=[],
                    error="Export or path not found on NFS server.",
                )
            return BrowseResponse(
                path=path, entries=[], error=error_msg or "Browse failed"
            )

        entries = []
        for line in stdout.decode("utf-8", errors="replace").split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[-1] if parts else line
            if name in (".", ".."):
                continue

            # Detect directory from permission bits (first column like ls -l)
            is_dir = (
                parts[0].startswith("d") if parts and parts[0] else name.endswith("/")
            )
            clean_name = name.rstrip("/")
            entry_path = f"{path}/{clean_name}".lstrip("/") if path else clean_name
            entries.append(
                DirectoryEntry(
                    name=clean_name,
                    path=entry_path,
                    is_dir=is_dir,
                    size=None,
                )
            )

        entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
        return BrowseResponse(path=path or "/", entries=entries)

    except asyncio.TimeoutError:
        return BrowseResponse(
            path=path, entries=[], error=f"Timeout connecting to {host}"
        )
    except Exception as e:
        return BrowseResponse(path=path, entries=[], error=str(e))


@router.get(
    "/filesystem/smb/browse",
    response_model=BrowseResponse,
    tags=["Filesystem Indexer"],
)
async def browse_smb_share(
    host: str,
    share: str,
    path: str = "",
    user: str = "",
    password: str = "",
    domain: str = "",
    _: User = Depends(require_admin),
):
    """
    Browse an SMB share using pysmb (userspace, no mount required).
    """
    if not host or not share:
        return BrowseResponse(
            path=path, entries=[], error="Host and share are required"
        )

    def _browse_smb() -> BrowseResponse:
        try:
            smb_module = importlib.import_module(
                "smb.SMBConnection"
            )  # pyright: ignore[reportMissingImports]
            SMBConnection = getattr(smb_module, "SMBConnection")

            # Resolve hostname to IP for SMB
            try:
                server_ip = socket.gethostbyname(host)
            except socket.gaierror:
                return BrowseResponse(
                    path=path, entries=[], error=f"Cannot resolve hostname: {host}"
                )

            # Create SMB connection
            conn = SMBConnection(
                user or "guest",
                password or "",
                "ragtime",  # client machine name
                host,  # server name
                domain=domain or "",
                use_ntlm_v2=True,
                is_direct_tcp=True,
            )

            try:
                connected = conn.connect(server_ip, 445, timeout=10)
                if not connected:
                    return BrowseResponse(
                        path=path, entries=[], error="Failed to connect to SMB server"
                    )
            except Exception as e:
                return BrowseResponse(
                    path=path, entries=[], error=f"Connection failed: {e}"
                )

            try:
                # List directory contents
                browse_path = path.lstrip("/") if path else ""
                file_list = conn.listPath(share, browse_path or "/")

                entries = []
                for f in file_list:
                    if f.filename in (".", ".."):
                        continue
                    entry_path = (
                        f"{path}/{f.filename}".lstrip("/") if path else f.filename
                    )
                    entries.append(
                        DirectoryEntry(
                            name=f.filename,
                            path=entry_path,
                            is_dir=f.isDirectory,
                            size=f.file_size if not f.isDirectory else None,
                        )
                    )

                # Sort: directories first, then by name
                entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))
                return BrowseResponse(path=path or "/", entries=entries)

            except Exception as e:
                error_msg = str(e)
                if "STATUS_ACCESS_DENIED" in error_msg:
                    return BrowseResponse(
                        path=path, entries=[], error="Access denied - check credentials"
                    )
                return BrowseResponse(path=path, entries=[], error=error_msg)
            finally:
                conn.close()

        except ImportError:
            return BrowseResponse(path=path, entries=[], error="pysmb not installed")
        except Exception as e:
            return BrowseResponse(path=path, entries=[], error=str(e))

    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _browse_smb),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        return BrowseResponse(
            path=path, entries=[], error=f"Timeout connecting to {host}"
        )


# -----------------------------------------------------------------------------
# Filesystem Indexer Endpoints (Admin only)
# -----------------------------------------------------------------------------


class FilesystemIndexStatsResponse(BaseModel):
    """Statistics for a filesystem index."""

    index_name: str
    embedding_count: int
    file_count: int
    last_indexed: Optional[str] = None
    chunk_count: Optional[int] = None
    estimated_memory_mb: Optional[float] = None
    pgvector_size_mb: Optional[float] = None
    vector_store_type: Optional[str] = None
    pgvector_count: Optional[int] = None
    faiss_count: Optional[int] = None


@router.post(
    "/tools/{tool_id}/filesystem/analyze",
    response_model=FilesystemAnalysisJobResponse,
    tags=["Filesystem Indexer"],
)
async def start_filesystem_analysis(
    tool_id: str,
    _user: User = Depends(require_admin),
):
    """
    Start filesystem analysis for a tool. Admin only.

    Analyzes the filesystem to estimate index size, suggest exclusions,
    and identify potential issues before indexing. Returns a job ID
    that can be polled for progress and results.
    """
    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Parse connection config
    try:
        fs_config = FilesystemConnectionConfig(**tool_config.connection_config)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid filesystem configuration: {str(e)}"
        ) from e

    # Start analysis
    job = await filesystem_indexer.start_analysis(
        tool_config_id=tool_id,
        config=fs_config,
    )

    return FilesystemAnalysisJobResponse(
        id=job.id,
        tool_config_id=job.tool_config_id,
        status=job.status,
        progress_percent=job.progress_percent,
        files_scanned=job.files_scanned,
        dirs_scanned=job.dirs_scanned,
        total_dirs_to_scan=job.total_dirs_to_scan,
        current_directory=job.current_directory,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        result=None,
    )


@router.get(
    "/tools/{tool_id}/filesystem/analyze/{job_id}",
    response_model=FilesystemAnalysisJobResponse,
    tags=["Filesystem Indexer"],
)
async def get_filesystem_analysis_status(
    tool_id: str,
    job_id: str,
    _user: User = Depends(require_admin),
):
    """
    Get filesystem analysis job status and results. Admin only.

    Poll this endpoint to track analysis progress. When status is 'completed',
    the result field will contain the full analysis.
    """
    result = await filesystem_indexer.get_analysis_job(job_id)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    job, analysis_result = result

    # Verify tool_id matches
    if job.tool_config_id != tool_id:
        raise HTTPException(
            status_code=404, detail="Analysis job not found for this tool"
        )

    return FilesystemAnalysisJobResponse(
        id=job.id,
        tool_config_id=job.tool_config_id,
        status=job.status,
        progress_percent=job.progress_percent,
        files_scanned=job.files_scanned,
        dirs_scanned=job.dirs_scanned,
        total_dirs_to_scan=job.total_dirs_to_scan,
        current_directory=job.current_directory,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        result=analysis_result,
    )


@router.post(
    "/tools/{tool_id}/filesystem/index",
    response_model=FilesystemIndexJobResponse,
    tags=["Filesystem Indexer"],
)
async def trigger_filesystem_index(
    tool_id: str,
    request: TriggerFilesystemIndexRequest = Body(
        default=TriggerFilesystemIndexRequest()
    ),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Trigger filesystem indexing for a tool. Admin only.

    Starts a background job to index files from the configured path.
    Uses incremental indexing by default (skips unchanged files).
    """
    logger.info(f"trigger_filesystem_index: full_reindex={request.full_reindex}")

    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Parse connection config
    try:
        fs_config = FilesystemConnectionConfig(**tool_config.connection_config)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid filesystem configuration: {str(e)}"
        ) from e

    # Validate pgvector is available
    if not await filesystem_indexer.ensure_pgvector_extension():
        raise HTTPException(
            status_code=503,
            detail="pgvector extension not available. Please install it: CREATE EXTENSION IF NOT EXISTS vector;",
        )

    # Start indexing
    job = await filesystem_indexer.trigger_index(
        tool_config_id=tool_id,
        config=fs_config,
        full_reindex=request.full_reindex,
    )

    return FilesystemIndexJobResponse(
        id=job.id,
        tool_config_id=job.tool_config_id,
        status=job.status,
        index_name=job.index_name,
        progress_percent=job.progress_percent,
        total_files=job.total_files,
        processed_files=job.processed_files,
        skipped_files=job.skipped_files,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/tools/{tool_id}/filesystem/jobs",
    response_model=List[FilesystemIndexJobResponse],
    tags=["Filesystem Indexer"],
)
async def list_filesystem_index_jobs(
    tool_id: str, _user: User = Depends(require_admin)
):
    """List filesystem indexing jobs for a tool. Admin only."""
    jobs = await filesystem_indexer.list_jobs(tool_config_id=tool_id)
    return [
        FilesystemIndexJobResponse(
            id=job.id,
            tool_config_id=job.tool_config_id,
            status=job.status,
            index_name=job.index_name,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            skipped_files=job.skipped_files,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            error_message=job.error_message,
            files_scanned=job.files_scanned,
            dirs_scanned=job.dirs_scanned,
            current_directory=job.current_directory,
            cancel_requested=job.cancel_requested,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.post(
    "/tools/{tool_id}/filesystem/jobs/{job_id}/cancel", tags=["Filesystem Indexer"]
)
async def cancel_filesystem_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """
    Cancel an active filesystem indexing job. Admin only.

    The job will be marked as cancelled and stop processing after the current file.
    """
    # Verify tool exists
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Request cancellation
    cancelled = await filesystem_indexer.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=400, detail="Job not found or already completed"
        )

    return {"success": True, "message": "Cancellation requested"}


@router.post(
    "/tools/{tool_id}/filesystem/jobs/{job_id}/retry",
    response_model=FilesystemIndexJobResponse,
    tags=["Filesystem Indexer"],
)
async def retry_filesystem_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """
    Retry a failed or cancelled filesystem indexing job. Admin only.

    Creates a new job using the same tool configuration.
    """
    # Verify tool exists
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Retry the job
    new_job = await filesystem_indexer.retry_job(job_id)
    if not new_job:
        raise HTTPException(
            status_code=400,
            detail="Job not found or not in a retryable state (must be failed or cancelled)",
        )

    return FilesystemIndexJobResponse(
        id=new_job.id,
        tool_config_id=new_job.tool_config_id,
        status=new_job.status,
        index_name=new_job.index_name,
        progress_percent=new_job.progress_percent,
        total_files=new_job.total_files,
        processed_files=new_job.processed_files,
        skipped_files=new_job.skipped_files,
        total_chunks=new_job.total_chunks,
        processed_chunks=new_job.processed_chunks,
        error_message=new_job.error_message,
        created_at=new_job.created_at,
        started_at=new_job.started_at,
        completed_at=new_job.completed_at,
    )


@router.get(
    "/tools/{tool_id}/filesystem/stats",
    response_model=FilesystemIndexStatsResponse,
    tags=["Filesystem Indexer"],
)
async def get_filesystem_index_stats(
    tool_id: str, _user: User = Depends(require_admin)
):
    """Get statistics for a filesystem index. Admin only."""
    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Parse connection config for index name
    try:
        fs_config = FilesystemConnectionConfig(**tool_config.connection_config)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid filesystem configuration: {str(e)}"
        ) from e

    stats = await filesystem_indexer.get_index_stats(
        fs_config.index_name, fs_config.vector_store_type
    )
    return FilesystemIndexStatsResponse(
        index_name=stats["index_name"],
        embedding_count=stats["embedding_count"],
        file_count=stats["file_count"],
        last_indexed=(
            stats["last_indexed"].isoformat() if stats["last_indexed"] else None
        ),
        chunk_count=stats.get("chunk_count"),
        estimated_memory_mb=stats.get("estimated_memory_mb"),
        pgvector_size_mb=stats.get("pgvector_size_mb"),
        vector_store_type=stats.get("vector_store_type"),
        pgvector_count=stats.get("pgvector_count"),
        faiss_count=stats.get("faiss_count"),
    )


@router.delete("/tools/{tool_id}/filesystem/index", tags=["Filesystem Indexer"])
async def delete_filesystem_index(tool_id: str, _user: User = Depends(require_admin)):
    """Delete all embeddings for a filesystem index. Admin only."""
    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    if tool_config.tool_type != ToolType.FILESYSTEM_INDEXER:
        raise HTTPException(
            status_code=400,
            detail=f"Tool type must be 'filesystem_indexer', got '{tool_config.tool_type.value}'",
        )

    # Parse connection config for index name
    try:
        fs_config = FilesystemConnectionConfig(**tool_config.connection_config)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid filesystem configuration: {str(e)}"
        ) from e

    deleted_count = await filesystem_indexer.delete_index(
        fs_config.index_name, vector_store_type=fs_config.vector_store_type
    )
    return {
        "success": True,
        "message": f"Deleted {deleted_count} embeddings from index '{fs_config.index_name}'",
    }


@router.get("/filesystem/pgvector-status", tags=["Filesystem Indexer"])
async def check_pgvector_status(_user: User = Depends(require_admin)):
    """Check if pgvector extension is installed. Admin only."""
    is_available = await filesystem_indexer.ensure_pgvector_extension()
    return {
        "available": is_available,
        "message": (
            "pgvector extension is installed"
            if is_available
            else "pgvector extension not available"
        ),
    }


@router.post("/garbage-collect", tags=["Admin"])
async def run_garbage_collection(_user: User = Depends(require_admin)):
    """
    Manually trigger garbage collection for orphaned embeddings. Admin only.

    Removes embeddings from pgvector tables that no longer have a corresponding
    tool configuration. This can happen when:
    - Tool configs were deleted before the embedding cleanup was added
    - The server crashed during a tool deletion

    This runs automatically at startup, but can also be triggered manually.
    """
    gc_results = await repository.cleanup_orphaned_embeddings()
    gc_total = sum(gc_results.values())
    return {
        "success": True,
        "total_deleted": gc_total,
        "details": gc_results,
        "message": (
            f"Garbage collected {gc_total} orphaned embedding(s)"
            if gc_total > 0
            else "No orphaned embeddings found"
        ),
    }


# -----------------------------------------------------------------------------
# Ollama Connection Testing
# -----------------------------------------------------------------------------


class OllamaTestRequest(BaseModel):
    """Request to test Ollama connection."""

    protocol: str = Field(default="http", description="Protocol: 'http' or 'https'")
    host: str = Field(default="localhost", description="Ollama server hostname or IP")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")
    embeddings_only: bool = Field(
        default=False,
        description="If True, only return models capable of generating embeddings",
    )


class OllamaModel(BaseModel):
    """Information about an available Ollama model."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    dimensions: Optional[int] = None
    is_embedding_model: bool = False


class OllamaTestResponse(BaseModel):
    """Response from Ollama connection test."""

    success: bool
    message: str
    models: List[OllamaModel] = []
    base_url: str = ""


@router.post("/ollama/test", response_model=OllamaTestResponse, tags=["Settings"])
async def test_ollama_connection(
    request: OllamaTestRequest, _user: User = Depends(require_admin)
):
    """
    Test connection to an Ollama server and retrieve available models.

    When embeddings_only=True, filters to only return models capable of
    generating embeddings (detected via embedding_length in model metadata).

    Queries Ollama's /api/show endpoint to get accurate dimension info.
    """
    base_url = f"{request.protocol}://{request.host}:{request.port}"

    try:
        ollama_models = await list_models(
            base_url=base_url,
            embeddings_only=request.embeddings_only,
            include_dimensions=True,
        )

        models = [
            OllamaModel(
                name=m.name,
                modified_at=m.modified_at,
                size=m.size,
                dimensions=m.dimensions,
                is_embedding_model=m.is_embedding_model,
            )
            for m in ollama_models
        ]

        filter_msg = " embedding" if request.embeddings_only else ""
        return OllamaTestResponse(
            success=True,
            message=f"Connected successfully. Found {len(models)}{filter_msg} model(s).",
            models=models,
            base_url=base_url,
        )

    except httpx.ConnectError:
        return OllamaTestResponse(
            success=False,
            message=f"Cannot connect to Ollama server at {base_url}. Is Ollama running?",
            base_url=base_url,
        )
    except httpx.TimeoutException:
        return OllamaTestResponse(
            success=False,
            message=f"Connection to {base_url} timed out.",
            base_url=base_url,
        )
    except httpx.HTTPStatusError as e:
        return OllamaTestResponse(
            success=False,
            message=f"HTTP error: {e.response.status_code}",
            base_url=base_url,
        )
    except Exception as e:
        return OllamaTestResponse(
            success=False, message=f"Connection failed: {str(e)}", base_url=base_url
        )


# -----------------------------------------------------------------------------
# Ollama Vision Models
# -----------------------------------------------------------------------------


class OllamaVisionModel(BaseModel):
    """Information about an available Ollama vision model."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    capabilities: Optional[List[str]] = None


class OllamaVisionModelsRequest(BaseModel):
    """Request to list vision-capable Ollama models."""

    protocol: str = Field(default="http", description="Protocol: 'http' or 'https'")
    host: str = Field(default="localhost", description="Ollama server hostname or IP")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")


class OllamaVisionModelsResponse(BaseModel):
    """Response with available vision models."""

    success: bool
    message: str
    models: List[OllamaVisionModel] = []
    base_url: str = ""


@router.post(
    "/ollama/vision-models",
    response_model=OllamaVisionModelsResponse,
    tags=["Settings"],
)
async def get_ollama_vision_models(
    request: OllamaVisionModelsRequest, _user: User = Depends(require_admin)
):
    """
    List vision-capable models from an Ollama server.

    Queries Ollama's /api/show endpoint for each model and returns those
    with "vision" in their capabilities array.
    """
    base_url = f"{request.protocol}://{request.host}:{request.port}"

    try:
        vision_models = await list_vision_models(base_url=base_url)

        models = [
            OllamaVisionModel(
                name=m.name,
                modified_at=m.modified_at,
                size=m.size,
                family=m.family,
                parameter_size=m.parameter_size,
                capabilities=m.capabilities,
            )
            for m in vision_models
        ]

        return OllamaVisionModelsResponse(
            success=True,
            message=f"Found {len(models)} vision-capable model(s).",
            models=models,
            base_url=base_url,
        )

    except httpx.ConnectError:
        return OllamaVisionModelsResponse(
            success=False,
            message=f"Cannot connect to Ollama server at {base_url}. Is Ollama running?",
            base_url=base_url,
        )
    except httpx.TimeoutException:
        return OllamaVisionModelsResponse(
            success=False,
            message=f"Connection to {base_url} timed out.",
            base_url=base_url,
        )
    except Exception as e:
        return OllamaVisionModelsResponse(
            success=False,
            message=f"Failed to list vision models: {str(e)}",
            base_url=base_url,
        )


# -----------------------------------------------------------------------------
# Embedding Model Fetching
# -----------------------------------------------------------------------------


class EmbeddingModelsRequest(BaseModel):
    """Request to fetch available embedding models from a provider."""

    provider: str = Field(..., description="Embedding provider: 'openai'")
    api_key: str = Field(..., description="API key for the provider")


class EmbeddingModel(BaseModel):
    """Information about an available embedding model."""

    id: str
    name: str
    dimensions: Optional[int] = None


class EmbeddingModelsResponse(BaseModel):
    """Response from embedding models fetch."""

    success: bool
    message: str
    models: List[EmbeddingModel] = []
    default_model: Optional[str] = None


# OpenAI embedding models - prioritized list (metadata from models.dev)
# These are the recommended models for most use cases
OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@router.post(
    "/embedding/models", response_model=EmbeddingModelsResponse, tags=["Settings"]
)
async def fetch_embedding_models(
    request: EmbeddingModelsRequest, _user: User = Depends(require_admin)
):
    """
    Fetch available embedding models from a provider given a valid API key.

    Uses models.dev metadata to identify embedding-capable models.
    Currently only supports OpenAI. Anthropic does not provide embedding models.
    """
    if request.provider == "openai":
        return await _fetch_openai_embedding_models(request.api_key)
    else:
        return EmbeddingModelsResponse(
            success=False,
            message=f"Unknown or unsupported embedding provider: {request.provider}. Supported: 'openai'",
        )


async def _fetch_openai_embedding_models(api_key: str) -> EmbeddingModelsResponse:
    """
    Fetch available embedding models from OpenAI API.

    Uses models.dev metadata to identify which models are
    embedding models and to get metadata like output dimensions.
    """
    try:
        # Get embedding model metadata from models.dev (cached)
        models_dev_embedding_models = await get_embedding_models()

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()

            data = response.json()
            models = []

            # Filter for embedding models using models.dev data
            for model in data.get("data", []):
                model_id = model.get("id", "")

                # Check if models.dev knows this is an embedding model
                model_info = models_dev_embedding_models.get(model_id)
                if model_info and model_info.provider == "openai":
                    models.append(
                        EmbeddingModel(
                            id=model_id,
                            name=model_id,
                            dimensions=model_info.output_vector_size,
                        )
                    )
                # Fallback: also include models with "embedding" in the name
                # (covers new models not yet in models.dev)
                elif "embedding" in model_id.lower() and model_id not in [
                    m.id for m in models
                ]:
                    models.append(
                        EmbeddingModel(id=model_id, name=model_id, dimensions=None)
                    )

            # Sort: priority models first, then alphabetically
            def sort_key(m: EmbeddingModel) -> tuple:
                try:
                    priority_idx = OPENAI_EMBEDDING_PRIORITY.index(m.id)
                except ValueError:
                    priority_idx = 999
                return (priority_idx, m.id)

            models.sort(key=sort_key)

            return EmbeddingModelsResponse(
                success=True,
                message=f"Found {len(models)} embedding model(s) (validated against models.dev metadata).",
                models=models,
                default_model=(
                    OPENAI_DEFAULT_EMBEDDING_MODEL
                    if any(m.id == OPENAI_DEFAULT_EMBEDDING_MODEL for m in models)
                    else (models[0].id if models else None)
                ),
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return EmbeddingModelsResponse(
                success=False,
                message="Invalid API key. Please check your OpenAI API key.",
            )
        return EmbeddingModelsResponse(
            success=False, message=f"OpenAI API error: {e.response.status_code}"
        )
    except httpx.TimeoutException:
        return EmbeddingModelsResponse(
            success=False, message="Request to OpenAI timed out."
        )
    except Exception as e:
        return EmbeddingModelsResponse(
            success=False, message=f"Failed to fetch OpenAI embedding models: {str(e)}"
        )


# -----------------------------------------------------------------------------
# LLM Provider Model Fetching
# -----------------------------------------------------------------------------


class LLMModelsRequest(BaseModel):
    """Request to fetch available models from an LLM provider."""

    provider: str = Field(
        ...,
        description="LLM provider: 'openai', 'anthropic', or 'github_copilot'",
    )
    api_key: str = Field(default="", description="API key/token for the provider")
    auth_mode: Optional[str] = Field(
        default=None,
        description="Auth mode for github_copilot: 'oauth' (Copilot token) or 'pat' (GitHub PAT).",
    )
    include_directory_models: bool = Field(
        default=False,
        description="Include models from models.dev directory to show known Copilot models.",
    )
    include_anthropic_models: bool = Field(
        default=False,
        description="Include Anthropic/Claude models from directory results.",
    )
    include_google_models: bool = Field(
        default=False,
        description="Include Google/Gemini models from directory results.",
    )


class LLMModel(BaseModel):
    """Information about an available LLM model."""

    id: str
    name: str
    created: Optional[int] = None
    group: Optional[str] = None
    is_latest: bool = False
    max_output_tokens: Optional[int] = None
    context_limit: Optional[int] = None
    capabilities: Optional[List[str]] = None
    supported_endpoints: Optional[List[str]] = None
    reasoning_supported: Optional[bool] = None
    thinking_budget_supported: Optional[bool] = None
    effort_levels: Optional[List[str]] = None


class LLMModelsResponse(BaseModel):
    """Response from LLM models fetch."""

    success: bool
    message: str
    models: List[LLMModel] = []
    default_model: Optional[str] = None


class AvailableModel(BaseModel):
    """A model available for chat."""

    id: str
    name: str
    provider: str  # 'openai' or 'anthropic'
    context_limit: int = 8192  # Max context window tokens
    max_output_tokens: Optional[int] = None  # Max output tokens for this model
    group: Optional[str] = None  # Model group for UI organization
    is_latest: bool = False  # Whether this is the latest version in its group
    created: Optional[int] = None
    capabilities: Optional[List[str]] = None
    supported_endpoints: Optional[List[str]] = None
    reasoning_supported: Optional[bool] = None
    thinking_budget_supported: Optional[bool] = None
    effort_levels: Optional[List[str]] = None


class ProviderModelState(BaseModel):
    """Readiness details for a specific model provider."""

    provider: str
    configured: bool = False
    connected: bool = False
    loading: bool = False
    available: bool = False
    error: Optional[str] = None


class AvailableModelsResponse(BaseModel):
    """Response with all available models from configured providers."""

    models: List[AvailableModel] = []
    default_model: Optional[str] = None
    automatic_default_model: Optional[str] = None
    current_model: Optional[str] = None  # Currently selected model in settings
    allowed_models: List[str] = []  # List of allowed model IDs (for settings UI)
    allowed_openapi_models: List[str] = []  # Separately curated OpenAPI model list
    models_loading: bool = False
    copilot_refresh_in_progress: bool = False
    provider_states: List[ProviderModelState] = []


# Sensible default models for each provider
OPENAI_DEFAULT_MODEL = ""
ANTHROPIC_DEFAULT_MODEL = ""


def _providers_equivalent(selected: str, actual: str) -> bool:
    """Check provider equivalence, treating github_models/github_copilot as aliases.

    Also treats openai as equivalent to github_copilot/github_models because
    GitHub Copilot proxies OpenAI models, so a model registered in
    allowedChatModels as 'openai::gpt-5.x-codex' should still match when the
    model is served by the github_copilot provider.
    """
    if selected == actual:
        return True
    if {selected, actual} <= {"github_copilot", "github_models"}:
        return True
    # GitHub Copilot proxies OpenAI models; allow openai-registered models to
    # match when served by github_copilot or github_models.
    if {selected, actual} <= {"openai", "github_copilot", "github_models"}:
        return True
    return False


def _normalize_provider_alias(provider: Optional[str]) -> str:
    value = (provider or "").strip().lower()
    return "github_copilot" if value == "github_models" else value


async def _is_model_discovery_loading() -> bool:
    """Return whether any provider model-discovery task is currently in flight."""
    async with _model_discovery_lock:
        return any(not task.done() for task in _model_discovery_inflight.values())


def _parse_model_identifier(value: str) -> tuple[Optional[str], str]:
    """Parse model identifier into (provider, model_id), supporting provider::model format."""
    raw = (value or "").strip()
    if not raw:
        return None, ""
    if "::" in raw:
        provider, _, model_id = raw.partition("::")
        provider = provider.strip().lower()
        model_id = model_id.strip()
        if (
            provider
            in {"openai", "anthropic", "ollama", "github_copilot", "github_models"}
            and model_id
        ):
            return provider, model_id
    return None, raw


def _build_scoped_model_identifier(model: AvailableModel) -> str:
    """Build provider-scoped model identifier for durable persistence."""
    return f"{model.provider}::{model.id}"


def _find_available_model_for_identifier(
    models: List[AvailableModel], identifier: Optional[str]
) -> Optional[AvailableModel]:
    """Find a model from available models by bare or provider-scoped identifier."""
    provider, model_id = _parse_model_identifier(identifier or "")
    if not model_id:
        return None

    if provider:
        for model in models:
            if model.id == model_id and _providers_equivalent(provider, model.provider):
                return model
        return None

    for model in models:
        if model.id == model_id:
            return model
    return None


def _find_discovered_model(models: List[LLMModel], model_id: str) -> Optional[LLMModel]:
    """Find a discovered model by exact id, tolerating normalized GitHub prefixes."""
    requested = str(model_id or "").strip().lstrip("/")
    if not requested:
        return None

    for model in models:
        candidate = str(model.id or "").strip().lstrip("/")
        if candidate == requested:
            return model
        if "/" in candidate and candidate.split("/", 1)[1] == requested:
            return model

    return None


def _resolve_ollama_chat_base_url(settings: AppSettings) -> str:
    """Resolve the effective Ollama base URL used for chat model discovery."""
    base_url = str(getattr(settings, "llm_ollama_base_url", "") or "").strip()
    if not base_url or base_url == "http://localhost:11434":
        base_url = str(getattr(settings, "ollama_base_url", "") or "").strip()
    return base_url


async def _fetch_github_provider_models(
    *,
    provider: str,
    settings: AppSettings,
    api_key: str = "",
    auth_mode: Optional[str] = None,
    include_directory_models: bool = False,
    include_anthropic_models: bool = False,
    include_google_models: bool = False,
    force_refresh: bool = False,
) -> LLMModelsResponse:
    """Fetch GitHub-backed models using the same PAT/OAuth flow across routes."""
    normalized_provider = provider.strip().lower()
    resolved_auth_mode, auth_error = _resolve_github_auth_mode(settings, auth_mode)
    if auth_error:
        return LLMModelsResponse(success=False, message=auth_error)
    assert resolved_auth_mode is not None

    token_override = str(api_key or "").strip()
    result: LLMModelsResponse

    if resolved_auth_mode == "pat":
        token = token_override or str(settings.github_models_api_token or "").strip()
        if not token:
            return LLMModelsResponse(
                success=False,
                message="GitHub Copilot PAT mode requires a token. Paste a fine-grained PAT with Models:read.",
            )
        result = await _fetch_github_models_catalog(token)
    else:
        token = token_override
        base_url = COPILOT_DEFAULT_BASE_URL
        refresh_token = ""
        if not token:
            token = str(settings.github_copilot_access_token or "").strip()
            base_url = settings.github_copilot_base_url or COPILOT_DEFAULT_BASE_URL
            refresh_token = str(settings.github_copilot_refresh_token or "").strip()
        if not token:
            return LLMModelsResponse(
                success=False,
                message="GitHub Copilot is not connected. Use OAuth connect in LLM settings first.",
            )

        if not token_override:
            refreshed = await ensure_copilot_token_fresh()
            if refreshed:
                token = refreshed

        result = await _fetch_github_copilot_models(
            token,
            base_url,
            force_refresh=force_refresh,
        )
        if not result.success:
            auth_failed = "authorization failed" in (result.message or "").lower()
            if refresh_token and auth_failed and not token_override:
                try:
                    exchanged_token, exchanged_expires_at = (
                        await exchange_github_token_for_copilot_token(refresh_token)
                    )
                    await repository.update_settings(
                        {
                            "github_copilot_access_token": exchanged_token,
                            "github_copilot_token_expires_at": exchanged_expires_at,
                        }
                    )
                    invalidate_settings_cache()
                    result = await _fetch_github_copilot_models(
                        exchanged_token,
                        base_url,
                        force_refresh=force_refresh,
                    )
                except Exception as exchange_error:
                    logger.warning(
                        "Failed to refresh Copilot bearer token from OAuth token: %s",
                        exchange_error,
                    )

    if include_directory_models and normalized_provider in {
        "github_copilot",
        "github_models",
    }:
        directory_result = await _fetch_copilot_directory_models(
            include_anthropic=include_anthropic_models,
            include_google=include_google_models,
        )
        result = _merge_llm_model_results(result, directory_result)

    return result


async def _validate_conversation_model_selection(
    *,
    provider: str,
    model_id: str,
    include_directory_models: bool = False,
    force_refresh: bool = False,
) -> None:
    """Force-refresh provider discovery before accepting a picker model change."""
    settings = await repository.get_settings()
    if not settings:
        raise HTTPException(
            status_code=503,
            detail="Application settings are unavailable",
        )

    normalized_provider = provider.strip().lower()
    normalized_model = model_id.strip().lstrip("/")
    result: Optional[LLMModelsResponse] = None

    if normalized_provider == "openai":
        api_key = str(settings.openai_api_key or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI is not configured")
        result = await _fetch_openai_models(api_key)
    elif normalized_provider == "anthropic":
        api_key = str(settings.anthropic_api_key or "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="Anthropic is not configured")
        result = await _fetch_anthropic_models(api_key)
    elif normalized_provider == "ollama":
        base_url = _resolve_ollama_chat_base_url(settings)
        if not base_url:
            raise HTTPException(status_code=400, detail="Ollama is not configured")
        result = await _fetch_ollama_llm_models(base_url)
    elif normalized_provider in {"github_models", "github_copilot"}:
        result = await _fetch_github_provider_models(
            provider=normalized_provider,
            settings=settings,
            include_directory_models=include_directory_models,
            include_anthropic_models=True,
            include_google_models=True,
            force_refresh=force_refresh,
        )

    if result is None:
        return

    if not result.success:
        raise HTTPException(
            status_code=503,
            detail=result.message or "Failed to refresh provider models",
        )

    if _find_discovered_model(result.models, normalized_model) is None:
        raise HTTPException(
            status_code=400,
            detail="Selected model is no longer available from the configured provider.",
        )


async def _validate_conversation_model_before_send(stored_model: str) -> None:
    """Verify the current conversation model still exists in the live provider catalog."""
    provider, model_id = _parse_model_identifier(stored_model)
    if not model_id:
        return

    if not provider:
        settings = await repository.get_settings()
        provider = str(getattr(settings, "llm_provider", "openai") or "openai").strip().lower()

    await _validate_conversation_model_selection(
        provider=provider,
        model_id=model_id,
        include_directory_models=False,
        force_refresh=(provider == "github_copilot"),
    )


def _identifier_in_allowed_models(identifier: str, allowed_models: List[str]) -> bool:
    """Check whether a candidate identifier matches any allowed model entry."""
    if not identifier:
        return False

    candidate_provider, candidate_model_id = _parse_model_identifier(identifier)
    if not candidate_model_id:
        return False

    for allowed in allowed_models:
        allowed_provider, allowed_model_id = _parse_model_identifier(allowed)
        if allowed_model_id != candidate_model_id:
            continue
        if allowed_provider is None or candidate_provider is None:
            return True
        if _providers_equivalent(allowed_provider, candidate_provider):
            return True

    return False


def _normalize_github_domain(url_or_domain: str) -> str:
    """Normalize enterprise URL/domain to plain host."""
    value = (url_or_domain or "").strip()
    if not value:
        return ""
    value = re.sub(r"^https?://", "", value, flags=re.IGNORECASE)
    return value.rstrip("/")


def _copilot_base_url_for_domain(domain: str) -> str:
    """Resolve Copilot API base URL for github.com vs enterprise domains."""
    normalized = _normalize_github_domain(domain)
    if not normalized or normalized == "github.com":
        return COPILOT_DEFAULT_BASE_URL
    return f"https://copilot-api.{normalized}"


# Re-export for backward compatibility within this module.
_exchange_github_token_for_copilot_token = exchange_github_token_for_copilot_token


class CopilotDeviceStartRequest(BaseModel):
    """Request to begin GitHub Copilot OAuth device flow."""

    deployment_type: str = Field(
        default="github.com",
        description="GitHub deployment type: 'github.com' or 'enterprise'.",
    )
    enterprise_url: Optional[str] = Field(
        default=None,
        description="GitHub Enterprise URL/domain (required when deployment_type='enterprise').",
    )


class CopilotDeviceStartResponse(BaseModel):
    """Response for GitHub Copilot OAuth device flow start."""

    success: bool
    request_id: str
    verification_uri: str
    verification_uri_complete: Optional[str] = None
    user_code: str
    interval: int
    expires_in: int
    deployment_type: str
    enterprise_url: Optional[str] = None


class CopilotDevicePollRequest(BaseModel):
    """Request to poll GitHub Copilot OAuth device flow status."""

    request_id: str


class CopilotDevicePollResponse(BaseModel):
    """Response for polling GitHub Copilot OAuth device flow status."""

    success: bool
    status: str
    message: str
    retry_after_seconds: Optional[int] = None


class CopilotAuthStatusResponse(BaseModel):
    """Current GitHub Copilot auth status."""

    connected: bool
    deployment_type: str
    enterprise_url: Optional[str] = None
    base_url: str
    token_expires_at: Optional[datetime] = None


@router.post(
    "/github-copilot/device/start",
    response_model=CopilotDeviceStartResponse,
    tags=["Settings"],
)
async def start_copilot_device_flow(
    request: CopilotDeviceStartRequest,
    _user: User = Depends(require_admin),
):
    """Start GitHub Copilot OAuth device flow and return verification code."""
    deployment_type = (request.deployment_type or "github.com").strip().lower()
    if deployment_type not in {"github.com", "enterprise"}:
        raise HTTPException(
            status_code=400,
            detail="deployment_type must be 'github.com' or 'enterprise'",
        )

    domain = "github.com"
    enterprise_url = None
    if deployment_type == "enterprise":
        domain = _normalize_github_domain(request.enterprise_url or "")
        if not domain:
            raise HTTPException(
                status_code=400,
                detail="enterprise_url is required for enterprise deployment",
            )
        enterprise_url = domain

    device_code_url = f"https://{domain}/login/device/code"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "ragtime",
            }

            # GitHub OAuth Apps do not always accept `models:read` in device flow.
            # Try broader scope first, then gracefully fall back to read:user.
            response = await client.post(
                device_code_url,
                headers=headers,
                json={
                    "client_id": GITHUB_COPILOT_CLIENT_ID,
                    "scope": "read:user models:read",
                },
            )
            if response.status_code == 400:
                logger.warning(
                    "GitHub device flow rejected models:read scope; retrying with read:user"
                )
                response = await client.post(
                    device_code_url,
                    headers=headers,
                    json={
                        "client_id": GITHUB_COPILOT_CLIENT_ID,
                        "scope": "read:user",
                    },
                )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"GitHub device flow request failed ({e.response.status_code})",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to start GitHub device flow: {str(e)}",
        ) from e

    request_id = secrets.token_urlsafe(24)
    interval = max(int(data.get("interval", 5)), 1)
    expires_in = max(int(data.get("expires_in", 900)), 60)
    _copilot_device_requests[request_id] = {
        "device_code": data.get("device_code", ""),
        "domain": domain,
        "deployment_type": deployment_type,
        "enterprise_url": enterprise_url,
        "interval": interval,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
    }

    return CopilotDeviceStartResponse(
        success=True,
        request_id=request_id,
        verification_uri=data.get(
            "verification_uri", "https://github.com/login/device"
        ),
        verification_uri_complete=data.get("verification_uri_complete"),
        user_code=data.get("user_code", ""),
        interval=interval,
        expires_in=expires_in,
        deployment_type=deployment_type,
        enterprise_url=enterprise_url,
    )


@router.post(
    "/github-copilot/device/poll",
    response_model=CopilotDevicePollResponse,
    tags=["Settings"],
)
async def poll_copilot_device_flow(
    request: CopilotDevicePollRequest,
    _user: User = Depends(require_admin),
):
    """Poll GitHub Copilot OAuth device flow and persist token on success."""
    state = _copilot_device_requests.get(request.request_id)
    if not state:
        raise HTTPException(
            status_code=404, detail="Device authorization request not found"
        )

    now = datetime.now(timezone.utc)
    if now >= state["expires_at"]:
        _copilot_device_requests.pop(request.request_id, None)
        return CopilotDevicePollResponse(
            success=False,
            status="expired",
            message="Device authorization code expired. Start again.",
        )

    access_token_url = f"https://{state['domain']}/login/oauth/access_token"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                access_token_url,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "ragtime",
                },
                json={
                    "client_id": GITHUB_COPILOT_CLIENT_ID,
                    "device_code": state["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            if not response.is_success:
                return CopilotDevicePollResponse(
                    success=False,
                    status="failed",
                    message=f"GitHub OAuth token request failed ({response.status_code}).",
                )
            data = response.json()
    except Exception as e:
        return CopilotDevicePollResponse(
            success=False,
            status="failed",
            message=f"GitHub OAuth polling failed: {str(e)}",
        )

    access_token = data.get("access_token")
    if access_token:
        oauth_expires_in = data.get("expires_in")
        oauth_expires_at = None
        if isinstance(oauth_expires_in, (int, float)) and int(oauth_expires_in) > 0:
            oauth_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=int(oauth_expires_in)
            )

        # Capture the OAuth refresh_token (ghr_*) if present.  GitHub Apps
        # with token expiration enabled return this alongside the access_token.
        # It can be used to obtain a new ghu_* after the original expires.
        oauth_refresh_token = (data.get("refresh_token") or "").strip()

        # Prefer Copilot bearer tokens for model/chat APIs.
        # Keep the raw OAuth token in refresh_token so we can re-exchange later.
        copilot_access_token = access_token
        copilot_expires_at = oauth_expires_at
        try:
            exchanged_token, exchanged_expires_at = (
                await _exchange_github_token_for_copilot_token(access_token)
            )
            copilot_access_token = exchanged_token
            copilot_expires_at = exchanged_expires_at or oauth_expires_at
        except Exception as exchange_error:
            logger.warning(
                "GitHub->Copilot token exchange failed; using OAuth token directly: %s",
                exchange_error,
            )

        base_url = _copilot_base_url_for_domain(state["domain"])
        settings_update: dict = {
            "github_copilot_access_token": copilot_access_token,
            # Keep OAuth token for future exchange refresh attempts.
            "github_copilot_refresh_token": access_token,
            "github_copilot_token_expires_at": copilot_expires_at,
            "github_copilot_enterprise_url": state.get("enterprise_url"),
            "github_copilot_base_url": base_url,
        }
        if oauth_refresh_token:
            settings_update["github_copilot_oauth_refresh_token"] = oauth_refresh_token
        await repository.update_settings(settings_update)
        invalidate_settings_cache()
        _copilot_device_requests.pop(request.request_id, None)
        return CopilotDevicePollResponse(
            success=True,
            status="connected",
            message="GitHub Copilot connected successfully.",
        )

    oauth_error = data.get("error")
    if oauth_error == "authorization_pending":
        retry_after = max(
            int(state.get("interval", 5)) + OAUTH_POLLING_SAFETY_MARGIN_SECONDS,
            1,
        )
        return CopilotDevicePollResponse(
            success=True,
            status="pending",
            message="Waiting for authorization in browser...",
            retry_after_seconds=retry_after,
        )

    if oauth_error == "slow_down":
        new_interval = int(data.get("interval") or state.get("interval", 5)) + 5
        state["interval"] = max(new_interval, 1)
        retry_after = state["interval"] + OAUTH_POLLING_SAFETY_MARGIN_SECONDS
        return CopilotDevicePollResponse(
            success=True,
            status="pending",
            message="GitHub requested slower polling; retrying...",
            retry_after_seconds=retry_after,
        )

    if oauth_error:
        _copilot_device_requests.pop(request.request_id, None)
        return CopilotDevicePollResponse(
            success=False,
            status="failed",
            message=f"GitHub OAuth failed: {oauth_error}",
        )

    return CopilotDevicePollResponse(
        success=False,
        status="failed",
        message="GitHub OAuth did not return a token.",
    )


@router.get(
    "/github-copilot/auth/status",
    response_model=CopilotAuthStatusResponse,
    tags=["Settings"],
)
async def get_copilot_auth_status(_user: User = Depends(require_admin)):
    """Get current GitHub Copilot auth status from settings."""
    app_settings = await repository.get_settings()
    connected = bool((app_settings.github_copilot_access_token or "").strip())
    enterprise_url = app_settings.github_copilot_enterprise_url or None
    deployment_type = "enterprise" if enterprise_url else "github.com"
    return CopilotAuthStatusResponse(
        connected=connected,
        deployment_type=deployment_type,
        enterprise_url=enterprise_url,
        base_url=app_settings.github_copilot_base_url or COPILOT_DEFAULT_BASE_URL,
        token_expires_at=app_settings.github_copilot_token_expires_at,
    )


@router.post("/github-copilot/auth/clear", tags=["Settings"])
async def clear_copilot_auth(_user: User = Depends(require_admin)):
    """Clear stored GitHub Copilot auth credentials."""
    await repository.update_settings(
        {
            "github_copilot_access_token": "",
            "github_copilot_refresh_token": "",
            "github_copilot_oauth_refresh_token": "",
            "github_copilot_token_expires_at": None,
            "github_copilot_enterprise_url": "",
            "github_copilot_base_url": COPILOT_DEFAULT_BASE_URL,
        }
    )
    invalidate_settings_cache()
    return {"success": True, "message": "GitHub Copilot credentials cleared."}


def _group_models(models: List[LLMModel], provider: str) -> List[LLMModel]:
    """
    Group models into families and identify the latest version.
    Does NOT remove dated versions, but marks grouping metadata.
    """
    # Use the same logic as _assign_model_groups but for LLMModel
    for model in models:
        mid = model.id.lower()
        provider_patterns = MODEL_FAMILY_PATTERNS.get(provider, [])
        found_group = False

        for pattern, group_name in provider_patterns:
            match = re.search(pattern, mid)
            if match:
                if group_name:
                    model.group = group_name
                elif match.lastindex:
                    model.group = _derive_group_label(
                        provider,
                        model.id.lower(),
                        match.group(1),
                    )
                found_group = True
                break

        if not found_group:
            model.group = f"Other {provider.title()}"

    # Identify 'is_latest' within each group
    grouped = defaultdict(list)
    for model in models:
        grouped[model.group].append(model)

    for _, group_models in grouped.items():
        if not group_models:
            continue

        # Sort: version (higher first), then created date, then ID length
        group_models.sort(
            key=lambda m: (-_extract_version(m.name), -(m.created or 0), len(m.id))
        )

        # Mark first as latest
        group_models[0].is_latest = True

    return models


def _derive_group_label(provider: str, model_id: str, capture: str) -> str:
    """Build a group label from regex captures for dynamic family patterns."""
    if provider in {"openai", "github_copilot", "github_models"} and "gpt-" in model_id:
        return f"GPT-{capture}"

    if provider == "ollama":
        return capture.title()

    return capture


def _extract_version(name: str) -> float:
    """Extract numeric version from model name.

    Handles various naming conventions:
    - Anthropic: 'Claude Haiku 4.5' -> 4.5
    - OpenAI display: 'GPT-4.1 Mini' -> 4.1
    - OpenAI ID: 'gpt-4.1-mini' -> 4.1
    - Dated versions: 'gpt-4-0613' -> 4.0 (base version, no sub-version)
    - Gemini: 'Gemini 3.1 Pro' -> 3.1, 'Gemini 3 Flash (Preview)' -> 3.0
    """
    name_lower = name.lower()

    # OpenAI: Extract version from gpt-X.Y pattern (e.g., gpt-4.1-mini -> 4.1)
    gpt_match = re.search(r"gpt-(\d+(?:\.\d+)?)", name_lower)
    if gpt_match:
        return float(gpt_match.group(1))

    # Anthropic/general: version at end of name (e.g., 'Claude Haiku 4.5' -> 4.5)
    end_match = re.search(r"(\d+(?:\.\d+)?)\s*$", name)
    if end_match:
        return float(end_match.group(1))

    # Gemini/general: version number followed by non-digit qualifier
    # e.g. 'Gemini 3.1 Pro' -> 3.1, 'Gemini 3 Flash (Preview)' -> 3.0
    mid_match = re.search(r"(\d+(?:\.\d+)?)\s+\w", name)
    if mid_match:
        return float(mid_match.group(1))

    return 0.0


def _assign_model_groups(models: List[AvailableModel]) -> List[AvailableModel]:
    """Assign UI group labels to models for better organization."""
    for model in models:
        mid = model.id.lower()
        provider_patterns = MODEL_FAMILY_PATTERNS.get(model.provider, [])
        found_group = False

        for pattern, group_name in provider_patterns:
            match = re.search(pattern, mid)
            if match:
                if group_name:
                    model.group = group_name
                elif match.lastindex:
                    model.group = _derive_group_label(
                        model.provider,
                        model.id.lower(),
                        match.group(1),
                    )
                found_group = True
                break

        if not found_group:
            model.group = f"Other {model.provider.title()}"

    # Grouping and is_latest logic
    grouped = defaultdict(list)
    for model in models:
        grouped[(model.provider, model.group)].append(model)

    for (provider, _group), group_models in grouped.items():
        if not group_models:
            continue

        if provider == "ollama":
            # For Ollama, use the ':latest' tag as the authoritative marker
            for m in group_models:
                if m.id.endswith(":latest"):
                    m.is_latest = True
        else:
            # Sort: version (higher first), then created date, then ID length
            group_models.sort(
                key=lambda m: (
                    -_extract_version(m.name),
                    -(m.created or 0),
                    len(m.id),
                )
            )
            # Mark the first one as latest
            group_models[0].is_latest = True

    return models


@router.post("/llm/models", response_model=LLMModelsResponse, tags=["Settings"])
async def fetch_llm_models(
    request: LLMModelsRequest, _user: User = Depends(require_admin)
):
    """
    Fetch available models from an LLM provider given a valid API key.

    Queries the provider's API and returns a list of available chat/completion models.
    """
    if request.provider == "openai":
        return await _fetch_openai_models(request.api_key)
    elif request.provider == "anthropic":
        return await _fetch_anthropic_models(request.api_key)
    elif request.provider in {"github_copilot", "github_models"}:
        settings = await repository.get_settings()
        return await _fetch_github_provider_models(
            provider=request.provider,
            settings=settings,
            api_key=request.api_key,
            auth_mode=request.auth_mode,
            include_directory_models=request.include_directory_models,
            include_anthropic_models=request.include_anthropic_models,
            include_google_models=request.include_google_models,
        )
    else:
        return LLMModelsResponse(
            success=False,
            message=f"Unknown provider: {request.provider}. Supported: 'openai', 'anthropic', 'github_copilot'",
        )


def _merge_llm_model_results(
    primary: LLMModelsResponse, extra: LLMModelsResponse
) -> LLMModelsResponse:
    """Merge model results by id while keeping primary metadata precedence."""
    if not primary.success and extra.success:
        return extra
    if primary.success and not extra.success:
        return primary
    if not primary.success and not extra.success:
        return primary

    merged: dict[str, LLMModel] = {m.id: m for m in primary.models}
    for model in extra.models:
        merged.setdefault(model.id, model)

    models = list(merged.values())
    models = _group_models(models, "github_copilot")
    models.sort(
        key=lambda m: (m.group or "", m.is_latest, m.created or 0, m.name.lower()),
        reverse=True,
    )

    return LLMModelsResponse(
        success=True,
        message=f"Found {len(models)} model(s).",
        models=models,
        default_model=primary.default_model or (models[0].id if models else None),
    )


async def _fetch_copilot_directory_models(
    include_anthropic: bool = False,
    include_google: bool = False,
) -> LLMModelsResponse:
    """Fetch known GitHub Copilot models from models.dev directory."""
    cache_key = (
        "copilot-directory:"
        f"anthropic={int(include_anthropic)}:google={int(include_google)}"
    )

    async def _fetch_uncached() -> LLMModelsResponse:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    MODELS_DEV_API_URL,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "ragtime",
                    },
                )
                response.raise_for_status()

            payload = response.json()
            provider_payload = (
                payload.get("github-copilot", {}) if isinstance(payload, dict) else {}
            )
            models_obj = (
                provider_payload.get("models", {})
                if isinstance(provider_payload, dict)
                else {}
            )
            if not isinstance(models_obj, dict):
                return LLMModelsResponse(
                    success=False,
                    message="models.dev github-copilot payload was malformed.",
                )

            models: list[LLMModel] = []
            for model_key, row in models_obj.items():
                if not isinstance(row, dict):
                    continue
                model_id = str(row.get("id") or model_key or "").strip()
                model_id = re.sub(r"/+", "/", model_id).lstrip("/")
                if not model_id:
                    continue

                model_id_l = model_id.lower()
                if not include_anthropic and (
                    "claude" in model_id_l or "anthropic" in model_id_l
                ):
                    continue
                if not include_google and (
                    "gemini" in model_id_l or "google" in model_id_l
                ):
                    continue

                output_modalities = row.get("modalities", {}).get("output")
                if isinstance(output_modalities, list) and output_modalities:
                    text_outputs = {str(x).lower() for x in output_modalities}
                    if "text" not in text_outputs:
                        continue

                output_limit = None
                if isinstance(row.get("limit"), dict):
                    limit_out = row.get("limit", {}).get("output")
                    if isinstance(limit_out, int):
                        output_limit = limit_out

                models.append(
                    LLMModel(
                        id=model_id,
                        name=str(row.get("name") or model_id),
                        max_output_tokens=output_limit,
                    )
                )

            if not models:
                return LLMModelsResponse(
                    success=False,
                    message="No matching directory models found for the selected filters.",
                )

            models = _group_models(models, "github_copilot")
            models.sort(
                key=lambda m: (
                    m.group or "",
                    m.is_latest,
                    m.created or 0,
                    m.name.lower(),
                ),
                reverse=True,
            )
            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} known Copilot model(s) from models.dev.",
                models=models,
                default_model=models[0].id if models else None,
            )
        except Exception as e:
            return LLMModelsResponse(
                success=False,
                message=f"Failed to fetch models.dev directory: {str(e)}",
            )

    return await _get_or_fetch_model_discovery(cache_key, _fetch_uncached)


async def _fetch_openai_models(api_key: str) -> LLMModelsResponse:
    """Fetch available models from OpenAI API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()

            data = response.json()
            models = []

            # Filter for chat-capable models using LiteLLM's function calling support
            for model in data.get("data", []):
                model_id = model.get("id", "")
                (
                    capabilities,
                    supported_endpoints,
                    reasoning_supported,
                    thinking_budget_supported,
                    effort_levels,
                ) = _extract_provider_capability_metadata(model)
                # Include GPT models suitable for chat (exclude embeddings, whisper, tts, dall-e, etc.)
                if model_id.startswith("gpt-") and not any(
                    x in model_id
                    for x in ["instruct", "vision", "realtime", "audio", "tts"]
                ):
                    # Check if model supports function calling (indicates chat capability)
                    if await supports_function_calling(model_id):
                        output_limit = await get_output_limit(model_id)
                        models.append(
                            LLMModel(
                                id=model_id,
                                name=model_id,
                                created=model.get("created"),
                                max_output_tokens=output_limit,
                                capabilities=capabilities or None,
                                supported_endpoints=supported_endpoints or None,
                                reasoning_supported=reasoning_supported,
                                thinking_budget_supported=thinking_budget_supported,
                                effort_levels=effort_levels or None,
                            )
                        )

            # Curate models to remove dated duplicates
            models = _group_models(models, "openai")

            # Sort: priority models first, then alphabetically
            # Priority sorting is now handled by is_latest flag for UI
            # Here we just ensure deterministic order or created time
            models.sort(
                key=lambda m: (m.group or "", m.is_latest, m.created or 0), reverse=True
            )

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} chat model(s).",
                models=models,
                default_model=(
                    OPENAI_DEFAULT_MODEL
                    if OPENAI_DEFAULT_MODEL
                    and any(m.id == OPENAI_DEFAULT_MODEL for m in models)
                    else (models[0].id if models else None)
                ),
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return LLMModelsResponse(
                success=False,
                message="Invalid API key. Please check your OpenAI API key.",
            )
        return LLMModelsResponse(
            success=False, message=f"OpenAI API error: {e.response.status_code}"
        )
    except httpx.TimeoutException:
        return LLMModelsResponse(success=False, message="Request to OpenAI timed out.")
    except Exception as e:
        return LLMModelsResponse(
            success=False, message=f"Failed to fetch OpenAI models: {str(e)}"
        )


async def _fetch_anthropic_models(api_key: str) -> LLMModelsResponse:
    """Fetch available models from Anthropic API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            )
            response.raise_for_status()

            data = response.json()
            models = []

            for model in data.get("data", []):
                model_id = model.get("id", "")
                display_name = model.get("display_name", model_id)
                (
                    capabilities,
                    supported_endpoints,
                    reasoning_supported,
                    thinking_budget_supported,
                    effort_levels,
                ) = _extract_provider_capability_metadata(model)

                if supported_endpoints:
                    register_model_supported_endpoints(model_id, supported_endpoints)
                if reasoning_supported or thinking_budget_supported:
                    register_model_reasoning_capabilities(
                        model_id,
                        reasoning_supported=bool(reasoning_supported),
                        thinking_budget_supported=bool(thinking_budget_supported),
                    )
                # All Claude models support function calling (chat capable)
                output_limit = await get_output_limit(model_id)
                models.append(
                    LLMModel(
                        id=model_id,
                        name=display_name,
                        created=None,
                        max_output_tokens=output_limit,
                        capabilities=capabilities or None,
                        supported_endpoints=supported_endpoints or None,
                        reasoning_supported=reasoning_supported,
                        thinking_budget_supported=thinking_budget_supported,
                        effort_levels=effort_levels or None,
                    )
                )

            # Curate models to remove dated duplicates
            models = _group_models(models, "anthropic")

            # Sort: Alphabetically
            models.sort(key=lambda m: (m.group or "", m.is_latest, m.id), reverse=True)

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} model(s).",
                models=models,
                default_model=(
                    ANTHROPIC_DEFAULT_MODEL
                    if ANTHROPIC_DEFAULT_MODEL
                    and any(m.id == ANTHROPIC_DEFAULT_MODEL for m in models)
                    else (models[0].id if models else None)
                ),
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return LLMModelsResponse(
                success=False,
                message="Invalid API key. Please check your Anthropic API key.",
            )
        return LLMModelsResponse(
            success=False, message=f"Anthropic API error: {e.response.status_code}"
        )
    except httpx.TimeoutException:
        return LLMModelsResponse(
            success=False, message="Request to Anthropic timed out."
        )
    except Exception as e:
        return LLMModelsResponse(
            success=False, message=f"Failed to fetch Anthropic models: {str(e)}"
        )


async def _fetch_ollama_llm_models(base_url: str) -> LLMModelsResponse:
    """Fetch available models from Ollama server for LLM use."""
    try:
        # Check reachability first
        reachable, error_msg = await is_reachable(base_url)
        if not reachable:
            return LLMModelsResponse(
                success=False,
                message=error_msg or f"Cannot connect to Ollama server at {base_url}.",
            )

        # List all models (not just embedding models, since this is for LLM)
        ollama_models = await ollama_list_models(
            base_url, embeddings_only=False, include_dimensions=False
        )

        models = [LLMModel(id=m.name, name=m.name) for m in ollama_models]

        # Fetch details for each model to get context window and filter embeddings
        # We allow this to be "best effort" - if details fail, we just don't update limit
        embedding_model_ids: set[str] = set()
        try:
            # Create a client for reuse
            async with httpx.AsyncClient(timeout=10.0) as client:
                tasks = []
                for m in models:
                    # Mark as supporting function calling
                    update_model_function_calling(m.id, True)
                    tasks.append(get_model_details(m.id, base_url, client))

                # Run all detail fetches concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for idx, details in enumerate(results):
                    if isinstance(details, dict) and details:
                        capabilities = extract_capabilities(details)
                        models[idx].capabilities = capabilities or None
                        models[idx].reasoning_supported = (
                            True if "thinking" in capabilities else None
                        )
                        # Filter out embedding-only models (nomic-embed-text, etc.)
                        if is_embedding_capable(details, models[idx].id):
                            embedding_model_ids.add(models[idx].id)
                            continue

                        # Extract context length from Ollama API
                        # (architecture-agnostic via extract_effective_context_length)
                        ctx_len = extract_effective_context_length(details)

                        if ctx_len:
                            update_model_limit(models[idx].id, ctx_len)
                            models[idx].max_output_tokens = ctx_len
                            models[idx].context_limit = ctx_len
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama model details: {e}")

        # Remove embedding-only models from the list
        if embedding_model_ids:
            models = [m for m in models if m.id not in embedding_model_ids]
            logger.debug(
                f"Filtered out {len(embedding_model_ids)} embedding-only Ollama model(s): {embedding_model_ids}"
            )

        return LLMModelsResponse(
            success=True,
            message=f"Found {len(models)} model(s).",
            models=models,
            default_model=models[0].id if models else None,
        )

    except Exception as e:
        return LLMModelsResponse(
            success=False,
            message=f"Failed to fetch Ollama models: {str(e)}",
        )


def _extract_context_limit_from_model_row(row: dict[str, Any]) -> int | None:
    """Extract context-window token limit from provider model metadata payloads."""

    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            coerced = int(value)
            return coerced if coerced > 0 else None
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                parsed = int(value)
                return parsed if parsed > 0 else None
        return None

    context_window_keys = [
        "context_limit",
        "context_window",
        "context_window_tokens",
        "context_length",
        "max_context_tokens",
        "max_context_window_tokens",
    ]

    input_budget_keys = [
        "max_input_tokens",
        "max_prompt_tokens",
        "input_tokens",
    ]

    capabilities_obj = row.get("capabilities")
    if isinstance(capabilities_obj, dict):
        capabilities_limits = capabilities_obj.get("limits")
        if isinstance(capabilities_limits, dict):
            for key in context_window_keys:
                parsed = _coerce_int(capabilities_limits.get(key))
                if parsed is not None:
                    return parsed
            for key in input_budget_keys:
                parsed = _coerce_int(capabilities_limits.get(key))
                if parsed is not None:
                    return parsed

    limits_obj = row.get("limits")
    if isinstance(limits_obj, dict):
        for key in context_window_keys:
            parsed = _coerce_int(limits_obj.get(key))
            if parsed is not None:
                return parsed
        for key in input_budget_keys:
            parsed = _coerce_int(limits_obj.get(key))
            if parsed is not None:
                return parsed

    for key in context_window_keys:
        parsed = _coerce_int(row.get(key))
        if parsed is not None:
            return parsed

    for key in input_budget_keys:
        parsed = _coerce_int(row.get(key))
        if parsed is not None:
            return parsed

    return None


def _extract_output_limit_from_model_row(row: dict[str, Any]) -> int | None:
    """Extract output-token limit from provider model metadata payloads."""

    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            coerced = int(value)
            return coerced if coerced > 0 else None
        if isinstance(value, str):
            value = value.strip()
            if value.isdigit():
                parsed = int(value)
                return parsed if parsed > 0 else None
        return None

    keys = ["max_output_tokens", "output_tokens"]

    capabilities_obj = row.get("capabilities")
    if isinstance(capabilities_obj, dict):
        capabilities_limits = capabilities_obj.get("limits")
        if isinstance(capabilities_limits, dict):
            for key in keys:
                parsed = _coerce_int(capabilities_limits.get(key))
                if parsed is not None:
                    return parsed

    limits_obj = row.get("limits")
    if isinstance(limits_obj, dict):
        for key in keys:
            parsed = _coerce_int(limits_obj.get(key))
            if parsed is not None:
                return parsed

    for key in keys:
        parsed = _coerce_int(row.get(key))
        if parsed is not None:
            return parsed

    return None


def _extract_provider_capability_metadata(
    row: dict[str, Any],
) -> tuple[list[str], list[str], bool | None, bool | None, list[str]]:
    """Extract normalized capability metadata from provider model payloads."""

    capabilities_out: list[str] = []
    supported_endpoints_out: list[str] = []
    reasoning_supported: bool | None = None
    thinking_budget_supported: bool | None = None
    effort_levels: list[str] = []

    supported_endpoints = row.get("supportedEndpoints")
    if isinstance(supported_endpoints, list):
        supported_endpoints_out = [str(item) for item in supported_endpoints]

    capabilities_obj = row.get("capabilities")
    if isinstance(capabilities_obj, list):
        capabilities_out = [str(item).lower() for item in capabilities_obj]
        if "reasoning" in capabilities_out or "reasoning_effort" in capabilities_out:
            reasoning_supported = True
        if "thinking_budget" in capabilities_out:
            thinking_budget_supported = True
    elif isinstance(capabilities_obj, dict):
        supports_obj = capabilities_obj.get("supports")
        if isinstance(supports_obj, list):
            capabilities_out = [str(flag).lower() for flag in supports_obj]
        elif isinstance(supports_obj, dict):
            capabilities_out = [
                str(flag).lower() for flag, enabled in supports_obj.items() if enabled
            ]
            reasoning_effort_value = supports_obj.get("reasoning_effort")
            if isinstance(reasoning_effort_value, list):
                effort_levels.extend(
                    str(level).lower() for level in reasoning_effort_value if level
                )

        thinking_obj = capabilities_obj.get("thinking")
        if isinstance(thinking_obj, dict):
            if thinking_obj.get("supported") is True:
                reasoning_supported = True

            thinking_types = thinking_obj.get("types")
            if isinstance(thinking_types, dict):
                enabled_obj = thinking_types.get("enabled")
                if (
                    isinstance(enabled_obj, dict)
                    and enabled_obj.get("supported") is True
                ):
                    thinking_budget_supported = True
                elif enabled_obj is True:
                    thinking_budget_supported = True

        effort_obj = capabilities_obj.get("effort")
        if isinstance(effort_obj, dict):
            if effort_obj.get("supported") is True:
                reasoning_supported = True
            for key, value in effort_obj.items():
                if key == "supported":
                    continue
                if isinstance(value, dict) and value.get("supported") is True:
                    effort_levels.append(str(key).lower())
                elif value is True:
                    effort_levels.append(str(key).lower())

        if capabilities_out:
            if (
                "reasoning" in capabilities_out
                or "reasoning_effort" in capabilities_out
            ):
                reasoning_supported = True
            if "thinking_budget" in capabilities_out:
                thinking_budget_supported = True

    effort_levels = [level for level in effort_levels if level]
    if effort_levels:
        reasoning_supported = True

    return (
        capabilities_out,
        supported_endpoints_out,
        reasoning_supported,
        thinking_budget_supported,
        effort_levels,
    )


async def _fetch_github_copilot_models(
    access_token: str,
    base_url: str = COPILOT_DEFAULT_BASE_URL,
    *,
    force_refresh: bool = False,
) -> LLMModelsResponse:
    """Fetch available models from GitHub Copilot API."""
    normalized_base = (base_url or COPILOT_DEFAULT_BASE_URL).rstrip("/")
    cache_key = (
        "copilot-models:"
        f"base={normalized_base}:token={_copilot_token_fingerprint(access_token)}"
    )

    def _normalize_model_id(raw_id: str) -> str:
        model_id = raw_id.strip()
        model_id = re.sub(r"/+", "/", model_id)
        if "/" in model_id:
            # GitHub Models catalog uses publisher-prefixed IDs like
            # "openai/gpt-4.1". Copilot/OpenAI-compatible chat APIs use
            # the model slug itself.
            model_id = model_id.split("/", 1)[1]
        # Some catalog payloads may include double-slash forms like
        # "anthropic//claude-haiku-4.5"; ensure we never surface "/model".
        model_id = model_id.lstrip("/")
        return model_id

    def _extract_model_rows(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            data = payload.get("data", payload)
            if isinstance(data, list):
                return [row for row in data if isinstance(row, dict)]
            return []
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        return []

    async def _fetch_uncached() -> LLMModelsResponse:
        endpoint_specs: list[tuple[str, dict[str, str]]] = [
            (
                f"{normalized_base}/models",
                build_copilot_headers(access_token=access_token),
            ),
        ]

        models_by_id: dict[str, LLMModel] = {}
        successful_sources = 0
        last_error: Optional[str] = None

        for url, headers in endpoint_specs:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(
                        url,
                        headers=headers,
                    )

                    if response.status_code in (401, 403):
                        return LLMModelsResponse(
                            success=False,
                            message="GitHub Copilot authorization failed. Please reconnect GitHub Copilot.",
                        )
                    if response.status_code == 404:
                        last_error = f"Endpoint not found at {url}"
                        continue

                    response.raise_for_status()
                    payload = response.json()
                    rows = _extract_model_rows(payload)
                    if not rows:
                        continue

                    successful_sources += 1
                    for row in rows:
                        raw_id = str(row.get("id", "")).strip()
                        if not raw_id:
                            continue

                        model_id = _normalize_model_id(raw_id)
                        if not model_id:
                            continue

                        # Skip obvious non-chat models.
                        if "embedding" in model_id.lower():
                            continue
                        output_modalities = row.get("supported_output_modalities")
                        if isinstance(output_modalities, list) and output_modalities:
                            if "text" not in [
                                str(m).lower() for m in output_modalities
                            ]:
                                continue

                        model_name = str(row.get("name") or model_id)
                        output_limit = _extract_output_limit_from_model_row(row)
                        context_limit = _extract_context_limit_from_model_row(row)
                        if not isinstance(output_limit, int):
                            output_limit = await get_output_limit(model_id)
                        else:
                            update_model_output_limit(model_id, output_limit)
                        if isinstance(context_limit, int):
                            update_model_limit(model_id, context_limit)
                        else:
                            context_limit = await get_context_limit(model_id)

                        existing = models_by_id.get(model_id)
                        created = (
                            row.get("created")
                            if isinstance(row.get("created"), int)
                            else None
                        )
                        (
                            capabilities,
                            supported_endpoints,
                            reasoning_supported,
                            thinking_budget_supported,
                            effort_levels,
                        ) = _extract_provider_capability_metadata(row)
                        candidate = LLMModel(
                            id=model_id,
                            name=model_name,
                            created=created,
                            max_output_tokens=output_limit,
                            context_limit=context_limit,
                            capabilities=capabilities or None,
                            supported_endpoints=supported_endpoints or None,
                            reasoning_supported=reasoning_supported,
                            thinking_budget_supported=thinking_budget_supported,
                            effort_levels=effort_levels or None,
                        )

                        # Cache supported API endpoints so the LLM builder
                        # knows when to use the Responses API.
                        if supported_endpoints:
                            register_model_supported_endpoints(
                                model_id, supported_endpoints
                            )

                        # Cache reasoning capability hints from Copilot model metadata.
                        if reasoning_supported or thinking_budget_supported:
                            register_model_reasoning_capabilities(
                                model_id,
                                reasoning_supported=bool(reasoning_supported),
                                thinking_budget_supported=bool(
                                    thinking_budget_supported
                                ),
                            )

                        # Prefer entries that include richer token limit metadata.
                        if existing is None:
                            models_by_id[model_id] = candidate
                        elif (
                            existing.max_output_tokens is None
                            and candidate.max_output_tokens is not None
                        ):
                            models_by_id[model_id] = candidate
            except Exception as e:
                last_error = str(e)

        if models_by_id and successful_sources > 0:
            models = _group_models(list(models_by_id.values()), "github_copilot")
            models.sort(
                key=lambda m: (
                    m.group or "",
                    m.is_latest,
                    m.created or 0,
                    m.name.lower(),
                ),
                reverse=True,
            )
            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} model(s) from {successful_sources} source(s).",
                models=models,
                default_model=models[0].id if models else None,
            )

        return LLMModelsResponse(
            success=False,
            message=f"Failed to fetch GitHub Copilot models: {last_error or 'unknown error'}",
        )

    return await _get_or_fetch_model_discovery(
        cache_key,
        _fetch_uncached,
        force_refresh=force_refresh,
    )


async def _fetch_github_models_catalog(github_token: str) -> LLMModelsResponse:
    """Fetch available models from GitHub Models catalog."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                GITHUB_MODELS_CATALOG_URL,
                headers={
                    "Authorization": f"Bearer {github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "ragtime",
                },
            )

        if response.status_code in (401, 403):
            return LLMModelsResponse(
                success=False,
                message="GitHub Models authorization failed. Reconnect GitHub auth and ensure model access is enabled.",
            )

        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else []

        models: list[LLMModel] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            model_id = str(row.get("id", "")).strip()
            model_id = re.sub(r"/+", "/", model_id).lstrip("/")
            if not model_id:
                continue
            if "embedding" in model_id.lower():
                continue

            output_modalities = row.get("supported_output_modalities")
            if isinstance(output_modalities, list) and output_modalities:
                if "text" not in [str(m).lower() for m in output_modalities]:
                    continue

            (
                capabilities,
                supported_endpoints,
                reasoning_supported,
                thinking_budget_supported,
                effort_levels,
            ) = _extract_provider_capability_metadata(row)

            if supported_endpoints:
                register_model_supported_endpoints(model_id, supported_endpoints)
            if reasoning_supported or thinking_budget_supported:
                register_model_reasoning_capabilities(
                    model_id,
                    reasoning_supported=bool(reasoning_supported),
                    thinking_budget_supported=bool(thinking_budget_supported),
                )

            model_name = str(row.get("name") or model_id)
            output_limit = _extract_output_limit_from_model_row(row)
            context_limit = _extract_context_limit_from_model_row(row)
            if not isinstance(output_limit, int):
                short_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
                output_limit = await get_output_limit(short_id)
            else:
                short_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
                update_model_output_limit(short_id, output_limit)
            short_id = model_id.split("/", 1)[1] if "/" in model_id else model_id
            if isinstance(context_limit, int):
                update_model_limit(short_id, context_limit)
            else:
                context_limit = await get_context_limit(short_id)

            models.append(
                LLMModel(
                    id=model_id,
                    name=model_name,
                    created=(
                        row.get("created")
                        if isinstance(row.get("created"), int)
                        else None
                    ),
                    max_output_tokens=output_limit,
                    context_limit=context_limit,
                    capabilities=capabilities or None,
                    supported_endpoints=supported_endpoints or None,
                    reasoning_supported=reasoning_supported,
                    thinking_budget_supported=thinking_budget_supported,
                    effort_levels=effort_levels or None,
                )
            )

        if not models:
            return LLMModelsResponse(
                success=False,
                message="GitHub Models catalog returned no chat-capable models for this account.",
            )

        models = _group_models(models, "github_models")
        models.sort(
            key=lambda m: (m.group or "", m.is_latest, m.created or 0, m.name.lower()),
            reverse=True,
        )
        return LLMModelsResponse(
            success=True,
            message=f"Found {len(models)} model(s) from GitHub Models catalog.",
            models=models,
            default_model=models[0].id if models else None,
        )
    except httpx.HTTPStatusError as e:
        return LLMModelsResponse(
            success=False,
            message=f"GitHub Models API error: {e.response.status_code}",
        )
    except Exception as e:
        return LLMModelsResponse(
            success=False,
            message=f"Failed to fetch GitHub Models catalog: {str(e)}",
        )


# =============================================================================
# Conversation/Chat Endpoints
# =============================================================================


@router.get(
    "/chat/available-models", response_model=AvailableModelsResponse, tags=["Chat"]
)
async def get_available_chat_models():
    """
    Get all available models from configured LLM providers.

    Returns models from configured providers (OpenAI, Anthropic, Ollama, GitHub Copilot).
    Provider fetches run in parallel to avoid blocking the event loop when one provider is slow.
    """
    # Kick off Copilot refresh in the background to avoid blocking model
    # discovery when another request is already refreshing.
    await ensure_copilot_token_fresh(mode="background")

    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse(
            copilot_refresh_in_progress=is_copilot_token_refresh_in_progress(),
            models_loading=await _is_model_discovery_loading(),
        )

    all_models: List[AvailableModel] = []
    default_model = None
    provider_states: dict[str, ProviderModelState] = {
        "openai": ProviderModelState(provider="openai"),
        "anthropic": ProviderModelState(provider="anthropic"),
        "ollama": ProviderModelState(provider="ollama"),
        "github_copilot": ProviderModelState(provider="github_copilot"),
    }

    # --- Build parallel fetch tasks for each configured provider ---

    async def _fetch_openai_task() -> tuple[str, LLMModelsResponse]:
        try:
            return ("openai", await _fetch_openai_models(app_settings.openai_api_key))
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            return (
                "openai",
                LLMModelsResponse(
                    success=False,
                    message=f"Failed to fetch OpenAI models: {str(e)}",
                ),
            )

    async def _fetch_anthropic_task() -> tuple[str, LLMModelsResponse]:
        try:
            return (
                "anthropic",
                await _fetch_anthropic_models(app_settings.anthropic_api_key),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")
            return (
                "anthropic",
                LLMModelsResponse(
                    success=False,
                    message=f"Failed to fetch Anthropic models: {str(e)}",
                ),
            )

    async def _fetch_ollama_task(url: str) -> tuple[str, LLMModelsResponse]:
        try:
            return ("ollama", await _fetch_ollama_llm_models(url))
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return (
                "ollama",
                LLMModelsResponse(
                    success=False,
                    message=f"Failed to fetch Ollama models: {str(e)}",
                ),
            )

    async def _fetch_github_pat_task(
        token: str, provider_name: str
    ) -> tuple[str, LLMModelsResponse]:
        try:
            result = await _fetch_github_models_catalog(token)
            return ("github_copilot", result)
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub models: {e}")
            return (
                "github_copilot",
                LLMModelsResponse(
                    success=False,
                    message=f"Failed to fetch GitHub models: {str(e)}",
                ),
            )

    async def _fetch_github_oauth_task(
        token: str, base_url: str
    ) -> tuple[str, LLMModelsResponse]:
        try:
            result = await _fetch_github_copilot_models(token, base_url)
            return ("github_copilot", result)
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub models: {e}")
            return (
                "github_copilot",
                LLMModelsResponse(
                    success=False,
                    message=f"Failed to fetch GitHub models: {str(e)}",
                ),
            )

    tasks: list[asyncio.Task] = []

    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        provider_states["openai"].configured = True
        provider_states["openai"].connected = True
        tasks.append(asyncio.create_task(_fetch_openai_task()))

    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        provider_states["anthropic"].configured = True
        provider_states["anthropic"].connected = True
        tasks.append(asyncio.create_task(_fetch_anthropic_task()))

    ollama_url = getattr(app_settings, "llm_ollama_base_url", "") or ""
    if not ollama_url or ollama_url == "http://localhost:11434":
        ollama_url = getattr(app_settings, "ollama_base_url", "") or ""
    if ollama_url:
        provider_states["ollama"].configured = True
        provider_states["ollama"].connected = True
        tasks.append(asyncio.create_task(_fetch_ollama_task(ollama_url)))

    github_models_token = (app_settings.github_models_api_token or "").strip()
    github_copilot_token = (app_settings.github_copilot_access_token or "").strip()
    github_refresh_token = (app_settings.github_copilot_refresh_token or "").strip()
    github_auth_mode, github_auth_error = _resolve_github_auth_mode(app_settings)
    provider_states["github_copilot"].loading = is_copilot_token_refresh_in_progress()
    if github_auth_error:
        provider_states["github_copilot"].configured = bool(
            github_models_token or github_copilot_token or github_refresh_token
        )
        provider_states["github_copilot"].connected = False
        provider_states["github_copilot"].error = github_auth_error
        logger.warning("Skipping GitHub model discovery: %s", github_auth_error)
    elif github_auth_mode == "pat" and github_models_token:
        provider_states["github_copilot"].configured = True
        provider_states["github_copilot"].connected = True
        tasks.append(
            asyncio.create_task(
                _fetch_github_pat_task(github_models_token, "github_models")
            )
        )
    elif github_auth_mode == "oauth" and (github_copilot_token or github_refresh_token):
        provider_states["github_copilot"].configured = True
        provider_states["github_copilot"].connected = bool(
            github_copilot_token and len(github_copilot_token) > 10
        )
        if github_copilot_token and len(github_copilot_token) > 10:
            tasks.append(
                asyncio.create_task(
                    _fetch_github_oauth_task(
                        github_copilot_token,
                        app_settings.github_copilot_base_url
                        or COPILOT_DEFAULT_BASE_URL,
                    )
                )
            )
        elif provider_states["github_copilot"].loading:
            provider_states["github_copilot"].error = (
                "Refreshing GitHub Copilot credentials."
            )
        else:
            provider_states["github_copilot"].error = (
                "GitHub Copilot access token is not available. Reconnect GitHub Copilot."
            )

    # --- Await all provider fetches in parallel ---
    results: list[tuple[str, LLMModelsResponse]] = []
    if tasks:
        results = await asyncio.gather(*tasks)

    # --- Process results in stable order ---
    for provider_key, result in results:
        normalized_provider = _normalize_provider_alias(provider_key)
        state = provider_states.get(normalized_provider)

        if not result.success:
            if state:
                state.available = False
                state.error = state.error or result.message
                if any(code in (result.message or "") for code in ["401", "403"]):
                    state.connected = False
            continue

        if state:
            state.available = bool(result.models)
            state.error = None if state.available else result.message
            if state.configured:
                state.connected = True

        for m in result.models:
            if provider_key == "github_copilot":
                # For PAT-mode catalog models, strip publisher prefix for context lookup
                context_model_id = (
                    m.id.split("/", 1)[1]
                    if github_auth_mode == "pat" and "/" in m.id
                    else m.id
                )
            else:
                context_model_id = m.id

            resolved_supported_endpoints = m.supported_endpoints
            if not resolved_supported_endpoints:
                if await requires_responses_api(context_model_id):
                    resolved_supported_endpoints = ["/responses"]
                elif await supports_responses_api(context_model_id):
                    resolved_supported_endpoints = [
                        "/chat/completions",
                        "/responses",
                    ]

            all_models.append(
                # Prefer provider-reported limits when available.
                # For Ollama, the model detail endpoint remains the source of truth.
                # For other providers, fall back to shared model-limits lookup.
                AvailableModel(
                    id=m.id,
                    name=m.name,
                    provider=provider_key,
                    context_limit=(
                        m.context_limit or 8192
                        if provider_key == "ollama"
                        else (
                            m.context_limit
                            if isinstance(m.context_limit, int) and m.context_limit > 0
                            else await get_context_limit(context_model_id)
                        )
                    ),
                    max_output_tokens=m.max_output_tokens,
                    created=m.created,
                    capabilities=m.capabilities,
                    supported_endpoints=resolved_supported_endpoints,
                    reasoning_supported=m.reasoning_supported,
                    thinking_budget_supported=m.thinking_budget_supported,
                    effort_levels=m.effort_levels,
                )
            )
        if not default_model and result.default_model:
            default_model = result.default_model

    # Use current configured LLM model as automatic default when available.
    current_model = (app_settings.llm_model or "").strip() or None
    current_model_match = _find_available_model_for_identifier(
        all_models, current_model
    )
    if current_model_match:
        default_model = current_model_match.id

    # Filter by allowed models if specified.
    # Supports legacy model IDs and provider-scoped keys: provider::model_id.
    allowed_models = [
        str(value).strip()
        for value in (app_settings.allowed_chat_models or [])
        if str(value).strip()
    ]
    if allowed_models:
        scoped_provider_by_model: dict[str, str] = {}
        legacy_model_ids: set[str] = set()
        for value in allowed_models:
            if "::" in value:
                provider, _, model_id = value.partition("::")
                provider = provider.strip().lower()
                model_id = model_id.strip()
                if (
                    provider
                    in {
                        "openai",
                        "anthropic",
                        "ollama",
                        "github_copilot",
                        "github_models",
                    }
                    and model_id
                ):
                    scoped_provider_by_model[model_id] = provider
                    continue
            legacy_model_ids.add(value)

        def _lookup_filter_provider(model_id: str) -> str:
            """Look up the filter provider, handling publisher-prefixed IDs.

            Catalog models use publisher-prefixed IDs (e.g. 'openai/gpt-4o')
            while directory/OAuth models use bare slugs ('gpt-4o').  The filter
            may have been saved in either format, so try both.
            """
            provider = scoped_provider_by_model.get(model_id, "")
            if not provider and "/" in model_id:
                provider = scoped_provider_by_model.get(model_id.split("/", 1)[1], "")
            return provider

        all_models = [
            model
            for model in all_models
            if (
                _providers_equivalent(_lookup_filter_provider(model.id), model.provider)
                or (
                    model.id in legacy_model_ids
                    and not _lookup_filter_provider(model.id)
                )
            )
        ]

    # Assign groups to models for UI organization
    all_models = _assign_model_groups(all_models)

    automatic_default_match = _find_available_model_for_identifier(
        all_models, default_model
    )
    if automatic_default_match is None and all_models:
        automatic_default_match = all_models[0]

    manual_default_identifier = (
        str(getattr(app_settings, "default_chat_model", "") or "").strip() or None
    )
    manual_default_match = _find_available_model_for_identifier(
        all_models, manual_default_identifier
    )

    effective_default_match = manual_default_match or automatic_default_match

    models_loading = await _is_model_discovery_loading()
    copilot_refresh_in_progress = is_copilot_token_refresh_in_progress()
    if copilot_refresh_in_progress and provider_states["github_copilot"].configured:
        provider_states["github_copilot"].loading = True
        if not provider_states["github_copilot"].connected:
            provider_states["github_copilot"].error = (
                provider_states["github_copilot"].error
                or "Refreshing GitHub Copilot credentials."
            )

    # When a configured provider never fetched (e.g. temporary auth mismatch),
    # keep availability false and attach a stable generic message.
    for state in provider_states.values():
        if (
            state.configured
            and not state.available
            and not state.loading
            and not state.error
        ):
            state.error = "No models available from provider."

    return AvailableModelsResponse(
        models=all_models,
        default_model=effective_default_match.id if effective_default_match else None,
        automatic_default_model=(
            _build_scoped_model_identifier(automatic_default_match)
            if automatic_default_match
            else None
        ),
        current_model=current_model,
        models_loading=models_loading,
        copilot_refresh_in_progress=copilot_refresh_in_progress,
        provider_states=list(provider_states.values()),
    )


@router.get("/chat/all-models", response_model=AvailableModelsResponse, tags=["Chat"])
async def get_all_chat_models(_user: User = Depends(require_admin)):
    """
    Get ALL available models from configured LLM providers (unfiltered).

    Used by the settings UI to show all models for selection.
    Provider fetches run in parallel.
    """
    # Kick off Copilot refresh in the background to avoid blocking model
    # discovery when another request is already refreshing.
    await ensure_copilot_token_fresh(mode="background")

    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse()

    all_models: List[AvailableModel] = []

    # --- Build parallel fetch tasks for each configured provider ---

    async def _fetch_openai_task() -> tuple[str, LLMModelsResponse | None]:
        try:
            return ("openai", await _fetch_openai_models(app_settings.openai_api_key))
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
            return ("openai", None)

    async def _fetch_anthropic_task() -> tuple[str, LLMModelsResponse | None]:
        try:
            return (
                "anthropic",
                await _fetch_anthropic_models(app_settings.anthropic_api_key),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")
            return ("anthropic", None)

    async def _fetch_ollama_task(url: str) -> tuple[str, LLMModelsResponse | None]:
        try:
            return ("ollama", await _fetch_ollama_llm_models(url))
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")
            return ("ollama", None)

    async def _fetch_github_pat_task(
        token: str, provider_name: str
    ) -> tuple[str, LLMModelsResponse | None]:
        try:
            result = await _fetch_github_models_catalog(token)
            if result.success and provider_name == "github_copilot":
                directory_result = await _fetch_copilot_directory_models(
                    include_anthropic=True,
                    include_google=True,
                )
                result = _merge_llm_model_results(result, directory_result)
            return ("github_copilot", result)
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub models: {e}")
            return ("github_copilot", None)

    async def _fetch_github_oauth_task(
        token: str, base_url: str
    ) -> tuple[str, LLMModelsResponse | None]:
        try:
            result = await _fetch_github_copilot_models(token, base_url)
            if result.success:
                directory_result = await _fetch_copilot_directory_models(
                    include_anthropic=True,
                    include_google=True,
                )
                result = _merge_llm_model_results(result, directory_result)
            return ("github_copilot", result)
        except Exception as e:
            logger.warning(f"Failed to fetch GitHub models: {e}")
            return ("github_copilot", None)

    tasks: list[asyncio.Task] = []

    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        tasks.append(asyncio.create_task(_fetch_openai_task()))

    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        tasks.append(asyncio.create_task(_fetch_anthropic_task()))

    ollama_url = getattr(app_settings, "llm_ollama_base_url", "") or ""
    if not ollama_url or ollama_url == "http://localhost:11434":
        ollama_url = getattr(app_settings, "ollama_base_url", "") or ""
    if ollama_url:
        tasks.append(asyncio.create_task(_fetch_ollama_task(ollama_url)))

    github_models_token = (app_settings.github_models_api_token or "").strip()
    github_copilot_token = (app_settings.github_copilot_access_token or "").strip()
    github_auth_mode, github_auth_error = _resolve_github_auth_mode(app_settings)
    if github_auth_error:
        logger.warning("Skipping GitHub model discovery: %s", github_auth_error)
    elif github_auth_mode == "pat" and github_models_token:
        tasks.append(
            asyncio.create_task(
                _fetch_github_pat_task(github_models_token, "github_models")
            )
        )
    elif (
        github_auth_mode == "oauth"
        and github_copilot_token
        and len(github_copilot_token) > 10
    ):
        tasks.append(
            asyncio.create_task(
                _fetch_github_oauth_task(
                    github_copilot_token,
                    app_settings.github_copilot_base_url or COPILOT_DEFAULT_BASE_URL,
                )
            )
        )

    # --- Await all provider fetches in parallel ---
    results: list[tuple[str, LLMModelsResponse | None]] = []
    if tasks:
        results = await asyncio.gather(*tasks)

    # --- Process results ---
    for provider_key, result in results:
        if not result or not result.success:
            continue

        for m in result.models:
            if provider_key == "github_copilot":
                context_model_id = (
                    m.id.split("/", 1)[1]
                    if github_auth_mode == "pat" and "/" in m.id
                    else m.id
                )
            else:
                context_model_id = m.id

            resolved_supported_endpoints = m.supported_endpoints
            if not resolved_supported_endpoints:
                if await requires_responses_api(context_model_id):
                    resolved_supported_endpoints = ["/responses"]
                elif await supports_responses_api(context_model_id):
                    resolved_supported_endpoints = [
                        "/chat/completions",
                        "/responses",
                    ]

            all_models.append(
                AvailableModel(
                    id=m.id,
                    name=m.name,
                    provider=provider_key,
                    context_limit=(
                        m.context_limit or 8192
                        if provider_key == "ollama"
                        else (
                            m.context_limit
                            if isinstance(m.context_limit, int) and m.context_limit > 0
                            else await get_context_limit(context_model_id)
                        )
                    ),
                    max_output_tokens=m.max_output_tokens,
                    created=m.created,
                    capabilities=m.capabilities,
                    supported_endpoints=resolved_supported_endpoints,
                    reasoning_supported=m.reasoning_supported,
                    thinking_budget_supported=m.thinking_budget_supported,
                    effort_levels=m.effort_levels,
                )
            )

    # Get currently allowed models from settings
    allowed_models = app_settings.allowed_chat_models or []

    # Assign groups to models for UI organization
    all_models = _assign_model_groups(all_models)

    return AvailableModelsResponse(
        models=all_models,
        default_model=app_settings.llm_model,
        current_model=app_settings.llm_model,
        allowed_models=allowed_models,
        allowed_openapi_models=app_settings.allowed_openapi_models or [],
    )


def _to_conversation_response(conv: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        user_id=conv.user_id,
        workspace_id=conv.workspace_id,
        username=conv.username,
        display_name=conv.display_name,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        active_task_id=conv.active_task_id,
        tool_output_mode=conv.tool_output_mode,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


class WorkspaceConversationStateResponse(BaseModel):
    conversations: List[ConversationResponse] = Field(
        default_factory=list,
        description="Conversation summaries for a workspace.",
    )
    interrupted_conversation_ids: List[str] = Field(
        default_factory=list,
        description="Conversation IDs that currently have interrupted tasks.",
    )


class ConversationTaskStateResponse(BaseModel):
    active_task: Optional[ChatTaskResponse] = Field(
        default=None,
        description="Current pending or running task for the conversation, if any.",
    )
    interrupted_task: Optional[ChatTaskResponse] = Field(
        default=None,
        description="Most recent interrupted task for the conversation when there is no active task.",
    )


class WorkspaceConversationStateSummaryRequest(BaseModel):
    workspace_ids: List[str] = Field(
        default_factory=list,
        description="Workspace IDs to summarize.",
    )


class WorkspaceConversationStateSummaryItem(BaseModel):
    workspace_id: str = Field(description="Workspace ID.")
    has_live_task: bool = Field(
        default=False,
        description="True when any conversation has a live task and is not interrupted.",
    )
    has_interrupted_task: bool = Field(
        default=False,
        description="True when any conversation has an interrupted task.",
    )


@router.get(
    "/conversations/workspace/{workspace_id}/chat-state",
    response_model=WorkspaceChatStateResponse,
)
async def get_workspace_chat_state(
    workspace_id: str,
    selected_conversation_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Return workspace conversations plus selected conversation task state."""
    await _assert_workspace_access(workspace_id, user, "viewer")
    return await build_workspace_chat_state(
        workspace_id=workspace_id,
        user_id=user.id,
        is_admin=(user.role == "admin"),
        selected_conversation_id=selected_conversation_id,
    )


def _to_chat_task_response(task: Any) -> ChatTaskResponse:
    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=task.streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


async def _assert_workspace_access(
    workspace_id: Optional[str], user: User, required_role: str
) -> None:
    if not workspace_id:
        return
    await userspace_service.enforce_workspace_role(
        workspace_id, user.id, required_role, is_admin=user.role == "admin"
    )


async def _resolve_selected_tool_ids_for_request(
    conversation: Conversation,
    user: User,
    workspace_id: Optional[str],
    required_role: str,
) -> tuple[Optional[str], set[str], Optional[dict[str, str]]]:
    requested_workspace_id = (workspace_id or "").strip() or None
    conversation_workspace_id = (conversation.workspace_id or "").strip() or None

    if (
        requested_workspace_id
        and conversation_workspace_id
        and requested_workspace_id != conversation_workspace_id
    ):
        raise HTTPException(
            status_code=400,
            detail="workspace_id does not match the conversation workspace",
        )

    effective_workspace_id = requested_workspace_id or conversation_workspace_id

    db = Prisma()
    await db.connect()
    try:
        conversation_tool_selections = await db.conversationtoolselection.find_many(
            where={"conversationId": conversation.id}
        )
        conversation_group_selections = (
            await db.conversationtoolgroupselection.find_many(
                where={"conversationId": conversation.id}
            )
        )
        conversation_selected_tool_ids = [
            s.toolConfigId for s in conversation_tool_selections
        ]
        conversation_selected_group_ids = [
            s.toolGroupId for s in conversation_group_selections
        ]
    finally:
        await db.disconnect()

    selected_tool_ids = set(conversation_selected_tool_ids)
    if conversation_selected_group_ids:
        group_tool_ids = await repository.get_tool_ids_for_groups(
            conversation_selected_group_ids
        )
        selected_tool_ids.update(group_tool_ids)

    workspace_context = None
    if effective_workspace_id:
        workspace = await userspace_service.enforce_workspace_role(
            effective_workspace_id,
            user.id,
            required_role,
        )
        # Workspace scope enforces strict selected-tool bounds.
        selected_tool_ids = set(workspace.selected_tool_ids)
        if workspace.selected_tool_group_ids:
            group_tool_ids = await repository.get_tool_ids_for_groups(
                workspace.selected_tool_group_ids
            )
            selected_tool_ids.update(group_tool_ids)
        workspace_context = {
            "workspace_id": effective_workspace_id,
            "user_id": user.id,
        }

    return effective_workspace_id, selected_tool_ids, workspace_context


async def _resolve_workspace_runtime_scope(
    conversation: Conversation,
    user: User,
    workspace_id: Optional[str],
    required_role: str,
) -> tuple[Optional[str], Optional[set[str]], Optional[dict[str, str]]]:
    effective_workspace_id, selected_tool_ids, workspace_context = (
        await _resolve_selected_tool_ids_for_request(
            conversation,
            user,
            workspace_id,
            required_role,
        )
    )

    # An empty selection means no system tools are available for this request.
    if not selected_tool_ids:
        blocked_tool_names = rag.get_blocked_config_tool_names([])
        return effective_workspace_id, blocked_tool_names, workspace_context

    # Get blocked tool names (tools NOT in selected set)
    blocked_tool_names = rag.get_blocked_config_tool_names(list(selected_tool_ids))

    return effective_workspace_id, blocked_tool_names, workspace_context


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """List chat conversations for the current user."""
    await _assert_workspace_access(workspace_id, user, "viewer")
    # Admins can see all, regular users only see their own
    is_admin = user.role == "admin"
    convs = await repository.list_conversations(
        user_id=user.id,
        include_all=is_admin,
        workspace_id=workspace_id,
    )
    return [_to_conversation_response(c) for c in convs]


@router.get(
    "/conversations/workspace/{workspace_id}/conversation-state",
    response_model=WorkspaceConversationStateResponse,
)
async def get_workspace_conversation_state(
    workspace_id: str,
    user: User = Depends(get_current_user),
):
    """Return combined workspace conversation summaries and interrupted-task IDs."""
    await _assert_workspace_access(workspace_id, user, "viewer")
    is_admin = user.role == "admin"
    convs = await repository.list_conversations(
        user_id=user.id,
        include_all=is_admin,
        workspace_id=workspace_id,
    )
    interrupted_conversation_ids = (
        await repository.get_interrupted_conversation_ids_for_workspace(workspace_id)
    )
    return WorkspaceConversationStateResponse(
        conversations=[_to_conversation_response(c) for c in convs],
        interrupted_conversation_ids=interrupted_conversation_ids,
    )


@router.post(
    "/conversations/workspaces/state-summary",
    response_model=List[WorkspaceConversationStateSummaryItem],
)
async def get_workspaces_conversation_state_summary(
    request: WorkspaceConversationStateSummaryRequest,
    user: User = Depends(get_current_user),
):
    """Return live/interrupted summary for multiple workspaces in one request."""
    workspace_ids = [
        wid.strip() for wid in request.workspace_ids if wid and wid.strip()
    ]
    if not workspace_ids:
        return []

    # Preserve order while deduplicating
    deduped_workspace_ids = list(dict.fromkeys(workspace_ids))
    is_admin = user.role == "admin"
    results: list[WorkspaceConversationStateSummaryItem] = []

    for workspace_id in deduped_workspace_ids:
        await _assert_workspace_access(workspace_id, user, "viewer")
        convs, interrupted_conversation_ids = await asyncio.gather(
            repository.list_conversations(
                user_id=user.id,
                include_all=is_admin,
                workspace_id=workspace_id,
            ),
            repository.get_interrupted_conversation_ids_for_workspace(workspace_id),
        )
        has_live_task = any(bool(conv.active_task_id) for conv in convs)
        has_interrupted_task = len(interrupted_conversation_ids) > 0
        results.append(
            WorkspaceConversationStateSummaryItem(
                workspace_id=workspace_id,
                has_live_task=has_live_task,
                has_interrupted_task=has_interrupted_task,
            )
        )

    return results


def _resolve_default_conversation_model(app_settings: Optional[AppSettings]) -> str:
    """Resolve default model for new conversations.

    Preference order:
    1. Manual default_chat_model override (if allowed)
    2. llm_model (if allowed)
    3. First allowed_chat_models entry (if any)
    4. Hard fallback
    """
    if app_settings is None:
        return "gpt-4-turbo"

    manual_default = str(getattr(app_settings, "default_chat_model", "") or "").strip()
    configured_model = str(app_settings.llm_model or "").strip()
    allowed_models = [
        str(value).strip()
        for value in (app_settings.allowed_chat_models or [])
        if str(value).strip()
    ]

    if allowed_models:
        if manual_default and _identifier_in_allowed_models(
            manual_default, allowed_models
        ):
            return manual_default
        if configured_model and _identifier_in_allowed_models(
            configured_model, allowed_models
        ):
            return configured_model
        return allowed_models[0]

    if manual_default:
        return manual_default
    if configured_model:
        return configured_model
    return "gpt-4-turbo"


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    user: User = Depends(get_current_user),
):
    """Create a new chat conversation for the current user."""
    # Get default model from settings (manual override or sensible automatic fallback).
    app_settings = await repository.get_settings()
    default_model = _resolve_default_conversation_model(app_settings)

    title = request.title if request and request.title else "Untitled Chat"
    model = request.model if request and request.model else default_model

    workspace_id = request.workspace_id if request else None

    await _assert_workspace_access(workspace_id, user, "editor")

    conv = await repository.create_conversation(
        title=title,
        model=model,
        user_id=user.id,
        workspace_id=workspace_id,
    )
    return _to_conversation_response(conv)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Get a specific conversation. Users can only access their own conversations."""
    await _assert_workspace_access(workspace_id, user, "viewer")
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.get(
    "/conversations/{conversation_id}/provider-debug-prompts",
    response_model=ProviderPromptDebugListResponse,
)
async def list_conversation_provider_debug_prompts(
    conversation_id: str,
    limit: int = 100,
    before: Optional[datetime] = None,
    message_index: int = Query(..., ge=0),
    workspace_id: Optional[str] = None,
    user: User = Depends(require_admin),
):
    """List provider prompt debug records for a conversation (admin + DEBUG_MODE only)."""
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Debug prompt capture is disabled")

    await _assert_workspace_access(workspace_id, user, "viewer")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=True,
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    records = await repository.list_provider_prompt_debug_records_for_conversation(
        conversation_id,
        limit=limit,
        before=before,
        message_index=message_index,
    )

    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        try:
            return json.dumps(content, ensure_ascii=True, default=str)
        except Exception:
            return str(content)

    enriched_records: list[ProviderPromptDebugRecord] = []
    for record in records:
        total_prompt_tokens = 0
        for message in record.rendered_provider_messages:
            if not isinstance(message, dict):
                continue
            text = _content_to_text(message.get("content"))
            if text:
                total_prompt_tokens += count_tokens(text)

        enriched_records.append(
            record.model_copy(update={"prompt_token_count": total_prompt_tokens})
        )

    return ProviderPromptDebugListResponse(records=enriched_records)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Delete a conversation. Users can only delete their own conversations."""
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    success = await repository.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@router.patch(
    "/conversations/{conversation_id}/title", response_model=ConversationResponse
)
async def update_conversation_title(
    conversation_id: str,
    body: dict,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Update a conversation's title. Users can only update their own conversations."""
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    conv = await repository.update_conversation_title(conversation_id, title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.patch(
    "/conversations/{conversation_id}/model", response_model=ConversationResponse
)
async def update_conversation_model(
    conversation_id: str,
    body: dict,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Update a conversation's model. Users can only update their own conversations."""
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    model = body.get("model", "").strip()
    model = model.lstrip("/")
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    provider = (body.get("provider") or "").strip().lower()
    if provider and provider not in {
        "openai",
        "anthropic",
        "ollama",
        "github_copilot",
        "github_models",
    }:
        raise HTTPException(
            status_code=400,
            detail="provider must be one of: openai, anthropic, ollama, github_copilot, github_models",
        )

    stored_model = f"{provider}::{model}" if provider else model

    if provider:
        await _validate_conversation_model_selection(
            provider=provider,
            model_id=model,
            include_directory_models=False,
            force_refresh=(provider == "github_copilot"),
        )

    conv = await repository.update_conversation_model(conversation_id, stored_model)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.patch(
    "/conversations/{conversation_id}/tool-output-mode",
    response_model=ConversationResponse,
)
async def update_conversation_tool_output_mode(
    conversation_id: str,
    body: dict,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Update a conversation's tool output mode. Users can only update their own conversations."""
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    mode = body.get("tool_output_mode", "").strip()
    if mode not in ["default", "show", "hide", "auto"]:
        raise HTTPException(
            status_code=400,
            detail="tool_output_mode must be 'default', 'show', 'hide', or 'auto'",
        )

    conv = await repository.update_conversation_tool_output_mode(conversation_id, mode)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.post(
    "/conversations/{conversation_id}/truncate", response_model=ConversationResponse
)
async def truncate_conversation(
    conversation_id: str,
    keep_count: int,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Truncate conversation messages to keep only the first N messages.
    Used when editing/resending a message to remove subsequent messages.
    """
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = await repository.truncate_messages(conversation_id, keep_count)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and get a response.
    Non-streaming version.
    """
    await _assert_workspace_access(workspace_id, user, "editor")

    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    _, blocked_tool_names, workspace_context = await _resolve_workspace_runtime_scope(
        conv,
        user,
        workspace_id,
        "editor",
    )

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

    await _validate_conversation_model_before_send(conv.model)

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Add user message
    conv = await repository.add_message(conversation_id, "user", user_message)
    schedule_title_generation(conversation_id, user_message)
    if conv is None:
        raise HTTPException(status_code=500, detail="Failed to add user message")

    # Build chat history for RAG
    chat_history: list[BaseMessage] = []
    for msg_idx, msg in enumerate(conv.messages[:-1]):  # Exclude the current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            if msg.events:
                chat_history.extend(
                    rebuild_tool_messages_from_events(msg.events, msg_idx)
                )
            else:
                chat_history.append(AIMessage(content=msg.content))

    # Generate response
    try:
        answer = await rag.process_query(
            user_message,
            chat_history,
            blocked_tool_names=blocked_tool_names,
            workspace_context=workspace_context,
            conversation_model=conv.model,
            conversation_id=conversation_id,
            user_id=user.id,
            message_index=len(conv.messages),
        )
    except Exception as e:
        logger.exception("Error processing message")
        answer = f"Error: {str(e)}"

    # Add assistant response
    conv = await repository.add_message(conversation_id, "assistant", answer)
    if conv is None:
        raise HTTPException(status_code=500, detail="Failed to add assistant message")

    return {
        "message": ChatMessage(
            role="assistant",
            content=answer,
            timestamp=conv.messages[-1].timestamp,
        ),
        "conversation": _to_conversation_response(conv),
    }


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    request: SendMessageRequest,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and stream the response.
    Returns SSE stream of tokens.
    """
    await _assert_workspace_access(workspace_id, user, "editor")

    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    _, blocked_tool_names, workspace_context = await _resolve_workspace_runtime_scope(
        conv,
        user,
        workspace_id,
        "editor",
    )

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

    await _validate_conversation_model_before_send(conv.model)

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Add user message
    await repository.add_message(conversation_id, "user", user_message)
    schedule_title_generation(conversation_id, user_message)

    # Refresh conversation to get updated messages
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=500, detail="Failed to store message")

    # Build chat history for RAG
    chat_history: list[BaseMessage] = []
    for msg_idx, msg in enumerate(conv.messages[:-1]):  # Exclude current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            if msg.events:
                chat_history.extend(
                    rebuild_tool_messages_from_events(msg.events, msg_idx)
                )
            else:
                chat_history.append(AIMessage(content=msg.content))

    async def stream_response():
        """Generate streaming response tokens."""
        chunk_id = f"chatcmpl-{int(time.time())}"
        full_response = ""
        chronological_events: list[dict[str, Any]] = (
            []
        )  # Collect events in order (content and tools)
        current_tool_call: dict[str, Any] | None = (
            None  # Track current tool call being built
        )

        try:
            # Use UI agent (with chart tool and enhanced prompt)
            async for event in rag.process_query_stream(
                user_message,
                chat_history,
                is_ui=True,
                blocked_tool_names=blocked_tool_names,
                workspace_context=workspace_context,
                conversation_model=conv.model,
                conversation_id=conversation_id,
                user_id=user.id,
                message_index=len(conv.messages),
            ):
                # Handle structured tool events
                if isinstance(event, dict):
                    event_type = event.get("type")
                    if event_type == "tool_start":
                        # Start tracking a new tool call
                        current_tool_call = {
                            "type": "tool",
                            "tool": event.get("tool"),
                            "input": event.get("input"),
                        }
                        tool_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": conv.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_call": {
                                            "type": "start",
                                            "tool": event.get("tool"),
                                            "input": event.get("input"),
                                        }
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
                    elif event_type == "tool_end":
                        # Complete the current tool call and save it
                        if current_tool_call is not None:
                            current_tool_call["output"] = event.get("output")
                            chronological_events.append(current_tool_call)
                            current_tool_call = None
                        tool_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": conv.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_call": {
                                            "type": "end",
                                            "tool": event.get("tool"),
                                            "output": event.get("output"),
                                        }
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
                    elif event_type == "reasoning":
                        # Reasoning/thinking content from LLM
                        reasoning_text = event.get("content", "")
                        if reasoning_text:
                            # Accumulate in chronological events
                            if (
                                chronological_events
                                and chronological_events[-1].get("type") == "reasoning"
                            ):
                                chronological_events[-1]["content"] += reasoning_text
                            else:
                                chronological_events.append(
                                    {"type": "reasoning", "content": reasoning_text}
                                )

                            reasoning_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": conv.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"reasoning": reasoning_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(reasoning_chunk)}\n\n"
                else:
                    # Handle regular text tokens
                    token = event
                    full_response += token
                    # Add content events when we get text (batch them for efficiency)
                    if (
                        chronological_events
                        and chronological_events[-1].get("type") == "content"
                    ):
                        # Append to last content event
                        chronological_events[-1]["content"] += token
                    else:
                        # Start new content event
                        chronological_events.append(
                            {"type": "content", "content": token}
                        )

                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": conv.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Persist the full response using chronological events.
            await repository.add_message(
                conversation_id,
                "assistant",
                full_response,
                events=chronological_events if chronological_events else None,
            )

            # Final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": conv.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error in streaming response")

            raw_error = str(e)
            friendly_error = (
                "An error occurred while generating the response. Please try again."
            )

            max_iters = None
            if rag and getattr(rag, "agent_executor", None):
                max_iters = getattr(rag.agent_executor, "max_iterations", None)

            if (
                "iteration" in raw_error.lower()
                or "max iterations" in raw_error.lower()
            ):
                limit_text = f" ({max_iters})" if max_iters else ""
                friendly_error = f"Stopped after reaching the max_iterations limit{limit_text}. Please narrow the request or retry."

            # Include any in-progress tool call that didn't complete
            if current_tool_call is not None:
                current_tool_call["output"] = "(interrupted)"
                chronological_events.append(current_tool_call)

            # Persist whatever we have so far using chronological events only.
            combined_response = full_response.strip()
            if friendly_error:
                combined_response = (
                    f"{combined_response}\n\n{friendly_error}"
                    if combined_response
                    else friendly_error
                )

            try:
                await repository.add_message(
                    conversation_id,
                    "assistant",
                    combined_response,
                    events=chronological_events if chronological_events else None,
                )
            except Exception:
                logger.exception("Failed to persist assistant message after error")

            error_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": conv.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n{friendly_error}"},
                        "finish_reason": "error",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# =============================================================================
# Retry Visualization Endpoint
# =============================================================================


class RetryTerminalToolRequest(BaseModel):
    tool_config_id: str = Field(min_length=1, description="Tool config ID to replay")
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Captured tool input to replay through the configured tool",
    )


class RetryTerminalToolResponse(BaseModel):
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


@router.post("/conversations/{conversation_id}/retry-visualization")
async def retry_visualization(
    conversation_id: str,
    request: RetryVisualizationRequest,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Retry a failed visualization tool call.

    Directly invokes the create_datatable or create_chart tool with the provided
    source data. No LLM call is needed since we have structured data.

    For datatables, source_data should be: {"columns": [...], "rows": [...]}
    For charts, source_data should be: {"labels": [...], "datasets": [...], "chart_type": "..."}
    """
    await _assert_workspace_access(workspace_id, user, "editor")
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        if request.tool_type == "datatable":
            # Extract data from source_data
            columns = request.source_data.get("columns", [])
            rows = request.source_data.get("rows", [])

            if not columns or not rows:
                return RetryVisualizationResponse(
                    success=False,
                    error="Invalid source_data: must contain 'columns' and 'rows'",
                )

            title = request.title or "Data"

            # Directly call the datatable tool
            output = await create_datatable(
                title=title,
                columns=columns,
                data=rows,
                description=f"Table with {len(rows)} rows",
            )

            return RetryVisualizationResponse(success=True, output=output)

        elif request.tool_type == "chart":
            # Extract chart data
            labels = request.source_data.get("labels", [])
            datasets = request.source_data.get("datasets", [])
            chart_type = request.source_data.get("chart_type", "bar")

            if not labels or not datasets:
                return RetryVisualizationResponse(
                    success=False,
                    error="Invalid source_data: must contain 'labels' and 'datasets'",
                )

            title = request.title or "Chart"

            # Directly call the chart tool
            output = await create_chart(
                chart_type=chart_type,
                title=title,
                labels=labels,
                datasets=datasets,
                description=f"Chart with {len(labels)} data points",
            )

            return RetryVisualizationResponse(success=True, output=output)
        else:
            return RetryVisualizationResponse(
                success=False, error=f"Unknown tool type: {request.tool_type}"
            )

    except Exception as e:
        logger.exception(f"Error retrying visualization: {e}")
        return RetryVisualizationResponse(success=False, error=str(e))


@router.post(
    "/conversations/{conversation_id}/retry-terminal-tool",
    response_model=RetryTerminalToolResponse,
)
async def retry_terminal_tool(
    conversation_id: str,
    request: RetryTerminalToolRequest,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Replay a terminal-classified runtime tool with its captured input."""
    await _assert_workspace_access(workspace_id, user, "editor")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

    conversation = await repository.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    _, selected_tool_ids, _ = await _resolve_selected_tool_ids_for_request(
        conversation,
        user,
        workspace_id,
        "editor",
    )
    if request.tool_config_id not in selected_tool_ids:
        raise HTTPException(status_code=404, detail="Tool not available")

    tool_config = await repository.get_tool_config(request.tool_config_id)
    if not tool_config or not tool_config.enabled:
        raise HTTPException(status_code=404, detail="Tool not found")

    if tool_config.tool_type not in {"ssh_shell", "odoo_shell"}:
        raise HTTPException(status_code=400, detail="Tool is not terminal-rerunnable")

    try:
        tool = await rag.build_primary_runtime_tool_from_config(tool_config.model_dump())
        if tool is None:
            raise HTTPException(status_code=500, detail="Failed to build tool")

        output = await tool.ainvoke(request.input)
        return RetryTerminalToolResponse(success=True, output=str(output))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrying terminal tool: {e}")
        return RetryTerminalToolResponse(success=False, error=str(e))


# =============================================================================
# Background Chat Task Endpoints
# =============================================================================


@router.post(
    "/conversations/{conversation_id}/messages/background",
    response_model=ChatTaskResponse,
)
async def send_message_background(
    conversation_id: str,
    request: SendMessageRequest,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and process it in the background.
    Returns a task object that can be polled for status and results.
    """
    await _assert_workspace_access(workspace_id, user, "editor")

    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    _, blocked_tool_names, workspace_context = await _resolve_workspace_runtime_scope(
        conv,
        user,
        workspace_id,
        "editor",
    )

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

    await _validate_conversation_model_before_send(conv.model)

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Check if there's already an active task
    existing_task = await repository.get_active_task_for_conversation(conversation_id)
    if existing_task:
        # Return the existing task instead of creating a new one
        return _to_chat_task_response(existing_task)

    # Add user message to conversation first
    await repository.add_message(conversation_id, "user", user_message)
    schedule_title_generation(conversation_id, user_message)

    # Start background task
    task_id = await background_task_service.start_task_async(
        conversation_id,
        user_message,
        blocked_tool_names=blocked_tool_names,
        workspace_context=workspace_context,
    )

    # Get the created task
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create background task")

    return _to_chat_task_response(task)


@router.get(
    "/conversations/{conversation_id}/task", response_model=Optional[ChatTaskResponse]
)
async def get_conversation_active_task(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Get the active (pending/running) task for a conversation, if any.
    Returns null if no active task.
    """
    await _assert_workspace_access(workspace_id, user, "viewer")
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    task = await repository.get_active_task_for_conversation(conversation_id)
    if not task:
        return None

    return _to_chat_task_response(task)


@router.get(
    "/conversations/{conversation_id}/interrupted-task",
    response_model=Optional[ChatTaskResponse],
)
async def get_conversation_interrupted_task(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Get the last interrupted task for a conversation, if any.
    Used to show Continue button after server restart.
    Returns null if no interrupted task.
    """
    await _assert_workspace_access(workspace_id, user, "viewer")
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    task = await repository.get_last_interrupted_task_for_conversation(conversation_id)
    if not task:
        return None

    return _to_chat_task_response(task)


@router.get(
    "/conversations/{conversation_id}/task-state",
    response_model=ConversationTaskStateResponse,
)
async def get_conversation_task_state(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Return combined active/interrupted task state for a conversation."""
    await _assert_workspace_access(workspace_id, user, "viewer")
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    active_task = await repository.get_active_task_for_conversation(conversation_id)
    interrupted_task = None
    if not active_task:
        interrupted_task = await repository.get_last_interrupted_task_for_conversation(
            conversation_id
        )

    return ConversationTaskStateResponse(
        active_task=_to_chat_task_response(active_task) if active_task else None,
        interrupted_task=(
            _to_chat_task_response(interrupted_task) if interrupted_task else None
        ),
    )


@router.get(
    "/conversations/workspace/{workspace_id}/interrupted-conversation-ids",
    response_model=list[str],
)
async def get_workspace_interrupted_conversation_ids(
    workspace_id: str,
    user: User = Depends(get_current_user),
):
    """
    Return conversation IDs that have at least one interrupted task in a workspace.
    Used to batch-check interrupted state instead of polling per-conversation.
    """
    await _assert_workspace_access(workspace_id, user, "viewer")
    return await repository.get_interrupted_conversation_ids_for_workspace(workspace_id)


@router.get("/tasks/{task_id}", response_model=ChatTaskResponse)
async def get_chat_task(task_id: str, since_version: int = 0):
    """
    Get a chat task by ID.
    Use this to poll for task status and streaming state.

    Query params:
        since_version: If provided, returns null streaming_state when version hasn't changed.
                      This reduces data transfer for polling clients.
    """
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # If client has current version and task is still running, omit streaming_state
    # to reduce data transfer. Client should use its cached version.
    streaming_state = task.streaming_state
    if since_version > 0 and streaming_state:
        current_version = streaming_state.version
        # Only omit if version matches AND task is still running
        # Always send full state when task completes so client gets final data
        if current_version <= since_version and task.status in (
            ChatTaskStatus.pending,
            ChatTaskStatus.running,
        ):
            streaming_state = None

    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


@router.get("/tasks/{task_id}/stream")
async def stream_chat_task(
    task_id: str,
    since_version: int = 0,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Stream updates for a chat task via SSE.
    """
    # Verify task exists and user has access
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    await _assert_workspace_access(workspace_id, user, "viewer")
    # Check conversation access
    has_access = await repository.check_conversation_access(
        task.conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")

    async def event_generator():
        # First partial update if we are behind
        # If task has state and it is newer than what client has, send valid state immediately
        if task.streaming_state and task.streaming_state.version > since_version:
            # Send initial state
            yield f"data: {json.dumps({'type': 'state', 'state': task.streaming_state.dict()})}\n\n"

        if task.status in [
            ChatTaskStatus.completed,
            ChatTaskStatus.failed,
            ChatTaskStatus.cancelled,
            ChatTaskStatus.interrupted,
        ]:
            yield f"data: {json.dumps({'type': 'completion', 'status': task.status.value})}\n\n"
            return

        queue = await task_event_bus.subscribe(task_id)
        try:
            while True:
                # Wait for data or disconnect
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"

                    if data.get("completed"):
                        break

                except asyncio.TimeoutError:
                    # Helper keep-alive
                    yield ": keep-alive\n\n"
                    # Check if task is still running (handle restarted server case)
                    t = await repository.get_chat_task(task_id)
                    if t and t.status not in (
                        ChatTaskStatus.pending,
                        ChatTaskStatus.running,
                    ):
                        yield f"data: {json.dumps({'type': 'completion', 'status': t.status.value})}\n\n"
                        break

                    # Detect orphaned tasks: DB says running but the
                    # asyncio task is no longer alive (silent crash,
                    # unhandled edge-case, etc.).
                    if t and t.status == ChatTaskStatus.running:
                        if not background_task_service.is_task_running(task_id):
                            _err = "Task processing stopped unexpectedly"
                            await repository.update_chat_task_status(
                                task_id, ChatTaskStatus.failed, _err
                            )
                            yield f"data: {json.dumps({'type': 'completion', 'status': 'failed', 'error': _err})}\n\n"
                            break

        finally:
            task_event_bus.unsubscribe(task_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}/events")
async def conversation_events(
    conversation_id: str,
    workspace_id: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """
    Subscribe to conversation events (e.g. title updates).
    """
    await _assert_workspace_access(workspace_id, user, "viewer")
    # Check conversation access
    has_access = await repository.check_conversation_access(
        conversation_id,
        user.id,
        is_admin=(user.role == "admin"),
        workspace_id=workspace_id,
    )
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")

    async def event_generator():
        channel = f"conversation:{conversation_id}"
        queue = await task_event_bus.subscribe(channel)
        try:
            while True:
                # Keep-alive heartbeat every 15s to handle load balancers
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            task_event_bus.unsubscribe(channel, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/tasks/{task_id}/cancel", response_model=ChatTaskResponse)
async def cancel_chat_task(task_id: str):
    """
    Cancel a running chat task.
    """
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in (ChatTaskStatus.pending, ChatTaskStatus.running):
        raise HTTPException(status_code=400, detail="Task is not running")

    # Cancel the task
    background_task_service.cancel_task(task_id)
    updated_task = await repository.cancel_chat_task(task_id)

    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to cancel task")

    return ChatTaskResponse(
        id=updated_task.id,
        conversation_id=updated_task.conversation_id,
        status=updated_task.status,
        user_message=updated_task.user_message,
        streaming_state=updated_task.streaming_state,
        response_content=updated_task.response_content,
        error_message=updated_task.error_message,
        created_at=updated_task.created_at,
        started_at=updated_task.started_at,
        completed_at=updated_task.completed_at,
        last_update_at=updated_task.last_update_at,
    )


@router.get("/conversations/{conversation_id}/members")
async def get_conversation_members(
    conversation_id: str,
    user: User = Depends(get_current_user),
):
    """Get conversation members"""
    db = Prisma()
    await db.connect()
    try:
        # Check if user has access to this conversation
        conversation = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"members": True}
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        is_admin = user.role == "admin"

        # Check membership
        user_member = next(
            (m for m in conversation.members if m.userId == user.id), None
        )
        if not is_admin and not user_member and conversation.userId != user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Return members
        return [{"user_id": m.userId, "role": m.role} for m in conversation.members]
    finally:
        await db.disconnect()


@router.put("/conversations/{conversation_id}/members")
async def update_conversation_members(
    conversation_id: str,
    request: dict,
    user: User = Depends(get_current_user),
):
    """Update conversation members (owner only)"""
    db = Prisma()
    await db.connect()
    try:
        # Check if user is owner
        conversation = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"members": True}
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Check if user is owner
        user_member = next(
            (m for m in conversation.members if m.userId == user.id), None
        )
        if not user_member or user_member.role != "owner":
            if conversation.userId != user.id:
                raise HTTPException(
                    status_code=403, detail="Only owner can manage members"
                )

        members = request.get("members", [])

        # Delete existing non-owner members
        await db.conversationmember.delete_many(
            where={"conversationId": conversation_id, "role": {"not": "owner"}}
        )

        # Add new members
        for member in members:
            mid = member["user_id"]
            role = member["role"]
            # Normalize non-owner members trying to be owner to editor
            if role == "owner" and mid != (conversation.userId or user.id):
                role = "editor"
            await db.conversationmember.create(
                data={"conversationId": conversation_id, "userId": mid, "role": role}
            )

        return {"success": True}
    finally:
        await db.disconnect()


@router.get("/conversations/{conversation_id}/tools")
async def get_conversation_tools(
    conversation_id: str,
    user: User = Depends(get_current_user),
):
    """Get conversation tool selections"""
    db = Prisma()
    await db.connect()
    try:
        # Check if user has access
        conversation = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"members": True}
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        is_admin = user.role == "admin"

        user_member = next(
            (m for m in conversation.members if m.userId == user.id), None
        )
        if not is_admin and not user_member and conversation.userId != user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get tool selections
        selections = await db.conversationtoolselection.find_many(
            where={"conversationId": conversation_id}
        )

        # Get tool group selections
        group_selections = await db.conversationtoolgroupselection.find_many(
            where={"conversationId": conversation_id}
        )

        return {
            "tool_config_ids": [s.toolConfigId for s in selections],
            "tool_group_ids": [s.toolGroupId for s in group_selections],
        }
    finally:
        await db.disconnect()


@router.put("/conversations/{conversation_id}/tools")
async def update_conversation_tools(
    conversation_id: str,
    request: dict,
    user: User = Depends(get_current_user),
):
    """Update conversation tool selections (owner/editor only)"""
    db = Prisma()
    await db.connect()
    try:
        # Check if user can edit
        conversation = await db.conversation.find_unique(
            where={"id": conversation_id}, include={"members": True}
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        user_member = next(
            (m for m in conversation.members if m.userId == user.id), None
        )
        if not user_member or user_member.role == "viewer":
            if conversation.userId != user.id:
                raise HTTPException(
                    status_code=403, detail="Only owner/editor can manage tools"
                )

        tool_config_ids = request.get("tool_config_ids", [])
        tool_group_ids = request.get("tool_group_ids", [])

        # Delete existing selections
        await db.conversationtoolselection.delete_many(
            where={"conversationId": conversation_id}
        )

        # Add new selections
        for tool_id in tool_config_ids:
            await db.conversationtoolselection.create(
                data={"conversationId": conversation_id, "toolConfigId": tool_id}
            )

        # Update group selections
        await db.conversationtoolgroupselection.delete_many(
            where={"conversationId": conversation_id}
        )
        for group_id in tool_group_ids:
            await db.conversationtoolgroupselection.create(
                data={"conversationId": conversation_id, "toolGroupId": group_id}
            )

        return {"success": True}
    finally:
        await db.disconnect()


# =============================================================================
# Schema Indexer Routes (for SQL database tools)
# =============================================================================


@router.post(
    "/tools/{tool_id}/schema/index",
    response_model=SchemaIndexJobResponse,
    tags=["Schema Indexer"],
)
async def trigger_schema_index(
    tool_id: str,
    request: TriggerSchemaIndexRequest = Body(default=TriggerSchemaIndexRequest()),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Trigger schema indexing for a SQL database tool. Admin only.

    Starts a background job to introspect the database schema and create
    embeddings for each table (table name, columns, constraints, indexes).

    The embeddings enable semantic search of the database schema, helping
    the AI write better queries by understanding the database structure.
    """
    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Only allow for SQL database tools that support schema indexing
    if tool_config.tool_type.value not in SCHEMA_INDEXER_CAPABLE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Schema indexing is only available for {', '.join(sorted(SCHEMA_INDEXER_CAPABLE_TYPES))} tools, got '{tool_config.tool_type.value}'",
        )

    # Check that schema indexing is enabled
    conn_config = tool_config.connection_config or {}
    if not conn_config.get("schema_index_enabled", False):
        raise HTTPException(
            status_code=400,
            detail="Schema indexing is not enabled for this tool. Enable it in the tool configuration.",
        )

    # Validate pgvector is available
    if not await ensure_pgvector_extension(logger_override=logger):
        raise HTTPException(
            status_code=503,
            detail="pgvector extension not available. Please install it: CREATE EXTENSION IF NOT EXISTS vector;",
        )

    # Start indexing
    job = await schema_indexer.trigger_index(
        tool_config_id=tool_id,
        tool_type=tool_config.tool_type.value,
        connection_config=conn_config,
        full_reindex=request.full_reindex,
        tool_name=safe_tool_name(tool_config.name) or None,
    )

    return schema_indexer.job_to_response(job)


@router.get(
    "/tools/{tool_id}/schema/status",
    response_model=Optional[SchemaIndexJobResponse],
    tags=["Schema Indexer"],
)
async def get_schema_index_status(tool_id: str, _user: User = Depends(require_admin)):
    """Get the latest schema indexing job status for a tool. Admin only."""
    job = await schema_indexer.get_latest_job(tool_id)
    return job


@router.get(
    "/tools/{tool_id}/schema/job/{job_id}",
    response_model=SchemaIndexJobResponse,
    tags=["Schema Indexer"],
)
async def get_schema_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """Get a specific schema indexing job status. Admin only."""
    job = await schema_indexer.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")
    return job


@router.post(
    "/tools/{tool_id}/schema/job/{job_id}/cancel",
    tags=["Schema Indexer"],
)
async def cancel_schema_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """Cancel an active schema indexing job. Admin only."""
    job = await schema_indexer.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")

    success = await schema_indexer.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (not active)")

    return {"message": "Cancellation requested", "job_id": job_id}


@router.delete(
    "/tools/{tool_id}/schema/index",
    tags=["Schema Indexer"],
)
async def delete_schema_index(tool_id: str, _user: User = Depends(require_admin)):
    """Delete all schema embeddings for a tool. Admin only."""
    success, message = await schema_indexer.delete_index(tool_id)
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {"message": message}


class SchemaStatsResponse(BaseModel):
    """Response for schema index statistics."""

    embedding_count: int = Field(description="Number of table embeddings")
    last_indexed_at: Optional[str] = Field(
        default=None, description="ISO timestamp of last successful index"
    )
    schema_hash: Optional[str] = Field(
        default=None, description="Hash of indexed schema for change detection"
    )
    # Memory estimation for the schema index stored in pgvector
    embedding_dimension: Optional[int] = Field(
        default=None, description="Dimension of embedding vectors"
    )
    estimated_memory_mb: Optional[float] = Field(
        default=None,
        description="Estimated memory size in MB (pgvector storage, not process RAM)",
    )


def _estimate_schema_index_memory_mb(
    embedding_count: int,
    embedding_dimension: int | None,
    avg_content_chars: int = 1500,
) -> float | None:
    """
    Estimate memory footprint for a schema index stored in pgvector.

    This is the storage footprint in PostgreSQL, not process RAM
    (schema indexes use pgvector, not FAISS in-memory indexes).

    Args:
        embedding_count: Number of embeddings
        embedding_dimension: Dimension of embedding vectors
        avg_content_chars: Average characters per schema entry (table definitions)

    Returns:
        Estimated size in MB, or None if dimension is unknown
    """
    if embedding_count == 0 or embedding_dimension is None:
        return None

    # Vector storage: embedding_count * dimensions * 4 bytes (float32)
    vector_bytes = embedding_count * embedding_dimension * 4

    # Content storage: schema text (table definitions are typically 1-3KB)
    content_bytes = embedding_count * avg_content_chars

    # Metadata overhead: ~200 bytes per embedding (uuid, index_name, table_name, etc.)
    metadata_bytes = embedding_count * 200

    # PostgreSQL row overhead: ~23 bytes per row + ~4 bytes tuple header
    pg_overhead_bytes = embedding_count * 27

    total_bytes = vector_bytes + content_bytes + metadata_bytes + pg_overhead_bytes

    # Convert to MB with 10% overhead for indexes
    return round(total_bytes * 1.1 / (1024 * 1024), 2)


@router.get(
    "/tools/{tool_id}/schema/stats",
    response_model=SchemaStatsResponse,
    tags=["Schema Indexer"],
)
async def get_schema_index_stats(tool_id: str, _user: User = Depends(require_admin)):
    """Get schema index statistics for a tool. Admin only."""
    # Get tool config first to get the name
    tool_config = await repository.get_tool_config(tool_id)

    # Get embedding count - pass both tool_id and safe tool name
    tool_name = safe_tool_name(tool_config.name) if tool_config else None
    embedding_count = await schema_indexer.get_embedding_count(tool_id, tool_name)
    last_indexed_at = None
    schema_hash = None

    if tool_config:
        conn_config = tool_config.connection_config or {}
        last_indexed_at = conn_config.get("last_schema_indexed_at")
        schema_hash = conn_config.get("schema_hash")

    # Get embedding dimension from app settings
    settings = await repository.get_settings()
    embedding_dimension = settings.embedding_dimension if settings else None

    # Estimate memory size
    estimated_memory_mb = _estimate_schema_index_memory_mb(
        embedding_count, embedding_dimension
    )

    return SchemaStatsResponse(
        embedding_count=embedding_count,
        last_indexed_at=last_indexed_at,
        schema_hash=schema_hash,
        embedding_dimension=embedding_dimension,
        estimated_memory_mb=estimated_memory_mb,
    )


@router.get(
    "/schema/jobs",
    response_model=List[SchemaIndexJobResponse],
    tags=["Schema Indexer"],
)
async def list_schema_jobs(
    limit: int = 50,
    _user: User = Depends(require_admin),
):
    """
    List all schema indexing jobs across all tools. Admin only.

    Returns jobs sorted by created_at descending.
    """
    jobs = await schema_indexer.list_all_jobs(limit=limit)
    return jobs


@router.post(
    "/tools/{tool_id}/schema/job/{job_id}/retry",
    response_model=SchemaIndexJobResponse,
    tags=["Schema Indexer"],
)
async def retry_schema_index_job(
    tool_id: str,
    job_id: str,
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Retry a failed or cancelled schema indexing job. Admin only.

    Creates a new job with a full reindex.
    """
    # Verify job belongs to tool
    existing_job = await schema_indexer.get_job_status(job_id)
    if not existing_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if existing_job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")

    # Retry the job
    new_job = await schema_indexer.retry_job(job_id)
    if not new_job:
        raise HTTPException(
            status_code=400, detail="Cannot retry job (not failed/cancelled)"
        )

    return schema_indexer.job_to_response(new_job)


# =============================================================================
# SolidWorks PDM Indexer Routes
# =============================================================================


@router.post(
    "/tools/{tool_id}/pdm/index",
    response_model=PdmIndexJobResponse,
    tags=["PDM Indexer"],
)
async def trigger_pdm_index(
    tool_id: str,
    request: TriggerPdmIndexRequest = Body(default=TriggerPdmIndexRequest()),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Trigger PDM metadata indexing for a SolidWorks PDM tool. Admin only.

    Starts a background job to extract document metadata from the PDM database
    and create embeddings for semantic search over parts, assemblies, drawings,
    materials, BOMs, and other PDM properties.
    """
    # Get the tool config
    tool_config = await repository.get_tool_config(tool_id)
    if not tool_config:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Only allow for solidworks_pdm tools
    expected_type = ToolType("solidworks_pdm")
    if tool_config.tool_type != expected_type:
        raise HTTPException(
            status_code=400,
            detail=f"PDM indexing is only available for solidworks_pdm tools, got '{tool_config.tool_type.value}'",
        )

    # Validate pgvector is available
    if not await ensure_pgvector_extension(logger_override=logger):
        raise HTTPException(
            status_code=503,
            detail="pgvector extension not available. Please install it: CREATE EXTENSION IF NOT EXISTS vector;",
        )

    # Start indexing
    job = await pdm_indexer.trigger_index(
        tool_config_id=tool_id,
        connection_config=tool_config.connection_config or {},
        full_reindex=request.full_reindex,
        tool_name=safe_tool_name(tool_config.name) or None,
    )

    return PdmIndexJobResponse(
        id=job.id,
        tool_config_id=job.tool_config_id,
        status=job.status,
        index_name=job.index_name,
        progress_percent=job.progress_percent,
        total_documents=job.total_documents,
        processed_documents=job.processed_documents,
        skipped_documents=job.skipped_documents,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/tools/{tool_id}/pdm/status",
    response_model=Optional[PdmIndexJobResponse],
    tags=["PDM Indexer"],
)
async def get_pdm_index_status(tool_id: str, _user: User = Depends(require_admin)):
    """Get the latest PDM indexing job status for a tool. Admin only."""
    job = await pdm_indexer.get_latest_job(tool_id)
    return job


@router.get(
    "/tools/{tool_id}/pdm/job/{job_id}",
    response_model=PdmIndexJobResponse,
    tags=["PDM Indexer"],
)
async def get_pdm_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """Get a specific PDM indexing job status. Admin only."""
    job = await pdm_indexer.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")
    return job


@router.post(
    "/tools/{tool_id}/pdm/job/{job_id}/cancel",
    tags=["PDM Indexer"],
)
async def cancel_pdm_index_job(
    tool_id: str, job_id: str, _user: User = Depends(require_admin)
):
    """Cancel an active PDM indexing job. Admin only."""
    job = await pdm_indexer.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")

    success = await pdm_indexer.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (not active)")

    return {"message": "Cancellation requested", "job_id": job_id}


@router.delete(
    "/tools/{tool_id}/pdm/index",
    tags=["PDM Indexer"],
)
async def delete_pdm_index(tool_id: str, _user: User = Depends(require_admin)):
    """Delete all PDM embeddings for a tool. Admin only."""
    success, message = await pdm_indexer.delete_index(tool_id)
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {"message": message}


class PdmStatsResponse(BaseModel):
    """Response for PDM index statistics."""

    document_count: int = Field(description="Number of indexed documents")
    embedding_count: int = Field(description="Number of embeddings")
    last_indexed_at: Optional[str] = Field(
        default=None, description="ISO timestamp of last successful index"
    )


@router.get(
    "/tools/{tool_id}/pdm/stats",
    response_model=PdmStatsResponse,
    tags=["PDM Indexer"],
)
async def get_pdm_index_stats(tool_id: str, _user: User = Depends(require_admin)):
    """Get PDM index statistics for a tool. Admin only."""
    # Get counts
    tool_config = await repository.get_tool_config(tool_id)
    tool_name = safe_tool_name(tool_config.name) if tool_config else None

    document_count = await pdm_indexer.get_document_count(tool_id, tool_name)
    embedding_count = await pdm_indexer.get_embedding_count(tool_id, tool_name)

    # Get last indexed timestamp from latest completed job
    last_indexed_at = None
    latest_job = await pdm_indexer.get_latest_job(tool_id)
    if (
        latest_job
        and latest_job.status.value == "completed"
        and latest_job.completed_at
    ):
        last_indexed_at = latest_job.completed_at.isoformat()

    return PdmStatsResponse(
        document_count=document_count,
        embedding_count=embedding_count,
        last_indexed_at=last_indexed_at,
    )


@router.get(
    "/pdm/jobs",
    response_model=List[PdmIndexJobResponse],
    tags=["PDM Indexer"],
)
async def list_pdm_jobs(
    limit: int = 50,
    _user: User = Depends(require_admin),
):
    """
    List all PDM indexing jobs across all tools. Admin only.

    Returns jobs sorted by created_at descending.
    """
    jobs = await pdm_indexer.list_all_jobs(limit=limit)
    return jobs


@router.post(
    "/tools/{tool_id}/pdm/job/{job_id}/retry",
    response_model=PdmIndexJobResponse,
    tags=["PDM Indexer"],
)
async def retry_pdm_index_job(
    tool_id: str,
    job_id: str,
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Retry a failed or cancelled PDM indexing job. Admin only.

    Creates a new job with a full reindex.
    """
    # Verify job belongs to tool
    existing_job = await pdm_indexer.get_job_status(job_id)
    if not existing_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if existing_job.tool_config_id != tool_id:
        raise HTTPException(status_code=404, detail="Job not found for this tool")

    # Retry the job
    new_job = await pdm_indexer.retry_job(job_id)
    if not new_job:
        raise HTTPException(
            status_code=400, detail="Cannot retry job (not failed/cancelled)"
        )

    return PdmIndexJobResponse(
        id=new_job.id,
        tool_config_id=new_job.tool_config_id,
        status=new_job.status,
        index_name=new_job.index_name,
        progress_percent=new_job.progress_percent,
        total_documents=new_job.total_documents,
        processed_documents=new_job.processed_documents,
        skipped_documents=new_job.skipped_documents,
        total_chunks=new_job.total_chunks,
        processed_chunks=new_job.processed_chunks,
        error_message=new_job.error_message,
        cancel_requested=new_job.cancel_requested,
        created_at=new_job.created_at,
        started_at=new_job.started_at,
        completed_at=new_job.completed_at,
    )
