"""
Indexer API routes.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import httpx
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.embedding_models import (
    OPENAI_EMBEDDING_PRIORITY,
    get_embedding_models,
)
from ragtime.core.encryption import decrypt_secret
from ragtime.core.event_bus import task_event_bus
from ragtime.core.git import check_repo_visibility as git_check_visibility
from ragtime.core.git import fetch_branches as git_fetch_branches
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import (
    MODEL_FAMILY_PATTERNS,
    get_context_limit,
    get_output_limit,
    supports_function_calling,
    update_model_function_calling,
    update_model_limit,
)
from ragtime.core.ollama import get_model_details, is_reachable
from ragtime.core.ollama import list_models
from ragtime.core.ollama import list_models as ollama_list_models
from ragtime.core.security import get_current_user, require_admin
from ragtime.core.sql_utils import (
    MssqlConnectionError,
    MysqlConnectionError,
    mssql_connect,
    mysql_connect,
)
from ragtime.core.ssh import (
    SSHConfig,
    SSHTunnel,
    build_ssh_tunnel_config,
    execute_ssh_command,
    ssh_tunnel_config_from_dict,
    test_ssh_connection,
)
from ragtime.core.validation import require_valid_embedding_provider
from ragtime.core.vision_models import list_vision_models
from ragtime.indexer.background_tasks import background_task_service
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.models import (
    AnalyzeIndexRequest,
    AppSettings,
    ChatMessage,
    ChatTaskResponse,
    ChatTaskStatus,
    CheckRepoVisibilityRequest,
    ConfigurationWarning,
    Conversation,
    ConversationResponse,
    CreateConversationRequest,
    CreateIndexRequest,
    CreateToolConfigRequest,
    EmbeddingStatus,
    FetchBranchesRequest,
    FetchBranchesResponse,
    FilesystemAnalysisJobResponse,
    FilesystemConnectionConfig,
    FilesystemIndexJobResponse,
    IndexAnalysisResult,
    IndexConfig,
    IndexInfo,
    IndexJobResponse,
    IndexStatus,
    MssqlDiscoverRequest,
    MssqlDiscoverResponse,
    MysqlDiscoverRequest,
    MysqlDiscoverResponse,
    OcrMode,
    PdmDiscoverRequest,
    PdmDiscoverResponse,
    PdmIndexJobResponse,
    PostgresDiscoverRequest,
    PostgresDiscoverResponse,
    RepoVisibilityResponse,
    RetryVisualizationRequest,
    RetryVisualizationResponse,
    SchemaIndexJobResponse,
    SendMessageRequest,
    ToolConfig,
    ToolTestRequest,
    ToolType,
    TriggerFilesystemIndexRequest,
    TriggerPdmIndexRequest,
    TriggerSchemaIndexRequest,
    UpdateSettingsRequest,
    UpdateToolConfigRequest,
    VectorStoreType,
)
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import SCHEMA_INDEXER_CAPABLE_TYPES, schema_indexer
from ragtime.indexer.service import indexer
from ragtime.indexer.title_generation import schedule_title_generation
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_backends import FAISS_INDEX_BASE_PATH
from ragtime.indexer.vector_utils import ensure_pgvector_extension
from ragtime.mcp.server import notify_tools_changed
from ragtime.rag import rag
from ragtime.tools.chart import create_chart
from ragtime.tools.datatable import create_datatable
from ragtime.tools.mssql import test_mssql_connection
from ragtime.tools.mysql import test_mysql_connection

if TYPE_CHECKING:
    from prisma.models import User

logger = get_logger(__name__)

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
    re-creates the FAISS index. Only works for indexes created from git repos.
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
            shutil.move(str(old_path), str(new_path))
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
                shutil.move(str(new_path), str(old_path))
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

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add both files to zip with a single directory for clarity
        zf.write(faiss_file, arcname=f"{name}/index.faiss")
        zf.write(pkl_file, arcname=f"{name}/index.pkl")

    zip_buffer.seek(0)

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

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add both files to zip with a single directory for clarity
        zf.write(faiss_file, arcname=f"{name}/index.faiss")
        zf.write(pkl_file, arcname=f"{name}/index.pkl")

    zip_buffer.seek(0)

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

    # Check if embedding config is changing
    current_settings = await repository.get_settings()
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


@router.post("/tools", response_model=ToolConfig, tags=["Tools"])
async def create_tool_config(
    request: CreateToolConfigRequest, _user: User = Depends(require_admin)
):
    """Create a new tool configuration. Admin only."""
    connection_config = request.connection_config.copy()

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
        allow_write=request.allow_write,
    )
    result = await repository.create_tool_config(config)

    # Reinitialize RAG agent to make the new tool available immediately
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed
    notify_tools_changed()

    # Auto-start indexing for filesystem_indexer tools
    if request.tool_type == ToolType.FILESYSTEM_INDEXER:
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

        try:
            # Quick ping-style check with short timeout
            result = await asyncio.wait_for(
                _heartbeat_check(tool.tool_type, tool.connection_config),
                timeout=5.0,  # Short timeout for heartbeat
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
                error="Heartbeat timeout (5s)",
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
                    from ragtime.indexer.vector_backends import FAISS_INDEX_BASE_PATH

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
                    from ragtime.indexer.vector_backends import get_faiss_backend

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
            "-c",
            "SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;",
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
            # Parse database names from output
            output = _stdout.decode("utf-8", errors="replace").strip()
            databases = [db.strip() for db in output.split("\n") if db.strip()]
            return PostgresDiscoverResponse(success=True, databases=databases)
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return PostgresDiscoverResponse(
                success=False, databases=[], error=f"Connection failed: {error}"
            )

    except asyncio.TimeoutError:
        return PostgresDiscoverResponse(
            success=False, databases=[], error="Connection timed out after 15 seconds"
        )
    except FileNotFoundError:
        return PostgresDiscoverResponse(
            success=False, databases=[], error="psql command not found"
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return PostgresDiscoverResponse(
            success=False, databases=[], error=f"Discovery failed: {error_msg}"
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
    ) -> tuple[bool, list[str], str | None]:
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

                # Query system databases
                cursor.execute(
                    """
                    SELECT name FROM sys.databases
                    WHERE database_id > 4  -- Exclude system databases
                    ORDER BY name
                    """
                )
                rows = cursor.fetchall() or []
                databases: list[str] = []
                for row in rows:
                    if isinstance(row, dict):
                        name = row.get("name")
                    elif row and len(row) > 0:
                        name = row[0]
                    else:
                        name = None
                    if name:
                        databases.append(str(name))
                return True, databases, None

        except MssqlConnectionError as e:
            return False, [], str(e)
        except Exception as e:
            return False, [], f"Connection error: {str(e)}"

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

        success, databases, error = await asyncio.to_thread(
            discover_databases, connect_host, connect_port
        )
        return MssqlDiscoverResponse(success=success, databases=databases, error=error)
    except asyncio.TimeoutError:
        return MssqlDiscoverResponse(
            success=False, databases=[], error="Connection timed out"
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return MssqlDiscoverResponse(
            success=False, databases=[], error=f"Discovery failed: {error_msg}"
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
    ) -> tuple[bool, list[str], str | None]:
        try:
            # Handle Docker container mode
            if request.container:
                container = request.container
                docker_network = request.docker_network

                # Build docker exec command to get credentials from env
                if docker_network:
                    exec_prefix = f"docker exec {container}"
                else:
                    exec_prefix = f"docker exec {container}"

                # Get credentials from container environment
                import subprocess

                def get_env_var(var_name: str) -> str | None:
                    try:
                        result = subprocess.run(
                            f"{exec_prefix} printenv {var_name}",
                            shell=True,
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
                mysql_cmd = f'{exec_prefix} mysql -u{user} -p"{password}" -N -e "SHOW DATABASES"'
                result = subprocess.run(
                    mysql_cmd,
                    shell=True,
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
                return True, databases, None

            # Direct connection mode (or through SSH tunnel)
            try:
                with mysql_connect(
                    host=connect_host,
                    port=connect_port,
                    user=request.user,
                    password=request.password,
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
                    return True, db_list, None
            except Exception as e:
                # Fallback: if global discovery fails, try connecting to the specific database if provided
                if request.database:
                    try:
                        with mysql_connect(
                            host=connect_host,
                            port=connect_port,
                            user=request.user,
                            password=request.password,
                            database=request.database,
                            connect_timeout=10,
                        ):
                            pass
                        return True, [request.database], None
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

        success, databases, error = await asyncio.to_thread(
            discover_databases, connect_host, connect_port
        )
        return MysqlDiscoverResponse(success=success, databases=databases, error=error)
    except asyncio.TimeoutError:
        return MysqlDiscoverResponse(
            success=False, databases=[], error="Connection timed out"
        )
    except Exception as e:
        error_msg = str(e)
        if "Authentication" in error_msg:
            error_msg = f"SSH tunnel authentication failed: {error_msg}"
        return MysqlDiscoverResponse(
            success=False, databases=[], error=f"Discovery failed: {error_msg}"
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
        # Quick check - list first few entries to verify read access
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
    privileged = False
    has_sys_admin = False

    try:
        # Read the capabilities from /proc/self/status
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    # CapEff is a hex bitmask of effective capabilities
                    cap_hex = line.split(":")[1].strip()
                    cap_int = int(cap_hex, 16)

                    # CAP_SYS_ADMIN is bit 21 (value 2097152 = 0x200000)
                    # In privileged mode, typically all bits are set
                    cap_sys_admin_bit = 1 << 21

                    has_sys_admin = bool(cap_int & cap_sys_admin_bit)

                    # Check if privileged (typically all caps = 0x3fffffffff or similar)
                    # A good heuristic: if many high-level caps are set, likely privileged
                    # CAP_MKNOD (bit 27), CAP_SYS_RAWIO (bit 17), CAP_SYS_PTRACE (bit 19)
                    high_privilege_bits = (1 << 27) | (1 << 17) | (1 << 19) | (1 << 21)
                    if (cap_int & high_privilege_bits) == high_privilege_bits:
                        privileged = True

                    break
    except Exception as e:
        logger.debug(f"Failed to read capabilities from /proc/self/status: {e}")

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
        result = execute_ssh_command(config, cmd)
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
                # Skip entries we can't access
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


# OpenAI embedding models - prioritized list (from LiteLLM community data)
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

    Uses LiteLLM's community-maintained model database to identify embedding models.
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

    Uses LiteLLM's community-maintained database to identify which models are
    embedding models and to get metadata like output dimensions.
    """
    try:
        # Get embedding model metadata from LiteLLM (cached)
        litellm_embedding_models = await get_embedding_models()

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()

            data = response.json()
            models = []

            # Filter for embedding models using LiteLLM data
            for model in data.get("data", []):
                model_id = model.get("id", "")

                # Check if LiteLLM knows this is an embedding model
                litellm_info = litellm_embedding_models.get(model_id)
                if litellm_info and litellm_info.provider == "openai":
                    models.append(
                        EmbeddingModel(
                            id=model_id,
                            name=model_id,
                            dimensions=litellm_info.output_vector_size,
                        )
                    )
                # Fallback: also include models with "embedding" in the name
                # (covers new models not yet in LiteLLM)
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
                message=f"Found {len(models)} embedding model(s) (validated against LiteLLM database).",
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

    provider: str = Field(..., description="LLM provider: 'openai' or 'anthropic'")
    api_key: str = Field(..., description="API key for the provider")


class LLMModel(BaseModel):
    """Information about an available LLM model."""

    id: str
    name: str
    created: Optional[int] = None
    group: Optional[str] = None
    is_latest: bool = False
    max_output_tokens: Optional[int] = None


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


class AvailableModelsResponse(BaseModel):
    """Response with all available models from configured providers."""

    models: List[AvailableModel] = []
    default_model: Optional[str] = None
    current_model: Optional[str] = None  # Currently selected model in settings
    allowed_models: List[str] = []  # List of allowed model IDs (for settings UI)


# Sensible default models for each provider
OPENAI_DEFAULT_MODEL = ""
ANTHROPIC_DEFAULT_MODEL = ""


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
                elif provider == "ollama":
                    model.group = match.group(1).title()
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


def _extract_version(name: str) -> float:
    """Extract numeric version from model name.

    Handles various naming conventions:
    - Anthropic: 'Claude Haiku 4.5' -> 4.5
    - OpenAI display: 'GPT-4.1 Mini' -> 4.1
    - OpenAI ID: 'gpt-4.1-mini' -> 4.1
    - Dated versions: 'gpt-4-0613' -> 4.0 (base version, no sub-version)
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
                elif model.provider == "ollama":
                    model.group = match.group(1).title()
                found_group = True
                break

        if not found_group:
            model.group = f"Other {model.provider.title()}"

    # Grouping and is_latest logic
    grouped = defaultdict(list)
    for model in models:
        grouped[(model.provider, model.group)].append(model)

    for _, group_models in grouped.items():
        if not group_models:
            continue

        # Sort: version (higher first), then created date, then ID length
        group_models.sort(
            key=lambda m: (-_extract_version(m.name), -(m.created or 0), len(m.id))
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
    else:
        return LLMModelsResponse(
            success=False,
            message=f"Unknown provider: {request.provider}. Supported: 'openai', 'anthropic'",
        )


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
                # All Claude models support function calling (chat capable)
                output_limit = await get_output_limit(model_id)
                models.append(
                    LLMModel(
                        id=model_id,
                        name=display_name,
                        created=None,
                        max_output_tokens=output_limit,
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

        # Fetch details for each model to get context window
        # We allow this to be "best effort" - if details fail, we just don't update limit
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
                        # Extract context length
                        # Ollama usually returns 'model_info' with parameters
                        # Or 'parameters' string which is hard to parse
                        # But /api/show usually has 'model_info' -> 'llama.context_length' or 'context_length'
                        model_info = details.get("model_info", {})

                        # Try common keys for context length
                        ctx_len = (
                            model_info.get("llama.context_length")
                            or model_info.get("context_length")
                            or model_info.get("max_tokens")
                        )

                        if ctx_len:
                            try:
                                limit = int(ctx_len)
                                update_model_limit(models[idx].id, limit)
                                # For Ollama, context_length is the max output
                                models[idx].max_output_tokens = limit
                            except (ValueError, TypeError):
                                pass
        except Exception as e:
            logger.warning(f"Failed to fetch Ollama model details: {e}")

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


# =============================================================================
# Conversation/Chat Endpoints
# =============================================================================


@router.get(
    "/chat/available-models", response_model=AvailableModelsResponse, tags=["Chat"]
)
async def get_available_chat_models():
    """
    Get all available models from configured LLM providers.

    Returns models from OpenAI and/or Anthropic based on which API keys are configured.
    """
    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse()

    all_models: List[AvailableModel] = []
    default_model = None

    # Fetch OpenAI models if API key is configured
    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        try:
            result = await _fetch_openai_models(app_settings.openai_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(
                        AvailableModel(
                            id=m.id,
                            name=m.name,
                            provider="openai",
                            context_limit=await get_context_limit(m.id),
                            max_output_tokens=m.max_output_tokens,
                            created=m.created,
                        )
                    )
                if not default_model and result.default_model:
                    default_model = result.default_model
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")

    # Fetch Anthropic models if API key is configured
    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        try:
            result = await _fetch_anthropic_models(app_settings.anthropic_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(
                        AvailableModel(
                            id=m.id,
                            name=m.name,
                            provider="anthropic",
                            context_limit=await get_context_limit(m.id),
                            max_output_tokens=m.max_output_tokens,
                            created=m.created,
                        )
                    )
                if not default_model and result.default_model:
                    default_model = result.default_model
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

    # Fetch Ollama models if LLM provider is Ollama and URL is configured
    if app_settings.llm_provider == "ollama":
        ollama_url = getattr(
            app_settings,
            "llm_ollama_base_url",
            app_settings.ollama_base_url,
        )
        if ollama_url:
            try:
                result = await _fetch_ollama_llm_models(ollama_url)
                if result.success:
                    for m in result.models:
                        all_models.append(
                            AvailableModel(
                                id=m.id,
                                name=m.name,
                                provider="ollama",
                                context_limit=await get_context_limit(m.id),
                                max_output_tokens=m.max_output_tokens,
                                created=m.created,
                            )
                        )
                    if not default_model and result.default_model:
                        default_model = result.default_model
            except Exception as e:
                logger.warning(f"Failed to fetch Ollama models: {e}")

    # Use current settings model as default if available
    current_model = app_settings.llm_model
    if current_model and any(m.id == current_model for m in all_models):
        default_model = current_model

    # Filter by allowed models if specified
    allowed_models = app_settings.allowed_chat_models or []
    if allowed_models:
        all_models = [m for m in all_models if m.id in allowed_models]
        # Ensure default model is in allowed list
        if default_model and default_model not in allowed_models:
            default_model = all_models[0].id if all_models else None

    # Assign groups to models for UI organization
    all_models = _assign_model_groups(all_models)

    return AvailableModelsResponse(
        models=all_models, default_model=default_model, current_model=current_model
    )


@router.get("/chat/all-models", response_model=AvailableModelsResponse, tags=["Chat"])
async def get_all_chat_models(_user: User = Depends(require_admin)):
    """
    Get ALL available models from configured LLM providers (unfiltered).

    Used by the settings UI to show all models for selection.
    """
    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse()

    all_models: List[AvailableModel] = []

    # Fetch OpenAI models if API key is configured
    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        try:
            result = await _fetch_openai_models(app_settings.openai_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(
                        AvailableModel(
                            id=m.id,
                            name=m.name,
                            provider="openai",
                            context_limit=await get_context_limit(m.id),
                            max_output_tokens=m.max_output_tokens,
                            created=m.created,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")

    # Fetch Anthropic models if API key is configured
    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        try:
            result = await _fetch_anthropic_models(app_settings.anthropic_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(
                        AvailableModel(
                            id=m.id,
                            name=m.name,
                            provider="anthropic",
                            context_limit=await get_context_limit(m.id),
                            max_output_tokens=m.max_output_tokens,
                            created=m.created,
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

    # Fetch Ollama models if LLM provider is Ollama and URL is configured
    if app_settings.llm_provider == "ollama":
        ollama_url = getattr(
            app_settings,
            "llm_ollama_base_url",
            app_settings.ollama_base_url,
        )
        if ollama_url:
            try:
                result = await _fetch_ollama_llm_models(ollama_url)
                if result.success:
                    for m in result.models:
                        all_models.append(
                            AvailableModel(
                                id=m.id,
                                name=m.name,
                                provider="ollama",
                                context_limit=await get_context_limit(m.id),
                                max_output_tokens=m.max_output_tokens,
                                created=m.created,
                            )
                        )
            except Exception as e:
                logger.warning(f"Failed to fetch Ollama models: {e}")

    # Get currently allowed models from settings
    allowed_models = app_settings.allowed_chat_models or []

    # Assign groups to models for UI organization
    all_models = _assign_model_groups(all_models)

    return AvailableModelsResponse(
        models=all_models,
        default_model=app_settings.llm_model,
        current_model=app_settings.llm_model,
        allowed_models=allowed_models,
    )


def _to_conversation_response(conv: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        user_id=conv.user_id,
        username=conv.username,
        display_name=conv.display_name,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        active_task_id=conv.active_task_id,
        tool_output_mode=conv.tool_output_mode,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user: User = Depends(get_current_user)):
    """List chat conversations for the current user."""
    # Admins can see all, regular users only see their own
    is_admin = user.role == "admin"
    convs = await repository.list_conversations(user_id=user.id, include_all=is_admin)
    return [_to_conversation_response(c) for c in convs]


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    user: User = Depends(get_current_user),
):
    """Create a new chat conversation for the current user."""
    # Get default model from app settings if not provided
    app_settings = await repository.get_settings()
    default_model = app_settings.llm_model if app_settings else "gpt-4-turbo"

    title = request.title if request and request.title else "New Chat"
    model = request.model if request and request.model else default_model

    conv = await repository.create_conversation(
        title=title, model=model, user_id=user.id
    )
    return _to_conversation_response(conv)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str, user: User = Depends(get_current_user)
):
    """Get a specific conversation. Users can only access their own conversations."""
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, user: User = Depends(get_current_user)
):
    """Delete a conversation. Users can only delete their own conversations."""
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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
    conversation_id: str, body: dict, user: User = Depends(get_current_user)
):
    """Update a conversation's title. Users can only update their own conversations."""
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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
    conversation_id: str, body: dict, user: User = Depends(get_current_user)
):
    """Update a conversation's model. Users can only update their own conversations."""
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    conv = await repository.update_conversation_model(conversation_id, model)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return _to_conversation_response(conv)


@router.patch(
    "/conversations/{conversation_id}/tool-output-mode",
    response_model=ConversationResponse,
)
async def update_conversation_tool_output_mode(
    conversation_id: str, body: dict, user: User = Depends(get_current_user)
):
    """Update a conversation's tool output mode. Users can only update their own conversations."""
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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
    conversation_id: str, keep_count: int, user: User = Depends(get_current_user)
):
    """
    Truncate conversation messages to keep only the first N messages.
    Used when editing/resending a message to remove subsequent messages.
    """
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and get a response.
    Non-streaming version.
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

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
    for msg in conv.messages[:-1]:  # Exclude the current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    # Generate response
    try:
        answer = await rag.process_query(user_message, chat_history)
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
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and stream the response.
    Returns SSE stream of tokens.
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

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
    for msg in conv.messages[:-1]:  # Exclude current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    async def stream_response():
        """Generate streaming response tokens."""
        chunk_id = f"chatcmpl-{int(time.time())}"
        full_response = ""
        tool_calls_collected: list[dict[str, Any]] = (
            []
        )  # Collect tool calls for storage (deprecated)
        chronological_events: list[dict[str, Any]] = (
            []
        )  # Collect events in order (content and tools)
        current_tool_call: dict[str, Any] | None = (
            None  # Track current tool call being built
        )

        try:
            # Use UI agent (with chart tool and enhanced prompt)
            async for event in rag.process_query_stream(
                user_message, chat_history, is_ui=True
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
                            # Also keep deprecated tool_calls format for backward compatibility
                            tool_calls_collected.append(
                                {
                                    "tool": current_tool_call["tool"],
                                    "input": current_tool_call.get("input"),
                                    "output": current_tool_call.get("output"),
                                }
                            )
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

            # Save the full response with tool calls and chronological events
            updated_conv = await repository.add_message(
                conversation_id,
                "assistant",
                full_response,
                tool_calls=tool_calls_collected if tool_calls_collected else None,
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
                # Also add to deprecated format
                tool_calls_collected.append(
                    {
                        "tool": current_tool_call["tool"],
                        "input": current_tool_call.get("input"),
                        "output": "(interrupted)",
                    }
                )

            # Persist whatever we have so far, including collected tool calls and events
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
                    tool_calls=tool_calls_collected if tool_calls_collected else None,
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


@router.post("/conversations/{conversation_id}/retry-visualization")
async def retry_visualization(
    conversation_id: str,
    request: RetryVisualizationRequest,
    user: User = Depends(get_current_user),
):
    """
    Retry a failed visualization tool call.

    Directly invokes the create_datatable or create_chart tool with the provided
    source data. No LLM call is needed since we have structured data.

    For datatables, source_data should be: {"columns": [...], "rows": [...]}
    For charts, source_data should be: {"labels": [...], "datasets": [...], "chart_type": "..."}
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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
    user: User = Depends(get_current_user),
):
    """
    Send a message to a conversation and process it in the background.
    Returns a task object that can be polled for status and results.
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(
            status_code=503, detail="RAG service initializing, please retry"
        )

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Check if there's already an active task
    existing_task = await repository.get_active_task_for_conversation(conversation_id)
    if existing_task:
        # Return the existing task instead of creating a new one
        return ChatTaskResponse(
            id=existing_task.id,
            conversation_id=existing_task.conversation_id,
            status=existing_task.status,
            user_message=existing_task.user_message,
            streaming_state=existing_task.streaming_state,
            response_content=existing_task.response_content,
            error_message=existing_task.error_message,
            created_at=existing_task.created_at,
            started_at=existing_task.started_at,
            completed_at=existing_task.completed_at,
            last_update_at=existing_task.last_update_at,
        )

    # Add user message to conversation first
    await repository.add_message(conversation_id, "user", user_message)
    schedule_title_generation(conversation_id, user_message)

    # Start background task
    task_id = await background_task_service.start_task_async(
        conversation_id, user_message
    )

    # Get the created task
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create background task")

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


@router.get(
    "/conversations/{conversation_id}/task", response_model=Optional[ChatTaskResponse]
)
async def get_conversation_active_task(
    conversation_id: str, user: User = Depends(get_current_user)
):
    """
    Get the active (pending/running) task for a conversation, if any.
    Returns null if no active task.
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    task = await repository.get_active_task_for_conversation(conversation_id)
    if not task:
        return None

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


@router.get(
    "/conversations/{conversation_id}/interrupted-task",
    response_model=Optional[ChatTaskResponse],
)
async def get_conversation_interrupted_task(
    conversation_id: str, user: User = Depends(get_current_user)
):
    """
    Get the last interrupted task for a conversation, if any.
    Used to show Continue button after server restart.
    Returns null if no interrupted task.
    """
    # Check access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    task = await repository.get_last_interrupted_task_for_conversation(conversation_id)
    if not task:
        return None

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
    user: User = Depends(get_current_user),
):
    """
    Stream updates for a chat task via SSE.
    """
    # Verify task exists and user has access
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check conversation access
    has_access = await repository.check_conversation_access(
        task.conversation_id, user.id, is_admin=(user.role == "admin")
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

        finally:
            task_event_bus.unsubscribe(task_id, queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}/events")
async def conversation_events(
    conversation_id: str,
    user: User = Depends(get_current_user),
):
    """
    Subscribe to conversation events (e.g. title updates).
    """
    # Check conversation access
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
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

    return SchemaIndexJobResponse(
        id=job.id,
        tool_config_id=job.tool_config_id,
        status=job.status,
        index_name=job.index_name,
        progress_percent=job.progress_percent,
        total_tables=job.total_tables,
        processed_tables=job.processed_tables,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


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

    return SchemaIndexJobResponse(
        id=new_job.id,
        tool_config_id=new_job.tool_config_id,
        status=new_job.status,
        index_name=new_job.index_name,
        progress_percent=new_job.progress_percent,
        total_tables=new_job.total_tables,
        processed_tables=new_job.processed_tables,
        total_chunks=new_job.total_chunks,
        processed_chunks=new_job.processed_chunks,
        error_message=new_job.error_message,
        cancel_requested=new_job.cancel_requested,
        created_at=new_job.created_at,
        started_at=new_job.started_at,
        completed_at=new_job.completed_at,
    )


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
