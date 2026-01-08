"""
Indexer API routes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import httpx
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.model_limits import get_context_limit
from ragtime.core.security import get_current_user, require_admin
from ragtime.core.sql_utils import MssqlConnectionError, mssql_connect
from ragtime.core.validation import require_valid_embedding_provider
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.models import (
    AnalyzeIndexRequest,
    AppSettings,
    CheckRepoVisibilityRequest,
    CreateIndexRequest,
    FetchBranchesRequest,
    FetchBranchesResponse,
    IndexAnalysisResult,
    IndexConfig,
    IndexInfo,
    IndexJobResponse,
    IndexStatus,
    PdmIndexJobResponse,
    RepoVisibilityResponse,
    SchemaIndexJobResponse,
    TriggerPdmIndexRequest,
    TriggerSchemaIndexRequest,
    UpdateSettingsRequest,
)
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import schema_indexer
from ragtime.indexer.service import indexer
from ragtime.indexer.utils import safe_tool_name

if TYPE_CHECKING:
    from prisma.models import User

logger = get_logger(__name__)

router = APIRouter(prefix="/indexes", tags=["Indexer"])

# Path to static files - React build lives under dist
STATIC_DIR = Path(__file__).parent / "static"
DIST_DIR = STATIC_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"

# Check if running in development mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


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
    enable_ocr: bool = Form(
        default=False,
        description="Enable OCR to extract text from images",
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
            enable_ocr=enable_ocr,
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
    from ragtime.core.git import check_repo_visibility as git_check_visibility

    # Get stored token if index_name provided
    stored_token = None
    if request.index_name:
        try:
            metadata = await repository.get_index_metadata(request.index_name)
            stored_token = getattr(metadata, "gitToken", None) if metadata else None
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
    from ragtime.core.git import fetch_branches as git_fetch_branches

    # Try to get stored token from existing index if index_name provided
    token = request.git_token
    if not token and request.index_name:
        try:
            metadata = await repository.get_index_metadata(request.index_name)
            token = getattr(metadata, "gitToken", None) if metadata else None
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


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, _user: User = Depends(require_admin)):
    """Delete a job record (must be completed, failed, or cancelled). Admin only."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [IndexStatus.PENDING, IndexStatus.PROCESSING]:
        raise HTTPException(
            status_code=400, detail="Cannot delete active job. Cancel it first."
        )

    await repository.delete_job(job_id)
    return {"message": f"Job '{job_id}' deleted successfully"}


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
    enable_ocr: bool = Form(
        default=False,
        description="Enable OCR to extract text from images (slower but captures text in screenshots, scanned docs, etc.)",
    ),
    _user: User = Depends(require_admin),
    _: None = Depends(require_valid_embedding_provider),
):
    """
    Upload an archive file and create a FAISS index from it. Admin only.

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
        enable_ocr=enable_ocr,
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
    Create a FAISS index from a git repository. Admin only.

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

    # Use provided token, or fall back to stored token
    git_token = request.git_token or getattr(metadata, "gitToken", None)

    # Get config from snapshot or use defaults
    config_snapshot = getattr(metadata, "configSnapshot", None)
    config_data: dict[str, Any] = (
        config_snapshot if isinstance(config_snapshot, dict) else {}
    )
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
        enable_ocr=config_data.get("enable_ocr", False),
        git_clone_timeout_minutes=config_data.get("git_clone_timeout_minutes", 5),
        git_history_depth=config_data.get("git_history_depth", 1),
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
    else:
        raise HTTPException(
            status_code=400,
            detail="Only git-based jobs can be retried. For upload-based indexes, please re-upload the file.",
        )


@router.delete("/{name}")
async def delete_index(name: str, _user: User = Depends(require_admin)):
    """Delete an index by name. Admin only."""
    if await indexer.delete_index(name):
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
    """Toggle an index's enabled status for RAG context. Admin only."""
    success = await repository.set_index_enabled(name, request.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
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
    enable_ocr: Optional[bool] = Field(
        default=None, description="Enable OCR for images"
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
        "enable_ocr",
        "git_clone_timeout_minutes",
        "git_history_depth",
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
    if request.enable_ocr is not None:
        new_config["enable_ocr"] = request.enable_ocr
    if request.git_clone_timeout_minutes is not None:
        new_config["git_clone_timeout_minutes"] = request.git_clone_timeout_minutes
    if request.git_history_depth is not None:
        new_config["git_history_depth"] = request.git_history_depth

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


@router.get("/{name}/download")
async def download_index(name: str, _user: User = Depends(require_admin)):
    """Download FAISS index files as a zip archive. Admin only.

    Returns a zip file containing the index.faiss and index.pkl files.
    """
    import io
    import zipfile

    from starlette.responses import StreamingResponse

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


# -----------------------------------------------------------------------------
# Settings Endpoints (Admin only)
# -----------------------------------------------------------------------------


@router.get("/settings", response_model=AppSettings, tags=["Settings"])
async def get_settings(_user: User = Depends(require_admin)):
    """Get current application settings. Admin only."""
    return await repository.get_settings()


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
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.mcp.server import notify_tools_changed
    from ragtime.rag import rag

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

    # Notify MCP clients that tools may have changed (e.g., aggregate_search toggle)
    notify_tools_changed()

    return UpdateSettingsResponse(settings=result, embedding_warning=embedding_warning)


from ragtime.indexer.models import EmbeddingStatus


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

from ragtime.indexer.models import (
    CreateToolConfigRequest,
    MssqlDiscoverRequest,
    MssqlDiscoverResponse,
    PdmDiscoverRequest,
    PdmDiscoverResponse,
    PostgresDiscoverRequest,
    PostgresDiscoverResponse,
    ToolConfig,
    ToolTestRequest,
    ToolType,
    UpdateToolConfigRequest,
)


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
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.mcp.server import notify_tools_changed
    from ragtime.rag import rag

    config = ToolConfig(
        name=request.name,
        tool_type=request.tool_type,
        description=request.description,
        connection_config=request.connection_config,
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
    import asyncio
    from datetime import datetime, timezone

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
    """Update an existing tool configuration. Admin only."""
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.mcp.server import notify_tools_changed
    from ragtime.rag import rag

    updates = request.model_dump(exclude_unset=True)
    config = await repository.update_tool_config(tool_id, updates)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Reinitialize RAG agent to pick up the tool changes
    invalidate_settings_cache()
    await rag.initialize()

    # Notify MCP clients that tools have changed (e.g., renamed)
    notify_tools_changed()

    return config


@router.delete("/tools/{tool_id}", tags=["Tools"])
async def delete_tool_config(tool_id: str, _user: User = Depends(require_admin)):
    """
    Delete a tool configuration. Admin only.

    For Odoo tools, also disconnects from the Docker network if no other
    tools are using it.
    For filesystem indexer tools, unmounts any SMB/NFS shares.
    """
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.mcp.server import notify_tools_changed
    from ragtime.rag import rag

    # Get the tool config before deleting to check for network cleanup
    tool = await repository.get_tool_config(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Check if this is an Odoo tool with a docker network
    docker_network = None
    if tool.tool_type == "odoo_shell" and tool.connection_config:
        docker_network = tool.connection_config.get("docker_network")

    # Check if this is a filesystem indexer tool (needs mount cleanup)
    is_filesystem_tool = tool.tool_type == "filesystem_indexer"

    # Delete the tool
    success = await repository.delete_tool_config(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Cleanup: unmount filesystem for filesystem indexer tools
    if is_filesystem_tool:
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
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.mcp.server import notify_tools_changed
    from ragtime.rag import rag

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
    import asyncio

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
    """
    import asyncio
    import subprocess

    try:
        # Connect to 'postgres' system database to list all databases
        cmd = [
            "psql",
            "-h",
            request.host,
            "-p",
            str(request.port),
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
            timeout=10.0,
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
            success=False, databases=[], error="Connection timed out after 10 seconds"
        )
    except FileNotFoundError:
        return PostgresDiscoverResponse(
            success=False, databases=[], error="psql command not found"
        )
    except Exception as e:
        return PostgresDiscoverResponse(
            success=False, databases=[], error=f"Discovery failed: {str(e)}"
        )


@router.post(
    "/tools/mssql/discover", response_model=MssqlDiscoverResponse, tags=["Tools"]
)
async def discover_mssql_databases(
    request: MssqlDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover available databases on an MSSQL server. Admin only.
    Connects to the server and lists all accessible databases.
    """
    import asyncio

    def discover_databases() -> tuple[bool, list[str], str | None]:
        try:
            with mssql_connect(
                host=request.host,
                port=request.port,
                user=request.user,
                password=request.password,
                database="master",  # Connect to master to list databases
                login_timeout=10,
                timeout=10,
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
        success, databases, error = await asyncio.to_thread(discover_databases)
        return MssqlDiscoverResponse(success=success, databases=databases, error=error)
    except asyncio.TimeoutError:
        return MssqlDiscoverResponse(
            success=False, databases=[], error="Connection timed out"
        )
    except Exception as e:
        return MssqlDiscoverResponse(
            success=False, databases=[], error=f"Discovery failed: {str(e)}"
        )


@router.post("/tools/pdm/discover", response_model=PdmDiscoverResponse, tags=["Tools"])
async def discover_pdm_schema(
    request: PdmDiscoverRequest, _user: User = Depends(require_admin)
):
    """
    Discover PDM schema metadata from a SolidWorks PDM database. Admin only.
    Returns available file extensions and variable names.
    """
    import asyncio

    def discover_schema() -> tuple[bool, list[str], list[str], int, str | None]:
        """Returns (success, file_extensions, variable_names, doc_count, error)."""
        try:
            with mssql_connect(
                host=request.host,
                port=request.port,
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

    try:
        success, extensions, variables, doc_count, error = await asyncio.to_thread(
            discover_schema
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
        return PdmDiscoverResponse(success=False, error=f"Discovery failed: {str(e)}")


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
    import subprocess
    import tempfile

    try:
        # Create temp directory for key generation
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
                    passphrase,  # Passphrase (empty string = no passphrase)
                    "-C",
                    comment,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            if process.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate keypair: {process.stderr}",
                )

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
from ragtime.indexer.models import FilesystemConnectionConfig


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
async def test_saved_tool_connection(tool_id: str):
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


async def _heartbeat_postgres(config: dict) -> ToolTestResponse:
    """Quick PostgreSQL heartbeat check."""
    import asyncio
    import subprocess

    host = config.get("host", "")
    port = config.get("port", 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")

    try:
        if host:
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


async def _heartbeat_mssql(config: dict) -> ToolTestResponse:
    """Quick MSSQL heartbeat check."""
    from ragtime.tools.mssql import test_mssql_connection

    host = config.get("host", "")
    port = config.get("port", 1433)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")

    if not host or not user or not database:
        return ToolTestResponse(success=False, message="MSSQL not configured")

    success, message, _ = await test_mssql_connection(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        timeout=5,
    )

    return ToolTestResponse(
        success=success,
        message="OK" if success else message[:100],
    )


async def _heartbeat_odoo(config: dict) -> ToolTestResponse:
    """Quick Odoo container/SSH heartbeat check."""
    import asyncio
    import subprocess

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


async def _heartbeat_ssh(config: dict) -> ToolTestResponse:
    """
    Quick SSH heartbeat check using socket-level port check.

    Uses asyncio sockets instead of paramiko to avoid blocking threads.
    This only verifies the SSH port is reachable, not full authentication.
    Full connection tests are done via the 'Test' button.
    """
    import asyncio

    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")

    if not host or not user:
        return ToolTestResponse(success=False, message="SSH not configured")

    try:
        # Use asyncio socket to check if SSH port is reachable
        # This is non-blocking and respects asyncio timeout
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=3.0
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
    import asyncio

    host = config.get("host", "")
    port = config.get("port", 1433)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")

    if not all([host, user, password, database]):
        return ToolTestResponse(success=False, message="PDM connection not configured")

    def check_connection() -> tuple[bool, str]:
        try:
            with mssql_connect(
                host=host,
                port=port,
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


async def _test_postgres_connection(config: dict) -> ToolTestResponse:
    """Test PostgreSQL connection."""
    import asyncio
    import subprocess

    host = config.get("host", "")
    port = config.get("port", 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")

    try:
        if host:
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
                success=False, message="Either host or container must be specified"
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
    """Test MSSQL/SQL Server connection."""
    from ragtime.tools.mssql import test_mssql_connection

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


async def _test_odoo_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection (Docker or SSH mode)."""
    mode = config.get("mode", "docker")

    if mode == "ssh":
        return await _test_odoo_ssh_connection(config)
    else:
        return await _test_odoo_docker_connection(config)


async def _test_odoo_ssh_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection via SSH using Paramiko."""
    import asyncio

    from ragtime.core.ssh import SSHConfig, execute_ssh_command, test_ssh_connection

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
    import asyncio
    import subprocess

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
    import asyncio

    from ragtime.core.ssh import SSHConfig, test_ssh_connection

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
async def discover_docker_resources():
    """
    Discover Docker networks and containers for tool configuration.

    Returns available networks, running containers, and which containers have Odoo.
    Also detects the current ragtime container's network.
    """
    import asyncio
    import json
    import subprocess

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
async def connect_to_network(network_name: str):
    """
    Connect the ragtime container to a Docker network.

    This enables container-to-container communication with services on that network.
    """
    import asyncio
    import subprocess

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
async def disconnect_from_network(network_name: str):
    """
    Disconnect the ragtime container from a Docker network.

    Used for cleanup when removing tools that required network access.
    """
    import asyncio
    import subprocess

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
    import asyncio
    import subprocess

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
    import asyncio
    import subprocess

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
    import asyncio
    import socket

    if not host:
        return SMBDiscoveryResponse(success=False, error="Host is required")

    def _discover_smb() -> SMBDiscoveryResponse:
        try:
            from smb.SMBConnection import (
                SMBConnection,
            )  # type: ignore[import-not-found,import-untyped]

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
    import asyncio
    import subprocess

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
    import asyncio
    import socket

    if not host or not share:
        return BrowseResponse(
            path=path, entries=[], error="Host and share are required"
        )

    def _browse_smb() -> BrowseResponse:
        try:
            from smb.SMBConnection import (
                SMBConnection,
            )  # type: ignore[import-not-found,import-untyped]

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

from ragtime.indexer.models import (
    FilesystemAnalysisJobResponse,
    FilesystemIndexJobResponse,
    TriggerFilesystemIndexRequest,
)


class FilesystemIndexStatsResponse(BaseModel):
    """Statistics for a filesystem index."""

    index_name: str
    embedding_count: int
    file_count: int
    last_indexed: Optional[str] = None


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

    stats = await filesystem_indexer.get_index_stats(fs_config.index_name)
    return FilesystemIndexStatsResponse(
        index_name=stats["index_name"],
        embedding_count=stats["embedding_count"],
        file_count=stats["file_count"],
        last_indexed=(
            stats["last_indexed"].isoformat() if stats["last_indexed"] else None
        ),
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

    deleted_count = await filesystem_indexer.delete_index(fs_config.index_name)
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


# -----------------------------------------------------------------------------
# Ollama Connection Testing
# -----------------------------------------------------------------------------


class OllamaTestRequest(BaseModel):
    """Request to test Ollama connection."""

    protocol: str = Field(default="http", description="Protocol: 'http' or 'https'")
    host: str = Field(default="localhost", description="Ollama server hostname or IP")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")


class OllamaModel(BaseModel):
    """Information about an available Ollama model."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None


class OllamaTestResponse(BaseModel):
    """Response from Ollama connection test."""

    success: bool
    message: str
    models: List[OllamaModel] = []
    base_url: str = ""


@router.post("/ollama/test", response_model=OllamaTestResponse, tags=["Settings"])
async def test_ollama_connection(request: OllamaTestRequest):
    """
    Test connection to an Ollama server and retrieve available models.

    Returns a list of available embedding models if the connection is successful.
    """
    base_url = f"{request.protocol}://{request.host}:{request.port}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First check if the server is reachable
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = []

            # Parse the models list from Ollama API response
            for model in data.get("models", []):
                models.append(
                    OllamaModel(
                        name=model.get("name", ""),
                        modified_at=model.get("modified_at"),
                        size=model.get("size"),
                    )
                )

            return OllamaTestResponse(
                success=True,
                message=f"Connected successfully. Found {len(models)} model(s).",
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
from ragtime.core.embedding_models import (
    OPENAI_EMBEDDING_PRIORITY,
    get_embedding_models,
)

OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@router.post(
    "/embedding/models", response_model=EmbeddingModelsResponse, tags=["Settings"]
)
async def fetch_embedding_models(request: EmbeddingModelsRequest):
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


class AvailableModelsResponse(BaseModel):
    """Response with all available models from configured providers."""

    models: List[AvailableModel] = []
    default_model: Optional[str] = None
    current_model: Optional[str] = None  # Currently selected model in settings
    allowed_models: List[str] = []  # List of allowed model IDs (for settings UI)


# Sensible default models for each provider
OPENAI_DEFAULT_MODEL = "gpt-4o"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Models to prioritize in the list (shown first)
OPENAI_PRIORITY_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]
ANTHROPIC_PRIORITY_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


@router.post("/llm/models", response_model=LLMModelsResponse, tags=["Settings"])
async def fetch_llm_models(request: LLMModelsRequest):
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

            # Filter for chat-capable models (gpt-* models, excluding embedding/audio/etc)
            for model in data.get("data", []):
                model_id = model.get("id", "")
                # Include GPT models suitable for chat (exclude embeddings, whisper, tts, dall-e, etc.)
                if model_id.startswith("gpt-") and not any(
                    x in model_id for x in ["instruct", "vision", "realtime", "audio"]
                ):
                    models.append(
                        LLMModel(
                            id=model_id, name=model_id, created=model.get("created")
                        )
                    )

            # Sort: priority models first, then alphabetically
            def sort_key(m: LLMModel) -> tuple:
                try:
                    priority_idx = OPENAI_PRIORITY_MODELS.index(m.id)
                except ValueError:
                    priority_idx = 999
                return (priority_idx, m.id)

            models.sort(key=sort_key)

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} chat model(s).",
                models=models,
                default_model=(
                    OPENAI_DEFAULT_MODEL
                    if any(m.id == OPENAI_DEFAULT_MODEL for m in models)
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
                models.append(LLMModel(id=model_id, name=display_name, created=None))

            # Sort: priority models first, then alphabetically
            def sort_key(m: LLMModel) -> tuple:
                try:
                    priority_idx = ANTHROPIC_PRIORITY_MODELS.index(m.id)
                except ValueError:
                    priority_idx = 999
                return (priority_idx, m.id)

            models.sort(key=sort_key)

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} model(s).",
                models=models,
                default_model=(
                    ANTHROPIC_DEFAULT_MODEL
                    if any(m.id == ANTHROPIC_DEFAULT_MODEL for m in models)
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


# =============================================================================
# Conversation/Chat Endpoints
# =============================================================================

from ragtime.indexer.models import (
    ChatMessage,
    Conversation,
    ConversationResponse,
    CreateConversationRequest,
    SendMessageRequest,
)


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
                        )
                    )
                if not default_model and result.default_model:
                    default_model = result.default_model
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

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

    return AvailableModelsResponse(
        models=all_models, default_model=default_model, current_model=current_model
    )


@router.get("/chat/all-models", response_model=AvailableModelsResponse, tags=["Chat"])
async def get_all_chat_models():
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
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

    # Get currently allowed models from settings
    allowed_models = app_settings.allowed_chat_models or []

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


@router.post(
    "/conversations/{conversation_id}/clear", response_model=ConversationResponse
)
async def clear_conversation(
    conversation_id: str, user: User = Depends(get_current_user)
):
    """Clear all messages in a conversation. Users can only clear their own conversations."""
    has_access = await repository.check_conversation_access(
        conversation_id, user.id, is_admin=(user.role == "admin")
    )
    if not has_access:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv = await repository.clear_conversation(conversation_id)
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
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    from ragtime.rag import rag

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

    # Auto-generate title from first user message if still "New Chat"
    if conv.title == "New Chat" and len(conv.messages) >= 2:
        first_msg = conv.messages[0].content[:50]
        new_title = first_msg + ("..." if len(conv.messages[0].content) > 50 else "")
        conv = (
            await repository.update_conversation_title(conversation_id, new_title)
            or conv
        )

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
    import json
    import time

    from fastapi.responses import StreamingResponse
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    from ragtime.rag import rag

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
            async for event in rag.process_query_stream(user_message, chat_history):
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

            # Auto-generate title if needed
            if (
                updated_conv
                and updated_conv.title == "New Chat"
                and len(updated_conv.messages) >= 2
            ):
                first_msg = updated_conv.messages[0].content[:50]
                new_title = first_msg + (
                    "..." if len(updated_conv.messages[0].content) > 50 else ""
                )
                await repository.update_conversation_title(conversation_id, new_title)

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
# Background Chat Task Endpoints
# =============================================================================

from ragtime.indexer.models import ChatTaskResponse, ChatTaskStatus


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
    from ragtime.indexer.background_tasks import background_task_service
    from ragtime.rag import rag

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


@router.post("/tasks/{task_id}/cancel", response_model=ChatTaskResponse)
async def cancel_chat_task(task_id: str):
    """
    Cancel a running chat task.
    """
    from ragtime.indexer.background_tasks import background_task_service

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

    # Only allow for SQL database tools
    if tool_config.tool_type not in (ToolType.POSTGRES, ToolType.MSSQL):
        raise HTTPException(
            status_code=400,
            detail=f"Schema indexing is only available for postgres and mssql tools, got '{tool_config.tool_type.value}'",
        )

    # Check that schema indexing is enabled
    conn_config = tool_config.connection_config or {}
    if not conn_config.get("schema_index_enabled", False):
        raise HTTPException(
            status_code=400,
            detail="Schema indexing is not enabled for this tool. Enable it in the tool configuration.",
        )

    # Validate pgvector is available
    if (
        not await schema_indexer._ensure_pgvector()
    ):  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # type: ignore[attr-defined]
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


@router.get(
    "/tools/{tool_id}/schema/stats",
    response_model=SchemaStatsResponse,
    tags=["Schema Indexer"],
)
async def get_schema_index_stats(tool_id: str, _user: User = Depends(require_admin)):
    """Get schema index statistics for a tool. Admin only."""
    # Get embedding count
    embedding_count = await schema_indexer.get_embedding_count(tool_id)

    # Get last indexed timestamp and hash from tool config
    tool_config = await repository.get_tool_config(tool_id)
    last_indexed_at = None
    schema_hash = None

    if tool_config:
        conn_config = tool_config.connection_config or {}
        last_indexed_at = conn_config.get("last_schema_indexed_at")
        schema_hash = conn_config.get("schema_hash")

    return SchemaStatsResponse(
        embedding_count=embedding_count,
        last_indexed_at=last_indexed_at,
        schema_hash=schema_hash,
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
    if (
        not await pdm_indexer._ensure_pgvector()
    ):  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # type: ignore[attr-defined]
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
