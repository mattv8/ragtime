"""
FAISS Indexer Service - Creates and manages vector indexes.

Job Recovery: Jobs are persisted to the database before processing starts.
If the server restarts mid-job (e.g., hot-reload), jobs in 'pending' or
'processing' state are automatically resumed on startup.
"""

import asyncio
import fnmatch
import json
import os
import re
import shutil
import subprocess
import tarfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from ragtime.config import settings
from ragtime.core.app_settings import invalidate_settings_cache
from ragtime.core.file_constants import (
    BINARY_EXTENSIONS,
    MINIFIED_PATTERNS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
)
from ragtime.core.logging import get_logger
from ragtime.indexer.document_parser import OCR_EXTENSIONS, extract_text_from_file
from ragtime.indexer.llm_exclusions import get_smart_exclusion_suggestions
from ragtime.indexer.models import (
    AnalyzeIndexRequest,
    AppSettings,
    FileTypeStats,
    IndexAnalysisResult,
    IndexConfig,
    IndexInfo,
    IndexJob,
    IndexStatus,
)
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

# Persistent storage for uploaded files
UPLOAD_TMP_DIR = Path(settings.index_data_path) / "_tmp"


# =============================================================================
# Git Provider Token Authentication
# =============================================================================
# Supported platforms and their HTTPS clone URL token formats:
#
# GitHub (github.com):
#   - Token prefixes: ghp_, gho_, ghu_, ghs_, ghr_, github_pat_
#   - Format: https://x-access-token:{token}@github.com/owner/repo.git
#   - Docs: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
#
# GitLab (gitlab.com and self-hosted):
#   - Token prefixes: glpat-, glptt-, gldt-, glsoat-
#   - Format: https://oauth2:{token}@{host}/owner/repo.git
#   - Note: Username can be any non-empty string; "oauth2" is conventional
#   - Docs: https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html
#
# Bitbucket (bitbucket.org):
#   - Token prefixes: (no standard prefix - API tokens)
#   - Format: https://x-bitbucket-api-token-auth:{token}@bitbucket.org/workspace/repo.git
#   - Docs: https://support.atlassian.com/bitbucket-cloud/docs/using-api-tokens/
#
# Generic (Gitea, Gogs, Forgejo, etc.):
#   - Format: https://{token}@{host}/owner/repo.git
#   - Most self-hosted Git servers accept token as username
# =============================================================================

# Token prefix patterns for provider detection (case-sensitive)
_GITHUB_TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")
_GITLAB_TOKEN_PREFIXES = ("glpat-", "glptt-", "gldt-", "glsoat-")


def _build_authenticated_git_url(git_url: str, token: Optional[str] = None) -> str:
    """
    Build a Git clone URL with embedded token authentication.

    Detects the Git provider from the URL hostname or token prefix and applies
    the appropriate authentication format.

    Args:
        git_url: The original HTTPS Git URL (e.g., https://github.com/owner/repo.git)
        token: Optional personal access token for authentication

    Returns:
        The URL with embedded credentials, or the original URL if no token provided
    """
    if not token:
        return git_url

    # Parse URL: protocol://host/path
    match = re.match(r"(https?://)([^/]+)(/.*)$", git_url)
    if not match:
        # Not a standard HTTP(S) URL, return as-is
        return git_url

    protocol, host, path = match.groups()
    host_lower = host.lower()

    # Priority 1: Detect by hostname (most reliable)
    if "github.com" in host_lower:
        # GitHub: https://x-access-token:{token}@github.com/...
        return f"{protocol}x-access-token:{token}@{host}{path}"

    if "gitlab" in host_lower:
        # GitLab (hosted or self-hosted with "gitlab" in hostname)
        return f"{protocol}oauth2:{token}@{host}{path}"

    if "bitbucket.org" in host_lower:
        # Bitbucket Cloud: https://x-bitbucket-api-token-auth:{token}@bitbucket.org/...
        return f"{protocol}x-bitbucket-api-token-auth:{token}@{host}{path}"

    # Priority 2: Detect by token prefix (for self-hosted instances)
    if token.startswith(_GITHUB_TOKEN_PREFIXES):
        # GitHub Enterprise or similar using GitHub token format
        return f"{protocol}x-access-token:{token}@{host}{path}"

    if token.startswith(_GITLAB_TOKEN_PREFIXES):
        # Self-hosted GitLab (e.g., git.example.com) detected by token prefix
        return f"{protocol}oauth2:{token}@{host}{path}"

    # Priority 3: Generic fallback
    # Most Git servers (Gitea, Gogs, Forgejo, etc.) accept token as username
    return f"{protocol}{token}@{host}{path}"


async def generate_index_description(
    index_name: str,
    documents: List,
    source_type: str,
    source: Optional[str] = None,
) -> str:
    """
    Auto-generate a description for an index using the LLM.

    Samples file paths and content to create a concise description
    that helps the AI understand what knowledge is available.
    """
    try:
        from langchain_openai import ChatOpenAI

        from ragtime.core.app_settings import get_app_settings

        app_settings = await get_app_settings()
        api_key = app_settings.get("openai_api_key", "")

        if not api_key:
            logger.debug("No OpenAI API key - skipping auto-description generation")
            return ""

        # Sample file paths for context (unique paths only)
        file_paths = list(
            set(doc.metadata.get("source", "unknown") for doc in documents[:100])
        )[:20]

        # Sample some content snippets
        content_samples = [doc.page_content[:500] for doc in documents[:5]]

        prompt = f"""Analyze this indexed codebase and write a brief description (1-2 sentences) for an AI assistant to understand what knowledge is available.

Index name: {index_name}
Source type: {source_type}
Source: {source or 'uploaded archive'}

Sample file paths:
{chr(10).join(f'- {p}' for p in file_paths)}

Sample content snippets:
{chr(10).join(f'---{chr(10)}{s}{chr(10)}' for s in content_samples)}

Write a concise description focusing on:
- What type of project/codebase this is
- Key technologies, frameworks, or domains covered
- What questions this index can help answer

Description:"""

        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use cheaper model for descriptions
            temperature=0.3,
            api_key=api_key,
        )

        response = await llm.ainvoke(prompt)
        content = response.content
        description = content.strip() if isinstance(content, str) else str(content)
        logger.info(
            f"Auto-generated description for {index_name}: {description[:100]}..."
        )
        return description

    except Exception as e:
        logger.warning(f"Failed to auto-generate description: {e}")
        return ""


class IndexerService:
    """Service for creating and managing FAISS indexes.

    Features job persistence and recovery: if the server restarts during
    processing, interrupted jobs are automatically resumed on startup.
    """

    def __init__(self, index_base_path: str = "/app/data/faiss_index"):
        self.index_base_path = Path(index_base_path)
        self.index_base_path.mkdir(parents=True, exist_ok=True)
        # Ensure tmp directory exists for upload persistence
        UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
        # In-memory cache for active jobs (transient state during processing)
        self._active_jobs: Dict[str, IndexJob] = {}
        # Cancellation flags for cooperative cancellation
        self._cancellation_flags: Dict[str, bool] = {}

    async def _reinitialize_rag_components(self) -> None:
        """
        Reinitialize RAG components to load newly created indexes.

        Called after successful index creation to make the new index
        immediately available for search without requiring a server restart.

        Note: The import of `rag` is done inside this method rather than at
        module level to avoid potential circular import issues, as
        rag.components imports from indexer.repository. Since this method
        is called only once per index creation, the performance impact is
        negligible.
        """
        try:
            from ragtime.rag.components import rag

            logger.info("Reinitializing RAG components to load new index")
            invalidate_settings_cache()
            await rag.initialize()
            logger.info("RAG components reinitialized successfully")
        except Exception as e:
            # Log but don't fail the indexing job if RAG reinitialization fails
            logger.warning(f"Failed to reinitialize RAG components: {e}")

    async def _maybe_reinitialize_rag(self, job: IndexJob) -> None:
        """
        Conditionally reinitialize RAG components for completed jobs.

        This is called in finally blocks after job status is persisted to ensure:
        1. Job status is saved before reinitialization
        2. Reinitialize errors don't affect job completion status
        3. New index is loaded even if there were warnings during indexing

        Args:
            job: The index job to check
        """
        if job.status == IndexStatus.COMPLETED:
            await self._reinitialize_rag_components()

    async def recover_interrupted_jobs(self) -> int:
        """
        Recover jobs that were interrupted by a server restart.

        Called during application startup. Finds jobs in 'pending' or 'processing'
        state and resumes them.

        Returns:
            Number of jobs recovered
        """
        jobs = await repository.list_jobs()
        interrupted = [
            j for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
        ]

        if not interrupted:
            return 0

        logger.info(f"Found {len(interrupted)} interrupted job(s) to recover")

        recovered = 0
        for job in interrupted:
            try:
                await self._resume_job(job)
                recovered += 1
            except Exception as e:
                logger.error(f"Failed to recover job {job.id}: {e}")
                job.status = IndexStatus.FAILED
                job.error_message = f"Recovery failed: {e}"
                job.completed_at = datetime.utcnow()
                await repository.update_job(job)

        # Clean up orphaned tmp directories (from deleted jobs)
        await self._cleanup_orphaned_tmp_dirs()

        return recovered

    async def _cleanup_orphaned_tmp_dirs(self) -> None:
        """Remove tmp directories for jobs that no longer exist."""
        if not UPLOAD_TMP_DIR.exists():
            return

        jobs = await repository.list_jobs()
        active_job_ids = {
            j.id
            for j in jobs
            if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
        }

        for tmp_path in UPLOAD_TMP_DIR.iterdir():
            if tmp_path.is_dir() and tmp_path.name not in active_job_ids:
                logger.info(f"Cleaning orphaned tmp directory: {tmp_path.name}")
                shutil.rmtree(tmp_path, ignore_errors=True)

    async def discover_orphan_indexes(self) -> int:
        """
        Discover FAISS indexes on disk that have no database metadata.

        Reads the index.pkl file to extract document count and creates
        database metadata entries for discovered indexes.

        Called during application startup.

        Returns:
            Number of indexes discovered and registered
        """
        import pickle

        # Get existing metadata from database
        db_metadata = await repository.list_index_metadata()
        known_names = {m.name for m in db_metadata}

        discovered = 0

        for path in self.index_base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Skip if already known
                if path.name in known_names:
                    continue

                # Check if it's a valid FAISS index
                faiss_file = path / "index.faiss"
                pkl_file = path / "index.pkl"

                if not (faiss_file.exists() and pkl_file.exists()):
                    continue

                logger.info(f"Discovered orphan index: {path.name}")

                # Try to extract document count from pickle file
                doc_count = 0
                chunk_count = 0
                try:
                    with open(pkl_file, "rb") as pkl_f:
                        data = pickle.load(pkl_f)

                    # FAISS pickle format: (InMemoryDocstore, index_to_docstore_id_dict)
                    if isinstance(data, tuple) and len(data) >= 2:
                        docstore, idx_to_id = data[0], data[1]
                        if hasattr(docstore, "_dict"):
                            doc_count = len(
                                docstore._dict
                            )  # noqa: SLF001  # type: ignore[union-attr]
                            chunk_count = doc_count  # chunks == documents for FAISS
                        elif isinstance(idx_to_id, dict):
                            doc_count = len(idx_to_id)
                            chunk_count = doc_count

                    logger.info(f"  Extracted {doc_count} documents from {path.name}")
                except Exception as e:
                    logger.warning(
                        f"  Could not extract doc count from {path.name}: {e}"
                    )

                # Calculate size
                size_bytes = sum(
                    p.stat().st_size for p in path.rglob("*") if p.is_file()
                )

                # Check for legacy metadata file
                legacy_meta: dict[str, Any] = {}
                meta_file = path / ".metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, encoding="utf-8") as meta_f:
                            legacy_meta = json.load(meta_f)
                    except Exception:
                        pass

                # Create database metadata
                await repository.upsert_index_metadata(
                    name=path.name,
                    path=str(path),
                    document_count=doc_count,
                    chunk_count=chunk_count,
                    size_bytes=size_bytes,
                    source_type=legacy_meta.get("source_type", "unknown"),
                    source=legacy_meta.get("source"),
                    config_snapshot=legacy_meta.get("config"),
                    description=legacy_meta.get("description", ""),
                )

                discovered += 1

        if discovered > 0:
            logger.info(f"Registered {discovered} orphan index(es) in database")

        return discovered

    async def _resume_job(self, job: IndexJob) -> None:
        """Resume an interrupted job based on its source type."""
        logger.info(f"Resuming job {job.id} ({job.source_type}): {job.name}")

        # Reset progress - we'll reprocess from scratch for simplicity
        job.status = IndexStatus.PENDING
        job.processed_files = 0
        job.processed_chunks = 0
        job.error_message = None
        await repository.update_job(job)

        # Cache for active processing
        self._active_jobs[job.id] = job

        if job.source_type == "git":
            # Check if the cloned repo still exists in _tmp from a previous attempt
            tmp_path = UPLOAD_TMP_DIR / job.id
            clone_dir = tmp_path / "repo"
            clone_complete_marker = tmp_path / ".clone_complete"

            if clone_dir.exists() and clone_complete_marker.exists():
                # Clone completed previously, skip directly to indexing
                logger.info(f"Found existing clone for job {job.id}, resuming indexing")
                asyncio.create_task(self._process_git(job, skip_clone=True))
            else:
                # Need to re-clone - clean up any partial clone first
                if tmp_path.exists():
                    shutil.rmtree(tmp_path, ignore_errors=True)
                asyncio.create_task(self._process_git(job))
        elif job.source_type == "upload":
            # Check if tmp file still exists
            tmp_path = UPLOAD_TMP_DIR / job.id
            if tmp_path.exists():
                # Find the archive file (not directories like 'extracted')
                archive_files = [f for f in tmp_path.iterdir() if f.is_file()]
                if archive_files:
                    archive_path = archive_files[0]

                    # Clean any previous extraction attempt
                    extracted_dir = tmp_path / "extracted"
                    if extracted_dir.exists():
                        shutil.rmtree(extracted_dir, ignore_errors=True)

                    temp_dir = tmp_path  # Use tmp dir for extraction
                    asyncio.create_task(
                        self._process_upload(job, archive_path, temp_dir)
                    )
                    return

            # No tmp file - mark as failed
            job.status = IndexStatus.FAILED
            job.error_message = "Upload file lost during restart - please re-upload"
            job.completed_at = datetime.utcnow()
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

    async def _get_embeddings(self, app_settings: AppSettings):
        """Get the configured embedding model based on app settings."""
        provider = app_settings.embedding_provider.lower()
        model = app_settings.embedding_model
        dimensions = getattr(app_settings, "embedding_dimensions", None)

        logger.info(
            f"Getting embeddings: provider={provider}, model={model}, dimensions={dimensions}"
        )

        if provider == "ollama":
            from langchain_ollama import (
                OllamaEmbeddings,
            )  # type: ignore[import-not-found]

            return OllamaEmbeddings(
                model=model,
                base_url=app_settings.ollama_base_url,
            )
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            if not app_settings.openai_api_key:
                raise ValueError(
                    "OpenAI embeddings selected but no API key configured in Settings"
                )
            # Pass dimensions for text-embedding-3-* models (supports MRL)
            kwargs: dict = {
                "model": model,
                "api_key": app_settings.openai_api_key,  # type: ignore[arg-type]
            }
            if dimensions and model.startswith("text-embedding-3"):
                kwargs["dimensions"] = dimensions
            return OpenAIEmbeddings(**kwargs)
        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. Use 'ollama' or 'openai'."
            )

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Detect OpenAI rate limit errors across libraries."""
        status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if status == 429:
            return True

        text = str(exc).lower()
        return "rate limit" in text or "rate_limit_exceeded" in text or "429" in text

    def _retry_delay_seconds(
        self, exc: Exception, attempt: int, base_delay: float = 1.5
    ) -> float:
        """Compute delay before retrying after a rate limit.

        Uses Retry-After headers or "try again in Xms" hints when available,
        otherwise falls back to exponential backoff capped to 30s.
        """
        headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
        retry_after_header = None
        if isinstance(headers, dict):
            retry_after_header = headers.get("retry-after")

        if retry_after_header:
            try:
                return max(base_delay, min(30.0, float(retry_after_header)))
            except (TypeError, ValueError):
                pass

        text = str(exc).lower()
        match = re.search(r"try again in ([0-9]+)ms", text)
        if match:
            try:
                return max(base_delay, min(30.0, int(match.group(1)) / 1000))
            except (TypeError, ValueError):
                pass

        return min(30.0, base_delay * (2**attempt))

    async def _append_embedding_dimension_warning(self, warnings: List[str]) -> None:
        """Warn when configured embeddings exceed pgvector's 2000-dim index limit."""
        try:
            settings = await repository.get_settings()
        except Exception as e:
            logger.debug(f"Could not load settings for embedding warning: {e}")
            return

        provider = settings.embedding_provider.lower()
        model = settings.embedding_model
        configured_dim = getattr(settings, "embedding_dimensions", None)
        tracked_dim = settings.embedding_dimension

        # Prefer explicit configuration, otherwise use tracked dimension from previous runs
        dimension = configured_dim or tracked_dim

        # Heuristic defaults for OpenAI text-embedding-3 models when dimension not set
        if provider == "openai" and dimension is None:
            if model.startswith("text-embedding-3-large"):
                dimension = 3072
            elif model.startswith("text-embedding-3-small"):
                dimension = 1536

        limit = 2000
        if dimension is not None and dimension > limit:
            warning = (
                f"Embedding dimension {dimension} exceeds pgvector's {limit}-dim index limit. "
                "Search will fall back to exact (non-indexed). "
                "Use a <=2000-dim model (e.g., set Embedding Dimensions to 1536 for OpenAI text-embedding-3-*) "
                "or choose a lower-dimension embedding model."
            )
            warnings.insert(0, warning)

    async def get_job(self, job_id: str) -> Optional[IndexJob]:
        """Get a job by ID (checks cache first, then database)."""
        # Check in-memory cache for active jobs
        if job_id in self._active_jobs:
            return self._active_jobs[job_id]
        # Fallback to database
        return await repository.get_job(job_id)

    async def list_jobs(self) -> List[IndexJob]:
        """List all jobs from database."""
        return await repository.list_jobs()

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job that is pending or processing."""
        job = await self.get_job(job_id)
        if not job:
            return False

        # Set cancellation flag for cooperative cancellation
        self._cancellation_flags[job_id] = True

        # Remove from active jobs cache
        self._active_jobs.pop(job_id, None)

        # Update status in database
        job.status = IndexStatus.FAILED
        job.error_message = "Job cancelled by user"
        job.completed_at = datetime.utcnow()
        await repository.update_job(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    def _is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        return self._cancellation_flags.get(job_id, False)

    async def list_indexes(self) -> List[IndexInfo]:
        """List all available indexes, enriching with database metadata."""
        indexes = []

        # Get metadata from database
        db_metadata = await repository.list_index_metadata()
        metadata_by_name = {m.name: m for m in db_metadata}

        for path in self.index_base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Check if it's a valid FAISS index
                if (path / "index.faiss").exists() or (path / "index.pkl").exists():
                    size_bytes = sum(
                        f.stat().st_size for f in path.rglob("*") if f.is_file()
                    )

                    # Try to get document count from database metadata first
                    doc_count = 0
                    created_at = None
                    enabled = True  # Default to enabled
                    description = ""
                    source_type = "upload"
                    source = None
                    git_branch = None

                    if path.name in metadata_by_name:
                        meta = metadata_by_name[path.name]
                        doc_count = meta.documentCount
                        created_at = meta.createdAt
                        enabled = meta.enabled
                        description = getattr(meta, "description", "")
                        source_type = getattr(meta, "sourceType", "upload")
                        source = getattr(meta, "source", None)
                        git_branch = getattr(meta, "gitBranch", None)
                        search_weight = getattr(meta, "searchWeight", 1.0)
                    else:
                        search_weight = 1.0
                        # Fallback to legacy .metadata.json file
                        meta_file = path / ".metadata.json"
                        if meta_file.exists():
                            try:
                                with open(meta_file, encoding="utf-8") as mf:
                                    legacy_meta = json.load(mf)
                                    doc_count = legacy_meta.get("document_count", 0)
                                    created_at_str = legacy_meta.get("created_at")
                                    if created_at_str:
                                        created_at = datetime.fromisoformat(
                                            created_at_str
                                        )
                            except Exception:
                                pass

                    indexes.append(
                        IndexInfo(
                            name=path.name,
                            path=str(path),
                            size_mb=round(size_bytes / (1024 * 1024), 2),
                            document_count=doc_count,
                            description=description,
                            enabled=enabled,
                            search_weight=search_weight,
                            source_type=source_type,
                            source=source,
                            git_branch=git_branch,
                            created_at=created_at,
                            last_modified=datetime.fromtimestamp(path.stat().st_mtime),
                        )
                    )

        return indexes

    async def delete_index(self, name: str) -> bool:
        """Delete an index by name (both files and metadata)."""
        index_path = self.index_base_path / name
        deleted_files = False

        if index_path.exists() and index_path.is_dir():
            shutil.rmtree(index_path)
            logger.info(f"Deleted index files: {name}")
            deleted_files = True

        # Also delete metadata from database
        metadata_deleted = await repository.delete_index_metadata(name)

        if metadata_deleted and not deleted_files:
            logger.info(f"Deleted index metadata without files present: {name}")

        return deleted_files or metadata_deleted

    async def analyze_git_repository(
        self, request: AnalyzeIndexRequest
    ) -> IndexAnalysisResult:
        """
        Analyze a git repository to estimate index size and suggest exclusions.

        This performs a shallow clone, scans files matching patterns, and provides:
        - Total file count, size, and estimated chunks
        - Breakdown by file extension
        - Suggested exclusion patterns for large/binary files
        - Warnings about potential issues

        The temporary clone is deleted after analysis.
        """
        temp_dir = UPLOAD_TMP_DIR / f"analysis_{uuid.uuid4().hex[:8]}"

        try:
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Build authenticated URL if token provided
            clone_url = _build_authenticated_git_url(request.git_url, request.git_token)

            # Shallow clone (depth=1) for speed
            logger.info(f"Shallow cloning {request.git_url} for analysis")
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    request.git_branch,
                    clone_url,
                    str(temp_dir / "repo"),
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for clone
                check=False,  # We handle the error ourselves
            )

            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")

            repo_dir = temp_dir / "repo"

            # Scan and analyze files
            return await self._analyze_directory(
                repo_dir,
                file_patterns=request.file_patterns,
                exclude_patterns=request.exclude_patterns,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                max_file_size_kb=request.max_file_size_kb,
                enable_ocr=request.enable_ocr,
            )

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def analyze_upload(
        self,
        file: BinaryIO,
        filename: str,
        file_patterns: List[str],
        exclude_patterns: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_file_size_kb: int = 500,
        enable_ocr: bool = False,
    ) -> IndexAnalysisResult:
        """
        Analyze an uploaded archive to estimate index size and suggest exclusions.

        Extracts the archive, scans files matching patterns, and provides:
        - Total file count, size, and estimated chunks
        - Breakdown by file extension
        - Suggested exclusion patterns for large/binary files
        - Warnings about potential issues

        The temporary extraction is deleted after analysis.
        """
        temp_dir = UPLOAD_TMP_DIR / f"analysis_{uuid.uuid4().hex[:8]}"

        try:
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Save uploaded file temporarily
            archive_path = temp_dir / filename
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(file, f)

            logger.info(f"Saved archive for analysis: {archive_path}")

            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)
            self._extract_archive(archive_path, extract_dir)

            # Find actual source directory
            source_dir = self._find_source_dir(extract_dir)

            # Scan and analyze files
            return await self._analyze_directory(
                source_dir,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_file_size_kb=max_file_size_kb,
                enable_ocr=enable_ocr,
            )

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _analyze_directory(
        self,
        source_dir: Path,
        file_patterns: List[str],
        exclude_patterns: List[str],
        chunk_size: int,
        chunk_overlap: int,
        max_file_size_kb: int = 500,
        enable_ocr: bool = False,
    ) -> IndexAnalysisResult:
        """
        Analyze a directory to estimate indexing results.
        """
        from collections import defaultdict

        max_file_size_bytes = max_file_size_kb * 1024

        # File extension stats
        ext_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "file_count": 0,
                "total_size": 0,
                "sample_files": [],
            }
        )

        # Use centralized constants from file_constants module
        # BINARY_EXTENSIONS and MINIFIED_PATTERNS are imported at module level

        total_files = 0
        total_size = 0
        skipped_oversized = 0  # Files exceeding max_file_size_kb
        matched_patterns: set = set()
        warnings: List[str] = []
        suggested_exclusions: List[str] = []
        large_files: List[tuple] = []  # (path, size)
        minified_files: List[str] = []

        # Compile exclude patterns
        def is_excluded(file_path: Path) -> bool:
            rel_path = str(file_path.relative_to(source_dir))
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
                # Also check just the filename
                if fnmatch.fnmatch(file_path.name, pattern):
                    return True
            return False

        # Compile include patterns
        def matches_include(file_path: Path) -> tuple:
            """Returns (matches, pattern) if file matches any include pattern."""
            rel_path = str(file_path.relative_to(source_dir))
            for pattern in file_patterns:
                if fnmatch.fnmatch(rel_path, pattern):
                    return (True, pattern)
                if fnmatch.fnmatch(file_path.name, pattern):
                    return (True, pattern)
            return (False, None)

        # Check if file matches minified patterns
        def is_minified(filename: str) -> bool:
            for pattern in MINIFIED_PATTERNS:
                if fnmatch.fnmatch(filename, pattern):
                    return True
            return False

        # Walk the directory
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue

            # Skip excluded
            if is_excluded(file_path):
                continue

            # Check if matches include patterns
            matches, pattern = matches_include(file_path)
            if not matches:
                continue

            matched_patterns.add(pattern)

            # Get file stats
            try:
                size = file_path.stat().st_size
            except OSError:
                continue

            # Skip zero-byte files
            if size == 0:
                continue

            # Skip files exceeding max size limit
            if size > max_file_size_bytes:
                skipped_oversized += 1
                continue

            ext = file_path.suffix.lower() or "(no extension)"
            rel_path = str(file_path.relative_to(source_dir))

            total_files += 1
            total_size += size

            # Track by extension
            stats = ext_stats[ext]
            stats["file_count"] += 1
            stats["total_size"] += size
            if len(stats["sample_files"]) < 5:
                stats["sample_files"].append(rel_path)

            # Check for issues
            if ext in BINARY_EXTENSIONS:
                # Should suggest excluding these
                pass

            # Track files approaching the limit (>80% of max)
            if size > max_file_size_bytes * 0.8:
                large_files.append((rel_path, size))

            if is_minified(file_path.name):
                minified_files.append(rel_path)

        # Calculate estimated chunks per file type
        for ext, stats in ext_stats.items():
            # Estimate chunks: (file_size - overlap) / (chunk_size - overlap)
            # Simplified: size / effective_chunk_size
            effective_chunk = chunk_size - chunk_overlap
            if effective_chunk > 0:
                stats["estimated_chunks"] = max(
                    1, stats["total_size"] // effective_chunk
                )
            else:
                stats["estimated_chunks"] = stats["file_count"]

        # Get smart exclusion suggestions (LLM if available, otherwise heuristics)
        # Extract repo name from source_dir for context
        repo_name = (
            source_dir.name if source_dir.name != "repo" else source_dir.parent.name
        )

        # Get smart suggestions (uses LLM if configured, falls back to heuristics)
        # Pass full ext_stats so LLM can see file counts and estimated chunks
        smart_exclusions, _used_llm = await get_smart_exclusion_suggestions(
            ext_stats=dict(ext_stats),  # Convert defaultdict to regular dict
            repo_name=repo_name,
        )
        suggested_exclusions.extend(smart_exclusions)

        # Add warnings based on what we found
        # Separate truly unparseable from parseable documents and OCR-eligible images
        ocr_images_found = [ext for ext in ext_stats if ext in OCR_EXTENSIONS]
        unparseable_found = [
            ext
            for ext in ext_stats
            if ext in UNPARSEABLE_BINARY_EXTENSIONS and ext not in OCR_EXTENSIONS
        ]
        parseable_docs_found = [
            ext for ext in ext_stats if ext in PARSEABLE_DOCUMENT_EXTENSIONS
        ]

        if ocr_images_found:
            if enable_ocr:
                warnings.append(
                    f"Found image types ({', '.join(ocr_images_found)}) that will be processed "
                    "with OCR to extract text."
                )
            else:
                warnings.append(
                    f"Found image types ({', '.join(ocr_images_found)}) that will be skipped. "
                    "Enable OCR to extract text from these files."
                )

        if unparseable_found:
            warnings.append(
                f"Found {len(unparseable_found)} binary file types that will be auto-skipped: "
                + ", ".join(unparseable_found[:5])
                + ("..." if len(unparseable_found) > 5 else "")
            )

        if parseable_docs_found:
            warnings.append(
                f"Found document types ({', '.join(parseable_docs_found)}) that will be parsed "
                "using document extractors (PDF, Word, Excel, PowerPoint, OpenDocument)."
            )

        if skipped_oversized > 0:
            warnings.append(
                f"Skipped {skipped_oversized} files exceeding the {max_file_size_kb}KB size limit. "
                "Increase max file size if you need to include them."
            )

        if large_files:
            threshold_kb = int(max_file_size_kb * 0.8)
            if len(large_files) > 5:
                warnings.append(
                    f"Found {len(large_files)} files over {threshold_kb}KB (approaching limit). "
                    f"Examples: {large_files[0][0]} ({large_files[0][1] // 1024}KB)"
                )
            else:
                for path, size in large_files[:3]:
                    warnings.append(f"Large file: {path} ({size // 1024}KB)")

        # Calculate totals
        total_estimated_chunks = sum(
            stats["estimated_chunks"] for stats in ext_stats.values()
        )

        # Estimate index size:
        # - Each chunk embedding ~6KB for 1536 dims (OpenAI) or ~3KB for 768 dims (Ollama)
        # - Plus metadata overhead
        # Use conservative 6KB per chunk estimate
        estimated_index_size_mb = (total_estimated_chunks * 6) / 1024

        # Build file type stats list (sorted by chunk count descending)
        file_type_list = [
            FileTypeStats(
                extension=ext,
                file_count=stats["file_count"],
                total_size_bytes=stats["total_size"],
                estimated_chunks=stats["estimated_chunks"],
                sample_files=stats["sample_files"],
            )
            for ext, stats in sorted(
                ext_stats.items(),
                key=lambda x: x[1]["estimated_chunks"],
                reverse=True,
            )
        ]

        # Add size warning if very large
        if estimated_index_size_mb > 1000:  # > 1GB
            warnings.insert(
                0,
                f"Estimated index size is {estimated_index_size_mb:.0f}MB. "
                "Consider adding more exclusion patterns or reducing included file types.",
            )

        # Warn if embedding dimensions exceed pgvector's index limit (applies to filesystem + doc indexing)
        await self._append_embedding_dimension_warning(warnings)

        return IndexAnalysisResult(
            total_files=total_files,
            total_size_bytes=total_size,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            estimated_chunks=total_estimated_chunks,
            estimated_index_size_mb=round(estimated_index_size_mb, 2),
            file_type_stats=file_type_list,
            suggested_exclusions=list(set(suggested_exclusions)),
            matched_file_patterns=list(matched_patterns),
            warnings=warnings,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def create_index_from_upload(
        self, file: BinaryIO, filename: str, config: IndexConfig
    ) -> IndexJob:
        """Create an index from an uploaded archive file.

        The uploaded file is stored in a persistent tmp directory so it
        survives server restarts and can be resumed if interrupted.
        """
        job_id = str(uuid.uuid4())[:8]

        job = IndexJob(
            id=job_id,
            name=config.name,
            config=config,
            source_type="upload",
            source_path=filename,
        )

        # Persist to database FIRST (before any processing)
        await repository.create_job(job)

        # Cache for active processing
        self._active_jobs[job_id] = job

        try:
            # Save uploaded file to PERSISTENT tmp location (survives restarts)
            tmp_dir = UPLOAD_TMP_DIR / job_id
            tmp_dir.mkdir(parents=True, exist_ok=True)

            if not os.access(tmp_dir, os.W_OK):
                raise PermissionError(f"Upload tmp directory not writable: {tmp_dir}")

            archive_path = tmp_dir / filename

            with open(archive_path, "wb") as f:
                shutil.copyfileobj(file, f)

            logger.info(
                f"Saved upload to tmp directory for job {job_id}: {archive_path}"
            )

            # Start processing in background
            asyncio.create_task(self._process_upload(job, archive_path, tmp_dir))

        except Exception as e:
            job.status = IndexStatus.FAILED
            job.error_message = str(e)
            await repository.update_job(job)
            self._active_jobs.pop(job_id, None)
            if "tmp_dir" in locals():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return job

    async def create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> IndexJob:
        """Create an index from a git repository."""
        job_id = str(uuid.uuid4())[:8]

        job = IndexJob(
            id=job_id,
            name=config.name,
            config=config,
            source_type="git",
            git_url=git_url,
            git_branch=branch,
            git_token=git_token,  # Kept in memory only, not persisted
        )

        # Persist to database
        await repository.create_job(job)

        # Cache for active processing
        self._active_jobs[job_id] = job

        # Start processing in background
        asyncio.create_task(self._process_git(job))

        return job

    async def _process_upload(self, job: IndexJob, archive_path: Path, temp_dir: Path):
        """Process an uploaded archive file (zip, tar, tar.gz, tar.bz2)."""
        try:
            job.status = IndexStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await repository.update_job(job)

            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled before extraction")
                return

            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            logger.info(f"Extracting {archive_path} to {extract_dir}")
            self._extract_archive(archive_path, extract_dir)

            # Check for cancellation after extraction
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled after extraction")
                return

            # Find the actual source directory (handle nested zips)
            source_dir = self._find_source_dir(extract_dir)

            # Create the index
            await self._create_faiss_index(job, source_dir)

            # Only mark completed if not cancelled
            if not self._is_cancelled(job.id):
                job.status = IndexStatus.COMPLETED
                job.completed_at = datetime.utcnow()

        except asyncio.CancelledError:
            # Job was cancelled - status already set by cancel_job()
            logger.info(f"Job {job.id} processing stopped due to cancellation")

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process upload for job {job.id}")
                job.status = IndexStatus.FAILED
                job.error_message = str(e)

        finally:
            # Cleanup temp directory and cancellation flag
            shutil.rmtree(temp_dir, ignore_errors=True)
            self._cancellation_flags.pop(job.id, None)
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

            # Reinitialize RAG components if job completed successfully
            await self._maybe_reinitialize_rag(job)

    async def _process_git(self, job: IndexJob, skip_clone: bool = False):
        """Process a git repository.

        Args:
            job: The index job to process
            skip_clone: If True, skip cloning (repo already exists from previous attempt)
        """
        # Use persistent _tmp directory so cloned repos survive restarts
        temp_dir = UPLOAD_TMP_DIR / job.id
        temp_dir.mkdir(parents=True, exist_ok=True)
        clone_dir = temp_dir / "repo"
        clone_complete_marker = temp_dir / ".clone_complete"

        try:
            job.status = IndexStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await repository.update_job(job)

            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled before cloning")
                return

            # Preserve token for metadata storage before clearing from job object
            stored_token = job.git_token

            if skip_clone:
                # Repo already cloned from previous attempt
                logger.info(
                    f"Skipping clone for job {job.id}, using existing repo at {clone_dir}"
                )
            else:
                # Clone repository
                logger.info(f"Cloning {job.git_url} branch {job.git_branch}")

                # Update job to show cloning phase
                job.error_message = (
                    "Cloning repository..."  # Use error_message as status hint
                )
                await repository.update_job(job)

                if not job.git_url:
                    raise ValueError("Git URL is required for git source type")

                # Build clone URL with token authentication
                clone_url = _build_authenticated_git_url(job.git_url, job.git_token)

                # Set environment to prevent git from prompting for credentials
                env = {
                    **os.environ,
                    "GIT_TERMINAL_PROMPT": "0",  # Disable credential prompting
                }

                try:
                    process = await asyncio.create_subprocess_exec(
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        job.git_branch or "main",
                        clone_url,
                        str(clone_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    # Timeout after 5 minutes for clone
                    _, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=300
                    )
                except TimeoutError:
                    process.kill()
                    raise RuntimeError(
                        "Git clone timed out after 5 minutes. The repository may be too large or unreachable."
                    )

                if process.returncode != 0:
                    error_msg = stderr.decode()
                    # Provide clearer error for authentication failures
                    if (
                        "could not read Username" in error_msg
                        or "Authentication failed" in error_msg
                    ):
                        raise RuntimeError(
                            "Git clone failed: Authentication required. "
                            "This is a private repository - please provide a valid access token."
                        )
                    raise RuntimeError(f"Git clone failed: {error_msg}")

                # Mark clone as complete so we can resume from here if interrupted
                clone_complete_marker.touch()
                logger.info("Clone complete, scanning files...")

            # Clear token from job object (not needed in memory after we have stored_token)
            job.git_token = None

            # Clear cloning status, update to scanning phase
            job.error_message = None
            await repository.update_job(job)

            # Check for cancellation after clone
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled after cloning")
                return

            # Create the index, passing the token for storage in metadata
            await self._create_faiss_index(job, clone_dir, git_token=stored_token)

            # Only mark completed if not cancelled
            if not self._is_cancelled(job.id):
                job.status = IndexStatus.COMPLETED
                job.completed_at = datetime.utcnow()

        except asyncio.CancelledError:
            # Job was cancelled - status already set by cancel_job()
            logger.info(f"Job {job.id} processing stopped due to cancellation")

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process git for job {job.id}")
                job.status = IndexStatus.FAILED
                job.error_message = str(e)

        finally:
            # Only clean up temp_dir if job completed (success or failure)
            # If job is still pending/processing (e.g., server shutdown), leave files for resume
            if job.status in (IndexStatus.COMPLETED, IndexStatus.FAILED):
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._cancellation_flags.pop(job.id, None)
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

            # Reinitialize RAG components if job completed successfully
            await self._maybe_reinitialize_rag(job)

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract an archive file (zip, tar, tar.gz, tar.bz2) to a directory."""
        filename_lower = archive_path.name.lower()

        if filename_lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        elif filename_lower.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir, filter="data")
        elif filename_lower.endswith((".tar.bz2", ".tbz2")):
            with tarfile.open(archive_path, "r:bz2") as tf:
                tf.extractall(extract_dir, filter="data")
        elif filename_lower.endswith(".tar"):
            with tarfile.open(archive_path, "r:") as tf:
                tf.extractall(extract_dir, filter="data")
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.name}")

    def _find_source_dir(self, extract_dir: Path) -> Path:
        """Find the actual source directory in extracted content."""
        # If there's a single directory, use that
        items = list(extract_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            return items[0]
        return extract_dir

    def _should_include_file(self, file_path: Path, config: IndexConfig) -> bool:
        """Check if a file should be included based on patterns."""
        rel_path = str(file_path)

        # Check excludes first
        for pattern in config.exclude_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return False

        # Check includes
        for pattern in config.file_patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True

        return False

    async def _create_faiss_index(
        self, job: IndexJob, source_dir: Path, git_token: Optional[str] = None
    ):
        """Create FAISS index from source directory."""
        from langchain_community.document_loaders import TextLoader
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        config = job.config

        # Collect all files to process (run in thread pool as rglob can be slow on large repos)
        def collect_files_sync() -> List[Path]:
            """Synchronous file collection - runs in thread pool."""
            files = []
            for pattern in config.file_patterns:
                # Use removeprefix instead of lstrip to avoid stripping too many characters
                # lstrip("**/") on "**/*.py" incorrectly gives ".py", removeprefix gives "*.py"
                glob_pattern = pattern.removeprefix("**/").removeprefix("*/")
                logger.debug(
                    f"Globbing with pattern: {glob_pattern} (original: {pattern})"
                )
                for file_path in source_dir.rglob(glob_pattern):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(source_dir)

                        # Check excludes - also use removeprefix instead of lstrip
                        skip = False
                        for exc_pattern in config.exclude_patterns:
                            clean_exc = exc_pattern.removeprefix("**/")
                            if fnmatch.fnmatch(str(rel_path), clean_exc):
                                skip = True
                                break

                        if not skip and file_path not in files:
                            files.append(file_path)
            return files

        all_files = await asyncio.to_thread(collect_files_sync)

        job.total_files = len(all_files)
        await repository.update_job(job)
        logger.info(f"Found {len(all_files)} files to index")

        # BINARY_EXTENSIONS is imported from ragtime.core.file_constants

        # Load documents
        # Note: TextLoader.load() is synchronous and can block the event loop,
        # especially with autodetect_encoding=True which uses chardet. We run
        # it in a thread pool to avoid blocking other async operations.
        documents = []
        skipped_binary = 0
        enable_ocr = config.enable_ocr

        def load_file_sync(file_path: Path) -> List:
            """Synchronous file loading - runs in thread pool."""
            from langchain_core.documents import Document as LangChainDocument

            ext_lower = file_path.suffix.lower()

            # Use document parser for Office/PDF files and images (when OCR enabled)
            if ext_lower in PARSEABLE_DOCUMENT_EXTENSIONS or (
                enable_ocr and ext_lower in OCR_EXTENSIONS
            ):
                content = extract_text_from_file(file_path, enable_ocr=enable_ocr)
                if content:
                    return [LangChainDocument(page_content=content)]
                return []

            # Use TextLoader for plain text files
            loader = TextLoader(str(file_path), autodetect_encoding=True)
            return loader.load()

        for file_path in all_files:
            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled during file loading")
                raise asyncio.CancelledError("Job cancelled by user")

            ext_lower = file_path.suffix.lower()

            # Skip truly unparseable binary files (executables, etc.)
            # When OCR is enabled, allow image files to be processed
            if ext_lower in UNPARSEABLE_BINARY_EXTENSIONS:
                if enable_ocr and ext_lower in OCR_EXTENSIONS:
                    # Process image with OCR
                    logger.debug(f"Processing image {file_path.name} with OCR")
                else:
                    skipped_binary += 1
                    job.processed_files += 1
                    continue

            # Log document parsing (no longer a warning since we can parse them)
            if ext_lower in PARSEABLE_DOCUMENT_EXTENSIONS:
                logger.debug(
                    f"Parsing document file {file_path.name} with document parser"
                )

            try:
                # Run synchronous file loading in thread pool to avoid blocking
                docs = await asyncio.to_thread(load_file_sync, file_path)

                # Add source metadata
                rel_path_str = str(file_path.relative_to(source_dir))
                for doc in docs:
                    doc.metadata["source"] = rel_path_str
                    doc.metadata["index_name"] = job.name

                documents.extend(docs)
                job.processed_files += 1

                # Update progress periodically and yield to event loop
                if job.processed_files % 50 == 0:
                    await repository.update_job(job)
                    logger.info(f"Loaded {job.processed_files}/{job.total_files} files")

            except Exception as e:
                # Debug level for common encoding issues, don't spam logs
                logger.debug(f"Skipped {file_path.name}: {e}")
                job.processed_files += 1

        if skipped_binary > 0:
            logger.info(f"Skipped {skipped_binary} binary files")

        if not documents:
            raise ValueError("No documents were loaded")

        logger.info(f"Loaded {len(documents)} documents, splitting into chunks...")

        # Split into chunks (run in thread pool to avoid blocking with many documents)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = await asyncio.to_thread(splitter.split_documents, documents)
        job.total_chunks = len(chunks)
        await repository.update_job(job)

        logger.info(f"Created {len(chunks)} chunks, generating embeddings...")

        # Get embedding settings from app settings
        app_settings = await repository.get_settings()
        embeddings = await self._get_embeddings(app_settings)

        logger.info(
            f"Using {app_settings.embedding_provider} embeddings with model: {app_settings.embedding_model}"
        )

        # Process in batches to show progress and throttle embedding calls
        batch_size = 50
        batch_pause_seconds = 0.5
        max_retries = 5
        db = None

        for i in range(0, len(chunks), batch_size):
            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled during embedding")
                raise asyncio.CancelledError("Job cancelled by user")

            batch = chunks[i : i + batch_size]
            batch_db = None
            attempt = 0

            while True:
                try:
                    batch_db = await asyncio.to_thread(
                        FAISS.from_documents, batch, embeddings
                    )
                    break
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < max_retries:
                        wait_seconds = self._retry_delay_seconds(e, attempt)
                        job.error_message = f"OpenAI rate limit hit, retrying batch {i // batch_size + 1} in {wait_seconds:.1f}s"
                        await repository.update_job(job)
                        logger.warning(job.error_message)
                        await asyncio.sleep(wait_seconds)
                        attempt += 1
                        continue
                    raise

            if db is None:
                db = batch_db
            else:
                db.merge_from(batch_db)

            # Update processed chunks progress
            job.error_message = None
            job.processed_chunks = min(i + batch_size, len(chunks))
            await repository.update_job(job)
            logger.info(f"Embedded {job.processed_chunks}/{len(chunks)} chunks")

            # Small pause between batches to stay under provider TPM limits
            if batch_pause_seconds > 0 and (i + batch_size) < len(chunks):
                await asyncio.sleep(batch_pause_seconds)

        if db is None:
            raise ValueError("No documents were embedded - FAISS index creation failed")

        # Save the index
        index_path = self.index_base_path / job.name
        index_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving index to {index_path}")
        await asyncio.to_thread(db.save_local, str(index_path))

        # Calculate index size
        size_bytes = sum(f.stat().st_size for f in index_path.rglob("*") if f.is_file())

        # Auto-generate description if not provided
        description = getattr(config, "description", "")
        if not description:
            description = await generate_index_description(
                index_name=job.name,
                documents=documents,
                source_type=job.source_type,
                source=job.git_url or job.source_path,
            )

        # Save metadata to database
        await repository.upsert_index_metadata(
            name=job.name,
            path=str(index_path),
            document_count=len(documents),
            chunk_count=len(chunks),
            size_bytes=size_bytes,
            source_type=job.source_type,
            source=job.git_url or job.source_path,
            config_snapshot=config.model_dump(),
            description=description,
            git_branch=job.git_branch if job.source_type == "git" else None,
            git_token=git_token,
        )

        logger.info(f"Index {job.name} created successfully!")


# Global indexer instance - uses configured path
indexer = IndexerService(index_base_path=settings.index_data_path)
