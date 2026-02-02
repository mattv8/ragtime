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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings, invalidate_settings_cache
from ragtime.core.embedding_models import get_embedding_models
from ragtime.core.file_constants import (
    BINARY_EXTENSIONS,
    MINIFIED_PATTERNS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
    get_embedding_safety_margin,
)
from ragtime.core.logging import get_logger
from ragtime.indexer.document_parser import OCR_EXTENSIONS
from ragtime.indexer.file_utils import (
    HARDCODED_EXCLUDES,
    build_authenticated_git_url,
    extract_archive,
    find_source_dir,
)
from ragtime.indexer.llm_exclusions import get_smart_exclusion_suggestions
from ragtime.indexer.memory_utils import (
    estimate_index_memory,
    estimate_memory_at_dimensions,
    get_embedding_dimension,
)
from ragtime.indexer.models import (
    AnalyzeIndexRequest,
    AppSettings,
    CommitHistoryInfo,
    CommitHistorySample,
    FileTypeStats,
    IndexAnalysisResult,
    IndexConfig,
    IndexInfo,
    IndexJob,
    IndexStatus,
    MemoryEstimate,
    OcrMode,
    VectorStoreType,
)
from ragtime.indexer.repository import repository
from ragtime.tools.git_history import _is_shallow_repository

logger = get_logger(__name__)

# Persistent storage for uploaded files
UPLOAD_TMP_DIR = Path(settings.index_data_path) / "_tmp"


async def generate_index_description(
    index_name: str,
    documents: List,
    source_type: str,
    source: Optional[str] = None,
) -> str:
    """
    Auto-generate a description for an index using the configured LLM.

    Samples file paths and content to create a concise description
    that helps the AI understand what knowledge is available.

    Uses the LLM provider configured in app settings (OpenAI, Anthropic, or Ollama).
    """
    try:
        from ragtime.core.app_settings import get_app_settings

        app_settings = await get_app_settings()
        provider = app_settings.get("llm_provider", "openai").lower()

        # Get the appropriate LLM based on configured provider
        llm = None

        if provider == "ollama":
            try:
                from langchain_ollama import (
                    ChatOllama,
                )  # type: ignore[reportMissingImports]

                base_url = app_settings.get("ollama_base_url", "http://localhost:11434")
                model = app_settings.get("llm_model", "llama3.2")
                llm = ChatOllama(model=model, base_url=base_url, temperature=0.3)
                logger.debug(f"Using Ollama for description generation: {model}")
            except ImportError:
                logger.warning("langchain-ollama not installed")

        elif provider == "anthropic":
            api_key = app_settings.get("anthropic_api_key", "")
            if api_key:
                try:
                    from langchain_anthropic import (
                        ChatAnthropic,
                    )  # type: ignore[import-untyped]

                    model = app_settings.get("llm_model", "claude-sonnet-4-20250514")
                    llm = ChatAnthropic(
                        model_name=model,
                        temperature=0.3,
                        api_key=api_key,
                        timeout=None,
                        stop=None,
                    )
                    logger.debug(f"Using Anthropic for description generation: {model}")
                except ImportError:
                    logger.warning("langchain-anthropic not installed")
            else:
                logger.debug("Anthropic selected but no API key configured")

        elif provider == "openai":
            api_key = app_settings.get("openai_api_key", "")
            if api_key:
                from langchain_openai import ChatOpenAI

                # Use cheaper model for descriptions
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    api_key=api_key,
                )
                logger.debug("Using OpenAI for description generation: gpt-4o-mini")
            else:
                logger.debug("OpenAI selected but no API key configured")

        if llm is None:
            logger.debug(
                f"No LLM available for description generation (provider: {provider})"
            )
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

    def _compute_clone_timeout_minutes(self, config: IndexConfig) -> int:
        """Derive clone timeout from depth unless explicitly overridden by user."""

        default_timeout = IndexConfig.model_fields["git_clone_timeout_minutes"].default
        user_timeout = getattr(config, "git_clone_timeout_minutes", default_timeout)

        # Respect explicit override
        if user_timeout != default_timeout:
            return user_timeout

        min_timeout = default_timeout or 5
        max_timeout = 120  # 2 hours cap for full history
        depth = getattr(config, "git_history_depth", 1)

        # depth=0 means full history
        if depth == 0:
            return max_timeout

        # Shallow or unset depth stays at minimum
        if depth <= 1:
            return min_timeout

        # Power curve (exponent > 1) for slow-then-fast growth
        max_depth = 1001  # Slider full + sentinel
        effective_depth = min(depth, max_depth)
        factor = (effective_depth / max_depth) ** 2.5
        timeout = min_timeout + (max_timeout - min_timeout) * factor

        return int(round(max(min_timeout, min(max_timeout, timeout))))

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
        state and resumes them. Also cleans up orphaned directories.

        Returns:
            Number of jobs recovered
        """
        jobs = await repository.list_jobs()
        interrupted = [
            j for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
        ]

        recovered = 0
        if interrupted:
            logger.info(f"Found {len(interrupted)} interrupted job(s) to recover")

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

        # Clean up orphaned directories (always run, after job recovery)
        await self._cleanup_orphaned_tmp_dirs()
        await self._cleanup_orphaned_git_repos()

        return recovered

    async def _cleanup_orphaned_tmp_dirs(self) -> None:
        """Remove tmp directories for jobs that no longer exist or are completed."""
        if not UPLOAD_TMP_DIR.exists():
            return

        jobs = await repository.list_jobs()
        # Keep tmp dirs for: active jobs (pending/processing) and failed upload jobs (for retry)
        keep_job_ids = {
            j.id
            for j in jobs
            if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
            or (j.status == IndexStatus.FAILED and j.source_type == "upload")
        }

        for tmp_path in UPLOAD_TMP_DIR.iterdir():
            if tmp_path.is_dir() and tmp_path.name not in keep_job_ids:
                logger.info(f"Cleaning orphaned tmp directory: {tmp_path.name}")
                await asyncio.to_thread(shutil.rmtree, tmp_path, ignore_errors=True)

    async def _cleanup_orphaned_git_repos(self) -> None:
        """Remove .git_repo directories for indexes that have no FAISS index.

        An orphaned git repo occurs when:
        1. A git clone completed but indexing failed
        2. There's no active (pending/processing) job that might complete

        This is safe to run on startup after job recovery has been attempted.
        """
        if not self.index_base_path.exists():
            return

        # Get active jobs that might still complete
        jobs = await repository.list_jobs()
        active_index_names = {
            j.name
            for j in jobs
            if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
        }

        for index_dir in self.index_base_path.iterdir():
            if not index_dir.is_dir() or index_dir.name.startswith("_"):
                continue

            git_repo = index_dir / ".git_repo"
            faiss_index = index_dir / "index.faiss"

            # If there's a git repo but no FAISS index, it might be orphaned
            if git_repo.exists() and not faiss_index.exists():
                # Don't delete if there's an active job for this index
                if index_dir.name in active_index_names:
                    logger.debug(
                        f"Keeping .git_repo for {index_dir.name}: active job exists"
                    )
                    continue

                # Safe to clean up
                logger.info(
                    f"Cleaning orphaned git repo: {index_dir.name}/.git_repo "
                    "(no FAISS index, no active job)"
                )
                await asyncio.to_thread(shutil.rmtree, git_repo, ignore_errors=True)

                # If the directory is now empty, remove it too
                try:
                    if index_dir.exists() and not any(index_dir.iterdir()):
                        index_dir.rmdir()
                        logger.info(f"Removed empty index directory: {index_dir.name}")
                except OSError:
                    pass  # Directory not empty or other issue

    async def discover_orphan_indexes(self) -> int:
        """
        Discover FAISS indexes on disk that have no database metadata.

        Reads the index.pkl file to extract document count and creates
        database metadata entries for discovered indexes.

        Called during application startup.

        Note: This only registers document indexes. Filesystem FAISS indexes
        are tracked via tool_configs and should not be added to index_metadata.

        Returns:
            Number of indexes discovered and registered
        """
        import pickle

        # Get existing metadata from database
        db_metadata = await repository.list_index_metadata()
        known_names = {m.name for m in db_metadata}

        # Get filesystem index names that use FAISS backend (not pgvector)
        # Only FAISS-backed filesystem indexes have index.faiss files on disk
        # We must NOT register these as document indexes
        tool_configs = await repository.list_tool_configs()
        filesystem_faiss_index_names = set()
        for tc in tool_configs:
            if tc.tool_type == "filesystem_indexer":
                conn_config = tc.connection_config or {}
                vector_store_type = conn_config.get("vector_store_type", "pgvector")
                # Only exclude FAISS-backed filesystem indexes
                if vector_store_type == "faiss":
                    index_name = conn_config.get("index_name")
                    if index_name:
                        filesystem_faiss_index_names.add(index_name)

        discovered = 0

        for path in self.index_base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Skip if already known in index_metadata
                if path.name in known_names:
                    continue

                # Skip if this is a FAISS-backed filesystem index (tracked in tool_configs)
                if path.name in filesystem_faiss_index_names:
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
                            docstore_dict = getattr(docstore, "_dict", {})
                            doc_count = len(docstore_dict)  # noqa: SLF001
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
            # Check if clone was completed before restart using temp marker
            temp_marker_dir = UPLOAD_TMP_DIR / job.id
            clone_complete_marker = temp_marker_dir / ".clone_complete"
            repo_dir = self.index_base_path / job.name / ".git_repo"

            if repo_dir.exists() and clone_complete_marker.exists():
                # Clone completed previously, skip directly to indexing
                logger.info(f"Found existing clone for job {job.id}, resuming indexing")
                asyncio.create_task(self._process_git(job, skip_clone=True))
            else:
                # Need to re-clone - clean up any partial clone first
                if repo_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, repo_dir, ignore_errors=True)
                if temp_marker_dir.exists():
                    await asyncio.to_thread(
                        shutil.rmtree, temp_marker_dir, ignore_errors=True
                    )
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
                        await asyncio.to_thread(
                            shutil.rmtree, extracted_dir, ignore_errors=True
                        )

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
        from ragtime.indexer.vector_utils import get_embeddings_model

        # Use dict-safe access for logging (app_settings may be dict or object)
        provider = (
            app_settings.get("embedding_provider")
            if isinstance(app_settings, dict)
            else getattr(app_settings, "embedding_provider", None)
        )
        model = (
            app_settings.get("embedding_model")
            if isinstance(app_settings, dict)
            else getattr(app_settings, "embedding_model", None)
        )
        dims = (
            app_settings.get("embedding_dimensions")
            if isinstance(app_settings, dict)
            else getattr(app_settings, "embedding_dimensions", None)
        )
        logger.info(
            f"Getting embeddings: provider={provider}, model={model}, dimensions={dims}"
        )

        return await get_embeddings_model(
            app_settings,
            allow_missing_api_key=False,
            return_none_on_error=False,
            logger_override=logger,
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
        from ragtime.indexer.vector_utils import append_embedding_dimension_warning

        await append_embedding_dimension_warning(warnings, logger_override=logger)

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

    async def _cleanup_failed_index_metadata(self, name: str) -> None:
        """Clean up optimistic index_metadata if a NEW index job fails.

        Only deletes metadata if the index has no actual data (chunk_count=0),
        indicating it was just created optimistically and the job failed before
        any real data was indexed. This preserves existing indexes on re-index failures.
        """
        try:
            metadata = await repository.get_index_metadata(name)
            if metadata and metadata.chunkCount == 0:
                # This was an optimistic placeholder - safe to clean up
                logger.info(f"Cleaning up optimistic metadata for failed index: {name}")
                await repository.delete_index_metadata(name)
        except Exception as e:
            # Don't fail the job cleanup if metadata cleanup fails
            logger.warning(f"Failed to cleanup metadata for {name}: {e}")

    async def _create_optimistic_index_metadata(
        self,
        config: IndexConfig,
        source_type: str,
        source: str | None,
        git_branch: str | None = None,
        git_token: str | None = None,
    ) -> None:
        """Create optimistic index_metadata so the index shows up in UI immediately.

        Values will be updated when job completes; if job fails, metadata will be
        cleaned up by _cleanup_failed_index_metadata().

        Args:
            config: Index configuration
            source_type: "upload" or "git"
            source: Source path/URL (filename for upload, git URL for git)
            git_branch: Git branch (for git source only)
            git_token: Git token (for git source only)
        """
        index_path = self.index_base_path / config.name
        await repository.upsert_index_metadata(
            name=config.name,
            path=str(index_path),
            document_count=0,
            chunk_count=0,
            size_bytes=0,
            source_type=source_type,
            source=source,
            config_snapshot=config.model_dump(),
            description=config.description or "",
            git_branch=git_branch,
            git_token=git_token,
            vector_store_type=config.vector_store_type,
        )

    async def list_indexes(self) -> List[IndexInfo]:
        """List all available document indexes.

        Only returns indexes that exist in index_metadata (document indexes).
        Filesystem FAISS indexes are managed separately via tool_configs.
        """
        indexes = []

        # Get metadata from database - this is the source of truth for document indexes
        db_metadata = await repository.list_index_metadata()

        for meta in db_metadata:
            path = Path(meta.path) if meta.path else self.index_base_path / meta.name

            # Determine vector store type from database
            vector_store_type_str = getattr(meta, "vectorStoreType", None)
            if vector_store_type_str:
                # Handle both string and enum values from database
                if hasattr(vector_store_type_str, "value"):
                    vector_store_type = VectorStoreType(vector_store_type_str.value)
                else:
                    vector_store_type = VectorStoreType(vector_store_type_str)
            else:
                vector_store_type = VectorStoreType.FAISS

            # Check if this is an optimistic/in-progress index (0 documents means job hasn't completed yet)
            is_optimistic = meta.documentCount == 0

            # Skip if directory doesn't exist on disk (for FAISS indexes)
            # But allow optimistic indexes through so they show in UI while job is processing
            if vector_store_type == VectorStoreType.FAISS and not is_optimistic:
                if not path.exists() or not path.is_dir():
                    logger.warning(
                        f"Index {meta.name} in database but not on disk: {path}"
                    )
                    continue

                # Verify it's a valid FAISS index
                if (
                    not (path / "index.faiss").exists()
                    and not (path / "index.pkl").exists()
                ):
                    continue

            size_bytes = (
                sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.exists()
                else 0
            )

            # Extract metadata fields
            doc_count = meta.documentCount
            chunk_count = getattr(meta, "chunkCount", 0)
            created_at = meta.createdAt
            last_modified = getattr(meta, "lastModified", None)
            enabled = meta.enabled
            description = getattr(meta, "description", "")
            source_type = getattr(meta, "sourceType", "upload")
            source = getattr(meta, "source", None)
            git_branch = getattr(meta, "gitBranch", None)
            search_weight = getattr(meta, "searchWeight", 1.0)
            config_snapshot_data = getattr(meta, "configSnapshot", None)
            has_stored_token = bool(getattr(meta, "gitToken", None))
            display_name = getattr(meta, "displayName", None)

            # Build config_snapshot from data if available
            config_snapshot = None
            if config_snapshot_data:
                from ragtime.indexer.models import IndexConfigSnapshot, OcrMode

                config_snapshot = IndexConfigSnapshot(
                    file_patterns=config_snapshot_data.get("file_patterns", ["**/*"]),
                    exclude_patterns=config_snapshot_data.get("exclude_patterns", []),
                    chunk_size=config_snapshot_data.get("chunk_size", 1000),
                    chunk_overlap=config_snapshot_data.get("chunk_overlap", 200),
                    max_file_size_kb=config_snapshot_data.get("max_file_size_kb", 500),
                    ocr_mode=OcrMode(config_snapshot_data.get("ocr_mode", "disabled")),
                    ocr_vision_model=config_snapshot_data.get("ocr_vision_model"),
                    git_clone_timeout_minutes=config_snapshot_data.get(
                        "git_clone_timeout_minutes", 5
                    ),
                    git_history_depth=config_snapshot_data.get("git_history_depth", 1),
                )

            # Check for git repo directory (git history)
            git_repo_path = path / ".git_repo"
            git_repo_size_mb = None
            has_git_history = False
            if git_repo_path.exists() and git_repo_path.is_dir():
                # Check if it has meaningful history using commit count
                # Repos with depth > 1 still have useful history to search
                has_git_history = not await _is_shallow_repository(git_repo_path)
                if has_git_history:
                    git_repo_size = sum(
                        f.stat().st_size
                        for f in git_repo_path.rglob("*")
                        if f.is_file()
                    )
                    git_repo_size_mb = round(git_repo_size / (1024 * 1024), 2)

            indexes.append(
                IndexInfo(
                    name=meta.name,
                    display_name=display_name,
                    path=str(path),
                    size_mb=round(size_bytes / (1024 * 1024), 2),
                    document_count=doc_count,
                    chunk_count=chunk_count,
                    description=description,
                    enabled=enabled,
                    search_weight=search_weight,
                    source_type=source_type,
                    source=source,
                    git_branch=git_branch,
                    has_stored_token=has_stored_token,
                    config_snapshot=config_snapshot,
                    created_at=created_at,
                    last_modified=last_modified
                    or (
                        datetime.fromtimestamp(path.stat().st_mtime)
                        if path.exists()
                        else datetime.utcnow()
                    ),
                    git_repo_size_mb=git_repo_size_mb,
                    has_git_history=has_git_history,
                    vector_store_type=vector_store_type,
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

    async def _sample_commit_history(
        self,
        repo_dir: Path,
        git_url: str,
        git_branch: str,
        git_token: Optional[str] = None,
    ) -> Optional[CommitHistoryInfo]:
        """
        Sample commit history at various depths for depth-to-date interpolation.

        Uses remote refs to get total commit count and samples commits at
        logarithmically distributed depths (1, 10, 50, 100, 500, 1000, etc).
        """
        try:
            # Get total commit count from remote (fast, uses refs)
            clone_url = build_authenticated_git_url(git_url, git_token)

            # Fetch commit count from remote
            count_result = subprocess.run(
                ["git", "rev-list", "--count", f"origin/{git_branch}"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if count_result.returncode != 0:
                # Fallback: try to get count from ls-remote
                ls_result = subprocess.run(
                    [
                        "git",
                        "ls-remote",
                        "--refs",
                        clone_url,
                        f"refs/heads/{git_branch}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if ls_result.returncode != 0:
                    logger.warning("Could not get remote commit count")
                    return None
                # Can't get count from ls-remote, but we tried
                total_commits = 0
            else:
                total_commits = int(count_result.stdout.strip())

            if total_commits == 0:
                # Try fetching just enough history to sample
                # Fetch 1001 commits to sample at various depths
                fetch_result = subprocess.run(
                    ["git", "fetch", "--deepen=1000", "origin", git_branch],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if fetch_result.returncode != 0:
                    logger.warning(f"Could not deepen fetch: {fetch_result.stderr}")

                # Now count local commits
                count_result = subprocess.run(
                    ["git", "rev-list", "--count", f"origin/{git_branch}"],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if count_result.returncode == 0:
                    total_commits = int(count_result.stdout.strip())

            # Define sample depths (logarithmically distributed)
            sample_depths = [0, 10, 50, 100, 500, 1000, 5000, 10000]
            # Filter to only depths within our available commits
            available_depths = [d for d in sample_depths if d < total_commits or d == 0]

            samples: List[CommitHistorySample] = []
            oldest_date: Optional[str] = None
            newest_date: Optional[str] = None

            for depth in available_depths:
                # Get commit at this depth: git log --skip=N -1 --format="%H %aI"
                log_result = subprocess.run(
                    [
                        "git",
                        "log",
                        f"--skip={depth}",
                        "-1",
                        "--format=%H %aI",
                        f"origin/{git_branch}",
                    ],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )

                if log_result.returncode == 0 and log_result.stdout.strip():
                    parts = log_result.stdout.strip().split(" ", 1)
                    if len(parts) == 2:
                        commit_hash, date = parts
                        samples.append(
                            CommitHistorySample(
                                depth=depth,
                                date=date,
                                hash=commit_hash[:7],
                            )
                        )

                        if depth == 0:
                            newest_date = date
                        oldest_date = date  # Keep updating to get the oldest

            # If we have samples, also try to get the actual oldest commit
            if total_commits > 0 and total_commits > sample_depths[-1]:
                # Get the very last commit
                log_result = subprocess.run(
                    [
                        "git",
                        "log",
                        "--reverse",
                        "-1",
                        "--format=%H %aI",
                        f"origin/{git_branch}",
                    ],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if log_result.returncode == 0 and log_result.stdout.strip():
                    parts = log_result.stdout.strip().split(" ", 1)
                    if len(parts) == 2:
                        oldest_date = parts[1]

            return CommitHistoryInfo(
                total_commits=total_commits,
                samples=samples,
                oldest_date=oldest_date,
                newest_date=newest_date,
            )

        except Exception as e:
            logger.warning(f"Failed to sample commit history: {e}")
            return None

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
            clone_url = build_authenticated_git_url(request.git_url, request.git_token)

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

            # Sample commit history for depth-to-date interpolation
            commit_history = await self._sample_commit_history(
                repo_dir,
                request.git_url,
                request.git_branch,
                request.git_token,
            )

            # Scan and analyze files
            analysis_result = await self._analyze_directory(
                repo_dir,
                file_patterns=request.file_patterns,
                exclude_patterns=request.exclude_patterns,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                max_file_size_kb=request.max_file_size_kb,
                ocr_mode=request.ocr_mode,
                ocr_vision_model=request.ocr_vision_model,
            )

            # Add commit history to the result
            analysis_result.commit_history = commit_history

            return analysis_result

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
        ocr_mode: str = "disabled",
        ocr_vision_model: Optional[str] = None,
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
            extract_archive(archive_path, extract_dir)

            # Find actual source directory
            source_dir = find_source_dir(extract_dir)

            # Scan and analyze files
            return await self._analyze_directory(
                source_dir,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_file_size_kb=max_file_size_kb,
                ocr_mode=ocr_mode,
                ocr_vision_model=ocr_vision_model,
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
        ocr_mode: str = "disabled",
        ocr_vision_model: Optional[str] = None,
    ) -> IndexAnalysisResult:
        """
        Analyze a directory to estimate indexing results.
        """
        from collections import defaultdict

        ocr_enabled = ocr_mode != "disabled"
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

        # Combine user patterns with hardcoded excludes
        all_excludes = list(exclude_patterns) + HARDCODED_EXCLUDES

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
            for pattern in all_excludes:
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
            if ocr_enabled:
                ocr_method = "Ollama Vision" if ocr_mode == "ollama" else "Tesseract"
                warnings.append(
                    f"Found image types ({', '.join(ocr_images_found)}) that will be processed "
                    f"with {ocr_method} to extract text."
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

        # Calculate memory estimates
        memory_estimate = None
        total_memory_with_existing_mb = None

        try:
            app_settings = await repository.get_settings()
            embedding_models = await get_embedding_models()

            embedding_dim = get_embedding_dimension(
                model=app_settings.embedding_model,
                embedding_models=embedding_models,
                tracked_dim=getattr(app_settings, "embedding_dimension", None),
            )

            if embedding_dim:
                # Estimate memory for this index
                mem_est = estimate_index_memory(
                    num_chunks=total_estimated_chunks,
                    embedding_dim=embedding_dim,
                    avg_chunk_chars=chunk_size,
                )

                # Generate dimension comparison table
                dim_breakdown = estimate_memory_at_dimensions(
                    num_chunks=total_estimated_chunks,
                    embedding_models=embedding_models,
                    avg_chunk_chars=chunk_size,
                )

                memory_estimate = MemoryEstimate(
                    embedding_dimension=embedding_dim,
                    steady_memory_mb=round(
                        mem_est["steady_memory_bytes"] / (1024 * 1024), 1
                    ),
                    peak_memory_mb=round(
                        mem_est["peak_memory_bytes"] / (1024 * 1024), 1
                    ),
                    dimension_breakdown=dim_breakdown,
                )

                # Calculate total memory with existing indexes
                existing_indexes = await repository.list_index_metadata()
                existing_memory = sum(
                    (idx.steadyMemoryBytes or 0) for idx in existing_indexes
                )
                total_memory_with_existing_mb = round(
                    (existing_memory + mem_est["steady_memory_bytes"]) / (1024 * 1024),
                    1,
                )

                # Add memory warning if total is high
                if total_memory_with_existing_mb > 8000:  # > 8GB
                    warnings.append(
                        f"Total RAM after adding this index: ~{total_memory_with_existing_mb / 1024:.1f}GB. "
                        "Consider using sequential index loading in Settings to reduce peak memory."
                    )

        except Exception as e:
            logger.debug(f"Could not calculate memory estimate: {e}")

        return IndexAnalysisResult(
            total_files=total_files,
            total_size_bytes=total_size,
            total_size_mb=round(total_size / (1024 * 1024), 2),
            estimated_chunks=total_estimated_chunks,
            estimated_index_size_mb=round(estimated_index_size_mb, 2),
            memory_estimate=memory_estimate,
            total_memory_with_existing_mb=total_memory_with_existing_mb,
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

        # Create optimistic metadata so index shows up in UI immediately
        await self._create_optimistic_index_metadata(
            config=config,
            source_type="upload",
            source=filename,
        )

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
            # Clean up optimistic metadata for failed new indexes
            await self._cleanup_failed_index_metadata(config.name)
            raise

        return job

    async def retry_upload_job(self, failed_job: IndexJob) -> IndexJob:
        """Retry a failed upload job using preserved tmp files.

        Returns a new job that will process the same upload.
        Raises if tmp files no longer exist.
        """
        tmp_path = UPLOAD_TMP_DIR / failed_job.id

        if not tmp_path.exists():
            raise ValueError(
                "Upload files no longer available. Please re-upload the file."
            )

        # Find the archive file (not directories like 'extracted')
        archive_files = [f for f in tmp_path.iterdir() if f.is_file()]
        if not archive_files:
            raise ValueError(
                "Upload archive file not found. Please re-upload the file."
            )

        archive_path = archive_files[0]

        # Create new job with same config
        job_id = str(uuid.uuid4())[:8]
        job = IndexJob(
            id=job_id,
            name=failed_job.name,
            config=failed_job.config,
            source_type="upload",
            source_path=archive_path.name,
        )

        # Persist to database
        await repository.create_job(job)

        # Create optimistic metadata
        await self._create_optimistic_index_metadata(
            config=failed_job.config,
            source_type="upload",
            source=archive_path.name,
        )

        # Cache for active processing
        self._active_jobs[job_id] = job

        # Move tmp files to new job id directory
        new_tmp_dir = UPLOAD_TMP_DIR / job_id
        await asyncio.to_thread(shutil.move, str(tmp_path), str(new_tmp_dir))

        # Clean any previous extraction attempt
        extracted_dir = new_tmp_dir / "extracted"
        if extracted_dir.exists():
            await asyncio.to_thread(shutil.rmtree, extracted_dir, ignore_errors=True)

        # Update archive path for new location
        new_archive_path = new_tmp_dir / archive_path.name

        # Start processing in background
        asyncio.create_task(self._process_upload(job, new_archive_path, new_tmp_dir))

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

        # Create optimistic metadata so index shows up in UI immediately
        await self._create_optimistic_index_metadata(
            config=config,
            source_type="git",
            source=git_url,
            git_branch=branch,
            git_token=git_token,
        )

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
            extract_archive(archive_path, extract_dir)

            # Check for cancellation after extraction
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled after extraction")
                return

            # Find the actual source directory (handle nested zips)
            source_dir = find_source_dir(extract_dir)

            # Create the index
            await self._create_faiss_index(job, source_dir)

            # Only mark completed if not cancelled
            if not self._is_cancelled(job.id):
                job.status = IndexStatus.COMPLETED
                job.completed_at = datetime.utcnow()

        except asyncio.CancelledError:
            # Job was cancelled - status already set by cancel_job()
            logger.info(f"Job {job.id} processing stopped due to cancellation")
            # Clean up optimistic metadata for cancelled new indexes
            await self._cleanup_failed_index_metadata(job.name)

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process upload for job {job.id}")
                job.status = IndexStatus.FAILED
                job.error_message = str(e)
                # Clean up optimistic metadata for failed new indexes
                await self._cleanup_failed_index_metadata(job.name)

        finally:
            # Only cleanup temp directory on success (preserve for retry on failure)
            if job.status == IndexStatus.COMPLETED:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self._cancellation_flags.pop(job.id, None)
            self._active_jobs.pop(job.id, None)

            # Update job status - gracefully handle database disconnection
            try:
                await repository.update_job(job)
            except Exception as db_err:
                logger.warning(
                    f"Job {job.id}: Could not update job status (database may be disconnected): {db_err}"
                )

            # Reinitialize RAG components if job completed successfully
            try:
                await self._maybe_reinitialize_rag(job)
            except Exception as rag_err:
                logger.warning(
                    f"Job {job.id}: Could not reinitialize RAG components: {rag_err}"
                )

    async def _process_git(self, job: IndexJob, skip_clone: bool = False):
        """Process a git repository.

        Git repos are stored persistently in the index directory for efficient re-indexing.
        On re-index, we use `git fetch` to get latest changes instead of a full re-clone.

        Args:
            job: The index job to process
            skip_clone: If True, skip cloning (repo already exists from previous attempt)
        """
        # Persistent location for git repo (survives re-indexing)
        index_dir = self.index_base_path / job.name
        repo_dir = index_dir / ".git_repo"

        # Temp marker for job recovery (if server restarts mid-clone)
        temp_marker_dir = UPLOAD_TMP_DIR / job.id
        clone_complete_marker = temp_marker_dir / ".clone_complete"

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

            # Determine if we have an existing repo to update
            existing_repo = repo_dir.exists() and (repo_dir / ".git").exists()

            if skip_clone and clone_complete_marker.exists():
                # Resuming from a previous attempt - repo should be ready
                logger.info(f"Resuming job {job.id}, using existing repo at {repo_dir}")
            elif existing_repo:
                # Re-indexing: fetch updates instead of full clone
                await self._fetch_git_updates(job, repo_dir)
                clone_complete_marker.parent.mkdir(parents=True, exist_ok=True)
                clone_complete_marker.touch()
            else:
                # Fresh clone
                await self._clone_git_repo(job, repo_dir)
                clone_complete_marker.parent.mkdir(parents=True, exist_ok=True)
                clone_complete_marker.touch()

            # Clear token from job object (not needed in memory after we have stored_token)
            job.git_token = None

            # Clear cloning status, update to scanning phase
            job.error_message = None
            await repository.update_job(job)

            # Check for cancellation after clone/fetch
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled after cloning")
                return

            # Create the index, passing the token for storage in metadata
            await self._create_faiss_index(job, repo_dir, git_token=stored_token)

            # Only mark completed if not cancelled
            if not self._is_cancelled(job.id):
                job.status = IndexStatus.COMPLETED
                job.completed_at = datetime.utcnow()

        except asyncio.CancelledError:
            # Job was cancelled - status already set by cancel_job()
            logger.info(f"Job {job.id} processing stopped due to cancellation")
            # Clean up optimistic metadata for cancelled new indexes
            await self._cleanup_failed_index_metadata(job.name)

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process git for job {job.id}")
                job.status = IndexStatus.FAILED
                job.error_message = str(e)
                # Clean up optimistic metadata for failed new indexes
                await self._cleanup_failed_index_metadata(job.name)

        finally:
            # Clean up temp marker directory
            if job.status in (IndexStatus.COMPLETED, IndexStatus.FAILED):
                await asyncio.to_thread(
                    shutil.rmtree, temp_marker_dir, ignore_errors=True
                )
            self._cancellation_flags.pop(job.id, None)
            self._active_jobs.pop(job.id, None)

            # Update job status - gracefully handle database disconnection
            try:
                await repository.update_job(job)
            except Exception as db_err:
                logger.warning(
                    f"Job {job.id}: Could not update job status (database may be disconnected): {db_err}"
                )

            # Reinitialize RAG components if job completed successfully
            try:
                await self._maybe_reinitialize_rag(job)
            except Exception as rag_err:
                logger.warning(
                    f"Job {job.id}: Could not reinitialize RAG components: {rag_err}"
                )

    async def _clone_git_repo(self, job: IndexJob, repo_dir: Path) -> None:
        """Clone a git repository to the specified directory.

        Args:
            job: The index job with git configuration
            repo_dir: Target directory for the clone
        """
        logger.info(f"Cloning {job.git_url} branch {job.git_branch}")

        job.error_message = "Cloning repository..."
        await repository.update_job(job)

        if not job.git_url:
            raise ValueError("Git URL is required for git source type")

        # Ensure parent directory exists and target is clean
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists():
            await asyncio.to_thread(shutil.rmtree, repo_dir)

        clone_url = build_authenticated_git_url(job.git_url, job.git_token)
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

        clone_timeout_minutes = self._compute_clone_timeout_minutes(job.config)
        clone_timeout_seconds = clone_timeout_minutes * 60
        history_depth = getattr(job.config, "git_history_depth", 1)

        # Build git clone command based on depth setting
        git_args = ["git", "clone", "--progress"]
        if history_depth == 0:
            logger.info(
                "Cloning with full history (may take a long time for large repos)"
            )
        elif history_depth == 1:
            git_args.extend(["--depth", "1"])
            logger.info("Shallow clone (latest commit only)")
        else:
            git_args.extend(["--depth", str(history_depth)])
            logger.info(f"Cloning with depth {history_depth} commits")

        git_args.extend(
            [
                "--branch",
                job.git_branch or "main",
                clone_url,
                str(repo_dir),
            ]
        )

        logger.info(f"Clone timeout set to {clone_timeout_minutes} minutes")

        try:
            process = await asyncio.create_subprocess_exec(
                *git_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            stderr_output = await self._stream_clone_progress(
                process, job, clone_timeout_seconds
            )
        except TimeoutError as exc:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            # Clean up partial clone on timeout
            await asyncio.to_thread(shutil.rmtree, repo_dir, ignore_errors=True)
            error_msg = (
                f"Git clone timed out after {clone_timeout_minutes} minutes. "
                "The repository may be too large or the network connection is slow. "
                "Try increasing the clone timeout in Advanced Options."
            )
            logger.error(f"Job {job.id}: {error_msg}")
            raise RuntimeError(error_msg) from exc

        if process.returncode != 0:
            # Clean up failed clone
            await asyncio.to_thread(shutil.rmtree, repo_dir, ignore_errors=True)
            error_msg = stderr_output
            if (
                "could not read Username" in error_msg
                or "Authentication failed" in error_msg
            ):
                raise RuntimeError(
                    "Git clone failed: Authentication required. "
                    "This is a private repository - please provide a valid access token."
                )
            raise RuntimeError(f"Git clone failed: {error_msg}")

        logger.info("Clone complete")

    async def _fetch_git_updates(self, job: IndexJob, repo_dir: Path) -> None:
        """Fetch updates from remote and reset to latest.

        Handles depth changes:
        - If requesting more history than we have, uses --deepen
        - If requesting full history from shallow, uses --unshallow
        - Otherwise just fetches latest commits

        Args:
            job: The index job with git configuration
            repo_dir: Existing git repository directory
        """
        if not job.git_url:
            raise ValueError("Git URL is required for git source type")

        logger.info(f"Fetching updates for {job.git_url} branch {job.git_branch}")
        job.error_message = "Fetching latest changes..."
        await repository.update_job(job)

        # Update remote URL with current token (in case token changed)
        fetch_url = build_authenticated_git_url(job.git_url, job.git_token)
        subprocess.run(
            ["git", "remote", "set-url", "origin", fetch_url],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        clone_timeout_minutes = self._compute_clone_timeout_minutes(job.config)
        clone_timeout_seconds = clone_timeout_minutes * 60
        history_depth = getattr(job.config, "git_history_depth", 1)

        # Check current depth of repo
        depth_result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        current_depth = (
            int(depth_result.stdout.strip()) if depth_result.returncode == 0 else 1
        )

        # Check if repo is shallow
        is_shallow = (repo_dir / ".git" / "shallow").exists()

        # Determine fetch strategy
        branch = job.git_branch or "main"
        if history_depth == 0 and is_shallow:
            logger.info("Unshallowing repo for full history")
            fetch_args = ["git", "fetch", "--unshallow", "--progress", "origin", branch]
        elif history_depth > current_depth and is_shallow:
            deepen_amount = history_depth - current_depth
            logger.info(
                f"Deepening repo from {current_depth} to {history_depth} commits (+{deepen_amount})"
            )
            fetch_args = [
                "git",
                "fetch",
                f"--deepen={deepen_amount}",
                "--progress",
                "origin",
                branch,
            ]
        else:
            logger.info(
                f"Fetching latest changes (depth: {current_depth} -> {history_depth})"
            )
            fetch_args = ["git", "fetch", "--progress", "origin", branch]

        try:
            process = await asyncio.create_subprocess_exec(
                *fetch_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repo_dir,
                env=env,
            )
            stderr_output = await self._stream_clone_progress(
                process, job, clone_timeout_seconds
            )
        except TimeoutError as exc:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            error_msg = (
                f"Git fetch timed out after {clone_timeout_minutes} minutes. "
                "Try increasing the clone timeout in Advanced Options."
            )
            logger.error(f"Job {job.id}: {error_msg}")
            raise RuntimeError(error_msg) from exc

        if process.returncode != 0:
            error_msg = stderr_output
            if (
                "could not read Username" in error_msg
                or "Authentication failed" in error_msg
            ):
                raise RuntimeError(
                    "Git fetch failed: Authentication required. "
                    "This is a private repository - please provide a valid access token."
                )
            raise RuntimeError(f"Git fetch failed: {error_msg}")

        # Reset to the fetched branch
        reset_result = subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if reset_result.returncode != 0:
            raise RuntimeError(f"Git reset failed: {reset_result.stderr}")

        logger.info("Fetch complete")

    async def _stream_clone_progress(
        self, process: Any, job: IndexJob, timeout_seconds: int
    ) -> str:
        """Stream git clone progress and update job status.

        Args:
            process: The git clone subprocess
            job: The index job to update
            timeout_seconds: Timeout for the entire operation

        Returns:
            The complete stderr output as a string
        """
        stderr_chunks: List[str] = []
        last_update_time = asyncio.get_event_loop().time()
        update_interval = 1.0  # Update job status at most once per second

        # Pattern to match git progress output like:
        # "Receiving objects:  45% (12345/27000), 156.00 MiB | 5.23 MiB/s"
        progress_pattern = re.compile(
            r"(Receiving objects|Resolving deltas|Updating files):\s+(\d+)%"
        )

        async def read_with_timeout():
            start_time = asyncio.get_event_loop().time()
            nonlocal last_update_time
            last_update_time = start_time

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout_seconds - elapsed

                if remaining <= 0:
                    raise TimeoutError("Clone operation timed out")

                try:
                    # Read a chunk of stderr (git progress uses \r for updates)
                    chunk = await asyncio.wait_for(
                        process.stderr.read(1024),  # type: ignore[union-attr]
                        timeout=min(remaining, 5.0),  # Check every 5 seconds max
                    )

                    if not chunk:
                        break  # EOF

                    text = chunk.decode("utf-8", errors="replace")
                    stderr_chunks.append(text)

                    # Parse progress and update job
                    current_time = asyncio.get_event_loop().time()

                    if current_time - last_update_time >= update_interval:
                        # Find the latest progress percentage
                        match = progress_pattern.search(text)
                        if match:
                            phase = match.group(1)
                            percent = match.group(2)
                            job.error_message = f"Cloning: {phase} {percent}%"
                            await repository.update_job(job)
                            last_update_time = current_time

                except asyncio.TimeoutError:
                    # Just a read timeout, check if process is still running
                    if process.returncode is not None:
                        break
                    continue

            # Wait for process to complete
            await process.wait()

        await read_with_timeout()

        # Clear the cloning message
        job.error_message = None
        await repository.update_job(job)

        return "".join(stderr_chunks)

    async def _index_git_history(
        self, repo_dir: Path, index_name: str, depth: int
    ) -> List:
        """Extract git commit history as documents for indexing.

        Args:
            repo_dir: Path to the git repository
            index_name: Name of the index for metadata
            depth: Number of commits to index (0 = all)

        Returns:
            List of LangChain Document objects with commit information
        """
        from langchain_core.documents import Document as LangChainDocument

        documents: List[LangChainDocument] = []

        try:
            # Build git log command with stats and filenames
            # Use a unique separator to split commits reliably
            commit_sep = "---COMMIT_SEPARATOR---"
            # Format: hash|author|date|subject, followed by body
            # --stat gives per-file stats AND summary line with totals
            # Format: " filename | +N -M" per file, then " N files changed, X insertions(+), Y deletions(-)"
            log_format = f"{commit_sep}%H|%an|%aI|%s%n%b"
            cmd = ["git", "log", f"--format={log_format}", "--stat"]
            if depth > 0:
                cmd.extend(["-n", str(depth)])

            # Run git log in thread pool to avoid blocking event loop
            def run_git_log():
                result = subprocess.run(
                    cmd,
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for large histories
                    check=False,
                )
                return result.returncode, result.stdout, result.stderr

            returncode, stdout, stderr = await asyncio.to_thread(run_git_log)

            if returncode != 0:
                logger.warning(f"Git log failed: {stderr}")
                return documents

            # Parse commits in thread pool for large histories
            def parse_commits(log_output: str) -> List[LangChainDocument]:
                parsed_docs: List[LangChainDocument] = []
                commit_blocks = log_output.split(commit_sep)

                for block in commit_blocks:
                    block = block.strip()
                    if not block:
                        continue

                    lines = block.split("\n")
                    if not lines:
                        continue

                    # First line is the header: hash|author|date|subject
                    header_parts = lines[0].split("|")
                    if len(header_parts) < 4:
                        continue

                    commit_hash = header_parts[0]
                    author = header_parts[1]
                    date = header_parts[2]
                    subject = "|".join(header_parts[3:])  # Subject may contain |

                    # Parse remaining lines for --stat format:
                    # - Body lines (commit message after subject)
                    # - File stat lines: " filename | N +/-" or " filename | Bin X -> Y"
                    # - Summary line: " N files changed, X insertions(+), Y deletions(-)"
                    body_lines: List[str] = []
                    file_names: List[str] = []
                    total_additions = 0
                    total_deletions = 0
                    files_changed = 0
                    in_stat_section = False

                    for line in lines[1:]:
                        # Check for file stat line: " filename | N +/-" pattern
                        # Example: " README.md | 2 +-"
                        if " | " in line and not in_stat_section:
                            in_stat_section = True

                        if in_stat_section:
                            # Summary line: " 3 files changed, 45 insertions(+), 12 deletions(-)"
                            if "files changed" in line or "file changed" in line:
                                parts = line.split(",")
                                for part in parts:
                                    part = part.strip()
                                    if "file" in part:
                                        try:
                                            files_changed = int(part.split()[0])
                                        except (ValueError, IndexError):
                                            pass
                                    elif "insertion" in part:
                                        try:
                                            total_additions = int(part.split()[0])
                                        except (ValueError, IndexError):
                                            pass
                                    elif "deletion" in part:
                                        try:
                                            total_deletions = int(part.split()[0])
                                        except (ValueError, IndexError):
                                            pass
                            elif " | " in line:
                                # File stat line: extract just the filename
                                # Format: " path/to/file.ext | 42 ++++---"
                                file_part = line.split(" | ")[0].strip()
                                if file_part:
                                    file_names.append(file_part)
                        elif line.strip():
                            # Before stat section, it's part of the commit body
                            body_lines.append(line)

                    body = "\n".join(body_lines).strip()

                    # Create a searchable document with compact formatting
                    content = f"[Commit {commit_hash[:8]}] {subject}\nAuthor: {author}\nDate: {date}"

                    # Add stats summary
                    if files_changed > 0:
                        content += f"\nChanges: +{total_additions}/-{total_deletions} in {files_changed} file(s)"

                    # Add file list on same line or next line (no blank line)
                    if file_names:
                        if len(file_names) <= 10:
                            content += f"\nFiles: {', '.join(file_names)}"
                        else:
                            content += f"\nFiles: {', '.join(file_names[:10])}, +{len(file_names) - 10} more"

                    # Add body if present
                    if body and body.strip() != "":
                        content += f"\nMessage:\n{body}"

                    doc = LangChainDocument(
                        page_content=content,
                        metadata={
                            "source": f"git:commit:{commit_hash[:8]}",
                            "index_name": index_name,
                            "type": "git_commit",
                            "commit_hash": commit_hash,
                            "author": author,
                            "date": date,
                            "additions": total_additions,
                            "deletions": total_deletions,
                            "files_changed": files_changed,
                        },
                    )
                    parsed_docs.append(doc)

                return parsed_docs

            documents = await asyncio.to_thread(parse_commits, stdout)
            logger.info(f"Extracted {len(documents)} commits from git history")

        except subprocess.TimeoutExpired:
            logger.warning("Git log timed out - repository may have very large history")
        except Exception as e:
            logger.warning(f"Failed to index git history: {e}")

        return documents

    async def _create_faiss_index(
        self, job: IndexJob, source_dir: Path, git_token: Optional[str] = None
    ):
        """Create FAISS index from source directory."""
        from langchain_community.document_loaders import TextLoader
        from langchain_community.vectorstores import FAISS

        config = job.config

        # Collect all files to process (run in thread pool as rglob can be slow on large repos)
        logger.info(
            f"Job {job.id}: Scanning directory for matching files (this may take a while for large repos)..."
        )
        job.error_message = "Scanning files..."
        await repository.update_job(job)

        def collect_files_sync() -> List[Path]:
            """Synchronous file collection - runs in thread pool."""
            files = []
            # Combine user patterns with hardcoded excludes (module-level constant)
            all_excludes = list(config.exclude_patterns) + HARDCODED_EXCLUDES

            # Separate file-extension patterns from path patterns
            # File-extension patterns (*.js, *.min.js) are matched against filename only
            # Path patterns (**/, */) are matched against the full relative path
            filename_excludes = []
            path_excludes = []
            for exc in all_excludes:
                # Pattern is a filename pattern if it starts with * but has no path separators
                if exc.startswith("*") and "/" not in exc and "\\" not in exc:
                    filename_excludes.append(exc)
                else:
                    path_excludes.append(exc)

            # Add hardcoded minified patterns to filename excludes
            filename_excludes.extend(MINIFIED_PATTERNS)

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
                        rel_path_str = str(rel_path)
                        filename = file_path.name

                        skip = False

                        # Check filename-based excludes (extension patterns like *.min.js)
                        for exc_pattern in filename_excludes:
                            if fnmatch.fnmatch(filename, exc_pattern):
                                logger.debug(
                                    f"Skipping file matching pattern {exc_pattern}: {rel_path_str}"
                                )
                                skip = True
                                break

                        # Check path-based excludes (directory patterns like **/vendor/**)
                        if not skip:
                            for exc_pattern in path_excludes:
                                clean_exc = exc_pattern.removeprefix("**/")
                                if fnmatch.fnmatch(rel_path_str, clean_exc):
                                    skip = True
                                    break

                        if not skip and file_path not in files:
                            files.append(file_path)
            return files

        all_files = await asyncio.to_thread(collect_files_sync)

        job.total_files = len(all_files)
        job.error_message = None  # Clear "Scanning files..." message
        await repository.update_job(job)
        logger.info(f"Found {len(all_files)} files to index")

        # BINARY_EXTENSIONS is imported from ragtime.core.file_constants

        # Load documents in parallel
        # Use asyncio.Semaphore to limit concurrent file loads (prevents memory spikes
        # and I/O saturation). Files are loaded concurrently while respecting limits.
        documents = []
        skipped_binary = 0
        ocr_mode = config.ocr_mode
        ocr_vision_model = config.ocr_vision_model
        ocr_enabled = ocr_mode != OcrMode.DISABLED

        # Concurrency limit: balance between parallelism and resource usage
        # Higher values improve throughput but increase memory and I/O pressure
        max_concurrent_loads = min(32, os.cpu_count() or 8)
        load_semaphore = asyncio.Semaphore(max_concurrent_loads)

        # For Ollama vision mode, we need async extraction and the base URL
        ollama_base_url = None
        if ocr_mode == OcrMode.OLLAMA:
            settings = await get_app_settings()
            ollama_base_url = settings.get("ollama_base_url")

        async def load_file_async(file_path: Path) -> tuple[Path, List, str | None]:
            """Async file loading with OCR support. Returns (path, docs, error)."""
            from langchain_core.documents import Document as LangChainDocument

            from ragtime.indexer.document_parser import extract_text_from_file_async

            async with load_semaphore:
                try:
                    ext_lower = file_path.suffix.lower()

                    # Use document parser for Office/PDF files and images (when OCR enabled)
                    if ext_lower in PARSEABLE_DOCUMENT_EXTENSIONS or (
                        ocr_enabled and ext_lower in OCR_EXTENSIONS
                    ):
                        content = await extract_text_from_file_async(
                            file_path,
                            ocr_mode=ocr_mode.value,
                            ocr_vision_model=ocr_vision_model,
                            ollama_base_url=ollama_base_url,
                        )
                        if content:
                            return (
                                file_path,
                                [LangChainDocument(page_content=content)],
                                None,
                            )
                        return (file_path, [], None)

                    # Use TextLoader for plain text files (run in thread pool)
                    def load_text():
                        loader = TextLoader(str(file_path), autodetect_encoding=True)
                        return loader.load()

                    docs = await asyncio.to_thread(load_text)
                    return (file_path, docs, None)
                except Exception as e:
                    return (file_path, [], str(e))

        # Build list of files to load (filtering out unparseable binaries)
        files_to_load = []
        for file_path in all_files:
            # Check for cancellation before processing
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled during file preparation")
                raise asyncio.CancelledError("Job cancelled by user")

            ext_lower = file_path.suffix.lower()

            # Skip truly unparseable binary files (executables, etc.)
            # When OCR is enabled, allow image files to be processed
            if ext_lower in UNPARSEABLE_BINARY_EXTENSIONS:
                if ocr_enabled and ext_lower in OCR_EXTENSIONS:
                    # Process image with OCR
                    ocr_method = (
                        "Ollama Vision" if ocr_mode == OcrMode.OLLAMA else "Tesseract"
                    )
                    logger.debug(f"Processing image {file_path.name} with {ocr_method}")
                    files_to_load.append(file_path)
                else:
                    skipped_binary += 1
                    job.processed_files += 1
            else:
                # Log document parsing (no longer a warning since we can parse them)
                if ext_lower in PARSEABLE_DOCUMENT_EXTENSIONS:
                    logger.debug(
                        f"Parsing document file {file_path.name} with document parser"
                    )
                files_to_load.append(file_path)

        # Load files in parallel batches
        # Process in batches to allow periodic progress updates and cancellation checks
        batch_size = max_concurrent_loads * 2  # 2x concurrency for good pipeline
        total_to_load = len(files_to_load)
        logger.info(
            f"Loading {total_to_load} files with {max_concurrent_loads} concurrent workers"
        )

        for batch_start in range(0, total_to_load, batch_size):
            # Check for cancellation at batch boundaries
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled during file loading")
                raise asyncio.CancelledError("Job cancelled by user")

            batch_end = min(batch_start + batch_size, total_to_load)
            batch_files = files_to_load[batch_start:batch_end]

            # Load all files in this batch concurrently
            load_tasks = [load_file_async(fp) for fp in batch_files]
            results = await asyncio.gather(*load_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, BaseException):
                    # Unexpected exception during gather
                    logger.debug(f"File load exception: {result}")
                    job.processed_files += 1
                    continue

                # result is now guaranteed to be tuple[Path, List, str | None]
                file_path, docs, error = result
                if error:
                    logger.debug(f"Skipped {file_path.name}: {error}")
                    job.processed_files += 1
                    continue

                # Add source metadata
                rel_path_str = str(file_path.relative_to(source_dir))
                for doc in docs:
                    doc.metadata["source"] = rel_path_str
                    doc.metadata["index_name"] = job.name

                documents.extend(docs)
                job.processed_files += 1

            # Update progress after each batch
            await repository.update_job(job)
            logger.info(f"Loaded {job.processed_files}/{job.total_files} files")

            # Brief yield to event loop between batches
            await asyncio.sleep(0)

        if skipped_binary > 0:
            logger.info(f"Skipped {skipped_binary} binary files")

        # Index git commit history if depth > 1
        history_depth = getattr(config, "git_history_depth", 1)
        if history_depth != 1:
            commit_docs = await self._index_git_history(
                source_dir, job.name, history_depth
            )
            if commit_docs:
                documents.extend(commit_docs)
                logger.info(f"Added {len(commit_docs)} commit history documents")

        if not documents:
            raise ValueError("No documents were loaded")

        logger.info(f"Loaded {len(documents)} documents, splitting into chunks...")

        # Get app settings for chunking configuration
        app_settings = await repository.get_settings()

        # Determine if we should use token-based chunking
        use_tokens = getattr(app_settings, "chunking_use_tokens", True)
        if use_tokens:
            logger.info("Using token-based chunking for accurate chunk sizes")
        else:
            logger.info("Using character-based chunking")

        # Split into chunks using parallel process pool
        # This runs CPU-intensive tiktoken work in separate processes,
        # leaving the main event loop responsive for API/UI/MCP
        from ragtime.indexer.chunking import chunk_documents_parallel

        chunks = await chunk_documents_parallel(
            documents=documents,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_tokens=use_tokens,
            batch_size=100,  # Larger batches for efficiency
            progress_callback=None,  # Progress logged from the function itself
        )

        job.total_chunks = len(chunks)
        await repository.update_job(job)

        logger.info(f"Created {len(chunks)} chunks, generating embeddings...")

        # Get embeddings model
        embeddings = await self._get_embeddings(app_settings)

        logger.info(
            f"Using {app_settings.embedding_provider} embeddings with model: {app_settings.embedding_model}"
        )

        # Process in batches to show progress and throttle embedding calls
        from ragtime.indexer.vector_utils import EMBEDDING_SUB_BATCH_SIZE

        batch_size = EMBEDDING_SUB_BATCH_SIZE
        batch_pause_seconds = 0.5
        max_retries = 5
        db = None

        # Get embedding model context limit dynamically from LiteLLM or Ollama API
        from ragtime.core.embedding_models import get_embedding_model_context_limit
        from ragtime.core.tokenization import count_tokens

        max_allowed_tokens = await get_embedding_model_context_limit(
            model_name=app_settings.embedding_model,
            provider=app_settings.embedding_provider,
            ollama_base_url=app_settings.ollama_base_url,
        )
        logger.debug(
            f"Embedding model context limit: {max_allowed_tokens} tokens "
            f"(provider: {app_settings.embedding_provider}, model: {app_settings.embedding_model})"
        )

        # Validate and truncate oversized chunks before embedding
        # Some chunks may exceed the limit due to headers, overlap, or small files
        # that weren't split but combined with headers exceed the limit
        #
        # IMPORTANT: We use tiktoken (cl100k_base) for counting, but embedding models
        # may use different tokenizers (e.g., BERT WordPiece for nomic-embed-text).
        # BERT tokenizers typically produce MORE tokens than tiktoken for the same text
        # because they have smaller vocabularies (~30k vs ~100k tokens).
        # We use aggressive safety margins to account for this mismatch.
        truncated_count = 0

        # Get safety margin based on embedding provider
        safety_margin = get_embedding_safety_margin(app_settings.embedding_provider)

        for chunk in chunks:
            tokens = count_tokens(chunk.page_content)
            if tokens > max_allowed_tokens:
                source = chunk.metadata.get("source", "unknown")
                chunker = chunk.metadata.get("chunker", "unknown")
                # Truncate to fit - use safety margin to account for tokenizer differences
                content = chunk.page_content
                target_chars = int(
                    len(content) * (max_allowed_tokens / tokens) * safety_margin
                )
                chunk.page_content = content[:target_chars]

                # Verify the truncated content is under limit (re-count)
                new_tokens = count_tokens(chunk.page_content)
                if truncated_count < 5:  # Log first 5
                    logger.warning(
                        f"Truncated oversized chunk from {tokens} to {new_tokens} tokens "
                        f"(source: {source}, chunker: {chunker}, target: {max_allowed_tokens})"
                    )
                truncated_count += 1

        if truncated_count > 0:
            logger.warning(
                f"Truncated {truncated_count} oversized chunks to fit embedding context limit "
                f"(safety_margin={safety_margin:.0%})"
            )

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

        # Auto-generate description if not provided, but preserve existing description if available
        description = getattr(config, "description", "")
        if not description:
            # Check if there's an existing description in the database
            existing_metadata = await repository.get_index_metadata(job.name)
            if existing_metadata and existing_metadata.description:
                description = existing_metadata.description
                logger.info(f"Preserving existing description for {job.name}")
            else:
                description = await generate_index_description(
                    index_name=job.name,
                    documents=documents,
                    source_type=job.source_type,
                    source=job.git_url or job.source_path,
                )
                logger.info(f"Generated new description for {job.name}")

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
            vector_store_type=config.vector_store_type,
        )

        logger.info(f"Index {job.name} created successfully!")


# Global indexer instance - uses configured path
indexer = IndexerService(index_base_path=settings.index_data_path)
