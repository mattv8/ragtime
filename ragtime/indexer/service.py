"""
FAISS Indexer Service - Creates and manages vector indexes.

Job Recovery: Jobs are persisted to the database before processing starts.
If the server restarts mid-job (e.g., hot-reload), jobs in 'pending' or
'processing' state are automatically resumed on startup.
"""

import asyncio
import functools
import json
import os
import pickle
import re
import shutil
import sqlite3
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from ragtime.config import settings
from ragtime.core import openrouter
from ragtime.core.app_settings import get_app_settings, invalidate_settings_cache
from ragtime.core.copilot_auth import ensure_copilot_token_fresh
from ragtime.core.datetimes import utc_now
from ragtime.core.embedding_models import (
    get_embedding_model_context_limit,
    get_embedding_models,
)
from ragtime.core.file_constants import (
    MINIFIED_PATTERNS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
    get_embedding_safety_margin,
)
from ragtime.core.logging import get_logger
from ragtime.core.model_providers import normalize_provider_name, resolve_provider_api_key, resolve_provider_base_url
from ragtime.core.tokenization import count_tokens
from ragtime.core.userspace_limits import (
    ARCHIVE_MAX_FILE_COUNT_DEFAULT,
    ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES,
)
from ragtime.indexer.chunking import (
    chunk_documents_parallel,
    chunking_implementation_fingerprint,
    is_context_length_error,
    pool_manager,
    rechunk_documents_batch,
    rechunk_oversized_content,
)
from ragtime.indexer.document_parser import OCR_EXTENSIONS, extract_text_from_file_async, extract_text_from_file_process_safe
from ragtime.indexer.embedding_errors import iter_exception_chain
from ragtime.indexer.faiss_artifacts import prepare_faiss_artifact
from ragtime.indexer.file_utils import (
    build_authenticated_git_url,
    collect_files_recursive,
    extract_archive,
    find_source_dir,
    get_directory_size_bytes,
    get_matching_file_pattern,
    get_matching_pattern,
    git_clone_fraction_from_line,
    is_excluded_by_patterns,
    is_excluded_directory,
    should_index_file_type,
)
from ragtime.indexer.indexing_pipeline import DEFAULT_BATCH_TEXT_BYTES, MAX_SOURCE_TEXT_BYTES, BoundedIndexingPipeline
from ragtime.indexer.indexing_spool import IndexingSpool
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
    IndexJobPhase,
    IndexStatus,
    MemoryEstimate,
    OcrMode,
    OcrProvider,
    VectorStoreType,
)
from ragtime.indexer.repository import repository
from ragtime.indexer.resource_governor import ResourceRequest, resource_governor
from ragtime.indexer.resource_workers import run_resource_task
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_utils import (
    EMBEDDING_SUB_BATCH_SIZE,
    append_embedding_dimension_warning,
    count_faiss_docstore_stats,
    embed_documents_subbatched,
    get_embeddings_model,
)

logger = get_logger(__name__)

ANALYZE_ONLY_GIT_TOKEN_METADATA_TTL_SECONDS = 24 * 60 * 60

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

    Uses the LLM provider configured in app settings.
    """
    try:
        app_settings = await get_app_settings()
        provider = app_settings.get("llm_provider", "openai").lower()

        # Get the appropriate LLM based on configured provider
        llm: Any = None

        if provider == "ollama":
            try:
                from langchain_ollama import (
                    ChatOllama,
                )  # type: ignore[reportMissingImports]

                from ragtime.core.ollama import (
                    KEEP_ALIVE,
                    NUM_GPU,
                    get_model_details,
                    has_capability,
                )

                base_url = app_settings.get("ollama_base_url", "http://localhost:11434")
                model = app_settings.get("llm_model", "llama3.2")

                # Detect thinking capability
                reasoning = None
                try:
                    details = await get_model_details(model, base_url)
                    if details and has_capability(details, "thinking"):
                        reasoning = True
                        logger.debug(f"Ollama model '{model}' supports thinking; enabling reasoning mode")
                except Exception:
                    pass

                llm = ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=0.3,
                    num_gpu=NUM_GPU,
                    keep_alive=KEEP_ALIVE,
                    reasoning=reasoning,
                )
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
                from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

                # Use cheaper model for descriptions
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.3,
                    api_key=SecretStr(str(api_key)),
                )
                logger.debug("Using OpenAI for description generation: gpt-4o-mini")
            else:
                logger.debug("OpenAI selected but no API key configured")

        elif provider == "openrouter":
            api_key = resolve_provider_api_key(app_settings, "openrouter", "llm")
            if api_key:
                from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

                model = app_settings.get("llm_model", "openai/gpt-4o-mini")
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    api_key=SecretStr(str(api_key)),
                    base_url=openrouter.DEFAULT_BASE_URL,
                )
                logger.debug(f"Using OpenRouter for description generation: {model}")
            else:
                logger.debug("OpenRouter selected but no API key configured")

        elif provider == "github_copilot":
            # GitHub Copilot uses OAuth flow. Refresh the short-lived HMAC token
            # before calling the Copilot API.
            refreshed = await ensure_copilot_token_fresh()
            token = refreshed or app_settings.get("github_copilot_access_token", "")
            if token:
                from langchain_openai import ChatOpenAI

                base_url = app_settings.get("github_copilot_base_url") or "https://api.githubcopilot.com"
                model = app_settings.get("llm_model", "gpt-4o")
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    api_key=SecretStr(str(token)),
                    base_url=str(base_url).rstrip("/"),
                    default_headers={
                        "Openai-Intent": "conversation-edits",
                        "x-initiator": "agent",
                        "User-Agent": "ragtime",
                    },
                )
                logger.debug(f"Using GitHub Copilot for description generation: {model}")
            else:
                logger.debug("GitHub Copilot selected but no OAuth token configured")

        elif provider == "github_models":
            token = app_settings.get("github_models_api_token", "")
            if token:
                from langchain_openai import ChatOpenAI

                model = app_settings.get("llm_model", "openai/gpt-4.1")
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    api_key=SecretStr(str(token)),
                    base_url="https://models.github.ai/inference",
                    default_headers={
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                        "User-Agent": "ragtime",
                    },
                )
                logger.debug(f"Using GitHub Models for description generation: {model}")
            else:
                logger.debug("GitHub Models selected but no PAT token configured")

        if llm is None:
            logger.debug(f"No LLM available for description generation (provider: {provider})")
            return ""

        # Sample file paths for context (unique paths only)
        file_paths = list(set(doc.metadata.get("source", "unknown") for doc in documents[:100]))[:20]

        # Sample some content snippets
        content_samples = [doc.page_content[:500] for doc in documents[:5]]

        prompt = f"""Analyze this indexed content and write a brief description (1-2 sentences) for an AI assistant to understand what knowledge is available.

The index may contain anything that has been ingested: source code, technical documentation, business records, scanned paperwork, contracts, policies, manuals, drawings, reference data, or any other material. Do not assume it is a codebase unless the sample evidence clearly indicates that.

Index name: {index_name}
Source type: {source_type}
Source: {source or "uploaded archive"}

Sample file paths:
{chr(10).join(f"- {p}" for p in file_paths)}

Sample content snippets:
{chr(10).join(f"---{chr(10)}{s}{chr(10)}" for s in content_samples)}

Write a concise description focusing on:
- What kind of content this index actually contains (code, documentation, business records, paperwork, etc.) based on the samples
- Key topics, domains, technologies, or document types covered
- What kinds of questions this index can help answer

Description:"""

        response = await llm.ainvoke(prompt)
        content = response.content
        description = content.strip() if isinstance(content, str) else str(content)
        logger.info(f"Auto-generated description for {index_name}: {description[:100]}...")
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
        # Strong references to background tasks so they are not garbage-collected.
        # asyncio only keeps weak references to tasks created via create_task().
        self._processing_tasks: Dict[str, asyncio.Task] = {}
        # Successful attempts remain available until the published generation
        # has been verified by the live RAG loader.
        self._completed_spools: Dict[str, Path] = {}
        # Cancellation flags for cooperative cancellation
        self._cancellation_flags: Dict[str, bool] = {}
        self._git_job_creation_lock = asyncio.Lock()

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

    async def _reinitialize_rag_components(self, index_name: Optional[str] = None) -> bool:
        """
        Refresh RAG components to load newly created indexes.

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

            logger.info("Refreshing RAG components to load new index")
            invalidate_settings_cache()
            if index_name:
                loaded = await rag.load_faiss_index_from_metadata(index_name)
                if loaded:
                    logger.info(f"RAG components refreshed successfully for index '{index_name}'")
                    return True
                else:
                    logger.warning(f"RAG refresh completed but index '{index_name}' was not loaded")
                    return False
            else:
                await rag.initialize()
                logger.info("RAG components reinitialized successfully")
                return True
        except Exception as e:
            # Log but don't fail the indexing job if RAG reinitialization fails
            logger.warning(f"Failed to reinitialize RAG components: {e}")
            return False

    async def _maybe_reinitialize_rag(self, job: IndexJob) -> bool:
        """
        Conditionally reinitialize RAG components for completed jobs.

        This is called before publishing completed job status to ensure:
        1. Completed jobs are queryable as soon as completion is visible
        2. Reinitialize errors don't affect job completion status
        3. New index is loaded even if there were warnings during indexing

        Args:
            job: The index job to check
        """
        if job.status == IndexStatus.COMPLETED:
            loaded = await self._reinitialize_rag_components(job.name)
            spool_root = self._completed_spools.pop(job.id, None) if loaded else None
            if spool_root is not None:
                await asyncio.to_thread(shutil.rmtree, spool_root, ignore_errors=True)
            return loaded
        return False

    async def recover_interrupted_jobs(self) -> int:
        """
        Recover jobs that were interrupted by a server restart.

        Called during application startup. Finds jobs in 'pending' or 'processing'
        state and resumes them. Also cleans up orphaned directories.

        Returns:
            Number of jobs recovered
        """
        jobs = await repository.list_jobs()
        await self._reconcile_published_spools(jobs)
        interrupted = [j for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)]

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
                    job.phase = IndexJobPhase.FAILED
                    job.error_message = f"Recovery failed: {e}"
                    job.completed_at = utc_now()
                    await repository.update_job(job)

        # Clean up orphaned directories (always run, after job recovery)
        await self._cleanup_orphaned_tmp_dirs()
        await self._cleanup_orphaned_git_repos()

        return recovered

    async def _reconcile_published_spools(self, jobs: List[IndexJob]) -> None:
        """Remove only journals durably published by a terminal completed job.

        A process restart loses ``_completed_spools``.  The journal markers
        below make recovery safe without treating failed/resumable attempts or
        persistent git clones as disposable.
        """
        for job in jobs:
            if job.status != IndexStatus.COMPLETED:
                continue
            metadata = await repository.get_index_metadata(job.name)
            if not metadata or not getattr(metadata, "path", None):
                continue
            job_root = UPLOAD_TMP_DIR / "indexing" / job.id
            if not job_root.is_dir():
                continue
            for attempt in job_root.iterdir():
                if not attempt.is_dir() or attempt.is_symlink():
                    continue
                try:
                    spool = await asyncio.to_thread(IndexingSpool.open_existing, attempt)
                    try:
                        published = spool.get_state("published_generation")
                        index_name = spool.get_state("published_index_name")
                    finally:
                        await asyncio.to_thread(spool.close)
                    if index_name == job.name and published and Path(str(published)).resolve() == Path(metadata.path).resolve():
                        await asyncio.to_thread(shutil.rmtree, attempt, ignore_errors=True)
                except (OSError, ValueError, sqlite3.Error):
                    continue

    async def _cleanup_orphaned_tmp_dirs(self) -> None:
        """Remove tmp directories for jobs that no longer exist or are completed."""
        if not UPLOAD_TMP_DIR.exists():
            return

        jobs = await repository.list_jobs()
        # Keep tmp dirs for: active jobs (pending/processing) and failed upload jobs (for retry)
        keep_job_ids = {
            j.id for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING) or (j.status == IndexStatus.FAILED and j.source_type == "upload")
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
        active_index_names = {j.name for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)}

        for index_dir in self.index_base_path.iterdir():
            if not index_dir.is_dir() or index_dir.name.startswith("_"):
                continue

            git_repo = index_dir / ".git_repo"
            # A published document artifact may live in an immutable generation.
            faiss_index = index_dir / "index.faiss"
            generations = index_dir / ".generations"
            has_completed_generation = generations.is_dir() and any(
                child.is_dir() and (child / "index.faiss").is_file() and (child / "index.pkl").is_file() for child in generations.iterdir()
            )

            # If there's a git repo but no FAISS index, it might be orphaned
            if git_repo.exists() and not faiss_index.exists() and not has_completed_generation:
                # Don't delete if there's an active job for this index
                if index_dir.name in active_index_names:
                    logger.debug(f"Keeping .git_repo for {index_dir.name}: active job exists")
                    continue

                # Safe to clean up
                logger.info(f"Cleaning orphaned git repo: {index_dir.name}/.git_repo (no FAISS index, no active job)")
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

                # Orphan discovery never promotes unpublished generations.
                faiss_file = path / "index.faiss"
                pkl_file = path / "index.pkl"

                if not (faiss_file.exists() and pkl_file.exists()):
                    continue

                logger.info(f"Discovered orphan index: {path.name}")

                # Try to extract document/chunk counts from pickle file.
                # document_count = unique source files; chunk_count = chunks.
                doc_count = 0
                chunk_count = 0
                try:
                    with open(pkl_file, "rb") as pkl_f:
                        data = pickle.load(pkl_f)
                    doc_count, chunk_count = count_faiss_docstore_stats(data)

                    logger.info(f"  Extracted {doc_count} source file(s) and {chunk_count} chunk(s) from {path.name}")
                except Exception as e:
                    logger.warning(f"  Could not extract doc count from {path.name}: {e}")

                # Calculate size in thread to avoid blocking event loop
                size_bytes = await asyncio.to_thread(get_directory_size_bytes, path)

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
        job.phase = IndexJobPhase.PREPARING
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
                self._processing_tasks[job.id] = asyncio.create_task(self._process_git(job, skip_clone=True))
            else:
                # Need to re-clone - clean up any partial clone first
                if repo_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, repo_dir, ignore_errors=True)
                if temp_marker_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, temp_marker_dir, ignore_errors=True)
                self._processing_tasks[job.id] = asyncio.create_task(self._process_git(job))
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
                        await asyncio.to_thread(shutil.rmtree, extracted_dir, ignore_errors=True)

                    temp_dir = tmp_path  # Use tmp dir for extraction
                    self._processing_tasks[job.id] = asyncio.create_task(self._process_upload(job, archive_path, temp_dir))
                    return

            # No tmp file - mark as failed
            job.status = IndexStatus.FAILED
            job.phase = IndexJobPhase.FAILED
            job.error_message = "Upload file lost during restart - please re-upload"
            job.completed_at = utc_now()
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)
            self._processing_tasks.pop(job.id, None)

    async def _get_embeddings(self, app_settings: AppSettings) -> Embeddings:
        """Get the configured embedding model based on app settings."""
        # Use dict-safe access for logging (app_settings may be dict or object)
        provider = app_settings.get("embedding_provider") if isinstance(app_settings, dict) else getattr(app_settings, "embedding_provider", None)
        model = app_settings.get("embedding_model") if isinstance(app_settings, dict) else getattr(app_settings, "embedding_model", None)
        dims = app_settings.get("embedding_dimensions") if isinstance(app_settings, dict) else getattr(app_settings, "embedding_dimensions", None)
        logger.info(f"Getting embeddings: provider={provider}, model={model}, dimensions={dims}")

        embeddings = await get_embeddings_model(
            app_settings,
            allow_missing_api_key=False,
            return_none_on_error=False,
            logger_override=logger,
        )
        if embeddings is None:
            raise ValueError("Embedding model could not be initialized")
        return embeddings

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        """Detect OpenAI rate limit errors across libraries."""
        for current in iter_exception_chain(exc):
            status = getattr(current, "status_code", None) or getattr(current, "http_status", None)
            if status == 429:
                return True

            text = str(current).lower()
            if "rate limit" in text or "rate_limit_exceeded" in text or "429" in text:
                return True
        return False

    def _retry_delay_seconds(self, exc: Exception, attempt: int, base_delay: float = 1.5) -> float:
        """Compute delay before retrying after a rate limit.

        Uses Retry-After headers or "try again in Xms" hints when available,
        otherwise falls back to exponential backoff capped to 30s.
        """
        retry_after_header = None
        for current in iter_exception_chain(exc):
            headers = getattr(getattr(current, "response", None), "headers", {}) or {}
            if hasattr(headers, "get"):
                retry_after_header = headers.get("retry-after")
            if retry_after_header:
                break

        if retry_after_header:
            try:
                return max(base_delay, min(30.0, float(retry_after_header)))
            except (TypeError, ValueError):
                pass

        for current in iter_exception_chain(exc):
            text = str(current).lower()
            match = re.search(r"try again in ([0-9]+)ms", text)
            if match:
                try:
                    return max(base_delay, min(30.0, int(match.group(1)) / 1000))
                except (TypeError, ValueError):
                    pass

        return min(30.0, base_delay * (2**attempt))

    async def _embed_batch_with_fallback(
        self,
        batch: list,
        embeddings,
        job_id: str = "",
    ) -> "FAISS | None":
        """
        Embed a batch using micro-batches, falling back to individual chunks.

        Called when progressive re-chunking retries are exhausted. Processes
        chunks in small micro-batches to isolate problematic chunks, then
        embeds those individually. As a last resort, truncates content that
        still exceeds the embedding model's context limit.

        Args:
            batch: List of Document objects to embed
            embeddings: Embedding model instance
            job_id: Job ID for cancellation checks

        Returns:
            Merged FAISS index, or None if all chunks failed
        """
        result_db = None
        skipped = 0
        micro_batch_size = 50

        for j in range(0, len(batch), micro_batch_size):
            # Check for cancellation between micro-batches
            if job_id and self._is_cancelled(job_id):
                raise asyncio.CancelledError("Job cancelled by user")

            micro_batch = batch[j : j + micro_batch_size]
            try:
                micro_db = await asyncio.to_thread(FAISS.from_documents, micro_batch, embeddings)
            except Exception as e:
                if not is_context_length_error(e):
                    raise

                # Micro-batch failed - process chunks individually
                logger.info(f"Micro-batch {j // micro_batch_size + 1} failed, processing {len(micro_batch)} chunks individually")
                micro_db = None
                for doc in micro_batch:
                    # Check for cancellation between individual chunks
                    if job_id and self._is_cancelled(job_id):
                        raise asyncio.CancelledError("Job cancelled by user")

                    try:
                        single_db = await asyncio.to_thread(FAISS.from_documents, [doc], embeddings)
                    except Exception as single_e:
                        if not is_context_length_error(single_e):
                            raise

                        # This chunk exceeds model context limit - log details
                        # and truncate as last resort
                        tokens = count_tokens(doc.page_content)
                        source = doc.metadata.get("source", "unknown")
                        logger.warning(
                            f"Chunk exceeds embedding context limit at {tokens} tiktoken tokens, {len(doc.page_content)} chars - source: {source} - truncating"
                        )

                        # Binary character truncation
                        single_db = None
                        content = doc.page_content
                        for pct in (0.75, 0.50, 0.25):
                            doc.page_content = content[: int(len(content) * pct)]
                            try:
                                single_db = await asyncio.to_thread(FAISS.from_documents, [doc], embeddings)
                                break
                            except Exception:
                                pass

                        if single_db is None:
                            logger.error(f"Skipping chunk after all truncation attempts - source: {source}")
                            skipped += 1
                            continue

                    if result_db is None:
                        result_db = single_db
                    else:
                        assert single_db is not None
                        await asyncio.to_thread(result_db.merge_from, single_db)

                continue  # Skip the merge below, already handled

            if result_db is None:
                result_db = micro_db
            else:
                assert micro_db is not None
                await asyncio.to_thread(result_db.merge_from, micro_db)

        if skipped > 0:
            logger.warning(f"Skipped {skipped} chunks that couldn't be embedded")

        return result_db

    async def _append_embedding_dimension_warning(self, warnings: List[str]) -> None:
        """Warn when configured embeddings exceed pgvector's 2000-dim index limit."""
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

        # Set cancellation flag for cooperative cancellation.
        # The background task checks this flag at multiple points and will
        # raise asyncio.CancelledError, which triggers cleanup in its finally
        # block (including popping _active_jobs and shutting down the pool).
        # We intentionally do NOT pop _active_jobs here to avoid triggering
        # premature shutdown_process_pool calls while workers are still busy.
        self._cancellation_flags[job_id] = True

        # Update status in database
        job.status = IndexStatus.FAILED
        job.phase = IndexJobPhase.CANCELLED
        job.error_message = "Job cancelled by user"
        job.completed_at = utc_now()
        await repository.update_job(job)

        logger.info(f"Cancelled job {job_id}")
        return True

    async def shutdown(self) -> None:
        """Cancel active document indexing tasks before process-pool teardown."""
        logger.info("Document indexer service shutting down")

        for job_id, job in list(self._active_jobs.items()):
            task = self._processing_tasks.get(job_id)
            if task is not None and task.done():
                continue

            self._cancellation_flags[job_id] = True
            if job.status in (IndexStatus.PENDING, IndexStatus.PROCESSING):
                job.status = IndexStatus.FAILED
                job.phase = IndexJobPhase.CANCELLED
                job.error_message = "Job cancelled due to server shutdown"
                job.completed_at = utc_now()
                try:
                    await repository.update_job(job)
                except Exception as exc:
                    logger.warning(f"Job {job_id}: Could not persist shutdown cancellation: {exc}")

        for job_id, task in list(self._processing_tasks.items()):
            if task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Cancelled document indexing task {job_id}")

        self._active_jobs.clear()
        self._processing_tasks.clear()
        self._cancellation_flags.clear()

    def _is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled."""
        return self._cancellation_flags.get(job_id, False)

    async def _cleanup_failed_index_metadata(self, name: str) -> None:
        """Handle index_metadata when an index job fails.

        For both new and re-indexed indexes, metadata is preserved so the index
        remains visible in the UI with an "Incomplete" badge and Retry button.

        For re-indexes with previous successful data on disk, attempts to restore
        the document/chunk counts from FAISS files so the old data remains searchable.
        """
        try:
            metadata = await repository.get_index_metadata(name)
            if not metadata:
                return

            if metadata.chunkCount == 0:
                completed_count = await repository.count_completed_jobs(name)

                if completed_count == 0:
                    # First-time index that failed — preserve metadata so the
                    # index card stays visible with Retry button in the UI.
                    logger.info(f"Preserving metadata for failed new index '{name}' (visible in UI with Retry button)")
                else:
                    # Re-index failed but previous data may still exist on disk.
                    # Try to restore counts from FAISS files.
                    index_path = self.index_base_path / name
                    pkl_file = index_path / "index.pkl"
                    if pkl_file.exists():
                        try:
                            with open(pkl_file, "rb") as f:
                                data = pickle.load(f)
                            doc_count, chunk_count = count_faiss_docstore_stats(data)
                            if chunk_count > 0:
                                size_bytes = await asyncio.to_thread(
                                    get_directory_size_bytes,
                                    index_path,
                                )
                                logger.info(
                                    f"Restoring metadata counts for '{name}' from FAISS data: "
                                    f"{doc_count} source file(s), {chunk_count} chunk(s), {size_bytes} bytes"
                                )
                                await repository.update_index_metadata_counts(
                                    name=name,
                                    document_count=doc_count,
                                    chunk_count=chunk_count,
                                    size_bytes=size_bytes,
                                )
                                return
                        except Exception as pkl_err:
                            logger.warning(f"Could not restore counts from FAISS for '{name}': {pkl_err}")

                    logger.info(f"Preserving metadata for '{name}' despite failure (found {completed_count} completed jobs, no FAISS data to restore from)")

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
        analyze_only_git_token: bool = False,
    ) -> None:
        """Create optimistic index_metadata so the index shows up in UI immediately.

        Values will be updated when job completes; if job fails, metadata will be
        cleaned up by _cleanup_failed_index_metadata().

        On re-index, preserves existing description and config_snapshot to avoid
        overwriting user customizations with defaults.

        Args:
            config: Index configuration
            source_type: "upload" or "git"
            source: Source path/URL (filename for upload, git URL for git)
            git_branch: Git branch (for git source only)
            git_token: Git token (for git source only)
        """
        index_path = self.index_base_path / config.name

        # Check if this is a re-index (existing metadata)
        existing_metadata = await repository.get_index_metadata(config.name)

        existing_snapshot = getattr(existing_metadata, "configSnapshot", None) if existing_metadata else None
        is_analyze_only_placeholder = (
            existing_metadata is not None
            and isinstance(existing_snapshot, dict)
            and existing_snapshot.get("_analyze_only_git_token") is True
            and (existing_metadata.documentCount or 0) == 0
            and (existing_metadata.chunkCount or 0) == 0
        )

        # Preserve existing description and config_snapshot on re-index
        # to avoid overwriting user customizations with defaults
        if existing_metadata and not is_analyze_only_placeholder:
            # Preserve non-empty descriptions, but allow empty ones to be regenerated
            # This ensures user-set descriptions are kept while still allowing AI generation
            existing_desc = existing_metadata.description
            if existing_desc and existing_desc.strip():
                description = existing_desc
            else:
                # Empty or whitespace-only description - allow regeneration at job completion
                description = config.description or ""

            # Always preserve existing config_snapshot to maintain user customizations
            # (e.g., git_history_depth, chunk settings, reindex_interval_hours)
            config_snapshot = existing_snapshot or config.model_dump(mode="json")

            # Preserve existing document/chunk counts on re-index so that if the
            # new job fails or is cancelled, the metadata still reflects the
            # previously indexed data rather than showing 0 documents.
            # Counts are only updated to new values when the job completes.
            document_count = existing_metadata.documentCount or 0
            chunk_count = existing_metadata.chunkCount or 0
            size_bytes = existing_metadata.sizeBytes or 0

            logger.debug(f"Re-indexing '{config.name}': preserving existing metadata")
        else:
            description = config.description or ""
            config_snapshot = config.model_dump(mode="json")
            if analyze_only_git_token:
                config_snapshot["_analyze_only_git_token"] = True
            document_count = 0
            chunk_count = 0
            size_bytes = 0
            logger.debug(f"Creating new index '{config.name}': using config from request")

        await repository.upsert_index_metadata(
            name=config.name,
            path=str(index_path),
            document_count=document_count,
            chunk_count=chunk_count,
            size_bytes=size_bytes,
            source_type=source_type,
            source=source,
            config_snapshot=config_snapshot,
            description=description,
            git_branch=git_branch,
            git_token=git_token,
            vector_store_type=config.vector_store_type,
        )

    async def _prune_stale_analyze_only_git_metadata(self) -> None:
        try:
            stale_metadata = await repository.list_stale_analyze_only_git_metadata(ANALYZE_ONLY_GIT_TOKEN_METADATA_TTL_SECONDS)
            for metadata in stale_metadata:
                if await repository.count_jobs(metadata.name) > 0:
                    continue
                index_path = Path(metadata.path) if metadata.path else self.index_base_path / metadata.name
                if index_path.exists():
                    continue
                await repository.delete_index_metadata(metadata.name)
                logger.info("Pruned abandoned git analyze metadata for index '%s'", metadata.name)
        except Exception:
            logger.debug("Failed to prune stale analyze-only git metadata", exc_info=True)

    @staticmethod
    def _is_git_auth_error(detail: str) -> bool:
        normalized = detail.lower()
        return any(
            marker in normalized
            for marker in (
                "invalid username or token",
                "authentication failed",
                "bad credentials",
                "could not read username",
            )
        )

    async def _delete_analyze_only_git_metadata_if_unused(self, name: str) -> None:
        metadata = await repository.get_index_metadata(name)
        if not metadata:
            return
        snapshot = getattr(metadata, "configSnapshot", None)
        if not isinstance(snapshot, dict) or snapshot.get("_analyze_only_git_token") is not True:
            return
        if (metadata.documentCount or 0) > 0 or (metadata.chunkCount or 0) > 0:
            return
        if await repository.count_jobs(metadata.name) > 0:
            return
        index_path = Path(metadata.path) if metadata.path else self.index_base_path / metadata.name
        if index_path.exists():
            return
        await repository.delete_index_metadata(metadata.name)

    async def list_indexes(self) -> List[IndexInfo]:
        """List all available document indexes.

        Only returns indexes that exist in index_metadata (document indexes).
        Filesystem FAISS indexes are managed separately via tool_configs.
        """
        # Get metadata from database - this is the source of truth for document indexes
        db_metadata = await repository.list_index_metadata()
        active_index_names = await repository.list_active_index_names()

        async def _build_index_info(meta) -> Optional[IndexInfo]:
            """Build IndexInfo for a single index. Returns None if index should be skipped."""
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
            is_active = meta.name in active_index_names

            # Diagnose missing FAISS artifacts while retaining persisted rows for management.
            if vector_store_type == VectorStoreType.FAISS and not is_optimistic and not is_active:
                if not path.exists() or not path.is_dir():
                    logger.warning(f"Index {meta.name} in database but not on disk: {path}")
                # Both files are required; a partial pair is not a usable index.
                elif not (path / "index.faiss").exists() or not (path / "index.pkl").exists():
                    logger.warning(f"Index {meta.name} in database but missing FAISS artifacts: {path}")

            # Extract metadata fields
            doc_count = meta.documentCount
            chunk_count = getattr(meta, "chunkCount", 0)
            created_at = meta.createdAt
            last_modified = getattr(meta, "lastModified", None)
            size_bytes = getattr(meta, "sizeBytes", 0) or 0
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
                    ocr_provider=(OcrProvider(config_snapshot_data["ocr_provider"]) if config_snapshot_data.get("ocr_provider") else None),
                    ocr_vision_model=config_snapshot_data.get("ocr_vision_model"),
                    git_clone_timeout_minutes=config_snapshot_data.get("git_clone_timeout_minutes", 5),
                    git_history_depth=config_snapshot_data.get("git_history_depth", 1),
                    reindex_interval_hours=config_snapshot_data.get("reindex_interval_hours", 0),
                    reindex_start_minute=config_snapshot_data.get("reindex_start_minute"),
                    reindex_timezone=config_snapshot_data.get("reindex_timezone"),
                )

            # Keep /indexes request-time work cheap. Git history tooling maintains
            # the .git_repo clone independently, but we avoid running git commands
            # or recursive size scans here because this endpoint is hit frequently.
            git_repo_path = path / ".git_repo"
            git_repo_size_mb = None
            has_git_history = source_type == "git" and git_repo_path.exists() and git_repo_path.is_dir()

            return IndexInfo(
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
                last_modified=last_modified or created_at,
                git_repo_size_mb=git_repo_size_mb,
                has_git_history=has_git_history,
                vector_store_type=vector_store_type,
            )

        # Keep the hot path parallel, but each task should remain metadata-only.
        results = await asyncio.gather(
            *[_build_index_info(meta) for meta in db_metadata],
            return_exceptions=True,
        )

        indexes = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Error building index info: {result}")
                continue
            if result is not None:
                indexes.append(result)

        return indexes

    async def delete_index(self, name: str) -> bool:
        """Delete an index by name (both files and metadata)."""
        index_path = self.index_base_path / name
        deleted_files = False

        if index_path.exists() and index_path.is_dir():
            # Use thread pool to avoid blocking event loop
            # Large indexes with 100K+ files can take seconds to delete
            await asyncio.to_thread(shutil.rmtree, index_path)
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

        All git subprocess calls are run in a thread pool to avoid blocking
        the event loop.
        """

        def _run_git(*args, timeout: int = 30) -> subprocess.CompletedProcess:
            """Helper to run git commands with consistent options."""
            return subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        try:
            # Get total commit count from remote (fast, uses refs)
            clone_url = build_authenticated_git_url(git_url, git_token)

            # Fetch commit count from remote - run in thread to avoid blocking.
            # Use 10s timeout to avoid blocking on very large repos.
            try:
                count_result = await asyncio.to_thread(_run_git, "rev-list", "--count", f"origin/{git_branch}", timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("git rev-list --count timed out for commit history sampling")
                return None

            if count_result.returncode != 0:
                # Fallback: try to get count from ls-remote
                ls_result = await asyncio.to_thread(
                    subprocess.run,
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
                fetch_result = await asyncio.to_thread(_run_git, "fetch", "--deepen=1000", "origin", git_branch, timeout=60)
                if fetch_result.returncode != 0:
                    logger.warning(f"Could not deepen fetch: {fetch_result.stderr}")

                # Now count local commits
                count_result = await asyncio.to_thread(_run_git, "rev-list", "--count", f"origin/{git_branch}", timeout=10)
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
                log_result = await asyncio.to_thread(
                    _run_git,
                    "log",
                    f"--skip={depth}",
                    "-1",
                    "--format=%H %aI",
                    f"origin/{git_branch}",
                    timeout=10,
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

                # Yield to event loop after each git operation
                await asyncio.sleep(0)

            # If we have samples, also try to get the actual oldest commit
            if total_commits > 0 and total_commits > sample_depths[-1]:
                # Get the very last commit
                log_result = await asyncio.to_thread(
                    _run_git,
                    "log",
                    "--reverse",
                    "-1",
                    "--format=%H %aI",
                    f"origin/{git_branch}",
                    timeout=10,
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

    async def analyze_git_repository(self, request: AnalyzeIndexRequest) -> IndexAnalysisResult:
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
        optimistic_index_name: str | None = None

        try:
            await self._prune_stale_analyze_only_git_metadata()
            temp_dir.mkdir(parents=True, exist_ok=True)

            if request.index_name and request.git_token:
                index_name = safe_tool_name(request.index_name)
                if index_name:
                    optimistic_index_name = index_name
                    await self._create_optimistic_index_metadata(
                        config=IndexConfig(
                            name=index_name,
                            file_patterns=request.file_patterns,
                            exclude_patterns=request.exclude_patterns,
                            chunk_size=request.chunk_size,
                            chunk_overlap=request.chunk_overlap,
                            max_file_size_kb=request.max_file_size_kb,
                            ocr_mode=request.ocr_mode,
                            ocr_provider=request.ocr_provider,
                            ocr_vision_model=request.ocr_vision_model,
                        ),
                        source_type="git",
                        source=request.git_url,
                        git_branch=request.git_branch,
                        git_token=request.git_token,
                        analyze_only_git_token=True,
                    )

            # Build authenticated URL if token provided
            clone_url = build_authenticated_git_url(request.git_url, request.git_token)

            # Shallow clone (depth=1) for speed - use async subprocess to avoid blocking
            logger.info(f"Shallow cloning {request.git_url} for analysis")
            env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                request.git_branch,
                clone_url,
                str(temp_dir / "repo"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=120,  # 2 minute timeout for clone
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.wait()
                raise RuntimeError("Git clone timed out after 2 minutes. The repository may be too large for analysis.") from exc

            if process.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")

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
                ocr_provider=request.ocr_provider.value if request.ocr_provider else None,
                ocr_vision_model=request.ocr_vision_model,
            )

            # Add commit history to the result
            analysis_result.commit_history = commit_history

            return analysis_result

        except Exception as exc:
            detail = str(exc).strip()
            if optimistic_index_name and self._is_git_auth_error(detail):
                try:
                    await self._delete_analyze_only_git_metadata_if_unused(optimistic_index_name)
                except Exception:
                    logger.debug("Failed to delete analyze-only git metadata after auth failure", exc_info=True)
            raise

        finally:
            # Cleanup in thread to avoid blocking event loop
            await asyncio.to_thread(shutil.rmtree, temp_dir, True)

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
        ocr_provider: Optional[str] = None,
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

            # Save uploaded file temporarily - use thread pool to avoid blocking event loop
            # Large archives (100MB+) can take seconds to copy
            archive_path = temp_dir / filename

            def copy_archive():
                with open(archive_path, "wb") as f:
                    shutil.copyfileobj(file, f)

            await asyncio.to_thread(copy_archive)

            logger.info(f"Saved archive for analysis: {archive_path}")

            # Extract archive in thread to avoid blocking event loop
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            # Read archive extraction limits from settings
            app_settings = await get_app_settings()
            max_total_size = app_settings.get("archive_max_total_size_bytes", ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES)
            max_file_count = app_settings.get("archive_max_file_count", ARCHIVE_MAX_FILE_COUNT_DEFAULT)
            await asyncio.to_thread(extract_archive, archive_path, extract_dir, max_total_size, max_file_count)

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
                ocr_provider=ocr_provider,
                ocr_vision_model=ocr_vision_model,
            )

        finally:
            # Cleanup in thread to avoid blocking event loop
            await asyncio.to_thread(shutil.rmtree, temp_dir, True)

    async def _analyze_directory(
        self,
        source_dir: Path,
        file_patterns: List[str],
        exclude_patterns: List[str],
        chunk_size: int,
        chunk_overlap: int,
        max_file_size_kb: int = 500,
        ocr_mode: str = "disabled",
        ocr_provider: Optional[str] = None,
        ocr_vision_model: Optional[str] = None,
    ) -> IndexAnalysisResult:
        """
        Analyze a directory to estimate indexing results.
        """
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
        # MINIFIED_PATTERNS is imported at module level

        total_files = 0
        total_size = 0
        skipped_oversized = 0  # Files exceeding max_file_size_kb
        matched_patterns: set = set()
        warnings: List[str] = []
        suggested_exclusions: List[str] = []
        large_files: List[tuple] = []  # (path, size)
        minified_files: List[str] = []

        # Check if file matches minified patterns
        def is_minified(filename: str) -> bool:
            return get_matching_pattern(filename, filename, MINIFIED_PATTERNS) is not None

        # Walk the directory in a thread to avoid blocking the event loop.
        # os.walk lets us prune excluded directories before descending into them.
        _source_dir = source_dir

        def _collect_file_entries():
            """Collect file metadata in a background thread."""
            entries = []
            for dirpath, dirnames, filenames in os.walk(_source_dir):
                current_dir = Path(dirpath)
                dirnames[:] = [
                    dirname
                    for dirname in dirnames
                    if not is_excluded_directory(
                        current_dir / dirname,
                        source_dir,
                        exclude_patterns,
                    )
                ]

                for filename in filenames:
                    fp = current_dir / filename
                    if is_excluded_by_patterns(
                        fp,
                        source_dir,
                        exclude_patterns,
                        skip_minified=False,
                    ):
                        continue
                    matched_pattern = get_matching_file_pattern(
                        fp,
                        source_dir,
                        file_patterns,
                    )
                    if not should_index_file_type(
                        fp,
                        matches_include_pattern=matched_pattern is not None,
                        ocr_enabled=ocr_enabled,
                    ):
                        continue
                    try:
                        if not fp.is_file():
                            continue
                        st_size = fp.stat().st_size
                    except OSError:
                        continue
                    entries.append((fp, st_size, matched_pattern or "(content match)"))
            return entries

        file_entries = await asyncio.to_thread(_collect_file_entries)

        for file_path, size, pattern in file_entries:
            matched_patterns.add(pattern)

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
                stats["estimated_chunks"] = max(1, stats["total_size"] // effective_chunk)
            else:
                stats["estimated_chunks"] = stats["file_count"]

        # Get smart exclusion suggestions (LLM if available, otherwise heuristics)
        # Extract repo name from source_dir for context
        repo_name = source_dir.name if source_dir.name != "repo" else source_dir.parent.name

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
        unparseable_found = [ext for ext in ext_stats if ext in UNPARSEABLE_BINARY_EXTENSIONS and ext not in OCR_EXTENSIONS]
        parseable_docs_found = [ext for ext in ext_stats if ext in PARSEABLE_DOCUMENT_EXTENSIONS]

        if ocr_images_found:
            if ocr_enabled:
                ocr_method = f"Vision ({ocr_provider or 'default provider'})" if ocr_mode == "vision" else "Tesseract"
                warnings.append(f"Found image types ({', '.join(ocr_images_found)}) that will be processed with {ocr_method} to extract text.")
            else:
                warnings.append(f"Found image types ({', '.join(ocr_images_found)}) that will be skipped. Enable OCR to extract text from these files.")

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
                f"Skipped {skipped_oversized} files exceeding the {max_file_size_kb}KB size limit. Increase max file size if you need to include them."
            )

        if large_files:
            threshold_kb = int(max_file_size_kb * 0.8)
            if len(large_files) > 5:
                warnings.append(
                    f"Found {len(large_files)} files over {threshold_kb}KB (approaching limit). Examples: {large_files[0][0]} ({large_files[0][1] // 1024}KB)"
                )
            else:
                for path, size in large_files[:3]:
                    warnings.append(f"Large file: {path} ({size // 1024}KB)")

        # Calculate totals
        total_estimated_chunks = sum(stats["estimated_chunks"] for stats in ext_stats.values())

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
                f"Estimated index size is {estimated_index_size_mb:.0f}MB. Consider adding more exclusion patterns or reducing included file types.",
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
                    steady_memory_mb=round(mem_est["steady_memory_bytes"] / (1024 * 1024), 1),
                    peak_memory_mb=round(mem_est["peak_memory_bytes"] / (1024 * 1024), 1),
                    dimension_breakdown=dim_breakdown,
                )

                # Calculate total memory with existing indexes
                existing_indexes = await repository.list_index_metadata()
                existing_memory = sum((idx.steadyMemoryBytes or 0) for idx in existing_indexes)
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

    async def create_index_from_upload(self, file: BinaryIO, filename: str, config: IndexConfig) -> IndexJob:
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

            # Copy in thread to avoid blocking event loop on large uploads
            def _copy_upload():
                with open(archive_path, "wb") as f:
                    shutil.copyfileobj(file, f)

            await asyncio.to_thread(_copy_upload)

            logger.info(f"Saved upload to tmp directory for job {job_id}: {archive_path}")

            # Start processing in background — hold strong reference to prevent GC
            self._processing_tasks[job.id] = asyncio.create_task(self._process_upload(job, archive_path, tmp_dir))

        except Exception as e:
            job.status = IndexStatus.FAILED
            job.phase = IndexJobPhase.FAILED
            job.error_message = str(e)
            job.completed_at = utc_now()
            await repository.update_job(job)
            self._active_jobs.pop(job_id, None)
            self._processing_tasks.pop(job_id, None)
            if "tmp_dir" in locals():
                await asyncio.to_thread(shutil.rmtree, tmp_dir, True)
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
            raise ValueError("Upload files no longer available. Please re-upload the file.")

        # Find the archive file (not directories like 'extracted')
        archive_files = [f for f in tmp_path.iterdir() if f.is_file()]
        if not archive_files:
            raise ValueError("Upload archive file not found. Please re-upload the file.")

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

        # Start processing in background — hold strong reference to prevent GC
        self._processing_tasks[job.id] = asyncio.create_task(self._process_upload(job, new_archive_path, new_tmp_dir))

        return job

    async def create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> IndexJob:
        """Create an index from a git repository."""
        async with self._git_job_creation_lock:
            active_job = await repository.get_active_job_for_index(config.name)
            if active_job is not None:
                logger.info(f"Reusing active git indexing job {active_job.id} for index '{config.name}'")
                return active_job

            return await self._create_git_index_job_locked(git_url, branch, config, git_token)

    async def try_create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> IndexJob | None:
        """Create an index from a git repository if no active job already exists."""
        async with self._git_job_creation_lock:
            if await repository.get_active_job_for_index(config.name) is not None:
                return None

            return await self._create_git_index_job_locked(git_url, branch, config, git_token)

    async def _create_git_index_job_locked(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> IndexJob:
        """Create and start a git indexing job while holding _git_job_creation_lock."""
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

        # Start processing in background — hold strong reference to prevent GC
        self._processing_tasks[job.id] = asyncio.create_task(self._process_git(job))

        return job

    async def _process_upload(self, job: IndexJob, archive_path: Path, temp_dir: Path):
        """Process an uploaded archive file (zip, tar, tar.gz, tar.bz2)."""
        try:
            job.status = IndexStatus.PROCESSING
            job.started_at = utc_now()
            await repository.update_job(job)

            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled before extraction")
                return

            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            logger.info(f"Extracting {archive_path} to {extract_dir}")

            # Read archive extraction limits from settings
            app_settings = await get_app_settings()
            max_total_size = app_settings.get("archive_max_total_size_bytes", ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES)
            max_file_count = app_settings.get("archive_max_file_count", ARCHIVE_MAX_FILE_COUNT_DEFAULT)
            await asyncio.to_thread(extract_archive, archive_path, extract_dir, max_total_size, max_file_count)

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
                job.phase = IndexJobPhase.COMPLETED
                job.completed_at = utc_now()

        except asyncio.CancelledError:
            logger.info(f"Job {job.id} processing stopped due to cancellation")
            job.status = IndexStatus.FAILED
            job.phase = IndexJobPhase.CANCELLED
            if not job.error_message:
                job.error_message = "Job cancelled by user"
            if job.completed_at is None:
                job.completed_at = utc_now()
            # Clean up optimistic metadata for cancelled new indexes
            await self._cleanup_failed_index_metadata(job.name)

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process upload for job {job.id}")
                job.status = IndexStatus.FAILED
                job.phase = IndexJobPhase.FAILED
                job.error_message = str(e) or repr(e)
                job.completed_at = utc_now()
                # Clean up optimistic metadata for failed new indexes
                await self._cleanup_failed_index_metadata(job.name)

        finally:
            # Only cleanup temp directory on success (preserve for retry on failure)
            if job.status == IndexStatus.COMPLETED:
                await asyncio.to_thread(shutil.rmtree, temp_dir, True)
            self._cancellation_flags.pop(job.id, None)
            self._active_jobs.pop(job.id, None)
            self._processing_tasks.pop(job.id, None)

            # Release this job's chunking pool so its workers (which each import
            # Chonkie/tree-sitter) are terminated and the per-job pool entry
            # removed from the manager. Running in a thread because the
            # underlying terminate + join is blocking.
            try:
                await asyncio.to_thread(pool_manager.release, job.id)
            except Exception:
                pass

            # Load into RAG before publishing completed status so newly completed
            # indexes are immediately queryable without a restart.
            try:
                await self._maybe_reinitialize_rag(job)
            except Exception as rag_err:
                logger.warning(f"Job {job.id}: Could not reinitialize RAG components: {rag_err}")

            # Update job status - gracefully handle database disconnection
            try:
                await repository.update_job(job)
            except Exception as db_err:
                logger.warning(f"Job {job.id}: Could not update job status (database may be disconnected): {db_err}")

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
            job.started_at = utc_now()
            await repository.update_job(job)

            # Check for cancellation
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled before cloning")
                return

            # Preserve token for metadata storage
            # We used to clear job.git_token here, but that causes issues on resume/retry
            # because the token is cleared from DB and not available for the resumed job.
            # Since gitToken is encrypted in IndexJob (repository.create_job handles encryption),
            # it is safe to leave it in the DB record.
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

            # Check for cancellation after clone/fetch
            if self._is_cancelled(job.id):
                logger.info(f"Job {job.id} was cancelled after cloning")
                return

            # Create the index, passing the token for storage in metadata
            await self._create_faiss_index(job, repo_dir, git_token=stored_token)

            # Only mark completed if not cancelled
            if not self._is_cancelled(job.id):
                job.status = IndexStatus.COMPLETED
                job.phase = IndexJobPhase.COMPLETED
                job.completed_at = utc_now()

        except asyncio.CancelledError:
            logger.info(f"Job {job.id} processing stopped due to cancellation")
            job.status = IndexStatus.FAILED
            job.phase = IndexJobPhase.CANCELLED
            if not job.error_message:
                job.error_message = "Job cancelled by user"
            if job.completed_at is None:
                job.completed_at = utc_now()
            # Clean up optimistic metadata for cancelled new indexes
            await self._cleanup_failed_index_metadata(job.name)

        except Exception as e:
            if not self._is_cancelled(job.id):
                logger.exception(f"Failed to process git for job {job.id}")
                job.status = IndexStatus.FAILED
                job.phase = IndexJobPhase.FAILED
                job.error_message = str(e) or repr(e)
                job.completed_at = utc_now()
                # Clean up optimistic metadata for failed new indexes
                await self._cleanup_failed_index_metadata(job.name)

        finally:
            # Clean up temp marker directory
            if job.status in (IndexStatus.COMPLETED, IndexStatus.FAILED):
                await asyncio.to_thread(shutil.rmtree, temp_marker_dir, ignore_errors=True)
            self._cancellation_flags.pop(job.id, None)
            self._active_jobs.pop(job.id, None)
            self._processing_tasks.pop(job.id, None)

            # Release this job's chunking pool. Each indexing job owns its own
            # pool keyed by job.id, so cancellation/termination of one job
            # cannot block or kill another job's workers.
            try:
                await asyncio.to_thread(pool_manager.release, job.id)
            except Exception:
                pass

            # Load into RAG before publishing completed status so newly completed
            # indexes are immediately queryable without a restart.
            try:
                await self._maybe_reinitialize_rag(job)
            except Exception as rag_err:
                logger.warning(f"Job {job.id}: Could not reinitialize RAG components: {rag_err}")

            # Update job status - gracefully handle database disconnection
            try:
                await repository.update_job(job)
            except Exception as db_err:
                logger.warning(f"Job {job.id}: Could not update job status (database may be disconnected): {db_err}")

    async def _clone_git_repo(self, job: IndexJob, repo_dir: Path) -> None:
        """Clone a git repository to the specified directory.

        Args:
            job: The index job with git configuration
            repo_dir: Target directory for the clone
        """
        logger.info(f"Cloning {job.git_url} branch {job.git_branch}")

        job.phase = IndexJobPhase.CLONING
        job.error_message = None
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
            logger.info("Cloning with full history (may take a long time for large repos)")
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
            stderr_output = await self._stream_clone_progress(process, job, clone_timeout_seconds)
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
            if "could not read Username" in error_msg or "Authentication failed" in error_msg:
                raise RuntimeError("Git clone failed: Authentication required. This is a private repository - please provide a valid access token.")
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

        def repo_git_args(*args: str) -> List[str]:
            # Scope safe.directory trust to this repo for this command only.
            # This handles cases where directory ownership changed after initial clone.
            return [
                "git",
                "-c",
                f"safe.directory={repo_dir.resolve()}",
                *args,
            ]

        def format_dubious_ownership_error(error_msg: str, operation: str) -> Optional[str]:
            if "detected dubious ownership" not in error_msg:
                return None
            repo_path = repo_dir.resolve()
            return (
                f"Git {operation} failed: repository permissions changed for '{repo_path}'. "
                "Fix by running this inside the ragtime container: "
                f"`git config --global --add safe.directory {repo_path}` "
                "or by restoring ownership so the runtime user owns that directory."
            )

        logger.info(f"Fetching updates for {job.git_url} branch {job.git_branch}")
        job.phase = IndexJobPhase.CLONING
        job.error_message = None
        await repository.update_job(job)

        # Update remote URL with current token (in case token changed)
        # Run in thread to avoid blocking event loop
        fetch_url = build_authenticated_git_url(job.git_url, job.git_token)
        remote_result = await asyncio.to_thread(
            subprocess.run,
            repo_git_args("remote", "set-url", "origin", fetch_url),
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if remote_result.returncode != 0:
            remote_error = remote_result.stderr or remote_result.stdout
            ownership_error = format_dubious_ownership_error(remote_error, "remote update")
            if ownership_error:
                raise RuntimeError(ownership_error)
            raise RuntimeError(f"Git remote update failed: {remote_error}")

        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        clone_timeout_minutes = self._compute_clone_timeout_minutes(job.config)
        clone_timeout_seconds = clone_timeout_minutes * 60
        history_depth = getattr(job.config, "git_history_depth", 1)

        # Check current depth of repo - run in thread to avoid blocking.
        # Use a short timeout (10s) because rev-list --count can take
        # very long on large repos (e.g., Odoo with 100k+ commits).
        try:
            depth_result = await asyncio.to_thread(
                subprocess.run,
                repo_git_args("rev-list", "--count", "HEAD"),
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            current_depth = int(depth_result.stdout.strip()) if depth_result.returncode == 0 else 1
        except subprocess.TimeoutExpired:
            logger.warning("git rev-list --count timed out (large repo), assuming deep history")
            current_depth = 999999  # Assume large depth so we don't re-clone

        # Check if repo is shallow
        is_shallow = (repo_dir / ".git" / "shallow").exists()

        # Determine fetch strategy
        branch = job.git_branch or "main"
        if history_depth == 0 and is_shallow:
            logger.info("Unshallowing repo for full history")
            fetch_args = repo_git_args("fetch", "--unshallow", "--progress", "origin", branch)
        elif history_depth > current_depth and is_shallow:
            deepen_amount = history_depth - current_depth
            logger.info(f"Deepening repo from {current_depth} to {history_depth} commits (+{deepen_amount})")
            fetch_args = repo_git_args(
                "fetch",
                f"--deepen={deepen_amount}",
                "--progress",
                "origin",
                branch,
            )
        else:
            logger.info(f"Fetching latest changes (depth: {current_depth} -> {history_depth})")
            fetch_args = repo_git_args("fetch", "--progress", "origin", branch)

        try:
            process = await asyncio.create_subprocess_exec(
                *fetch_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repo_dir,
                env=env,
            )
            stderr_output = await self._stream_clone_progress(process, job, clone_timeout_seconds)
        except TimeoutError as exc:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            error_msg = f"Git fetch timed out after {clone_timeout_minutes} minutes. Try increasing the clone timeout in Advanced Options."
            logger.error(f"Job {job.id}: {error_msg}")
            raise RuntimeError(error_msg) from exc

        if process.returncode != 0:
            error_msg = stderr_output
            ownership_error = format_dubious_ownership_error(error_msg, "fetch")
            if ownership_error:
                raise RuntimeError(ownership_error)
            if "could not read Username" in error_msg or "Authentication failed" in error_msg:
                raise RuntimeError("Git fetch failed: Authentication required. This is a private repository - please provide a valid access token.")
            raise RuntimeError(f"Git fetch failed: {error_msg}")

        # Reset to the fetched branch - run in thread to avoid blocking
        reset_result = await asyncio.to_thread(
            subprocess.run,
            repo_git_args("reset", "--hard", f"origin/{branch}"),
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if reset_result.returncode != 0:
            reset_error = reset_result.stderr or reset_result.stdout
            ownership_error = format_dubious_ownership_error(reset_error, "reset")
            if ownership_error:
                raise RuntimeError(ownership_error)
            raise RuntimeError(f"Git reset failed: {reset_error}")

        logger.info("Fetch complete")

    async def _stream_clone_progress(self, process: Any, job: IndexJob, timeout_seconds: int) -> str:
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

        # Initialize clone progress so the UI can immediately show a cloning
        # phase instead of an idle "preparing" state.
        if job.clone_progress is None:
            job.clone_progress = 0.0

        async def read_with_timeout():
            start_time = asyncio.get_event_loop().time()
            nonlocal last_update_time
            last_update_time = start_time
            buffer = ""
            last_fraction = job.clone_progress or 0.0

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

                    # git refreshes progress with \r and ends phases with \n;
                    # split on both so each refresh is parsed for the latest
                    # monotonic overall fraction.
                    buffer += text
                    parts = re.split(r"[\r\n]+", buffer)
                    buffer = parts.pop() if parts else ""
                    for line in parts:
                        if not line:
                            continue
                        fraction = git_clone_fraction_from_line(line)
                        if fraction is not None and fraction > last_fraction:
                            last_fraction = fraction

                    # Throttle job status writes to at most once per second.
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_update_time >= update_interval:
                        if last_fraction != (job.clone_progress or 0.0):
                            job.clone_progress = last_fraction
                            await repository.update_job(job)
                            last_update_time = current_time

                except asyncio.TimeoutError:
                    # Just a read timeout, check if process is still running
                    if process.returncode is not None:
                        break
                    continue

            # Parse any trailing buffered progress line.
            if buffer:
                fraction = git_clone_fraction_from_line(buffer)
                if fraction is not None and fraction > last_fraction:
                    last_fraction = fraction
                    job.clone_progress = last_fraction

            # Wait for process to complete
            await process.wait()

        await read_with_timeout()

        # Clear the cloning state once the clone finishes.
        job.clone_progress = None
        await repository.update_job(job)

        return "".join(stderr_chunks)

    async def _index_git_history(self, repo_dir: Path, index_name: str, depth: int) -> List:
        """Extract git commit history as documents for indexing.

        Args:
            repo_dir: Path to the git repository
            index_name: Name of the index for metadata
            depth: Number of commits to index (0 = all)

        Returns:
            List of LangChain Document objects with commit information
        """
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

    async def _create_faiss_index(self, job: IndexJob, source_dir: Path, git_token: Optional[str] = None):
        """Create FAISS index from source directory."""
        config = job.config
        job.phase = IndexJobPhase.SCANNING
        job.error_message = None
        await repository.update_job(job)
        ocr_enabled = config.ocr_mode != OcrMode.DISABLED
        files_with_sizes = await asyncio.to_thread(
            collect_files_recursive,
            source_dir,
            config.file_patterns,
            config.exclude_patterns,
            max_file_size_bytes=config.max_file_size_kb * 1024,
            ocr_enabled=ocr_enabled,
        )
        job.total_files = len(files_with_sizes)
        job.processed_files = 0
        job.phase = IndexJobPhase.LOADING
        await repository.update_job(job)
        app_settings = await repository.get_settings()

        def setting(name: str, default: Any = None) -> Any:
            return app_settings.get(name, default) if isinstance(app_settings, dict) else getattr(app_settings, name, default)

        def source_snapshot() -> dict[str, Any]:
            files = [
                {
                    "path": str(path.relative_to(source_dir)),
                    "size": size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path, size in files_with_sizes
            ]
            snapshot: dict[str, Any] = {"files": files}
            if job.source_type == "git":
                try:
                    snapshot["head"] = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
                except (OSError, subprocess.CalledProcessError):
                    snapshot["head"] = None
            return snapshot

        fingerprint = json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "provider": setting("embedding_provider"),
                "model": setting("embedding_model"),
                "dimensions": setting("embedding_dimensions"),
                "chunking_use_tokens": setting("chunking_use_tokens", True),
                "chunking_implementation": chunking_implementation_fingerprint(),
                "source": await asyncio.to_thread(source_snapshot),
            },
            sort_keys=True,
            default=str,
        )
        spool = await asyncio.to_thread(IndexingSpool.open_attempt, UPLOAD_TMP_DIR / "indexing", job.id, fingerprint)
        pipeline = BoundedIndexingPipeline(spool, job_id=job.id, cancelled=lambda: self._is_cancelled(job.id))
        vision_base_url = vision_api_key = None
        effective_ocr_provider = config.ocr_provider.value if config.ocr_provider else None
        if config.ocr_mode == OcrMode.VISION:
            runtime_settings = await get_app_settings()
            effective_ocr_provider = normalize_provider_name(effective_ocr_provider or str(runtime_settings.get("default_ocr_provider") or "ollama"))
            vision_base_url = resolve_provider_base_url(runtime_settings, effective_ocr_provider, "llm")
            vision_api_key = runtime_settings.get("openai_api_key" if effective_ocr_provider == "openai" else f"{effective_ocr_provider}_api_key")
        effective_ocr_vision_model = config.ocr_vision_model or (
            runtime_settings.get("default_ocr_vision_model") if config.ocr_mode == OcrMode.VISION else None
        )

        async def load(path: Path) -> list[LangChainDocument]:
            if config.ocr_mode == OcrMode.VISION and (path.suffix.lower() in PARSEABLE_DOCUMENT_EXTENSIONS or path.suffix.lower() in OCR_EXTENSIONS):
                request = resource_governor.estimate_request(
                    job_id=job.id,
                    stage="loading",
                    text_bytes=min(path.stat().st_size, MAX_SOURCE_TEXT_BYTES),
                    record_count=1,
                    cpu_slots=0,
                    provider_key=f"ocr:{effective_ocr_provider}",
                )
                async with resource_governor.acquire(request):
                    text = await extract_text_from_file_async(
                        path,
                        ocr_mode=config.ocr_mode.value,
                        ocr_provider=effective_ocr_provider,
                        ocr_vision_model=effective_ocr_vision_model,
                        vision_base_url=vision_base_url,
                        vision_api_key=vision_api_key,
                    )
                    if len(text.encode("utf-8")) > MAX_SOURCE_TEXT_BYTES:
                        raise ValueError(f"Extracted text for {path} exceeds {MAX_SOURCE_TEXT_BYTES} byte limit")
            elif path.suffix.lower() not in PARSEABLE_DOCUMENT_EXTENSIONS and path.suffix.lower() not in OCR_EXTENSIONS:
                # Plain text is source-size bounded and needs no native parser.
                request = ResourceRequest(job.id, "loading", 32 * 1024 * 1024, cpu_slots=0)
                async with resource_governor.acquire(request):
                    documents = await asyncio.to_thread(TextLoader(str(path), autodetect_encoding=True).load)
                return documents
            else:
                request = resource_governor.estimate_request(job_id=job.id, stage="loading", text_bytes=path.stat().st_size, record_count=1, cpu_slots=1)
                parser = functools.partial(extract_text_from_file_process_safe, ocr_mode=config.ocr_mode.value)
                text = await run_resource_task(request, parser, (str(path),), timeout_seconds=300.0)
            return [LangChainDocument(page_content=text)] if text else []

        async def loaded_progress(processed: int) -> None:
            job.processed_files = processed
            await repository.update_job(job)

        async def chunk_progress(processed_documents: int, total_chunks: int) -> None:
            job.processed_chunks = min(job.total_chunks, job.processed_chunks + processed_documents)
            await repository.update_job(job)

        async def embedding_progress(committed: int) -> None:
            job.processed_chunks = committed
            await repository.update_job(job)

        try:
            if not await asyncio.to_thread(spool.is_stage_complete, "documents"):
                await pipeline.stage_documents((path for path, _ in files_with_sizes), source_dir, job.name, load, loaded_progress)
                if job.source_type == "git" and config.git_history_depth != 1:
                    await pipeline.stage_extra_documents(await self._index_git_history(source_dir, job.name, config.git_history_depth))
                await asyncio.to_thread(spool.mark_stage_complete, "documents")
            summary = await asyncio.to_thread(spool.summary)
            if not summary["document_count"]:
                raise ValueError("No indexable documents were loaded; check file patterns, exclusions, and OCR settings")
            job.processed_files = job.total_files
            job.total_chunks = summary["document_count"]
            job.processed_chunks = 0
            job.phase = IndexJobPhase.CHUNKING
            await repository.update_job(job)
            if not await asyncio.to_thread(spool.is_stage_complete, "chunks"):
                await pipeline.stage_chunks(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    use_tokens=setting("chunking_use_tokens", True),
                    max_documents=max(1, resource_governor.batch_document_limit()),
                    progress=chunk_progress,
                )
                await asyncio.to_thread(spool.mark_stage_complete, "chunks")
            summary = await asyncio.to_thread(spool.summary)
            job.total_chunks = summary["chunk_count"]
            job.processed_chunks = summary["embedded_count"]
            job.phase = IndexJobPhase.EMBEDDING
            await repository.update_job(job)
            embeddings = await self._get_embeddings(app_settings)
            context_limit = await get_embedding_model_context_limit(
                model_name=setting("embedding_model"),
                provider=setting("embedding_provider"),
                ollama_base_url=setting("ollama_base_url"),
                llama_cpp_base_url=setting("llama_cpp_base_url"),
                lmstudio_base_url=setting("lmstudio_base_url"),
            )
            safe_token_limit = int(context_limit * get_embedding_safety_margin(setting("embedding_provider")))
            if not await asyncio.to_thread(spool.is_stage_complete, "embeddings"):
                await pipeline.stage_embeddings(
                    embeddings,
                    max_documents=max(1, resource_governor.batch_document_limit()),
                    max_text_bytes=DEFAULT_BATCH_TEXT_BYTES,
                    resource_job_id=job.id,
                    safe_token_limit=safe_token_limit,
                    chunk_overlap=config.chunk_overlap,
                    progress=embedding_progress,
                )
                await asyncio.to_thread(spool.mark_stage_complete, "embeddings")
            summary = await asyncio.to_thread(spool.summary)
            job.processed_chunks = summary["embedded_count"]
            job.total_chunks = summary["embedded_count"]
            job.phase = IndexJobPhase.FINALIZING
            await repository.update_job(job)
            # The artifact worker opens its own readonly connection; it must
            # never race this attempt's single SQLite writer.
            await asyncio.to_thread(spool.close)
            artifact = await prepare_faiss_artifact(job.id, self.index_base_path / job.name, spool.root)
            journal = await asyncio.to_thread(IndexingSpool.open_existing, spool.root, readonly=False)
            try:
                # This marker is durable before metadata publication, allowing
                # startup reconciliation to distinguish a prepared artifact
                # from the generation that actually became authoritative.
                await asyncio.to_thread(journal.set_state, "prepared_generation", str(artifact.generation_path))
            finally:
                await asyncio.to_thread(journal.close)
            existing = await repository.get_index_metadata(job.name)
            description = config.description or (getattr(existing, "description", "") if existing else "")
            if not description:
                sample_spool = await asyncio.to_thread(IndexingSpool.open_existing, spool.root)
                try:
                    sample_documents = []
                    for batch in sample_spool.iter_documents(max_documents=5, max_text_bytes=DEFAULT_BATCH_TEXT_BYTES):
                        for record in batch.records:
                            content = await asyncio.to_thread(sample_spool._file(record.text_path).read_text, encoding="utf-8")
                            sample_documents.append(
                                LangChainDocument(
                                    page_content=content,
                                    metadata=record.metadata,
                                )
                            )
                            if len(sample_documents) == 5:
                                break
                        break
                finally:
                    await asyncio.to_thread(sample_spool.close)
                description = await generate_index_description(
                    index_name=job.name, documents=sample_documents, source_type=job.source_type, source=job.git_url or job.source_path
                )
            snapshot = config.model_dump(mode="json")
            existing_snapshot = getattr(existing, "configSnapshot", None) if existing else None
            if isinstance(existing_snapshot, dict):
                for key in (
                    "file_patterns",
                    "exclude_patterns",
                    "chunk_size",
                    "chunk_overlap",
                    "max_file_size_kb",
                    "ocr_mode",
                    "ocr_provider",
                    "ocr_vision_model",
                    "git_clone_timeout_minutes",
                    "git_history_depth",
                    "reindex_interval_hours",
                    "reindex_start_minute",
                    "reindex_timezone",
                ):
                    if key in existing_snapshot:
                        snapshot[key] = existing_snapshot[key]
            await repository.upsert_index_metadata(
                name=job.name,
                path=str(artifact.generation_path),
                document_count=artifact.document_count,
                chunk_count=artifact.chunk_count,
                size_bytes=artifact.size_bytes,
                source_type=job.source_type,
                source=job.git_url or job.source_path,
                config_snapshot=snapshot,
                description=description,
                git_branch=job.git_branch if job.source_type == "git" else None,
                git_token=git_token,
                vector_store_type=config.vector_store_type,
            )
            journal = await asyncio.to_thread(IndexingSpool.open_existing, spool.root, readonly=False)
            try:
                await asyncio.to_thread(journal.set_state, "published_generation", str(artifact.generation_path))
                await asyncio.to_thread(journal.set_state, "published_index_name", job.name)
            finally:
                await asyncio.to_thread(journal.close)
            self._completed_spools[job.id] = spool.root
        finally:
            # Successful attempts are removed only after metadata publication;
            # failed attempts retain their journal for fingerprinted recovery.
            try:
                await asyncio.to_thread(spool.close)
            except Exception:
                pass
        return


# Global indexer instance - uses configured path
indexer = IndexerService(index_base_path=settings.index_data_path)


# Global indexer instance - uses configured path
indexer = IndexerService(index_base_path=settings.index_data_path)
