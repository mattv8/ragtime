"""
Filesystem Indexer Service - Creates and manages filesystem indexes.

This service handles:
- Indexing files from Docker volumes, SMB shares, NFS mounts, or local paths
- Incremental indexing (skip unchanged files based on SHA-256 hash)
- Storing embeddings in PostgreSQL (pgvector) or FAISS (in-memory)
- Progress tracking and job management
- Persistent mount management (SMB/NFS shares mounted once, reused across jobs)
"""

import asyncio
import mimetypes
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ragtime.core.database import get_db
from ragtime.core.file_constants import (
    DOCUMENT_EXTENSIONS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
    UNPARSEABLE_BINARY_EXTENSIONS,
)
from ragtime.core.logging import get_logger
from ragtime.indexer.document_parser import (
    OCR_EXTENSIONS,
    is_ocr_supported,
    is_supported_document,
)
from ragtime.indexer.file_utils import compute_file_hash, matches_pattern
from ragtime.indexer.models import (
    FilesystemAnalysisJob,
    FilesystemAnalysisResult,
    FilesystemAnalysisStatus,
    FilesystemConnectionConfig,
    FilesystemFileMetadata,
    FilesystemIndexJob,
    FilesystemIndexStatus,
    FileTypeStats,
    OcrMode,
    VectorStoreType,
)
from ragtime.indexer.repository import repository
from ragtime.indexer.vector_backends import (
    VectorStoreBackend,
    get_backend,
    get_faiss_backend,
)
from ragtime.indexer.vector_utils import (
    ensure_embedding_column,
    ensure_pgvector_extension,
    get_embeddings_model,
)

logger = get_logger(__name__)


@dataclass
class MountInfo:
    """Tracks state of a mounted filesystem."""

    tool_config_id: str
    mount_point: str
    mount_type: str  # "smb" or "nfs"
    effective_path: str  # mount_point + base_path offset
    mounted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reference_count: int = 0  # Number of active jobs using this mount


class FilesystemIndexerService:
    """Service for creating and managing filesystem indexes with pgvector."""

    def __init__(self):
        self._active_jobs: Dict[str, FilesystemIndexJob] = {}
        self._cancellation_flags: Dict[str, bool] = {}  # job_id -> should_cancel
        self._running_tasks: Dict[str, asyncio.Task] = {}  # job_id -> task
        self._shutdown = False
        # Analysis job tracking (in-memory since analysis is fast)
        self._analysis_jobs: Dict[str, FilesystemAnalysisJob] = {}
        self._analysis_results: Dict[str, FilesystemAnalysisResult] = {}
        # Persistent mount tracking: tool_config_id -> MountInfo
        self._mounts: Dict[str, MountInfo] = {}
        self._mount_lock = threading.Lock()  # Protect mount operations
        # Note: Ollama vision OCR uses centralized semaphore from ollama_concurrency

    async def _append_embedding_dimension_warning(self, warnings: List[str]) -> None:
        """Warn when embedding dimensions exceed pgvector's 2000-dim index limit."""
        try:
            settings = await repository.get_settings()
        except Exception as e:
            logger.debug(f"Could not load settings for embedding warning: {e}")
            return

        model = settings.embedding_model
        # Use configured dimension if set, otherwise use tracked dimension
        dimension = (
            getattr(settings, "embedding_dimensions", None)
            or settings.embedding_dimension
        )

        limit = 2000
        if dimension is not None and dimension > limit:
            model_info = f" ({model})" if model else ""
            warnings.insert(
                0,
                (
                    f"Embedding dimension {dimension}{model_info} exceeds pgvector's {limit}-dim index limit. "
                    "Search will fall back to exact (non-indexed). Use a model with <=2000 dimensions, "
                    "or configure lower dimensions in Embedding Configuration."
                ),
            )

    def _get_mount_key(self, config: FilesystemConnectionConfig) -> str:
        """Generate a unique key for a mount based on its configuration."""
        if config.mount_type == "smb":
            return f"smb://{config.smb_host}/{config.smb_share}"
        elif config.mount_type == "nfs":
            return f"nfs://{config.nfs_host}:{config.nfs_export}"
        else:
            return f"local:{config.base_path}"

    async def _check_mount_health(self, mount_info: MountInfo) -> bool:
        """Check if a mount point is still accessible."""
        try:
            mount_path = Path(mount_info.mount_point)
            # Quick check: can we list the directory?
            await asyncio.to_thread(lambda: list(mount_path.iterdir())[:1])
            return True
        except Exception as e:
            logger.warning(
                f"Mount health check failed for {mount_info.mount_point}: {e}"
            )
            return False

    async def _do_mount_smb(
        self, config: FilesystemConnectionConfig, mount_point: str
    ) -> None:
        """Execute SMB mount command."""
        smb_path = f"//{config.smb_host}/{config.smb_share}"
        mount_opts = []
        if config.smb_user:
            mount_opts.append(f"user={config.smb_user}")
        if config.smb_password:
            mount_opts.append(f"password={config.smb_password}")
        if config.smb_domain:
            mount_opts.append(f"domain={config.smb_domain}")
        mount_opts.extend(["vers=3.0", "sec=ntlmssp"])

        cmd = [
            "mount",
            "-t",
            "cifs",
            smb_path,
            mount_point,
            "-o",
            ",".join(mount_opts),
        ]

        logger.info(f"Mounting SMB share {smb_path} to {mount_point}")
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown mount error"
            raise RuntimeError(f"Failed to mount SMB share: {error_msg}")

    async def _do_mount_nfs(
        self, config: FilesystemConnectionConfig, mount_point: str
    ) -> None:
        """Execute NFS mount command."""
        nfs_path = f"{config.nfs_host}:{config.nfs_export}"
        mount_opts = config.nfs_options or "ro,soft,timeo=30"

        cmd = ["mount", "-t", "nfs", nfs_path, mount_point, "-o", mount_opts]

        logger.info(f"Mounting NFS export {nfs_path} to {mount_point}")
        result = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown mount error"
            raise RuntimeError(f"Failed to mount NFS export: {error_msg}")

    async def _do_unmount(self, mount_point: str) -> None:
        """Unmount a filesystem and cleanup mount point."""
        logger.info(f"Unmounting {mount_point}")
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["umount", mount_point],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.warning(f"Error unmounting {mount_point}: {e}")
        try:
            shutil.rmtree(mount_point, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up mount point {mount_point}: {e}")

    @asynccontextmanager
    async def _mount_filesystem(
        self, config: FilesystemConnectionConfig, tool_config_id: str | None = None
    ) -> AsyncIterator[Path]:
        """
        Context manager for filesystem access with persistent mount reuse.

        For docker_volume/local mounts, yields base_path directly.
        For smb/nfs mounts:
          - Reuses existing mount if available and healthy
          - Creates new mount if needed
          - Tracks reference count for cleanup safety
          - Does NOT unmount on exit (mounts persist until tool deletion)

        Args:
            config: Filesystem connection configuration
            tool_config_id: Tool config ID for mount tracking (optional for backward compat)
        """
        mount_type = config.mount_type

        # Docker volume / local - no mounting needed
        if mount_type in ("docker_volume", "local"):
            yield Path(config.base_path)
            return

        mount_key = self._get_mount_key(config)

        # Check for existing mount
        with self._mount_lock:
            existing_mount = self._mounts.get(mount_key)

        if existing_mount:
            # Verify mount is still healthy
            if await self._check_mount_health(existing_mount):
                logger.info(f"Reusing existing mount for {mount_key}")
                with self._mount_lock:
                    existing_mount.reference_count += 1
                    existing_mount.last_accessed = datetime.now(timezone.utc)
                try:
                    yield Path(existing_mount.effective_path)
                finally:
                    with self._mount_lock:
                        existing_mount.reference_count -= 1
                return
            else:
                # Mount is stale, remove it and remount
                logger.warning(f"Existing mount {mount_key} is stale, remounting")
                await self._do_unmount(existing_mount.mount_point)
                with self._mount_lock:
                    del self._mounts[mount_key]

        # Create new mount
        prefix = "ragtime_smb_" if mount_type == "smb" else "ragtime_nfs_"
        mount_point = tempfile.mkdtemp(prefix=prefix)

        try:
            if mount_type == "smb":
                await self._do_mount_smb(config, mount_point)
            elif mount_type == "nfs":
                await self._do_mount_nfs(config, mount_point)
            else:
                shutil.rmtree(mount_point, ignore_errors=True)
                raise ValueError(f"Unknown mount type: {mount_type}")

            # Calculate effective path
            rel_path = config.base_path.lstrip("/")
            effective_path = (
                Path(mount_point) / rel_path if rel_path else Path(mount_point)
            )

            logger.info(f"Mount successful, effective path: {effective_path}")

            # Register the persistent mount
            mount_info = MountInfo(
                tool_config_id=tool_config_id or mount_key,
                mount_point=mount_point,
                mount_type=mount_type,
                effective_path=str(effective_path),
                reference_count=1,
            )
            with self._mount_lock:
                self._mounts[mount_key] = mount_info

            try:
                yield effective_path
            finally:
                # Decrement reference count but DON'T unmount
                with self._mount_lock:
                    if mount_key in self._mounts:
                        self._mounts[mount_key].reference_count -= 1

        except Exception:
            # Only cleanup on mount failure, not on job completion
            shutil.rmtree(mount_point, ignore_errors=True)
            raise

    async def unmount_tool(self, tool_config_id: str) -> bool:
        """
        Unmount filesystem for a tool config (called on tool deletion).

        Args:
            tool_config_id: The tool config ID to unmount

        Returns:
            True if unmounted, False if not found or still in use
        """
        mount_key_to_remove = None

        with self._mount_lock:
            for mount_key, mount_info in self._mounts.items():
                if mount_info.tool_config_id == tool_config_id:
                    if mount_info.reference_count > 0:
                        logger.warning(
                            f"Cannot unmount {mount_key}: {mount_info.reference_count} active jobs"
                        )
                        return False
                    mount_key_to_remove = mount_key
                    break

        if mount_key_to_remove:
            mount_info = self._mounts[mount_key_to_remove]
            await self._do_unmount(mount_info.mount_point)
            with self._mount_lock:
                del self._mounts[mount_key_to_remove]
            logger.info(f"Unmounted filesystem for tool {tool_config_id}")
            return True

        return False

    async def cleanup_all_mounts(self) -> int:
        """
        Unmount all filesystems (called on service shutdown).

        Returns:
            Number of mounts cleaned up
        """
        cleaned = 0
        with self._mount_lock:
            mount_keys = list(self._mounts.keys())

        for mount_key in mount_keys:
            mount_info = self._mounts.get(mount_key)
            if mount_info:
                await self._do_unmount(mount_info.mount_point)
                with self._mount_lock:
                    del self._mounts[mount_key]
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} mounts on shutdown")
        return cleaned

    def get_active_mounts(self) -> List[Dict[str, Any]]:
        """Get information about all active mounts for debugging."""
        with self._mount_lock:
            return [
                {
                    "mount_key": key,
                    "tool_config_id": info.tool_config_id,
                    "mount_point": info.mount_point,
                    "mount_type": info.mount_type,
                    "effective_path": info.effective_path,
                    "mounted_at": info.mounted_at.isoformat(),
                    "last_accessed": info.last_accessed.isoformat(),
                    "reference_count": info.reference_count,
                }
                for key, info in self._mounts.items()
            ]

    async def _cleanup_stale_jobs(self) -> int:
        """
        Clean up any jobs left in pending/indexing state from a previous run.

        This handles cases where the server was restarted while indexing was
        in progress. Those jobs will never complete, so mark them as failed.

        Returns the number of orphaned jobs cleaned up.
        """
        try:
            db = await get_db()

            # Find all jobs stuck in pending or indexing state
            result = await db.execute_raw(
                """
                UPDATE filesystem_index_jobs
                SET status = 'failed',
                    error_message = 'Job interrupted by server restart',
                    completed_at = NOW()
                WHERE status IN ('pending', 'indexing')
                RETURNING id
                """
            )

            # result is the count of updated rows
            count = result if isinstance(result, int) else 0
            if count > 0:
                logger.info(f"Cleaned up {count} orphaned filesystem indexing job(s)")
            return count

        except Exception as e:
            logger.warning(f"Failed to clean up orphaned jobs: {e}")
            return 0

    async def ensure_pgvector_extension(self) -> bool:
        """
        Ensure pgvector extension is installed in the database.

        Returns True if pgvector is available, False otherwise.
        """
        return await ensure_pgvector_extension(logger_override=logger)

    async def ensure_embedding_column(
        self, embedding_dim: int = 1536, index_lists: int = 100
    ) -> bool:
        """
        Ensure the embedding column exists with the expected dimension.

        If the column exists but the dimension changed (e.g., provider/model swap),
        alter the column and rebuild the index to match the new size.

        Note: pgvector has a 2000-dimension limit for both IVFFlat and HNSW indexes.
        For embeddings > 2000 dimensions, we skip index creation and use exact search.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_lists: IVFFlat lists parameter (higher = slower build, faster query)
        """
        return await ensure_embedding_column(
            table_name="filesystem_embeddings",
            index_name="filesystem_embeddings_embedding_idx",
            embedding_dim=embedding_dim,
            index_lists=index_lists,
            logger_override=logger,
        )

    async def validate_path_access(
        self, config: FilesystemConnectionConfig
    ) -> Dict[str, Any]:
        """
        Validate that the configured path is accessible.

        Returns a dict with:
        - success: bool
        - message: str
        - file_count: int (if success)
        - sample_files: list[str] (if success)
        """
        try:
            base_path = Path(config.base_path)

            if not base_path.exists():
                return {
                    "success": False,
                    "message": f"Path does not exist: {config.base_path}",
                }

            if not base_path.is_dir():
                return {
                    "success": False,
                    "message": f"Path is not a directory: {config.base_path}",
                }

            # Check read access
            if not os.access(base_path, os.R_OK):
                return {
                    "success": False,
                    "message": f"No read access to path: {config.base_path}",
                }

            # Count files matching patterns
            matching_files = self._collect_files(config, limit=100)
            sample_files = [str(f.relative_to(base_path)) for f in matching_files[:10]]

            return {
                "success": True,
                "message": f"Path accessible, found {len(matching_files)}+ matching files",
                "file_count": len(matching_files),
                "sample_files": sample_files,
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error accessing path: {str(e)}",
            }

    def _collect_files(
        self,
        config: FilesystemConnectionConfig,
        limit: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        cancel_check: Optional[callable] = None,
    ) -> List[Path]:
        """
        Collect files matching the configuration patterns.

        Respects file patterns, exclude patterns, extension whitelist, and size limits.
        Also intelligently skips cloud placeholder files (OneDrive/iCloud):
        - Symlinks (not followed)
        - Zero-byte files (cloud-only placeholders)

        Args:
            config: Filesystem configuration
            limit: Maximum number of files to collect
            progress_callback: Optional callback(files_found, current_dir) for progress updates
        """
        import os

        base_path = Path(config.base_path)
        matching_files: List[Path] = []
        max_size_bytes = config.max_file_size_mb * 1024 * 1024
        dirs_scanned = 0

        # Build exclude patterns for matching - uses shared matches_pattern from file_utils
        exclude_patterns = config.exclude_patterns

        # Use os.walk for better control and progress tracking
        walker = (
            os.walk(base_path, followlinks=False)
            if config.recursive
            else [(str(base_path), [], os.listdir(base_path))]
        )

        for dirpath, dirnames, filenames in walker:
            dirs_scanned += 1
            current_dir = Path(dirpath)

            # Check for cancellation between directories
            if cancel_check and cancel_check():
                return matching_files

            # Update progress every 10 directories or on first directory
            if progress_callback and (dirs_scanned == 1 or dirs_scanned % 10 == 0):
                rel_dir = (
                    str(current_dir.relative_to(base_path))
                    if current_dir != base_path
                    else "/"
                )
                progress_callback(len(matching_files), rel_dir)

            # Check if we've hit the limit
            if limit and len(matching_files) >= limit:
                return matching_files

            for filename in filenames:
                if cancel_check and cancel_check():
                    return matching_files
                file_path = current_dir / filename

                try:
                    # Skip symlinks - use lstat to avoid following
                    # Wrap in try/except as SMB can fail even on lstat
                    try:
                        stat_info = file_path.lstat()
                        # Check if it's a symlink using the mode
                        if stat.S_ISLNK(stat_info.st_mode):
                            continue
                    except OSError:
                        # Can't even lstat - skip this file (broken symlink, cloud placeholder, etc.)
                        continue

                    if not file_path.is_file():
                        continue

                    # Check if file matches patterns (use shared matches_pattern)
                    if not matches_pattern(filename, config.file_patterns):
                        continue

                    # Check file size and skip zero-byte files
                    # Re-use stat_info from lstat if available
                    file_size = stat_info.st_size
                    if file_size == 0:
                        continue
                    if file_size > max_size_bytes:
                        continue

                    # Check exclude patterns (use shared matches_pattern)
                    rel_path = str(file_path.relative_to(base_path))
                    if matches_pattern(rel_path, exclude_patterns):
                        continue

                    matching_files.append(file_path)

                    # Stop at max_total_files
                    if len(matching_files) >= config.max_total_files:
                        logger.warning(
                            f"Reached max_total_files limit ({config.max_total_files})"
                        )
                        return matching_files

                    if limit and len(matching_files) >= limit:
                        return matching_files

                except OSError as e:
                    # Skip any file that causes I/O errors (network issues, permissions, etc.)
                    logger.debug(f"Skipping inaccessible file {file_path}: {e}")
                    continue

        # Final progress update
        if progress_callback:
            progress_callback(len(matching_files), "Complete")

        return matching_files

    async def trigger_index(
        self,
        tool_config_id: str,
        config: FilesystemConnectionConfig,
        full_reindex: bool = False,
    ) -> FilesystemIndexJob:
        """
        Trigger filesystem indexing for a tool configuration.

        Args:
            tool_config_id: ID of the tool config
            config: Filesystem connection configuration
            full_reindex: If True, re-index all files regardless of hash
        """
        # Validate pgvector is available
        if not await self.ensure_pgvector_extension():
            raise RuntimeError(
                "pgvector extension not available. Please install it: "
                "CREATE EXTENSION IF NOT EXISTS vector;"
            )

        job = FilesystemIndexJob(
            id=str(uuid.uuid4()),
            tool_config_id=tool_config_id,
            index_name=config.index_name,
            status=FilesystemIndexStatus.PENDING,
        )

        # Create job in database
        await self._create_job(job)
        self._active_jobs[job.id] = job

        # Start processing in background
        task = asyncio.create_task(self._process_index(job, config, full_reindex))
        self._running_tasks[job.id] = task

        return job

    async def _create_job(self, job: FilesystemIndexJob) -> None:
        """Create a filesystem index job in the database."""
        db = await get_db()
        await db.filesystemindexjob.create(
            data={
                "id": job.id,
                "toolConfigId": job.tool_config_id,
                "status": job.status.value,
                "indexName": job.index_name,
                "totalFiles": job.total_files,
                "processedFiles": job.processed_files,
                "skippedFiles": job.skipped_files,
                "totalChunks": job.total_chunks,
                "processedChunks": job.processed_chunks,
                "errorMessage": job.error_message,
                "createdAt": job.created_at,
            }
        )

    async def _update_job(self, job: FilesystemIndexJob) -> None:
        """Update a filesystem index job in the database."""
        db = await get_db()
        await db.filesystemindexjob.update(
            where={"id": job.id},
            data={
                "status": job.status.value,
                "totalFiles": job.total_files,
                "processedFiles": job.processed_files,
                "skippedFiles": job.skipped_files,
                "totalChunks": job.total_chunks,
                "processedChunks": job.processed_chunks,
                "errorMessage": job.error_message,
                "startedAt": job.started_at,
                "completedAt": job.completed_at,
            },
        )

    async def get_job(self, job_id: str) -> Optional[FilesystemIndexJob]:
        """Get a filesystem index job by ID."""
        if job_id in self._active_jobs:
            return self._active_jobs[job_id]

        db = await get_db()
        prisma_job = await db.filesystemindexjob.find_unique(where={"id": job_id})

        if not prisma_job:
            return None

        return FilesystemIndexJob(
            id=prisma_job.id,
            tool_config_id=prisma_job.toolConfigId,
            status=FilesystemIndexStatus(prisma_job.status),
            index_name=prisma_job.indexName,
            total_files=prisma_job.totalFiles,
            processed_files=prisma_job.processedFiles,
            skipped_files=prisma_job.skippedFiles,
            total_chunks=prisma_job.totalChunks,
            processed_chunks=prisma_job.processedChunks,
            error_message=prisma_job.errorMessage,
            created_at=prisma_job.createdAt,
            started_at=prisma_job.startedAt,
            completed_at=prisma_job.completedAt,
        )

    async def list_jobs(
        self, tool_config_id: Optional[str] = None
    ) -> List[FilesystemIndexJob]:
        """List filesystem index jobs, optionally filtered by tool config."""
        db = await get_db()
        where = {"toolConfigId": tool_config_id} if tool_config_id else {}
        prisma_jobs = await db.filesystemindexjob.find_many(
            where=where,
            order={"createdAt": "desc"},
            take=50,
        )

        jobs = []
        for j in prisma_jobs:
            # Check if this job is active in memory (has live progress data)
            if j.id in self._active_jobs:
                # Use in-memory job which has real-time progress data
                jobs.append(self._active_jobs[j.id])
            else:
                # Use DB data for completed/historical jobs
                jobs.append(
                    FilesystemIndexJob(
                        id=j.id,
                        tool_config_id=j.toolConfigId,
                        status=FilesystemIndexStatus(j.status),
                        index_name=j.indexName,
                        total_files=j.totalFiles,
                        processed_files=j.processedFiles,
                        skipped_files=j.skippedFiles,
                        total_chunks=j.totalChunks,
                        processed_chunks=j.processedChunks,
                        error_message=j.errorMessage,
                        created_at=j.createdAt,
                        started_at=j.startedAt,
                        completed_at=j.completedAt,
                        cancel_requested=False,
                    )
                )

        return jobs

    async def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of an active indexing job.

        Returns True if cancellation was requested, False if job not found or already complete.
        """
        # Check if job is active in memory
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            if job.status in (
                FilesystemIndexStatus.PENDING,
                FilesystemIndexStatus.INDEXING,
            ):
                logger.info(f"Requesting cancellation for active job {job_id}")
                self._cancellation_flags[job_id] = True
                # Surface cancellation to clients immediately
                job.error_message = "Cancellation requested"
                job.cancel_requested = True
                await self._update_job(job)
                return True
            return False

        # Check database for the job
        db = await get_db()
        prisma_job = await db.filesystemindexjob.find_unique(where={"id": job_id})
        if not prisma_job:
            return False

        # Can only cancel pending/indexing jobs
        if prisma_job.status in ("pending", "indexing"):
            # Job is not running in memory but stuck in DB - directly mark as cancelled
            logger.info(
                f"Directly cancelling orphaned job {job_id} (not in active jobs)"
            )
            await db.filesystemindexjob.update(
                where={"id": job_id},
                data={
                    "status": "cancelled",
                    "errorMessage": "Job cancelled (was orphaned)",
                    "completedAt": datetime.now(timezone.utc),
                },
            )
            return True

        return False

    async def shutdown(self) -> None:
        """Shutdown the service and cancel all running tasks."""
        logger.info("Filesystem indexer service shutting down")
        self._shutdown = True

        # Cancel all running tasks
        for job_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Cancelled filesystem indexing task {job_id}")

        self._running_tasks.clear()
        self._active_jobs.clear()
        self._cancellation_flags.clear()

    async def retry_job(self, job_id: str) -> Optional[FilesystemIndexJob]:
        """
        Retry a failed or cancelled filesystem indexing job.

        Creates a new job using the same tool configuration.

        Args:
            job_id: The ID of the failed/cancelled job to retry

        Returns:
            The new job if successful, None if the original job wasn't found
            or wasn't in a retryable state.
        """
        from ragtime.indexer.repository import IndexerRepository

        repo = IndexerRepository()

        # Get the failed job
        failed_job = await self.get_job(job_id)
        if not failed_job:
            return None

        # Can only retry failed or cancelled jobs
        if failed_job.status not in (
            FilesystemIndexStatus.FAILED,
            FilesystemIndexStatus.CANCELLED,
        ):
            logger.warning(f"Cannot retry job {job_id} with status {failed_job.status}")
            return None

        # Get the tool config to start a new job
        tool_config = await repo.get_tool_config(failed_job.tool_config_id)
        if not tool_config:
            logger.error(
                f"Cannot retry job {job_id}: tool config {failed_job.tool_config_id} not found"
            )
            return None

        # Parse the connection config
        try:
            fs_config = FilesystemConnectionConfig(**tool_config.connection_config)
        except Exception as e:
            logger.error(f"Cannot retry job {job_id}: invalid config - {e}")
            return None

        # Start a new job
        logger.info(f"Retrying failed job {job_id} for tool {tool_config.name}")
        new_job = await self.trigger_index(
            tool_config_id=tool_config.id,
            config=fs_config,
            full_reindex=False,  # Incremental by default
        )

        return new_job

    def _is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been marked for cancellation."""
        return self._cancellation_flags.get(job_id, False)

    async def _get_file_metadata(
        self,
        index_name: str,
        file_path: str,
    ) -> Optional[FilesystemFileMetadata]:
        """Get file metadata for incremental indexing."""
        db = await get_db()
        result = await db.filesystemfilemetadata.find_first(
            where={"indexName": index_name, "filePath": file_path}
        )
        if not result:
            return None

        return FilesystemFileMetadata(
            id=result.id,
            index_name=result.indexName,
            file_path=result.filePath,
            file_hash=result.fileHash,
            file_size=int(result.fileSize),
            mime_type=result.mimeType,
            chunk_count=result.chunkCount,
            last_indexed=result.lastIndexed,
        )

    async def _upsert_file_metadata(self, metadata: FilesystemFileMetadata) -> None:
        """Upsert file metadata for tracking."""
        db = await get_db()
        await db.filesystemfilemetadata.upsert(
            where={
                "indexName_filePath": {
                    "indexName": metadata.index_name,
                    "filePath": metadata.file_path,
                }
            },
            data={
                "create": {
                    "id": metadata.id or str(uuid.uuid4()),
                    "indexName": metadata.index_name,
                    "filePath": metadata.file_path,
                    "fileHash": metadata.file_hash,
                    "fileSize": metadata.file_size,
                    "mimeType": metadata.mime_type,
                    "chunkCount": metadata.chunk_count,
                    "lastIndexed": metadata.last_indexed,
                },
                "update": {
                    "fileHash": metadata.file_hash,
                    "fileSize": metadata.file_size,
                    "mimeType": metadata.mime_type,
                    "chunkCount": metadata.chunk_count,
                    "lastIndexed": metadata.last_indexed,
                },
            },
        )

    async def _delete_file_embeddings(
        self,
        index_name: str,
        file_path: str,
        backend: Optional[VectorStoreBackend] = None,
    ) -> None:
        """Delete embeddings for a specific file (before re-indexing).

        Args:
            index_name: Name of the index
            file_path: Relative path of the file
            backend: Vector store backend to use (defaults to pgvector for backward compat)
        """
        if backend:
            await backend.delete_file_embeddings(index_name, file_path)
        else:
            # Legacy pgvector path (backward compatibility)
            db = await get_db()
            await db.filesystemembedding.delete_many(
                where={"indexName": index_name, "filePath": file_path}
            )

    async def _insert_embeddings(
        self,
        index_name: str,
        file_path: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        backend: Optional[VectorStoreBackend] = None,
    ) -> int:
        """Insert embeddings for a file.

        Args:
            index_name: Name of the index
            file_path: Relative path of the file
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: Additional metadata for each chunk
            backend: Vector store backend to use (defaults to pgvector for backward compat)

        Returns:
            Number of embeddings inserted
        """
        if backend:
            return await backend.store_embeddings(
                index_name, file_path, chunks, embeddings, metadata
            )

        # Legacy pgvector path (backward compatibility)
        import json

        db = await get_db()

        # Ensure embedding column exists with correct dimensions
        if embeddings:
            embedding_dim = len(embeddings[0])
            # Get ivfflat_lists from app settings
            from ragtime.core.app_settings import get_app_settings

            app_settings = await get_app_settings()
            index_lists = app_settings.get("ivfflat_lists", 100)
            success = await self.ensure_embedding_column(embedding_dim, index_lists)
            if not success:
                raise RuntimeError(
                    f"Failed to ensure embedding column with dimension {embedding_dim}"
                )

        # Serialize metadata to JSON string
        metadata_json = json.dumps(metadata).replace("'", "''")

        inserted = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_id = str(uuid.uuid4())
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

            # Insert using raw SQL because Prisma doesn't support vector type
            await db.execute_raw(
                f"""
                INSERT INTO filesystem_embeddings
                (id, index_name, file_path, chunk_index, content, metadata, embedding, created_at)
                VALUES (
                    '{embedding_id}',
                    '{index_name}',
                    '{file_path.replace("'", "''")}',
                    {i},
                    '{chunk.replace("'", "''")}',
                    '{metadata_json}'::jsonb,
                    '{embedding_str}'::vector,
                    NOW()
                )
            """
            )
            inserted += 1

        return inserted

    async def _process_index(
        self,
        job: FilesystemIndexJob,
        config: FilesystemConnectionConfig,
        full_reindex: bool,
    ) -> None:
        """Process filesystem indexing job."""
        try:
            job.status = FilesystemIndexStatus.INDEXING
            job.started_at = datetime.utcnow()
            await self._update_job(job)

            # Get app settings for embeddings
            from ragtime.core.app_settings import get_app_settings

            app_settings = await get_app_settings()

            # Check for embedding configuration mismatch
            # Auto-correct by forcing full re-index when config changes
            from ragtime.indexer.repository import IndexerRepository

            repo = IndexerRepository()
            settings = await repo.get_settings()

            current_config_hash = settings.get_embedding_config_hash()
            tracking_needs_update = (
                settings.embedding_dimension is None
                or settings.embedding_config_hash is None
            )

            if settings.embedding_config_hash is not None:
                if settings.embedding_config_hash != current_config_hash:
                    # Mismatch detected - auto-correct by treating as full reindex
                    logger.warning(
                        f"Embedding config changed: {settings.embedding_config_hash} -> {current_config_hash}. "
                        "Auto-triggering full re-index to correct dimension mismatch."
                    )
                    full_reindex = True  # Force full reindex to clear old embeddings
                    tracking_needs_update = True

            # Initialize embeddings
            embeddings = await self._get_embeddings(app_settings)

            # Get the appropriate vector store backend based on config
            vector_store_type = getattr(
                config, "vector_store_type", VectorStoreType.PGVECTOR
            )
            backend = get_backend(vector_store_type)
            logger.info(
                f"Using {vector_store_type.value} backend for index '{config.index_name}'"
            )

            # Mount filesystem if needed (SMB/NFS) and process files
            async with self._mount_filesystem(
                config, job.tool_config_id
            ) as effective_path:
                # Resolve OCR vision model - use config value if set, otherwise global default
                resolved_ocr_vision_model = config.ocr_vision_model
                if not resolved_ocr_vision_model and config.ocr_mode == OcrMode.OLLAMA:
                    resolved_ocr_vision_model = app_settings.get(
                        "default_ocr_vision_model"
                    )

                # Log OCR mode being used
                if config.ocr_mode == OcrMode.OLLAMA and resolved_ocr_vision_model:
                    logger.info(
                        f"Using Ollama Vision OCR with model: {resolved_ocr_vision_model}"
                    )
                elif config.ocr_mode == OcrMode.TESSERACT:
                    logger.info("Using Tesseract OCR")
                else:
                    logger.debug(f"OCR mode: {config.ocr_mode}")

                # Create a working config with the effective path for local access
                working_config = FilesystemConnectionConfig(
                    mount_type=config.mount_type,
                    base_path=str(effective_path),  # Use mounted path
                    index_name=config.index_name,
                    file_patterns=config.file_patterns,
                    exclude_patterns=config.exclude_patterns,
                    recursive=config.recursive,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    max_file_size_mb=config.max_file_size_mb,
                    max_total_files=config.max_total_files,
                    ocr_mode=config.ocr_mode,
                    ocr_vision_model=resolved_ocr_vision_model,
                )

                # Collect files to process - run in thread to avoid blocking on network filesystems
                logger.info(f"Collecting files from {effective_path}...")

                # Progress tracking for file collection
                collection_progress = {"files_found": 0, "current_dir": "Starting..."}

                def update_collection_progress(files_found: int, current_dir: str):
                    collection_progress["files_found"] = files_found
                    collection_progress["current_dir"] = current_dir

                # Start file collection in background
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._collect_files,
                        working_config,
                        None,
                        update_collection_progress,
                        lambda: self._is_cancelled(job.id),
                    )

                    # Poll for completion while updating job progress
                    while not future.done():
                        await asyncio.sleep(1)  # Check every second
                        if self._is_cancelled(job.id):
                            job.cancel_requested = True
                            job.error_message = "Cancellation requested"
                            await self._update_job(job)
                        # Update job with collection progress
                        job.files_scanned = collection_progress["files_found"]
                        job.current_directory = (
                            collection_progress["current_dir"][:50] + "..."
                            if len(collection_progress["current_dir"]) > 50
                            else collection_progress["current_dir"]
                        )
                        await self._update_job(job)

                    # Get the result
                    files = future.result()

                # Stop if cancellation was requested during collection
                if self._is_cancelled(job.id):
                    logger.info(f"Job {job.id} cancelled during collection")
                    job.status = FilesystemIndexStatus.CANCELLED
                    job.error_message = "Cancelled by user"
                    job.completed_at = datetime.utcnow()
                    job.cancel_requested = True
                    await self._update_job(job)
                    return

                job.total_files = len(files)
                await self._update_job(job)

                # Yield after file collection
                await asyncio.sleep(0)

                if not files:
                    job.status = FilesystemIndexStatus.COMPLETED
                    job.completed_at = datetime.utcnow()
                    job.error_message = "No files matched the configured patterns"
                    await self._update_job(job)
                    return

                logger.info(f"Found {len(files)} files to process")

                if full_reindex:
                    logger.info(
                        f"Full re-index requested. Clearing existing data for index '{config.index_name}'"
                    )
                    await self.delete_index(config.index_name)

                base_path = effective_path

                # Process files in parallel batches (like document indexer)
                # Concurrency limit: balance between parallelism and resource usage
                max_concurrent = min(32, os.cpu_count() or 8)
                file_semaphore = asyncio.Semaphore(max_concurrent)
                batch_size = max_concurrent * 2  # 2x concurrency for good pipeline

                logger.info(
                    f"Processing {len(files)} files with {max_concurrent} concurrent workers"
                )

                # Track OCR files for progress (OCR_EXTENSIONS imported at module level)
                ocr_file_count = 0
                ocr_files_processed = 0
                if config.ocr_mode != OcrMode.DISABLED:
                    ocr_file_count = sum(
                        1 for f in files if f.suffix.lower() in OCR_EXTENSIONS
                    )
                    if ocr_file_count > 0:
                        logger.info(
                            f"Found {ocr_file_count} image files for OCR processing"
                        )

                # Track files loaded during parallel processing (for progress updates)
                files_loaded_in_batch = 0
                files_loaded_lock = asyncio.Lock()

                async def process_single_file(
                    file_path: Path,
                ) -> tuple[str, list[str], str | None, str | None]:
                    """Process a single file: hash, check, load, chunk.

                    Returns: (rel_path, chunks, current_hash, error)
                    """
                    nonlocal ocr_files_processed, files_loaded_in_batch
                    async with file_semaphore:
                        rel_path = str(file_path.relative_to(base_path))
                        try:
                            # Check if file changed (incremental indexing)
                            current_hash = await asyncio.to_thread(
                                compute_file_hash, file_path
                            )

                            existing_meta = None
                            if not full_reindex:
                                existing_meta = await self._get_file_metadata(
                                    config.index_name, rel_path
                                )

                            if (
                                not full_reindex
                                and existing_meta
                                and existing_meta.file_hash == current_hash
                            ):
                                # File unchanged, skip
                                return (rel_path, [], current_hash, "unchanged")

                            # Update progress for OCR files
                            suffix = file_path.suffix.lower()
                            is_ocr_file = (
                                suffix in OCR_EXTENSIONS
                                and working_config.ocr_mode != OcrMode.DISABLED
                            )
                            if is_ocr_file and ocr_file_count > 0:
                                ocr_files_processed += 1
                                # Update job status to show OCR progress
                                job.current_directory = f"Vision OCR: {ocr_files_processed}/{ocr_file_count} images"
                                await self._update_job(job)

                            # Load and chunk the file
                            use_token_chunking = app_settings.get(
                                "chunking_use_tokens", True
                            )
                            chunks = await self._load_and_chunk_file(
                                file_path, working_config, use_token_chunking
                            )

                            # Track progress - increment counter after loading each file
                            async with files_loaded_lock:
                                files_loaded_in_batch += 1
                                # Update job periodically (every 5 files)
                                if files_loaded_in_batch % 5 == 0:
                                    await self._update_job(job)

                            return (rel_path, chunks, current_hash, None)
                        except Exception as e:
                            return (rel_path, [], None, str(e))

                # Process files in batches
                for batch_start in range(0, len(files), batch_size):
                    # Check for cancellation at batch boundaries
                    if self._is_cancelled(job.id):
                        logger.info(f"Job {job.id} cancelled by user")
                        job.status = FilesystemIndexStatus.CANCELLED
                        job.error_message = "Cancelled by user"
                        job.completed_at = datetime.utcnow()
                        job.cancel_requested = True
                        await self._update_job(job)
                        return

                    batch_end = min(batch_start + batch_size, len(files))
                    batch_files = files[batch_start:batch_end]

                    # Reset batch progress counter
                    files_loaded_in_batch = 0

                    # Initial progress update for this batch
                    job.current_directory = "Loading files"
                    await self._update_job(job)

                    # Process batch in parallel
                    tasks = [process_single_file(fp) for fp in batch_files]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Update progress after file loading phase completes for this batch
                    job.current_directory = (
                        f"Embedding batch {batch_start // batch_size + 1}"
                    )
                    await self._update_job(job)

                    # Collect chunks for batch embedding
                    files_to_embed: list[tuple[str, list[str], str, Path]] = []

                    for i, result in enumerate(results):
                        file_path = batch_files[i]
                        if isinstance(result, BaseException):
                            logger.debug(f"File processing exception: {result}")
                            job.processed_files += 1
                            continue

                        rel_path, chunks, current_hash, error = result

                        if error == "unchanged":
                            job.skipped_files += 1
                            continue

                        if error:
                            logger.debug(f"Skipped {rel_path}: {error}")
                            job.processed_files += 1
                            continue

                        if not chunks:
                            job.processed_files += 1
                            continue

                        # current_hash is guaranteed to be set when no error
                        assert current_hash is not None
                        files_to_embed.append(
                            (rel_path, chunks, current_hash, file_path)
                        )

                    # Update after categorizing results
                    await self._update_job(job)

                    # Batch embed all chunks from this batch
                    if files_to_embed:
                        # Flatten all chunks for batch embedding
                        all_chunks = []
                        chunk_file_map = []  # Track which file each chunk belongs to
                        for rel_path, chunks, _, _ in files_to_embed:
                            for chunk in chunks:
                                all_chunks.append(chunk)
                                chunk_file_map.append(rel_path)

                        # Generate embeddings for all chunks at once
                        try:
                            all_embeddings = await asyncio.to_thread(
                                embeddings.embed_documents, all_chunks
                            )
                        except Exception as e:
                            logger.error(f"Batch embedding failed: {e}")
                            for rel_path, _, _, _ in files_to_embed:
                                job.processed_files += 1
                            await self._update_job(job)
                            continue

                        # Validate embedding dimension (first in batch)
                        if all_embeddings and tracking_needs_update:
                            current_dim = len(all_embeddings[0])
                            if (
                                settings.embedding_dimension is not None
                                and settings.embedding_dimension != current_dim
                            ):
                                if not full_reindex:
                                    job.status = FilesystemIndexStatus.FAILED
                                    job.error_message = "Embedding dimension changed. A full re-index is required."
                                    job.completed_at = datetime.utcnow()
                                    await self._update_job(job)
                                    return
                                settings.embedding_dimension = current_dim
                                settings.embedding_config_hash = current_config_hash

                        # Distribute embeddings back to files and insert
                        embed_idx = 0
                        for rel_path, chunks, current_hash, file_path in files_to_embed:
                            chunk_count = len(chunks)
                            file_embeddings = all_embeddings[
                                embed_idx : embed_idx + chunk_count
                            ]
                            embed_idx += chunk_count

                            try:
                                # Delete old embeddings for this file
                                await self._delete_file_embeddings(
                                    config.index_name, rel_path, backend=backend
                                )

                                # Insert new embeddings
                                inserted = await self._insert_embeddings(
                                    index_name=config.index_name,
                                    file_path=rel_path,
                                    chunks=chunks,
                                    embeddings=file_embeddings,
                                    metadata={"source": rel_path},
                                    backend=backend,
                                )

                                # Update embedding tracking after first success
                                if inserted > 0 and tracking_needs_update:
                                    await repo.update_embedding_tracking(
                                        dimension=len(file_embeddings[0]),
                                        config_hash=current_config_hash,
                                    )
                                    tracking_needs_update = False
                                    settings.embedding_dimension = len(
                                        file_embeddings[0]
                                    )
                                    settings.embedding_config_hash = current_config_hash

                                # Update file metadata
                                file_stat = file_path.stat()
                                await self._upsert_file_metadata(
                                    FilesystemFileMetadata(
                                        index_name=config.index_name,
                                        file_path=rel_path,
                                        file_hash=current_hash,
                                        file_size=file_stat.st_size,
                                        mime_type=mimetypes.guess_type(str(file_path))[
                                            0
                                        ],
                                        chunk_count=chunk_count,
                                        last_indexed=datetime.utcnow(),
                                    )
                                )

                                job.processed_files += 1
                                job.processed_chunks += chunk_count
                                job.total_chunks += chunk_count

                            except Exception as e:
                                logger.warning(
                                    f"Error inserting embeddings for {rel_path}: {e}"
                                )
                                job.processed_files += 1

                    # Update job progress after each batch
                    await self._update_job(job)

                    if job.processed_files > 0 and job.processed_files % 50 == 0:
                        logger.info(
                            f"Processed {job.processed_files}/{job.total_files} files, "
                            f"skipped {job.skipped_files} unchanged"
                        )

            # Finalize the vector store (FAISS saves to disk here, pgvector is no-op)
            await backend.finalize_index(config.index_name)

            # Update tool config with last indexed timestamp (after the for loop, inside async with)
            await self._update_tool_config_last_indexed(job.tool_config_id)

            job.status = FilesystemIndexStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            logger.info(
                f"Indexing completed: {job.processed_files} processed, "
                f"{job.skipped_files} skipped, {job.total_chunks} chunks "
                f"(backend: {vector_store_type.value})"
            )

        except asyncio.CancelledError:
            # Task was cancelled during shutdown - don't try to update DB
            logger.info(f"Filesystem indexing task {job.id} cancelled")
            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)
            raise
        except Exception as e:
            logger.exception(f"Filesystem indexing failed: {e}")
            job.status = FilesystemIndexStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()

            # Only try to update DB if not shutting down
            if not self._shutdown:
                try:
                    await self._update_job(job)
                except RuntimeError as db_error:
                    if "Database is not connected" in str(db_error):
                        logger.warning(
                            f"Cannot update job {job.id} - DB disconnected during shutdown"
                        )
                    else:
                        raise
            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)
        else:
            # Success path - update DB and clean up
            await self._update_job(job)

            # Post-completion actions for FAISS indexes
            if vector_store_type == VectorStoreType.FAISS:
                try:
                    # 1. Load the new index into memory immediately
                    from ragtime.indexer.vector_backends import get_faiss_backend

                    await get_faiss_backend().load_index(job.index_name, embeddings)

                    # 2. Re-enable tool if it was auto-disabled
                    tool_config = await repository.get_tool_config(job.tool_config_id)
                    if tool_config and not tool_config.enabled:
                        logger.info(
                            f"Re-enabling filesystem tool {job.tool_config_id} after successful indexing"
                        )
                        await repository.update_tool_config(
                            job.tool_config_id, {"enabled": True}
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to post-process FAISS index after completion: {e}"
                    )

            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)

    async def _update_tool_config_last_indexed(self, tool_config_id: str) -> None:
        """Update the tool config with last indexed timestamp."""
        try:
            # Use repository to get decrypted config and properly re-encrypt on update
            tool_config = await repository.get_tool_config(tool_config_id)
            if tool_config and tool_config.connection_config:
                connection_config = dict(tool_config.connection_config)
                connection_config["last_indexed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                await repository.update_tool_config(
                    tool_config_id, {"connection_config": connection_config}
                )
        except Exception as e:
            logger.warning(f"Failed to update last_indexed_at: {e}")

    async def _get_embeddings(self, app_settings: dict):
        """Get the configured embedding model based on app settings."""
        return await get_embeddings_model(
            app_settings,
            logger_override=logger,
        )

    async def _load_and_chunk_file(
        self,
        file_path: Path,
        config: FilesystemConnectionConfig,
        use_token_chunking: bool = True,
    ) -> List[str]:
        """Load a file and split it into chunks.

        Uses Chonkie for all text chunking:
        - CodeChunker for code files (AST-based with Magika detection)
        - RecursiveChunker for plain text and documents
        - Semantic chunking for images (keeps classification with description)

        When use_token_chunking=True, chunk sizes are measured in tokens (tiktoken)
        rather than characters, ensuring chunks don't exceed embedding model limits.

        Text extraction (PDF, DOCX, images with OCR, etc.) is handled by
        document_parser via _read_file_content() before chunking.
        """
        from ragtime.core.file_constants import OCR_EXTENSIONS
        from ragtime.indexer.chunking import (
            _chunk_with_chonkie_code,
            _chunk_with_recursive,
            chunk_semantic_segments,
        )

        try:
            file_path_str = str(file_path)
            metadata = {"source": file_path_str}
            suffix = file_path.suffix.lower()

            # For images with Ollama OCR: use semantic chunking to keep
            # classification metadata together with description
            if suffix in OCR_EXTENSIONS and config.ocr_mode.value == "ollama":
                docs = await self._load_image_with_semantic_chunks(
                    file_path, config, metadata
                )
                if docs:
                    return [doc.page_content for doc in docs]
                # Fall through to standard processing if semantic extraction fails

            # Read file content (handles PDF, DOCX, images with OCR, etc.)
            content = await self._read_file_content_async(
                file_path,
                ocr_mode=(
                    config.ocr_mode.value
                    if hasattr(config.ocr_mode, "value")
                    else str(config.ocr_mode)
                ),
                ocr_vision_model=config.ocr_vision_model,
            )
            if not content:
                return []

            # Try AST-based code chunking with auto language detection
            try:
                docs = await asyncio.to_thread(
                    _chunk_with_chonkie_code,
                    content,
                    config.chunk_size,
                    config.chunk_overlap,
                    metadata,
                    use_token_chunking,
                )
                return [doc.page_content for doc in docs]
            except (ValueError, RuntimeError, LookupError) as e:
                # Language not supported by Chonkie - use recursive chunker
                err_lower = str(e).lower()
                if (
                    "not supported" in err_lower
                    or "detected language" in err_lower
                    or "could not find language" in err_lower
                ):
                    logger.debug(
                        f"Code chunking not available for {file_path.name}, "
                        f"using recursive chunker"
                    )
                    docs = await asyncio.to_thread(
                        _chunk_with_recursive,
                        content,
                        config.chunk_size,
                        config.chunk_overlap,
                        metadata,
                        use_token_chunking,
                    )
                    return [doc.page_content for doc in docs]
                raise

        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {e}")
            return []

    async def _load_image_with_semantic_chunks(
        self,
        file_path: Path,
        config: FilesystemConnectionConfig,
        metadata: dict,
    ) -> List:
        """
        Load an image and chunk using semantic boundaries from vision OCR.

        Uses structured OCR output to keep related content together:
        - OCR text stays together (or is chunked if very large)
        - Classification metadata (description + tags) always stays together

        Args:
            file_path: Path to image file
            config: Filesystem indexer config
            metadata: Base metadata for chunks

        Returns:
            List of Document objects, or empty list if extraction fails
        """
        from ragtime.core.app_settings import get_app_settings
        from ragtime.indexer.chunking import chunk_semantic_segments
        from ragtime.indexer.document_parser import extract_image_structured_async

        # Get Ollama base URL
        app_settings = await get_app_settings()
        ollama_base_url = app_settings.get("ollama_base_url", "http://localhost:11434")

        # Get structured OCR result (semaphore is applied at vision_models level)
        result = await extract_image_structured_async(
            file_path,
            ocr_vision_model=config.ocr_vision_model,
            ollama_base_url=ollama_base_url,
        )

        if not result:
            logger.debug(f"Structured OCR failed for {file_path}, using fallback")
            return []

        # Handle raw text fallback (when structured parsing failed)
        if result.raw_text and not result.extracted_text:
            from ragtime.indexer.chunking import _chunk_with_recursive

            docs = await asyncio.to_thread(
                _chunk_with_recursive,
                result.raw_text,
                config.chunk_size,
                config.chunk_overlap,
                metadata,
            )
            return docs

        # Get semantic segments and chunk them
        segments = result.get_semantic_segments()
        if not segments:
            logger.debug(f"No semantic segments extracted from {file_path}")
            return []

        docs = chunk_semantic_segments(
            segments,
            config.chunk_size,
            config.chunk_overlap,
            metadata,
        )

        logger.debug(
            f"Semantic chunking for {file_path.name}: "
            f"{len(segments)} segments -> {len(docs)} chunks"
        )
        return docs

    async def _read_file_content_async(
        self,
        file_path: Path,
        ocr_mode: str = "disabled",
        ocr_vision_model: Optional[str] = None,
    ) -> str:
        """Read file content using the unified document parser (handles OCR, docs, text).

        Uses async extraction to support Ollama vision OCR.
        """
        from ragtime.core.app_settings import get_app_settings
        from ragtime.indexer.document_parser import extract_text_from_file_async

        # Get Ollama base URL from app settings for vision OCR
        ollama_base_url = None
        if ocr_mode == "ollama":
            app_settings = await get_app_settings()
            ollama_base_url = app_settings.get(
                "ollama_base_url", "http://localhost:11434"
            )

        # Semaphore is applied at vision_models level for Ollama requests
        text = await extract_text_from_file_async(
            file_path,
            ocr_mode=ocr_mode,  # type: ignore
            ocr_vision_model=ocr_vision_model,
            ollama_base_url=ollama_base_url,
        )

        if text:
            return text

        logger.debug(f"No content extracted from {file_path}")
        return ""

    # =========================================================================
    # FILESYSTEM ANALYSIS
    # =========================================================================

    async def start_analysis(
        self,
        tool_config_id: str,
        config: FilesystemConnectionConfig,
    ) -> FilesystemAnalysisJob:
        """
        Start a filesystem analysis job.

        Runs in background and provides progress updates.
        """
        job = FilesystemAnalysisJob(
            id=str(uuid.uuid4()),
            tool_config_id=tool_config_id,
            status=FilesystemAnalysisStatus.PENDING,
        )

        self._analysis_jobs[job.id] = job

        # Start analysis in background
        asyncio.create_task(self._run_analysis(job, config))

        return job

    async def get_analysis_job(
        self, job_id: str
    ) -> Optional[tuple[FilesystemAnalysisJob, Optional[FilesystemAnalysisResult]]]:
        """Get an analysis job and its result if completed."""
        job = self._analysis_jobs.get(job_id)
        if not job:
            return None
        result = self._analysis_results.get(job_id)
        return (job, result)

    async def _run_analysis(
        self,
        job: FilesystemAnalysisJob,
        config: FilesystemConnectionConfig,
    ) -> None:
        """Run filesystem analysis with progress tracking."""
        import time
        from collections import defaultdict

        start_time = time.time()

        try:
            job.status = FilesystemAnalysisStatus.SCANNING

            # Mount filesystem if needed (SMB/NFS)
            async with self._mount_filesystem(
                config, job.tool_config_id
            ) as effective_path:
                base_path = effective_path

                if not base_path.exists():
                    job.status = FilesystemAnalysisStatus.FAILED
                    job.error_message = f"Path does not exist: {effective_path}"
                    job.completed_at = datetime.now(timezone.utc)
                    return

                if not base_path.is_dir():
                    job.status = FilesystemAnalysisStatus.FAILED
                    job.error_message = f"Path is not a directory: {effective_path}"
                    job.completed_at = datetime.now(timezone.utc)
                    return

                # First pass: count directories for progress estimation
                # Run in thread to avoid blocking
                def count_dirs() -> int:
                    count = 0
                    try:
                        for entry in base_path.rglob("*"):
                            if entry.is_dir():
                                count += 1
                    except PermissionError:
                        pass
                    return max(count, 1)  # At least 1 to avoid division by zero

                job.total_dirs_to_scan = await asyncio.to_thread(count_dirs)
                await asyncio.sleep(0)  # Yield

                # File extension stats
                ext_stats: Dict[str, Dict[str, Any]] = defaultdict(
                    lambda: {
                        "file_count": 0,
                        "total_size": 0,
                        "sample_files": [],
                    }
                )

                max_size_bytes = config.max_file_size_mb * 1024 * 1024
                exclude_patterns = config.exclude_patterns

                total_files = 0
                total_size = 0
                skipped_size = 0
                skipped_excluded = 0
                warnings: List[str] = []
                suggested_exclusions: List[str] = []
                large_files: List[tuple] = []
                dirs_processed = 0

                # Walk directory tree with progress updates
                def scan_directory() -> None:
                    nonlocal total_files, total_size, skipped_size
                    nonlocal skipped_excluded, dirs_processed

                    for pattern in config.file_patterns:
                        glob_pattern = pattern.removeprefix("**/").removeprefix("*/")
                        glob_func = (
                            base_path.rglob if config.recursive else base_path.glob
                        )

                        current_dir = ""
                        for file_path in glob_func(glob_pattern):
                            # Track directory changes for progress
                            file_dir = str(file_path.parent)
                            if file_dir != current_dir:
                                current_dir = file_dir
                                dirs_processed += 1
                                job.dirs_scanned = dirs_processed
                                # Truncate for display
                                rel_dir = file_dir.replace(str(base_path), "")
                                job.current_directory = (
                                    rel_dir[:50] + "..."
                                    if len(rel_dir) > 50
                                    else rel_dir
                                )

                            # Skip symlinks
                            if file_path.is_symlink():
                                continue

                            if not file_path.is_file():
                                continue

                            ext = file_path.suffix.lower() or "(no extension)"
                            rel_path = str(file_path.relative_to(base_path))

                            # Check exclude patterns (use shared matches_pattern)
                            if matches_pattern(rel_path, exclude_patterns):
                                skipped_excluded += 1
                                continue

                            # Get file size
                            try:
                                file_size = file_path.stat().st_size
                                if file_size == 0:
                                    continue  # Skip zero-byte files
                                if file_size > max_size_bytes:
                                    skipped_size += 1
                                    large_files.append((rel_path, file_size))
                                    continue
                            except OSError:
                                continue

                            # Track stats
                            job.files_scanned = total_files + 1
                            total_files += 1
                            total_size += file_size

                            stats = ext_stats[ext]
                            stats["file_count"] += 1
                            stats["total_size"] += file_size
                            if len(stats["sample_files"]) < 5:
                                stats["sample_files"].append(rel_path)

                            # Safety limit
                            if total_files >= config.max_total_files:
                                return

                # Run scan in thread
                await asyncio.to_thread(scan_directory)

                job.status = FilesystemAnalysisStatus.ANALYZING

                # Calculate estimated chunks per file type
                for ext, stats in ext_stats.items():
                    effective_chunk = config.chunk_size - config.chunk_overlap
                    if effective_chunk > 0:
                        # For images/OCR, the file size is NOT representative of text content.
                        # Apply a drastic reduction factor (1/100) to estimate actual text content.
                        if ext in OCR_EXTENSIONS:
                            density_factor = 0.01
                        # PDF/Office docs contain significant formatting overhead/binary data
                        # Apply reduction factor (1/10)
                        elif ext in PARSEABLE_DOCUMENT_EXTENSIONS:
                            density_factor = 0.1
                        else:
                            density_factor = 1.0

                        estimated_text_size = stats["total_size"] * density_factor

                        stats["estimated_chunks"] = max(
                            1, int(estimated_text_size // effective_chunk)
                        )
                    else:
                        stats["estimated_chunks"] = stats["file_count"]

                # Get LLM-powered exclusion suggestions
                from ragtime.indexer.llm_exclusions import (
                    get_smart_exclusion_suggestions,
                )

                smart_exclusions, _used_llm = await get_smart_exclusion_suggestions(
                    ext_stats=dict(ext_stats),
                    repo_name=config.index_name or base_path.name,
                )
                suggested_exclusions.extend(smart_exclusions)

                # Generate warnings
                if skipped_size > 0:
                    warnings.append(
                        f"Skipped {skipped_size} files exceeding the {config.max_file_size_mb}MB size limit."
                    )

                if skipped_excluded > 0:
                    warnings.append(
                        f"Skipped {skipped_excluded} files matching exclude patterns."
                    )

                if total_files >= config.max_total_files:
                    warnings.append(
                        f"Reached max_total_files limit ({config.max_total_files}). "
                        "Consider adding more exclude patterns."
                    )

                # Check for parseable documents that can be handled
                parseable_found = [
                    ext for ext in ext_stats if ext in PARSEABLE_DOCUMENT_EXTENSIONS
                ]
                if parseable_found:
                    doc_count = sum(
                        ext_stats[ext]["file_count"] for ext in parseable_found
                    )
                    warnings.append(
                        f"Found {doc_count} document files ({', '.join(parseable_found)}) - "
                        "these will be parsed using document extractors."
                    )

                # Check for OCR-eligible images
                ocr_images_found = [ext for ext in ext_stats if ext in OCR_EXTENSIONS]
                if ocr_images_found:
                    img_count = sum(
                        ext_stats[ext]["file_count"] for ext in ocr_images_found
                    )
                    ocr_mode_str = (
                        config.ocr_mode.value
                        if hasattr(config.ocr_mode, "value")
                        else str(config.ocr_mode)
                    )
                    if ocr_mode_str != "disabled":
                        ocr_method = (
                            "Tesseract"
                            if ocr_mode_str == "tesseract"
                            else f"Ollama Vision ({config.ocr_vision_model or 'not configured'})"
                        )
                        warnings.append(
                            f"Found {img_count} image files ({', '.join(ocr_images_found)}) - "
                            f"these will be processed with {ocr_method} OCR to extract text."
                        )
                    else:
                        warnings.append(
                            f"Found {img_count} image files ({', '.join(ocr_images_found)}) - "
                            "these will be skipped. Enable OCR to extract text from images."
                        )

                # Calculate totals
                total_estimated_chunks = sum(
                    stats["estimated_chunks"] for stats in ext_stats.values()
                )

                # Estimate pgvector index size (embeddings + overhead)
                # Each embedding: 4 bytes per dimension + metadata
                # Assume 1536 dimensions (OpenAI default), ~6KB per embedding
                estimated_index_size_mb = (total_estimated_chunks * 6) / 1024

                # Build file type stats list
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

                # Size warning
                if estimated_index_size_mb > 500:
                    warnings.insert(
                        0,
                        f"Estimated index size is {estimated_index_size_mb:.0f}MB. "
                        "Consider adding more exclusion patterns to reduce size.",
                    )

                # Warn if embedding dimension exceeds pgvector index limit
                await self._append_embedding_dimension_warning(warnings)

                elapsed = time.time() - start_time

                result = FilesystemAnalysisResult(
                    total_files=total_files,
                    total_size_bytes=total_size,
                    total_size_mb=round(total_size / (1024 * 1024), 2),
                    estimated_chunks=total_estimated_chunks,
                    estimated_index_size_mb=round(estimated_index_size_mb, 2),
                    file_type_stats=file_type_list,
                    suggested_exclusions=list(set(suggested_exclusions)),
                    warnings=warnings,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    analysis_duration_seconds=round(elapsed, 2),
                    directories_scanned=job.dirs_scanned,
                )

                self._analysis_results[job.id] = result
                job.status = FilesystemAnalysisStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)

                logger.info(
                    f"Filesystem analysis completed: {total_files} files, "
                    f"{total_estimated_chunks} chunks, {elapsed:.1f}s"
                )

        except Exception as e:
            logger.exception(f"Filesystem analysis failed: {e}")
            job.status = FilesystemAnalysisStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)

    async def delete_index(
        self,
        index_name: str,
        vector_store_type: Optional[VectorStoreType] = None,
    ) -> int:
        """
        Delete all embeddings and metadata for an index.

        Args:
            index_name: Name of the index to delete
            vector_store_type: Vector store type (auto-detects if not specified)

        Returns the number of embeddings deleted.
        """
        db = await get_db()
        deleted_count = 0

        # Delete from pgvector (always try, for backward compatibility)
        if vector_store_type in (None, VectorStoreType.PGVECTOR):
            result = await db.filesystemembedding.delete_many(
                where={"indexName": index_name}
            )
            deleted_count += result
            logger.info(f"Deleted {result} pgvector embeddings for '{index_name}'")

        # Delete from FAISS if applicable
        if vector_store_type in (None, VectorStoreType.FAISS):
            faiss_deleted = await get_faiss_backend().delete_index(index_name)
            if faiss_deleted > 0:
                deleted_count += faiss_deleted
                logger.info(f"Deleted FAISS index for '{index_name}'")

        # Delete file metadata (shared between both backends)
        await db.filesystemfilemetadata.delete_many(where={"indexName": index_name})

        logger.info(f"Deleted index '{index_name}': {deleted_count} embeddings removed")
        return deleted_count

    async def get_index_stats(
        self,
        index_name: str,
        vector_store_type: Optional[VectorStoreType] = None,
    ) -> Dict[str, Any]:
        """Get statistics for a filesystem index.

        Args:
            index_name: Name of the index
            vector_store_type: Vector store type (auto-detects if not specified)
        """
        from prisma.errors import TableNotFoundError

        from ragtime.core.app_settings import get_app_settings

        db = await get_db()
        app_settings = await get_app_settings()

        # Count embeddings from both backends
        # Tables may not exist if no indexing has been done yet
        pgvector_count = 0
        faiss_count = 0

        if vector_store_type in (None, VectorStoreType.PGVECTOR):
            try:
                pgvector_count = await db.filesystemembedding.count(
                    where={"indexName": index_name}
                )
            except TableNotFoundError:
                # Table doesn't exist yet - no embeddings
                pgvector_count = 0

        if vector_store_type in (None, VectorStoreType.FAISS):
            faiss_stats = await get_faiss_backend().get_index_stats(index_name)
            faiss_count = faiss_stats.get("embedding_count", 0)

        embedding_count = pgvector_count + faiss_count

        # Count unique files - table may not exist yet
        file_count = 0
        latest_file = None
        try:
            file_count = await db.filesystemfilemetadata.count(
                where={"indexName": index_name}
            )

            # Get latest indexed timestamp
            latest_file = await db.filesystemfilemetadata.find_first(
                where={"indexName": index_name},
                order={"lastIndexed": "desc"},
            )
        except TableNotFoundError:
            # Table doesn't exist yet - no files indexed
            file_count = 0
            latest_file = None

        # Calculate memory - for FAISS use actual disk size, for pgvector estimate
        estimated_memory_mb = 0.0

        if faiss_count > 0 and vector_store_type == VectorStoreType.FAISS:
            # For FAISS, use actual disk size from stats
            faiss_stats = await get_faiss_backend().get_index_stats(index_name)
            estimated_memory_mb = faiss_stats.get("size_mb") or 0.0
        elif pgvector_count > 0:
            # For pgvector, estimate from embedding dimensions
            embedding_dimension = app_settings.get("embedding_dimension")
            if embedding_dimension and embedding_count > 0:
                # Memory formula: embeddings * dimensions * 4 bytes (float32) * overhead
                bytes_per_embedding = embedding_dimension * 4 * 1.15
                estimated_memory_mb = (embedding_count * bytes_per_embedding) / (
                    1024 * 1024
                )

        return {
            "index_name": index_name,
            "embedding_count": embedding_count,
            "chunk_count": embedding_count,
            "file_count": file_count,
            "last_indexed": latest_file.lastIndexed if latest_file else None,
            "estimated_memory_mb": estimated_memory_mb,
            "vector_store_type": (
                vector_store_type.value if vector_store_type else "auto"
            ),
            "pgvector_count": pgvector_count,
            "faiss_count": faiss_count,
        }


# Global service instance
filesystem_indexer = FilesystemIndexerService()
