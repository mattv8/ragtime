"""
Filesystem Indexer Service - Creates and manages pgvector-based filesystem indexes.

This service handles:
- Indexing files from Docker volumes, SMB shares, NFS mounts, or local paths
- Incremental indexing (skip unchanged files based on SHA-256 hash)
- Storing embeddings in PostgreSQL using pgvector
- Progress tracking and job management
"""

import asyncio
import fnmatch
import hashlib
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import uuid

from ragtime.core.logging import get_logger
from ragtime.core.database import get_db
from ragtime.indexer.models import (
    FilesystemConnectionConfig,
    FilesystemIndexJob,
    FilesystemIndexStatus,
    FilesystemFileMetadata,
)

logger = get_logger(__name__)


class FilesystemIndexerService:
    """Service for creating and managing filesystem indexes with pgvector."""

    def __init__(self):
        self._active_jobs: Dict[str, FilesystemIndexJob] = {}
        self._cancellation_flags: Dict[str, bool] = {}  # job_id -> should_cancel
        self._pgvector_validated = False

    async def cleanup_orphaned_jobs(self) -> int:
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
        if self._pgvector_validated:
            return True

        try:
            db = await get_db()

            # Check if pgvector extension exists
            result = await db.query_raw(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            )

            if not result:
                # Try to create the extension
                logger.info("pgvector extension not found, attempting to create...")
                try:
                    await db.execute_raw(
                        "CREATE EXTENSION IF NOT EXISTS vector"
                    )
                    logger.info("pgvector extension created successfully")
                except Exception as e:
                    logger.error(f"Failed to create pgvector extension: {e}")
                    logger.error(
                        "Please run: CREATE EXTENSION IF NOT EXISTS vector; "
                        "as a superuser in your PostgreSQL database"
                    )
                    return False

            self._pgvector_validated = True
            return True

        except Exception as e:
            logger.error(f"Error checking pgvector extension: {e}")
            return False

    async def ensure_embedding_column(self, embedding_dim: int = 1536) -> bool:
        """
        Ensure the embedding column exists with the expected dimension.

        If the column exists but the dimension changed (e.g., provider/model swap),
        alter the column and rebuild the index to match the new size.

        Note: pgvector has a 2000-dimension limit for both IVFFlat and HNSW indexes.
        For embeddings > 2000 dimensions, we skip index creation and use exact search.
        """
        try:
            db = await get_db()

            # Inspect existing column and its dimension (atttypmod = dims + 4 for vector)
            result = await db.query_raw("""
                SELECT a.atttypmod AS typmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE c.relname = 'filesystem_embeddings'
                  AND n.nspname = 'public'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
            """)

            current_dim: Optional[int] = None
            if result:
                typmod = result[0].get("typmod")
                if isinstance(typmod, int) and typmod > 4:
                    current_dim = typmod - 4

            dimension_changed = current_dim is not None and current_dim != embedding_dim

            if not result:
                logger.info(f"Adding embedding column (vector({embedding_dim}))")
                await db.execute_raw(f"""
                    ALTER TABLE filesystem_embeddings
                    ADD COLUMN IF NOT EXISTS embedding vector({embedding_dim})
                """)
            elif dimension_changed:
                logger.info(
                    f"Updating embedding column dimension {current_dim} -> {embedding_dim}"
                )
                # Drop old index before altering column
                await db.execute_raw(
                    "DROP INDEX IF EXISTS filesystem_embeddings_embedding_idx"
                )
                await db.execute_raw(
                    f"ALTER TABLE filesystem_embeddings "
                    f"ALTER COLUMN embedding TYPE vector({embedding_dim})"
                )

            # pgvector max dimension for ANN indexes is 2000
            # For higher dimensions, skip indexing and use exact (sequential) search
            if embedding_dim > 2000:
                logger.warning(
                    f"Embedding dimension {embedding_dim} exceeds pgvector's 2000-dim index limit. "
                    "Using exact (non-indexed) search which may be slower for large datasets. "
                    "To enable fast indexed search, set 'Embedding Dimensions' to 1536 or less "
                    "in Settings (OpenAI text-embedding-3-* models support dimension reduction)."
                )
                # Drop any existing index since it's incompatible
                await db.execute_raw(
                    "DROP INDEX IF EXISTS filesystem_embeddings_embedding_idx"
                )
                return True

            # Check if index exists
            index_info = await db.query_raw("""
                SELECT am.amname AS index_type
                FROM pg_index i
                JOIN pg_class c ON i.indexrelid = c.oid
                JOIN pg_am am ON c.relam = am.oid
                WHERE c.relname = 'filesystem_embeddings_embedding_idx'
            """)

            # Recreate index if dimension changed or no index exists
            if dimension_changed or not index_info:
                await db.execute_raw(
                    "DROP INDEX IF EXISTS filesystem_embeddings_embedding_idx"
                )
                logger.info(f"Creating IVFFlat index for {embedding_dim}-dim embeddings")
                await db.execute_raw("""
                    CREATE INDEX filesystem_embeddings_embedding_idx
                    ON filesystem_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

            return True

        except Exception as e:
            logger.error(f"Error ensuring embedding column: {e}")
            return False

    async def validate_path_access(self, config: FilesystemConnectionConfig) -> Dict[str, Any]:
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
        limit: Optional[int] = None
    ) -> List[Path]:
        """
        Collect files matching the configuration patterns.

        Respects file patterns, exclude patterns, extension whitelist, and size limits.
        Also intelligently skips cloud placeholder files (OneDrive/iCloud):
        - Symlinks (not followed)
        - Zero-byte files (cloud-only placeholders)
        """
        base_path = Path(config.base_path)
        matching_files: List[Path] = []
        max_size_bytes = config.max_file_size_mb * 1024 * 1024

        # Normalize allowed extensions
        allowed_exts = set(
            ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            for ext in config.allowed_extensions
        )

        def should_exclude(rel_path: str) -> bool:
            for pattern in config.exclude_patterns:
                clean_pattern = pattern.removeprefix("**/")
                if fnmatch.fnmatch(rel_path, clean_pattern) or fnmatch.fnmatch(rel_path, pattern):
                    return True
            return False

        for pattern in config.file_patterns:
            glob_pattern = pattern.removeprefix("**/").removeprefix("*/")
            glob_func = base_path.rglob if config.recursive else base_path.glob

            for file_path in glob_func(glob_pattern):
                if limit and len(matching_files) >= limit:
                    return matching_files

                # Skip symlinks - cloud services often use these for placeholders
                if file_path.is_symlink():
                    continue

                if not file_path.is_file():
                    continue

                # Check extension whitelist
                if file_path.suffix.lower() not in allowed_exts:
                    continue

                # Check file size and skip zero-byte files (cloud placeholders)
                try:
                    file_size = file_path.stat().st_size
                    if file_size == 0:
                        # Skip zero-byte files (OneDrive/iCloud cloud-only placeholders)
                        continue
                    if file_size > max_size_bytes:
                        continue
                except OSError:
                    continue

                # Check exclude patterns
                rel_path = str(file_path.relative_to(base_path))
                if should_exclude(rel_path):
                    continue

                if file_path not in matching_files:
                    matching_files.append(file_path)

                    # Stop at max_total_files
                    if len(matching_files) >= config.max_total_files:
                        logger.warning(
                            f"Reached max_total_files limit ({config.max_total_files})"
                        )
                        return matching_files

        return matching_files

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file for change detection."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Error hashing file {file_path}: {e}")
            return ""

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
        asyncio.create_task(
            self._process_index(job, config, full_reindex)
        )

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
            }
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

    async def list_jobs(self, tool_config_id: Optional[str] = None) -> List[FilesystemIndexJob]:
        """List filesystem index jobs, optionally filtered by tool config."""
        db = await get_db()
        where = {"toolConfigId": tool_config_id} if tool_config_id else {}
        prisma_jobs = await db.filesystemindexjob.find_many(
            where=where,
            order={"createdAt": "desc"},
            take=50,
        )

        return [
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
            )
            for j in prisma_jobs
        ]

    async def cancel_job(self, job_id: str) -> bool:
        """
        Request cancellation of an active indexing job.

        Returns True if cancellation was requested, False if job not found or already complete.
        """
        # Check if job is active in memory
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            if job.status in (FilesystemIndexStatus.PENDING, FilesystemIndexStatus.INDEXING):
                logger.info(f"Requesting cancellation for active job {job_id}")
                self._cancellation_flags[job_id] = True
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
            logger.info(f"Directly cancelling orphaned job {job_id} (not in active jobs)")
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
            }
        )

    async def _delete_file_embeddings(self, index_name: str, file_path: str) -> None:
        """Delete embeddings for a specific file (before re-indexing)."""
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
    ) -> int:
        """Insert embeddings for a file into pgvector."""
        import json
        db = await get_db()

        # Ensure embedding column exists with correct dimensions
        if embeddings:
            embedding_dim = len(embeddings[0])
            success = await self.ensure_embedding_column(embedding_dim)
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
            await db.execute_raw(f"""
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
            """)
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
                    # Mismatch detected - require full reindex
                    if not full_reindex:
                        job.status = FilesystemIndexStatus.FAILED
                        job.completed_at = datetime.utcnow()
                        job.error_message = (
                            f"Embedding configuration mismatch: indexes were created with "
                            f"'{settings.embedding_config_hash}' but current config is "
                            f"'{current_config_hash}'. A full re-index is required. "
                            "Use the 'Full Re-index' button to rebuild all embeddings."
                        )
                        await self._update_job(job)
                        logger.error(job.error_message)
                        return
                    logger.warning(
                        f"Full re-index with config change: {settings.embedding_config_hash} -> {current_config_hash}"
                    )
                    tracking_needs_update = True

            # Initialize embeddings
            embeddings = await self._get_embeddings(app_settings)

            # Collect files to process - run in thread to avoid blocking on network filesystems
            logger.info(f"Collecting files from {config.base_path}...")
            files = await asyncio.to_thread(self._collect_files, config)
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

            base_path = Path(config.base_path)

            # Process files
            for file_path in files:
                # Check for cancellation at start of each file
                if self._is_cancelled(job.id):
                    logger.info(f"Job {job.id} cancelled by user")
                    job.status = FilesystemIndexStatus.CANCELLED
                    job.error_message = "Cancelled by user"
                    job.completed_at = datetime.utcnow()
                    await self._update_job(job)
                    return

                rel_path = str(file_path.relative_to(base_path))

                try:
                    # Check if file changed (incremental indexing) - run in thread for network filesystems
                    current_hash = await asyncio.to_thread(self._compute_file_hash, file_path)
                    existing_meta = await self._get_file_metadata(
                        config.index_name, rel_path
                    )

                    if not full_reindex and existing_meta and existing_meta.file_hash == current_hash:
                        # File unchanged, skip
                        job.skipped_files += 1
                        # Update progress every 10 skipped files (they're fast)
                        if job.skipped_files % 10 == 0:
                            await self._update_job(job)
                            await asyncio.sleep(0)  # Yield on skip updates too
                        continue

                    # Load and chunk the file
                    chunks = await self._load_and_chunk_file(file_path, config)
                    if not chunks:
                        # No content extracted - count as skipped (empty/unparseable file)
                        job.skipped_files += 1
                        await self._update_job(job)
                        await asyncio.sleep(0)
                        continue

                    # Delete old embeddings for this file
                    await self._delete_file_embeddings(config.index_name, rel_path)

                    # Generate embeddings - run in thread to avoid blocking event loop
                    chunk_embeddings = await asyncio.to_thread(
                        embeddings.embed_documents, chunks
                    )

                    # Validate embedding dimension against stored tracking
                    if chunk_embeddings:
                        current_dim = len(chunk_embeddings[0])
                        if (
                            settings.embedding_dimension is not None
                            and settings.embedding_dimension != current_dim
                        ):
                            if not full_reindex:
                                job.status = FilesystemIndexStatus.FAILED
                                job.error_message = (
                                    "Embedding dimension changed. A full re-index is required to "
                                    "rebuild embeddings with the new provider/model."
                                )
                                job.completed_at = datetime.utcnow()
                                await self._update_job(job)
                                logger.error(job.error_message)
                                return
                            logger.warning(
                                f"Embedding dimension changed {settings.embedding_dimension} -> {current_dim}; "
                                "continuing full re-index"
                            )
                            settings.embedding_dimension = current_dim
                            settings.embedding_config_hash = current_config_hash
                            tracking_needs_update = True

                    # Yield to event loop to keep server responsive
                    await asyncio.sleep(0)

                    # Insert new embeddings
                    inserted = await self._insert_embeddings(
                        index_name=config.index_name,
                        file_path=rel_path,
                        chunks=chunks,
                        embeddings=chunk_embeddings,
                        metadata={"source": rel_path},
                    )

                    # After first successful insert, record embedding dimension and config hash
                    if inserted > 0 and chunk_embeddings and tracking_needs_update:
                        await repo.update_embedding_tracking(
                            dimension=len(chunk_embeddings[0]),
                            config_hash=current_config_hash,
                        )
                        tracking_needs_update = False
                        settings.embedding_dimension = len(chunk_embeddings[0])
                        settings.embedding_config_hash = current_config_hash

                    # Update file metadata
                    file_stat = file_path.stat()
                    await self._upsert_file_metadata(FilesystemFileMetadata(
                        index_name=config.index_name,
                        file_path=rel_path,
                        file_hash=current_hash,
                        file_size=file_stat.st_size,
                        mime_type=mimetypes.guess_type(str(file_path))[0],
                        chunk_count=len(chunks),
                        last_indexed=datetime.utcnow(),
                    ))

                    job.processed_files += 1
                    job.processed_chunks += len(chunks)
                    job.total_chunks += len(chunks)

                    # Update job progress in database after each file
                    await self._update_job(job)

                    # Yield to event loop after each file to keep server responsive
                    await asyncio.sleep(0)

                    if job.processed_files % 10 == 0:
                        logger.info(
                            f"Processed {job.processed_files}/{job.total_files} files, "
                            f"skipped {job.skipped_files} unchanged"
                        )

                except Exception as e:
                    logger.warning(f"Error processing file {rel_path}: {e}")
                    job.processed_files += 1
                    # Yield even on errors
                    await asyncio.sleep(0)
                    continue

            # Update tool config with last indexed timestamp
            await self._update_tool_config_last_indexed(job.tool_config_id)

            job.status = FilesystemIndexStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            logger.info(
                f"Indexing completed: {job.processed_files} processed, "
                f"{job.skipped_files} skipped, {job.total_chunks} chunks"
            )

        except Exception as e:
            logger.exception(f"Filesystem indexing failed: {e}")
            job.status = FilesystemIndexStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()

        finally:
            await self._update_job(job)
            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)  # Clean up cancellation flag

    async def _update_tool_config_last_indexed(self, tool_config_id: str) -> None:
        """Update the tool config with last indexed timestamp."""
        try:
            import json
            from prisma import Json
            db = await get_db()
            tool_config = await db.toolconfig.find_unique(where={"id": tool_config_id})
            if tool_config:
                # connectionConfig is stored as JSON string, parse it
                if isinstance(tool_config.connectionConfig, str):
                    connection_config = json.loads(tool_config.connectionConfig)
                else:
                    connection_config = dict(tool_config.connectionConfig)  # type: ignore
                connection_config["last_indexed_at"] = datetime.utcnow().isoformat()
                await db.toolconfig.update(
                    where={"id": tool_config_id},
                    data={"connectionConfig": Json(connection_config)}
                )
        except Exception as e:
            logger.warning(f"Failed to update last_indexed_at: {e}")

    async def _get_embeddings(self, app_settings: dict):
        """Get the configured embedding model based on app settings."""
        provider = app_settings.get("embedding_provider", "ollama").lower()
        model = app_settings.get("embedding_model", "nomic-embed-text")
        dimensions = app_settings.get("embedding_dimensions")

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=model,
                base_url=app_settings.get("ollama_base_url", "http://localhost:11434"),
            )
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = app_settings.get("openai_api_key", "")
            if not api_key:
                raise ValueError(
                    "OpenAI embeddings selected but no API key configured in Settings"
                )
            # Pass dimensions for text-embedding-3-* models (supports MRL)
            kwargs = {"model": model, "api_key": api_key}
            if dimensions and model.startswith("text-embedding-3"):
                kwargs["dimensions"] = dimensions
                logger.info(f"Using OpenAI embeddings with {dimensions} dimensions")
            return OpenAIEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    async def _load_and_chunk_file(
        self,
        file_path: Path,
        config: FilesystemConnectionConfig,
    ) -> List[str]:
        """Load a file and split it into chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        try:
            # Read file content
            content = await asyncio.to_thread(self._read_file_content, file_path)
            if not content:
                return []

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )

            chunks = splitter.split_text(content)
            return chunks

        except Exception as e:
            logger.warning(f"Error loading file {file_path}: {e}")
            return []

    def _read_file_content(self, file_path: Path) -> str:
        """Read file content, handling various encodings and document formats."""
        from ragtime.indexer.document_parser import extract_text_from_file, DOCUMENT_EXTENSIONS

        suffix = file_path.suffix.lower()

        # Use document parser for Office/PDF files
        if suffix in {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".odt", ".ods", ".odp"}:
            return extract_text_from_file(file_path)

        # Plain text files - try various encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path} with {encoding}: {e}")
                return ""

        logger.warning(f"Could not decode file {file_path} with any encoding")
        return ""

    async def delete_index(self, index_name: str) -> int:
        """
        Delete all embeddings and metadata for an index.

        Returns the number of embeddings deleted.
        """
        db = await get_db()

        # Delete embeddings
        result = await db.filesystemembedding.delete_many(
            where={"indexName": index_name}
        )

        # Delete file metadata
        await db.filesystemfilemetadata.delete_many(
            where={"indexName": index_name}
        )

        logger.info(f"Deleted index '{index_name}': {result} embeddings removed")
        return result

    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics for a filesystem index."""
        db = await get_db()

        # Count embeddings
        embedding_count = await db.filesystemembedding.count(
            where={"indexName": index_name}
        )

        # Count unique files
        file_count = await db.filesystemfilemetadata.count(
            where={"indexName": index_name}
        )

        # Get latest indexed timestamp
        latest_file = await db.filesystemfilemetadata.find_first(
            where={"indexName": index_name},
            order={"lastIndexed": "desc"},
        )

        return {
            "index_name": index_name,
            "embedding_count": embedding_count,
            "file_count": file_count,
            "last_indexed": latest_file.lastIndexed if latest_file else None,
        }


# Global service instance
filesystem_indexer = FilesystemIndexerService()
