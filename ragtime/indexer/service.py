"""
FAISS Indexer Service - Creates and manages vector indexes.

Job Recovery: Jobs are persisted to the database before processing starts.
If the server restarts mid-job (e.g., hot-reload), jobs in 'pending' or
'processing' state are automatically resumed on startup.
"""

import asyncio
import fnmatch
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, BinaryIO
import uuid

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.indexer.models import IndexJob, IndexStatus, IndexConfig, IndexInfo, AppSettings
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

# Persistent storage for uploaded files (survives restarts)
UPLOAD_STAGING_DIR = Path(settings.index_data_path) / "staging"


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
        from ragtime.core.app_settings import get_app_settings
        from langchain_openai import ChatOpenAI

        app_settings = await get_app_settings()
        api_key = app_settings.get("openai_api_key", "")

        if not api_key:
            logger.debug("No OpenAI API key - skipping auto-description generation")
            return ""

        # Sample file paths for context (unique paths only)
        file_paths = list(set(
            doc.metadata.get("source", "unknown")
            for doc in documents[:100]
        ))[:20]

        # Sample some content snippets
        content_samples = [
            doc.page_content[:500] for doc in documents[:5]
        ]

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
            openai_api_key=api_key,
            max_tokens=150,
        )

        response = await llm.ainvoke(prompt)
        description = response.content.strip()
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
        # Ensure staging directory exists for upload persistence
        UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
        # In-memory cache for active jobs (transient state during processing)
        self._active_jobs: Dict[str, IndexJob] = {}

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
            j for j in jobs
            if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)
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

        # Clean up orphaned staging directories (from deleted jobs)
        await self._cleanup_orphaned_staging()

        return recovered

    async def _cleanup_orphaned_staging(self) -> None:
        """Remove staging directories for jobs that no longer exist."""
        if not UPLOAD_STAGING_DIR.exists():
            return

        jobs = await repository.list_jobs()
        active_job_ids = {j.id for j in jobs if j.status in (IndexStatus.PENDING, IndexStatus.PROCESSING)}

        for staging_path in UPLOAD_STAGING_DIR.iterdir():
            if staging_path.is_dir() and staging_path.name not in active_job_ids:
                logger.info(f"Cleaning orphaned staging directory: {staging_path.name}")
                shutil.rmtree(staging_path, ignore_errors=True)

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
            # Git jobs can be re-cloned
            asyncio.create_task(self._process_git(job))
        elif job.source_type == "upload":
            # Check if staged file still exists
            staged_path = UPLOAD_STAGING_DIR / job.id
            if staged_path.exists():
                # Find the archive file (not directories like 'extracted')
                archive_files = [f for f in staged_path.iterdir() if f.is_file()]
                if archive_files:
                    archive_path = archive_files[0]

                    # Clean any previous extraction attempt
                    extracted_dir = staged_path / "extracted"
                    if extracted_dir.exists():
                        shutil.rmtree(extracted_dir, ignore_errors=True)

                    temp_dir = staged_path  # Use staging as temp dir
                    asyncio.create_task(self._process_upload(job, archive_path, temp_dir))
                    return

            # No staged file - mark as failed
            job.status = IndexStatus.FAILED
            job.error_message = "Upload file lost during restart - please re-upload"
            job.completed_at = datetime.utcnow()
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

    async def _get_embeddings(self, app_settings: AppSettings):
        """Get the configured embedding model based on app settings."""
        provider = app_settings.embedding_provider.lower()
        model = app_settings.embedding_model

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model=model,
                base_url=app_settings.ollama_base_url,
            )
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not app_settings.openai_api_key:
                raise ValueError("OpenAI embeddings selected but no API key configured in Settings")
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=app_settings.openai_api_key,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}. Use 'ollama' or 'openai'.")

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

        # Remove from active jobs cache (will stop processing on next iteration)
        self._active_jobs.pop(job_id, None)

        # Update status in database
        job.status = IndexStatus.FAILED
        job.error_message = "Job cancelled by user"
        job.completed_at = datetime.utcnow()
        await repository.update_job(job)

        logger.info(f"Cancelled job {job_id}")
        return True

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
                    size_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

                    # Try to get document count from database metadata first
                    doc_count = 0
                    created_at = None
                    enabled = True  # Default to enabled
                    description = ""

                    if path.name in metadata_by_name:
                        meta = metadata_by_name[path.name]
                        doc_count = meta.documentCount
                        created_at = meta.createdAt
                        enabled = meta.enabled
                        description = getattr(meta, "description", "")
                    else:
                        # Fallback to legacy .metadata.json file
                        meta_file = path / ".metadata.json"
                        if meta_file.exists():
                            try:
                                with open(meta_file) as f:
                                    meta = json.load(f)
                                    doc_count = meta.get("document_count", 0)
                                    created_at_str = meta.get("created_at")
                                    if created_at_str:
                                        created_at = datetime.fromisoformat(created_at_str)
                            except Exception:
                                pass

                    indexes.append(IndexInfo(
                        name=path.name,
                        path=str(path),
                        size_mb=round(size_bytes / (1024 * 1024), 2),
                        document_count=doc_count,
                        description=description,
                        enabled=enabled,
                        created_at=created_at,
                        last_modified=datetime.fromtimestamp(path.stat().st_mtime)
                    ))

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
        await repository.delete_index_metadata(name)

        return deleted_files

    async def create_index_from_upload(
        self,
        file: BinaryIO,
        filename: str,
        config: IndexConfig
    ) -> IndexJob:
        """Create an index from an uploaded archive file.

        The uploaded file is stored in a persistent staging directory so it
        survives server restarts and can be resumed if interrupted.
        """
        job_id = str(uuid.uuid4())[:8]

        job = IndexJob(
            id=job_id,
            name=config.name,
            config=config,
            source_type="upload",
            source_path=filename
        )

        # Persist to database FIRST (before any processing)
        await repository.create_job(job)

        # Cache for active processing
        self._active_jobs[job_id] = job

        # Save uploaded file to PERSISTENT staging location (survives restarts)
        staging_dir = UPLOAD_STAGING_DIR / job_id
        staging_dir.mkdir(parents=True, exist_ok=True)
        archive_path = staging_dir / filename

        try:
            with open(archive_path, "wb") as f:
                shutil.copyfileobj(file, f)

            logger.info(f"Staged upload for job {job_id}: {archive_path}")

            # Start processing in background
            asyncio.create_task(self._process_upload(job, archive_path, staging_dir))

        except Exception as e:
            job.status = IndexStatus.FAILED
            job.error_message = str(e)
            await repository.update_job(job)
            self._active_jobs.pop(job_id, None)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        return job

    async def create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig
    ) -> IndexJob:
        """Create an index from a git repository."""
        job_id = str(uuid.uuid4())[:8]

        job = IndexJob(
            id=job_id,
            name=config.name,
            config=config,
            source_type="git",
            git_url=git_url,
            git_branch=branch
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

            # Extract archive
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()

            logger.info(f"Extracting {archive_path} to {extract_dir}")
            self._extract_archive(archive_path, extract_dir)

            # Find the actual source directory (handle nested zips)
            source_dir = self._find_source_dir(extract_dir)

            # Create the index
            await self._create_faiss_index(job, source_dir)

            job.status = IndexStatus.COMPLETED
            job.completed_at = datetime.utcnow()

        except Exception as e:
            logger.exception(f"Failed to process upload for job {job.id}")
            job.status = IndexStatus.FAILED
            job.error_message = str(e)

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

    async def _process_git(self, job: IndexJob):
        """Process a git repository."""
        temp_dir = Path(tempfile.mkdtemp())

        try:
            job.status = IndexStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await repository.update_job(job)

            # Clone repository
            clone_dir = temp_dir / "repo"
            logger.info(f"Cloning {job.git_url} branch {job.git_branch}")

            process = await asyncio.create_subprocess_exec(
                "git", "clone",
                "--depth", "1",
                "--branch", job.git_branch,
                job.git_url,
                str(clone_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"Git clone failed: {stderr.decode()}")

            # Create the index
            await self._create_faiss_index(job, clone_dir)

            job.status = IndexStatus.COMPLETED
            job.completed_at = datetime.utcnow()

        except Exception as e:
            logger.exception(f"Failed to process git for job {job.id}")
            job.status = IndexStatus.FAILED
            job.error_message = str(e)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            await repository.update_job(job)
            self._active_jobs.pop(job.id, None)

    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract an archive file (zip, tar, tar.gz, tar.bz2) to a directory."""
        filename_lower = archive_path.name.lower()

        if filename_lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif filename_lower.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(extract_dir, filter='data')
        elif filename_lower.endswith((".tar.bz2", ".tbz2")):
            with tarfile.open(archive_path, 'r:bz2') as tf:
                tf.extractall(extract_dir, filter='data')
        elif filename_lower.endswith(".tar"):
            with tarfile.open(archive_path, 'r:') as tf:
                tf.extractall(extract_dir, filter='data')
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

    async def _create_faiss_index(self, job: IndexJob, source_dir: Path):
        """Create FAISS index from source directory."""
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        config = job.config

        # Collect all files to process
        all_files = []
        for pattern in config.file_patterns:
            # Use removeprefix instead of lstrip to avoid stripping too many characters
            # lstrip("**/") on "**/*.py" incorrectly gives ".py", removeprefix gives "*.py"
            glob_pattern = pattern.removeprefix("**/").removeprefix("*/")
            logger.debug(f"Globbing with pattern: {glob_pattern} (original: {pattern})")
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

                    if not skip and file_path not in all_files:
                        all_files.append(file_path)

        job.total_files = len(all_files)
        await repository.update_job(job)
        logger.info(f"Found {len(all_files)} files to index")

        # Load documents
        documents = []
        for file_path in all_files:
            try:
                loader = TextLoader(str(file_path), autodetect_encoding=True)
                docs = loader.load()

                # Add source metadata
                rel_path = str(file_path.relative_to(source_dir))
                for doc in docs:
                    doc.metadata["source"] = rel_path
                    doc.metadata["index_name"] = job.name

                documents.extend(docs)
                job.processed_files += 1

                # Update progress periodically
                if job.processed_files % 50 == 0:
                    await repository.update_job(job)
                    logger.info(f"Loaded {job.processed_files}/{job.total_files} files")

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not documents:
            raise Exception("No documents were loaded")

        logger.info(f"Loaded {len(documents)} documents, splitting into chunks...")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        job.total_chunks = len(chunks)
        await repository.update_job(job)

        logger.info(f"Created {len(chunks)} chunks, generating embeddings...")

        # Get embedding settings from app settings
        app_settings = await repository.get_settings()
        embeddings = await self._get_embeddings(app_settings)

        logger.info(f"Using {app_settings.embedding_provider} embeddings with model: {app_settings.embedding_model}")

        # Process in batches to show progress
        batch_size = 100
        db = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            if db is None:
                db = await asyncio.to_thread(FAISS.from_documents, batch, embeddings)
            else:
                batch_db = await asyncio.to_thread(FAISS.from_documents, batch, embeddings)
                db.merge_from(batch_db)

            # Update processed chunks progress
            job.processed_chunks = min(i + batch_size, len(chunks))
            await repository.update_job(job)
            logger.info(f"Embedded {job.processed_chunks}/{len(chunks)} chunks")

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
        )

        logger.info(f"Index {job.name} created successfully!")


# Global indexer instance - uses configured path
indexer = IndexerService(index_base_path=settings.index_data_path)
