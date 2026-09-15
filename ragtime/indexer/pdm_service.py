"""
SolidWorks PDM Indexer Service - Creates and manages pgvector-based PDM indexes.

This service handles:
- Connecting to PDM SQL Server database
- Extracting document metadata and variables
- Building structured text for each document
- Storing embeddings in PostgreSQL using pgvector
- Progress tracking and job management
- Incremental indexing (skip unchanged documents based on metadata hash)

PDM Database Structure:
- Documents: Main document table (DocumentID, Filename, etc.)
- DocumentsInProjects: Junction table linking documents to folders (ProjectID, DocumentID)
- VariableValue: Variable values per document/configuration
- Variable: Variable name definitions (ID -> Name mapping)
- Projects: Folder/project information (ProjectID, Name, Path, etc.)
- BomSheets: BOM relationships
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from ragtime.core.app_setting_defaults import DEFAULT_IVFFLAT_LISTS
from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.indexer.embedding_errors import EmbeddingOperationError, build_embedding_configuration_error
from ragtime.indexer.models import (
    PdmBomComponentModel,
    PdmConfigurationStateModel,
    PdmDocumentStateModel,
    PdmIndexJob,
    PdmIndexJobResponse,
    PdmIndexStatus,
    PdmPropertyValueModel,
    SolidworksPdmConnectionConfig,
)
from ragtime.indexer.pdm_source import (
    PdmDocumentRecord,
    PdmSqlSource,
    PdmVariableDef,
)
from ragtime.indexer.pdm_source import (
    build_pdm_extension_filter as _build_pdm_extension_filter,
)
from ragtime.indexer.repository import repository
from ragtime.indexer.vector_utils import (
    EMBEDDING_SUB_BATCH_SIZE,
    PDM_COLUMNS,
    embed_documents_subbatched,
    ensure_embedding_column,
    ensure_pgvector_extension,
    get_embeddings_model,
    search_pgvector_embeddings,
)

# pylint: disable=not-callable

logger = get_logger(__name__)


_ALLOWED_PDM_DOCUMENT_TYPES = frozenset({"SLDPRT", "SLDASM", "SLDDRW"})
_PDM_ROLE_DEFAULT_NAMES = {"part_number": "Part Number", "description": "Description"}


class PdmIndexerService:
    """Service for creating and managing SolidWorks PDM indexes with pgvector."""

    def __init__(self) -> None:
        self._active_jobs: Dict[str, PdmIndexJob] = {}
        self._cancellation_flags: Dict[str, bool] = {}  # job_id -> should_cancel
        self._running_tasks: Dict[str, asyncio.Task] = {}  # job_id -> task
        self._shutdown = False
        self._variable_defs: list[PdmVariableDef] = []
        self._record_states: dict[int, PdmDocumentStateModel] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    async def trigger_index(
        self,
        tool_config_id: str,
        connection_config: dict,
        full_reindex: bool = False,
        tool_name: str | None = None,
    ) -> PdmIndexJob:
        """
        Trigger PDM metadata indexing for a tool config.

        Args:
            tool_config_id: The tool configuration ID
            connection_config: PDM connection configuration
            full_reindex: If True, re-index all documents regardless of hash
            tool_name: Safe tool name for display (if None, uses tool_config_id)

        Returns:
            The created PdmIndexJob
        """
        # Check for existing active job
        existing_job = await self.get_active_job(tool_config_id)
        if existing_job and existing_job.status in (
            PdmIndexStatus.PENDING,
            PdmIndexStatus.INDEXING,
        ):
            logger.info(f"PDM index job already running for tool {tool_config_id}")
            return existing_job

        # Create job
        job_id = str(uuid.uuid4())
        safe_name = tool_name or tool_config_id
        index_name = f"pdm_{safe_name}"

        job = PdmIndexJob(
            id=job_id,
            tool_config_id=tool_config_id,
            status=PdmIndexStatus.PENDING,
            index_name=index_name,
            created_at=datetime.now(timezone.utc),
        )

        # Store job in database
        await self._create_job(job)
        self._active_jobs[job_id] = job
        self._cancellation_flags[job_id] = False

        # Start background processing
        task = asyncio.create_task(self._process_index(job, connection_config, full_reindex))
        self._running_tasks[job_id] = task

        logger.info(f"Started PDM indexing job {job_id} for tool {tool_config_id}")
        return job

    async def get_job_status(self, job_id: str) -> Optional[PdmIndexJobResponse]:
        """Get the current status of a PDM indexing job."""
        # Check in-memory first
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            return PdmIndexJobResponse(
                id=job.id,
                tool_config_id=job.tool_config_id,
                status=job.status,
                index_name=job.index_name,
                current_step=job.current_step,
                progress_percent=job.progress_percent,
                total_documents=job.total_documents,
                processed_documents=job.processed_documents,
                skipped_documents=job.skipped_documents,
                extracted_documents=job.extracted_documents,
                total_chunks=job.total_chunks,
                processed_chunks=job.processed_chunks,
                error_message=job.error_message,
                cancel_requested=job.cancel_requested,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
            )

        # Fall back to database
        return await self._get_job_from_db(job_id)

    async def get_active_job(self, tool_config_id: str) -> Optional[PdmIndexJob]:
        """Get any active job for a tool config."""
        # Check in-memory first
        for job in self._active_jobs.values():
            if job.tool_config_id == tool_config_id:
                return job

        # Check database for pending/indexing jobs
        try:
            db: Any = await get_db()
            prisma_job = await db.pdmindexjob.find_first(
                where={
                    "toolConfigId": tool_config_id,
                    "status": {"in": ["pending", "indexing"]},
                },
                order={"createdAt": "desc"},
            )
            if prisma_job:
                return self._prisma_job_to_model(prisma_job)
        except Exception as e:
            logger.warning(f"Error checking for active PDM job: {e}")

        return None

    async def get_latest_job(self, tool_config_id: str) -> Optional[PdmIndexJobResponse]:
        """Get the most recent job for a tool config."""
        try:
            db: Any = await get_db()
            prisma_job = await db.pdmindexjob.find_first(
                where={"toolConfigId": tool_config_id},
                order={"createdAt": "desc"},
            )
            if prisma_job:
                job = self._prisma_job_to_model(prisma_job)
                return PdmIndexJobResponse(
                    id=job.id,
                    tool_config_id=job.tool_config_id,
                    status=job.status,
                    index_name=job.index_name,
                    current_step=job.current_step,
                    progress_percent=job.progress_percent,
                    total_documents=job.total_documents,
                    processed_documents=job.processed_documents,
                    skipped_documents=job.skipped_documents,
                    extracted_documents=job.extracted_documents,
                    total_chunks=job.total_chunks,
                    processed_chunks=job.processed_chunks,
                    error_message=job.error_message,
                    cancel_requested=False,  # Not tracked in DB
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
        except Exception as e:
            logger.warning(f"Error getting latest PDM job: {e}")

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of an active job."""
        # Check if job is active in memory
        if job_id in self._cancellation_flags:
            self._cancellation_flags[job_id] = True
            if job_id in self._active_jobs:
                self._active_jobs[job_id].cancel_requested = True
            logger.info(f"Cancellation requested for PDM job {job_id}")
            return True

        # Check database for orphaned jobs (running in DB but not in memory)
        db: Any = await get_db()
        prisma_job = await db.pdmindexjob.find_unique(where={"id": job_id})
        if not prisma_job:
            return False

        # Can only cancel pending/indexing jobs
        if prisma_job.status in ("pending", "indexing"):
            # Job is not running in memory but stuck in DB - directly mark as cancelled
            logger.info(f"Directly cancelling orphaned PDM job {job_id} (not in active jobs)")
            await db.pdmindexjob.update(
                where={"id": job_id},
                data={
                    "status": "cancelled",
                    "currentStep": "Cancelled (orphaned)",
                    "errorMessage": "Job cancelled (was orphaned after restart)",
                    "completedAt": datetime.now(timezone.utc),
                },
            )
            return True

        return False

    async def _cleanup_stale_jobs(self) -> int:
        """
        Clean up any jobs left in pending/indexing state from a previous run.

        This handles cases where the server was restarted while indexing was
        in progress. Those jobs will never complete, so mark them as failed.

        Returns the number of orphaned jobs cleaned up.
        """
        try:
            db: Any = await get_db()

            # Find all jobs stuck in pending or indexing state
            # Note: Column names use snake_case as defined in Prisma @map() directives
            result = await db.execute_raw(
                """
                UPDATE pdm_index_jobs
                SET status = 'failed',
                    current_step = 'Failed (server restart)',
                    error_message = 'Job interrupted by server restart',
                    completed_at = NOW()
                WHERE status IN ('pending', 'indexing')
                RETURNING id
                """
            )

            # result is the count of updated rows
            count = result if isinstance(result, int) else 0
            if count > 0:
                logger.info(f"Cleaned up {count} orphaned PDM indexing job(s)")
            return count

        except Exception as e:
            logger.warning(f"Failed to clean up orphaned PDM jobs: {e}")
            return 0

    async def shutdown(self) -> None:
        """Shutdown the service and cancel all running tasks."""
        logger.info("PDM indexer service shutting down")
        self._shutdown = True

        # Cancel all running tasks
        for job_id, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Cancelled PDM indexing task {job_id}")

        self._running_tasks.clear()
        self._active_jobs.clear()
        self._cancellation_flags.clear()

    async def list_all_jobs(self, limit: int = 50) -> List[PdmIndexJobResponse]:
        """
        List all PDM indexing jobs across all tools.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of PDM job responses, sorted by created_at desc
        """
        jobs: List[PdmIndexJobResponse] = []

        try:
            db: Any = await get_db()

            # Get jobs from database
            db_jobs = await db.pdmindexjob.find_many(
                take=limit,
                order={"createdAt": "desc"},
            )

            for db_job in db_jobs:
                # Check if this job is still active in memory (fresher data)
                if db_job.id in self._active_jobs:
                    job = self._active_jobs[db_job.id]
                    jobs.append(
                        PdmIndexJobResponse(
                            id=job.id,
                            tool_config_id=job.tool_config_id,
                            status=job.status,
                            index_name=job.index_name,
                            current_step=job.current_step,
                            progress_percent=job.progress_percent,
                            total_documents=job.total_documents,
                            processed_documents=job.processed_documents,
                            skipped_documents=job.skipped_documents,
                            extracted_documents=job.extracted_documents,
                            total_chunks=job.total_chunks,
                            processed_chunks=job.processed_chunks,
                            error_message=job.error_message,
                            cancel_requested=job.cancel_requested,
                            created_at=job.created_at,
                            started_at=job.started_at,
                            completed_at=job.completed_at,
                        )
                    )
                else:
                    # Use database data
                    job = self._prisma_job_to_model(db_job)
                    jobs.append(
                        PdmIndexJobResponse(
                            id=job.id,
                            tool_config_id=job.tool_config_id,
                            status=job.status,
                            index_name=job.index_name,
                            current_step=job.current_step,
                            progress_percent=job.progress_percent,
                            total_documents=job.total_documents,
                            processed_documents=job.processed_documents,
                            skipped_documents=job.skipped_documents,
                            extracted_documents=job.extracted_documents,
                            total_chunks=job.total_chunks,
                            processed_chunks=job.processed_chunks,
                            error_message=job.error_message,
                            cancel_requested=False,
                            created_at=job.created_at,
                            started_at=job.started_at,
                            completed_at=job.completed_at,
                        )
                    )

        except Exception as e:
            logger.warning(f"Error listing PDM jobs: {e}")

        return jobs

    async def retry_job(self, job_id: str) -> Optional[PdmIndexJob]:
        """Retry a failed or cancelled PDM indexing job."""
        try:
            db: Any = await get_db()

            # Get the original job
            db_job = await db.pdmindexjob.find_unique(where={"id": job_id})
            if not db_job:
                logger.warning(f"Cannot retry: job {job_id} not found")
                return None

            job_status = str(db_job.status)

            # Only allow retry for failed or cancelled jobs
            if job_status not in ("failed", "cancelled"):
                logger.warning(f"Cannot retry: job {job_id} has status {job_status}")
                return None

            # Get the tool config to get connection details
            tool_config = await repository.get_tool_config(db_job.toolConfigId)
            if not tool_config or not tool_config.id:
                logger.warning(f"Cannot retry: tool config {db_job.toolConfigId} not found")
                return None

            # Trigger a new indexing job
            return await self.trigger_index(
                tool_config_id=tool_config.id,
                connection_config=tool_config.connection_config or {},
                full_reindex=True,  # Force full reindex on retry
            )

        except Exception as e:
            logger.error(f"Error retrying PDM job {job_id}: {e}")
            return None

    async def delete_index(self, tool_config_id: str) -> Tuple[bool, str]:
        """Delete all PDM embeddings for a tool config."""
        try:
            db: Any = await get_db()

            # Find index name (try both conventions)
            # First try tool name pattern
            tool_config = await repository.get_tool_config(tool_config_id)
            index_names = [f"pdm_{tool_config_id}"]
            if tool_config:
                index_names.insert(0, f"pdm_{tool_config.name}")

            deleted_embeddings = 0
            deleted_metadata = 0
            deleted_state = 0

            for index_name in index_names:
                # Delete embeddings
                result = await db.execute_raw(
                    "DELETE FROM pdm_embeddings WHERE index_name = $1",
                    index_name,
                )
                if isinstance(result, int):
                    deleted_embeddings += result

                # Delete document metadata
                result = await db.execute_raw(
                    "DELETE FROM pdm_document_metadata WHERE index_name = $1",
                    index_name,
                )
                if isinstance(result, int):
                    deleted_metadata += result

                result = await db.execute_raw(
                    "DELETE FROM pdm_document_state WHERE index_name = $1",
                    index_name,
                )
                if isinstance(result, int):
                    deleted_state += result

            # Delete jobs
            await db.pdmindexjob.delete_many(where={"toolConfigId": tool_config_id})

            logger.info(
                f"Deleted PDM index for tool {tool_config_id}: {deleted_embeddings} embeddings, {deleted_metadata} metadata records, {deleted_state} state records"
            )
            return True, f"Deleted {deleted_embeddings} PDM embeddings"

        except Exception as e:
            logger.error(f"Error deleting PDM index: {e}")
            return False, str(e)

    async def get_embedding_count(self, tool_config_id: str, tool_name: str | None = None) -> int:
        """Get the number of PDM embeddings for a tool."""
        names_to_check = []
        if tool_name:
            names_to_check.append(f"pdm_{tool_name}")
        names_to_check.append(f"pdm_{tool_config_id}")

        try:
            db: Any = await get_db()
            for index_name in names_to_check:
                result = await db.query_raw(
                    "SELECT COUNT(*) as count FROM pdm_embeddings WHERE index_name = $1",
                    index_name,
                )
                if result and int(result[0].get("count", 0)) > 0:
                    return int(result[0].get("count", 0))
        except Exception as e:
            logger.warning(f"Error getting PDM embedding count: {e}")
        return 0

    async def get_document_count(self, tool_config_id: str, tool_name: str | None = None) -> int:
        """Get the number of indexed PDM documents for a tool."""
        names_to_check = []
        if tool_name:
            names_to_check.append(f"pdm_{tool_name}")
        names_to_check.append(f"pdm_{tool_config_id}")

        try:
            db: Any = await get_db()
            for index_name in names_to_check:
                result = await db.query_raw(
                    "SELECT COUNT(*) as count FROM pdm_document_metadata WHERE index_name = $1",
                    index_name,
                )
                if result and int(result[0].get("count", 0)) > 0:
                    return int(result[0].get("count", 0))
        except Exception as e:
            logger.warning(f"Error getting PDM document count: {e}")
        return 0

    # =========================================================================
    # PDM Data Extraction
    # =========================================================================

    async def extract_documents_batched(
        self,
        config: SolidworksPdmConnectionConfig,
        variable_map: Dict[int, str],
        max_documents: int | None = None,
        batch_size: int = 1000,
        on_batch_extracted: Callable[[int, int], None] | None = None,
    ) -> AsyncIterator[List[PdmDocumentRecord]]:
        """Delegate batched checked-in extraction to the PDM SQL source adapter."""
        async for batch in PdmSqlSource(config).extract_documents_batched(
            variable_map,
            max_documents,
            batch_size,
            on_batch_extracted,
        ):
            yield batch

    async def extract_documents(
        self,
        config: SolidworksPdmConnectionConfig,
        max_documents: int | None = None,
    ) -> AsyncIterator[PdmDocumentRecord]:
        """Extract documents with metadata from PDM database.

        This is a compatibility wrapper around extract_documents_batched
        that yields individual documents instead of batches.
        """
        variable_map = await self._get_variable_map(config)

        async for batch in self.extract_documents_batched(
            config=config,
            variable_map=variable_map,
            max_documents=max_documents,
            batch_size=500,
        ):
            for doc in batch:
                yield doc

    async def _get_variable_map(self, config: SolidworksPdmConnectionConfig) -> Dict[int, str]:
        """Discover configured and role-default variables through the adapter."""
        variable_defs = await PdmSqlSource(config).discover_variables()
        self._variable_defs = variable_defs
        names = {str(name).casefold() for name in (config.variable_names or [])}
        names.update({default_name.casefold() for default_name in _PDM_ROLE_DEFAULT_NAMES.values()})
        return {definition.variable_id: definition.name for definition in variable_defs if definition.name.casefold() in names}

    @staticmethod
    def _resolve_role_variables(
        variable_defs: list[PdmVariableDef],
        config: SolidworksPdmConnectionConfig,
    ) -> dict[str, Optional[int]]:
        """Resolve role IDs by configured/default names, never vault-specific IDs.

        Priority for each role:
        1. Explicit override in config (part_number_variable/description_variable)
        2. Default name match ("Part Number"/"Description", case-insensitive)
        3. None if no match found

        Variable_defs are sorted by ID for deterministic resolution.
        """
        resolved: dict[str, Optional[int]] = {}
        # Sort defs by variable_id for determinism when multiple defs have the same name
        sorted_defs = sorted(variable_defs, key=lambda d: d.variable_id)

        for role, default_name in _PDM_ROLE_DEFAULT_NAMES.items():
            # Check for explicit override attribute (e.g., part_number_variable)
            override_attr = f"{role}_variable"
            override_value = getattr(config, override_attr, None)

            if override_value:
                # Explicit override: find matching def
                match = next(
                    (d for d in sorted_defs if d.name.casefold() == override_value.casefold()),
                    None,
                )
                resolved[role] = match.variable_id if match else None
            else:
                # Default match: find def with default name
                match = next(
                    (d for d in sorted_defs if d.name.casefold() == default_name.casefold()),
                    None,
                )
                resolved[role] = match.variable_id if match else None

        return resolved

    @staticmethod
    def _state_from_record(
        record: PdmDocumentRecord,
        role_map: dict[str, Optional[int]],
        variable_map: Dict[int, str],
    ) -> PdmDocumentStateModel:
        """Convert a resolved adapter record to its persisted checked-in state."""
        configurations_by_id = {item.configuration_id: item for item in record.configurations}
        document_values: dict[str, PdmPropertyValueModel] = {}
        configuration_values: dict[int, dict[str, PdmPropertyValueModel]] = {}
        for (configuration_id, variable_id), raw in record.resolved_values.items():
            value = PdmPropertyValueModel(
                variable_id=variable_id,
                variable_name=variable_map.get(variable_id, raw.variable_name),
                value_text=raw.value_text,
                is_blank=raw.value_text == "",
                origin_revision=raw.revision_no,
                project_scope_id=raw.project_id,
                value_int=raw.value_int,
                value_float=raw.value_float,
                value_date=raw.value_date,
            )
            ref = configurations_by_id.get(configuration_id)
            if ref and ref.name == "@":
                document_values[value.variable_name] = value
            elif ref:
                configuration_values.setdefault(configuration_id, {})[value.variable_name] = value
            else:
                # Configuration ID has no membership reference (dropped value)
                logger.debug(
                    f"Dropped value for document {record.document_id}: configuration_id={configuration_id}, variable={value.variable_name} ({variable_id})"
                )

        configurations = [
            PdmConfigurationStateModel(
                configuration_id=ref.configuration_id,
                name=ref.name,
                values=configuration_values.get(ref.configuration_id, {}),
            )
            for ref in sorted(record.configurations, key=lambda item: item.configuration_id)
            if ref.name != "@"
        ]

        def role_value(role: str) -> Optional[str]:
            variable_id = role_map.get(role)
            role_name = "part number" if role == "part_number" else "description"
            document_value = next(
                (
                    value
                    for value in document_values.values()
                    if value.variable_id == variable_id or (variable_id is None and value.variable_name.casefold() == role_name)
                ),
                None,
            )
            if document_value and not document_value.is_blank:
                return document_value.value_text
            for configuration in configurations:
                value = next(
                    (
                        item
                        for item in configuration.values.values()
                        if item.variable_id == variable_id or (variable_id is None and item.variable_name.casefold() == role_name)
                    ),
                    None,
                )
                if value and not value.is_blank:
                    return value.value_text
            return None

        warnings: list[str] = []
        if record.membership_fallback:
            warnings.append("Configuration membership was derived from eligible property values.")
        if record.has_beyond_latest_values:
            warnings.append("Values beyond the checked-in target revision were excluded.")
        document_type = record.filename.rsplit(".", 1)[-1].upper() if "." in record.filename else "UNKNOWN"
        return PdmDocumentStateModel(
            document_id=record.document_id,
            filename=record.filename,
            document_type=document_type,
            target_revision=record.latest_revision,
            folder_paths=record.folder_paths,
            part_number=role_value("part_number"),
            description=role_value("description"),
            document_values=document_values,
            configurations=configurations,
            bom_components=[
                PdmBomComponentModel(
                    document_id=item.document_id,
                    filename=item.filename,
                    configuration=item.configuration,
                    quantity=None,
                )
                for item in record.bom_children
            ],
            membership_fallback=record.membership_fallback,
            has_beyond_latest_values=record.has_beyond_latest_values,
            warnings=warnings,
        )

    def _metadata_hash_for(
        self,
        item: Any,
        role_map: dict[str, Optional[int]],
        variable_map: Dict[int, str],
    ) -> str:
        """Hash real records from normalized state while retaining fake-doc seams."""
        if isinstance(item, PdmDocumentRecord):
            state = self._state_from_record(item, role_map, variable_map)
            self._record_states[id(item)] = state
            return state.compute_metadata_hash()
        return item.compute_metadata_hash()

    # =========================================================================
    # Background Processing
    # =========================================================================

    async def _process_index(
        self,
        job: PdmIndexJob,
        connection_config: dict,
        full_reindex: bool,
    ):
        """Main processing loop for PDM indexing with step-by-step progress tracking."""
        try:
            # Update job status
            job.status = PdmIndexStatus.INDEXING
            job.started_at = datetime.now(timezone.utc)
            job.current_step = "Initializing"
            await self._update_job(job)

            # Ensure pgvector is available
            if not await self._ensure_pgvector():
                raise RuntimeError("pgvector extension not available")

            # Get app settings for embedding configuration
            from ragtime.core.app_settings import get_app_settings

            job.current_step = "Checking embedding configuration"
            await self._update_job(job)

            app_settings = await get_app_settings()
            settings = await repository.get_settings()

            # Check for embedding configuration mismatch
            # Auto-correct by forcing full re-index when config changes
            current_config_hash = settings.get_embedding_config_hash()
            tracking_needs_update = settings.embedding_dimension is None or settings.embedding_config_hash is None

            if settings.embedding_config_hash is not None:
                if settings.embedding_config_hash != current_config_hash:
                    # Mismatch detected - auto-correct by treating as full reindex
                    logger.warning(
                        f"Embedding config changed: {settings.embedding_config_hash} -> {current_config_hash}. "
                        "Auto-triggering full re-index to correct dimension mismatch."
                    )
                    full_reindex = True  # Force full reindex to clear old embeddings
                    tracking_needs_update = True

            # Get embeddings provider
            job.current_step = "Initializing embedding provider"
            await self._update_job(job)

            embeddings = await self._get_embeddings(app_settings)
            if embeddings is None:
                raise RuntimeError("No embedding provider configured. Configure an embedding provider in Settings.")

            # Check embedding dimension and ensure column matches
            test_embedding = await embeddings.aembed_documents(["test"])
            embedding_dim = len(test_embedding[0])
            index_lists = app_settings.get("ivfflat_lists", DEFAULT_IVFFLAT_LISTS)
            # This will raise RuntimeError with detailed message if it fails
            await self._ensure_embedding_column(embedding_dim, index_lists)

            # Update tracking if needed
            if tracking_needs_update:
                await repository.update_settings(
                    {
                        "embedding_dimension": embedding_dim,
                        "embedding_config_hash": current_config_hash,
                    }
                )
                logger.info(f"Updated embedding tracking: dim={embedding_dim}, hash={current_config_hash}")

            # Parse connection config
            config = SolidworksPdmConnectionConfig(**connection_config)

            # Count documents first
            job.current_step = "Counting documents in PDM"
            await self._update_job(job)

            doc_count = await self._count_documents(config)
            job.total_documents = doc_count
            await self._update_job(job)

            if doc_count == 0:
                job.status = PdmIndexStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.current_step = "Completed"
                job.error_message = "No documents found matching criteria"
                await self._update_job(job)
                return

            logger.info(f"PDM indexing: Found {doc_count} documents to process")

            # Check for cancellation
            if self._cancellation_flags.get(job.id, False):
                job.status = PdmIndexStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                job.current_step = "Cancelled"
                await self._update_job(job)
                return

            # If full reindex, clear existing embeddings
            if full_reindex:
                job.current_step = "Clearing existing embeddings"
                await self._update_job(job)
                await self._clear_embeddings(job.index_name)

            # Get variable mapping
            job.current_step = "Loading PDM variable mappings"
            await self._update_job(job)
            variable_map = await self._get_variable_map(config)
            logger.info(f"PDM indexing: Loaded {len(variable_map)} variable mappings")

            # Process documents in batches using optimized extraction
            extraction_batch_size = 1000  # Documents per SQL batch
            processed = 0
            skipped = 0
            extracted = 0
            embedding_batch: List[Any] = []
            role_map = self._resolve_role_variables(self._variable_defs, config)

            job.current_step = f"Extracting documents from PDM (0/{doc_count})"
            await self._update_job(job)

            async for doc_batch in self.extract_documents_batched(
                config=config,
                variable_map=variable_map,
                max_documents=config.max_documents,
                batch_size=extraction_batch_size,
            ):
                # Check for cancellation at batch boundaries
                if self._cancellation_flags.get(job.id, False):
                    job.status = PdmIndexStatus.CANCELLED
                    job.completed_at = datetime.now(timezone.utc)
                    job.current_step = "Cancelled"
                    await self._update_job(job)
                    logger.info(f"PDM indexing cancelled for job {job.id}")
                    return

                extracted += len(doc_batch)
                job.extracted_documents = extracted
                job.current_step = f"Extracting documents from PDM ({extracted}/{doc_count})"
                await self._update_job(job)

                stored_hashes: dict[int, str] = {}
                if not full_reindex:
                    stored_hashes = await self._get_stored_hashes(
                        job.index_name,
                        [doc.document_id for doc in doc_batch],
                    )

                # Process each document in the extraction batch
                for doc in doc_batch:
                    # Check if document has changed (skip if unchanged and not full reindex)
                    if not full_reindex:
                        current_hash = self._metadata_hash_for(doc, role_map, variable_map)
                        stored_hash = stored_hashes.get(doc.document_id)
                        if stored_hash == current_hash:
                            skipped += 1
                            job.skipped_documents = skipped
                            continue

                    embedding_batch.append(doc)

                    # Process embedding batch when full
                    if len(embedding_batch) >= EMBEDDING_SUB_BATCH_SIZE:
                        job.current_step = f"Generating embeddings ({processed}/{doc_count - skipped})"
                        await self._update_job(job)

                        await self._process_batch(job, embedding_batch, embeddings, role_map, variable_map)
                        processed += len(embedding_batch)
                        job.processed_documents = processed
                        await self._update_job(job)
                        embedding_batch = []

            # Process remaining embedding batch
            if embedding_batch:
                job.current_step = f"Generating embeddings ({processed}/{doc_count - skipped})"
                await self._update_job(job)

                await self._process_batch(job, embedding_batch, embeddings, role_map, variable_map)
                processed += len(embedding_batch)
                job.processed_documents = processed

            # Mark completed
            job.status = PdmIndexStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.current_step = "Completed"
            await self._update_job(job)

            # Clean up
            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)

            logger.info(f"PDM indexing completed for job {job.id}: {processed} processed, {skipped} skipped")

        except asyncio.CancelledError:
            # Task was cancelled during shutdown - don't try to update DB
            logger.info(f"PDM indexing task {job.id} cancelled")
            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)
            raise
        except Exception as e:
            logger.exception(f"PDM indexing failed: {e}")
            job.status = PdmIndexStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.current_step = "Failed"
            job.error_message = str(e)

            # Only try to update DB if not shutting down
            if not self._shutdown:
                try:
                    await self._update_job(job)
                except RuntimeError as db_error:
                    if "Database is not connected" in str(db_error):
                        logger.warning(f"Cannot update job {job.id} - DB disconnected during shutdown")
                    else:
                        raise

            self._active_jobs.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)
            self._running_tasks.pop(job.id, None)

    async def _process_batch(
        self,
        job: PdmIndexJob,
        documents: List[Any],
        embeddings,
        role_map: Optional[Dict[str, Optional[int]]] = None,
        variable_map: Optional[Dict[int, str]] = None,
    ):
        """Process a batch of documents - generate embeddings and store."""
        if not documents:
            return

        db: Any = await get_db()
        role_map = role_map or {}
        variable_map = variable_map or {}

        states = [
            self._record_states.pop(id(doc), None) or self._state_from_record(doc, role_map, variable_map) if isinstance(doc, PdmDocumentRecord) else None
            for doc in documents
        ]
        texts = [state.to_embedding_text() if state is not None else doc.to_embedding_text() for doc, state in zip(documents, states)]

        # Generate embeddings in sub-batches to keep event loop responsive
        doc_embeddings = await embed_documents_subbatched(embeddings, texts, logger_override=logger)

        # Store embeddings and metadata
        for doc, state, text, embedding in zip(documents, states, texts, doc_embeddings):
            metadata_hash = state.compute_metadata_hash() if state is not None else doc.compute_metadata_hash()
            document_type = state.document_type if state is not None else doc.document_type
            part_number = state.part_number if state is not None else (doc.part_number or "")
            filename = state.filename if state is not None else doc.filename
            folder_path = (state.folder_paths[0] if state and state.folder_paths else "") if state is not None else (doc.folder_path or "")
            variables = {name: value.value_text for name, value in state.document_values.items()} if state is not None else doc.variables
            revision = state.target_revision if state is not None else doc.revision_no

            # Upsert embedding - use parameters for all user-provided values
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await db.execute_raw(
                """
                INSERT INTO pdm_embeddings
                    (id, index_name, document_id, document_type, content, part_number,
                     filename, folder_path, metadata, embedding, created_at)
                VALUES
                    (gen_random_uuid(), $1, $2,
                     $3, $4, $5, $6,
                     $7, $8::jsonb, $9::vector, NOW())
                ON CONFLICT (index_name, document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    part_number = EXCLUDED.part_number,
                    folder_path = EXCLUDED.folder_path,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
            """,
                job.index_name,
                doc.document_id,
                document_type,
                text,
                part_number or "",
                filename,
                folder_path,
                json.dumps(variables),
                embedding_str,
            )

            # Upsert document metadata
            await db.execute_raw(
                """
                INSERT INTO pdm_document_metadata
                    (id, index_name, document_id, filename, revision_no, metadata_hash, last_indexed)
                VALUES
                    (gen_random_uuid(), $1, $2,
                     $3, $4, $5, NOW())
                ON CONFLICT (index_name, document_id)
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    revision_no = EXCLUDED.revision_no,
                    metadata_hash = EXCLUDED.metadata_hash,
                    last_indexed = NOW()
            """,
                job.index_name,
                doc.document_id,
                filename,
                revision,
                metadata_hash,
            )

            state_json = (
                state.model_dump_json()
                if state is not None
                else json.dumps(
                    {
                        "document_id": doc.document_id,
                        "filename": filename,
                        "document_type": document_type,
                        "target_revision": revision,
                        "part_number": part_number or None,
                        "document_values": variables,
                    }
                )
            )
            await db.execute_raw(
                """
                INSERT INTO pdm_document_state
                    (id, index_name, document_id, filename, part_number, target_revision, state_json, metadata_hash, extracted_at)
                VALUES (gen_random_uuid(), $1, $2, $3, $4, $5, $6::jsonb, $7, NOW())
                ON CONFLICT (index_name, document_id) DO UPDATE SET
                    filename = EXCLUDED.filename, part_number = EXCLUDED.part_number,
                    target_revision = EXCLUDED.target_revision, state_json = EXCLUDED.state_json,
                    metadata_hash = EXCLUDED.metadata_hash, extracted_at = NOW()
                """,
                job.index_name,
                doc.document_id,
                filename,
                part_number or "",
                revision,
                state_json,
                metadata_hash,
            )

        job.total_chunks += len(documents)
        job.processed_chunks += len(documents)

    async def _count_documents(self, config: SolidworksPdmConnectionConfig) -> int:
        """Count matching documents through the adapter."""
        return await PdmSqlSource(config).count_documents()

    async def _get_stored_hash(self, index_name: str, document_id: int) -> Optional[str]:
        """Get the stored metadata hash for a document."""
        try:
            db: Any = await get_db()
            result = await db.query_raw(
                "SELECT metadata_hash FROM pdm_document_metadata WHERE index_name = $1 AND document_id = $2",
                index_name,
                document_id,
            )
            if result and result[0].get("metadata_hash"):
                return result[0]["metadata_hash"]
        except Exception:
            pass
        return None

    async def _get_stored_hashes(self, index_name: str, document_ids: list[int]) -> dict[int, str]:
        """Get stored metadata hashes for a batch of documents."""
        if not document_ids:
            return {}

        try:
            db: Any = await get_db()
            placeholders = ", ".join(f"${position}" for position in range(2, len(document_ids) + 2))
            result = await db.query_raw(
                (f"SELECT document_id, metadata_hash FROM pdm_document_metadata WHERE index_name = $1 AND document_id IN ({placeholders})"),
                index_name,
                *document_ids,
            )
            return {int(row["document_id"]): row["metadata_hash"] for row in (result or []) if row.get("document_id") is not None and row.get("metadata_hash")}
        except Exception:
            return {}

    async def _clear_embeddings(self, index_name: str):
        """Clear all embeddings for an index."""
        db: Any = await get_db()
        await db.execute_raw("DELETE FROM pdm_embeddings WHERE index_name = $1", index_name)
        await db.execute_raw("DELETE FROM pdm_document_metadata WHERE index_name = $1", index_name)
        await db.execute_raw("DELETE FROM pdm_document_state WHERE index_name = $1", index_name)
        logger.info(f"Cleared PDM embeddings for index {index_name}")

    # =========================================================================
    # Database Helpers
    # =========================================================================

    async def _ensure_pgvector(self) -> bool:
        return await ensure_pgvector_extension(logger_override=logger)

    async def _ensure_embedding_column(self, dimension: int, index_lists: int = 100) -> bool:
        """
        Ensure the embedding column exists with the correct dimension.

        If the column doesn't exist, add it. If it exists but dimension changed,
        alter the column and rebuild the index.

        Args:
            dimension: Dimension of embedding vectors
            index_lists: IVFFlat lists parameter (higher = slower build, faster query)
        """
        return await ensure_embedding_column(
            table_name="pdm_embeddings",
            index_name="pdm_embeddings_embedding_idx",
            embedding_dim=dimension,
            index_lists=index_lists,
            logger_override=logger,
        )

    async def _get_embeddings(self, app_settings: dict):
        """Get the configured embedding model based on app settings."""
        return await get_embeddings_model(app_settings, logger_override=logger)

    async def _create_job(self, job: PdmIndexJob):
        """Create a new job in the database."""
        db: Any = await get_db()
        await db.pdmindexjob.create(
            data={
                "id": job.id,
                "toolConfigId": job.tool_config_id,
                "status": job.status.value,
                "indexName": job.index_name,
                "totalDocuments": job.total_documents,
                "processedDocuments": job.processed_documents,
                "skippedDocuments": job.skipped_documents,
                "totalChunks": job.total_chunks,
                "processedChunks": job.processed_chunks,
                "errorMessage": job.error_message,
                "createdAt": job.created_at,
                "startedAt": job.started_at,
                "completedAt": job.completed_at,
            }
        )

    async def _update_job(self, job: PdmIndexJob):
        """Update job status in database."""
        db: Any = await get_db()
        try:
            await db.pdmindexjob.update(
                where={"id": job.id},
                data={
                    "status": job.status.value,
                    "currentStep": job.current_step,
                    "totalDocuments": job.total_documents,
                    "processedDocuments": job.processed_documents,
                    "skippedDocuments": job.skipped_documents,
                    "extractedDocuments": job.extracted_documents,
                    "totalChunks": job.total_chunks,
                    "processedChunks": job.processed_chunks,
                    "errorMessage": job.error_message,
                    "startedAt": job.started_at,
                    "completedAt": job.completed_at,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to update PDM job: {e}")

    async def _get_job_from_db(self, job_id: str) -> Optional[PdmIndexJobResponse]:
        """Get a job from the database."""
        try:
            db: Any = await get_db()
            prisma_job = await db.pdmindexjob.find_unique(where={"id": job_id})
            if prisma_job:
                job = self._prisma_job_to_model(prisma_job)
                return PdmIndexJobResponse(
                    id=job.id,
                    tool_config_id=job.tool_config_id,
                    status=job.status,
                    index_name=job.index_name,
                    current_step=job.current_step,
                    progress_percent=job.progress_percent,
                    total_documents=job.total_documents,
                    processed_documents=job.processed_documents,
                    skipped_documents=job.skipped_documents,
                    extracted_documents=job.extracted_documents,
                    total_chunks=job.total_chunks,
                    processed_chunks=job.processed_chunks,
                    error_message=job.error_message,
                    cancel_requested=False,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
        except Exception as e:
            logger.warning(f"Error getting PDM job from DB: {e}")
        return None

    def _prisma_job_to_model(self, prisma_job) -> PdmIndexJob:
        """Convert Prisma job to PdmIndexJob model."""
        return PdmIndexJob(
            id=prisma_job.id,
            tool_config_id=prisma_job.toolConfigId,
            status=PdmIndexStatus(str(prisma_job.status)),
            index_name=prisma_job.indexName,
            current_step=getattr(prisma_job, "currentStep", None),
            total_documents=prisma_job.totalDocuments or 0,
            processed_documents=prisma_job.processedDocuments or 0,
            skipped_documents=prisma_job.skippedDocuments or 0,
            extracted_documents=getattr(prisma_job, "extractedDocuments", 0) or 0,
            total_chunks=prisma_job.totalChunks or 0,
            processed_chunks=prisma_job.processedChunks or 0,
            error_message=prisma_job.errorMessage,
            created_at=prisma_job.createdAt,
            started_at=prisma_job.startedAt,
            completed_at=prisma_job.completedAt,
        )


# Singleton instance
pdm_indexer = PdmIndexerService()


# =============================================================================
# Search Functions
# =============================================================================


async def search_pdm_index(
    query: str,
    index_name: str,
    document_type: str | None = None,
    max_results: int = 10,
) -> str:
    """
    Search PDM embeddings for relevant document information.

    Args:
        query: Natural language query about PDM documents (parts, assemblies, etc.)
        index_name: The PDM index name (usually 'pdm_{tool_name}')
        document_type: Optional filter: SLDPRT, SLDASM, SLDDRW, or None for all
        max_results: Maximum number of results to return

    Returns:
        Formatted string with matching PDM documents
    """
    try:
        # Build document type filter before doing embedding work so invalid
        # caller input never reaches SQL construction or expensive providers.
        extra_where = None
        if document_type:
            normalized_document_type = document_type.strip().upper()
            if normalized_document_type not in _ALLOWED_PDM_DOCUMENT_TYPES:
                allowed = ", ".join(sorted(_ALLOWED_PDM_DOCUMENT_TYPES))
                return f"Error: Invalid document_type filter. Allowed values: {allowed}"
            extra_where = f"document_type = '{normalized_document_type}'"

        # Get embedding model
        from ragtime.core.app_settings import get_app_settings

        app_settings = await get_app_settings()
        embeddings = await get_embeddings_model(
            app_settings,
            return_none_on_error=True,
            logger_override=logger,
        )

        if embeddings is None:
            return f"Error: {build_embedding_configuration_error(app_settings, operation='query')}"

        # Generate query embedding
        embedding = await embeddings.aembed_query(query)

        # Search using centralized pgvector search
        results = await search_pgvector_embeddings(
            table_name="pdm_embeddings",
            query_embedding=embedding,
            index_name=index_name,
            max_results=max_results,
            columns=PDM_COLUMNS,
            extra_where=extra_where,
            logger_override=logger,
        )

        if not results:
            return "No matching documents found in PDM index."

        output_parts = []
        for result in results:
            similarity = result.get("similarity", 0)
            content = result.get("content", "")
            filename = result.get("filename", "")
            doc_type = result.get("document_type", "")
            part_number = result.get("part_number", "")

            header = f"[{filename}]"
            if part_number:
                header += f" (PN: {part_number})"
            header += f" [{doc_type}] (similarity: {similarity:.3f})"

            output_parts.append(f"{header}\n{content}")

        return "\n\n---\n\n".join(output_parts)

    except EmbeddingOperationError as e:
        logger.error(f"Error searching PDM index: {e}")
        return f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Error searching PDM index: {e}")
        return f"Error searching PDM: {str(e)}"


def _escape_like(value: str) -> str:
    """Escape a user string for a PostgreSQL LIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def lookup_pdm_documents(
    index_name: str,
    document_id: int | None = None,
    filename: str | None = None,
    part_number: str | None = None,
    configuration: str | None = None,
    max_results: int = 10,
) -> str:
    """Look up deterministic checked-in PDM document snapshots without embeddings."""
    if document_id is None and not filename and not part_number:
        return "Error: Provide document_id, filename, or part_number."
    limit = max(1, min(50, int(max_results)))
    filters = ["index_name = $1"]
    params: list[Any] = [index_name]
    if document_id is not None:
        params.append(int(document_id))
        filters.append(f"document_id = ${len(params)}")
    if filename:
        params.append(f"%{_escape_like(filename)}%")
        filters.append(f"filename ILIKE ${len(params)} ESCAPE '\\'")
    if part_number:
        params.append(f"%{_escape_like(part_number)}%")
        filters.append(f"(part_number ILIKE ${len(params)} ESCAPE '\\' OR state_json::text ILIKE ${len(params)} ESCAPE '\\')")
    query = (
        "SELECT document_id, filename, part_number, target_revision, state_json, extracted_at "
        "FROM pdm_document_state WHERE " + " AND ".join(filters) + f" ORDER BY filename LIMIT {limit}"
    )
    try:
        db: Any = await get_db()
        rows = await db.query_raw(query, *params)
    except Exception as exc:
        logger.error(f"Error looking up PDM documents: {exc}")
        return f"Error looking up PDM documents: {exc}"

    output: list[str] = []
    for row in rows or []:
        try:
            raw_state = row.get("state_json", {})
            if isinstance(raw_state, str):
                raw_state = json.loads(raw_state)
            state = PdmDocumentStateModel.model_validate(raw_state)
        except Exception as exc:
            logger.warning(f"Skipping invalid PDM document state: {exc}")
            continue
        header = f"[{state.filename}]"
        if state.part_number:
            header += f" (PN: {state.part_number})"
        lines = [
            f"{header} [{state.document_type}] target revision {state.target_revision}",
            "Source: indexed snapshot (checked-in view)",
            f"Extracted at: {row.get('extracted_at', '')}",
        ]
        if state.folder_paths:
            lines.append("Folders: " + ", ".join(sorted(state.folder_paths)))
        if state.warnings:
            lines.append("Warnings: " + "; ".join(state.warnings))
        flags = []
        if state.membership_fallback:
            flags.append("membership fallback")
        if state.has_beyond_latest_values:
            flags.append("values beyond latest excluded")
        if flags:
            lines.append("Flags: " + ", ".join(flags))
        if state.document_values:
            lines.append("@ document values:")
            for value in sorted(state.document_values.values(), key=lambda item: (item.variable_id, item.variable_name)):
                if value.is_blank:
                    lines.append(f"- {value.variable_name}: (empty, cleared at v{value.origin_revision})")
                else:
                    lines.append(f"- {value.variable_name}: {value.value_text} (v{value.origin_revision})")
        configurations = state.configurations
        if configuration is not None:
            is_document_scope = configuration.casefold() == "@"
            configurations = [] if is_document_scope else [item for item in configurations if item.name.casefold() == configuration.casefold()]
            if not configurations and not is_document_scope:
                available = ", ".join(item.name for item in state.configurations) or "none"
                lines.append(f"No configuration named {configuration}; available: {available}")
        for item in sorted(configurations, key=lambda entry: entry.configuration_id):
            lines.append(f"Configuration: {item.name} (ID: {item.configuration_id})")
            for value in sorted(item.values.values(), key=lambda entry: (entry.variable_id, entry.variable_name)):
                if value.is_blank:
                    lines.append(f"- {value.variable_name}: (empty, cleared at v{value.origin_revision})")
                else:
                    lines.append(f"- {value.variable_name}: {value.value_text} (v{value.origin_revision})")
        output.append("\n".join(lines))
    return "\n\n---\n\n".join(output) if output else "No matching documents found in PDM index."
