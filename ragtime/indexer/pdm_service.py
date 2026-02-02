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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, cast

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.indexer.models import (
    PdmDocumentInfo,
    PdmIndexJob,
    PdmIndexJobResponse,
    PdmIndexStatus,
    SolidworksPdmConnectionConfig,
)
from ragtime.indexer.repository import repository
from ragtime.indexer.vector_utils import (
    ensure_embedding_column,
    ensure_pgvector_extension,
    get_embeddings_model,
)

# pylint: disable=not-callable

logger = get_logger(__name__)


class PdmIndexerService:
    """Service for creating and managing SolidWorks PDM indexes with pgvector."""

    def __init__(self):
        self._active_jobs: Dict[str, PdmIndexJob] = {}
        self._cancellation_flags: Dict[str, bool] = {}  # job_id -> should_cancel
        self._running_tasks: Dict[str, asyncio.Task] = {}  # job_id -> task
        self._shutdown = False

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
        task = asyncio.create_task(
            self._process_index(job, connection_config, full_reindex)
        )
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

    async def get_latest_job(
        self, tool_config_id: str
    ) -> Optional[PdmIndexJobResponse]:
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
            logger.info(
                f"Directly cancelling orphaned PDM job {job_id} (not in active jobs)"
            )
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
                logger.warning(
                    f"Cannot retry: tool config {db_job.toolConfigId} not found"
                )
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

            for index_name in index_names:
                # Delete embeddings
                result = await db.execute_raw(
                    f"DELETE FROM pdm_embeddings WHERE index_name = '{index_name}'"
                )
                if isinstance(result, int):
                    deleted_embeddings += result

                # Delete document metadata
                result = await db.execute_raw(
                    f"DELETE FROM pdm_document_metadata WHERE index_name = '{index_name}'"
                )
                if isinstance(result, int):
                    deleted_metadata += result

            # Delete jobs
            await db.pdmindexjob.delete_many(where={"toolConfigId": tool_config_id})

            logger.info(
                f"Deleted PDM index for tool {tool_config_id}: "
                f"{deleted_embeddings} embeddings, {deleted_metadata} metadata records"
            )
            return True, f"Deleted {deleted_embeddings} PDM embeddings"

        except Exception as e:
            logger.error(f"Error deleting PDM index: {e}")
            return False, str(e)

    async def get_embedding_count(
        self, tool_config_id: str, tool_name: str | None = None
    ) -> int:
        """Get the number of PDM embeddings for a tool."""
        names_to_check = []
        if tool_name:
            names_to_check.append(f"pdm_{tool_name}")
        names_to_check.append(f"pdm_{tool_config_id}")

        try:
            db: Any = await get_db()
            for index_name in names_to_check:
                result = await db.query_raw(
                    f"SELECT COUNT(*) as count FROM pdm_embeddings "
                    f"WHERE index_name = '{index_name}'"
                )
                if result and int(result[0].get("count", 0)) > 0:
                    return int(result[0].get("count", 0))
        except Exception as e:
            logger.warning(f"Error getting PDM embedding count: {e}")
        return 0

    async def get_document_count(
        self, tool_config_id: str, tool_name: str | None = None
    ) -> int:
        """Get the number of indexed PDM documents for a tool."""
        names_to_check = []
        if tool_name:
            names_to_check.append(f"pdm_{tool_name}")
        names_to_check.append(f"pdm_{tool_config_id}")

        try:
            db: Any = await get_db()
            for index_name in names_to_check:
                result = await db.query_raw(
                    f"SELECT COUNT(*) as count FROM pdm_document_metadata "
                    f"WHERE index_name = '{index_name}'"
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
    ) -> AsyncIterator[List[PdmDocumentInfo]]:
        """
        Extract documents with metadata from PDM database in batches.

        Uses optimized bulk queries instead of N+1 pattern:
        1. Fetch documents in batches with OFFSET/FETCH
        2. Bulk fetch all variables for the batch in one query
        3. Bulk fetch all configurations for the batch in one query
        4. Bulk fetch all BOM components for assemblies in one query

        Args:
            config: PDM connection configuration
            variable_map: Mapping of variable ID to variable name
            max_documents: Optional limit on total documents
            batch_size: Number of documents per batch (default 1000)
            on_batch_extracted: Optional callback(extracted_count, total_count)

        Yields:
            Lists of PdmDocumentInfo objects (one list per batch)
        """
        try:
            import pymssql  # type: ignore[import-untyped]

            pymssql = cast(Any, pymssql)
            connect_fn = cast(Any, getattr(pymssql, "connect", None))
            if not callable(connect_fn):
                raise RuntimeError("pymssql.connect is not available")
        except ImportError as exc:
            raise RuntimeError("pymssql not installed for PDM database access") from exc

        # Build file extension filter
        ext_filter = " OR ".join(
            [f"d.Filename LIKE '%{ext}'" for ext in (config.file_extensions or [])]
        )
        if not ext_filter:
            ext_filter = "1=1"

        deleted_filter = "AND d.Deleted = 0" if config.exclude_deleted else ""
        dip_deleted_filter = (
            "AND (dip.Deleted IS NULL OR dip.Deleted = 0)"
            if config.exclude_deleted
            else ""
        )

        # Get total count first
        total_count = await self._count_documents(config)
        if max_documents:
            total_count = min(total_count, max_documents)

        var_ids = list(variable_map.keys())
        var_ids_str = ",".join(str(v) for v in var_ids) if var_ids else "0"

        def extract_batch(offset: int, limit: int) -> List[PdmDocumentInfo]:
            """Extract a single batch of documents with all related data."""
            conn: Any = connect_fn(
                server=config.host,
                port=str(config.port or 1433),
                user=config.user,
                password=config.password,
                database=config.database,
                login_timeout=30,
                timeout=300,
            )
            cursor: Any = conn.cursor(as_dict=True)

            # Step 1: Get batch of documents
            cursor.execute(
                f"""
                SELECT
                    d.DocumentID,
                    d.Filename,
                    d.LatestRevisionNo,
                    p.Path AS FolderPath
                FROM Documents d
                LEFT JOIN DocumentsInProjects dip ON d.DocumentID = dip.DocumentID
                    {dip_deleted_filter}
                LEFT JOIN Projects p ON dip.ProjectID = p.ProjectID
                WHERE ({ext_filter})
                    {deleted_filter}
                ORDER BY d.DocumentID
                OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY
            """
            )
            doc_rows = cursor.fetchall()
            if not doc_rows:
                conn.close()
                return []

            # Build document ID list and revision map
            doc_ids = [row["DocumentID"] for row in doc_rows]
            doc_ids_str = ",".join(str(d) for d in doc_ids)
            revision_map = {
                row["DocumentID"]: row["LatestRevisionNo"] or 1 for row in doc_rows
            }

            # Step 2: Bulk fetch all variables for this batch
            variables_by_doc: Dict[int, Dict[str, str]] = {
                doc_id: {} for doc_id in doc_ids
            }
            if var_ids:
                cursor.execute(
                    f"""
                    SELECT DISTINCT
                        vv.DocumentID,
                        vv.VariableID,
                        vv.ValueText
                    FROM VariableValue vv
                    WHERE vv.DocumentID IN ({doc_ids_str})
                        AND vv.VariableID IN ({var_ids_str})
                        AND vv.ValueText IS NOT NULL
                        AND vv.ValueText != ''
                """
                )
                for var_row in cursor.fetchall():
                    doc_id = var_row["DocumentID"]
                    var_id = var_row["VariableID"]
                    var_name = variable_map.get(var_id)
                    if var_name and var_row["ValueText"] and doc_id in variables_by_doc:
                        # Only use value if revision matches or we don't have one yet
                        variables_by_doc[doc_id][var_name] = var_row["ValueText"]

            # Step 3: Bulk fetch configurations if enabled
            configs_by_doc: Dict[int, List[dict]] = {doc_id: [] for doc_id in doc_ids}
            if config.include_configurations:
                cursor.execute(
                    f"""
                    SELECT DISTINCT
                        vv.DocumentID,
                        dc.ConfigurationName,
                        vv_pn.ValueText AS PartNumber,
                        vv_desc.ValueText AS Description
                    FROM VariableValue vv
                    INNER JOIN DocumentConfiguration dc
                        ON vv.ConfigurationID = dc.ConfigurationID
                    LEFT JOIN VariableValue vv_pn
                        ON vv.DocumentID = vv_pn.DocumentID
                        AND vv.ConfigurationID = vv_pn.ConfigurationID
                        AND vv.RevisionNo = vv_pn.RevisionNo
                        AND vv_pn.VariableID = 122
                    LEFT JOIN VariableValue vv_desc
                        ON vv.DocumentID = vv_desc.DocumentID
                        AND vv.ConfigurationID = vv_desc.ConfigurationID
                        AND vv.RevisionNo = vv_desc.RevisionNo
                        AND vv_desc.VariableID = 58
                    WHERE vv.DocumentID IN ({doc_ids_str})
                """
                )
                for cfg_row in cursor.fetchall():
                    doc_id = cfg_row["DocumentID"]
                    config_name = cfg_row["ConfigurationName"]
                    if config_name and doc_id in configs_by_doc:
                        configs_by_doc[doc_id].append(
                            {
                                "name": config_name,
                                "part_number": cfg_row["PartNumber"] or "",
                                "description": cfg_row["Description"] or "",
                            }
                        )

            # Step 4: Bulk fetch BOM components for assemblies if enabled
            bom_by_doc: Dict[int, List[dict]] = {doc_id: [] for doc_id in doc_ids}
            if config.include_bom:
                # Find assembly doc IDs
                assembly_ids = [
                    row["DocumentID"]
                    for row in doc_rows
                    if "." in row["Filename"]
                    and row["Filename"].rsplit(".", 1)[-1].upper() == "SLDASM"
                ]
                if assembly_ids:
                    assembly_ids_str = ",".join(str(d) for d in assembly_ids)
                    cursor.execute(
                        f"""
                        SELECT
                            bs.SourceDocumentID,
                            bsr.RowDocumentID AS ChildFileID,
                            d2.Filename AS ChildFilename,
                            dc.ConfigurationName AS ChildConfigName
                        FROM BomSheets bs
                        INNER JOIN BomSheetRow bsr ON bs.BomDocumentID = bsr.BomDocumentID
                        INNER JOIN Documents d2 ON bsr.RowDocumentID = d2.DocumentID
                        LEFT JOIN DocumentConfiguration dc ON bsr.RowConfigurationID = dc.ConfigurationID
                        WHERE bs.SourceDocumentID IN ({assembly_ids_str})
                    """
                    )
                    for bom_row in cursor.fetchall():
                        doc_id = bom_row["SourceDocumentID"]
                        if doc_id in bom_by_doc:
                            bom_by_doc[doc_id].append(
                                {
                                    "document_id": bom_row["ChildFileID"],
                                    "filename": bom_row["ChildFilename"],
                                    "configuration": bom_row["ChildConfigName"] or "",
                                    "quantity": 1,
                                }
                            )

            conn.close()

            # Build document info objects
            documents = []
            for row in doc_rows:
                doc_id = row["DocumentID"]
                filename = row["Filename"]
                revision = revision_map.get(doc_id, 1)
                folder_path = row["FolderPath"]

                doc_type = "UNKNOWN"
                if "." in filename:
                    doc_type = filename.rsplit(".", 1)[-1].upper()

                variables = variables_by_doc.get(doc_id, {})

                doc_info = PdmDocumentInfo(
                    document_id=doc_id,
                    filename=filename,
                    document_type=doc_type,
                    folder_path=folder_path if config.include_folder_path else None,
                    revision_no=revision,
                    part_number=variables.get("Part Number"),
                    description=variables.get("Description"),
                    material=variables.get("Material"),
                    author=variables.get("Author"),
                    stocked_status=variables.get("Stocked Status"),
                    variables=variables,
                    configurations=configs_by_doc.get(doc_id, []),
                    bom_components=bom_by_doc.get(doc_id, []),
                )
                documents.append(doc_info)

            return documents

        # Process in batches
        offset = 0
        total_extracted = 0
        effective_limit = max_documents or total_count

        while offset < effective_limit:
            current_batch_size = min(batch_size, effective_limit - offset)

            # Run extraction in thread to avoid blocking
            batch_docs = await asyncio.to_thread(
                extract_batch, offset, current_batch_size
            )

            if not batch_docs:
                break

            total_extracted += len(batch_docs)

            # Call progress callback if provided
            if on_batch_extracted:
                on_batch_extracted(total_extracted, total_count)

            yield batch_docs

            offset += current_batch_size

            # Small delay to allow other async tasks to run
            await asyncio.sleep(0.01)

    async def extract_documents(
        self,
        config: SolidworksPdmConnectionConfig,
        max_documents: int | None = None,
    ) -> AsyncIterator[PdmDocumentInfo]:
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

    async def _get_variable_map(
        self, config: SolidworksPdmConnectionConfig
    ) -> Dict[int, str]:
        """Get a mapping of variable ID to variable name."""
        try:
            import pymssql  # type: ignore[import-untyped]

            pymssql = cast(Any, pymssql)
            connect_fn = cast(Any, getattr(pymssql, "connect", None))
            if not callable(connect_fn):
                return {}
            connect_fn = cast(Callable[..., Any], connect_fn)
        except ImportError:
            return {}

        connect_fn_callable: Callable[..., Any] = cast(Callable[..., Any], connect_fn)

        def run_query() -> Dict[int, str]:
            conn: Any = cast(Callable[..., Any], connect_fn_callable)(
                server=config.host,
                port=str(config.port or 1433),
                user=config.user,
                password=config.password,
                database=config.database,
                login_timeout=30,
                timeout=60,
            )
            cursor: Any = conn.cursor(as_dict=True)

            # Get all variables and map name -> ID
            cursor.execute("SELECT VariableID, VariableName FROM Variable")
            var_map: Dict[int, str] = {}
            rows = cursor.fetchall() or []
            for row in rows:
                var_name = row["VariableName"]
                if var_name in (config.variable_names or []):
                    var_map[row["VariableID"]] = var_name

            conn.close()
            return var_map

        return await asyncio.to_thread(run_query)

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

            # Get embeddings provider
            job.current_step = "Initializing embedding provider"
            await self._update_job(job)

            embeddings = await self._get_embeddings(app_settings)
            if embeddings is None:
                raise RuntimeError(
                    "No embedding provider configured. Configure an embedding provider in Settings."
                )

            # Check embedding dimension and ensure column matches
            test_embedding = await asyncio.to_thread(
                embeddings.embed_documents, ["test"]
            )
            embedding_dim = len(test_embedding[0])
            index_lists = app_settings.get("ivfflat_lists", 100)
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
                logger.info(
                    f"Updated embedding tracking: dim={embedding_dim}, hash={current_config_hash}"
                )

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
            embedding_batch_size = 50  # Documents per embedding batch
            processed = 0
            skipped = 0
            extracted = 0
            embedding_batch: List[PdmDocumentInfo] = []

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
                job.current_step = (
                    f"Extracting documents from PDM ({extracted}/{doc_count})"
                )
                await self._update_job(job)

                # Process each document in the extraction batch
                for doc in doc_batch:
                    # Check if document has changed (skip if unchanged and not full reindex)
                    if not full_reindex:
                        current_hash = doc.compute_metadata_hash()
                        stored_hash = await self._get_stored_hash(
                            job.index_name, doc.document_id
                        )
                        if stored_hash == current_hash:
                            skipped += 1
                            job.skipped_documents = skipped
                            continue

                    embedding_batch.append(doc)

                    # Process embedding batch when full
                    if len(embedding_batch) >= embedding_batch_size:
                        job.current_step = (
                            f"Generating embeddings ({processed}/{doc_count - skipped})"
                        )
                        await self._update_job(job)

                        await self._process_batch(job, embedding_batch, embeddings)
                        processed += len(embedding_batch)
                        job.processed_documents = processed
                        await self._update_job(job)
                        embedding_batch = []

            # Process remaining embedding batch
            if embedding_batch:
                job.current_step = (
                    f"Generating embeddings ({processed}/{doc_count - skipped})"
                )
                await self._update_job(job)

                await self._process_batch(job, embedding_batch, embeddings)
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

            logger.info(
                f"PDM indexing completed for job {job.id}: "
                f"{processed} processed, {skipped} skipped"
            )

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
            job.error_message = str(e)[:500]

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

    async def _process_batch(
        self,
        job: PdmIndexJob,
        documents: List[PdmDocumentInfo],
        embeddings,
    ):
        """Process a batch of documents - generate embeddings and store."""
        if not documents:
            return

        db: Any = await get_db()

        # Generate text content for each document
        texts = [doc.to_embedding_text() for doc in documents]

        # Generate embeddings
        doc_embeddings = await asyncio.to_thread(embeddings.embed_documents, texts)

        # Store embeddings and metadata
        for doc, text, embedding in zip(documents, texts, doc_embeddings):
            metadata_hash = doc.compute_metadata_hash()

            # Upsert embedding - use parameters for all user-provided values
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            # Escape single quotes in strings for SQL
            safe_filename = doc.filename.replace("'", "''")
            safe_doc_type = doc.document_type.replace("'", "''")
            await db.execute_raw(
                f"""
                INSERT INTO pdm_embeddings
                    (id, index_name, document_id, document_type, content, part_number,
                     filename, folder_path, metadata, embedding, created_at)
                VALUES
                    (gen_random_uuid(), '{job.index_name}', {doc.document_id},
                     '{safe_doc_type}', $1, $2, '{safe_filename}',
                     $3, $4::jsonb, '{embedding_str}'::vector, NOW())
                ON CONFLICT (index_name, document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    part_number = EXCLUDED.part_number,
                    folder_path = EXCLUDED.folder_path,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
            """,
                text,
                doc.part_number or "",
                doc.folder_path or "",
                json.dumps(doc.variables),
            )

            # Upsert document metadata
            await db.execute_raw(
                f"""
                INSERT INTO pdm_document_metadata
                    (id, index_name, document_id, filename, revision_no, metadata_hash, last_indexed)
                VALUES
                    (gen_random_uuid(), '{job.index_name}', {doc.document_id},
                     '{safe_filename}', {doc.revision_no}, '{metadata_hash}', NOW())
                ON CONFLICT (index_name, document_id)
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    revision_no = EXCLUDED.revision_no,
                    metadata_hash = EXCLUDED.metadata_hash,
                    last_indexed = NOW()
            """
            )

        job.total_chunks += len(documents)
        job.processed_chunks += len(documents)

    async def _count_documents(self, config: SolidworksPdmConnectionConfig) -> int:
        """Count documents matching the filter criteria."""
        try:
            import pymssql  # type: ignore[import-untyped]

            pymssql = cast(Any, pymssql)
            connect_fn = cast(Any, getattr(pymssql, "connect", None))
            if not callable(connect_fn):
                return 0
            connect_fn = cast(Callable[..., Any], connect_fn)
        except ImportError:
            return 0

        connect_fn_callable: Callable[..., Any] = cast(Callable[..., Any], connect_fn)

        def run_count() -> int:
            conn: Any = cast(Callable[..., Any], connect_fn_callable)(
                server=config.host,
                port=str(config.port or 1433),
                user=config.user,
                password=config.password,
                database=config.database,
                login_timeout=30,
                timeout=60,
            )
            cursor: Any = conn.cursor()

            # Extensions may already include the dot (e.g., '.SLDPRT'), so use them as-is
            ext_filter = " OR ".join(
                [f"Filename LIKE '%{ext}'" for ext in (config.file_extensions or [])]
            )
            if not ext_filter:
                ext_filter = "1=1"

            deleted_filter = "AND Deleted = 0" if config.exclude_deleted else ""

            cursor.execute(
                f"""
                SELECT COUNT(*) FROM Documents
                WHERE ({ext_filter}) {deleted_filter}
            """
            )
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else 0

        return await asyncio.to_thread(run_count)

    async def _get_stored_hash(
        self, index_name: str, document_id: int
    ) -> Optional[str]:
        """Get the stored metadata hash for a document."""
        try:
            db: Any = await get_db()
            result = await db.query_raw(
                f"""
                SELECT metadata_hash FROM pdm_document_metadata
                WHERE index_name = '{index_name}' AND document_id = {document_id}
            """
            )
            if result and result[0].get("metadata_hash"):
                return result[0]["metadata_hash"]
        except Exception:
            pass
        return None

    async def _clear_embeddings(self, index_name: str):
        """Clear all embeddings for an index."""
        db: Any = await get_db()
        await db.execute_raw(
            f"DELETE FROM pdm_embeddings WHERE index_name = '{index_name}'"
        )
        await db.execute_raw(
            f"DELETE FROM pdm_document_metadata WHERE index_name = '{index_name}'"
        )
        logger.info(f"Cleared PDM embeddings for index {index_name}")

    # =========================================================================
    # Database Helpers
    # =========================================================================

    async def _ensure_pgvector(self) -> bool:
        return await ensure_pgvector_extension(logger_override=logger)

    async def _ensure_embedding_column(
        self, dimension: int, index_lists: int = 100
    ) -> bool:
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
    from ragtime.indexer.vector_utils import PDM_COLUMNS, search_pgvector_embeddings

    try:
        # Get embedding model
        from ragtime.core.app_settings import get_app_settings

        app_settings = await get_app_settings()
        embeddings = await get_embeddings_model(
            app_settings,
            return_none_on_error=True,
            logger_override=logger,
        )

        if embeddings is None:
            return "Error: No embedding provider configured"

        # Generate query embedding
        query_embedding = await asyncio.to_thread(embeddings.embed_documents, [query])
        embedding = query_embedding[0]

        # Build document type filter
        extra_where = None
        if document_type:
            extra_where = f"document_type = '{document_type.upper()}'"

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

    except Exception as e:
        logger.error(f"Error searching PDM index: {e}")
        return f"Error searching PDM: {str(e)}"
