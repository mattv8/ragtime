"""
Schema Indexer Service - Creates and manages pgvector-based schema indexes for SQL databases.

This service handles:
- Introspecting database schemas from PostgreSQL, MSSQL, and MySQL/MariaDB
- Converting schema definitions to embeddable text chunks
- Storing embeddings in PostgreSQL using pgvector
- Progress tracking and job management
- Incremental indexing based on schema hash

Each table becomes one embedding chunk containing:
- Table name and type (TABLE/VIEW)
- All column definitions with types and constraints
- Primary key information
- Foreign key relationships
- Index definitions
"""

import asyncio
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.ssh import (SSHTunnel, build_ssh_tunnel_config,
                              ssh_tunnel_config_from_dict)
from ragtime.indexer.models import (SchemaIndexConfig, SchemaIndexJob,
                                    SchemaIndexJobResponse, SchemaIndexStatus,
                                    TableSchemaInfo)
from ragtime.indexer.repository import repository
from ragtime.indexer.utils import safe_tool_name
from ragtime.indexer.vector_utils import (SCHEMA_COLUMNS,
                                          ensure_embedding_column,
                                          ensure_pgvector_extension,
                                          get_embeddings_model,
                                          search_pgvector_embeddings)

logger = get_logger(__name__)


# =============================================================================
# Database Type Constants
# =============================================================================

DB_TYPE_POSTGRES = "postgres"
DB_TYPE_MSSQL = "mssql"
DB_TYPE_MYSQL = "mysql"

# Tool types that support schema indexing
SCHEMA_INDEXER_CAPABLE_TYPES = frozenset(
    {DB_TYPE_POSTGRES, DB_TYPE_MSSQL, DB_TYPE_MYSQL}
)


class SchemaIndexerService:
    """Service for creating and managing SQL database schema indexes with pgvector."""

    def __init__(self):
        self._active_jobs: Dict[str, SchemaIndexJob] = {}
        self._active_tasks: Dict[str, asyncio.Task] = (
            {}
        )  # prevent GC of fire-and-forget tasks
        self._cancellation_flags: Dict[str, bool] = {}  # job_id -> should_cancel

    # =========================================================================
    # Public API
    # =========================================================================

    async def trigger_index(
        self,
        tool_config_id: str,
        tool_type: str,
        connection_config: dict,
        full_reindex: bool = False,
        tool_name: str | None = None,
    ) -> SchemaIndexJob:
        """
        Trigger schema indexing for a SQL database tool.

        Args:
            tool_config_id: The tool configuration ID
            tool_type: Type of database (postgres, mssql, mysql)
            connection_config: Database connection configuration
            full_reindex: If True, re-index all tables regardless of hash
            tool_name: Safe tool name for display (if None, uses tool_config_id)

        Returns:
            The created SchemaIndexJob
        """
        # Check for existing active job
        existing_job = await self.get_active_job(tool_config_id)
        if existing_job and existing_job.status in (
            SchemaIndexStatus.PENDING,
            SchemaIndexStatus.INDEXING,
        ):
            logger.info(f"Schema index job already running for tool {tool_config_id}")
            return existing_job

        # Create job - use tool_name for display, but still key by tool_config_id
        job_id = str(uuid.uuid4())
        safe_name = tool_name or tool_config_id
        index_name = f"schema_{safe_name}"

        job = SchemaIndexJob(
            id=job_id,
            tool_config_id=tool_config_id,
            status=SchemaIndexStatus.PENDING,
            index_name=index_name,
            created_at=datetime.now(timezone.utc),
        )

        # Store job in database
        await self._create_job(job)
        self._active_jobs[job_id] = job
        self._cancellation_flags[job_id] = False

        # Start background processing — hold a strong reference so the task
        # is not garbage-collected before completion (asyncio keeps only weak refs).
        task = asyncio.create_task(
            self._process_index(job, tool_type, connection_config, full_reindex)
        )
        self._active_tasks[job_id] = task

        logger.info(f"Started schema indexing job {job_id} for tool {tool_config_id}")
        return job

    async def get_job_status(self, job_id: str) -> Optional[SchemaIndexJobResponse]:
        """Get the current status of a schema indexing job."""
        # Check in-memory first
        if job_id in self._active_jobs:
            return self.job_to_response(self._active_jobs[job_id])

        # Fall back to database
        return await self._get_job_from_db(job_id)

    async def get_active_job(self, tool_config_id: str) -> Optional[SchemaIndexJob]:
        """Get any active job for a tool config."""
        # Check in-memory first
        for job in self._active_jobs.values():
            if job.tool_config_id == tool_config_id:
                return job

        # Check database for pending/indexing jobs
        try:
            db = await get_db()
            prisma_job = await db.schemaindexjob.find_first(
                where={
                    "toolConfigId": tool_config_id,
                    "status": {"in": ["pending", "indexing"]},
                },
                order={"createdAt": "desc"},
            )
            if prisma_job:
                return self._prisma_job_to_model(prisma_job)
        except Exception as e:
            logger.warning(f"Error checking for active schema job: {e}")

        return None

    async def get_latest_job(
        self, tool_config_id: str
    ) -> Optional[SchemaIndexJobResponse]:
        """Get the most recent job for a tool config."""
        try:
            db = await get_db()
            prisma_job = await db.schemaindexjob.find_first(
                where={"toolConfigId": tool_config_id},
                order={"createdAt": "desc"},
            )
            if prisma_job:
                job = self._prisma_job_to_model(prisma_job)
                return self.job_to_response(job)
        except Exception as e:
            logger.warning(f"Error getting latest schema job: {e}")

        return None

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of an active job."""
        # Check if job is active in memory
        if job_id in self._cancellation_flags:
            self._cancellation_flags[job_id] = True
            if job_id in self._active_jobs:
                self._active_jobs[job_id].cancel_requested = True
            logger.info(f"Cancellation requested for schema job {job_id}")
            return True

        # Check database for orphaned jobs (running in DB but not in memory)
        db: Any = await get_db()
        prisma_job = await db.schemaindexjob.find_unique(where={"id": job_id})
        if not prisma_job:
            return False

        # Can only cancel pending/indexing jobs
        if prisma_job.status in ("pending", "indexing"):
            # Job is not running in memory but stuck in DB - directly mark as cancelled
            logger.info(
                f"Directly cancelling orphaned schema job {job_id} (not in active jobs)"
            )
            await db.schemaindexjob.update(
                where={"id": job_id},
                data={
                    "status": "cancelled",
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
                UPDATE schema_index_jobs
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
                logger.info(f"Cleaned up {count} orphaned schema indexing job(s)")
            return count

        except Exception as e:
            logger.warning(f"Failed to clean up orphaned schema jobs: {e}")
            return 0

    async def shutdown(self) -> None:
        """Shutdown the service and clear active job tracking."""
        logger.info("Schema indexer service shutting down")
        self._active_jobs.clear()
        self._cancellation_flags.clear()

    async def list_all_jobs(self, limit: int = 50) -> List[SchemaIndexJobResponse]:
        """
        List all schema indexing jobs across all tools.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of schema job responses, sorted by created_at desc
        """
        jobs: List[SchemaIndexJobResponse] = []

        try:
            db = await get_db()

            # Get jobs from database
            db_jobs = await db.schemaindexjob.find_many(
                take=limit,
                order={"createdAt": "desc"},
            )

            for db_job in db_jobs:
                # Check if this job is still active in memory (fresher data)
                if db_job.id in self._active_jobs:
                    jobs.append(self.job_to_response(self._active_jobs[db_job.id]))
                else:
                    # Use database data - convert to model first
                    job = self._prisma_job_to_model(db_job)
                    jobs.append(
                        self.job_to_response(job, cancel_requested_override=False)
                    )

        except Exception as e:
            logger.warning(f"Error listing schema jobs: {e}")

        return jobs

    async def retry_job(self, job_id: str) -> Optional[SchemaIndexJob]:
        """
        Retry a failed or cancelled schema indexing job.

        Args:
            job_id: The job ID to retry

        Returns:
            New job if successful, None if original job not found or not retryable
        """
        try:
            db = await get_db()

            # Get the original job
            db_job = await db.schemaindexjob.find_unique(where={"id": job_id})
            if not db_job:
                logger.warning(f"Cannot retry: job {job_id} not found")
                return None

            # Convert to get status as string
            job_status = str(db_job.status)

            # Only allow retry for failed or cancelled jobs
            if job_status not in ("failed", "cancelled"):
                logger.warning(f"Cannot retry: job {job_id} has status {job_status}")
                return None

            # Get the tool config to get connection details
            tool_config = await repository.get_tool_config(db_job.toolConfigId)
            if not tool_config:
                logger.warning(
                    f"Cannot retry: tool config {db_job.toolConfigId} not found"
                )
                return None

            # Trigger a new indexing job
            return await self.trigger_index(
                tool_config_id=tool_config.id,
                tool_type=tool_config.tool_type.value,
                connection_config=tool_config.connection_config or {},
                full_reindex=True,  # Force full reindex on retry
                tool_name=safe_tool_name(tool_config.name) or None,
            )

        except Exception as e:
            logger.error(f"Error retrying schema job {job_id}: {e}")
            return None

    async def delete_index(
        self, tool_config_id: str, tool_name: str | None = None
    ) -> Tuple[bool, str]:
        """
        Delete all schema embeddings for a tool config.

        Args:
            tool_config_id: The tool configuration ID
            tool_name: Safe tool name used in index_name (preferred)

        Returns:
            Tuple of (success, message)
        """
        # Try tool_name first (new convention), then tool_config_id (legacy)
        names_to_delete = []
        if tool_name:
            names_to_delete.append(f"schema_{tool_name}")
        names_to_delete.append(f"schema_{tool_config_id}")

        try:
            db = await get_db()

            total_deleted = 0
            for index_name in names_to_delete:
                # Delete embeddings
                result = await db.execute_raw(
                    f"DELETE FROM schema_embeddings WHERE index_name = '{index_name}'"
                )
                deleted_count = result if isinstance(result, int) else 0
                total_deleted += deleted_count

            # Delete jobs
            await db.schemaindexjob.delete_many(where={"toolConfigId": tool_config_id})

            logger.info(
                f"Deleted schema index for tool {tool_config_id}: "
                f"{total_deleted} embeddings removed"
            )
            return True, f"Deleted {total_deleted} schema embeddings"

        except Exception as e:
            logger.error(f"Error deleting schema index: {e}")
            return False, str(e)

    async def get_embedding_count(
        self, tool_config_id: str, tool_name: str | None = None
    ) -> int:
        """Get the number of schema embeddings for a tool.

        Args:
            tool_config_id: The tool configuration ID (for backwards compatibility)
            tool_name: Safe tool name used in index_name (preferred)

        Returns:
            Number of embeddings, checking both naming conventions
        """
        # Try tool_name first (new convention), then tool_config_id (legacy)
        names_to_check = []
        if tool_name:
            names_to_check.append(f"schema_{tool_name}")
        names_to_check.append(f"schema_{tool_config_id}")

        try:
            db = await get_db()
            for index_name in names_to_check:
                result = await db.query_raw(
                    f"SELECT COUNT(*) as count FROM schema_embeddings "
                    f"WHERE index_name = '{index_name}'"
                )
                if result and int(result[0].get("count", 0)) > 0:
                    return int(result[0].get("count", 0))
        except Exception as e:
            logger.warning(f"Error getting embedding count: {e}")
        return 0

    # =========================================================================
    # Schema Introspection
    # =========================================================================

    async def test_connection(
        self, tool_type: str, connection_config: dict
    ) -> tuple[bool, str | None]:
        """
        Test database connectivity before starting indexing.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if tool_type == DB_TYPE_POSTGRES:
                host = connection_config.get("host", "")
                port = connection_config.get("port", 5432)
                user = connection_config.get("user", "")
                password = connection_config.get("password", "")
                database = connection_config.get("database", "")
                container = connection_config.get("container", "")

                # Set up SSH tunnel if configured
                tunnel: SSHTunnel | None = None
                tunnel_cfg_dict = build_ssh_tunnel_config(connection_config, host, port)
                if tunnel_cfg_dict and host:
                    tunnel_cfg = ssh_tunnel_config_from_dict(
                        tunnel_cfg_dict, default_remote_port=5432
                    )
                    if tunnel_cfg:
                        tunnel = SSHTunnel(tunnel_cfg)
                        local_port = tunnel.start()
                        host = "127.0.0.1"
                        port = local_port

                try:
                    # Simple SELECT 1 to verify connection
                    result = await self._execute_postgres_query(
                        "SELECT 1 AS ok",
                        host,
                        port,
                        user,
                        password,
                        database,
                        container,
                    )
                    if not result:
                        return False, "Connection test returned no results"
                    return True, None
                finally:
                    if tunnel:
                        tunnel.stop()
            elif tool_type == DB_TYPE_MSSQL:
                # For MSSQL, we'll test during introspection
                # as it uses pymssql which handles connection testing
                return True, None
            elif tool_type == DB_TYPE_MYSQL:
                # For MySQL, we'll test during introspection
                # as it uses pymysql which handles connection testing
                return True, None
            else:
                return False, f"Unsupported database type: {tool_type}"
        except Exception as e:
            error_msg = str(e)
            # Extract meaningful error from full traceback
            if "password authentication failed" in error_msg.lower():
                return (
                    False,
                    "Password authentication failed. Check username and password.",
                )
            if (
                "could not connect" in error_msg.lower()
                or "connection refused" in error_msg.lower()
            ):
                return False, "Could not connect to database. Check host and port."
            if "does not exist" in error_msg.lower():
                return False, "Database does not exist. Check database name."
            return False, f"Connection test failed: {error_msg}"

    # Type for optional progress callback: (total_discovered, introspected_count, current_table_name) -> Awaitable
    IntrospectionCallback = Callable[[int, int, str], Awaitable[None]]

    async def introspect_schema(
        self,
        tool_type: str,
        connection_config: dict,
        on_progress: "IntrospectionCallback | None" = None,
    ) -> List[TableSchemaInfo]:
        """
        Introspect database schema and return structured table information.

        Args:
            tool_type: Database type (postgres, mssql, mysql)
            connection_config: Database connection configuration
            on_progress: Optional async callback(total, introspected, table_name)
                         called after each table is introspected

        Returns:
            List of TableSchemaInfo objects for each table/view
        """
        if tool_type == DB_TYPE_POSTGRES:
            return await self._introspect_postgres(connection_config, on_progress)
        elif tool_type == DB_TYPE_MSSQL:
            return await self._introspect_mssql(connection_config, on_progress)
        elif tool_type == DB_TYPE_MYSQL:
            return await self._introspect_mysql(connection_config, on_progress)
        else:
            raise ValueError(f"Unsupported database type: {tool_type}")

    async def _introspect_postgres(
        self,
        connection_config: dict,
        on_progress: "IntrospectionCallback | None" = None,
    ) -> List[TableSchemaInfo]:
        """Introspect PostgreSQL database schema."""
        host = connection_config.get("host", "")
        port = connection_config.get("port", 5432)
        user = connection_config.get("user", "")
        password = connection_config.get("password", "")
        database = connection_config.get("database", "")
        container = connection_config.get("container", "")

        # Set up SSH tunnel if configured
        tunnel: SSHTunnel | None = None
        tunnel_cfg_dict = build_ssh_tunnel_config(connection_config, host, port)
        if tunnel_cfg_dict and host:
            tunnel_cfg = ssh_tunnel_config_from_dict(
                tunnel_cfg_dict, default_remote_port=5432
            )
            if tunnel_cfg:
                tunnel = SSHTunnel(tunnel_cfg)
                local_port = tunnel.start()
                logger.debug(
                    f"SSH tunnel established: localhost:{local_port} -> "
                    f"{tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}"
                )
                host = "127.0.0.1"
                port = local_port

        try:
            return await self._introspect_postgres_impl(
                host, port, user, password, database, container, on_progress
            )
        finally:
            if tunnel:
                tunnel.stop()

    async def _introspect_postgres_impl(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
        on_progress: "IntrospectionCallback | None" = None,
    ) -> List[TableSchemaInfo]:
        """Internal PostgreSQL introspection (after tunnel setup).

        Discovers tables, regular views, and materialized views, then
        introspects columns, constraints, and indexes for each one.
        """

        # Query tables and regular views from information_schema
        tables_query = """
        SELECT
            t.table_schema,
            t.table_name,
            t.table_type,
            pg_catalog.obj_description(
                (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass,
                'pg_class'
            ) as table_comment,
            (SELECT reltuples::bigint FROM pg_class
             WHERE oid = (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass
            ) as row_estimate
        FROM information_schema.tables t
        WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        ORDER BY t.table_schema, t.table_name
        """

        # Materialized views are NOT in information_schema.tables;
        # they live in pg_class with relkind = 'm'.
        matview_query = """
        SELECT
            n.nspname AS table_schema,
            c.relname AS table_name,
            'MATERIALIZED VIEW' AS table_type,
            pg_catalog.obj_description(c.oid, 'pg_class') AS table_comment,
            c.reltuples::bigint AS row_estimate
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'm'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        ORDER BY n.nspname, c.relname
        """

        tables: List[TableSchemaInfo] = []

        try:
            # Discover all tables, views, and materialized views
            table_rows = await self._execute_postgres_query(
                tables_query, host, port, user, password, database, container
            )
            matview_rows = await self._execute_postgres_query(
                matview_query, host, port, user, password, database, container
            )

            # Combine and deduplicate (matviews won't overlap with info_schema)
            all_rows = table_rows + matview_rows
            total_discovered = len(all_rows)

            logger.info(
                f"Discovered {total_discovered} objects "
                f"({len(table_rows)} tables/views, {len(matview_rows)} materialized views)"
            )

            # Report initial discovery count
            if on_progress and total_discovered > 0:
                await on_progress(total_discovered, 0, "")

            for idx, row in enumerate(all_rows):
                schema = row.get("table_schema", "public")
                table_name = row.get("table_name", "")
                table_type_raw = row.get("table_type", "BASE TABLE")
                row_estimate = row.get("row_estimate")

                full_name = f"{schema}.{table_name}"

                # Normalize table type for display
                if "MATERIALIZED" in table_type_raw.upper():
                    table_type = "MATERIALIZED VIEW"
                elif "VIEW" in table_type_raw.upper():
                    table_type = "VIEW"
                else:
                    table_type = "TABLE"

                # Get columns for this table/view
                columns = await self._get_postgres_columns(
                    schema, table_name, host, port, user, password, database, container
                )

                # Get primary key (not applicable for views, but query is safe)
                pk = await self._get_postgres_primary_key(
                    schema, table_name, host, port, user, password, database, container
                )

                # Get foreign keys
                fks = await self._get_postgres_foreign_keys(
                    schema, table_name, host, port, user, password, database, container
                )

                # Get indexes (materialized views can have indexes)
                indexes = await self._get_postgres_indexes(
                    schema, table_name, host, port, user, password, database, container
                )

                tables.append(
                    TableSchemaInfo(
                        table_schema=schema,
                        table_name=table_name,
                        full_name=full_name,
                        table_type=table_type,
                        columns=columns,
                        primary_key=pk,
                        foreign_keys=fks,
                        indexes=indexes,
                        row_count_estimate=int(row_estimate) if row_estimate else None,
                    )
                )

                # Report progress after each table
                if on_progress:
                    await on_progress(total_discovered, idx + 1, full_name)

            logger.info(
                f"Introspected {len(tables)} tables/views/materialized views from PostgreSQL"
            )
            return tables

        except Exception as e:
            logger.error(f"Error introspecting PostgreSQL schema: {e}")
            raise

    async def _execute_postgres_query(
        self,
        query: str,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[dict]:
        """Execute a PostgreSQL query and return results as list of dicts."""
        # Escape the query for shell
        escaped_query = query.replace("'", "'\\''")

        if host:
            # Direct connection
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
                "-t",  # Tuples only (no headers/footers)
                "-A",  # Unaligned output
                "-F",
                "\t",  # Tab-separated
                "-c",
                query,
            ]
            env = {"PGPASSWORD": password}
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, env=env, timeout=60
            )
        elif container:
            # Docker container
            inner_cmd = (
                f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" '
                f'-d "$POSTGRES_DB" -t -A -F "\t" -c \'{escaped_query}\''
            )
            cmd = ["docker", "exec", "-i", container, "bash", "-c", inner_cmd]
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=60
            )
        else:
            raise ValueError("No PostgreSQL connection configured (host or container)")

        if result.returncode != 0:
            raise RuntimeError(f"PostgreSQL query failed: {result.stderr}")

        # Parse tab-separated output
        rows = []
        output = result.stdout.strip()
        if not output:
            return rows

        # For simple queries, parse assuming column order matches
        # This is a simplified parser - we'll use JSON output for complex queries
        for line in output.split("\n"):
            if line.strip():
                parts = line.split("\t")
                # Map to dict based on expected columns
                if len(parts) >= 5:  # tables_query
                    rows.append(
                        {
                            "table_schema": parts[0],
                            "table_name": parts[1],
                            "table_type": parts[2],
                            "table_comment": parts[3] if parts[3] != "\\N" else None,
                            "row_estimate": parts[4] if parts[4] != "\\N" else None,
                        }
                    )
                elif len(parts) == 1:
                    # Generic single-column result (e.g. connection check)
                    rows.append({"result": parts[0]})

        return rows

    async def _get_postgres_columns(
        self,
        schema: str,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[dict]:
        """Get column information for a PostgreSQL table, view, or materialized view."""
        # information_schema.columns works for tables and regular views
        query = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        columns = await self._execute_postgres_simple_query(
            query,
            ["name", "type", "nullable", "default"],
            host,
            port,
            user,
            password,
            database,
            container,
        )

        # Fallback for materialized views (not in information_schema)
        if not columns:
            matview_query = f"""
            SELECT
                a.attname AS column_name,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                CASE WHEN a.attnotnull THEN 'NO' ELSE 'YES' END AS is_nullable,
                pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS column_default
            FROM pg_attribute a
            LEFT JOIN pg_attrdef d ON a.attrelid = d.adrelid AND a.attnum = d.adnum
            WHERE a.attrelid = (quote_ident('{schema}') || '.' || quote_ident('{table_name}'))::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """
            columns = await self._execute_postgres_simple_query(
                matview_query,
                ["name", "type", "nullable", "default"],
                host,
                port,
                user,
                password,
                database,
                container,
            )

        return columns

    async def _get_postgres_primary_key(
        self,
        schema: str,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[str]:
        """Get primary key columns for a PostgreSQL table."""
        query = f"""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_schema = '{schema}'
            AND tc.table_name = '{table_name}'
        ORDER BY kcu.ordinal_position
        """
        results = await self._execute_postgres_simple_query(
            query, ["column_name"], host, port, user, password, database, container
        )
        return [r["column_name"] for r in results]

    async def _get_postgres_foreign_keys(
        self,
        schema: str,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[dict]:
        """Get foreign key relationships for a PostgreSQL table."""
        query = f"""
        SELECT
            tc.constraint_name,
            kcu.column_name,
            ccu.table_schema || '.' || ccu.table_name as references_table,
            ccu.column_name as references_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = '{schema}'
            AND tc.table_name = '{table_name}'
        """
        results = await self._execute_postgres_simple_query(
            query,
            ["name", "column", "references_table", "references_column"],
            host,
            port,
            user,
            password,
            database,
            container,
        )

        # Group by constraint name
        fk_map: Dict[str, dict] = {}
        for r in results:
            name = r["name"]
            if name not in fk_map:
                fk_map[name] = {
                    "name": name,
                    "columns": [],
                    "references_table": r["references_table"],
                    "references_columns": [],
                }
            fk_map[name]["columns"].append(r["column"])
            fk_map[name]["references_columns"].append(r["references_column"])

        return list(fk_map.values())

    async def _get_postgres_indexes(
        self,
        schema: str,
        table_name: str,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[dict]:
        """Get index definitions for a PostgreSQL table."""
        query = f"""
        SELECT
            i.relname as index_name,
            ix.indisunique as is_unique,
            array_to_string(array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)), ',') as columns
        FROM pg_class t
        JOIN pg_index ix ON t.oid = ix.indrelid
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
        WHERE n.nspname = '{schema}'
            AND t.relname = '{table_name}'
            AND NOT ix.indisprimary
        GROUP BY i.relname, ix.indisunique
        """
        results = await self._execute_postgres_simple_query(
            query,
            ["name", "unique", "columns"],
            host,
            port,
            user,
            password,
            database,
            container,
        )

        return [
            {
                "name": r["name"],
                "unique": r["unique"] == "t" or r["unique"] is True,
                "columns": r["columns"].split(",") if r["columns"] else [],
            }
            for r in results
        ]

    async def _execute_postgres_simple_query(
        self,
        query: str,
        column_names: List[str],
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        container: str,
    ) -> List[dict]:
        """Execute a simple PostgreSQL query with known columns."""
        escaped_query = query.replace("'", "'\\''")

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
                "-t",
                "-A",
                "-F",
                "\t",
                "-c",
                query,
            ]
            env = {"PGPASSWORD": password}
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, env=env, timeout=60
            )
        elif container:
            inner_cmd = (
                f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" '
                f'-d "$POSTGRES_DB" -t -A -F "\t" -c \'{escaped_query}\''
            )
            cmd = ["docker", "exec", "-i", container, "bash", "-c", inner_cmd]
            result = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=60
            )
        else:
            return []

        if result.returncode != 0:
            logger.warning(f"Query failed: {result.stderr}")
            return []

        rows = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split("\t")
                row = {}
                for i, col_name in enumerate(column_names):
                    if i < len(parts):
                        val = parts[i]
                        row[col_name] = None if val == "\\N" else val
                    else:
                        row[col_name] = None
                rows.append(row)

        return rows

    async def _introspect_mssql(
        self,
        connection_config: dict,
        on_progress: "IntrospectionCallback | None" = None,
    ) -> List[TableSchemaInfo]:
        """Introspect MSSQL/SQL Server database schema."""
        try:
            import pymssql
        except ImportError:
            raise RuntimeError("pymssql not installed for MSSQL schema introspection")

        host = connection_config.get("host", "")
        port = connection_config.get("port", 1433)
        user = connection_config.get("user", "")
        password = connection_config.get("password", "")
        database = connection_config.get("database", "")

        # Build SSH tunnel config dict from top-level ssh_tunnel_* fields
        tunnel_cfg_dict = build_ssh_tunnel_config(connection_config, host, port)

        tables = []

        def run_introspection() -> List[TableSchemaInfo]:
            tunnel: SSHTunnel | None = None
            actual_host = host
            actual_port = port

            try:
                # Set up SSH tunnel if configured
                if tunnel_cfg_dict:
                    tunnel_cfg = ssh_tunnel_config_from_dict(
                        tunnel_cfg_dict, default_remote_port=1433
                    )
                    if tunnel_cfg:
                        tunnel = SSHTunnel(tunnel_cfg)
                        local_port = tunnel.start()
                        actual_host = "127.0.0.1"
                        actual_port = local_port
                        logger.debug(
                            f"SSH tunnel established: localhost:{local_port} -> "
                            f"{tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}"
                        )

                conn = pymssql.connect(
                    server=actual_host,
                    port=str(actual_port),
                    user=user,
                    password=password,
                    database=database,
                    login_timeout=30,
                    timeout=60,
                )
                cursor = conn.cursor(as_dict=True)

                # Get all tables and views
                cursor.execute(
                    """
                    SELECT
                        s.name AS table_schema,
                        t.name AS table_name,
                        CASE WHEN t.type = 'V' THEN 'VIEW' ELSE 'TABLE' END AS table_type,
                        p.rows AS row_estimate
                    FROM sys.tables t
                    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
                    LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1)
                    UNION ALL
                    SELECT
                        s.name AS table_schema,
                        v.name AS table_name,
                        'VIEW' AS table_type,
                        NULL AS row_estimate
                    FROM sys.views v
                    INNER JOIN sys.schemas s ON v.schema_id = s.schema_id
                    ORDER BY table_schema, table_name
                """
                )

                table_rows = cursor.fetchall()
                result_tables = []

                for row in table_rows:
                    schema = row["table_schema"]
                    table_name = row["table_name"]
                    table_type = row["table_type"]
                    row_estimate = row.get("row_estimate")
                    full_name = f"{schema}.{table_name}"

                    # Get columns
                    cursor.execute(
                        f"""
                        SELECT
                            c.name AS column_name,
                            t.name + CASE
                                WHEN t.name IN ('varchar', 'nvarchar', 'char', 'nchar')
                                    THEN '(' + CASE WHEN c.max_length = -1 THEN 'MAX' ELSE CAST(c.max_length AS VARCHAR) END + ')'
                                WHEN t.name IN ('decimal', 'numeric')
                                    THEN '(' + CAST(c.precision AS VARCHAR) + ',' + CAST(c.scale AS VARCHAR) + ')'
                                ELSE ''
                            END AS data_type,
                            c.is_nullable,
                            OBJECT_DEFINITION(c.default_object_id) AS column_default
                        FROM sys.columns c
                        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
                        WHERE c.object_id = OBJECT_ID('{schema}.{table_name}')
                        ORDER BY c.column_id
                    """
                    )
                    columns = [
                        {
                            "name": c["column_name"],
                            "type": c["data_type"],
                            "nullable": c["is_nullable"],
                            "default": c["column_default"],
                        }
                        for c in cursor.fetchall()
                    ]

                    # Get primary key
                    cursor.execute(
                        f"""
                        SELECT col.name AS column_name
                        FROM sys.indexes idx
                        INNER JOIN sys.index_columns ic ON idx.object_id = ic.object_id AND idx.index_id = ic.index_id
                        INNER JOIN sys.columns col ON ic.object_id = col.object_id AND ic.column_id = col.column_id
                        WHERE idx.object_id = OBJECT_ID('{schema}.{table_name}')
                            AND idx.is_primary_key = 1
                        ORDER BY ic.key_ordinal
                    """
                    )
                    pk = [r["column_name"] for r in cursor.fetchall()]

                    # Get foreign keys
                    cursor.execute(
                        f"""
                        SELECT
                            fk.name AS constraint_name,
                            COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS column_name,
                            OBJECT_SCHEMA_NAME(fkc.referenced_object_id) + '.' + OBJECT_NAME(fkc.referenced_object_id) AS references_table,
                            COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS references_column
                        FROM sys.foreign_keys fk
                        INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
                        WHERE fk.parent_object_id = OBJECT_ID('{schema}.{table_name}')
                    """
                    )

                    fk_map: Dict[str, dict] = {}
                    for r in cursor.fetchall():
                        name = r["constraint_name"]
                        if name not in fk_map:
                            fk_map[name] = {
                                "name": name,
                                "columns": [],
                                "references_table": r["references_table"],
                                "references_columns": [],
                            }
                        fk_map[name]["columns"].append(r["column_name"])
                        fk_map[name]["references_columns"].append(
                            r["references_column"]
                        )
                    fks = list(fk_map.values())

                    # Get indexes (non-primary key)
                    cursor.execute(
                        f"""
                        SELECT
                            idx.name AS index_name,
                            idx.is_unique,
                            STRING_AGG(col.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS columns
                        FROM sys.indexes idx
                        INNER JOIN sys.index_columns ic ON idx.object_id = ic.object_id AND idx.index_id = ic.index_id
                        INNER JOIN sys.columns col ON ic.object_id = col.object_id AND ic.column_id = col.column_id
                        WHERE idx.object_id = OBJECT_ID('{schema}.{table_name}')
                            AND idx.is_primary_key = 0
                            AND idx.name IS NOT NULL
                        GROUP BY idx.name, idx.is_unique
                    """
                    )
                    indexes = [
                        {
                            "name": r["index_name"],
                            "unique": r["is_unique"],
                            "columns": r["columns"].split(",") if r["columns"] else [],
                        }
                        for r in cursor.fetchall()
                    ]

                    result_tables.append(
                        TableSchemaInfo(
                            table_schema=schema,
                            table_name=table_name,
                            full_name=full_name,
                            table_type=table_type,
                            columns=columns,
                            primary_key=pk,
                            foreign_keys=fks,
                            indexes=indexes,
                            row_count_estimate=(
                                int(row_estimate) if row_estimate else None
                            ),
                        )
                    )

                conn.close()
                return result_tables
            finally:
                if tunnel:
                    tunnel.stop()

        tables = await asyncio.to_thread(run_introspection)
        logger.info(f"Introspected {len(tables)} tables/views from MSSQL")

        # Report final progress (MSSQL introspection runs synchronously in one batch)
        if on_progress and tables:
            await on_progress(len(tables), len(tables), tables[-1].full_name)

        return tables

    async def _introspect_mysql(
        self,
        connection_config: dict,
        on_progress: "IntrospectionCallback | None" = None,
    ) -> List[TableSchemaInfo]:
        """Introspect MySQL/MariaDB database schema."""
        try:
            import pymysql
            import pymysql.cursors
        except ImportError:
            raise RuntimeError("pymysql not installed for MySQL schema introspection")

        host = connection_config.get("host", "")
        port = connection_config.get("port", 3306)
        user = connection_config.get("user", "")
        password = connection_config.get("password", "")
        database = connection_config.get("database", "")

        # Build SSH tunnel config dict from top-level ssh_tunnel_* fields
        tunnel_cfg_dict = build_ssh_tunnel_config(connection_config, host, port)

        tables: List[TableSchemaInfo] = []

        def run_introspection() -> List[TableSchemaInfo]:
            tunnel: SSHTunnel | None = None
            actual_host = host
            actual_port = port

            try:
                # Set up SSH tunnel if configured
                if tunnel_cfg_dict:
                    tunnel_cfg = ssh_tunnel_config_from_dict(
                        tunnel_cfg_dict, default_remote_port=3306
                    )
                    if tunnel_cfg:
                        tunnel = SSHTunnel(tunnel_cfg)
                        local_port = tunnel.start()
                        actual_host = "127.0.0.1"
                        actual_port = local_port
                        logger.debug(
                            f"SSH tunnel established: localhost:{local_port} -> "
                            f"{tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}"
                        )

                conn = pymysql.connect(
                    host=actual_host,
                    port=actual_port,
                    user=user,
                    password=password,
                    database=database,
                    cursorclass=pymysql.cursors.DictCursor,
                    connect_timeout=30,
                    read_timeout=60,
                    charset="utf8mb4",
                )
                cursor = conn.cursor()

                # Get all tables and views
                cursor.execute(
                    """
                    SELECT
                        TABLE_SCHEMA AS table_schema,
                        TABLE_NAME AS table_name,
                        TABLE_TYPE AS table_type,
                        TABLE_ROWS AS row_estimate
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = %s
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                    """,
                    (database,),
                )

                table_rows = cursor.fetchall()
                result_tables: List[TableSchemaInfo] = []

                for row in table_rows:
                    schema = row["table_schema"]
                    table_name = row["table_name"]
                    table_type_raw = row["table_type"]
                    row_estimate = row.get("row_estimate")
                    full_name = f"{schema}.{table_name}"

                    # Normalize table type
                    table_type = "VIEW" if "VIEW" in table_type_raw.upper() else "TABLE"

                    # Get columns
                    cursor.execute(
                        """
                        SELECT
                            COLUMN_NAME AS column_name,
                            COLUMN_TYPE AS data_type,
                            IS_NULLABLE AS is_nullable,
                            COLUMN_DEFAULT AS column_default,
                            COLUMN_KEY AS column_key,
                            EXTRA AS extra,
                            COLUMN_COMMENT AS column_comment
                        FROM information_schema.COLUMNS
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                        ORDER BY ORDINAL_POSITION
                        """,
                        (schema, table_name),
                    )
                    columns = [
                        {
                            "name": c["column_name"],
                            "type": c["data_type"],
                            "nullable": c["is_nullable"] == "YES",
                            "default": c["column_default"],
                            "comment": c.get("column_comment") or None,
                        }
                        for c in cursor.fetchall()
                    ]

                    # Get primary key columns
                    cursor.execute(
                        """
                        SELECT COLUMN_NAME
                        FROM information_schema.KEY_COLUMN_USAGE
                        WHERE TABLE_SCHEMA = %s
                          AND TABLE_NAME = %s
                          AND CONSTRAINT_NAME = 'PRIMARY'
                        ORDER BY ORDINAL_POSITION
                        """,
                        (schema, table_name),
                    )
                    pk = [r["COLUMN_NAME"] for r in cursor.fetchall()]

                    # Get foreign keys
                    cursor.execute(
                        """
                        SELECT
                            kcu.CONSTRAINT_NAME AS constraint_name,
                            kcu.COLUMN_NAME AS column_name,
                            kcu.REFERENCED_TABLE_SCHEMA AS ref_schema,
                            kcu.REFERENCED_TABLE_NAME AS ref_table,
                            kcu.REFERENCED_COLUMN_NAME AS ref_column
                        FROM information_schema.KEY_COLUMN_USAGE kcu
                        WHERE kcu.TABLE_SCHEMA = %s
                          AND kcu.TABLE_NAME = %s
                          AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
                        ORDER BY kcu.CONSTRAINT_NAME, kcu.ORDINAL_POSITION
                        """,
                        (schema, table_name),
                    )

                    fk_map: Dict[str, dict] = {}
                    for r in cursor.fetchall():
                        name = r["constraint_name"]
                        ref_table = f"{r['ref_schema']}.{r['ref_table']}"
                        if name not in fk_map:
                            fk_map[name] = {
                                "name": name,
                                "columns": [],
                                "references_table": ref_table,
                                "references_columns": [],
                            }
                        fk_map[name]["columns"].append(r["column_name"])
                        fk_map[name]["references_columns"].append(r["ref_column"])
                    fks = list(fk_map.values())

                    # Get indexes (non-primary key)
                    cursor.execute(
                        """
                        SELECT
                            INDEX_NAME AS index_name,
                            NON_UNIQUE AS non_unique,
                            GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) AS columns
                        FROM information_schema.STATISTICS
                        WHERE TABLE_SCHEMA = %s
                          AND TABLE_NAME = %s
                          AND INDEX_NAME != 'PRIMARY'
                        GROUP BY INDEX_NAME, NON_UNIQUE
                        """,
                        (schema, table_name),
                    )
                    indexes = [
                        {
                            "name": r["index_name"],
                            "unique": r["non_unique"] == 0,
                            "columns": r["columns"].split(",") if r["columns"] else [],
                        }
                        for r in cursor.fetchall()
                    ]

                    result_tables.append(
                        TableSchemaInfo(
                            table_schema=schema,
                            table_name=table_name,
                            full_name=full_name,
                            table_type=table_type,
                            columns=columns,
                            primary_key=pk,
                            foreign_keys=fks,
                            indexes=indexes,
                            row_count_estimate=(
                                int(row_estimate) if row_estimate else None
                            ),
                        )
                    )

                conn.close()
                return result_tables

            finally:
                if tunnel:
                    try:
                        tunnel.stop()
                    except Exception:
                        pass

        tables = await asyncio.to_thread(run_introspection)
        logger.info(f"Introspected {len(tables)} tables/views from MySQL/MariaDB")

        # Report final progress (MySQL introspection runs synchronously in one batch)
        if on_progress and tables:
            await on_progress(len(tables), len(tables), tables[-1].full_name)

        return tables

    # =========================================================================
    # Embedding Generation and Storage
    # =========================================================================

    async def _process_index(
        self,
        job: SchemaIndexJob,
        tool_type: str,
        connection_config: dict,
        full_reindex: bool,
    ):
        """Main processing loop for schema indexing."""
        try:
            # Update job status
            job.status = SchemaIndexStatus.INDEXING
            job.started_at = datetime.now(timezone.utc)
            await self._update_job(job)

            # Early connection test - fail fast if credentials are wrong
            logger.info(f"Testing {tool_type} connection...")
            success, error_msg = await self.test_connection(
                tool_type, connection_config
            )
            if not success:
                raise RuntimeError(f"Database connection failed: {error_msg}")

            # Ensure pgvector is available
            if not await self._ensure_pgvector():
                raise RuntimeError("pgvector extension not available")

            # Get app settings for embedding configuration
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

            embeddings = await self._get_embeddings(settings)

            if embeddings is None:
                raise RuntimeError(
                    "No embedding provider configured. Configure an embedding provider in Settings."
                )

            # Check embedding dimension and ensure column matches
            # Timeout after 60 seconds to prevent indefinite hangs
            try:
                test_embedding = await asyncio.wait_for(
                    asyncio.to_thread(embeddings.embed_documents, ["test"]),
                    timeout=60.0,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "Embedding provider timed out after 60 seconds. "
                    "Check that the embedding service is running and responsive."
                )
            embedding_dim = len(test_embedding[0])
            index_lists = settings.ivfflat_lists or 100
            # This will raise RuntimeError with detailed message if it fails
            await self._ensure_embedding_column(embedding_dim, index_lists)

            # Update tracking if needed (first index or config change)
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

            # Introspect database schema with progress reporting
            logger.info(f"Introspecting {tool_type} schema...")
            job.status_detail = "Discovering tables..."

            async def on_introspection_progress(
                total: int, introspected: int, table_name: str
            ):
                job.total_tables = total
                job.introspected_tables = introspected
                if introspected == 0:
                    job.status_detail = f"Discovered {total} tables, introspecting..."
                elif introspected < total:
                    job.status_detail = (
                        f"Introspecting {introspected}/{total}: {table_name}"
                    )
                else:
                    job.status_detail = f"Introspected {total} tables"
                await self._update_job(job)

            tables = await self.introspect_schema(
                tool_type, connection_config, on_progress=on_introspection_progress
            )

            if not tables:
                job.status = SchemaIndexStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                job.error_message = "No tables found in database"
                await self._update_job(job)
                return

            job.total_tables = len(tables)
            job.introspected_tables = len(tables)
            job.status_detail = None  # Clear detail for embedding phase
            await self._update_job(job)

            # Check for cancellation
            if self._cancellation_flags.get(job.id, False):
                job.status = SchemaIndexStatus.CANCELLED
                job.completed_at = datetime.now(timezone.utc)
                await self._update_job(job)
                return

            # Compute schema hash for change detection
            schema_hash = self._compute_schema_hash(tables)

            # Check if schema has changed (skip if same hash and not full reindex)
            if not full_reindex:
                stored_hash = connection_config.get("schema_hash")
                if stored_hash == schema_hash:
                    logger.info("Schema unchanged, skipping re-index")
                    job.status = SchemaIndexStatus.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
                    job.processed_tables = len(tables)
                    await self._update_job(job)
                    return

            # Clear existing embeddings for this index
            await self._clear_embeddings(job.index_name)

            # Generate embeddings for each table
            for table in tables:
                if self._cancellation_flags.get(job.id, False):
                    job.status = SchemaIndexStatus.CANCELLED
                    job.completed_at = datetime.now(timezone.utc)
                    await self._update_job(job)
                    return

                try:
                    # Convert table to embedding text
                    content = table.to_embedding_text()

                    # Generate embedding with 120s timeout per table
                    try:
                        embedding = await asyncio.wait_for(
                            asyncio.to_thread(embeddings.embed_documents, [content]),
                            timeout=120.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Embedding timed out for table {table.full_name}, skipping"
                        )
                        continue

                    # Store in database
                    await self._insert_embedding(
                        index_name=job.index_name,
                        table_name=table.full_name,
                        table_schema=table.table_schema,
                        content=content,
                        embedding=embedding[0],
                        metadata={
                            "table_type": table.table_type,
                            "column_count": len(table.columns),
                            "has_pk": len(table.primary_key) > 0,
                            "fk_count": len(table.foreign_keys),
                            "index_count": len(table.indexes),
                            "row_estimate": table.row_count_estimate,
                        },
                    )

                    job.processed_tables += 1
                    job.processed_chunks += 1
                    job.total_chunks = job.total_tables  # One chunk per table

                except Exception as e:
                    logger.warning(f"Error processing table {table.full_name}: {e}")

                # Update progress for every table to keep UI responsive
                await self._update_job(job)

            # Update schema hash in connection config
            await self._update_schema_hash(job.tool_config_id, schema_hash)

            # Mark complete
            job.status = SchemaIndexStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            await self._update_job(job)

            logger.info(
                f"Schema indexing completed: {job.processed_tables} tables indexed"
            )

        except Exception as e:
            logger.error(f"Schema indexing failed: {e}", exc_info=True)
            job.status = SchemaIndexStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            try:
                await self._update_job(job)
            except Exception as db_err:
                logger.warning(
                    f"Job {job.id}: Could not update job status (database may be disconnected): {db_err}"
                )

        finally:
            # Clean up in-memory tracking
            self._active_jobs.pop(job.id, None)
            self._active_tasks.pop(job.id, None)
            self._cancellation_flags.pop(job.id, None)

    async def _get_embeddings(self, settings):
        return await get_embeddings_model(
            settings,
            allow_missing_api_key=True,
            return_none_on_error=True,
            logger_override=logger,
        )

    async def _ensure_pgvector(self) -> bool:
        return await ensure_pgvector_extension(logger_override=logger)

    async def _ensure_embedding_column(
        self, embedding_dim: int = 1536, index_lists: int = 100
    ) -> bool:
        """Ensure the embedding column exists with the correct dimension.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_lists: IVFFlat lists parameter (higher = slower build, faster query)
        """
        return await ensure_embedding_column(
            table_name="schema_embeddings",
            index_name="schema_embeddings_embedding_idx",
            embedding_dim=embedding_dim,
            index_lists=index_lists,
            logger_override=logger,
        )

    async def _clear_embeddings(self, index_name: str):
        """Clear all embeddings for an index."""
        try:
            db = await get_db()
            await db.execute_raw(
                f"DELETE FROM schema_embeddings WHERE index_name = '{index_name}'"
            )
        except Exception as e:
            logger.warning(f"Error clearing embeddings: {e}")

    async def _insert_embedding(
        self,
        index_name: str,
        table_name: str,
        table_schema: str,
        content: str,
        embedding: List[float],
        metadata: dict,
    ):
        """Insert a schema embedding into the database."""
        try:
            db = await get_db()
            embedding_id = str(uuid.uuid4())
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            escaped_content = content.replace("'", "''")
            metadata_json = json.dumps(metadata).replace("'", "''")

            await db.execute_raw(
                f"""
                INSERT INTO schema_embeddings
                (id, index_name, table_name, table_schema, content, metadata, embedding, created_at)
                VALUES (
                    '{embedding_id}',
                    '{index_name}',
                    '{table_name}',
                    '{table_schema}',
                    '{escaped_content}',
                    '{metadata_json}'::jsonb,
                    '{embedding_str}'::vector,
                    NOW()
                )
                ON CONFLICT (index_name, table_name) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
            """
            )
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            raise

    def _compute_schema_hash(self, tables: List[TableSchemaInfo]) -> str:
        """Compute a hash of the schema for change detection."""
        schema_data = []
        for table in sorted(tables, key=lambda t: t.full_name):
            schema_data.append(
                {
                    "name": table.full_name,
                    "type": table.table_type,
                    "columns": table.columns,
                    "pk": table.primary_key,
                    "fks": table.foreign_keys,
                }
            )
        schema_json = json.dumps(schema_data, sort_keys=True)
        return hashlib.sha256(schema_json.encode()).hexdigest()[:16]

    async def _update_schema_hash(self, tool_config_id: str, schema_hash: str):
        """Update the schema hash and timestamp in the tool config."""
        try:
            # Use repository to get decrypted config and properly re-encrypt on update
            tool_config = await repository.get_tool_config(tool_config_id)
            if tool_config and tool_config.connection_config:
                connection_config = dict(tool_config.connection_config)
                connection_config["schema_hash"] = schema_hash
                connection_config["last_schema_indexed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                await repository.update_tool_config(
                    tool_config_id, {"connection_config": connection_config}
                )
        except Exception as e:
            logger.warning(f"Error updating schema hash: {e}")

    # =========================================================================
    # Job Persistence
    # =========================================================================

    async def _create_job(self, job: SchemaIndexJob):
        """Create a job record in the database."""
        try:
            db = await get_db()
            await db.schemaindexjob.create(
                data={
                    "id": job.id,
                    "toolConfigId": job.tool_config_id,
                    "status": job.status.value,
                    "indexName": job.index_name,
                    "totalTables": job.total_tables,
                    "processedTables": job.processed_tables,
                    "totalChunks": job.total_chunks,
                    "processedChunks": job.processed_chunks,
                    "createdAt": job.created_at,
                }
            )
        except Exception as e:
            logger.error(f"Error creating schema job: {e}")

    async def _update_job(self, job: SchemaIndexJob):
        """Update a job record in the database."""
        try:
            db = await get_db()
            await db.schemaindexjob.update(
                where={"id": job.id},
                data={
                    "status": job.status.value,
                    "totalTables": job.total_tables,
                    "processedTables": job.processed_tables,
                    "totalChunks": job.total_chunks,
                    "processedChunks": job.processed_chunks,
                    "errorMessage": job.error_message,
                    "startedAt": job.started_at,
                    "completedAt": job.completed_at,
                },
            )
        except Exception as e:
            logger.warning(f"Error updating schema job: {e}")

    async def _get_job_from_db(self, job_id: str) -> Optional[SchemaIndexJobResponse]:
        """Get a job from the database."""
        try:
            db = await get_db()
            prisma_job = await db.schemaindexjob.find_unique(where={"id": job_id})
            if prisma_job:
                job = self._prisma_job_to_model(prisma_job)
                return self.job_to_response(job)
        except Exception as e:
            logger.warning(f"Error getting job from db: {e}")
        return None

    def _prisma_job_to_model(self, prisma_job) -> SchemaIndexJob:
        """Convert Prisma job to model."""
        return SchemaIndexJob(
            id=prisma_job.id,
            tool_config_id=prisma_job.toolConfigId,
            status=SchemaIndexStatus(prisma_job.status),
            index_name=prisma_job.indexName,
            total_tables=prisma_job.totalTables,
            processed_tables=prisma_job.processedTables,
            total_chunks=prisma_job.totalChunks,
            processed_chunks=prisma_job.processedChunks,
            error_message=prisma_job.errorMessage,
            created_at=prisma_job.createdAt,
            started_at=prisma_job.startedAt,
            completed_at=prisma_job.completedAt,
        )

    @staticmethod
    def job_to_response(
        job: SchemaIndexJob, cancel_requested_override: bool | None = None
    ) -> SchemaIndexJobResponse:
        """Convert a SchemaIndexJob to a SchemaIndexJobResponse."""
        return SchemaIndexJobResponse(
            id=job.id,
            tool_config_id=job.tool_config_id,
            status=job.status,
            index_name=job.index_name,
            progress_percent=job.progress_percent,
            total_tables=job.total_tables,
            processed_tables=job.processed_tables,
            introspected_tables=job.introspected_tables,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            error_message=job.error_message,
            cancel_requested=(
                cancel_requested_override
                if cancel_requested_override is not None
                else job.cancel_requested
            ),
            status_detail=job.status_detail,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )


# Global singleton
schema_indexer = SchemaIndexerService()


# =============================================================================
# Search Functions
# =============================================================================


async def search_schema_index(
    query: str,
    index_name: str,
    max_results: int = 5,
) -> str:
    """
    Search schema embeddings for relevant table information.

    Args:
        query: Natural language query about database schema
        index_name: The schema index name (usually 'schema_{tool_config_id}')
        max_results: Maximum number of results to return

    Returns:
        Formatted string with matching table schemas
    """
    try:
        # Get embedding model
        settings = await repository.get_settings()
        if settings is None:
            return "Error: Application settings not configured"

        embeddings = await get_embeddings_model(
            settings,
            return_none_on_error=True,
            logger_override=logger,
        )

        if embeddings is None:
            return "Error: No embedding provider configured. Please configure OpenAI or Ollama in Settings."

        # Generate query embedding
        query_embedding = await asyncio.to_thread(embeddings.embed_documents, [query])
        embedding = query_embedding[0]

        # Search using centralized pgvector search
        results = await search_pgvector_embeddings(
            table_name="schema_embeddings",
            query_embedding=embedding,
            index_name=index_name,
            max_results=max_results,
            columns=SCHEMA_COLUMNS,
            logger_override=logger,
        )

        if not results:
            return "No matching tables found in schema index."

        output_parts = []
        for result in results:
            similarity = result.get("similarity") or 0.0
            content = result.get("content") or ""
            table_name = result.get("table_name") or "unknown"

            output_parts.append(
                f"[{table_name}] (similarity: {similarity:.3f})\n{content}"
            )

        return "\n\n---\n\n".join(output_parts)

    except Exception as e:
        logger.error(f"Error searching schema index: {e}")
        return f"Error searching schema: {str(e)}"
