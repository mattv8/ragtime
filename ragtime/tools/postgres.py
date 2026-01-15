"""
PostgreSQL query tool for executing read-only SQL queries.
"""

import asyncio
import re
import subprocess

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger
from ragtime.core.security import sanitize_output, validate_sql_query
from ragtime.core.sql_utils import add_table_metadata_to_psql_output

logger = get_logger(__name__)


class PostgresQueryInput(BaseModel):
    """Input schema for PostgreSQL queries."""

    query: str = Field(
        description="SQL SELECT query to execute. Must be read-only and include LIMIT clause. "
        "For JSONB name fields, use (column->>'en_US') syntax."
    )
    description: str = Field(
        default="",
        description="Brief description of what this query retrieves (for logging)",
        alias="reason",
    )

    model_config = {"populate_by_name": True}


async def execute_postgres_query(query: str, description: str) -> str:
    """
    Execute a read-only SQL query against PostgreSQL.
    Runs asynchronously to avoid blocking the event loop.

    Args:
        query: SQL query to execute.
        description: Brief description for logging purposes.

    Returns:
        String output from the PostgreSQL query.
    """
    logger.info(f"PostgreSQL Query: {description}")
    logger.debug(f"SQL: {query[:200]}...")

    # Get settings from database
    app_settings = await get_app_settings()
    max_query_results = app_settings["max_query_results"]
    query_timeout = app_settings["query_timeout"]
    postgres_host = app_settings["postgres_host"]
    postgres_port = app_settings["postgres_port"]
    postgres_user = app_settings["postgres_user"]
    postgres_password = app_settings["postgres_password"]
    postgres_db = app_settings["postgres_db"]
    postgres_container = app_settings["postgres_container"]

    # Validate the query
    is_safe, reason = validate_sql_query(query)
    if not is_safe:
        error_msg = f"Security validation failed: {reason}"
        logger.warning(error_msg)
        return f"Error: {error_msg}"

    # Enforce max results by modifying LIMIT if needed
    limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
    if limit_match:
        current_limit = int(limit_match.group(1))
        if current_limit > max_query_results:
            query = re.sub(
                r"LIMIT\s+\d+", f"LIMIT {max_query_results}", query, flags=re.IGNORECASE
            )
            logger.info(f"Reduced LIMIT from {current_limit} to {max_query_results}")

    # Escape single quotes in query for shell
    escaped_query = query.replace("'", "'\\''")

    # Build command based on configuration
    if postgres_host:
        # Direct connection using psql
        cmd = [
            "psql",
            "-h",
            postgres_host,
            "-p",
            str(postgres_port),
            "-U",
            postgres_user,
            "-d",
            postgres_db,
            "-c",
            query,
        ]
        env = {"PGPASSWORD": postgres_password}
    else:
        # Docker exec method
        cmd = [
            "docker",
            "exec",
            "-i",
            postgres_container,
            "bash",
            "-c",
            f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \'{escaped_query}\'',
        ]
        env = None

    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            ),
            timeout=query_timeout,
        )
        stdout, stderr = await process.communicate()

        output = stdout.decode("utf-8", errors="replace").strip()
        errors = stderr.decode("utf-8", errors="replace").strip()

        if process.returncode != 0:
            logger.error(f"PostgreSQL error: {errors}")
            return f"Error executing query: {errors}"

        logger.debug(f"Result rows: {output.count(chr(10))}")
        if output:
            # Add table metadata for UI rendering, then sanitize
            output_with_metadata = add_table_metadata_to_psql_output(output)
            return sanitize_output(output_with_metadata)
        return "Query executed successfully (no results)"

    except asyncio.TimeoutError:
        logger.error(f"PostgreSQL query timed out after {query_timeout}s")
        return f"Error: Query timed out after {query_timeout} seconds"
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        return "Error: Required command (docker or psql) not available"
    except Exception as e:
        logger.exception("Unexpected error in PostgreSQL query")
        return f"Error: {str(e)}"


# Create LangChain tool
postgres_tool = StructuredTool.from_function(
    coroutine=execute_postgres_query,
    name="postgres_query",
    description="""Execute raw SQL queries against the PostgreSQL database.
Use this for:
- Complex aggregations and reports
- Cross-table analytics
- Performance-critical queries

IMPORTANT:
- Only SELECT queries allowed (read-only)
- MUST include LIMIT clause
- For JSONB name columns, use: (name->>'en_US')
- Tables use snake_case (sale_order, res_partner, etc.)

Example: SELECT id, name, amount_total FROM sale_order WHERE state = 'sale' LIMIT 20;
""",
    args_schema=PostgresQueryInput,
)
