"""
Shared utilities for SQL database tools (MSSQL, MySQL, etc.).

Consolidates common patterns for connection, query execution, testing,
and schema generation across database tool implementations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import format_query_result
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict
from ragtime.core.tool_timeouts import resolve_effective_tool_timeout
from ragtime.tools._query_helpers import validate_and_prepare_query
from ragtime.tools.descriptions import format_tool_write_access_sentence

logger = get_logger(__name__)

# Maximum timeout for any tool execution (5 minutes)
# AI can request up to this limit; configured per-tool timeout is the default
MAX_TOOL_TIMEOUT_SECONDS = 300


def resolve_effective_timeout(requested_timeout: int | None, timeout_max_seconds: int) -> int:
    """Resolve runtime timeout using per-tool max (0 max = unlimited)."""
    return resolve_effective_tool_timeout(requested_timeout, timeout_max_seconds)


def create_query_input_schema(
    db_name: str,
    identifier_example: str,
    limit_clause_example: str,
    default_timeout: int,
    timeout_max_seconds: int,
) -> type[BaseModel]:
    """
    Create a Pydantic input schema for SQL query tools with database-specific help text.

    Args:
        db_name: Database name (e.g., "MSSQL", "MySQL").
        identifier_example: Example of database-specific identifier format.
        limit_clause_example: Example of result limit syntax.
        default_timeout: Default query timeout in seconds.
        timeout_max_seconds: Maximum timeout allowed.

    Returns:
        A Pydantic BaseModel class for query input validation.
    """
    timeout_hint = "Use 0 for no timeout." if timeout_max_seconds == 0 else "Use 0 or omit to use the configured maximum."

    class QueryInput(BaseModel):
        """Input schema for SQL queries."""

        query: str = Field(description=(f"SQL SELECT query to execute. Must be read-only. {limit_clause_example} For {db_name}, use {identifier_example}"))
        description: str = Field(
            default="",
            description="Brief description of what this query retrieves (for logging)",
            alias="reason",
        )
        timeout: int = Field(
            default=default_timeout,
            ge=0,
            le=max(timeout_max_seconds, MAX_TOOL_TIMEOUT_SECONDS),
            description=(
                f"Query timeout in seconds (default: {default_timeout}, "
                f"max: {'unlimited' if timeout_max_seconds == 0 else timeout_max_seconds}). "
                f"{timeout_hint}"
            ),
        )

        model_config = {"populate_by_name": True}

    return QueryInput


async def execute_sql_query_async(
    query: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    db_type: str,
    connect_fn: Callable[..., Any],
    timeout: int = 30,
    max_results: int = 100,
    allow_write: bool = False,
    require_result_limit: bool = True,
    enforce_result_limit: bool = True,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
    metadata_max_length: int | None = 30000,
    max_output_length: int | None = 50000,
    include_ascii: bool = True,
    default_remote_port: int = 3306,
    error_formatter: Callable[[Exception, str | None], str] | None = None,
) -> str:
    """
    Execute a read-only SQL query against a database.

    Runs in a thread pool to avoid blocking the event loop.

    Args:
        query: SQL query to execute.
        host: Database server hostname.
        port: Database server port.
        user: Database username.
        password: Database password.
        database: Database name.
        db_type: Database type constant (e.g., DB_TYPE_MYSQL, DB_TYPE_MSSQL).
        connect_fn: Callable that returns a database connection given (host, port, user, password, database, timeout).
        timeout: Query timeout in seconds.
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        require_result_limit: Whether LIMIT/TOP is required for SELECT.
        enforce_result_limit: Whether existing LIMIT/TOP clauses are capped to max_results.
        description: Brief description for logging purposes.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.
        metadata_max_length: Maximum embedded JSON metadata length, or None for no cap.
        max_output_length: Maximum formatted output length, or None for no truncation.
        include_ascii: Whether to include a readable ASCII fallback table.
        error_formatter: Optional callable that takes (exception, database) and returns
            a user-facing error string. When None, raw exception strings are surfaced.
        default_remote_port: Default remote port for SSH tunnel (e.g., 3306 for MySQL).

    Returns:
        String output from the query.
    """
    logger.info(f"SQL Query ({db_type}): {description}")
    logger.debug(f"SQL: {query[:200]}...")

    is_safe, prepared_query = validate_and_prepare_query(
        query,
        max_results=max_results,
        allow_write=allow_write,
        db_type=db_type,
        require_result_limit=require_result_limit,
        enforce_result_limit=enforce_result_limit,
    )
    if not is_safe:
        error_msg = f"Security validation failed: {prepared_query}"
        logger.warning(error_msg)
        return f"Error: {error_msg}"
    query = prepared_query

    def run_query() -> str:
        """Execute query in thread pool."""
        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=default_remote_port)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = connect_fn(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch results
            rows = cursor.fetchall()

            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else None

            if not rows and not columns:
                return "Query executed successfully (no results)"

            return format_query_result(
                rows if isinstance(rows, list) else list(rows),
                columns,
                include_metadata=include_metadata,
                metadata_max_length=metadata_max_length,
                max_output_length=max_output_length,
                include_ascii=include_ascii,
            )

        except Exception as e:
            logger.exception(f"Unexpected error in SQL query ({db_type})")
            if error_formatter:
                return error_formatter(e, database)
            return f"Error: {str(e)}"

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if tunnel:
                try:
                    tunnel.stop()
                except Exception:
                    pass

    try:
        if timeout > 0:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, run_query),
                timeout=timeout + 5,  # Allow extra time for connection
            )
            return result

        result = await asyncio.get_event_loop().run_in_executor(None, run_query)
        return result

    except asyncio.TimeoutError:
        logger.error(f"SQL query timed out after {timeout}s ({db_type})")
        return f"Error: Query timed out after {timeout} seconds"


def create_sql_tool(
    name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    db_type: str,
    db_display_name: str,
    identifier_example: str,
    limit_clause_hint: str,
    connect_fn: Callable[..., Any],
    timeout: int = 30,
    timeout_max_seconds: int = MAX_TOOL_TIMEOUT_SECONDS,
    max_results: int = 100,
    allow_write: bool = False,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
    default_remote_port: int = 3306,
    error_formatter: Callable[[Exception, str | None], str] | None = None,
) -> StructuredTool:
    """
    Create a configured SQL query tool for LangChain.

    Args:
        name: Tool name (used in LangChain agent).
        host: Database server hostname.
        port: Database server port.
        user: Database username.
        password: Database password.
        database: Database name.
        db_type: Database type constant (e.g., DB_TYPE_MYSQL, DB_TYPE_MSSQL).
        db_display_name: Display name for tool description (e.g., "MySQL/MariaDB").
        identifier_example: Database-specific identifier format example (e.g.,
            "backticks for identifiers: `table_name`.`column_name`").
        limit_clause_hint: Display hint for result limit syntax (e.g., "SELECT TOP 100 ..." or "SELECT ... LIMIT 100").
        connect_fn: Callable that returns a database connection.
        timeout: Query timeout in seconds (default for AI).
        timeout_max_seconds: Maximum timeout allowed.
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        description: Description for LLM context.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.
        include_metadata: Whether to include result metadata.
        default_remote_port: Default remote port for SSH tunnel.
        error_formatter: Optional callable (exception, database) -> user-facing error string.

    Returns:
        Configured StructuredTool instance.
    """
    # Create input schema with this tool's default timeout
    QueryInput = create_query_input_schema(
        db_name=db_display_name,
        identifier_example=identifier_example,
        limit_clause_example=limit_clause_hint,
        default_timeout=timeout,
        timeout_max_seconds=timeout_max_seconds,
    )

    async def execute_query(query: str = "", description: str = "", timeout: int = timeout, **_: Any) -> str:
        """Execute SQL query using configured connection."""
        if not query or not query.strip():
            return "Error: 'query' parameter is required. Provide a SQL query to execute."
        if not description:
            description = "SQL query"

        effective_timeout = resolve_effective_timeout(timeout, timeout_max_seconds)

        return await execute_sql_query_async(
            query=query,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            db_type=db_type,
            connect_fn=connect_fn,
            timeout=effective_timeout,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
            default_remote_port=default_remote_port,
            error_formatter=error_formatter,
        )

    tool_description = f"Query the {name} {db_display_name} database using SQL."
    if description:
        tool_description += f" This database contains: {description}"
    tool_description += f" {format_tool_write_access_sentence(allow_write)}"
    tool_description += f" {limit_clause_hint}"

    return StructuredTool.from_function(
        coroutine=execute_query,
        name=f"query_{name.lower().replace(' ', '_').replace('-', '_')}",
        description=tool_description,
        args_schema=QueryInput,
    )


async def test_sql_connection(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    timeout: int,
    ssh_tunnel_config: dict[str, Any] | None,
    connect_fn: Callable[..., Any],
    version_query: str,
    db_name_query: str,
    default_remote_port: int = 3306,
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Test SQL database connection and return status.

    Args:
        host: Database server hostname.
        port: Database server port.
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Connection timeout in seconds.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.
        connect_fn: Callable that returns a database connection.
        version_query: SQL query to retrieve server version.
        db_name_query: SQL query to retrieve database name.
        default_remote_port: Default remote port for SSH tunnel.

    Returns:
        Tuple of (success, message, details).
    """

    def do_test() -> tuple[bool, str, dict[str, Any] | None]:
        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=default_remote_port)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = connect_fn(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()

            # Get server version
            cursor.execute(version_query)
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "Unknown"

            # Get database name
            cursor.execute(db_name_query)
            db_row = cursor.fetchone()
            db_name = db_row[0] if db_row else database

            details: dict[str, Any] = {
                "version": str(version).split("\n")[0] if version else "Unknown",
                "database": db_name,
            }

            if tunnel:
                details["mode"] = "ssh_tunnel"
                details["ssh_host"] = ssh_tunnel_config.get("ssh_tunnel_host", "") if ssh_tunnel_config else ""
            else:
                details["host"] = host
                details["port"] = port

            msg = f"Connected to {str(db_name)} successfully"
            if tunnel:
                msg += " (via SSH tunnel)"

            return True, msg, details

        except Exception as e:
            error_str = str(e)
            logger.error(f"SQL connection test failed: {error_str}")
            if tunnel:
                return False, "Cannot connect to database through SSH tunnel", None
            return False, f"Connection test failed: {error_str}", None

        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            if tunnel:
                try:
                    tunnel.stop()
                except Exception:
                    pass

    try:
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, do_test),
            timeout=timeout + 5,
        )
    except asyncio.TimeoutError:
        return False, f"Connection timed out after {timeout} seconds", None
