"""
MSSQL (SQL Server) query tool for executing read-only SQL queries.

Uses pymssql for native TDS protocol support (no ODBC required).
Follows the same patterns as the PostgreSQL tool for consistency.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import DB_TYPE_MSSQL, normalize_mssql_error_message
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict
from ragtime.tools._db_shared import (
    MAX_TOOL_TIMEOUT_SECONDS,
    create_sql_tool,
    resolve_effective_timeout,
)

logger = get_logger(__name__)


# Legacy schema for backwards compatibility (default 30s timeout)
class MssqlQueryInput(BaseModel):
    """Input schema for MSSQL queries."""

    query: str = Field(
        description=(
            "SQL SELECT query to execute. Must be read-only. "
            "Include TOP n clause to limit results (e.g., SELECT TOP 100 ...). "
            "For MSSQL, use square brackets for identifiers: [TableName].[ColumnName]"
        )
    )
    description: str = Field(
        default="",
        description="Brief description of what this query retrieves (for logging)",
        alias="reason",
    )
    timeout: int = Field(
        default=30,
        ge=0,
        le=MAX_TOOL_TIMEOUT_SECONDS,
        description=f"Query timeout in seconds (default: 30, max: {MAX_TOOL_TIMEOUT_SECONDS}). Use 0 for no timeout.",
    )

    model_config = {"populate_by_name": True}


def _mssql_connect(host: str, port: int, user: str, password: str, database: str, timeout: int) -> Any:
    """Create and return a pymssql connection."""
    try:
        import pymssql  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pymssql package not installed. Install with: pip install pymssql")

    return pymssql.connect(  # type: ignore[attr-defined]
        server=host,
        port=str(port),
        user=user,
        password=password,
        database=database,
        login_timeout=timeout,
        timeout=timeout,
        as_dict=True,
    )


async def execute_mssql_query_async(
    query: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
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
) -> str:
    """
    Execute a read-only SQL query against MSSQL Server.

    Runs in a thread pool to avoid blocking the event loop.

    Args:
        query: SQL query to execute.
        host: MSSQL server hostname.
        port: MSSQL server port (default 1433).
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Query timeout in seconds.
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        require_result_limit: Whether TOP/FETCH is required for SELECT.
        enforce_result_limit: Whether existing TOP/FETCH clauses are capped to max_results.
        description: Brief description for logging purposes.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.
        metadata_max_length: Maximum embedded JSON metadata length, or None for no cap.
        max_output_length: Maximum formatted output length, or None for no truncation.
        include_ascii: Whether to include a readable ASCII fallback table.

    Returns:
        String output from the MSSQL query.
    """

    def run_query() -> str:
        """Execute query in thread pool."""
        try:
            import pymssql  # type: ignore[import-untyped]
        except ImportError:
            return "Error: pymssql package not installed. Install with: pip install pymssql"

        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=1433)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = _mssql_connect(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()

            # Use shared query execution
            from ragtime.core.sql_utils import format_query_result
            from ragtime.tools._query_helpers import validate_and_prepare_query

            is_safe, prepared_query = validate_and_prepare_query(
                query,
                max_results=max_results,
                allow_write=allow_write,
                db_type=DB_TYPE_MSSQL,
                require_result_limit=require_result_limit,
                enforce_result_limit=enforce_result_limit,
            )
            if not is_safe:
                error_msg = f"Security validation failed: {prepared_query}"
                logger.warning(error_msg)
                return f"Error: {error_msg}"

            cursor.execute(prepared_query)

            # Fetch results
            rows = cursor.fetchall()

            # Get column names from cursor description
            columns = [col[0] for col in cursor.description] if cursor.description else None

            if not rows and not columns:
                return "Query executed successfully (no results)"

            return format_query_result(
                rows,
                columns,
                include_metadata=include_metadata,
                metadata_max_length=metadata_max_length,
                max_output_length=max_output_length,
                include_ascii=include_ascii,
            )

        except pymssql.OperationalError as e:
            error_str = str(e)
            logger.error(f"MSSQL connection error: {error_str}")
            normalized_error = normalize_mssql_error_message(error_str, database=database)
            if tunnel:
                return f"Error: Cannot connect to SQL Server through SSH tunnel"
            return f"Error: {normalized_error}"

        except pymssql.ProgrammingError as e:
            logger.error(f"MSSQL query error: {e}")
            return f"Error: Query failed - {e}"

        except Exception as e:
            logger.exception("Unexpected error in MSSQL query")
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
        import asyncio

        if timeout > 0:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, run_query),
                timeout=timeout + 5,  # Allow extra time for connection
            )
            return result

        result = await asyncio.get_event_loop().run_in_executor(None, run_query)
        return result

    except asyncio.TimeoutError:
        logger.error(f"MSSQL query timed out after {timeout}s")
        return f"Error: Query timed out after {timeout} seconds"


def _mssql_error_formatter(exc: Exception, database: str | None) -> str:
    """Format MSSQL-specific errors for LLM consumption."""
    try:
        import pymssql  # type: ignore[import-untyped]
    except ImportError:
        return f"Error: {exc}"
    if isinstance(exc, (pymssql.OperationalError, pymssql.ProgrammingError)):
        return f"Error: {normalize_mssql_error_message(str(exc), database=database)}"
    return f"Error: {exc}"


def create_mssql_tool(
    name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    timeout: int = 30,
    timeout_max_seconds: int = MAX_TOOL_TIMEOUT_SECONDS,
    max_results: int = 100,
    allow_write: bool = False,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> Any:
    """
    Create a configured MSSQL query tool for LangChain.

    Args:
        name: Tool name (used in LangChain agent).
        host: MSSQL server hostname.
        port: MSSQL server port.
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Query timeout in seconds (default for AI).
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        description: Description for LLM context.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        Configured StructuredTool instance.
    """
    return create_sql_tool(
        name=name,
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        db_type=DB_TYPE_MSSQL,
        db_display_name="MSSQL/SQL Server",
        identifier_example="square brackets for identifiers: [TableName].[ColumnName]",
        limit_clause_hint="Include TOP n clause to limit results (e.g., SELECT TOP 100 ...).",
        connect_fn=_mssql_connect,
        timeout=timeout,
        timeout_max_seconds=timeout_max_seconds,
        max_results=max_results,
        allow_write=allow_write,
        description=description,
        ssh_tunnel_config=ssh_tunnel_config,
        include_metadata=include_metadata,
        default_remote_port=1433,
        error_formatter=_mssql_error_formatter,
    )


async def test_mssql_connection(
    host: str = "",
    port: int = 1433,
    user: str = "",
    password: str = "",
    database: str = "",
    timeout: int = 10,
    ssh_tunnel_config: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Test MSSQL connection and return status.

    Args:
        host: MSSQL server hostname.
        port: MSSQL server port.
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Connection timeout in seconds.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        Tuple of (success, message, details).
    """

    def do_test() -> tuple[bool, str, dict[str, Any] | None]:
        try:
            import pymssql  # type: ignore[import-untyped]
        except ImportError:
            return False, "pymssql package not installed", None

        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=1433)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = _mssql_connect(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()

            # Get server version
            cursor.execute("SELECT @@VERSION")
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "Unknown"

            # Get database info
            cursor.execute("SELECT DB_NAME() AS db_name")
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

        except pymssql.OperationalError as e:
            error_str = str(e)
            normalized_error = normalize_mssql_error_message(error_str, database=database)
            if tunnel:
                return False, "Cannot connect to SQL Server through SSH tunnel", None
            return False, normalized_error, None

        except Exception as e:
            return False, f"Connection test failed: {str(e)}", None

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
        import asyncio

        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, do_test),
            timeout=timeout + 5,
        )
    except asyncio.TimeoutError:
        return False, f"Connection timed out after {timeout} seconds", None


# Legacy tool export (for backwards compatibility if needed)
# Note: Prefer using create_mssql_tool() with ToolConfig for dynamic instances
mssql_tool = None  # Placeholder - instantiate via ToolConfig
