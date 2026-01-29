"""
MSSQL (SQL Server) query tool for executing read-only SQL queries.

Uses pymssql for native TDS protocol support (no ODBC required).
Follows the same patterns as the PostgreSQL tool for consistency.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (
    DB_TYPE_MSSQL,
    enforce_max_results,
    format_query_result,
    validate_sql_query,
)
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict

logger = get_logger(__name__)


class MssqlQueryInput(BaseModel):
    """Input schema for MSSQL queries."""

    query: str = Field(
        description=(
            "SQL SELECT query to execute. Must be read-only and include "
            "TOP n clause (e.g., SELECT TOP 100 * FROM table). "
            "For MSSQL, use square brackets for identifiers: [TableName].[ColumnName]"
        )
    )
    description: str = Field(
        default="",
        description="Brief description of what this query retrieves (for logging)",
        alias="reason",
    )

    model_config = {"populate_by_name": True}


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
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
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
        description: Brief description for logging purposes.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        String output from the MSSQL query.
    """
    logger.info(f"MSSQL Query: {description}")
    logger.debug(f"SQL: {query[:200]}...")

    # Validate the query
    is_safe, reason = validate_sql_query(
        query, enable_write=allow_write, db_type=DB_TYPE_MSSQL
    )
    if not is_safe:
        error_msg = f"Security validation failed: {reason}"
        logger.warning(error_msg)
        return f"Error: {error_msg}"

    # Enforce max results
    query = enforce_max_results(query, max_results, db_type=DB_TYPE_MSSQL)

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
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(
                        f"SSH tunnel established: localhost:{local_port} -> "
                        f"{tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}"
                    )

            conn = pymssql.connect(  # type: ignore[attr-defined]
                server=actual_host,
                port=str(actual_port),
                user=user,
                password=password,
                database=database,
                login_timeout=timeout,
                timeout=timeout,
                as_dict=True,
            )
            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch results
            rows = cursor.fetchall()

            # Get column names from cursor description
            columns = (
                [col[0] for col in cursor.description] if cursor.description else None
            )

            if not rows and not columns:
                return "Query executed successfully (no results)"

            return format_query_result(rows, columns, include_metadata=include_metadata)

        except pymssql.OperationalError as e:
            error_str = str(e)
            logger.error(f"MSSQL connection error: {error_str}")
            # Clean up sensitive info from error messages
            if "Login failed" in error_str:
                return "Error: Login failed - check username and password"
            if "Cannot open database" in error_str:
                return f"Error: Cannot open database '{database}' - check database name and permissions"
            if tunnel:
                return f"Error: Cannot connect to SQL Server through SSH tunnel"
            return f"Error: Connection failed - {error_str}"

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
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, run_query),
            timeout=timeout + 5,  # Allow extra time for connection
        )
        return result

    except asyncio.TimeoutError:
        logger.error(f"MSSQL query timed out after {timeout}s")
        return f"Error: Query timed out after {timeout} seconds"


def create_mssql_tool(
    name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    timeout: int = 30,
    max_results: int = 100,
    allow_write: bool = False,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> StructuredTool:
    """
    Create a configured MSSQL query tool for LangChain.

    Args:
        name: Tool name (used in LangChain agent).
        host: MSSQL server hostname.
        port: MSSQL server port.
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Query timeout in seconds.
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        description: Description for LLM context.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        Configured StructuredTool instance.
    """

    async def execute_query(query: str = "", description: str = "", **_: Any) -> str:
        """Execute MSSQL query using configured connection."""
        if not query or not query.strip():
            return (
                "Error: 'query' parameter is required. Provide a SQL query to execute."
            )
        if not description:
            description = "SQL query"

        return await execute_mssql_query_async(
            query=query,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=timeout,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
        )

    tool_description = f"Query the {name} MSSQL/SQL Server database using SQL."
    if description:
        tool_description += f" This database contains: {description}"
    tool_description += (
        " Include TOP n clause to limit results (e.g., SELECT TOP 100 ...). "
        "SELECT queries only unless writes are enabled."
    )

    return StructuredTool.from_function(
        coroutine=execute_query,
        name=f"query_{name.lower().replace(' ', '_').replace('-', '_')}",
        description=tool_description,
        args_schema=MssqlQueryInput,
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
                tunnel_cfg = ssh_tunnel_config_from_dict(
                    ssh_tunnel_config, default_remote_port=1433
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

            conn = pymssql.connect(  # type: ignore[attr-defined]
                server=actual_host,
                port=str(actual_port),
                user=user,
                password=password,
                database=database,
                login_timeout=timeout,
                timeout=timeout,
            )
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
                "version": version.split("\n")[0] if version else "Unknown",
                "database": db_name,
            }

            if tunnel:
                details["mode"] = "ssh_tunnel"
                details["ssh_host"] = (
                    ssh_tunnel_config.get("ssh_tunnel_host", "")
                    if ssh_tunnel_config
                    else ""
                )
            else:
                details["host"] = host
                details["port"] = port

            msg = f"Connected to {db_name} successfully"
            if tunnel:
                msg += " (via SSH tunnel)"

            return True, msg, details

        except pymssql.OperationalError as e:
            error_str = str(e)
            if "Login failed" in error_str:
                return False, "Login failed - check username and password", None
            if "Cannot open database" in error_str:
                return False, f"Cannot open database '{database}'", None
            if tunnel:
                return False, "Cannot connect to SQL Server through SSH tunnel", None
            return False, f"Connection failed: {error_str}", None

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
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, do_test),
            timeout=timeout + 5,
        )
    except asyncio.TimeoutError:
        return False, f"Connection timed out after {timeout} seconds", None


# Legacy tool export (for backwards compatibility if needed)
# Note: Prefer using create_mssql_tool() with ToolConfig for dynamic instances
mssql_tool = None  # Placeholder - instantiate via ToolConfig
