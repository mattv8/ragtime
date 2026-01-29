"""
MySQL/MariaDB query tool for executing read-only SQL queries.

Uses PyMySQL for pure-Python MySQL/MariaDB connectivity.
Follows the same patterns as the MSSQL tool for consistency.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (
    DB_TYPE_MYSQL,
    enforce_max_results,
    format_query_result,
    validate_sql_query,
)
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict

logger = get_logger(__name__)


class MysqlQueryInput(BaseModel):
    """Input schema for MySQL queries."""

    query: str = Field(
        description=(
            "SQL SELECT query to execute. Must be read-only and include "
            "LIMIT clause (e.g., SELECT * FROM table LIMIT 100). "
            "For MySQL, use backticks for identifiers: `table_name`.`column_name`"
        )
    )
    description: str = Field(
        default="",
        description="Brief description of what this query retrieves (for logging)",
        alias="reason",
    )

    model_config = {"populate_by_name": True}


async def execute_mysql_query_async(
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
    Execute a read-only SQL query against MySQL/MariaDB.

    Runs in a thread pool to avoid blocking the event loop.

    Args:
        query: SQL query to execute.
        host: MySQL server hostname.
        port: MySQL server port (default 3306).
        user: Database username.
        password: Database password.
        database: Database name.
        timeout: Query timeout in seconds.
        max_results: Maximum rows to return.
        allow_write: Whether to allow write operations.
        description: Brief description for logging purposes.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        String output from the MySQL query.
    """
    logger.info(f"MySQL Query: {description}")
    logger.debug(f"SQL: {query[:200]}...")

    # Validate the query
    is_safe, reason = validate_sql_query(
        query, enable_write=allow_write, db_type=DB_TYPE_MYSQL
    )
    if not is_safe:
        error_msg = f"Security validation failed: {reason}"
        logger.warning(error_msg)
        return f"Error: {error_msg}"

    # Enforce max results
    query = enforce_max_results(query, max_results, db_type=DB_TYPE_MYSQL)

    def run_query() -> str:
        """Execute query in thread pool."""
        try:
            import pymysql  # type: ignore[import-untyped]
            import pymysql.cursors  # type: ignore[import-untyped]
        except ImportError:
            return "Error: pymysql package not installed. Install with: pip install pymysql"

        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(
                    ssh_tunnel_config, default_remote_port=3306
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
                connect_timeout=timeout,
                read_timeout=timeout,
                write_timeout=timeout,
                charset="utf8mb4",
            )
            cursor = conn.cursor()
            cursor.execute(query)

            # Fetch results
            rows = cursor.fetchall()
            if not rows:
                return "Query executed successfully (no results)"

            # Get column names from cursor description
            columns = (
                [col[0] for col in cursor.description] if cursor.description else None
            )

            return format_query_result(rows, columns, include_metadata=include_metadata)

        except pymysql.OperationalError as e:
            error_str = str(e)
            logger.error(f"MySQL connection error: {error_str}")
            # Clean up sensitive info from error messages
            if "Access denied" in error_str:
                return "Error: Access denied - check username and password"
            if "Unknown database" in error_str:
                return f"Error: Unknown database '{database}' - check database name"
            if "Can't connect" in error_str:
                if tunnel:
                    return f"Error: Cannot connect to MySQL through SSH tunnel"
                return f"Error: Cannot connect to MySQL server at {host}:{port}"
            return f"Error: Connection failed - {error_str}"

        except pymysql.ProgrammingError as e:
            logger.error(f"MySQL query error: {e}")
            return f"Error: Query failed - {e}"

        except Exception as e:
            logger.exception("Unexpected error in MySQL query")
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
        logger.error(f"MySQL query timed out after {timeout}s")
        return f"Error: Query timed out after {timeout} seconds"


def create_mysql_tool(
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
    Create a configured MySQL query tool for LangChain.

    Args:
        name: Tool name (used in LangChain agent).
        host: MySQL server hostname.
        port: MySQL server port.
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
        """Execute MySQL query using configured connection."""
        if not query or not query.strip():
            return (
                "Error: 'query' parameter is required. Provide a SQL query to execute."
            )
        if not description:
            description = "SQL query"

        return await execute_mysql_query_async(
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

    tool_description = f"Query the {name} MySQL/MariaDB database using SQL."
    if description:
        tool_description += f" This database contains: {description}"
    tool_description += (
        " Include LIMIT clause to limit results (e.g., SELECT ... LIMIT 100). "
        "SELECT queries only unless writes are enabled."
    )

    return StructuredTool.from_function(
        coroutine=execute_query,
        name=f"query_{name.lower().replace(' ', '_').replace('-', '_')}",
        description=tool_description,
        args_schema=MysqlQueryInput,
    )


async def test_mysql_connection(
    host: str = "",
    port: int = 3306,
    user: str = "",
    password: str = "",
    database: str = "",
    container: str = "",
    docker_network: str = "",
    timeout: int = 10,
    ssh_tunnel_config: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """
    Test MySQL connection and return status.
    Supports both direct connections and Docker container mode.

    Args:
        host: MySQL server hostname (for direct mode).
        port: MySQL server port (for direct mode).
        user: Database username (for direct mode).
        password: Database password (for direct mode).
        database: Database name.
        container: Docker container name (for container mode).
        docker_network: Docker network name (for container mode).
        timeout: Connection timeout in seconds.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.

    Returns:
        Tuple of (success, message, details).
    """

    def do_test() -> tuple[bool, str, dict[str, Any] | None]:
        try:
            import pymysql  # type: ignore[import-untyped]
        except ImportError:
            return False, "pymysql package not installed", None

        # Docker container mode
        if container:
            import subprocess

            exec_prefix = f"docker exec {container}"

            def get_env_var(var_name: str) -> str | None:
                try:
                    result = subprocess.run(
                        f"{exec_prefix} printenv {var_name}",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    return result.stdout.strip() if result.returncode == 0 else None
                except Exception:
                    return None

            # Get credentials from container environment
            db_user = get_env_var("MYSQL_USER") or "root"
            if db_user == "root":
                db_password = get_env_var("MYSQL_ROOT_PASSWORD") or ""
            else:
                db_password = get_env_var("MYSQL_PASSWORD") or ""
            db_name = database or get_env_var("MYSQL_DATABASE") or ""

            if not db_name:
                return (
                    False,
                    "No database specified and MYSQL_DATABASE not set in container",
                    None,
                )

            # Test via docker exec mysql command
            mysql_cmd = f'{exec_prefix} mysql -u{db_user} -p"{db_password}" -N -e "SELECT VERSION()" {db_name}'
            try:
                result = subprocess.run(
                    mysql_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )

                if result.returncode == 0:
                    version = result.stdout.strip() or "Unknown"
                    details = {
                        "version": version,
                        "database": db_name,
                        "container": container,
                        "mode": "docker",
                    }
                    return (
                        True,
                        f"Connected to {db_name} in container {container}",
                        details,
                    )
                else:
                    error_msg = result.stderr.strip() or "Unknown error"
                    if "Access denied" in error_msg:
                        return (
                            False,
                            "Access denied - check container credentials",
                            None,
                        )
                    if "Unknown database" in error_msg:
                        return False, f"Unknown database '{db_name}'", None
                    return False, f"Connection failed: {error_msg}", None

            except subprocess.TimeoutExpired:
                return False, f"Connection timed out after {timeout}s", None
            except Exception as e:
                return False, f"Docker test failed: {str(e)}", None

        # Direct connection mode (with optional SSH tunnel)
        conn = None
        cursor = None
        tunnel: SSHTunnel | None = None
        actual_host = host
        actual_port = port

        try:
            # Set up SSH tunnel if configured
            if ssh_tunnel_config:
                tunnel_cfg = ssh_tunnel_config_from_dict(
                    ssh_tunnel_config, default_remote_port=3306
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
                connect_timeout=timeout,
                read_timeout=timeout,
            )
            cursor = conn.cursor()

            # Get server version
            cursor.execute("SELECT VERSION()")
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "Unknown"

            # Get database name
            cursor.execute("SELECT DATABASE()")
            db_row = cursor.fetchone()
            db_name = db_row[0] if db_row else database

            details: dict[str, Any] = {
                "version": version,
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

        except pymysql.OperationalError as e:
            error_str = str(e)
            if "Access denied" in error_str:
                return False, "Access denied - check username and password", None
            if "Unknown database" in error_str:
                return False, f"Unknown database '{database}'", None
            if "Can't connect" in error_str:
                if tunnel:
                    return False, "Cannot connect to MySQL through SSH tunnel", None
                return False, f"Cannot connect to MySQL server at {host}:{port}", None
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
# Note: Prefer using create_mysql_tool() with ToolConfig for dynamic instances
mysql_tool = None  # Placeholder - instantiate via ToolConfig
