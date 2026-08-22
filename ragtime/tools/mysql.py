"""
MySQL/MariaDB query tool for executing read-only SQL queries.

Uses PyMySQL for pure-Python MySQL/MariaDB connectivity.
Follows the same patterns as the MSSQL tool for consistency.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import DB_TYPE_MYSQL
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict
from ragtime.tools._db_shared import (
    MAX_TOOL_TIMEOUT_SECONDS,
    create_sql_tool,
    resolve_effective_timeout,
)

logger = get_logger(__name__)


# Legacy schema for backwards compatibility (default 30s timeout)
class MysqlQueryInput(BaseModel):
    """Input schema for MySQL queries."""

    query: str = Field(
        description=(
            "SQL SELECT query to execute. Must be read-only. "
            "Include LIMIT clause to limit results (e.g., SELECT * FROM table LIMIT 100). "
            "For MySQL, use backticks for identifiers: `table_name`.`column_name`"
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


def _mysql_connect(host: str, port: int, user: str, password: str, database: str, timeout: int) -> Any:
    """Create and return a pymysql connection."""
    try:
        import pymysql  # type: ignore[import-untyped]
        import pymysql.cursors  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pymysql package not installed. Install with: pip install pymysql")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=timeout,
        read_timeout=timeout,
        write_timeout=timeout,
        charset="utf8mb4",
    )


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
        require_result_limit: Whether LIMIT is required for SELECT.
        enforce_result_limit: Whether existing LIMIT clauses are capped to max_results.
        description: Brief description for logging purposes.
        ssh_tunnel_config: Optional SSH tunnel configuration dict.
        metadata_max_length: Maximum embedded JSON metadata length, or None for no cap.
        max_output_length: Maximum formatted output length, or None for no truncation.
        include_ascii: Whether to include a readable ASCII fallback table.

    Returns:
        String output from the MySQL query.
    """

    def run_query() -> str:
        """Execute query in thread pool."""
        try:
            import pymysql  # type: ignore[import-untyped]
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
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=3306)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = _mysql_connect(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()

            # Use shared query execution
            from ragtime.core.sql_utils import format_query_result
            from ragtime.tools._query_helpers import validate_and_prepare_query

            is_safe, prepared_query = validate_and_prepare_query(
                query,
                max_results=max_results,
                allow_write=allow_write,
                db_type=DB_TYPE_MYSQL,
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
                list(rows),
                columns,
                include_metadata=include_metadata,
                metadata_max_length=metadata_max_length,
                max_output_length=max_output_length,
                include_ascii=include_ascii,
            )

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
        logger.error(f"MySQL query timed out after {timeout}s")
        return f"Error: Query timed out after {timeout} seconds"


def _mysql_error_formatter(exc: Exception, database: str | None) -> str:
    """Format MySQL-specific errors for LLM consumption."""
    error_str = str(exc)
    if "Access denied" in error_str:
        return "Error: Access denied - check username and password"
    if "Unknown database" in error_str:
        return f"Error: Unknown database '{database}' - check database name"
    if "Can't connect" in error_str:
        return f"Error: Cannot connect to MySQL server"
    return f"Error: Connection failed - {error_str}"


def create_mysql_tool(
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
    Create a configured MySQL query tool for LangChain.

    Args:
        name: Tool name (used in LangChain agent).
        host: MySQL server hostname.
        port: MySQL server port.
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
        db_type=DB_TYPE_MYSQL,
        db_display_name="MySQL/MariaDB",
        identifier_example="backticks for identifiers: `table_name`.`column_name`",
        limit_clause_hint="Include LIMIT clause to limit results (e.g., SELECT ... LIMIT 100).",
        connect_fn=_mysql_connect,
        timeout=timeout,
        timeout_max_seconds=timeout_max_seconds,
        max_results=max_results,
        allow_write=allow_write,
        description=description,
        ssh_tunnel_config=ssh_tunnel_config,
        include_metadata=include_metadata,
        default_remote_port=3306,
        error_formatter=_mysql_error_formatter,
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

            def get_env_var(var_name: str) -> str | None:
                try:
                    result = subprocess.run(
                        ["docker", "exec", container, "printenv", var_name],
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
            try:
                result = subprocess.run(
                    ["docker", "exec", container, "mysql", f"-u{db_user}", f"-p{db_password}", "-N", "-e", "SELECT VERSION()", db_name],
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
                tunnel_cfg = ssh_tunnel_config_from_dict(ssh_tunnel_config, default_remote_port=3306)
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    actual_host = "127.0.0.1"
                    actual_port = local_port
                    logger.debug(f"SSH tunnel established: localhost:{local_port} -> {tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}")

            conn = _mysql_connect(actual_host, actual_port, user, password, database, timeout)
            cursor = conn.cursor()

            # Get server version
            cursor.execute("SELECT VERSION()")
            version_row = cursor.fetchone()
            version = version_row[0] if version_row else "Unknown"

            # Get database name
            cursor.execute("SELECT DATABASE()")
            db_row = cursor.fetchone()
            db_name = db_row[0] if db_row else database

            direct_details: dict[str, Any] = {
                "version": version,
                "database": db_name,
            }

            if tunnel:
                direct_details["mode"] = "ssh_tunnel"
                direct_details["ssh_host"] = ssh_tunnel_config.get("ssh_tunnel_host", "") if ssh_tunnel_config else ""
            else:
                direct_details["host"] = host
                direct_details["port"] = port

            msg = f"Connected to {db_name} successfully"
            if tunnel:
                msg += " (via SSH tunnel)"

            return True, msg, direct_details

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
        import asyncio

        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, do_test),
            timeout=timeout + 5,
        )
    except asyncio.TimeoutError:
        return False, f"Connection timed out after {timeout} seconds", None


# Legacy tool export (for backwards compatibility if needed)
# Note: Prefer using create_mysql_tool() with ToolConfig for dynamic instances
mysql_tool = None  # Placeholder - instantiate via ToolConfig
