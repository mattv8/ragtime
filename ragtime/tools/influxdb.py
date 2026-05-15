"""
InfluxDB 2.x Flux query tool.

Follows the same runtime patterns as MSSQL/MySQL tools:
- Dynamic input schema with timeout controls
- Default read-only execution with optional write enablement
- SSH tunnel support
- Centralized query validation and max-result enforcement
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.core.sql_utils import (
    DB_TYPE_INFLUXDB,
    enforce_max_results,
    format_query_result,
    validate_sql_query,
)
from ragtime.core.ssh import SSHTunnel, ssh_tunnel_config_from_dict

logger = get_logger(__name__)

# Maximum timeout for any tool execution (5 minutes)
# AI can request up to this limit; configured per-tool timeout is the default
MAX_TOOL_TIMEOUT_SECONDS = 300


def resolve_effective_timeout(requested_timeout: int, timeout_max_seconds: int) -> int:
    """Resolve runtime timeout using per-tool max (0 = unlimited)."""
    requested = max(0, int(requested_timeout))
    max_timeout = max(0, int(timeout_max_seconds))

    if max_timeout == 0:
        return requested
    return min(requested, max_timeout)


def _build_influxdb_url(host: str, port: int, use_https: bool) -> str:
    """Build InfluxDB API base URL from host/port/protocol toggle."""
    scheme = "https" if use_https else "http"
    return f"{scheme}://{host}:{port}"


def create_influxdb_query_input(
    default_timeout: int, timeout_max_seconds: int
) -> type[BaseModel]:
    """Create InfluxdbQueryInput with dynamic default timeout."""

    class InfluxdbQueryInputModel(BaseModel):
        """Input schema for InfluxDB Flux queries."""

        query: str = Field(
            description=(
                "Flux query to execute. Must include a result limiter with "
                "`|> limit(n: <value>)`, for example: "
                'from(bucket: "my-bucket") |> range(start: -1h) |> limit(n: 100).'
            )
        )
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
                "Use 0 for no timeout."
            ),
        )

        model_config = {"populate_by_name": True}

    return InfluxdbQueryInputModel


# Legacy schema for backwards compatibility (default 30s timeout)
class InfluxdbQueryInput(BaseModel):
    """Input schema for InfluxDB Flux queries."""

    query: str = Field(
        description=(
            "Flux query to execute. Must include a result limiter with "
            "`|> limit(n: <value>)`, for example: "
            'from(bucket: "my-bucket") |> range(start: -1h) |> limit(n: 100).'
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


async def execute_influxdb_query_async(
    query: str,
    host: str,
    port: int,
    use_https: bool,
    token: str,
    org: str,
    timeout: int = 30,
    max_results: int = 100,
    allow_write: bool = False,
    require_result_limit: bool = True,
    enforce_result_limit: bool = True,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> str:
    """
    Execute a Flux query against InfluxDB 2.x.

    Runs in a thread pool to avoid blocking the event loop.
    """
    logger.info(f"InfluxDB Query: {description}")
    logger.debug(f"Flux: {query[:200]}...")

    is_safe, reason = validate_sql_query(
        query,
        enable_write=allow_write,
        db_type=DB_TYPE_INFLUXDB,
        require_result_limit=require_result_limit,
    )
    if not is_safe:
        error_msg = f"Security validation failed: {reason}"
        logger.warning(error_msg)
        return f"Error: {error_msg}"

    if enforce_result_limit:
        query = enforce_max_results(query, max_results, db_type=DB_TYPE_INFLUXDB)

    def run_query() -> str:
        """Execute query in thread pool."""
        try:
            from influxdb_client import InfluxDBClient  # type: ignore[import-untyped]
        except ImportError:
            return (
                "Error: influxdb-client package not installed. "
                "Install with: pip install influxdb-client"
            )

        tunnel: SSHTunnel | None = None
        client: Any = None

        try:
            scheme = "https" if use_https else "http"
            effective_url = _build_influxdb_url(host, port, use_https)

            if ssh_tunnel_config:
                tunnel_dict = dict(ssh_tunnel_config)
                tunnel_dict.setdefault("host", host)
                tunnel_dict.setdefault("port", port)

                tunnel_cfg = ssh_tunnel_config_from_dict(
                    tunnel_dict, default_remote_port=port
                )
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    effective_url = f"{scheme}://127.0.0.1:{local_port}"
                    logger.debug(
                        f"SSH tunnel established: localhost:{local_port} -> "
                        f"{tunnel_cfg.remote_host}:{tunnel_cfg.remote_port}"
                    )

            client = InfluxDBClient(
                url=effective_url,
                token=token,
                org=org,
                timeout=timeout * 1000 if timeout > 0 else None,
            )

            query_api = client.query_api()
            tables = query_api.query(query=query, org=org)

            rows: list[dict[str, Any]] = []
            for table in tables:
                for record in table.records:
                    values = {
                        k: v
                        for k, v in record.values.items()
                        if k not in {"result", "table"}
                    }
                    rows.append(values)

            if not rows:
                return "Query executed successfully (no results)"

            # Stable column order for deterministic rendering.
            column_names = sorted({key for row in rows for key in row.keys()})
            return format_query_result(
                rows,
                column_names,
                include_metadata=include_metadata,
            )

        except Exception as e:
            error_str = str(e)
            logger.error(f"InfluxDB query error: {error_str}")
            if "unauthorized" in error_str.lower() or "token" in error_str.lower():
                return "Error: Authentication failed - check token and organization"
            if "timeout" in error_str.lower():
                return f"Error: Query timed out after {timeout} seconds"
            if tunnel:
                return "Error: Cannot connect to InfluxDB through SSH tunnel"
            return f"Error: {error_str}"

        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass
            if tunnel:
                try:
                    tunnel.stop()
                except Exception:
                    pass

    try:
        if timeout > 0:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, run_query),
                timeout=timeout + 5,
            )

        return await asyncio.get_event_loop().run_in_executor(None, run_query)

    except asyncio.TimeoutError:
        logger.error(f"InfluxDB query timed out after {timeout}s")
        return f"Error: Query timed out after {timeout} seconds"


def create_influxdb_tool(
    name: str,
    host: str,
    port: int,
    use_https: bool,
    token: str,
    org: str,
    timeout: int = 30,
    timeout_max_seconds: int = MAX_TOOL_TIMEOUT_SECONDS,
    max_results: int = 100,
    allow_write: bool = False,
    description: str = "",
    ssh_tunnel_config: dict[str, Any] | None = None,
    include_metadata: bool = True,
) -> StructuredTool:
    """Create a configured InfluxDB query tool for LangChain."""
    QueryInput = create_influxdb_query_input(timeout, timeout_max_seconds)

    async def execute_query(
        query: str = "", description: str = "", timeout: int = timeout, **_: Any
    ) -> str:
        """Execute InfluxDB query using configured connection."""
        if not query or not query.strip():
            return (
                "Error: 'query' parameter is required. Provide a Flux query to execute."
            )
        if not description:
            description = "Flux query"

        effective_timeout = resolve_effective_timeout(timeout, timeout_max_seconds)

        return await execute_influxdb_query_async(
            query=query,
            host=host,
            port=port,
            use_https=use_https,
            token=token,
            org=org,
            timeout=effective_timeout,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
            ssh_tunnel_config=ssh_tunnel_config,
            include_metadata=include_metadata,
        )

    tool_description = f"Query the {name} InfluxDB database using Flux."
    if description:
        tool_description += f" This database contains: {description}"
    tool_description += (
        " Include |> limit(n: ...) to restrict results. "
        "Read-only queries only unless writes are enabled."
    )

    return StructuredTool.from_function(
        coroutine=execute_query,
        name=f"query_{name.lower().replace(' ', '_').replace('-', '_')}",
        description=tool_description,
        args_schema=QueryInput,
    )


async def test_influxdb_connection(
    host: str,
    port: int,
    use_https: bool,
    token: str,
    org: str,
    bucket: str = "",
    timeout: int = 10,
    ssh_tunnel_config: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Test InfluxDB connectivity and return status."""

    def do_test() -> tuple[bool, str, dict[str, Any] | None]:
        try:
            from influxdb_client import InfluxDBClient  # type: ignore[import-untyped]
        except ImportError:
            return False, "influxdb-client package not installed", None

        tunnel: SSHTunnel | None = None
        client: Any = None

        try:
            scheme = "https" if use_https else "http"
            effective_url = _build_influxdb_url(host, port, use_https)

            if ssh_tunnel_config:
                tunnel_dict = dict(ssh_tunnel_config)
                tunnel_dict.setdefault("host", host)
                tunnel_dict.setdefault("port", port)
                tunnel_cfg = ssh_tunnel_config_from_dict(
                    tunnel_dict, default_remote_port=port
                )
                if tunnel_cfg:
                    tunnel = SSHTunnel(tunnel_cfg)
                    local_port = tunnel.start()
                    effective_url = f"{scheme}://127.0.0.1:{local_port}"

            client = InfluxDBClient(
                url=effective_url,
                token=token,
                org=org or "",
                timeout=timeout * 1000 if timeout > 0 else None,
            )

            # Step 1: confirm server is reachable (no auth required)
            health = client.health()
            if not health or getattr(health, "status", "").lower() != "pass":
                status = getattr(health, "status", "unknown")
                return False, f"InfluxDB health check failed (status={status})", None

            version = getattr(health, "version", "")

            # Step 2: validate token is actually authorized by running a
            # minimal Flux query.  This catches revoked/wrong tokens that
            # would otherwise look fine because /health needs no auth.
            if not org:
                # Cannot validate token without an org; report partial success
                return (
                    True,
                    f"Server reachable (InfluxDB {version}) but org is required to validate token",
                    {"status": "pass", "version": version, "warning": "org_not_set"},
                )

            try:
                tables = client.query_api().query("buckets() |> limit(n: 1)", org=org)
                bucket_names = [
                    r.values.get("name", "") for t in tables for r in t.records
                ]
            except Exception as qe:
                qe_str = str(qe)
                if "401" in qe_str or "unauthorized" in qe_str.lower():
                    return (
                        False,
                        "Token authentication failed — check the token and organization name",
                        None,
                    )
                if "not found" in qe_str.lower() or "organization" in qe_str.lower():
                    return (
                        False,
                        f"Organization '{org}' not found on this InfluxDB server",
                        None,
                    )
                return False, f"Token validation failed: {qe_str[:200]}", None

            details: dict[str, Any] = {
                "status": "pass",
                "version": version,
                "org": org,
                "accessible_buckets": len(bucket_names),
            }
            if bucket:
                details["bucket"] = bucket
            if tunnel:
                details["mode"] = "ssh_tunnel"
                details["ssh_host"] = (
                    ssh_tunnel_config.get("ssh_tunnel_host", "")
                    if ssh_tunnel_config
                    else ""
                )

            bucket_hint = (
                f" ({len(bucket_names)} bucket(s) accessible)" if bucket_names else ""
            )
            return True, f"InfluxDB connection successful{bucket_hint}", details

        except Exception as e:
            error_str = str(e)
            if "unauthorized" in error_str.lower() or "401" in error_str:
                return (
                    False,
                    "Token authentication failed — check the token and organization name",
                    None,
                )
            if tunnel:
                return False, "Cannot connect to InfluxDB through SSH tunnel", None
            return False, f"Connection failed: {error_str}", None

        finally:
            if client:
                try:
                    client.close()
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
