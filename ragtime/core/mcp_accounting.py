"""
MCP request logging and analytics.

Records each MCP HTTP request and provides aggregation queries
for the admin usage dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger

logger = get_logger(__name__)


async def log_mcp_request(
    *,
    user_id: str | None = None,
    username: str | None = None,
    route_name: str = "default",
    auth_method: str = "none",
    http_method: str = "POST",
    status_code: int = 200,
) -> None:
    """Record an MCP HTTP request."""
    try:
        db = await get_db()
        await db.mcprequestlog.create(
            data={
                "id": str(uuid.uuid4()),
                "userId": user_id,
                "username": username,
                "routeName": route_name,
                "authMethod": auth_method,
                "httpMethod": http_method,
                "statusCode": status_code,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log MCP request: {e}")


async def get_mcp_usage_by_user(
    *,
    since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Per-user MCP request summary for admin dashboard."""
    db = await get_db()

    where_clauses: list[str] = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"m.created_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            COALESCE(m.user_id, '__anonymous__') AS user_id,
            COALESCE(u.username, m.username, 'anonymous') AS username,
            u.display_name,
            m.auth_method,
            m.route_name,
            COUNT(*)::int AS total_requests,
            COUNT(*) FILTER (WHERE m.status_code >= 200 AND m.status_code < 400)::int AS success_count,
            COUNT(*) FILTER (WHERE m.status_code >= 400)::int AS error_count
        FROM mcp_request_logs m
        LEFT JOIN users u ON m.user_id = u.id
        {where_sql}
        GROUP BY COALESCE(m.user_id, '__anonymous__'), COALESCE(u.username, m.username, 'anonymous'), u.display_name, m.auth_method, m.route_name
        ORDER BY total_requests DESC
    """

    return await db.query_raw(query, *params)


async def get_mcp_daily_trend(
    *,
    since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Daily MCP request trend for admin dashboard."""
    db = await get_db()

    where_clauses: list[str] = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"created_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            DATE(created_at) AS date,
            COUNT(*)::int AS total_requests,
            COUNT(*) FILTER (WHERE status_code >= 200 AND status_code < 400)::int AS success_count,
            COUNT(*) FILTER (WHERE status_code >= 400)::int AS error_count,
            COUNT(DISTINCT user_id)::int AS unique_users
        FROM mcp_request_logs
        {where_sql}
        GROUP BY DATE(created_at)
        ORDER BY date ASC
    """

    return await db.query_raw(query, *params)


async def get_mcp_usage_by_route(
    *,
    since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Per-route MCP request summary for admin dashboard."""
    db = await get_db()

    where_clauses: list[str] = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"created_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            route_name,
            COUNT(*)::int AS total_requests,
            COUNT(*) FILTER (WHERE status_code >= 200 AND status_code < 400)::int AS success_count,
            COUNT(*) FILTER (WHERE status_code >= 400)::int AS error_count,
            COUNT(DISTINCT user_id)::int AS unique_users
        FROM mcp_request_logs
        {where_sql}
        GROUP BY route_name
        ORDER BY total_requests DESC
    """

    return await db.query_raw(query, *params)
