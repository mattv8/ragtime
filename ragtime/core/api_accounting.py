"""
API request logging and analytics.

Records each OpenAI-compatible API request and provides aggregation
queries for the admin usage dashboard.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import normalize_provider_name

logger = get_logger(__name__)


async def log_api_request(
    *,
    user_id: str | None = None,
    provider: str = "",
    model: str = "",
    endpoint: str = "/v1/chat/completions",
    http_method: str = "POST",
    status_code: int = 200,
    streaming: bool = False,
) -> None:
    """Record an API HTTP request."""
    try:
        db = await get_db()
        await db.apirequestlog.create(
            data={
                "id": str(uuid.uuid4()),
                "userId": user_id,
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "httpMethod": http_method,
                "statusCode": status_code,
                "streaming": streaming,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to log API request: {e}")


async def get_api_daily_trend(
    *,
    since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Daily API request trend for admin dashboard."""
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
        FROM api_request_logs
        {where_sql}
        GROUP BY DATE(created_at)
        ORDER BY date ASC
    """

    rows = await db.query_raw(query, *params)
    for row in rows:
        row["provider"] = normalize_provider_name(
            row.get("provider"),
            model_id=row.get("model"),
        )
    return rows


async def get_api_provider_model_breakdown(
    *,
    since: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Provider/model breakdown for API requests."""
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
            provider,
            model,
            'api'::text AS request_source,
            COUNT(*)::int AS total_requests,
            0::int AS total_input_tokens,
            0::int AS total_output_tokens,
            0::int AS total_tokens
        FROM api_request_logs
        WHERE endpoint = '/v1/chat/completions'
          AND status_code >= 200
          AND status_code < 400
          {'AND ' + ' AND '.join(where_clauses) if where_clauses else ''}
        GROUP BY provider, model
        ORDER BY total_requests DESC
    """

    return await db.query_raw(query, *params)
