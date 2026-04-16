"""
Usage accounting service for per-user request and token tracking.

Records each chat attempt at request start, finalizes it after
completion or failure, and supports recovery for interrupted runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from ragtime.core.database import get_db
from ragtime.core.logging import get_logger
from ragtime.core.model_limits import normalize_provider_name
from ragtime.core.tokenization import count_tokens

logger = get_logger(__name__)


def _estimate_input_tokens(
    user_message: str, chat_history: Optional[list] = None
) -> int:
    """Estimate input tokens from the user message and optional chat history."""
    tokens = count_tokens(user_message)
    if chat_history:
        for msg in chat_history:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                tokens += count_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        tokens += count_tokens(part.get("text", ""))
    return tokens


def _estimate_output_tokens(
    response_text: str, events: Optional[list[dict]] = None
) -> int:
    """Estimate output tokens from the response text and optional events."""
    tokens = count_tokens(response_text) if response_text else 0
    if events:
        for ev in events:
            ev_type = ev.get("type")
            if ev_type == "tool":
                tool_input = ev.get("input")
                if isinstance(tool_input, dict):
                    import json

                    tokens += count_tokens(json.dumps(tool_input, default=str))
                tool_output = ev.get("output", "")
                if tool_output:
                    tokens += count_tokens(str(tool_output))
            elif ev_type == "reasoning":
                tokens += count_tokens(ev.get("content", ""))
    return tokens


async def create_usage_attempt(
    *,
    user_id: str,
    request_source: str,
    provider: str = "",
    model: str = "",
    conversation_id: Optional[str] = None,
    input_tokens: int = 0,
) -> str:
    """Create a usage attempt record at request start. Returns the attempt ID."""
    attempt_id = str(uuid.uuid4())
    try:
        db = await get_db()
        await db.userusageattempt.create(
            data={
                "id": attempt_id,
                "userId": user_id,
                "requestSource": request_source,
                "provider": provider,
                "model": model,
                "conversationId": conversation_id,
                "inputTokens": input_tokens,
                "totalTokens": input_tokens,
                "tokensEstimated": True,
                "status": "started",
            }
        )
    except Exception as e:
        logger.warning(f"Failed to create usage attempt: {e}")
    return attempt_id


async def bind_usage_attempt_task(attempt_id: str, chat_task_id: str) -> None:
    """Bind a chat task ID to an existing usage attempt."""
    if not attempt_id:
        return
    try:
        db = await get_db()
        await db.userusageattempt.update(
            where={"id": attempt_id},
            data={"chatTaskId": chat_task_id},
        )
    except Exception as e:
        logger.warning(f"Failed to bind task to usage attempt: {e}")


async def finalize_usage_attempt(
    attempt_id: str,
    *,
    status: str,
    output_tokens: int = 0,
    input_tokens: Optional[int] = None,
    failure_reason: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Finalize a usage attempt with final status and token counts.

    Idempotent: if already finalized (finalizedAt set), this is a no-op.
    """
    if not attempt_id:
        return
    try:
        db = await get_db()
        existing = await db.userusageattempt.find_unique(where={"id": attempt_id})
        if not existing or existing.finalizedAt is not None:
            return  # Already finalized or missing

        update_data: dict[str, Any] = {
            "status": status,
            "outputTokens": output_tokens,
            "finalizedAt": datetime.utcnow(),
        }
        if input_tokens is not None:
            update_data["inputTokens"] = input_tokens
        if failure_reason is not None:
            update_data["failureReason"] = (
                failure_reason[:500] if failure_reason else None
            )
        if provider is not None:
            update_data["provider"] = provider
        if model is not None:
            update_data["model"] = model

        # Compute total
        final_input = input_tokens if input_tokens is not None else existing.inputTokens
        update_data["totalTokens"] = final_input + output_tokens

        await db.userusageattempt.update(
            where={"id": attempt_id},
            data=update_data,
        )
    except Exception as e:
        logger.warning(f"Failed to finalize usage attempt {attempt_id}: {e}")


async def finalize_stale_attempts_for_tasks(task_ids: list[str]) -> int:
    """Finalize any open usage attempts linked to the given task IDs as interrupted."""
    if not task_ids:
        return 0
    count = 0
    try:
        db = await get_db()
        open_attempts = await db.userusageattempt.find_many(
            where={
                "chatTaskId": {"in": task_ids},
                "finalizedAt": None,
            }
        )
        for attempt in open_attempts:
            await db.userusageattempt.update(
                where={"id": attempt.id},
                data={
                    "status": "interrupted",
                    "finalizedAt": datetime.utcnow(),
                    "failureReason": "Task interrupted by server restart",
                },
            )
            count += 1
    except Exception as e:
        logger.warning(f"Failed to finalize stale usage attempts: {e}")
    return count


async def get_user_usage_summary(
    *,
    user_id: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Get per-user usage summary for admin dashboard.

    Returns aggregated rows with: user_id, username, display_name,
    total_requests, total_input_tokens, total_output_tokens, total_tokens,
    completed_count, failed_count, cancelled_count, interrupted_count.
    """
    db = await get_db()

    where_clauses = []
    params: list[Any] = []
    param_idx = 1

    if user_id:
        where_clauses.append(f"u.user_id = ${param_idx}")
        params.append(user_id)
        param_idx += 1
    if since:
        where_clauses.append(f"u.started_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1
    if until:
        where_clauses.append(f"u.started_at < ${param_idx}::timestamp")
        params.append(until.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            u.user_id,
            usr.username,
            usr.display_name,
            COUNT(*)::int AS total_requests,
            COALESCE(SUM(u.input_tokens), 0)::int AS total_input_tokens,
            COALESCE(SUM(u.output_tokens), 0)::int AS total_output_tokens,
            COALESCE(SUM(u.total_tokens), 0)::int AS total_tokens,
            COUNT(*) FILTER (WHERE u.status = 'completed')::int AS completed_count,
            COUNT(*) FILTER (WHERE u.status = 'failed')::int AS failed_count,
            COUNT(*) FILTER (WHERE u.status = 'cancelled')::int AS cancelled_count,
            COUNT(*) FILTER (WHERE u.status = 'interrupted')::int AS interrupted_count,
            COUNT(*) FILTER (WHERE u.status = 'started')::int AS started_count
        FROM user_usage_attempts u
        JOIN users usr ON u.user_id = usr.id
        {where_sql}
        GROUP BY u.user_id, usr.username, usr.display_name
        ORDER BY total_tokens DESC
    """

    rows = await db.query_raw(query, *params)
    for row in rows:
        row["provider"] = normalize_provider_name(
            row.get("provider"),
            model_id=row.get("model"),
        )
    return rows


async def get_provider_model_breakdown(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Get usage breakdown by provider and model."""
    db = await get_db()

    where_clauses = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"started_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1
    if until:
        where_clauses.append(f"started_at < ${param_idx}::timestamp")
        params.append(until.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            provider,
            model,
            request_source,
            COUNT(*)::int AS total_requests,
            COALESCE(SUM(input_tokens), 0)::int AS total_input_tokens,
            COALESCE(SUM(output_tokens), 0)::int AS total_output_tokens,
            COALESCE(SUM(total_tokens), 0)::int AS total_tokens
        FROM user_usage_attempts
        {where_sql}
        GROUP BY provider, model, request_source
        ORDER BY total_tokens DESC
    """

    rows = await db.query_raw(query, *params)
    return rows


async def get_daily_usage_trend(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Get daily usage trend for the admin dashboard."""
    db = await get_db()

    where_clauses = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"started_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1
    if until:
        where_clauses.append(f"started_at < ${param_idx}::timestamp")
        params.append(until.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            DATE(started_at) AS date,
            COUNT(*)::int AS total_requests,
            COALESCE(SUM(input_tokens), 0)::int AS total_input_tokens,
            COALESCE(SUM(output_tokens), 0)::int AS total_output_tokens,
            COALESCE(SUM(total_tokens), 0)::int AS total_tokens,
            COUNT(*) FILTER (WHERE status = 'completed')::int AS completed_count,
            COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count
        FROM user_usage_attempts
        {where_sql}
        GROUP BY DATE(started_at)
        ORDER BY date ASC
    """

    rows = await db.query_raw(query, *params)
    return rows


async def get_user_daily_usage_series(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Get per-user daily usage series for the admin dashboard."""
    db = await get_db()

    where_clauses = []
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"u.started_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1
    if until:
        where_clauses.append(f"u.started_at < ${param_idx}::timestamp")
        params.append(until.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"""
        SELECT
            u.user_id,
            usr.username,
            usr.display_name,
            DATE(u.started_at) AS date,
            COUNT(*)::int AS total_requests,
            COALESCE(SUM(u.input_tokens), 0)::int AS total_input_tokens,
            COALESCE(SUM(u.output_tokens), 0)::int AS total_output_tokens,
            COALESCE(SUM(u.total_tokens), 0)::int AS total_tokens
        FROM user_usage_attempts u
        JOIN users usr ON u.user_id = usr.id
        {where_sql}
        GROUP BY u.user_id, usr.username, usr.display_name, DATE(u.started_at)
        ORDER BY date ASC, total_tokens DESC, usr.username ASC
    """

    rows = await db.query_raw(query, *params)
    return rows


async def get_usage_earliest_date() -> str | None:
    """Return the earliest usage attempt date as an ISO date string, or None."""
    db = await get_db()
    rows = await db.query_raw(
        "SELECT MIN(started_at)::date::text AS earliest FROM user_usage_attempts"
    )
    if rows and rows[0].get("earliest"):
        return rows[0]["earliest"]
    return None


async def get_daily_provider_failures(
    *,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """Get daily failure/interrupted counts by provider and model."""
    db = await get_db()

    where_clauses = ["status IN ('failed', 'interrupted')"]
    params: list[Any] = []
    param_idx = 1

    if since:
        where_clauses.append(f"started_at >= ${param_idx}::timestamp")
        params.append(since.isoformat())
        param_idx += 1
    if until:
        where_clauses.append(f"started_at < ${param_idx}::timestamp")
        params.append(until.isoformat())
        param_idx += 1

    where_sql = f"WHERE {' AND '.join(where_clauses)}"

    query = f"""
        SELECT
            DATE(started_at) AS date,
            provider,
            model,
            COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
            COUNT(*) FILTER (WHERE status = 'interrupted')::int AS interrupted_count
        FROM user_usage_attempts
        {where_sql}
        GROUP BY DATE(started_at), provider, model
        ORDER BY date ASC, (COUNT(*)) DESC
    """

    rows = await db.query_raw(query, *params)
    return rows
