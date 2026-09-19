"""Fresh, fail-closed policy checks for Ragtime-hosted generation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator

from fastapi import HTTPException, status

from ragtime.core.database import get_db

_hosted_execution_principals: ContextVar[tuple[str | None, ...]] = ContextVar("hosted_execution_principals", default=())


@contextmanager
def hosted_execution_context(*user_ids: str | None) -> Iterator[None]:
    """Bind caller/owner principals to the current request or task context."""
    token = _hosted_execution_principals.set(tuple(user_ids))
    try:
        yield
    finally:
        _hosted_execution_principals.reset(token)


def effective_hosted_execution_enabled(global_enabled: bool, user_override: bool | None = None) -> bool:
    """Pure global-first precedence rule shared by request and bulk callers."""
    return bool(global_enabled) and (user_override is None or bool(user_override))


def _combined_principals(explicit: Iterable[str | None]) -> tuple[str | None, ...]:
    """Combine explicit and bound principals; explicit callers cannot bypass owners."""
    principals: list[str | None] = []
    for user_id in (*explicit, *_hosted_execution_principals.get()):
        normalized = str(user_id).strip() if user_id is not None else None
        if normalized not in principals:
            principals.append(normalized)
    return tuple(principals)


async def _load_global_enabled(db: Any) -> bool:
    app_settings = await db.appsettings.find_unique(where={"id": "default"})
    return bool(app_settings and getattr(app_settings, "hostedChatEnabled", False))


async def hosted_execution_enabled(user_id: str | None = None) -> bool:
    """Resolve the policy directly from persistence; database failures deny."""
    try:
        db = await get_db()
        global_enabled = await _load_global_enabled(db)
        if not global_enabled:
            return False
        if user_id is None:
            return True
        user = await db.user.find_unique(where={"id": user_id})
        if user is None:
            return False
        return effective_hosted_execution_enabled(global_enabled, getattr(user, "hostedChatEnabled", None))
    except Exception:
        return False


async def require_hosted_execution(*user_ids: str | None) -> None:
    """Require global and every explicit or context-bound user policy."""
    principals = _combined_principals(user_ids)
    try:
        db = await get_db()
        global_enabled = await _load_global_enabled(db)
        if not global_enabled:
            enabled = False
        else:
            user_ids_to_load = [user_id for user_id in principals if user_id]
            if not user_ids_to_load:
                enabled = True
            else:
                rows = await db.user.find_many(where={"id": {"in": user_ids_to_load}})
                rows_by_id = {str(row.id): row for row in rows}
                enabled = all(
                    user_id in rows_by_id and effective_hosted_execution_enabled(global_enabled, getattr(rows_by_id[user_id], "hostedChatEnabled", None))
                    for user_id in user_ids_to_load
                )
    except Exception:
        enabled = False
    if not enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "hosted_execution_disabled", "message": "Hosted execution is disabled for this user or instance."},
        )
