"""Fail-closed generation access policy for Chat and User Space."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator, Literal

from fastapi import HTTPException, status

from ragtime.core.database import get_db

GenerationSurface = Literal["chat", "userspace", "v1"]
_principals: ContextVar[tuple[str | None, ...]] = ContextVar("generation_principals", default=())
_surface: ContextVar[GenerationSurface | None] = ContextVar("generation_surface", default=None)


def current_generation_surface() -> GenerationSurface | None:
    """Return only the route/task-bound trusted surface for callback capture."""
    return _surface.get()


@contextmanager
def generation_context(surface: GenerationSurface, *user_ids: str | None) -> Iterator[None]:
    """Bind a trusted generation surface and its caller/owner principals."""
    # Nested provider/retry scopes must not shed the original caller/owner.
    principals: list[str | None] = list(_principals.get())
    for user_id in user_ids:
        normalized = str(user_id).strip() if user_id is not None else None
        if normalized not in principals:
            principals.append(normalized)
    principal_token = _principals.set(tuple(principals))
    surface_token = _surface.set(surface)
    try:
        yield
    finally:
        _surface.reset(surface_token)
        _principals.reset(principal_token)


def effective_generation_enabled(global_enabled: bool, user_override: bool | None = None) -> bool:
    return bool(global_enabled) and (user_override is None or bool(user_override))


def _combined_principals(explicit: Iterable[str | None]) -> tuple[str | None, ...]:
    values: list[str | None] = []
    for user_id in (*explicit, *_principals.get()):
        normalized = str(user_id).strip() if user_id is not None else None
        if normalized not in values:
            values.append(normalized)
    return tuple(values)


def _fields(surface: GenerationSurface) -> tuple[str, str, str, str]:
    if surface == "chat":
        return "chatEnabled", "chatEnabled", "chat_generation_disabled", "Chat generation is disabled for this user or instance."
    if surface == "userspace":
        return (
            "userspaceGenerationEnabled",
            "userspaceGenerationEnabled",
            "userspace_generation_disabled",
            "User Space generation is disabled for this user or instance.",
        )
    return "", "", "", ""


async def _enabled(surface: GenerationSurface | None, user_ids: Iterable[str | None]) -> bool:
    # OpenAI-compatible API requests are explicitly trusted by their route.
    if surface == "v1":
        return True
    if surface not in {"chat", "userspace"}:
        return False
    global_field, user_field, _, _ = _fields(surface)
    try:
        db = await get_db()
        app_settings = await db.appsettings.find_unique(where={"id": "default"})
        global_enabled = bool(app_settings and getattr(app_settings, global_field, False))
        if not global_enabled:
            return False
        principals = [user_id for user_id in _combined_principals(user_ids) if user_id]
        if not principals:
            return True
        rows = await db.user.find_many(where={"id": {"in": principals}})
        by_id = {str(row.id): row for row in rows}
        return all(user_id in by_id and effective_generation_enabled(global_enabled, getattr(by_id[user_id], user_field, None)) for user_id in principals)
    except Exception:
        return False


async def chat_generation_enabled(user_id: str | None = None) -> bool:
    return await _enabled("chat", (user_id,))


async def userspace_generation_enabled(user_id: str | None = None) -> bool:
    return await _enabled("userspace", (user_id,))


async def require_generation(*user_ids: str | None, surface: GenerationSurface | None = None) -> None:
    trusted_surface = surface or _surface.get()
    if await _enabled(trusted_surface, user_ids):
        return
    _, _, code, message = (
        _fields(trusted_surface)
        if trusted_surface in {"chat", "userspace"}
        else ("", "", "generation_scope_unknown", "Generation is unavailable for this request.")
    )
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail={"code": code, "message": message})


async def require_chat_generation(*user_ids: str | None) -> None:
    await require_generation(*user_ids, surface="chat")


async def require_userspace_generation(*user_ids: str | None) -> None:
    await require_generation(*user_ids, surface="userspace")
