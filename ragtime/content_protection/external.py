"""Small transport adapters for content-protection boundaries.

This module deliberately contains no policy decisions.  External transports use
it to construct only server-derived identities and invoke the core service at
each execute/release boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, cast

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionError


def _core() -> Any:
    # Kept lazy so independently-started route modules do not require the core
    # package to have completed import during application startup.
    return service


def context_for_principal(
    principal: Any | None,
    *,
    surface: str,
    mcp_route: str | None = None,
    tool_id: str | None = None,
    resource_id: str | None = None,
    public: bool = False,
    baseline: str | None = None,
) -> Any:
    """Build a core context from a verified principal, never request data."""
    user_id = getattr(principal, "user_id", None) if principal is not None else None
    if baseline is None:
        baseline = "public" if public else ("user" if user_id else "service")
    return _core().ProtectionContext(
        user_id=str(user_id) if user_id else None,
        audience_user_ids=(str(user_id),) if user_id else (),
        surface=surface,
        mcp_route=mcp_route,
        tool_id=tool_id,
        resource_id=resource_id,
        public=public,
        baseline=baseline,
    )


@contextmanager
def external_protection_context(principal: Any | None, **kwargs: Any) -> Iterator[Any]:
    """Bind a request-local protection context for cached MCP servers."""
    core = _core()
    context = context_for_principal(principal, **kwargs)
    with core.protection_context(context):
        yield context


async def authorize_external_content(
    candidate: Any,
    *,
    direction: str,
    principal: Any | None = None,
    context: Any | None = None,
    surface: str,
    mcp_route: str | None = None,
    tool_id: str | None = None,
    resource_id: str | None = None,
    operation: str | None = None,
    public: bool = False,
    baseline: str | None = None,
    supporting_context: Any = None,
    execution_completed: bool = False,
) -> None:
    """Authorize an unmodified external candidate at one concrete boundary.

    ``execution_completed`` is trusted adapter state, never client input.  It
    is set only after a successful operation has produced its result, so a
    denied release can honestly tell an external harness not to replay it.
    """
    core = _core()
    effective_context = context or context_for_principal(
        principal,
        surface=surface,
        mcp_route=mcp_route,
        tool_id=tool_id,
        resource_id=resource_id,
        public=public,
        baseline=baseline,
    )
    try:
        await core.authorize_content(
            candidate,
            direction=direction,
            context=effective_context,
            tool_id=tool_id,
            operation=operation,
            supporting_context=supporting_context,
        )
    except ContentProtectionError as exc:
        if execution_completed and direction == "outbound" and exc.code == "content_denied":
            raise ContentProtectionError(
                exc.code,
                exc.request_id,
                reason=exc.reason,
                reason_code=exc.reason_code,
                recovery_eligible=True,
                execution_status="completed_response_withheld",
            ) from exc
        raise


def public_error_detail(exc: Exception) -> dict[str, str]:
    """Return the core fixed error envelope without exposing candidate data."""
    detail = getattr(exc, "public_detail", None)
    if callable(detail):
        # BYO callers own their logical-turn state across requests, so never
        # expose a hosted server's attempt counter through these transports.
        return {key: value for key, value in dict(cast(Callable[[], Mapping[str, str]], detail)()).items() if key != "attempts_remaining"}
    return {"code": "content_unavailable", "message": "This content is not available under your access profile."}
