"""Hosted-chat adapters for the transport-neutral content protection service.

This module deliberately contains no policy resolution.  It supplies the
execution ordering needed by LangChain: arguments are authorized before a
tool coroutine is entered and results before it is returned to the agent.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, cast
from uuid import uuid4

from ragtime.content_protection import service
from ragtime.content_protection.models import ContentProtectionError

_MAX_BUFFER_BYTES = 1024 * 1024


@dataclass
class _ToolSelection:
    """Mutable turn state shared with LangChain child tasks."""

    tool_id: str | None = None


_selected_tool_state: ContextVar[_ToolSelection | None] = ContextVar("content_protection_selected_tool_state", default=None)


def _effective_context(context: Any) -> Any:
    context = _merge_context(_service().current_context(), context)
    state = _selected_tool_state.get()
    return replace(context, tool_id=state.tool_id) if context is not None and state and state.tool_id else context


def _merge_context(outer: Any, inner: Any) -> Any:
    """Keep route-derived scope and audience when binding nested work."""
    if outer is None or not hasattr(inner, "surface"):
        return inner
    audience = tuple(dict.fromkeys((*getattr(outer, "audience_user_ids", ()), *getattr(inner, "audience_user_ids", ()))))
    return replace(
        inner,
        user_id=getattr(inner, "user_id", None) or getattr(outer, "user_id", None),
        audience_user_ids=audience,
        surface=getattr(outer, "surface", getattr(inner, "surface", "chat")),
        mcp_route=getattr(outer, "mcp_route", None) or getattr(inner, "mcp_route", None),
        resource_id=getattr(outer, "resource_id", None) or getattr(inner, "resource_id", None),
        public=bool(getattr(outer, "public", False) or getattr(inner, "public", False)),
        baseline=getattr(outer, "baseline", getattr(inner, "baseline", "user")),
    )


async def _tool_context(context: Any, tool_id: str) -> tuple[Any, bool]:
    service = _service()
    resolved = replace(context, tool_id=tool_id) if context is not None else None
    required = resolved is not None and await service.classification_required(resolved)
    if required:
        state = _selected_tool_state.get()
        if state is not None:
            state.tool_id = tool_id
    return resolved, required


def _service() -> Any:
    # Keep the policy service as the sole owner of configuration, identity and
    # cache state.  Importing lazily also avoids initializing it during module
    # discovery.
    return service


def _ensure_active_attempt() -> None:
    """Stop work owned by an attempt which became terminal while awaiting."""
    guard = getattr(_service(), "ensure_active_attempt", None)
    if callable(guard):
        guard()


def _terminal_error() -> ContentProtectionError | None:
    """Return a swallowed protection failure, without trusting framework events."""
    getter = getattr(_service(), "terminal_error", None)
    return cast(ContentProtectionError | None, getter()) if callable(getter) else None


def hosted_context(
    *,
    user_id: str | None,
    owner_user_id: str | None,
    surface: str = "chat",
    public: bool = False,
) -> Any:
    service = _service()
    audience = (None,) if public else tuple(user for user in (user_id, owner_user_id) if user)
    context = service.ProtectionContext(
        user_id=user_id,
        audience_user_ids=audience,
        surface=surface,
        public=public,
        baseline="public" if public else ("user" if user_id else "service"),
    )
    return _merge_context(service.current_context(), context)


async def authorize_inbound(candidate: Any, *, context: Any, supporting_context: Any = None) -> None:
    await _service().authorize_content(
        candidate,
        direction="inbound",
        context=_effective_context(context),
        supporting_context=supporting_context,
    )


async def authorize_history(candidate: Any, *, context: Any) -> None:
    await _service().authorize_content(candidate, direction="stored_readback", context=_effective_context(context))


async def authorize_assistant(candidate: Any, *, context: Any, supporting_context: Any = None) -> None:
    await _service().authorize_content(
        candidate,
        direction="assistant_response",
        context=_effective_context(context),
        supporting_context=supporting_context,
    )


async def authorize_auxiliary(candidate: Any, *, context: Any, operation: str) -> None:
    await _service().authorize_content(
        candidate,
        direction="assistant_response",
        context=_effective_context(context),
        operation=operation,
    )


async def authorize_persistence(candidate: Any, *, direction: str, operation: str | None = None) -> None:
    """Guard a persistence sink using the active turn context when present."""
    service = _service()
    await service.authorize_content(candidate, direction=direction, context=_effective_context(service.current_context()), operation=operation)


def bind_context(context: Any) -> Any:
    class _BoundContext:
        def __enter__(self) -> Any:
            service = _service()
            self._context = _merge_context(service.current_context(), context)
            current_selection = _selected_tool_state.get()
            self._tool_token = None if current_selection is not None else _selected_tool_state.set(_ToolSelection())
            self._service_context = service.protection_context(self._context)
            return self._service_context.__enter__()

        def __exit__(self, *args: Any) -> None:
            try:
                self._service_context.__exit__(*args)
            finally:
                if self._tool_token is not None:
                    _selected_tool_state.reset(self._tool_token)

    return _BoundContext()


def public_error_event(exc: Exception) -> dict[str, str]:
    """Return only the core's fixed, transport-safe error envelope."""
    detail = getattr(exc, "public_detail", None)
    if callable(detail):
        return {"type": "error", **dict(cast(Callable[[], Mapping[str, str]], detail)())}
    return {"type": "error", "code": "classifier_unavailable", "message": "Content protection is unavailable."}


def _run_async_from_sync(awaitable: Any) -> Any:
    """Run a guard from a synchronous tool without dropping ContextVars."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}
    context = contextvars.copy_context()

    def runner() -> None:
        try:
            result["value"] = context.run(asyncio.run, awaitable)
        except BaseException as exc:  # re-raise on the framework thread
            result["error"] = exc

    thread = threading.Thread(target=runner, name="content-protection-sync-tool", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def wrap_tools(tools: list[Any], clone: Any, *, tool_ids_by_name: dict[str, str] | None = None) -> list[Any]:
    """Wrap every coroutine tool at its executor boundary.

    ``current_context`` is read inside each invocation, rather than captured
    when an executor is cached, so queued work uses the caller bound to its
    active turn.
    """
    wrapped: list[Any] = []
    tool_ids_by_name = tool_ids_by_name or {}
    for tool in tools:
        original = getattr(tool, "coroutine", None)
        original_func = getattr(tool, "func", None)
        tool_name = str(getattr(tool, "name", ""))
        canonical_tool_id = tool_ids_by_name.get(tool_name) or tool_name
        if original is None and original_func is None:
            wrapped.append(tool)
            continue

        async def guarded(*args: Any, _original: Any = original, _tool_id: str = canonical_tool_id, **kwargs: Any) -> Any:
            service = _service()
            context = service.current_context()
            tool_context, _ = await _tool_context(context, _tool_id)
            arguments: Any = kwargs if kwargs else (args[0] if args else {})
            await service.authorize_content(
                arguments,
                direction="proposed_operation",
                context=tool_context,
                tool_id=_tool_id,
                operation="tool_call",
            )
            _ensure_active_attempt()
            result = await _original(*args, **kwargs)
            await service.authorize_content(
                result,
                direction="tool_result",
                context=tool_context,
                tool_id=_tool_id,
                operation="tool_result",
                supporting_context=arguments,
            )
            _ensure_active_attempt()
            return result

        def guarded_func(*args: Any, _original_func: Any = original_func, _tool_id: str = canonical_tool_id, **kwargs: Any) -> Any:
            if _original_func is None:
                # A coroutine-only tool is not synchronously executable.  Do
                # not silently bypass its guard on a synchronous call path.
                raise RuntimeError("Content-protected tool requires asynchronous execution")

            async def invoke() -> Any:
                service = _service()
                context = service.current_context()
                tool_context, _ = await _tool_context(context, _tool_id)
                arguments: Any = kwargs if kwargs else (args[0] if args else {})
                await service.authorize_content(arguments, direction="proposed_operation", context=tool_context, tool_id=_tool_id, operation="tool_call")
                _ensure_active_attempt()
                result = await asyncio.to_thread(_original_func, *args, **kwargs)
                await service.authorize_content(
                    result, direction="tool_result", context=tool_context, tool_id=_tool_id, operation="tool_result", supporting_context=arguments
                )
                _ensure_active_attempt()
                return result

            return _run_async_from_sync(invoke())

        wrapped.append(clone(tool, coroutine=guarded if original is not None else None, func=guarded_func))
    return wrapped


async def buffered_stream(
    stream: AsyncIterator[Any],
    *,
    context: Any,
    tool_ids_by_name: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """Release completed model output only after its applicable policy approves it."""
    service = _service()
    # Master-off and never-classify turns retain native token streaming; do not
    # accumulate a full response merely to decide it does not need inspection.
    root_required = await service.classification_required(context)
    if not root_required:
        config = await service.load_config()
        may_require_tool = bool(
            getattr(config, "enabled", False)
            and any(
                getattr(requirement, "scope_kind", None) == "tool" and getattr(requirement, "mode", None) == "require"
                for requirement in getattr(config, "requirements", ())
            )
        )
        if not may_require_tool:
            async for event in stream:
                yield event
            return

    tool_ids_by_name = tool_ids_by_name or {}
    active_protection = root_required
    pending: list[str] = []
    pending_reasoning: list[str] = []
    pending_bytes = 0

    def append_fragment(target: list[str], fragment: str) -> None:
        nonlocal pending_bytes
        fragment_bytes = len(fragment.encode("utf-8"))
        if pending_bytes + fragment_bytes > _MAX_BUFFER_BYTES:
            raise ContentProtectionError("content_unclassifiable", str(uuid4()))
        target.append(fragment)
        pending_bytes += fragment_bytes

    async def flush() -> AsyncIterator[Any]:
        nonlocal pending_bytes
        if pending_reasoning:
            reasoning = "".join(pending_reasoning)
            pending_reasoning.clear()
            if reasoning:
                await authorize_assistant(reasoning, context=context)
                yield {"type": "reasoning", "content": reasoning}
        if pending:
            text = "".join(pending)
            pending.clear()
            if text:
                await authorize_assistant(text, context=context)
                yield text
        pending_bytes = 0

    async for event in stream:
        terminal = _terminal_error()
        if terminal is not None:
            raise terminal
        if isinstance(event, str):
            if active_protection:
                append_fragment(pending, event)
            else:
                yield event
            continue
        if isinstance(event, dict) and event.get("type") == "reasoning":
            if active_protection:
                append_fragment(pending_reasoning, str(event.get("content") or ""))
            else:
                yield event
            continue
        async for released in flush():
            _ensure_active_attempt()
            yield released
        if isinstance(event, dict) and event.get("type") == "tool_start":
            tool_name = str(event.get("tool") or "")
            tool_id = tool_ids_by_name.get(tool_name) or tool_name
            tool_context, tool_required = await _tool_context(context, tool_id)
            await service.authorize_content(
                event.get("input") or {},
                direction="proposed_operation",
                context=tool_context,
                tool_id=tool_id,
                operation="tool_call",
            )
            _ensure_active_attempt()
            active_protection = active_protection or tool_required
        elif isinstance(event, dict) and active_protection:
            # Never send raw observer payloads (including on_tool_error output)
            # through the transport without an authorization boundary.
            await authorize_assistant(event, context=context)
            _ensure_active_attempt()
        terminal = _terminal_error()
        if terminal is not None:
            raise terminal
        yield event
    async for released in flush():
        _ensure_active_attempt()
        yield released
