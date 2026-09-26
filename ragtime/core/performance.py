"""Payload-free request and operation timing helpers."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, ParamSpec, TypeVar, cast

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_LOGGER = logging.getLogger("ragtime.performance")
_REQUEST_STATE: contextvars.ContextVar[_RequestState | None] = contextvars.ContextVar("ragtime_performance_request", default=None)
_MAX_OPERATIONS = 32
_MONITOR_INTERVAL_SECONDS = 0.5

# Hardcoded thresholds in seconds; zero disables each warning type
_SLOW_REQUEST_THRESHOLD_SECONDS = 1.0
_SLOW_OPERATION_THRESHOLD_SECONDS = 1.0
_EVENT_LOOP_LAG_THRESHOLD_SECONDS = 0.5

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class PerformanceSettings:
    """Warning thresholds in seconds; zero disables each warning type."""

    slow_request_threshold_seconds: float = _SLOW_REQUEST_THRESHOLD_SECONDS
    slow_operation_threshold_seconds: float = _SLOW_OPERATION_THRESHOLD_SECONDS
    event_loop_lag_threshold_seconds: float = _EVENT_LOOP_LAG_THRESHOLD_SECONDS


_DEFAULT_SETTINGS = PerformanceSettings()


@dataclass(slots=True)
class _OperationTotal:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0


@dataclass(slots=True)
class _RequestState:
    request_id: str
    service: str
    settings: PerformanceSettings
    operations: dict[str, _OperationTotal] = field(default_factory=dict)
    closed: bool = False
    client_disconnected: bool = False

    def add_operation(self, name: str, elapsed: float) -> None:
        if self.closed:
            return
        total = self.operations.get(name)
        if total is None:
            if len(self.operations) >= _MAX_OPERATIONS:
                return
            total = self.operations[name] = _OperationTotal()
        total.count += 1
        total.total += elapsed
        total.maximum = max(total.maximum, elapsed)


def get_request_id() -> str | None:
    """Return the generated ID for the currently instrumented request."""
    state = _REQUEST_STATE.get()
    return state.request_id if state is not None and not state.closed else None


@contextmanager
def track_operation(name: str) -> Generator[None, None, None]:
    """Measure a developer-named operation and aggregate it into its request."""
    started = monotonic()
    outcome = "success"
    exception_type: str | None = None
    try:
        yield
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except BaseException as exc:
        outcome = "failed"
        exception_type = type(exc).__name__
        raise
    finally:
        elapsed = max(0.0, monotonic() - started)
        state = _REQUEST_STATE.get()
        if state is not None and not state.closed:
            state.add_operation(name, elapsed)
            if _exceeds(elapsed, state.settings.slow_operation_threshold_seconds):
                _log_operation_warning(name, elapsed, state.service, state.request_id, outcome, exception_type)
        else:
            settings = state.settings if state is not None else _DEFAULT_SETTINGS
            service = state.service if state is not None else "ragtime"
            request_id = state.request_id if state is not None else None
            if _exceeds(elapsed, settings.slow_operation_threshold_seconds):
                _log_operation_warning(name, elapsed, service, request_id, outcome, exception_type)


def timed_operation(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate synchronous or asynchronous functions with ``track_operation``."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(function):
            async_function = cast(Callable[P, Awaitable[Any]], function)

            @functools.wraps(function)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with track_operation(name):
                    return await async_function(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(function)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with track_operation(name):
                return function(*args, **kwargs)

        return sync_wrapper

    return decorate


class SlowRequestMiddleware:
    """Pure ASGI middleware that measures request start and final-body latency."""

    def __init__(self, app: ASGIApp, *, service: str = "ragtime", settings: PerformanceSettings | None = None) -> None:
        self.app = app
        self.service = service
        self.settings = settings or _DEFAULT_SETTINGS
        self._lag_monitor_task: asyncio.Task[None] | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(scope, receive, send)
        elif scope["type"] == "http":
            await self._http(scope, receive, send)
        else:
            await self.app(scope, receive, send)

    async def _http(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = monotonic()
        state = _RequestState(uuid.uuid4().hex, self.service, self.settings)
        token = _REQUEST_STATE.set(state)
        completed = False
        status: int | None = None
        ttfb: float | None = None
        is_sse = False

        async def monitored_receive() -> Message:
            """Wrap receive to detect client disconnect."""
            message = await receive()
            if message["type"] == "http.disconnect":
                state.client_disconnected = True
            return message

        async def timed_send(message: Message) -> None:
            nonlocal completed, status, ttfb, is_sse
            message_type = message["type"]
            if message_type == "http.response.start":
                now = monotonic()
                status = cast(int, message.get("status"))
                ttfb = max(0.0, now - started)
                is_sse = _is_sse(message)
                if _exceeds(ttfb, self.settings.slow_request_threshold_seconds):
                    self._log_request("slow response start", state, scope, ttfb, ttfb, status, "started", logging.WARNING)
            await send(message)
            if message_type == "http.response.body" and not message.get("more_body", False):
                completed = True
                elapsed = max(0.0, monotonic() - started)
                level = logging.DEBUG if is_sse or not _exceeds(elapsed, self.settings.slow_request_threshold_seconds) else logging.WARNING
                label = "sse request complete" if is_sse else "request complete"
                self._log_request(label, state, scope, elapsed, ttfb, status, "completed", level)
                state.closed = True

        try:
            await self.app(scope, monitored_receive, timed_send)
            if not completed:
                elapsed = max(0.0, monotonic() - started)
                if state.client_disconnected:
                    level = logging.DEBUG if is_sse or not _exceeds(elapsed, self.settings.slow_request_threshold_seconds) else logging.WARNING
                    self._log_request("client disconnected", state, scope, elapsed, ttfb, status, "disconnected", level)
                else:
                    self._log_request(
                        "request ended without final body",
                        state,
                        scope,
                        elapsed,
                        ttfb,
                        status,
                        "incomplete",
                        logging.WARNING,
                    )
        except asyncio.CancelledError:
            elapsed = max(0.0, monotonic() - started)
            self._log_request(
                "background/app cancellation" if completed else "request cancelled",
                state,
                scope,
                elapsed,
                ttfb,
                status,
                "cancelled",
                logging.WARNING,
            )
            raise
        except BaseException:
            elapsed = max(0.0, monotonic() - started)
            self._log_request(
                "background/app failure" if completed else "request failed",
                state,
                scope,
                elapsed,
                ttfb,
                status,
                "failed",
                logging.ERROR,
            )
            raise
        finally:
            state.closed = True
            _REQUEST_STATE.reset(token)

    async def _lifespan(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def monitored_send(message: Message) -> None:
            if message["type"] == "lifespan.startup.complete":
                self._start_lag_monitor()
            await send(message)
            if message["type"] in {"lifespan.shutdown.complete", "lifespan.shutdown.failed"}:
                await self._stop_lag_monitor()

        try:
            await self.app(scope, receive, monitored_send)
        finally:
            await self._stop_lag_monitor()

    def _start_lag_monitor(self) -> None:
        if self.settings.event_loop_lag_threshold_seconds > 0 and (self._lag_monitor_task is None or self._lag_monitor_task.done()):
            self._lag_monitor_task = asyncio.create_task(self._monitor_lag())

    async def _stop_lag_monitor(self) -> None:
        task, self._lag_monitor_task = self._lag_monitor_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _monitor_lag(self) -> None:
        while True:
            started = monotonic()
            await asyncio.sleep(_MONITOR_INTERVAL_SECONDS)
            lag = max(0.0, monotonic() - started - _MONITOR_INTERVAL_SECONDS)
            if _exceeds(lag, self.settings.event_loop_lag_threshold_seconds):
                if _LOGGER.isEnabledFor(logging.WARNING):
                    pid = os.getpid()
                    _LOGGER.warning(
                        "event loop lag service=%s pid=%s lag=%.3fs performance=%s",
                        self.service,
                        pid,
                        lag,
                        json.dumps({"service": self.service, "pid": pid, "lag": lag}, sort_keys=True),
                    )

    def _log_request(
        self,
        label: str,
        state: _RequestState,
        scope: Scope,
        elapsed: float,
        ttfb: float | None,
        status: int | None,
        outcome: str,
        level: int,
    ) -> None:
        if not _LOGGER.isEnabledFor(level):
            return
        operations = {name: {"count": total.count, "total": round(total.total, 6), "max": round(total.maximum, 6)} for name, total in state.operations.items()}
        pid = os.getpid()
        performance = {
            "service": self.service,
            "pid": pid,
            "request_id": state.request_id,
            "method": scope.get("method", "<unknown>"),
            "route": _route_template(scope),
            "elapsed": round(elapsed, 6),
            "ttfb": None if ttfb is None else round(ttfb, 6),
            "status": status,
            "outcome": outcome,
            "operations": operations,
        }
        _LOGGER.log(
            level,
            "%s service=%s pid=%s request_id=%s method=%s route=%s elapsed=%.3fs ttfb=%s status=%s outcome=%s summary=%s",
            label,
            self.service,
            pid,
            state.request_id,
            performance["method"],
            performance["route"],
            elapsed,
            "none" if ttfb is None else f"{ttfb:.3f}s",
            status,
            outcome,
            json.dumps(performance, sort_keys=True),
            extra={"performance": performance},
        )


def _exceeds(elapsed: float, threshold: float) -> bool:
    return threshold > 0 and elapsed >= threshold


def _log_operation_warning(
    name: str,
    elapsed: float,
    service: str,
    request_id: str | None,
    outcome: str,
    exception_type: str | None,
) -> None:
    if not _LOGGER.isEnabledFor(logging.WARNING):
        return
    pid = os.getpid()
    summary: dict[str, Any] = {
        "operation": name,
        "elapsed": elapsed,
        "service": service,
        "pid": pid,
        "request_id": request_id,
        "outcome": outcome,
    }
    if exception_type is not None:
        summary["exception_type"] = exception_type
    _LOGGER.warning(
        "slow operation name=%s elapsed=%.3fs service=%s pid=%s request_id=%s outcome=%s summary=%s",
        name,
        elapsed,
        service,
        pid,
        request_id or "none",
        outcome,
        json.dumps(summary, sort_keys=True),
        extra={"performance": summary},
    )


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    for attribute in ("path_format", "path"):
        value = getattr(route, attribute, None)
        if isinstance(value, str):
            return value
    return "<unmatched>"


def _is_sse(message: Message) -> bool:
    for name, value in message.get("headers", ()):
        if name.lower() == b"content-type" and value.lower().startswith(b"text/event-stream"):
            return True
    return False
