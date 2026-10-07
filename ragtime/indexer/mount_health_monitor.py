import asyncio
import contextlib
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from fastapi import HTTPException

from ragtime.core.logging import get_logger
from ragtime.core.mount_health import MountHealthChecker, MountHealthEntry, redact_mount_source
from ragtime.core.runtime_manager_client import runtime_manager_enabled, runtime_manager_request
from ragtime.indexer.models import MountHealthStatus, MountProblem

logger = get_logger(__name__)

MOUNT_HEALTH_CHECK_INTERVAL_SECONDS = 60.0
ERROR_LOG_INTERVAL_SECONDS = 600.0
RUNTIME_STARTUP_ERROR_GRACE_SECONDS = 300.0


class MountHealthMonitor:
    """Cached health for mounts visible to Ragtime and its runtime manager."""

    def __init__(
        self,
        *,
        checker: MountHealthChecker | None = None,
        interval_seconds: float = MOUNT_HEALTH_CHECK_INTERVAL_SECONDS,
        runtime_enabled: Callable[[], bool] = runtime_manager_enabled,
        runtime_request: Callable[..., Awaitable[Any]] = runtime_manager_request,
    ) -> None:
        self._checker = checker or MountHealthChecker()
        self._interval_seconds = interval_seconds
        self._runtime_enabled = runtime_enabled
        self._runtime_request = runtime_request
        self._lock = asyncio.Lock()
        self._cycle_task: asyncio.Task[MountHealthStatus] | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._status = MountHealthStatus(status="unknown", checked_at=None, runtime_checked=False, problems=[])
        self._reported: dict[tuple[Literal["ragtime", "runtime"], str], MountProblem] = {}
        self._runtime_problems: list[MountProblem] = []
        self._last_runtime_error_log = 0.0
        self._last_background_error_log = 0.0
        self._runtime_reached = False
        self._started_at: float | None = None

    def snapshot(self) -> MountHealthStatus:
        """Return the cached status without probing mounts."""
        return self._status.model_copy(deep=True)

    def start(self) -> None:
        """Start background checks without delaying application startup."""
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._started_at = time.monotonic()
        self._task = asyncio.create_task(self._run(), name="mount-health-monitor")

    async def stop(self) -> None:
        """Stop the periodic task cleanly."""
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None
        self._stop_event = None

    async def recheck(self) -> MountHealthStatus:
        """Run an immediate serialized local and runtime health check."""
        return await self._check_once(runtime_recheck=True)

    async def _run(self) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return
        while not stop_event.is_set():
            try:
                await self._check_once(runtime_recheck=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_background_error(exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval_seconds)
            except asyncio.TimeoutError:
                pass

    async def _check_once(self, *, runtime_recheck: bool) -> MountHealthStatus:
        async with self._lock:
            if self._cycle_task is None:
                self._cycle_task = asyncio.create_task(self._run_cycle(runtime_recheck=runtime_recheck))
            task = self._cycle_task
        return await asyncio.shield(task)

    async def _run_cycle(self, *, runtime_recheck: bool) -> MountHealthStatus:
        try:
            local_entries = await self._checker.check_async()
            runtime_entries: list[dict[str, Any]] = []
            runtime_checked = False
            if self._runtime_enabled():
                runtime_entries, runtime_checked = await self._runtime_entries(recheck=runtime_recheck)

            problems = self._problems_from_entries("ragtime", local_entries)
            if runtime_checked:
                self._runtime_problems = self._problems_from_entries("runtime", runtime_entries)
            problems.extend(self._runtime_problems)
            checked_at = self._checked_at(local_entries, runtime_entries)
            self._log_transitions(problems)
            self._status = MountHealthStatus(
                status="degraded" if problems else "ok",
                checked_at=checked_at,
                runtime_checked=runtime_checked,
                problems=problems,
            )
            return self.snapshot()
        finally:
            async with self._lock:
                if self._cycle_task is asyncio.current_task():
                    self._cycle_task = None

    async def _runtime_entries(self, *, recheck: bool) -> tuple[list[dict[str, Any]], bool]:
        try:
            response = await self._runtime_request(
                "POST" if recheck else "GET",
                "/mounts/health/recheck" if recheck else "/mounts/health",
                timeout_override_seconds=30 if recheck else 15,
                retry_safe=False,
                surface_error_status=True,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                logger.debug("Runtime mount health endpoint is unavailable on this runtime version")
            else:
                self._log_runtime_error("Runtime mount health check failed", exc)
            return [], False
        except Exception as exc:
            self._log_runtime_error("Runtime mount health check failed", exc)
            return [], False
        if not isinstance(response, dict) or response.get("checked_at") is None:
            return [], False
        mounts = response.get("mounts")
        self._runtime_reached = True
        return ([entry for entry in mounts if isinstance(entry, dict)] if isinstance(mounts, list) else []), True

    def _log_runtime_error(self, message: str, error: Exception) -> None:
        now = time.monotonic()
        started_at = self._started_at
        if not self._runtime_reached and (started_at is None or now - started_at < RUNTIME_STARTUP_ERROR_GRACE_SECONDS):
            logger.debug("%s: %s", message, error)
        elif now - self._last_runtime_error_log >= ERROR_LOG_INTERVAL_SECONDS:
            logger.warning("%s: %s", message, error)
            self._last_runtime_error_log = now
        else:
            logger.debug("%s: %s", message, error)

    def _log_background_error(self, error: Exception) -> None:
        now = time.monotonic()
        if now - self._last_background_error_log >= ERROR_LOG_INTERVAL_SECONDS:
            logger.warning("Background mount health check failed", exc_info=error)
            self._last_background_error_log = now
        else:
            logger.debug("Background mount health check failed", exc_info=error)

    @staticmethod
    def _problems_from_entries(container: Literal["ragtime", "runtime"], entries: list[MountHealthEntry] | list[dict[str, Any]]) -> list[MountProblem]:
        problems: list[MountProblem] = []
        for entry in entries:
            value = entry.to_dict() if isinstance(entry, MountHealthEntry) else entry
            if not value.get("reported") or value.get("state") not in {"failed", "unresponsive"}:
                continue
            mount_point = value.get("mount_point")
            fstype = value.get("fstype")
            source = value.get("source")
            if not isinstance(mount_point, str) or not isinstance(fstype, str) or not isinstance(source, str):
                continue
            failing_since = _parse_utc(value.get("failing_since"))
            error = value.get("error")
            problems.append(
                MountProblem(
                    container=container,
                    mount_point=mount_point,
                    fstype=fstype,
                    source=redact_mount_source(source),
                    state=value["state"],
                    error=error if isinstance(error, str) else None,
                    failing_since=failing_since,
                )
            )
        return problems

    def _checked_at(self, local_entries: list[MountHealthEntry], runtime_entries: list[dict[str, Any]]) -> datetime | None:
        checked_at = [entry.checked_at for entry in local_entries]
        for entry in runtime_entries:
            parsed = _parse_utc(entry.get("checked_at"))
            if parsed is not None:
                checked_at.append(parsed)
        return max(checked_at) if checked_at else datetime.now(timezone.utc)

    def _log_transitions(self, problems: list[MountProblem]) -> None:
        current: dict[tuple[Literal["ragtime", "runtime"], str], MountProblem] = {(problem.container, problem.mount_point): problem for problem in problems}
        for key, problem in current.items():
            if key not in self._reported:
                logger.warning("Mount health problem in %s at %s (source %s)", problem.container, problem.mount_point, problem.source)
        for key, problem in self._reported.items():
            if key not in current:
                logger.info("Mount health recovered in %s at %s (source %s)", problem.container, problem.mount_point, problem.source)
        self._reported = current


def _parse_utc(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


mount_health_monitor = MountHealthMonitor()
