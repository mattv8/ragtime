import asyncio
import contextlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


TOOL_HEALTH_CHECK_INTERVAL_SECONDS = 30.0
TOOL_HEALTH_STALE_AFTER_SECONDS = TOOL_HEALTH_CHECK_INTERVAL_SECONDS * 3
TOOL_HEALTH_FAILURES_BEFORE_OFFLINE = 2
# Cold-start grace window for provisional statuses seeded from persisted test
# results. A process restart inherently pauses live heartbeats; during that gap
# we trust the last persisted DB state so default-all workspaces do not briefly
# resolve to zero tools. This window must comfortably exceed a slow container
# boot (image pull, Postgres init, migrations) so the seed survives until the
# first live heartbeat lands and overwrites it.
TOOL_HEALTH_COLD_START_GRACE_SECONDS = 600.0


@dataclass(frozen=True)
class ToolHeartbeatStatus:
    tool_id: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: datetime | None = None
    # True only for statuses seeded from persisted test results at startup.
    # Provisional statuses use the longer cold-start grace window for freshness
    # and are always overwritten by the first live heartbeat (which carries a
    # newer checked_at). They never survive a normal heartbeat cycle.
    provisional: bool = False

    def checked_at_iso(self) -> str:
        checked_at = self.checked_at or datetime.now(timezone.utc)
        return checked_at.isoformat()


@dataclass(frozen=True)
class ToolHealthCheckResult:
    statuses: dict[str, ToolHeartbeatStatus]
    changed_tool_ids: set[str]


def get_heartbeat_timeout_seconds(
    connection_config: dict[str, Any] | None,
    tool_type: str | Any | None = None,
) -> float:
    """Return the heartbeat timeout for a tool connection."""
    config = connection_config or {}
    has_ssh_tunnel = bool(config.get("ssh_tunnel_enabled", False))
    has_remote_docker = bool(config.get("docker_ssh_enabled", False))
    tool_type_value = getattr(tool_type, "value", tool_type)
    if tool_type_value == "ssh_shell":
        return 15.0
    return 15.0 if has_ssh_tunnel or has_remote_docker else 5.0


def _tool_attr(tool: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool, dict):
        return tool.get(name, default)
    return getattr(tool, name, default)


class ToolHealthMonitor:
    """Background heartbeat cache for runtime tool availability."""

    def __init__(
        self,
        *,
        interval_seconds: float = TOOL_HEALTH_CHECK_INTERVAL_SECONDS,
        stale_after_seconds: float = TOOL_HEALTH_STALE_AFTER_SECONDS,
        failures_before_offline: int = TOOL_HEALTH_FAILURES_BEFORE_OFFLINE,
        cold_start_grace_seconds: float = TOOL_HEALTH_COLD_START_GRACE_SECONDS,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.failures_before_offline = max(1, failures_before_offline)
        self.cold_start_grace_seconds = max(stale_after_seconds, cold_start_grace_seconds)
        self._statuses: dict[str, ToolHeartbeatStatus] = {}
        self._failure_counts: dict[str, int] = {}
        self._subscribers: set[asyncio.Queue[ToolHealthCheckResult]] = set()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._seeded = False

    def get_status(self, tool_id: str | None) -> ToolHeartbeatStatus | None:
        if not tool_id:
            return None
        return self._statuses.get(tool_id)

    def get_statuses(self) -> dict[str, ToolHeartbeatStatus]:
        return dict(self._statuses)

    async def subscribe(self) -> asyncio.Queue[ToolHealthCheckResult]:
        queue: asyncio.Queue[ToolHealthCheckResult] = asyncio.Queue(maxsize=10)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue[ToolHealthCheckResult]) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def is_status_fresh(
        self,
        status: ToolHeartbeatStatus | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        if status is None or status.checked_at is None:
            return False
        reference = now or datetime.now(timezone.utc)
        checked_at = status.checked_at
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        limit = self.cold_start_grace_seconds if status.provisional else self.stale_after_seconds
        return (reference - checked_at).total_seconds() <= limit

    def is_tool_healthy(self, tool_id: str | None) -> bool:
        status = self.get_status(tool_id)
        return bool(status and status.alive and self.is_status_fresh(status))

    def get_unavailable_reason(self, tool_id: str | None) -> str | None:
        if not tool_id:
            return "Missing tool id"
        status = self.get_status(tool_id)
        if status is None:
            return "No recent heartbeat"
        if not self.is_status_fresh(status):
            return "Heartbeat stale"
        if not status.alive:
            return status.error or "Heartbeat failed"
        return None

    def filter_healthy_tool_config_dicts(self, tool_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [config for config in tool_configs if self.is_tool_healthy(str(config.get("id") or ""))]

    def healthy_tool_ids_for_configs(self, tool_configs: list[Any]) -> list[str]:
        healthy_ids: list[str] = []
        for tool in tool_configs:
            tool_id = str(_tool_attr(tool, "id", "") or "")
            if tool_id and self.is_tool_healthy(tool_id):
                healthy_ids.append(tool_id)
        return healthy_ids

    def _coerce_persisted_status(self, tool: Any, *, now: datetime) -> ToolHeartbeatStatus | None:
        """Build a provisional status from a tool's persisted test result.

        Returns None when the tool has no usable persisted result or when that
        result is older than the cold-start grace window. Runs only at startup
        seeding time, never on the request hot path.
        """
        tool_id = str(_tool_attr(tool, "id", "") or "")
        if not tool_id:
            return None

        last_test_result = _tool_attr(tool, "last_test_result", None)
        if last_test_result is None:
            last_test_result = _tool_attr(tool, "lastTestResult", None)
        last_test_at = _tool_attr(tool, "last_test_at", None)
        if last_test_at is None:
            last_test_at = _tool_attr(tool, "lastTestAt", None)
        if last_test_result is None or last_test_at is None:
            return None

        if isinstance(last_test_at, str):
            try:
                last_test_at = datetime.fromisoformat(last_test_at)
            except ValueError:
                return None
        if not isinstance(last_test_at, datetime):
            return None
        if last_test_at.tzinfo is None:
            last_test_at = last_test_at.replace(tzinfo=timezone.utc)
        if (now - last_test_at).total_seconds() > self.cold_start_grace_seconds:
            return None

        last_test_error = _tool_attr(tool, "last_test_error", None)
        if last_test_error is None:
            last_test_error = _tool_attr(tool, "lastTestError", None)
        # Clamp checked_at to now so a backward host-clock shift across the
        # restart can never stamp the seed in the future. A future-dated seed
        # would otherwise out-rank the first live heartbeat in the
        # newest-wins guard in _store_statuses and pin a stale state.
        seed_checked_at = min(last_test_at, now)
        return ToolHeartbeatStatus(
            tool_id=tool_id,
            alive=bool(last_test_result),
            error=None if last_test_result else last_test_error,
            checked_at=seed_checked_at,
            provisional=True,
        )

    async def seed_from_persisted_results(self, tool_configs: list[Any] | None = None) -> dict[str, ToolHeartbeatStatus]:
        """Warm-start in-memory health from persisted test results, exactly once.

        Must be awaited during startup BEFORE the first reader (e.g.
        ``rag.initialize()``) consults tool availability, so default-all
        workspaces never observe an empty health cache. Idempotent: subsequent
        calls are no-ops. Seeds only tools that have no live status yet, so a
        heartbeat that landed first is never clobbered. Seeded entries are
        provisional and are superseded by the first live heartbeat.
        """
        if self._seeded:
            return {}

        if tool_configs is None:
            from ragtime.indexer.repository import repository

            try:
                tool_configs = await repository.list_tool_configs(enabled_only=True)
            except Exception:
                logger.warning("Tool health cold-start seed skipped: failed to load tool configs", exc_info=True)
                return {}

        now = datetime.now(timezone.utc)
        seeded: dict[str, ToolHeartbeatStatus] = {}
        async with self._lock:
            if self._seeded:
                return {}
            for tool in tool_configs:
                tool_id = str(_tool_attr(tool, "id", "") or "")
                if not tool_id or tool_id in self._statuses:
                    continue
                status = self._coerce_persisted_status(tool, now=now)
                if status is None:
                    continue
                self._statuses[tool_id] = status
                seeded[tool_id] = status
            self._seeded = True

        if seeded:
            logger.info(
                "Cold-start seeded %d tool heartbeat status(es) from persisted results",
                len(seeded),
            )
        return seeded

    async def check_once(self, tool_configs: list[Any] | None = None) -> ToolHealthCheckResult:
        """Refresh heartbeat status for enabled tools and persist the latest result."""
        if tool_configs is None:
            from ragtime.indexer.repository import repository

            tool_configs = await repository.list_tool_configs(enabled_only=True)

        if not tool_configs:
            return ToolHealthCheckResult(statuses={}, changed_tool_ids=set())

        try:
            from ragtime.indexer.routes import _heartbeat_check
        except Exception as exc:
            logger.warning("Heartbeat helper unavailable: %s", exc)
            now = datetime.now(timezone.utc)
            failure_statuses = {
                str(_tool_attr(tool, "id", "") or ""): ToolHeartbeatStatus(
                    tool_id=str(_tool_attr(tool, "id", "") or ""),
                    alive=False,
                    error="Heartbeat checker unavailable",
                    checked_at=now,
                )
                for tool in tool_configs
                if _tool_attr(tool, "id", None)
            }
            return await self._store_statuses(failure_statuses)

        async def check_single_tool(tool: Any) -> ToolHeartbeatStatus | None:
            tool_id = str(_tool_attr(tool, "id", "") or "")
            if not tool_id:
                return None

            connection_config = _tool_attr(tool, "connection_config", {}) or {}
            tool_type = _tool_attr(tool, "tool_type", "")
            heartbeat_timeout = get_heartbeat_timeout_seconds(connection_config, tool_type)
            start_time = time.monotonic()
            checked_at = datetime.now(timezone.utc)

            try:
                result = await asyncio.wait_for(
                    _heartbeat_check(tool_type, connection_config),
                    timeout=heartbeat_timeout,
                )
                latency_ms = (time.monotonic() - start_time) * 1000
                return ToolHeartbeatStatus(
                    tool_id=tool_id,
                    alive=bool(result.success),
                    latency_ms=round(latency_ms, 1) if result.success else None,
                    error=result.message if not result.success else None,
                    checked_at=checked_at,
                )
            except asyncio.TimeoutError:
                return ToolHeartbeatStatus(
                    tool_id=tool_id,
                    alive=False,
                    error=f"Heartbeat timeout ({int(heartbeat_timeout)}s)",
                    checked_at=checked_at,
                )
            except Exception as exc:
                logger.debug("Heartbeat check failed for tool %s", tool_id, exc_info=True)
                return ToolHeartbeatStatus(
                    tool_id=tool_id,
                    alive=False,
                    error=str(exc) or exc.__class__.__name__,
                    checked_at=checked_at,
                )

        results = await asyncio.gather(
            *(check_single_tool(tool) for tool in tool_configs),
            return_exceptions=True,
        )
        statuses: dict[str, ToolHeartbeatStatus] = {}
        for result in results:
            if isinstance(result, ToolHeartbeatStatus):
                statuses[result.tool_id] = result
            elif isinstance(result, BaseException):
                logger.debug("Tool heartbeat task failed: %s", result)

        return await self._store_statuses(statuses)

    async def _store_statuses(
        self,
        statuses: dict[str, ToolHeartbeatStatus],
        *,
        require_consecutive_failures: bool = True,
    ) -> ToolHealthCheckResult:
        changed_tool_ids: set[str] = set()
        now = datetime.now(timezone.utc)
        subscribers: list[asyncio.Queue[ToolHealthCheckResult]] = []
        stored_statuses: dict[str, ToolHeartbeatStatus] = {}
        async with self._lock:
            for tool_id, status in statuses.items():
                previous = self._statuses.get(tool_id)
                if previous and previous.checked_at and status.checked_at:
                    previous_checked_at = previous.checked_at
                    status_checked_at = status.checked_at
                    if previous_checked_at.tzinfo is None:
                        previous_checked_at = previous_checked_at.replace(tzinfo=timezone.utc)
                    if status_checked_at.tzinfo is None:
                        status_checked_at = status_checked_at.replace(tzinfo=timezone.utc)
                    if status_checked_at < previous_checked_at:
                        continue
                # Observed health reflects what consumers see (includes a fresh
                # provisional seed) and drives change notifications.
                previous_observed_healthy = bool(previous and previous.alive and self.is_status_fresh(previous, now=now))
                if status.alive:
                    self._failure_counts.pop(tool_id, None)
                elif previous is not None and previous_observed_healthy and not previous.provisional and require_consecutive_failures:
                    # Hysteresis only smooths flaps between CONFIRMED live states.
                    # A provisional seed is a guess, so the first live failure
                    # must replace it immediately rather than earn extra grace.
                    failure_count = self._failure_counts.get(tool_id, 0) + 1
                    self._failure_counts[tool_id] = failure_count
                    if failure_count < self.failures_before_offline:
                        logger.info(
                            "Ignoring transient heartbeat failure %d/%d for tool %s: %s",
                            failure_count,
                            self.failures_before_offline,
                            tool_id,
                            status.error or "Heartbeat failed",
                        )
                        continue
                current_healthy = bool(status.alive and self.is_status_fresh(status, now=now))
                if previous is None or previous_observed_healthy != current_healthy:
                    changed_tool_ids.add(tool_id)
                self._statuses[tool_id] = status
                stored_statuses[tool_id] = status
            if changed_tool_ids:
                subscribers = list(self._subscribers)

        if stored_statuses:
            await self._persist_statuses(stored_statuses)
        result = ToolHealthCheckResult(statuses=stored_statuses, changed_tool_ids=changed_tool_ids)
        if changed_tool_ids:
            for queue in subscribers:
                if queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(result)
        return result

    async def record_tool_test_result(
        self,
        tool_id: str,
        *,
        success: bool,
        message: str | None = None,
        latency_ms: float | None = None,
        checked_at: datetime | None = None,
    ) -> ToolHealthCheckResult:
        """Record an explicit saved-tool test result in the shared health cache."""
        if not tool_id:
            return ToolHealthCheckResult(statuses={}, changed_tool_ids=set())

        status = ToolHeartbeatStatus(
            tool_id=tool_id,
            alive=success,
            latency_ms=latency_ms if success else None,
            error=None if success else message,
            checked_at=checked_at or datetime.now(timezone.utc),
        )
        return await self._store_statuses({tool_id: status}, require_consecutive_failures=False)

    async def _persist_statuses(self, statuses: dict[str, ToolHeartbeatStatus]) -> None:
        if not statuses:
            return

        from ragtime.indexer.repository import repository

        await asyncio.gather(
            *(
                repository.update_tool_test_result(
                    tool_id,
                    success=status.alive,
                    error=status.error if not status.alive else None,
                )
                for tool_id, status in statuses.items()
            ),
            return_exceptions=True,
        )

    def start(
        self,
        *,
        on_change: Callable[[ToolHealthCheckResult], Awaitable[None]] | None = None,
    ) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(on_change=on_change))

    async def stop(self) -> None:
        task = self._task
        if not task:
            return
        if self._stop_event:
            self._stop_event.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None
        self._stop_event = None

    async def _run(
        self,
        *,
        on_change: Callable[[ToolHealthCheckResult], Awaitable[None]] | None,
    ) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            return

        while not stop_event.is_set():
            try:
                result = await self.check_once()
                if result.changed_tool_ids and on_change:
                    await on_change(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Background tool heartbeat check failed", exc_info=True)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass


tool_health_monitor = ToolHealthMonitor()
