
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


@dataclass(frozen=True)
class ToolHeartbeatStatus:
    tool_id: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: datetime | None = None

    def checked_at_iso(self) -> str:
        checked_at = self.checked_at or datetime.now(timezone.utc)
        return checked_at.isoformat()


@dataclass(frozen=True)
class ToolHealthCheckResult:
    statuses: dict[str, ToolHeartbeatStatus]
    changed_tool_ids: set[str]


def get_heartbeat_timeout_seconds(connection_config: dict[str, Any] | None) -> float:
    """Return the heartbeat timeout for a tool connection."""
    has_ssh_tunnel = bool((connection_config or {}).get("ssh_tunnel_enabled", False))
    return 15.0 if has_ssh_tunnel else 5.0


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
    ) -> None:
        self.interval_seconds = interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self._statuses: dict[str, ToolHeartbeatStatus] = {}
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    def get_status(self, tool_id: str | None) -> ToolHeartbeatStatus | None:
        if not tool_id:
            return None
        return self._statuses.get(tool_id)

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
        return (reference - checked_at).total_seconds() <= self.stale_after_seconds

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

    def filter_healthy_tool_config_dicts(
        self, tool_configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            config
            for config in tool_configs
            if self.is_tool_healthy(str(config.get("id") or ""))
        ]

    def healthy_tool_ids_for_configs(self, tool_configs: list[Any]) -> list[str]:
        healthy_ids: list[str] = []
        for tool in tool_configs:
            tool_id = str(_tool_attr(tool, "id", "") or "")
            if tool_id and self.is_tool_healthy(tool_id):
                healthy_ids.append(tool_id)
        return healthy_ids

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
            heartbeat_timeout = get_heartbeat_timeout_seconds(connection_config)
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
        self, statuses: dict[str, ToolHeartbeatStatus]
    ) -> ToolHealthCheckResult:
        changed_tool_ids: set[str] = set()
        now = datetime.now(timezone.utc)
        async with self._lock:
            for tool_id, status in statuses.items():
                previous = self._statuses.get(tool_id)
                previous_healthy = bool(
                    previous and previous.alive and self.is_status_fresh(previous, now=now)
                )
                current_healthy = bool(
                    status.alive and self.is_status_fresh(status, now=now)
                )
                if previous is None or previous_healthy != current_healthy:
                    changed_tool_ids.add(tool_id)
                self._statuses[tool_id] = status

        await self._persist_statuses(statuses)
        return ToolHealthCheckResult(statuses=statuses, changed_tool_ids=changed_tool_ids)

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
        return await self._store_statuses({tool_id: status})

    async def _persist_statuses(
        self, statuses: dict[str, ToolHeartbeatStatus]
    ) -> None:
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
