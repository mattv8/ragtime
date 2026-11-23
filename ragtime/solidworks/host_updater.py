"""Update observed SolidWorks hosts when they are safe to stop."""

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Any

from ragtime.solidworks.host_activity import HostActivity, HostBusyError, host_activity, host_key

QUIET_SECONDS = 300
LOOP_SECONDS = 60
CHECK_SECONDS = 300
BACKOFF_SECONDS = (1800, 7200, 28800, 86400)
GATE_FEATURE = "stop-if-idle"
EXCLUSIVE_WAIT_SECONDS = 30
GENERIC_MESSAGE = "The automatic update could not run. Ragtime tries again later."
logger = logging.getLogger(__name__)


def get_host_artifact() -> Any:
    from ragtime.solidworks.host_artifact import get_host_artifact as load_artifact  # inline-import: keep

    return load_artifact()


def parse_host_version(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
        return None
    parts = value.split(".")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _enabled(config: dict[str, Any]) -> bool:
    return (config.get("connection_config") or {}).get("auto_update_host", True) is not False


@dataclass
class _HostState:
    tool_ids: set[str] = field(default_factory=set)
    installed_version: str | None = None
    target_version: str | None = None
    state: str = "pending"
    reason: str | None = None
    failures: int = 0
    next_attempt: float | None = None
    next_attempt_at: str | None = None
    last_attempt_at: str | None = None
    message: str | None = None
    urgent: bool = False
    urgent_checked: bool = False
    last_check: float | None = None
    failed_version: str | None = None


class SolidworksHostUpdater:
    def __init__(
        self,
        *,
        status_service_factory: Callable[[], Any],
        installer_factory: Callable[[], Any],
        list_tools: Callable[[], Awaitable[list[dict[str, Any]]]],
        activity: HostActivity,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        artifact: Callable[[], Any] = get_host_artifact,
    ) -> None:
        self.status_service_factory = status_service_factory
        self.installer_factory = installer_factory
        self.list_tools = list_tools
        self.activity = activity
        self.clock = clock
        self.now = now
        self.artifact = artifact
        self._states: dict[str, _HostState] = {}
        self._tool_keys: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None

    def _forget(self, key: str) -> None:
        state = self._states.pop(key, None)
        if state:
            for tool_id in state.tool_ids:
                if self._tool_keys.get(tool_id) == key:
                    self._tool_keys.pop(tool_id, None)

    def observe(self, config: dict[str, Any], host_version: Any) -> None:
        tool_id, key, packaged = config.get("id"), host_key(config), self.artifact()
        installed = parse_host_version(host_version)
        target = parse_host_version(packaged.host_version) if packaged else None
        if not isinstance(tool_id, str) or not tool_id or not key or installed is None or target is None or packaged is None:
            return
        old_key = self._tool_keys.get(tool_id)
        if old_key and old_key != key and old_key in self._states:
            self._states[old_key].tool_ids.discard(tool_id)
        self._tool_keys[tool_id] = key
        state = self._states.setdefault(key, _HostState())
        state.tool_ids.add(tool_id)
        versions_unchanged = (state.installed_version, state.target_version) == (host_version, packaged.host_version)
        state.installed_version, state.target_version = host_version, packaged.host_version
        if state.failed_version is not None and state.failed_version != packaged.host_version:
            state.failures = 0
            state.next_attempt = state.next_attempt_at = state.failed_version = None
        if not _enabled(config):
            state.state, state.reason = "off", None
        elif installed == target:
            state.state, state.reason = "current", None
            state.failures, state.urgent, state.message = 0, False, None
            state.next_attempt = state.next_attempt_at = state.failed_version = None
        elif installed > target:
            state.state, state.reason = "newer", None
        elif state.state != "updating" and not (state.state == "failed" and state.next_attempt is not None and self.clock() < state.next_attempt):
            # A status read must not erase why a pending update is waiting; the loop refreshes the reason.
            if state.state != "pending" or not versions_unchanged:
                state.reason = None
            state.state = "pending"

    def request_now(self, config: dict[str, Any]) -> None:
        key = host_key(config)
        if key and key in self._states:
            state = self._states[key]
            if not state.urgent:
                state.urgent_checked = False
            state.urgent = True

    def note_manual_action(self, config: dict[str, Any], status: dict[str, Any]) -> None:
        key = host_key(config)
        if not status.get("installed"):
            if key:
                self._forget(key)
        elif status.get("host_version"):
            self.observe(config, status["host_version"])

    def state_for(self, tool_id: str) -> dict[str, Any] | None:
        state = self._states.get(self._tool_keys.get(tool_id, ""))
        if state is None:
            return None
        return {
            "enabled": state.state != "off",
            "state": state.state,
            "installed_version": state.installed_version,
            "target_version": state.target_version,
            "reason": state.reason,
            "last_attempt_at": state.last_attempt_at,
            "next_attempt_at": state.next_attempt_at,
            "message": state.message,
        }

    def _failed(self, state: _HostState, message: str) -> None:
        state.failures += 1
        delay = BACKOFF_SECONDS[min(state.failures - 1, len(BACKOFF_SECONDS) - 1)]
        now = self.now()
        state.state, state.reason, state.message = "failed", None, message
        state.last_attempt_at = now.isoformat()
        state.next_attempt, state.next_attempt_at = self.clock() + delay, (now + timedelta(seconds=delay)).isoformat()
        state.failed_version = state.target_version

    async def run_once(self) -> None:
        tools = await self.list_tools()
        candidates = sorted(
            ((key, state) for key, state in self._states.items() if state.state in ("pending", "failed")),
            key=lambda item: not item[1].urgent,
        )
        for key, state in candidates:
            matching = [tool for tool in tools if tool.get("enabled", True) and tool.get("tool_type") == "solidworks" and host_key(tool) == key]
            if not matching:
                self._forget(key)
                continue
            for tool in matching:
                tool_id = tool.get("id")
                if isinstance(tool_id, str) and tool_id:
                    state.tool_ids.add(tool_id)
                    self._tool_keys[tool_id] = key
            if any(not _enabled(tool) for tool in matching):
                state.state, state.reason = "off", None
                continue
            packaged = self.artifact()
            target = parse_host_version(packaged.host_version) if packaged else None
            if target is None or packaged is None:
                continue
            if state.failed_version and state.failed_version != packaged.host_version:
                state.failures = 0
                state.next_attempt = state.next_attempt_at = state.failed_version = None
                state.target_version = packaged.host_version
            if state.next_attempt is not None and self.clock() < state.next_attempt:
                continue
            if state.last_check is not None and self.clock() - state.last_check < CHECK_SECONDS and not (state.urgent and not state.urgent_checked):
                continue
            snapshot = self.activity.snapshot(key)
            if snapshot.exclusive:
                continue
            if snapshot.in_flight > 0 or (not state.urgent and snapshot.idle_seconds is not None and snapshot.idle_seconds < QUIET_SECONDS):
                state.state, state.reason = "pending", "in_use"
                continue
            config = matching[0]
            try:
                await self.activity.run_exclusive(key, EXCLUSIVE_WAIT_SECONDS, partial(self._check_and_update, key, state, config, target))
            except HostBusyError:
                state.state, state.reason = "pending", "in_use"

    async def _check_and_update(self, key: str, state: _HostState, config: dict[str, Any], target: tuple[int, int, int]) -> None:
        snapshot = self.activity.snapshot(key)
        if snapshot.in_flight > 0 or (not state.urgent and snapshot.idle_seconds is not None and snapshot.idle_seconds < QUIET_SECONDS):
            state.state, state.reason = "pending", "in_use"
            return
        state.last_check = self.clock()
        if state.urgent:
            state.urgent_checked = True
        status = await self.status_service_factory().host_status(config, include_pdm=False)
        session = status.get("session") or {}
        if not status.get("reachable") or not status.get("installed") or not status.get("running"):
            state.state, state.reason = "pending", "logged_off" if session.get("interactive") is False else "not_running"
            return
        if status.get("instances_reported") is not True:
            state.state, state.reason = "pending", "in_use"
            return
        instances = (status.get("solidworks") or {}).get("instances")
        if not isinstance(instances, list) or any(not isinstance(item, dict) or type(item.get("busy")) is not bool for item in instances):
            state.state, state.reason = "pending", "in_use"
            return
        if any(item["busy"] for item in instances):
            state.state, state.reason = "pending", "in_use"
            return
        if any(
            not isinstance(item.get("mode"), str) or type(item.get("owned")) is not bool or type(item.get("documents")) is not int or item["documents"] < 0
            for item in instances
        ):
            state.state, state.reason = "pending", "in_use"
            return
        if any(item["mode"] == "background" and item["owned"] and item["documents"] > 0 for item in instances):
            state.state, state.reason = "pending", "documents_open"
            return
        host_activity_status = status.get("activity")
        if host_activity_status and (
            host_activity_status.get("in_flight", 0) > 0
            or host_activity_status.get("pdm_busy")
            or (not state.urgent and host_activity_status.get("idle_seconds", 0) < QUIET_SECONDS)
        ):
            state.state, state.reason = "pending", "in_use"
            return
        installed = parse_host_version(status.get("host_version"))
        if installed is None:
            return
        if installed >= target:
            self.observe(config, status["host_version"])
            return
        try:
            state.state, state.reason = "updating", None
            state.last_attempt_at = self.now().isoformat()
            logger.info("SolidWorks host update started: tool=%s installed=%s target=%s", config.get("id"), state.installed_version, state.target_version)
            gate = (0 if state.urgent else QUIET_SECONDS) if GATE_FEATURE in (status.get("host_features") or []) else None
            result = await self.installer_factory().install_host(config, idle_gate_seconds=gate)
            error = result.get("error") or {}
            if state.state == "current":
                pass
            elif result.get("running") and parse_host_version(result.get("host_version")) == target:
                self.observe(config, result["host_version"])
            elif error.get("code") == "host_busy":
                state.state, state.reason = "pending", "busy"
            else:
                self._failed(state, error.get("message") or GENERIC_MESSAGE)
                logger.warning("SolidWorks host update failed: tool=%s code=%s", config.get("id"), error.get("code", "unknown"))
            logger.info(
                "SolidWorks host update result: tool=%s installed=%s target=%s state=%s",
                config.get("id"),
                state.installed_version,
                state.target_version,
                state.state,
            )
        except asyncio.CancelledError:
            if state.state == "updating":
                state.state, state.reason = "pending", "busy"
                state.next_attempt = state.next_attempt_at = None
            raise
        except Exception as exc:
            if state.state == "updating":
                self._failed(state, GENERIC_MESSAGE)
                logger.warning("SolidWorks host update failed: tool=%s code=%s", config.get("id"), type(exc).__name__)
            else:
                raise
        finally:
            if state.state == "updating":
                self._failed(state, GENERIC_MESSAGE)

    async def _loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                logger.warning("SolidWorks host update loop failed: %s", type(exc).__name__)
            await asyncio.sleep(LOOP_SECONDS)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _list_tools() -> list[dict[str, Any]]:
    from ragtime.indexer.repository import repository  # inline-import: keep

    return [tool.model_dump() for tool in await repository.list_tool_configs(enabled_only=True)]


def _status_service() -> Any:
    from ragtime.solidworks.host_status import SolidworksHostStatusService  # inline-import: keep

    return SolidworksHostStatusService()


def _installer() -> Any:
    from ragtime.solidworks.host_installer import SolidworksHostInstaller  # inline-import: keep

    return SolidworksHostInstaller()


host_updater = SolidworksHostUpdater(
    status_service_factory=_status_service,
    installer_factory=_installer,
    list_tools=_list_tools,
    activity=host_activity,
)
