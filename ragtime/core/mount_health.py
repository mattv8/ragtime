"""Bounded, dependency-free mount responsiveness checks."""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

NETWORK_FSTYPES: frozenset[str] = frozenset({"cifs", "smb3", "nfs", "nfs4", "fuse.sshfs"})
DEFAULT_EXCLUDE_PREFIXES: tuple[str, ...] = ("/proc", "/sys", "/dev")
MountState = Literal["ok", "failed", "unresponsive"]


@dataclass(frozen=True)
class MountInfoEntry:
    """The mount details needed for health checks."""

    mount_id: int
    parent_id: int
    mount_point: str
    fstype: str
    source: str


@dataclass(frozen=True)
class MountHealthEntry:
    """The latest health result for one mount point."""

    mount_point: str
    fstype: str
    source: str
    state: MountState
    error: str | None
    failing_since: datetime | None
    consecutive_failures: int
    reported: bool
    recovered: bool
    checked_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        return {
            "mount_point": self.mount_point,
            "fstype": self.fstype,
            "source": self.source,
            "state": self.state,
            "error": self.error,
            "failing_since": self.failing_since.isoformat() if self.failing_since else None,
            "consecutive_failures": self.consecutive_failures,
            "reported": self.reported,
            "recovered": self.recovered,
            "checked_at": self.checked_at.isoformat(),
        }


def _unescape_mountinfo(value: str) -> str:
    for escaped, character in ((r"\040", " "), (r"\011", "\t"), (r"\012", "\n"), (r"\134", "\\")):
        value = value.replace(escaped, character)
    return value


def parse_mountinfo(text: str) -> list[MountInfoEntry]:
    """Parse Linux mountinfo text, ignoring malformed records."""
    entries: list[MountInfoEntry] = []
    for line in text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            if separator < 6 or len(fields) < separator + 3:
                continue
            entries.append(
                MountInfoEntry(
                    mount_id=int(fields[0]),
                    parent_id=int(fields[1]),
                    mount_point=_unescape_mountinfo(fields[4]),
                    fstype=fields[separator + 1],
                    source=fields[separator + 2],
                )
            )
        except (ValueError, IndexError):
            continue
    return entries


def read_mountinfo(path: str = "/proc/self/mountinfo") -> str:
    """Read mountinfo from the current process namespace."""
    with open(path, encoding="utf-8") as mountinfo:
        return mountinfo.read()


def redact_mount_source(source: str) -> str:
    """Remove user credentials from a mount source."""
    if source.startswith("//"):
        prefix, authority_and_path = "//", source[2:]
    elif "://" in source:
        prefix, authority_and_path = source.split("://", 1)
        prefix += "://"
    else:
        return source.split("@", 1)[1] if "@" in source else source
    authority, separator, path = authority_and_path.partition("/")
    authority = authority.rsplit("@", 1)[-1]
    return f"{prefix}{authority}{separator}{path}"


def default_probe(path: str) -> None:
    """Read one directory entry, allowing empty directories."""
    with os.scandir(path) as entries:
        next(entries, None)


@dataclass
class _InFlightProbe:
    future: Future[None]
    thread: threading.Thread


class MountHealthChecker:
    """Check network and automounted filesystems with bounded daemon probes."""

    def __init__(
        self,
        *,
        probe_timeout_seconds: float = 10.0,
        failures_before_report: int = 2,
        exclude_prefixes: Sequence[str] = (),
        max_probe_threads: int = 16,
        mountinfo_reader: Callable[[], str] = read_mountinfo,
        probe: Callable[[str], None] = default_probe,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._probe_timeout_seconds = probe_timeout_seconds
        self._failures_before_report = failures_before_report
        self._exclude_prefixes = (*DEFAULT_EXCLUDE_PREFIXES, tempfile.gettempdir(), *exclude_prefixes)
        self._max_probe_threads = max_probe_threads
        self._mountinfo_reader = mountinfo_reader
        self._probe = probe
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._inflight: dict[tuple[str, int], _InFlightProbe] = {}
        self._failures: dict[str, tuple[int, datetime]] = {}
        self._snapshot: list[MountHealthEntry] = []
        self._last_checked_at: datetime | None = None

    def _excluded(self, path: str) -> bool:
        return any(path == prefix or path.startswith(f"{prefix.rstrip('/')}/") for prefix in self._exclude_prefixes)

    def _candidates(self, entries: list[MountInfoEntry]) -> dict[str, tuple[MountInfoEntry, bool]]:
        grouped: dict[str, list[MountInfoEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.mount_point].append(entry)
        candidates: dict[str, tuple[MountInfoEntry, bool]] = {}
        for mount_point, group in grouped.items():
            autofs = [entry for entry in group if entry.fstype == "autofs"]
            non_autofs = [entry for entry in group if entry.fstype != "autofs"]
            if autofs:
                candidates[mount_point] = (non_autofs[-1] if non_autofs else autofs[-1], not non_autofs)
            elif group[-1].fstype in NETWORK_FSTYPES:
                candidates[mount_point] = (group[-1], False)
        return {path: candidate for path, candidate in candidates.items() if not self._excluded(path)}

    def _start_probe(self, key: tuple[str, int], path: str) -> _InFlightProbe:
        future: Future[None] = Future()

        def run() -> None:
            try:
                self._probe(path)
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(None)

        thread = threading.Thread(target=run, daemon=True, name=f"mount-health:{path}")
        record = _InFlightProbe(future, thread)
        thread.start()
        self._inflight[key] = record
        return record

    @staticmethod
    def _failure_from_future(future: Future[None]) -> tuple[MountState, str | None]:
        try:
            future.result()
        except OSError as error:
            return "failed", os.strerror(error.errno) if error.errno is not None else str(error)
        except BaseException as error:
            return "failed", type(error).__name__
        return "ok", None

    def check(self) -> list[MountHealthEntry]:
        """Run one serialized health check."""
        with self._lock:
            self._inflight = {key: record for key, record in self._inflight.items() if not record.future.done()}
            carried_alive = sum(r.thread.is_alive() for r in self._inflight.values())
            candidates = self._candidates(parse_mountinfo(self._mountinfo_reader()))
            pending: dict[str, tuple[MountInfoEntry, bool, _InFlightProbe]] = {}
            outcomes: dict[str, tuple[MountInfoEntry, bool, MountState, str | None]] = {}
            queued: list[tuple[str, MountInfoEntry, bool]] = []
            for mount_point, (entry, inactive_autofs) in candidates.items():
                key = (mount_point, entry.mount_id)
                record = self._inflight.get(key)
                if record:
                    outcomes[mount_point] = (entry, inactive_autofs, "unresponsive", "Previous check still pending")
                else:
                    queued.append((mount_point, entry, inactive_autofs))
            deadline = time.monotonic() + self._probe_timeout_seconds
            while queued and time.monotonic() < deadline:
                running = sum(record.thread.is_alive() for _, _, record in pending.values())
                while queued and carried_alive + running < self._max_probe_threads and time.monotonic() < deadline:
                    mount_point, entry, inactive_autofs = queued.pop(0)
                    try:
                        record = self._start_probe((mount_point, entry.mount_id), mount_point)
                    except RuntimeError:
                        outcomes[mount_point] = (entry, inactive_autofs, "failed", "Could not start mount probe")
                    else:
                        pending[mount_point] = (entry, inactive_autofs, record)
                        running += 1
                waiting = [record.future for _, _, record in pending.values() if not record.future.done()]
                if not waiting:
                    if carried_alive >= self._max_probe_threads:
                        break
                    continue
                wait(waiting, timeout=max(0.0, deadline - time.monotonic()), return_when=FIRST_COMPLETED)
            for mount_point, entry, inactive_autofs in queued:
                outcomes[mount_point] = (entry, inactive_autofs, "unresponsive", "Probe capacity exhausted")
            for mount_point, (entry, inactive_autofs, record) in pending.items():
                future = record.future
                if future.done():
                    state, error = self._failure_from_future(future)
                else:
                    state, error = "unresponsive", f"No response within {self._probe_timeout_seconds:g}s"
                outcomes[mount_point] = (entry, inactive_autofs, state, error)

            active_after_probe = self._candidates(parse_mountinfo(self._mountinfo_reader()))
            checked_at = self._clock()
            results: list[MountHealthEntry] = []
            for mount_point, (entry, inactive_autofs, state, error) in outcomes.items():
                recovered = state == "ok" and inactive_autofs and mount_point in active_after_probe and not active_after_probe[mount_point][1]
                if state == "ok":
                    self._failures.pop(mount_point, None)
                    count, failing_since = 0, None
                else:
                    previous_count, previous_since = self._failures.get(mount_point, (0, checked_at))
                    count, failing_since = previous_count + 1, previous_since
                    self._failures[mount_point] = (count, failing_since)
                results.append(
                    MountHealthEntry(
                        mount_point=mount_point,
                        fstype=entry.fstype,
                        source="host automount" if inactive_autofs else redact_mount_source(entry.source),
                        state=state,
                        error=error,
                        failing_since=failing_since,
                        consecutive_failures=count,
                        reported=state != "ok" and count >= self._failures_before_report,
                        recovered=recovered,
                        checked_at=checked_at,
                    )
                )
            self._failures = {path: failure for path, failure in self._failures.items() if path in candidates}
            with self._snapshot_lock:
                self._snapshot = results
                self._last_checked_at = checked_at
            return list(results)

    async def check_async(self) -> list[MountHealthEntry]:
        """Run a check without blocking the event loop."""
        return await asyncio.to_thread(self.check)

    def snapshot(self) -> list[MountHealthEntry]:
        """Return cached results without probing or waiting for a running check."""
        with self._snapshot_lock:
            return list(self._snapshot)

    def last_checked_at(self) -> datetime | None:
        """Return when a check last completed."""
        with self._snapshot_lock:
            return self._last_checked_at

    def problems(self) -> list[MountHealthEntry]:
        """Return reported problems from the cached results."""
        return [entry for entry in self.snapshot() if entry.reported]
