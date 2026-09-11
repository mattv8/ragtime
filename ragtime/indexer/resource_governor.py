"""Process-wide, conservative admission control for indexing work.

The governor deliberately has no database or pipeline imports.  It samples host
state in a worker thread and exposes the last sample to request handlers.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

try:  # psutil is an application dependency, but keep import failures safe.
    import psutil
except ImportError:  # pragma: no cover - exercised on constrained installations
    psutil = None  # type: ignore[assignment]

Stage = Literal["loading", "chunking", "embedding", "finalizing", "index_loading"]
Kind = Literal["document_job", "index_load", "shared"]
Reason = Literal[
    "none",
    "memory_headroom",
    "memory_budget",
    "cpu_capacity",
    "provider_limit",
    "user_limit",
    "memory_pressure",
    "metrics_unavailable",
]
MiB = 1024 * 1024


class ResourceLimitError(RuntimeError):
    """A request can never fit the configured resource envelope."""

    def __init__(self, *, stage: Stage, required_bytes: int, budget_bytes: int, reason: Reason) -> None:
        self.stage = stage
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        self.reason = reason
        super().__init__(f"{stage} requires {required_bytes} bytes but budget is {budget_bytes} ({reason})")


@dataclass(frozen=True)
class ResourceRequest:
    job_id: str
    stage: Stage
    estimated_peak_bytes: int
    cpu_slots: int = 0
    provider_key: str | None = None
    kind: Kind = "document_job"
    diagnostic_source: str | None = None


@dataclass
class _Metrics:
    sampled_monotonic: float
    system_total_bytes: int | None = None
    system_available_bytes: int | None = None
    cgroup_limit_bytes: int | None = None
    cgroup_usage_bytes: int | None = None
    memory_source: Literal["cgroup_v2", "cgroup_v1", "system", "unavailable"] = "unavailable"
    cpu_capacity: float = 1.0
    pressure: bool = False
    memory_events: int = 0
    swap_out: int = 0
    application_rss_bytes: int | None = None
    sampled_at_utc: str | None = None


@dataclass
class _Waiter:
    request: ResourceRequest
    future: asyncio.Future["ResourceLease"]
    queued_at: float
    sequence: int


@dataclass
class _Worker:
    job_id: str
    resident_bytes: int = 0
    active: bool = False


class ResourceLease:
    def __init__(self, governor: "IndexingResourceGovernor", request: ResourceRequest) -> None:
        self._governor = governor
        self.request = request
        self.resident_bytes = 0
        self.peak_bytes = 0
        self._released = False

    def report_resident_bytes(self, value: int) -> None:
        self.resident_bytes = max(0, int(value))

    def report_peak_bytes(self, value: int) -> None:
        self.peak_bytes = max(self.peak_bytes, max(0, int(value)))

    async def __aenter__(self) -> "ResourceLease":
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._released:
            self._released = True
            await self._governor._release(self)


class _AcquireContext(AbstractAsyncContextManager[ResourceLease]):
    def __init__(self, governor: "IndexingResourceGovernor", request: ResourceRequest) -> None:
        self._governor = governor
        self._request = request
        self._lease: ResourceLease | None = None

    async def __aenter__(self) -> ResourceLease:
        self._lease = await self._governor._admit(self._request)
        return self._lease

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._lease is not None:
            await self._lease.__aexit__(exc_type, exc, traceback)


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _finite_limit(value: int | None, host_total: int | None) -> int | None:
    # Docker commonly exposes the host total as an effectively-unlimited limit.
    if value is None or value <= 0 or (host_total and value >= host_total):
        return None
    return value


def _read_pressure(path: Path) -> bool:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("full "):
                return float(dict(item.split("=", 1) for item in line.split()[1:]).get("avg10", "0")) >= 1.0
    except (OSError, ValueError):
        pass
    return False


def _cgroup_v2_metrics(root: Path, host_total: int | None) -> tuple[int | None, int | None, bool, int] | None:
    if not (root / "memory.current").exists():
        return None
    constraints: list[tuple[int, int]] = []
    primary_usage = _read_int(root / "memory.current")
    events = 0
    pressure = False
    # Walking ancestors gives the strictest visible constraint for nested cgroups.
    for current in (root, *root.parents):
        limit = _finite_limit(_read_int(current / "memory.max"), host_total)
        high = _finite_limit(_read_int(current / "memory.high"), host_total)
        usage = _read_int(current / "memory.current")
        # A scope's limit and current must remain paired.  Mixing the minimum
        # ancestor limit with a different ancestor's maximum current creates a
        # fictitious headroom value.
        if usage is not None:
            for constrained_limit in (limit, high):
                if constrained_limit is not None:
                    constraints.append((constrained_limit, usage))
        try:
            values = dict(line.split(maxsplit=1) for line in (current / "memory.events").read_text().splitlines())
            events += int(values.get("high", 0)) + int(values.get("max", 0)) + int(values.get("oom", 0))
        except (OSError, ValueError):
            pass
        pressure = pressure or _read_pressure(current / "memory.pressure")
        if current == current.parent:
            break
    if constraints:
        limit, usage = min(constraints, key=lambda item: item[0] - item[1])
        return limit, usage, pressure, events
    # Unlimited cgroups still have meaningful current usage for telemetry.
    return None, primary_usage, pressure, events


def _cgroup_v2_path() -> Path:
    """Resolve this process's v2 path below the mounted cgroup root."""
    root = Path("/sys/fs/cgroup")
    relative = "/"
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers == "":
                break
    except OSError:
        return root
    try:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            fields = line.split()
            separator = fields.index("-")
            if fields[separator + 1] != "cgroup2":
                continue
            mount_root, mount_point = fields[3], fields[4]
            suffix = relative
            if mount_root != "/" and suffix.startswith(mount_root):
                suffix = suffix[len(mount_root) :]
            return Path(mount_point) / suffix.lstrip("/")
    except (OSError, ValueError, IndexError):
        pass
    return root / relative.lstrip("/")


def _read_swap_out() -> int:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("pswpout "):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def _quota_capacity(root: Path, capacity: float) -> float:
    """Apply visible v2 or v1 CPU quota without rounding fractional quotas."""
    for current in (root, *root.parents):
        try:
            quota, period = (current / "cpu.max").read_text().split()
            if quota != "max":
                capacity = min(capacity, int(quota) / int(period))
        except (OSError, ValueError, ZeroDivisionError):
            pass
        try:
            v1_quota = _read_int(current / "cpu/cpu.cfs_quota_us")
            v1_period = _read_int(current / "cpu/cpu.cfs_period_us")
            if v1_quota is not None and v1_quota > 0 and v1_period:
                capacity = min(capacity, v1_quota / v1_period)
        except ZeroDivisionError:
            pass
        if current == current.parent:
            break
    return max(0.01, capacity)


def collect_resource_metrics() -> _Metrics:
    """Collect best-effort host/cgroup state. This function must run off-loop."""
    now = time.monotonic()
    if psutil is None:
        return _Metrics(sampled_monotonic=now)
    vm = psutil.virtual_memory()
    host_total, available = int(vm.total), int(vm.available)
    process = psutil.Process()
    rss = 0
    try:
        descendants = process.children(recursive=True)
        for proc in [process, *descendants]:
            rss += int(proc.memory_info().rss)
    except (psutil.Error, OSError):
        rss = int(process.memory_info().rss)
    affinity = len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else (os.cpu_count() or 1)
    capacity = min(float(affinity), float(os.cpu_count() or 1))
    root = _cgroup_v2_path()
    v2 = _cgroup_v2_metrics(root, host_total)
    if v2 is not None:
        limit, usage, pressure, events = v2
        capacity = _quota_capacity(root, capacity)
        return _Metrics(
            now, host_total, available, limit, usage, "cgroup_v2", capacity, pressure, events, _read_swap_out(), rss, datetime.now(timezone.utc).isoformat()
        )
    v1_limit = _finite_limit(_read_int(root / "memory/memory.limit_in_bytes"), host_total)
    v1_usage = _read_int(root / "memory/memory.usage_in_bytes")
    if v1_limit is not None or v1_usage is not None:
        return _Metrics(
            now,
            host_total,
            available,
            v1_limit,
            v1_usage,
            "cgroup_v1",
            _quota_capacity(root, capacity),
            swap_out=_read_swap_out(),
            application_rss_bytes=rss,
            sampled_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    return _Metrics(
        now,
        host_total,
        available,
        None,
        None,
        "system",
        capacity,
        swap_out=_read_swap_out(),
        application_rss_bytes=rss,
        sampled_at_utc=datetime.now(timezone.utc).isoformat(),
    )


class IndexingResourceGovernor:
    """A fair, cancellation-safe scheduler for transient indexing allocations."""

    def __init__(self, *, sampler: Callable[[], _Metrics] = collect_resource_metrics, clock: Callable[[], float] = time.monotonic) -> None:
        self._sampler, self._clock = sampler, clock
        self._metrics: _Metrics | None = None
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._leases: set[ResourceLease] = set()
        self._workers: dict[int, _Worker] = {}
        self._sample_task: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False
        self._sequence = 0
        self._round_robin_after: str | None = None
        self._memory_budget_mb = 0
        self._max_workers = 0
        self._max_batch_documents = 0
        self._sequential_loading = False
        self._worker_target = 1
        self._healthy_samples = 0
        self._cooldown_until = 0.0
        self._last_events = 0
        self._last_swap_out = 0
        self._has_sample = False
        self._swap_pressure_samples = 0
        self._lag_samples = 0
        self._observed_stage_peaks: dict[Stage, int] = {}
        self._provider_active: dict[str, int] = defaultdict(int)
        self._provider_limit: dict[str, int] = defaultdict(lambda: 1)
        self._event_loop_lag_ms: float | None = None

    async def start(self) -> None:
        if self._running:
            return
        self._running, self._stopping = True, False
        await self._refresh_metrics()
        self._sample_task = asyncio.create_task(self._sampling_loop(), name="indexing-resource-sampler")

    async def stop(self) -> None:
        self._stopping, self._running = True, False
        async with self._condition:
            for waiter in self._waiters:
                if not waiter.future.done():
                    waiter.future.cancel()
            self._waiters.clear()
            self._condition.notify_all()
        if self._sample_task:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
            self._sample_task = None
        # asyncio synchronization primitives become loop-bound after a waiter
        # blocks. Recreate this otherwise-empty condition so the module
        # singleton can cleanly serve a later asyncio.run() lifecycle.
        self._condition = asyncio.Condition()

    def configure(self, *, memory_budget_mb: int, max_workers: int, max_batch_documents: int, sequential_index_loading: bool) -> None:
        self._memory_budget_mb = max(0, int(memory_budget_mb))
        self._max_workers = max(0, int(max_workers))
        self._max_batch_documents = max(0, int(max_batch_documents))
        self._sequential_loading = bool(sequential_index_loading)
        # Condition notification must occur on the owning loop, but this method
        # intentionally does no blocking work and is safe before start().
        if self._running:
            asyncio.create_task(self._wake_waiters())

    async def _wake_waiters(self) -> None:
        async with self._condition:
            self._fail_impossible_waiters()
            self._grant_waiters()
            self._condition.notify_all()

    def acquire(self, request: ResourceRequest) -> AbstractAsyncContextManager[ResourceLease]:
        return _AcquireContext(self, request)

    def estimate_request(
        self,
        *,
        job_id: str,
        stage: Stage,
        text_bytes: int = 0,
        record_count: int = 0,
        dimensions: int = 0,
        steady_bytes: int = 0,
        cpu_slots: int = 1,
        provider_key: str | None = None,
        kind: Kind = "document_job",
    ) -> ResourceRequest:
        from ragtime.indexer.memory_utils import estimate_stage_peak_bytes

        estimate = estimate_stage_peak_bytes(stage=stage, text_bytes=text_bytes, record_count=record_count, dimensions=dimensions, steady_bytes=steady_bytes)
        # A measured peak is a lower bound for later equivalent stage requests.
        estimate = max(estimate, self._observed_stage_peaks.get(stage, 0))
        return ResourceRequest(job_id, stage, estimate, cpu_slots, provider_key, kind)

    def record_outcome(
        self, *, stage: Stage, elapsed_seconds: float, peak_bytes: int = 0, provider_key: str | None = None, throttled: bool = False, failed: bool = False
    ) -> None:
        # Failed envelopes are valuable observations too: callers use this to
        # make their single adaptive retry large enough for native cold starts.
        if peak_bytes > 0:
            self._observed_stage_peaks[stage] = max(self._observed_stage_peaks.get(stage, 0), int(peak_bytes))
        if throttled or failed:
            self._cooldown_until = max(self._cooldown_until, self._clock() + 30.0)
            if provider_key:
                self._provider_limit[provider_key] = 1
        elif provider_key and elapsed_seconds > 0 and self._healthy_samples >= 10:
            self._provider_limit[provider_key] = min(4, self._provider_limit[provider_key] + 1)

    def track_worker(self, pid: int, job_id: str) -> None:
        self._workers[pid] = _Worker(job_id)

    def untrack_worker(self, pid: int) -> None:
        self._workers.pop(pid, None)

    def set_worker_active(self, pid: int, active: bool) -> None:
        worker = self._workers.get(pid)
        if worker:
            worker.active = active

    def report_worker_resident_bytes(self, pid: int, value: int) -> None:
        """Worker integration hook; residency is charged once while it is idle."""
        worker = self._workers.get(pid)
        if worker:
            worker.resident_bytes = max(0, int(value))

    def batch_document_limit(self) -> int:
        return self._max_batch_documents or 10

    async def _admit(self, request: ResourceRequest) -> ResourceLease:
        if request.estimated_peak_bytes < 0 or request.cpu_slots < 0:
            raise ValueError("resource estimates and CPU slots must be non-negative")
        loop = asyncio.get_running_loop()
        waiter = _Waiter(request, loop.create_future(), self._clock(), self._sequence)
        self._sequence += 1
        async with self._condition:
            if self._stopping:
                raise asyncio.CancelledError("indexing resource governor is stopping")
            self._reject_impossible(request)
            self._waiters.append(waiter)
            self._grant_waiters()
        try:
            return await waiter.future
        except BaseException:
            async with self._condition:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                elif waiter.future.done() and not waiter.future.cancelled():
                    # Cancellation can race a grant between the final await
                    # checkpoint and returning from __aenter__. Reclaim the
                    # lease because no context manager will release it.
                    lease = waiter.future.result()
                    self._reclaim_granted_lease(lease)
                self._grant_waiters()
            raise

    async def _release(self, lease: ResourceLease) -> None:
        async with self._condition:
            self._leases.discard(lease)
            if lease.request.provider_key:
                self._provider_active[lease.request.provider_key] = max(0, self._provider_active[lease.request.provider_key] - 1)
            self._grant_waiters()
            self._condition.notify_all()

    def _effective_total(self) -> int:
        metrics = self._metrics
        if not metrics:
            return 0
        return metrics.cgroup_limit_bytes or metrics.system_total_bytes or 0

    def _budget(self) -> int:
        return self._memory_budget_mb * MiB if self._memory_budget_mb else self._effective_total() // 2

    def _committed(self) -> int:
        # Reservations are the not-yet-materialized allocations. Worker RSS is
        # separately retained for idle workers; active workers are measured in
        # process/cgroup usage and are not deducted twice from headroom.
        active = sum(lease.request.estimated_peak_bytes for lease in self._leases)
        idle = sum(worker.resident_bytes for worker in self._workers.values() if not worker.active)
        return active + idle

    def _unmaterialized_reservations(self) -> int:
        """Reservations not yet reflected by a worker's measured RSS report."""
        active = sum(max(0, lease.request.estimated_peak_bytes - lease.resident_bytes) for lease in self._leases)
        # Idle processes remain in sampled app/cgroup usage. Their tracked RSS
        # is policy accounting only, not a second subtraction from headroom.
        return active

    def _headroom(self) -> tuple[int, Reason]:
        metrics = self._metrics
        stale = not metrics or self._clock() - metrics.sampled_monotonic > 3.0
        unmaterialized = self._unmaterialized_reservations()
        budget = self._budget()
        if budget <= 0:
            return 0, "metrics_unavailable"
        candidates: list[tuple[int, Reason]] = [(budget - self._committed(), "memory_budget")]
        if stale:
            # Without live metrics only one conservative heavy operation may
            # proceed; unknown headroom never becomes unlimited.
            return (budget - self._committed(), "metrics_unavailable")
        if metrics is not None:
            system_available = metrics.system_available_bytes
            system_total = metrics.system_total_bytes
            if system_available is not None and system_total is not None:
                reserve = max(512 * MiB, system_total // 10)
                candidates.append((system_available - reserve - unmaterialized, "memory_headroom"))
            cgroup_limit = metrics.cgroup_limit_bytes
            cgroup_usage = metrics.cgroup_usage_bytes
            if cgroup_limit is not None and cgroup_usage is not None:
                reserve = max(256 * MiB, cgroup_limit // 10)
                candidates.append((cgroup_limit - cgroup_usage - reserve - unmaterialized, "memory_headroom"))
        return min(candidates, key=lambda item: item[0])

    def _cpu_limit(self) -> int:
        metrics = self._metrics
        capacity = metrics.cpu_capacity if metrics else 1.0
        automatic = min(16, max(1, int(capacity * 0.75)))
        return self._max_workers or automatic

    def _reject_impossible(self, request: ResourceRequest) -> None:
        budget = self._budget()
        metrics = self._metrics
        if budget and request.estimated_peak_bytes > budget:
            raise ResourceLimitError(stage=request.stage, required_bytes=request.estimated_peak_bytes, budget_bytes=budget, reason="memory_budget")
        if metrics and metrics.cgroup_limit_bytes:
            available = metrics.cgroup_limit_bytes - max(256 * MiB, metrics.cgroup_limit_bytes // 10)
            if request.estimated_peak_bytes > available:
                raise ResourceLimitError(stage=request.stage, required_bytes=request.estimated_peak_bytes, budget_bytes=available, reason="memory_headroom")
        if not budget:
            raise ResourceLimitError(stage=request.stage, required_bytes=request.estimated_peak_bytes, budget_bytes=0, reason="metrics_unavailable")

    def _can_admit(self, request: ResourceRequest) -> tuple[bool, Reason]:
        headroom, reason = self._headroom()
        if request.estimated_peak_bytes > headroom:
            return False, reason
        if reason == "metrics_unavailable" and self._leases:
            return False, reason
        cpu_used = sum(lease.request.cpu_slots for lease in self._leases)
        if request.cpu_slots and cpu_used + request.cpu_slots > min(self._cpu_limit(), self._worker_target):
            return False, "user_limit" if self._max_workers else "cpu_capacity"
        if request.provider_key and self._provider_active[request.provider_key] >= self._provider_limit[request.provider_key]:
            return False, "provider_limit"
        if self._sequential_loading and request.stage == "index_loading" and any(lease.request.stage == "index_loading" for lease in self._leases):
            return False, "user_limit"
        return True, "none"

    def _select_waiter(self) -> _Waiter | None:
        if not self._waiters:
            return None
        # Once a large finalization/load has waited, do not let producers keep
        # consuming the capacity it needs; small consumers may still drain.
        aged = next((w for w in self._waiters if self._clock() - w.queued_at >= 30 and w.request.stage in {"finalizing", "index_loading"}), None)
        candidates = self._waiters if aged is None else [w for w in self._waiters if w is aged or w.request.stage == "embedding"]
        jobs: list[str] = []
        for waiter in candidates:
            if waiter.request.job_id not in jobs:
                jobs.append(waiter.request.job_id)
        if self._round_robin_after in jobs:
            index = (jobs.index(self._round_robin_after) + 1) % len(jobs)
            jobs = jobs[index:] + jobs[:index]
        for job_id in jobs:
            waiter = next(w for w in candidates if w.request.job_id == job_id)
            allowed, _ = self._can_admit(waiter.request)
            if allowed:
                return waiter
        return None

    def _grant_waiters(self) -> None:
        self._waiters[:] = [waiter for waiter in self._waiters if not waiter.future.cancelled()]
        while (waiter := self._select_waiter()) is not None:
            self._waiters.remove(waiter)
            if waiter.future.cancelled():
                continue
            lease = ResourceLease(self, waiter.request)
            self._leases.add(lease)
            self._round_robin_after = waiter.request.job_id
            if waiter.request.provider_key:
                self._provider_active[waiter.request.provider_key] += 1
            if not waiter.future.done():
                waiter.future.set_result(lease)

    def _reclaim_granted_lease(self, lease: ResourceLease) -> None:
        self._leases.discard(lease)
        if lease.request.provider_key:
            self._provider_active[lease.request.provider_key] = max(0, self._provider_active[lease.request.provider_key] - 1)

    def _fail_impossible_waiters(self) -> None:
        for waiter in list(self._waiters):
            try:
                self._reject_impossible(waiter.request)
            except ResourceLimitError as error:
                self._waiters.remove(waiter)
                if not waiter.future.done():
                    waiter.future.set_exception(error)

    async def _refresh_metrics(self) -> None:
        try:
            metrics = await asyncio.to_thread(self._sampler)
        except Exception:
            metrics = _Metrics(sampled_monotonic=self._clock())
        if metrics.sampled_at_utc is None:
            metrics.sampled_at_utc = datetime.now(timezone.utc).isoformat()
        self._metrics = metrics
        event_pressure = self._has_sample and metrics.memory_events > self._last_events
        self._swap_pressure_samples = self._swap_pressure_samples + 1 if self._has_sample and metrics.swap_out > self._last_swap_out else 0
        pressure = metrics.pressure or event_pressure or self._swap_pressure_samples >= 3
        self._last_events = metrics.memory_events
        self._last_swap_out = metrics.swap_out
        self._has_sample = True
        headroom, _ = self._headroom()
        self._lag_samples = self._lag_samples + 1 if self._event_loop_lag_ms is not None and self._event_loop_lag_ms > 100 else 0
        if pressure or headroom <= 0 or self._lag_samples >= 3:
            self._worker_target = max(1, self._worker_target // 2)
            self._healthy_samples = 0
            self._cooldown_until = self._clock() + 30.0
        elif headroom > 0 and self._committed() < self._budget() * 0.7:
            self._healthy_samples += 1
            if self._healthy_samples >= 10 and self._waiters and self._clock() >= self._cooldown_until:
                self._worker_target = min(self._cpu_limit(), self._worker_target + 1)
                self._healthy_samples = 0
        else:
            self._healthy_samples = 0
        async with self._condition:
            self._fail_impossible_waiters()
            self._grant_waiters()
            self._condition.notify_all()

    async def _sampling_loop(self) -> None:
        expected = self._clock() + 1.0
        while self._running:
            await asyncio.sleep(max(0, expected - self._clock()))
            now = self._clock()
            self._event_loop_lag_ms = max(0.0, (now - expected) * 1000)
            expected = now + 1.0
            await self._refresh_metrics()

    def snapshot(self) -> dict[str, Any]:
        metrics = self._metrics
        stale = metrics is None or self._clock() - metrics.sampled_monotonic > 3.0
        headroom, reason = self._headroom()
        jobs: list[dict[str, Any]] = []
        for lease in self._leases:
            if lease.request.kind == "document_job":
                jobs.append(
                    {
                        "job_id": lease.request.job_id,
                        "stage": lease.request.stage,
                        "state": "running",
                        "reason": "none",
                        "committed_bytes": lease.request.estimated_peak_bytes,
                    }
                )
        for waiter in self._waiters:
            if waiter.request.kind == "document_job":
                _, wait_reason = self._can_admit(waiter.request)
                jobs.append({"job_id": waiter.request.job_id, "stage": waiter.request.stage, "state": "waiting", "reason": wait_reason, "committed_bytes": 0})
        workers_active = sum(worker.active for worker in self._workers.values())
        return {
            "sampled_at": metrics.sampled_at_utc if metrics and metrics.sampled_at_utc else datetime.now(timezone.utc).isoformat(),
            "stale": stale,
            "memory_source": metrics.memory_source if metrics else "unavailable",
            "system_available_bytes": metrics.system_available_bytes if metrics else None,
            "container_limit_bytes": metrics.cgroup_limit_bytes if metrics else None,
            "container_usage_bytes": metrics.cgroup_usage_bytes if metrics else None,
            "application_rss_bytes": metrics.application_rss_bytes if metrics else None,
            "effective_budget_bytes": self._budget(),
            "committed_bytes": self._committed(),
            "effective_cpu_capacity": metrics.cpu_capacity if metrics else 1.0,
            "event_loop_lag_ms": self._event_loop_lag_ms,
            "worker_limit": self._cpu_limit(),
            "worker_target": self._worker_target,
            "workers_active": workers_active,
            "workers_live": len(self._workers),
            "active_jobs": len({item["job_id"] for item in jobs if item["state"] == "running"}),
            "waiting_jobs": len({item["job_id"] for item in jobs if item["state"] == "waiting"}),
            "limiting_reason": "metrics_unavailable" if stale else (reason if headroom <= 0 else "none"),
            "jobs": jobs,
        }


resource_governor = IndexingResourceGovernor()
