import asyncio
import tempfile
import unittest
from pathlib import Path

from ragtime.indexer.resource_governor import (
    IndexingResourceGovernor,
    MiB,
    ResourceLimitError,
    ResourceRequest,
    _cgroup_v2_metrics,
    _Metrics,
    _quota_capacity,
)


class IndexingResourceGovernorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.metrics = _Metrics(
            sampled_monotonic=self.now,
            system_total_bytes=8 * 1024 * MiB,
            system_available_bytes=6 * 1024 * MiB,
            cgroup_limit_bytes=4 * 1024 * MiB,
            cgroup_usage_bytes=1 * 1024 * MiB,
            memory_source="cgroup_v2",
            cpu_capacity=4.0,
        )
        self.governor = IndexingResourceGovernor(sampler=lambda: self.metrics, clock=lambda: self.now)

    async def asyncTearDown(self) -> None:
        await self.governor.stop()

    async def test_snapshot_is_cached_and_admission_releases_reservation(self) -> None:
        await self.governor.start()
        request = ResourceRequest("one", "chunking", 512 * MiB, cpu_slots=1)
        async with self.governor.acquire(request):
            status = self.governor.snapshot()
            self.assertEqual(status["active_jobs"], 1)
            self.assertEqual(status["committed_bytes"], 512 * MiB)
        self.assertEqual(self.governor.snapshot()["committed_bytes"], 0)

    async def test_system_and_cgroup_reserves_are_independent(self) -> None:
        self.metrics.system_available_bytes = 700 * MiB
        self.metrics.cgroup_limit_bytes = 2 * 1024 * MiB
        self.metrics.cgroup_usage_bytes = 1200 * MiB
        await self.governor.start()
        # System has 188 MiB after its 512 MiB reserve, cgroup has 596 MiB.
        pending = asyncio.create_task(self.governor.acquire(ResourceRequest("one", "finalizing", 300 * MiB)).__aenter__())
        await asyncio.sleep(0)
        self.assertEqual(self.governor.snapshot()["limiting_reason"], "memory_headroom")
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending

    async def test_waiting_acquisition_is_cancellable(self) -> None:
        self.governor.configure(memory_budget_mb=512, max_workers=1, max_batch_documents=0, sequential_index_loading=False)
        await self.governor.start()
        first = ResourceRequest("one", "chunking", 400 * MiB, cpu_slots=1)
        second = ResourceRequest("two", "chunking", 400 * MiB, cpu_slots=1)
        async with self.governor.acquire(first):
            task = asyncio.create_task(self.governor.acquire(second).__aenter__())
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.governor.snapshot()["waiting_jobs"], 0)

    async def test_round_robin_admits_competing_jobs_in_turn(self) -> None:
        self.governor.configure(memory_budget_mb=1024, max_workers=1, max_batch_documents=0, sequential_index_loading=False)
        await self.governor.start()
        first = ResourceRequest("one", "chunking", 300 * MiB, cpu_slots=1)
        second = ResourceRequest("two", "chunking", 300 * MiB, cpu_slots=1)
        async with self.governor.acquire(first):
            waiting = asyncio.create_task(self.governor.acquire(second).__aenter__())
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
        lease = await waiting
        self.assertEqual(lease.request.job_id, "two")
        await lease.__aexit__(None, None, None)

    async def test_finalization_that_exceeds_budget_is_explicit(self) -> None:
        self.governor.configure(memory_budget_mb=256, max_workers=0, max_batch_documents=0, sequential_index_loading=False)
        await self.governor.start()
        with self.assertRaises(ResourceLimitError) as raised:
            async with self.governor.acquire(ResourceRequest("one", "finalizing", 300 * MiB)):
                pass
        self.assertEqual(raised.exception.reason, "memory_budget")

    async def test_worker_idle_residency_is_visible_once(self) -> None:
        await self.governor.start()
        self.governor.track_worker(22, "one")
        self.governor.report_worker_resident_bytes(22, 64 * MiB)
        self.assertEqual(self.governor.snapshot()["committed_bytes"], 64 * MiB)
        self.governor.set_worker_active(22, True)
        self.assertEqual(self.governor.snapshot()["committed_bytes"], 0)

    async def test_reported_residency_is_not_subtracted_twice_from_headroom(self) -> None:
        self.governor.configure(memory_budget_mb=4096, max_workers=0, max_batch_documents=0, sequential_index_loading=False)
        self.metrics.system_available_bytes = 2 * 1024 * MiB
        self.metrics.cgroup_usage_bytes = 1 * 1024 * MiB
        await self.governor.start()
        request = ResourceRequest("one", "chunking", 400 * MiB)
        async with self.governor.acquire(request) as lease:
            before, _ = self.governor._headroom()
            lease.report_resident_bytes(400 * MiB)
            after, _ = self.governor._headroom()
            self.assertEqual(after - before, 400 * MiB)

    async def test_reconfigure_fails_existing_waiter_that_can_no_longer_fit(self) -> None:
        self.governor.configure(memory_budget_mb=1024, max_workers=1, max_batch_documents=0, sequential_index_loading=False)
        await self.governor.start()
        async with self.governor.acquire(ResourceRequest("one", "chunking", 400 * MiB, cpu_slots=1)):
            pending = asyncio.create_task(self.governor.acquire(ResourceRequest("two", "finalizing", 300 * MiB, cpu_slots=1)).__aenter__())
            await asyncio.sleep(0)
            self.governor.configure(memory_budget_mb=256, max_workers=1, max_batch_documents=0, sequential_index_loading=False)
            await asyncio.sleep(0)
            with self.assertRaises(ResourceLimitError):
                await pending

    async def test_first_memory_event_and_three_lag_samples_shrink(self) -> None:
        await self.governor.start()
        self.governor._worker_target = 4
        self.metrics.memory_events = 1
        await self.governor._refresh_metrics()
        self.assertEqual(self.governor.snapshot()["worker_target"], 2)
        self.metrics.memory_events = 1
        self.governor._worker_target = 4
        self.governor._event_loop_lag_ms = 101
        await self.governor._refresh_metrics()
        await self.governor._refresh_metrics()
        self.assertEqual(self.governor.snapshot()["worker_target"], 4)
        await self.governor._refresh_metrics()
        self.assertEqual(self.governor.snapshot()["worker_target"], 2)

    async def test_snapshot_uses_sample_timestamp_and_keeps_multiple_operations(self) -> None:
        self.metrics.sampled_at_utc = "2026-09-10T00:00:00+00:00"
        await self.governor.start()
        async with self.governor.acquire(ResourceRequest("one", "chunking", 100 * MiB)):
            async with self.governor.acquire(ResourceRequest("one", "embedding", 10 * MiB)):
                status = self.governor.snapshot()
                self.assertEqual(status["sampled_at"], self.metrics.sampled_at_utc)
                self.assertEqual(len(status["jobs"]), 2)
                self.assertEqual(sum(item["committed_bytes"] for item in status["jobs"]), 110 * MiB)

    async def test_pressure_shrinks_and_cooldown_blocks_growth(self) -> None:
        await self.governor.start()
        self.governor._worker_target = 4
        self.metrics.pressure = True
        await self.governor._refresh_metrics()
        self.assertEqual(self.governor.snapshot()["worker_target"], 2)
        self.assertGreater(self.governor._cooldown_until, self.now)


class CgroupMetricTests(unittest.TestCase):
    def test_v2_keeps_limit_and_usage_from_same_scope_and_reads_memory_high(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"
            child = parent / "child"
            child.mkdir(parents=True)
            (parent / "memory.max").write_text("2000")
            (parent / "memory.current").write_text("100")
            (child / "memory.max").write_text("1000")
            (child / "memory.high").write_text("950")
            (child / "memory.current").write_text("900")
            result = _cgroup_v2_metrics(child, None)
            self.assertIsNotNone(result)
            assert result is not None
            limit, usage, _, _ = result
            self.assertEqual((limit, usage), (950, 900))

    def test_v2_unlimited_reports_primary_current_and_fractional_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "memory.current").write_text("123")
            (root / "memory.max").write_text("max")
            (root / "cpu.max").write_text("50000 100000")
            result = _cgroup_v2_metrics(root, None)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result[:2], (None, 123))
            self.assertEqual(_quota_capacity(root, 4.0), 0.5)


class GovernorLifecycleTests(unittest.TestCase):
    def test_one_governor_can_restart_in_independent_event_loops(self) -> None:
        governor = IndexingResourceGovernor(
            sampler=lambda: _Metrics(sampled_monotonic=0, system_total_bytes=4 * 1024 * MiB, system_available_bytes=3 * 1024 * MiB)
        )

        async def lifecycle() -> None:
            await governor.start()
            async with governor.acquire(ResourceRequest("job", "chunking", 10 * MiB)):
                pass
            await governor.stop()

        asyncio.run(lifecycle())
        asyncio.run(lifecycle())
