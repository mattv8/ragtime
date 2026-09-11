"""Real-process tests for bounded indexing worker supervision."""

import asyncio
import multiprocessing
import os
import sys
import time
import unittest
from contextlib import AbstractAsyncContextManager
from unittest import mock

from ragtime.indexer.resource_governor import ResourceRequest
from ragtime.indexer.resource_workers import (
    ResourceTaskError,
    ResourceTaskMemoryExceeded,
    ResourceTaskResultTooLarge,
    ResourceTaskTimeout,
    _install_parent_death_guard,
    run_resource_task,
)


def _return(value):
    return value


def _sleep(seconds):
    time.sleep(seconds)
    return "done"


def _large_result():
    return "x" * (5 * 1024 * 1024)


def _exit_without_result():
    os._exit(17)


def _die_when_parent_is_wrong():
    _install_parent_death_guard(os.getppid() + 1)
    time.sleep(2)


class _Lease:
    def report_resident_bytes(self, value):
        pass

    def report_peak_bytes(self, value):
        pass


class _Acquire(AbstractAsyncContextManager):
    async def __aenter__(self):
        return _Lease()

    async def __aexit__(self, *args):
        return None


class _Governor:
    def __init__(self):
        self.live = set()
        self.max_live = 0

    def acquire(self, request):
        return _Acquire()

    def track_worker(self, pid, job_id):
        self.live.add(pid)
        self.max_live = max(self.max_live, len(self.live))

    def untrack_worker(self, pid):
        self.live.discard(pid)

    def set_worker_active(self, pid, active):
        pass

    def record_outcome(self, **kwargs):
        pass


class ResourceWorkerTests(unittest.IsolatedAsyncioTestCase):
    def request(self, job_id="job-a"):
        return ResourceRequest(job_id=job_id, stage="chunking", estimated_peak_bytes=0, cpu_slots=1)

    async def test_returns_small_serializable_result_and_reaps_child(self):
        governor = _Governor()
        with mock.patch("ragtime.indexer.resource_workers.resource_governor", governor):
            self.assertEqual(await run_resource_task(self.request(), _return, ("ok",)), "ok")
        self.assertEqual(governor.live, set())

    async def test_deadline_terminates_unresponsive_child(self):
        governor = _Governor()
        with mock.patch("ragtime.indexer.resource_workers.resource_governor", governor):
            with self.assertRaises(ResourceTaskTimeout):
                await run_resource_task(self.request(), _sleep, (5,), timeout_seconds=0.1)
        self.assertEqual(governor.live, set())

    async def test_rejects_corpus_sized_ipc_result(self):
        governor = _Governor()
        with mock.patch("ragtime.indexer.resource_workers.resource_governor", governor):
            with self.assertRaises(ResourceTaskResultTooLarge):
                await run_resource_task(self.request(), _large_result, ())
        self.assertEqual(governor.live, set())

    async def test_eof_from_own_child_is_normalized_and_reaped(self):
        governor = _Governor()
        with mock.patch("ragtime.indexer.resource_workers.resource_governor", governor):
            with self.assertRaisesRegex(ResourceTaskError, "code 17"):
                await run_resource_task(self.request(), _exit_without_result, ())
        self.assertEqual(governor.live, set())

    async def test_concurrent_job_keys_leave_no_surviving_workers(self):
        governor = _Governor()
        with mock.patch("ragtime.indexer.resource_workers.resource_governor", governor):
            results = await asyncio.gather(*(run_resource_task(self.request(f"job-{number}"), _sleep, (0.1,)) for number in range(3)))
        self.assertEqual(results, ["done", "done", "done"])
        self.assertEqual(governor.live, set())

    async def test_memory_envelope_retries_once_with_measured_headroom(self):
        request = self.request()
        observed = 600 * 1024 * 1024
        calls = []

        async def once(candidate, *_args, **_kwargs):
            calls.append(candidate)
            if len(calls) == 1:
                raise ResourceTaskMemoryExceeded(candidate, observed)
            return "retried"

        with mock.patch("ragtime.indexer.resource_workers._run_resource_task_once", side_effect=once):
            self.assertEqual(await run_resource_task(request, _return, ("unused",)), "retried")

        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1].estimated_peak_bytes, observed)

    async def test_second_memory_envelope_failure_does_not_retry_again(self):
        request = self.request()

        async def once(candidate, *_args, **_kwargs):
            raise ResourceTaskMemoryExceeded(candidate, 10)

        with mock.patch("ragtime.indexer.resource_workers._run_resource_task_once", side_effect=once):
            with self.assertRaises(ResourceTaskMemoryExceeded):
                await run_resource_task(request, _return, ("unused",))

    @unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux-specific")
    def test_parent_death_guard_exits_disposable_child(self):
        process = multiprocessing.get_context("spawn").Process(target=_die_when_parent_is_wrong)
        process.start()
        process.join(timeout=3)
        self.assertFalse(process.is_alive())
        self.assertNotEqual(process.exitcode, 0)
