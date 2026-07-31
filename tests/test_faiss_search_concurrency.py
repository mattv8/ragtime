from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import os
import subprocess
import sys
import threading
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

_MODULE_PATH = Path(__file__).resolve().parents[1] / "ragtime" / "core" / "faiss_concurrency.py"
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "test_faiss_concurrency_module",
    _MODULE_PATH,
)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec for {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)

FaissSearchBusyError = _MODULE.FaissSearchBusyError
FaissSearchConcurrencyMode = _MODULE.FaissSearchConcurrencyMode
FaissSearchCoordinator = _MODULE.FaissSearchCoordinator


class _RecordingExecutor:
    def __init__(self) -> None:
        self.submissions = 0
        self.shutdown_calls = 0

    def submit(self, *_args, **_kwargs):
        self.submissions += 1
        raise AssertionError("submit should not be called")

    def shutdown(self, **_kwargs) -> None:
        self.shutdown_calls += 1


class _ForeignMode(Enum):
    PER_INDEX = "per_index"
    GLOBAL = "global"


class FaissSearchCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        coordinator = getattr(self, "coordinator", None)
        if coordinator is not None:
            coordinator.shutdown()

    async def test_per_index_serializes_same_index_searches(self) -> None:
        self.coordinator = FaissSearchCoordinator()
        first_started = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_operation() -> str:
            first_started.set()
            first_release.wait(timeout=1)
            return "first"

        def second_operation() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(
            self.coordinator.run(
                "shared-index",
                first_operation,
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )
        )
        await asyncio.to_thread(first_started.wait, 1)

        second_task = asyncio.create_task(
            self.coordinator.run(
                "shared-index",
                second_operation,
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )
        )
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())

        first_release.set()
        self.assertEqual(await first_task, "first")
        self.assertEqual(await second_task, "second")
        self.assertTrue(second_started.is_set())

    async def test_per_index_allows_different_indexes_to_overlap(self) -> None:
        self.coordinator = FaissSearchCoordinator()
        first_started = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_operation() -> str:
            first_started.set()
            first_release.wait(timeout=1)
            return "first"

        def second_operation() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(
            self.coordinator.run(
                "index-a",
                first_operation,
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )
        )
        await asyncio.to_thread(first_started.wait, 1)

        second_task = asyncio.create_task(
            self.coordinator.run(
                "index-b",
                second_operation,
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )
        )
        await asyncio.to_thread(second_started.wait, 1)
        self.assertTrue(second_started.is_set())

        first_release.set()
        self.assertEqual(await first_task, "first")
        self.assertEqual(await second_task, "second")

    async def test_global_serializes_all_indexes(self) -> None:
        self.coordinator = FaissSearchCoordinator()
        first_started = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_operation() -> str:
            first_started.set()
            first_release.wait(timeout=1)
            return "first"

        def second_operation() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(
            self.coordinator.run(
                "index-a",
                first_operation,
                mode=FaissSearchConcurrencyMode.GLOBAL,
            )
        )
        await asyncio.to_thread(first_started.wait, 1)

        second_task = asyncio.create_task(
            self.coordinator.run(
                "index-b",
                second_operation,
                mode=FaissSearchConcurrencyMode.GLOBAL,
            )
        )
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())

        first_release.set()
        self.assertEqual(await first_task, "first")
        self.assertEqual(await second_task, "second")
        self.assertTrue(second_started.is_set())

    async def test_timeout_raises_busy_error_without_submitting_operation(self) -> None:
        self.coordinator = FaissSearchCoordinator(acquire_timeout_seconds=0.05)
        first_started = threading.Event()
        first_release = threading.Event()

        def first_operation() -> str:
            first_started.set()
            first_release.wait(timeout=1)
            return "first"

        first_task = asyncio.create_task(
            self.coordinator.run(
                "shared-index",
                first_operation,
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )
        )
        await asyncio.to_thread(first_started.wait, 1)

        submit_mock = AsyncMock()
        self.coordinator._run_in_executor = submit_mock
        self.coordinator._executor = _RecordingExecutor()

        with self.assertRaises(FaissSearchBusyError):
            await self.coordinator.run(
                "shared-index",
                lambda: "second",
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )

        submit_mock.assert_not_awaited()
        self.assertEqual(self.coordinator._executor.submissions, 0)

        first_release.set()
        self.assertEqual(await first_task, "first")

    async def test_missing_or_invalid_mode_falls_back_to_per_index(self) -> None:
        for settings_payload in ({}, {"faiss_search_concurrency_mode": "nope"}):
            with self.subTest(settings_payload=settings_payload):
                self.coordinator = FaissSearchCoordinator()
                first_started = threading.Event()
                first_release = threading.Event()
                second_started = threading.Event()

                def first_operation() -> str:
                    first_started.set()
                    first_release.wait(timeout=1)
                    return "first"

                def second_operation() -> str:
                    second_started.set()
                    return "second"

                with patch.object(
                    _MODULE,
                    "get_app_settings",
                    new=AsyncMock(return_value=settings_payload),
                ):
                    first_task = asyncio.create_task(self.coordinator.run("shared-index", first_operation))
                    await asyncio.to_thread(first_started.wait, 1)

                    second_task = asyncio.create_task(self.coordinator.run("shared-index", second_operation))
                    await asyncio.sleep(0.05)
                    self.assertFalse(second_started.is_set())

                    first_release.set()
                    self.assertEqual(await first_task, "first")
                    self.assertEqual(await second_task, "second")

                self.coordinator.shutdown()
                del self.coordinator

    async def test_enum_like_mode_value_resolves_to_global(self) -> None:
        self.coordinator = FaissSearchCoordinator()
        first_started = threading.Event()
        first_release = threading.Event()
        second_started = threading.Event()

        def first_operation() -> str:
            first_started.set()
            first_release.wait(timeout=1)
            return "first"

        def second_operation() -> str:
            second_started.set()
            return "second"

        first_task = asyncio.create_task(
            self.coordinator.run(
                "index-a",
                first_operation,
                mode=_ForeignMode.GLOBAL,
            )
        )
        await asyncio.to_thread(first_started.wait, 1)

        second_task = asyncio.create_task(
            self.coordinator.run(
                "index-b",
                second_operation,
                mode=_ForeignMode.GLOBAL,
            )
        )
        await asyncio.sleep(0.05)
        self.assertFalse(second_started.is_set())

        first_release.set()
        self.assertEqual(await first_task, "first")
        self.assertEqual(await second_task, "second")

    def test_mode_enum_wire_values_match_expected_strings(self) -> None:
        self.assertEqual(FaissSearchConcurrencyMode.PER_INDEX.value, _ForeignMode.PER_INDEX.value)
        self.assertEqual(FaissSearchConcurrencyMode.GLOBAL.value, _ForeignMode.GLOBAL.value)

    async def test_timeout_race_releases_permit_for_later_work(self) -> None:
        self.coordinator = FaissSearchCoordinator(acquire_timeout_seconds=0.05)
        run_mock = AsyncMock(return_value="ok")
        self.coordinator._run_in_executor = run_mock

        async def fake_wait_for(awaitable, timeout):
            del timeout
            await asyncio.shield(awaitable)
            raise asyncio.TimeoutError

        with patch.object(_MODULE.asyncio, "wait_for", new=fake_wait_for):
            with self.assertRaises(FaissSearchBusyError):
                await self.coordinator.run(
                    "shared-index",
                    lambda: "never",
                    mode=FaissSearchConcurrencyMode.PER_INDEX,
                )

        run_mock.assert_not_awaited()

        result = await self.coordinator.run(
            "shared-index",
            lambda: "later",
            mode=FaissSearchConcurrencyMode.PER_INDEX,
        )
        self.assertEqual(result, "ok")

    def test_fresh_process_can_import_ragtime_main(self) -> None:
        if importlib.util.find_spec("fastapi") is None:
            self.skipTest("fastapi not installed in local test environment")

        repo_root = str(Path(__file__).resolve().parents[1])
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, "-c", "import ragtime.main"],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )

    async def test_executor_worker_sets_faiss_threads_before_operation(self) -> None:
        calls: list[str] = []
        fake_faiss = SimpleNamespace(omp_set_num_threads=lambda value: calls.append(f"omp:{value}"))
        self.coordinator = FaissSearchCoordinator(faiss_module=fake_faiss)

        result = await self.coordinator.run(
            "shared-index",
            lambda: calls.append("operation") or "ok",
            mode=FaissSearchConcurrencyMode.PER_INDEX,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, ["omp:1", "operation"])

    async def test_executor_is_dedicated_and_bounded(self) -> None:
        with patch.object(_MODULE.os, "cpu_count", return_value=64):
            self.coordinator = FaissSearchCoordinator()

        self.assertIsInstance(
            self.coordinator._executor,
            concurrent.futures.ThreadPoolExecutor,
        )
        self.assertEqual(self.coordinator._executor._max_workers, 4)
        self.assertEqual(
            self.coordinator._executor._thread_name_prefix,
            "ragtime-faiss-",
        )

    async def test_shutdown_rejects_new_work(self) -> None:
        self.coordinator = FaissSearchCoordinator()
        self.coordinator.shutdown()

        with self.assertRaises(RuntimeError):
            await self.coordinator.run(
                "shared-index",
                lambda: "never",
                mode=FaissSearchConcurrencyMode.PER_INDEX,
            )


if __name__ == "__main__":
    unittest.main()
