from __future__ import annotations

import asyncio
import importlib
import signal
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

worker_service: Any | None
runtime_import_error: ImportError | None
try:
    worker_service = importlib.import_module("runtime.worker.service")
except ImportError as exc:
    worker_service = None
    runtime_import_error = exc
else:
    runtime_import_error = None


class RuntimeDevserverCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if worker_service is None:
            self.skipTest(f"runtime worker unavailable: {runtime_import_error}")

    async def test_terminates_entire_process_group(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        process = SimpleNamespace(
            pid=1234,
            returncode=None,
            terminate=mock.Mock(),
            kill=mock.Mock(),
            wait=mock.AsyncMock(return_value=None),
        )

        with (
            mock.patch("runtime.worker.service.os.getpgid", return_value=4321),
            mock.patch("runtime.worker.service.os.killpg") as killpg,
        ):
            await service._terminate_devserver_process(process)

        killpg.assert_called_once_with(4321, signal.SIGTERM)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        process.wait.assert_awaited_once()

    async def test_escalates_process_group_after_timeout(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        process = SimpleNamespace(
            pid=1234,
            returncode=None,
            terminate=mock.Mock(),
            kill=mock.Mock(),
        )
        wait_calls = 0

        async def wait() -> None:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                await asyncio.sleep(10)

        process.wait = mock.AsyncMock(side_effect=wait)

        with (
            mock.patch("runtime.worker.service.os.getpgid", return_value=4321),
            mock.patch("runtime.worker.service.os.killpg") as killpg,
        ):
            await service._terminate_devserver_process(process, timeout=0.01)

        self.assertEqual(
            killpg.mock_calls,
            [
                mock.call(4321, signal.SIGTERM),
                mock.call(4321, signal.SIGKILL),
            ],
        )
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        self.assertEqual(process.wait.await_count, 2)


if __name__ == "__main__":
    unittest.main()