from __future__ import annotations

import asyncio
import importlib
import signal
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

worker_api: Any | None
runtime_import_error: ImportError | None
try:
    worker_api = importlib.import_module("runtime.worker.api")
except ImportError as exc:
    worker_api = None
    runtime_import_error = exc
else:
    runtime_import_error = None


class RuntimePtyCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if worker_api is None:
            self.skipTest(f"runtime worker unavailable: {runtime_import_error}")

    async def _assert_process_group_termination(self, *, timeout: float | None = None) -> None:
        assert worker_api is not None
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
            if timeout is not None and wait_calls == 1:
                await asyncio.sleep(10)

        if timeout is None:
            process.wait = mock.AsyncMock(return_value=None)
        else:
            process.wait = mock.AsyncMock(side_effect=wait)

        with (
            mock.patch("runtime.worker.api.os.getpgid", return_value=4321),
            mock.patch("runtime.worker.api.os.killpg") as killpg,
        ):
            if timeout is None:
                await worker_api._terminate_pty_process(process)
                self.assertEqual(killpg.mock_calls, [mock.call(4321, signal.SIGTERM)])
                process.wait.assert_awaited_once()
            else:
                await worker_api._terminate_pty_process(process, timeout=timeout)
                self.assertEqual(
                    killpg.mock_calls,
                    [
                        mock.call(4321, signal.SIGTERM),
                        mock.call(4321, signal.SIGKILL),
                    ],
                )
                self.assertEqual(process.wait.await_count, 2)

        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    async def test_terminates_entire_pty_process_group(self) -> None:
        await self._assert_process_group_termination()

    async def test_escalates_pty_process_group_after_timeout(self) -> None:
        await self._assert_process_group_termination(timeout=0.01)


if __name__ == "__main__":
    unittest.main()
