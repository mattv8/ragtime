from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from typing import Any
from unittest import mock


async def assert_process_group_termination(
    test_case: Any,
    *,
    terminate: Any,
    expected_patch_base: str,
    timeout: float | None = None,
) -> None:
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
        mock.patch(f"{expected_patch_base}.os.getpgid", return_value=4321),
        mock.patch(f"{expected_patch_base}.os.killpg") as killpg,
    ):
        if timeout is None:
            await terminate(process)
            test_case.assertEqual(killpg.mock_calls, [mock.call(4321, signal.SIGTERM)])
            process.wait.assert_awaited_once()
        else:
            await terminate(process, timeout=timeout)
            test_case.assertEqual(
                killpg.mock_calls,
                [
                    mock.call(4321, signal.SIGTERM),
                    mock.call(4321, signal.SIGKILL),
                ],
            )
            test_case.assertEqual(process.wait.await_count, 2)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()
