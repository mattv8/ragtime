from __future__ import annotations

import asyncio
import signal
import unittest
from unittest import mock

from ragtime.core import claude_code
from ragtime.core.claude_code import ClaudeCodeStatus


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid: int | None = None
        self.communicate_started = asyncio.Event()
        self.terminated = False
        self.killed = False
        self.waited = False

    async def communicate(self):  # type: ignore[no-untyped-def]
        self.communicate_started.set()
        await asyncio.Event().wait()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


class _PidProcess(_HangingProcess):
    def __init__(self, *, wait_timeouts: int = 0) -> None:
        super().__init__()
        self.pid = 12345
        self.wait_timeouts = wait_timeouts

    async def wait(self) -> int | None:
        self.waited = True
        if self.wait_timeouts > 0:
            self.wait_timeouts -= 1
            raise asyncio.TimeoutError()
        return self.returncode


class ClaudeCodeStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        claude_code._STATUS_IN_FLIGHT = None

    async def asyncTearDown(self) -> None:
        claude_code._STATUS_IN_FLIGHT = None

    async def test_run_claude_command_reaps_timed_out_process(self) -> None:
        process = _HangingProcess()

        async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
            return process

        with mock.patch("asyncio.create_subprocess_exec", fake_create_subprocess_exec):
            result_task = asyncio.create_task(claude_code._run_claude_command("claude", "--version", timeout=0.01))
            await asyncio.wait_for(process.communicate_started.wait(), timeout=1.0)
            result = await result_task

        self.assertTrue(result.timed_out)
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_uses_group_signal_only_when_requested(self) -> None:
        process = _PidProcess()

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=False)  # type: ignore[arg-type]

        killpg.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_can_signal_process_group(self) -> None:
        process = _PidProcess()

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=True)  # type: ignore[arg-type]

        killpg.assert_called_once_with(12345, signal.SIGTERM)
        self.assertFalse(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_escalates_group_signal_after_timeout(self) -> None:
        process = _PidProcess(wait_timeouts=1)

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=True)  # type: ignore[arg-type]

        self.assertEqual(
            killpg.mock_calls,
            [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)],
        )
        self.assertTrue(process.waited)

    async def test_get_claude_code_status_coalesces_concurrent_live_probes(self) -> None:
        calls = 0

        async def fake_probe() -> ClaudeCodeStatus:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return ClaudeCodeStatus(
                installed=True,
                command="/usr/local/bin/claude",
                version="1.0.0",
                has_oauth_token=False,
                has_cli_auth=True,
                auth_method="oauth",
                subscription_type="pro",
                available=True,
                error=None,
            )

        with mock.patch("ragtime.core.claude_code._probe_claude_code_status", fake_probe):
            results = await asyncio.gather(
                claude_code.get_claude_code_status(),
                claude_code.get_claude_code_status(),
                claude_code.get_claude_code_status(),
            )

        self.assertEqual(calls, 1)
        self.assertTrue(all(result.available for result in results))


if __name__ == "__main__":
    unittest.main()
