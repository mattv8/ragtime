from __future__ import annotations

import asyncio
import importlib
import signal
import tempfile
import unittest
from pathlib import Path
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

    async def _assert_process_group_termination(
        self,
        *,
        terminate,
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
                self.assertEqual(killpg.mock_calls, [mock.call(4321, signal.SIGTERM)])
                process.wait.assert_awaited_once()
            else:
                await terminate(process, timeout=timeout)
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

    async def test_terminates_entire_process_group(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        await self._assert_process_group_termination(
            terminate=service._terminate_devserver_process,
            expected_patch_base="runtime.worker.service",
        )

    async def test_escalates_process_group_after_timeout(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        await self._assert_process_group_termination(
            terminate=service._terminate_devserver_process,
            expected_patch_base="runtime.worker.service",
            timeout=0.01,
        )

    async def test_scheduling_startup_clears_stale_devserver_port(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            session = worker_service.WorkerSession(
                id="wkr-1",
                workspace_id="workspace-1",
                provider_session_id="mgr-1",
                workspace_root=workspace_root,
                workspace_files_path=workspace_root / "files",
                sandbox_spec=worker_service.SandboxSpec(
                    workspace_id="workspace-1",
                    workspace_files_path=workspace_root / "files",
                    rootfs_path=workspace_root / "rootfs",
                ),
                pty_access_token="token",
                workspace_env={},
                workspace_env_visibility={},
                workspace_mounts=[],
                mount_targets_to_clear=set(),
                state="running",
                devserver_running=False,
                devserver_port=50481,
                devserver_command=None,
                launch_framework="node",
                launch_cwd=".",
                last_error="Dev server exited with code 1: EADDRINUSE",
                runtime_operation_id=None,
                runtime_operation_phase="failed",
                runtime_operation_started_at=None,
                runtime_operation_updated_at=None,
                updated_at=worker_service.utc_now(),
            )

            with mock.patch.object(service, "_run_startup_pipeline", mock.AsyncMock(return_value=None)):
                service._schedule_startup_locked(session)
                task = service._startup_tasks[session.id]
                await task

        self.assertIsNone(session.devserver_port)
        self.assertEqual(session.state, "starting")
        self.assertEqual(session.runtime_operation_phase, "queued")


if __name__ == "__main__":
    unittest.main()
