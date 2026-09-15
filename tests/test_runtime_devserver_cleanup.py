from __future__ import annotations

import asyncio
import importlib
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from tests._runtime_cleanup_helpers import assert_process_group_termination

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

    def _build_session(self, workspace_root: Path) -> Any:
        assert worker_service is not None
        return worker_service.WorkerSession(
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

    async def test_terminates_entire_process_group(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        await assert_process_group_termination(
            self,
            terminate=service._terminate_devserver_process,
            expected_patch_base="runtime.worker.service",
        )

    async def test_escalates_process_group_after_timeout(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        await assert_process_group_termination(
            self,
            terminate=service._terminate_devserver_process,
            expected_patch_base="runtime.worker.service",
            timeout=0.01,
        )

    async def test_exec_timeout_terminates_entire_process_group(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()

        async def hang() -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return (b"", b"")

        wait_calls = 0

        async def wait() -> None:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                await asyncio.sleep(10)

        process = SimpleNamespace(
            pid=1234,
            returncode=None,
            communicate=mock.AsyncMock(side_effect=hang),
            wait=mock.AsyncMock(side_effect=wait),
            terminate=mock.Mock(),
            kill=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._build_session(Path(tmpdir))
            service._sessions[session.id] = session

            with (
                mock.patch.object(worker_service, "spawn_sandboxed", new=mock.AsyncMock(return_value=process)),
                mock.patch("runtime.worker.sandbox.os.getpgid", return_value=process.pid),
                mock.patch("runtime.worker.sandbox.os.killpg") as killpg,
            ):
                response = await service.exec_command(
                    session.id,
                    "sleep 30",
                    timeout_seconds=1,
                )

        self.assertTrue(response.timed_out)
        killpg.assert_any_call(process.pid, signal.SIGTERM)
        killpg.assert_any_call(process.pid, signal.SIGKILL)
        self.assertNotIn(mock.call(), process.kill.mock_calls)

    async def test_scheduling_startup_clears_stale_devserver_port(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            session = self._build_session(workspace_root)

            with mock.patch.object(service, "_run_startup_pipeline", mock.AsyncMock(return_value=None)):
                service._schedule_startup_locked(session)
                task = service._startup_tasks[session.id]
                await task

        self.assertIsNone(session.devserver_port)
        self.assertEqual(session.state, "starting")
        self.assertEqual(session.runtime_operation_phase, "queued")

    async def test_demand_start_reuses_active_startup_task(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        blocker = asyncio.Event()

        async def blocked_startup(*_args: Any, **_kwargs: Any) -> None:
            await blocker.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._build_session(Path(tmpdir))
            service._sessions[session.id] = session

            with mock.patch.object(service, "_run_startup_pipeline", side_effect=blocked_startup):
                with self.assertRaises(worker_service.HTTPException):
                    await service.build_preview_upstream_url(session.id, "")

                first_task = service._startup_tasks[session.id]
                first_operation_id = session.runtime_operation_id

                with self.assertRaises(worker_service.HTTPException):
                    await service.build_preview_upstream_url(session.id, "")

                self.assertIs(service._startup_tasks[session.id], first_task)
                self.assertFalse(first_task.cancelled())
                self.assertEqual(session.runtime_operation_id, first_operation_id)

                blocker.set()
                await first_task

    async def test_explicit_startup_restart_replaces_active_task(self) -> None:
        assert worker_service is not None
        service = worker_service.WorkerService()
        blocker = asyncio.Event()

        async def blocked_startup(*_args: Any, **_kwargs: Any) -> None:
            await blocker.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._build_session(Path(tmpdir))

            with mock.patch.object(service, "_run_startup_pipeline", side_effect=blocked_startup):
                service._schedule_startup_locked(session)
                first_task = service._startup_tasks[session.id]
                first_operation_id = session.runtime_operation_id

                service._schedule_startup_locked(session)
                second_task = service._startup_tasks[session.id]

                self.assertIsNot(second_task, first_task)
                await asyncio.sleep(0)
                self.assertTrue(first_task.done())
                with self.assertRaises(asyncio.CancelledError):
                    first_task.result()
                self.assertNotEqual(session.runtime_operation_id, first_operation_id)

                blocker.set()
                await second_task

    async def test_stop_does_not_block_unrelated_session_status_during_cleanup(self) -> None:
        """A slow process termination for A must not hold the worker state lock."""
        assert worker_service is not None
        service = worker_service.WorkerService()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        class _RunningProcess:
            returncode = None

        async def slow_terminate(_process: Any) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._build_session(Path(tmpdir))
            second = self._build_session(Path(tmpdir) / "other")
            second.id = "wkr-2"
            second.workspace_id = "workspace-2"
            service._sessions[first.id] = first
            service._sessions[second.id] = second
            service._devserver_processes[first.id] = _RunningProcess()

            with mock.patch.object(service, "_terminate_devserver_process", side_effect=slow_terminate):
                stop_task = asyncio.create_task(service.stop_session(first.id))
                try:
                    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                    response = await asyncio.wait_for(service.get_session(second.id), timeout=0.1)
                    self.assertEqual(response.worker_session_id, second.id)
                finally:
                    allow_cleanup.set()
                await stop_task

    async def test_same_workspace_start_waits_for_stop_cleanup(self) -> None:
        """A replacement pipeline must not provision until its workspace cleanup drains."""
        assert worker_service is not None
        service = worker_service.WorkerService()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        provision_started = asyncio.Event()

        class _RunningProcess:
            returncode = None

        async def slow_terminate(_process: Any) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        def record_provision(_spec: Any) -> None:
            provision_started.set()

        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._build_session(Path(tmpdir))
            first.runtime_operation_id = "op-1"
            replacement = self._build_session(Path(tmpdir) / "replacement")
            replacement.id = "wkr-2"
            replacement.runtime_operation_id = "op-2"
            replacement.state = "starting"
            service._sessions[first.id] = first
            service._sessions[replacement.id] = replacement
            service._devserver_processes[first.id] = _RunningProcess()

            with (
                mock.patch.object(service, "_terminate_devserver_process", side_effect=slow_terminate),
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=record_provision),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_resolve_devserver_command", return_value=worker_service.DevserverResolution()),
            ):
                stop_task = asyncio.create_task(service.stop_session(first.id))
                try:
                    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                    start_task = asyncio.create_task(service._run_startup_pipeline(replacement.id, "op-2"))
                    await asyncio.sleep(0)
                    self.assertFalse(provision_started.is_set())
                finally:
                    allow_cleanup.set()
                await stop_task
                await asyncio.wait_for(provision_started.wait(), timeout=1)
                await start_task

    async def test_same_workspace_start_waits_for_registered_stop_barrier(self) -> None:
        """A start queued while stop drains its startup task cannot overtake cleanup."""
        assert worker_service is not None
        service = worker_service.WorkerService()
        cancellation_started = asyncio.Event()
        release_cancelled_startup = asyncio.Event()
        provision_started = asyncio.Event()

        async def cancelled_startup() -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_started.set()
                await release_cancelled_startup.wait()
                raise

        def record_provision(_spec: Any) -> None:
            provision_started.set()

        with tempfile.TemporaryDirectory() as tmpdir:
            first = self._build_session(Path(tmpdir))
            first.runtime_operation_id = "op-1"
            replacement = self._build_session(Path(tmpdir) / "replacement")
            replacement.id = "wkr-2"
            replacement.runtime_operation_id = "op-2"
            replacement.state = "starting"
            service._sessions[first.id] = first
            service._sessions[replacement.id] = replacement
            service._startup_tasks[first.id] = asyncio.create_task(cancelled_startup())

            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=record_provision),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_resolve_devserver_command", return_value=worker_service.DevserverResolution()),
            ):
                stop_task = asyncio.create_task(service.stop_session(first.id))
                try:
                    await asyncio.wait_for(cancellation_started.wait(), timeout=1)
                    start_task = asyncio.create_task(service._run_startup_pipeline(replacement.id, "op-2"))
                    await asyncio.sleep(0)
                    self.assertFalse(provision_started.is_set())
                finally:
                    release_cancelled_startup.set()
                await stop_task
                await asyncio.wait_for(provision_started.wait(), timeout=1)
                await start_task

    async def test_stop_cancellation_keeps_cleanup_barrier_until_thread_finishes(self) -> None:
        """Cancelling the caller cannot release a cleanup thread's workspace fence."""
        assert worker_service is not None
        service = worker_service.WorkerService()
        cleanup_started = threading.Event()
        allow_cleanup = threading.Event()

        def slow_cleanup(_spec: Any) -> None:
            cleanup_started.set()
            allow_cleanup.wait(timeout=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._build_session(Path(tmpdir))
            service._sessions[session.id] = session
            with mock.patch("runtime.worker.service.cleanup_sandbox", side_effect=slow_cleanup):
                stop_task = asyncio.create_task(service.stop_session(session.id))
                try:
                    await asyncio.wait_for(asyncio.to_thread(cleanup_started.wait), timeout=1)
                    stop_task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await stop_task
                    self.assertIn(session.workspace_id, service._workspace_cleanup_tasks)
                finally:
                    allow_cleanup.set()
                await asyncio.wait_for(service._workspace_cleanup_tasks[session.workspace_id], timeout=1)

    async def test_shutdown_drains_tracked_background_cleanup(self) -> None:
        """Shutdown waits for tracked cleanup that runs outside the state lock."""
        assert worker_service is not None
        service = worker_service.WorkerService()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def blocked_cleanup() -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        cleanup_task = asyncio.create_task(blocked_cleanup())
        service._track_background_cleanup(cleanup_task)
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=1)
            shutdown_task = asyncio.create_task(service.shutdown())
            await asyncio.sleep(0)
            self.assertFalse(shutdown_task.done())
        finally:
            allow_cleanup.set()
        await asyncio.wait_for(shutdown_task, timeout=1)


if __name__ == "__main__":
    unittest.main()
