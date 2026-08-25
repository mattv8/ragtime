from __future__ import annotations

import asyncio
import importlib
import tempfile
import unittest
from pathlib import Path
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


class _StopPipeline(Exception):
    pass


class RuntimeWorkerStartupOverlapTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if worker_service is None:
            self.skipTest(f"runtime worker unavailable: {runtime_import_error}")

    def _service(self) -> Any:
        assert worker_service is not None
        return worker_service.WorkerService()

    def _install_session(
        self,
        service: Any,
        workspace_root: Path,
        *,
        session_id: str = "sess-1",
        operation_id: str = "op-1",
    ) -> Any:
        assert worker_service is not None
        workspace_files = workspace_root / "files"
        workspace_files.mkdir(parents=True, exist_ok=True)
        session = worker_service.WorkerSession(
            id=session_id,
            workspace_id="workspace-1",
            provider_session_id="mgr-1",
            workspace_root=workspace_root,
            workspace_files_path=workspace_files,
            sandbox_spec=worker_service.SandboxSpec(
                workspace_id="workspace-1",
                workspace_files_path=workspace_files,
                rootfs_path=workspace_root / "rootfs",
            ),
            pty_access_token="token",
            workspace_env={},
            workspace_env_visibility={},
            workspace_mounts=[],
            mount_targets_to_clear=set(),
            state="starting",
            devserver_running=False,
            devserver_port=None,
            devserver_command=None,
            launch_framework=None,
            launch_cwd=None,
            last_error=None,
            runtime_operation_id=operation_id,
            runtime_operation_phase="queued",
            runtime_operation_started_at=None,
            runtime_operation_updated_at=None,
            updated_at=worker_service.utc_now(),
        )
        service._sessions[session_id] = session
        return session

    async def test_object_storage_overlaps_deps_install(self) -> None:
        service = self._service()
        order: list[str] = []
        storage_started = asyncio.Event()

        async def fake_deps(_session: Any) -> str | None:
            order.append("deps_start")
            await asyncio.sleep(0)
            await asyncio.wait_for(storage_started.wait(), timeout=1.0)
            order.append("deps_end")
            return None

        async def fake_storage(_session: Any) -> str | None:
            order.append("storage_start")
            storage_started.set()
            order.append("storage_end")
            return None

        def stop_pipeline(*_args: Any, **_kwargs: Any) -> Any:
            order.append("post_join")
            raise _StopPipeline()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", side_effect=fake_deps),
                mock.patch.object(service, "_start_object_storage_locked", side_effect=fake_storage),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_set_operation_phase"),
                mock.patch.object(service, "_resolve_devserver_command", side_effect=stop_pipeline),
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                with self.assertRaises(_StopPipeline):
                    await service._run_startup_pipeline("sess-1", "op-1")

        self.assertEqual(order, ["deps_start", "storage_start", "storage_end", "deps_end", "post_join"])

    async def test_deps_failure_terminates_started_object_storage_before_mark_failed(self) -> None:
        service = self._service()
        terminated = mock.AsyncMock()
        failed = mock.AsyncMock()

        async def fake_deps_failure(_session: Any) -> str | None:
            await asyncio.sleep(0)
            return "npm ci failed"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", side_effect=fake_deps_failure),
                mock.patch.object(service, "_start_object_storage_locked", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_terminate_object_storage_locked", new=terminated),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_set_operation_phase"),
                mock.patch.object(service, "_mark_operation_failed", new=failed),
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                await service._run_startup_pipeline("sess-1", "op-1")

        terminated.assert_awaited_once_with("sess-1")
        failed.assert_awaited_once_with("sess-1", "op-1", "npm ci failed")

    async def test_object_storage_error_fails_pipeline_after_deps(self) -> None:
        service = self._service()

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_start_object_storage_locked", new=mock.AsyncMock(return_value="storage broke")),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_set_operation_phase") as set_phase,
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                await service._run_startup_pipeline("sess-1", "op-1")

        self.assertEqual(session.state, "running")
        self.assertFalse(session.devserver_running)
        self.assertEqual(session.last_error, "storage broke")
        set_phase.assert_any_call(session, "failed")

    async def test_dependency_error_wins_over_object_storage_exception(self) -> None:
        service = self._service()
        terminated = mock.AsyncMock()
        failed = mock.AsyncMock()
        storage_started = asyncio.Event()

        async def fake_storage(*_args: Any) -> str | None:
            storage_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                raise RuntimeError("storage startup boom")
            raise AssertionError("unreachable")

        async def fake_deps(_session: Any) -> str | None:
            await asyncio.wait_for(storage_started.wait(), timeout=1.0)
            return "npm ci failed"

        with tempfile.TemporaryDirectory() as tmpdir:
            self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=fake_deps),
                mock.patch.object(service, "_start_object_storage_for_pipeline", new=fake_storage),
                mock.patch.object(service, "_terminate_object_storage_locked", new=terminated),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_set_operation_phase"),
                mock.patch.object(service, "_mark_operation_failed", new=failed),
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                await asyncio.wait_for(service._run_startup_pipeline("sess-1", "op-1"), timeout=1.0)

        terminated.assert_awaited_once_with("sess-1")
        failed.assert_awaited_once_with("sess-1", "op-1", "npm ci failed")

    async def test_pipeline_cancellation_preserves_original_cancelled_error(self) -> None:
        service = self._service()
        terminated = mock.AsyncMock()
        storage_started = asyncio.Event()

        async def fake_storage(*_args: Any) -> str | None:
            storage_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                raise RuntimeError("storage cancel cleanup boom")
            raise AssertionError("unreachable")

        async def fake_deps(_session: Any) -> str | None:
            await asyncio.wait_for(storage_started.wait(), timeout=1.0)
            await asyncio.Future()
            raise AssertionError("unreachable")

        with tempfile.TemporaryDirectory() as tmpdir:
            self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=fake_deps),
                mock.patch.object(service, "_start_object_storage_for_pipeline", new=fake_storage),
                mock.patch.object(service, "_terminate_object_storage_locked", new=terminated),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_set_operation_phase"),
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                task = asyncio.create_task(service._run_startup_pipeline("sess-1", "op-1"))
                await asyncio.wait_for(storage_started.wait(), timeout=1.0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1.0)

        terminated.assert_awaited_once_with("sess-1")

    async def test_object_storage_start_cancellation_kills_unregistered_process(self) -> None:
        service = self._service()
        ready_wait_started = asyncio.Event()
        assert worker_service is not None

        class _FakeStream:
            async def readline(self) -> bytes:
                ready_wait_started.set()
                await asyncio.Future()
                raise AssertionError("unreachable")

            async def read(self) -> bytes:
                return b""

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStream()
                self.stderr = _FakeStream()
                self.returncode: int | None = None
                self.kill_called = False
                self.wait_called = False

            def kill(self) -> None:
                self.kill_called = True
                self.returncode = -9

            async def wait(self) -> int:
                self.wait_called = True
                return -9

        fake_process = _FakeProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_read_workspace_object_storage_config", return_value={"buckets": [{"name": "uploads"}]}),
                mock.patch.object(service, "_extract_object_storage_buckets", return_value=[{"name": "uploads"}]),
                mock.patch.object(worker_service.shutil, "which", return_value="/usr/bin/node"),
                mock.patch.object(service, "_pick_free_port", return_value=4501),
                mock.patch.object(worker_service.asyncio, "create_subprocess_exec", new=mock.AsyncMock(return_value=fake_process)),
            ):
                task = asyncio.create_task(service._start_object_storage_locked(session))
                await asyncio.wait_for(ready_wait_started.wait(), timeout=1.0)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        self.assertTrue(fake_process.kill_called)
        self.assertTrue(fake_process.wait_called)


if __name__ == "__main__":
    unittest.main()
