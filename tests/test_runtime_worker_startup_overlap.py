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

    async def test_dependency_failure_marks_startup_failed_without_storage_sidecar(self) -> None:
        service = self._service()
        failed = mock.AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._install_session(service, Path(tmpdir))
            with (
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value="npm ci failed")),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_mark_operation_failed", new=failed),
                mock.patch("runtime.worker.service.ensure_sandbox_ready"),
            ):
                await service._run_startup_pipeline("sess-1", "op-1")

        failed.assert_awaited_once_with("sess-1", "op-1", "npm ci failed")

    async def test_stop_invalidates_operation_before_cancelled_startup_can_commit(self) -> None:
        """Stopping a workspace fences a cancelled spawn before cleanup starts."""
        service = self._service()
        cancellation_observed = asyncio.Event()

        async def blocked_startup(session_id: str, operation_id: str) -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                session = service._sessions[session_id]
                self.assertIsNone(session.runtime_operation_id)
                self.assertNotEqual(session.runtime_operation_id, operation_id)
                cancellation_observed.set()
                raise

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._install_session(service, Path(tmpdir))
            with mock.patch.object(service, "_run_startup_pipeline", side_effect=blocked_startup):
                service._startup_tasks[session.id] = asyncio.create_task(service._run_startup_pipeline(session.id, "op-1"))
                await asyncio.sleep(0)
                response = await service.stop_session(session.id)

        self.assertTrue(cancellation_observed.is_set())
        self.assertEqual(response.state, "stopped")


if __name__ == "__main__":
    unittest.main()
