from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from pydantic import ValidationError

from ragtime.core.userspace_limits import USERSPACE_EXEC_TIMEOUT_HARD_CAP_SECONDS
from runtime.core.shared import RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS
from runtime.manager.models import RuntimeExecRequest
from runtime.worker import service as worker_service


class RuntimeExecTimeoutLimitTests(unittest.IsolatedAsyncioTestCase):
    def _build_session(self, workspace_root: Path) -> Any:
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
            devserver_port=None,
            devserver_command=None,
            launch_framework="node",
            launch_cwd=".",
            last_error=None,
            runtime_operation_id=None,
            runtime_operation_phase=None,
            runtime_operation_started_at=None,
            runtime_operation_updated_at=None,
            updated_at=worker_service.utc_now(),
        )

    def test_runtime_cap_matches_control_plane_cap(self) -> None:
        self.assertEqual(RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS, 3600)
        self.assertEqual(RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS, USERSPACE_EXEC_TIMEOUT_HARD_CAP_SECONDS)

    def test_runtime_exec_schema_accepts_hard_cap_and_rejects_above_it(self) -> None:
        self.assertEqual(RuntimeExecRequest(command="true", timeout_seconds=3600).timeout_seconds, 3600)
        with self.assertRaises(ValidationError):
            RuntimeExecRequest(command="true", timeout_seconds=3601)

    async def test_worker_clamps_direct_exec_to_hard_cap(self) -> None:
        service = worker_service.WorkerService()
        process = SimpleNamespace(
            returncode=0,
            communicate=mock.AsyncMock(return_value=(b"", b"")),
        )
        captured_timeout: int | None = None

        async def capture_timeout(awaitable: Any, *, timeout: int) -> tuple[bytes, bytes]:
            nonlocal captured_timeout
            captured_timeout = timeout
            return await awaitable

        with tempfile.TemporaryDirectory() as tmpdir:
            session = self._build_session(Path(tmpdir))
            service._sessions[session.id] = session
            with (
                mock.patch.object(worker_service, "spawn_sandboxed", new=mock.AsyncMock(return_value=process)),
                mock.patch.object(asyncio, "wait_for", new=capture_timeout),
            ):
                response = await service.exec_command(session.id, "true", timeout_seconds=5000)

        self.assertEqual(captured_timeout, RUNTIME_EXEC_TIMEOUT_HARD_CAP_SECONDS)
        self.assertEqual(response.exit_code, 0)
