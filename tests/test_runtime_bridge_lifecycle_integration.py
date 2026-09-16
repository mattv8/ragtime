from __future__ import annotations

import base64
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from runtime.worker.sandbox import SandboxSpec
from runtime.worker.service import DevserverResolution, WorkerService, WorkerSession


def _token(session_id: str, expiry: int) -> str:
    encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
    return f"{encode({'alg': 'HS256'})}.{encode({'kind': 'userspace_runtime_bridge', 'workspace_id': 'ws', 'session_id': session_id, 'iat': 1735689600, 'exp': expiry})}.signature"


class _Process:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0


class RuntimeBridgeLifecycleIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_startup_pipeline_refresh_and_recycle_keep_latest_private_token(self) -> None:
        """Exercise the real pipeline while only replacing provisioning/spawn boundaries."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = root / "files"
            files.mkdir()
            first = _token("session-1", 1735693200)
            latest = _token("session-1", 1735696800)
            service = WorkerService()
            session = WorkerSession(
                id="worker-1",
                workspace_id="ws",
                provider_session_id="provider",
                workspace_root=root,
                workspace_files_path=files,
                sandbox_spec=SandboxSpec(workspace_id="ws", workspace_files_path=files, rootfs_path=root / "rootfs"),
                pty_access_token="pty",
                workspace_env={"RAGTIME_BRIDGE_URL": "http://bridge", "RAGTIME_BRIDGE_TOKEN_FILE": "/run/.ragtime-bridge/token"},
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
                runtime_operation_id="start-1",
                runtime_operation_phase="queued",
                runtime_operation_started_at=None,
                runtime_operation_updated_at=None,
                updated_at=datetime.now(timezone.utc),
                bridge_credential_mode="worker_file",
                bridge_session_id="session-1",
                bridge_token_file_initial_token=first,
            )
            service._sessions[session.id] = session
            spawned = [_Process(1001), _Process(1002)]

            def provision(spec: SandboxSpec) -> None:
                spec.rootfs_path.mkdir(parents=True, exist_ok=True)

            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=provision),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_resolve_devserver_command", return_value=DevserverResolution(command=["python3", "app.py"], port=8000)),
                mock.patch("runtime.worker.service.spawn_sandboxed", new=mock.AsyncMock(side_effect=spawned)),
                mock.patch.object(service, "_wait_devserver_ready", new=mock.AsyncMock(return_value=True)),
                mock.patch("runtime.worker.sandbox.os.getpgid", side_effect=ProcessLookupError()),
            ):
                await service._run_startup_pipeline(session.id, "start-1")
                self.assertEqual(service._read_bridge_token_file(session), first)
                self.assertNotIn("RAGTIME_BRIDGE_TOKEN", session.workspace_env)
                before_pid = service._devserver_processes[session.id].pid
                refreshed = await service.refresh_bridge_credential(
                    session.id, token=latest, expected_session_id="session-1", expected_revision=0, request_id="refresh-1"
                )
                self.assertEqual(refreshed.revision, 1)
                self.assertEqual(service._devserver_processes[session.id].pid, before_pid)
                response = await service.restart_app(session.id, "restart-1")
                await service._startup_tasks[session.id]
                self.assertEqual(response.runtime_operation_phase, "queued")
                self.assertEqual(service._read_bridge_token_file(session), latest)
                self.assertEqual(session.bridge_session_id, "session-1")
                self.assertEqual(service._devserver_processes[session.id].pid, 1002)
