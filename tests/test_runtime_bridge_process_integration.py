from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any
from unittest import mock

from runtime.worker.sandbox import SandboxSpec
from runtime.worker.service import DevserverResolution, WorkerService, WorkerSession


def _token(session_id: str, expiry: int) -> str:
    def encode(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    payload = {
        "kind": "userspace_runtime_bridge",
        "workspace_id": "ws",
        "session_id": session_id,
        "iat": 1735689600,
        "exp": expiry,
    }
    return f"{encode({'alg': 'HS256'})}.{encode(payload)}.signature"


_HTTP_APP = """\
import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(os.environ['RAGTIME_BRIDGE_TOKEN_FILE'], encoding='utf-8') as token_file:
            token = token_file.read().strip()
        body = json.dumps({'token_hash': hashlib.sha256(token.encode()).hexdigest(), 'pid': os.getpid()}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()
"""


class RuntimeBridgeProcessIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_credential_refresh_and_recycle_with_real_http_process(self) -> None:
        """A local process reads the worker-managed token file on every request."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = root / "files"
            files.mkdir()
            app_path = files / "token_server.py"
            app_path.write_text(_HTTP_APP, encoding="utf-8")
            first = _token("session-1", 1735693200)
            latest = _token("session-1", 1735696800)
            service = WorkerService()
            service._devserver_start_timeout_seconds = 5
            session = WorkerSession(
                id="worker-1",
                workspace_id="ws",
                provider_session_id="provider",
                workspace_root=root,
                workspace_files_path=files,
                sandbox_spec=SandboxSpec(
                    workspace_id="ws",
                    workspace_files_path=files,
                    rootfs_path=root / "rootfs",
                ),
                pty_access_token="pty",
                workspace_env={
                    "RAGTIME_BRIDGE_URL": "http://bridge",
                    "RAGTIME_BRIDGE_TOKEN_FILE": "/run/.ragtime-bridge/token",
                },
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

            def provision(spec: SandboxSpec) -> None:
                spec.rootfs_path.mkdir(parents=True, exist_ok=True)

            async def spawn_local_process(
                spec: SandboxSpec,
                command: list[str],
                *,
                cwd: Path,
                env: dict[str, str],
                stdout: int | IO[Any] | None,
                stderr: int | IO[Any] | None,
                ensure_ready: bool,
            ) -> asyncio.subprocess.Process:
                self.assertNotIn("RAGTIME_BRIDGE_TOKEN", env)
                self.assertEqual(env["RAGTIME_BRIDGE_TOKEN_FILE"], "/run/.ragtime-bridge/token")
                local_env = {**os.environ, **env}
                # This is the only sandbox boundary replacement: the local process
                # maps the sandbox-internal path to its isolated temporary rootfs.
                local_env["RAGTIME_BRIDGE_TOKEN_FILE"] = str(spec.rootfs_path / "run" / ".ragtime-bridge" / "token")
                local_env["PORT"] = str(service._sessions[session.id].devserver_port)
                return await asyncio.create_subprocess_exec(
                    *command,
                    # The host process cannot use the sandbox's /workspace cwd.
                    cwd=str(spec.workspace_files_path),
                    env=local_env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )

            def fetch(port: int) -> dict[str, object]:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    return json.loads(response.read())

            with (
                mock.patch("runtime.worker.service.ensure_sandbox_ready", side_effect=provision),
                mock.patch.object(service, "_materialize_workspace_mounts", new=mock.AsyncMock()),
                mock.patch.object(service, "_run_workspace_bootstrap_if_needed", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(service, "_ensure_entrypoint_dependencies", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(
                    service,
                    "_resolve_devserver_command",
                    return_value=DevserverResolution(command=[sys.executable, str(app_path)]),
                ),
                mock.patch("runtime.worker.service.spawn_sandboxed", new=spawn_local_process),
                mock.patch("runtime.worker.service.cleanup_sandbox"),
            ):
                await service._run_startup_pipeline(session.id, "start-1")
                first_process = service._devserver_processes[session.id]
                first_http = await asyncio.to_thread(fetch, session.devserver_port or 0)
                self.assertEqual(first_http["pid"], first_process.pid)
                self.assertEqual(first_http["token_hash"], hashlib.sha256(first.encode()).hexdigest())

                refreshed = await service.refresh_bridge_credential(
                    session.id,
                    token=latest,
                    expected_session_id="session-1",
                    expected_revision=0,
                    request_id="refresh-1",
                )
                refreshed_http = await asyncio.to_thread(fetch, session.devserver_port or 0)
                self.assertEqual(refreshed.revision, 1)
                self.assertEqual(refreshed_http["pid"], first_process.pid)
                self.assertEqual(refreshed_http["token_hash"], hashlib.sha256(latest.encode()).hexdigest())

                await service.restart_app(session.id, "restart-1")
                await service._startup_tasks[session.id]
                second_process = service._devserver_processes[session.id]
                recycled_http = await asyncio.to_thread(fetch, session.devserver_port or 0)
                self.assertNotEqual(second_process.pid, first_process.pid)
                self.assertEqual(recycled_http["pid"], second_process.pid)
                self.assertEqual(recycled_http["token_hash"], hashlib.sha256(latest.encode()).hexdigest())
                self.assertEqual(session.bridge_session_id, "session-1")

                stopped = await service.stop_session(session.id)
                self.assertEqual(stopped.state, "stopped")
                self.assertIsNotNone(second_process.returncode)
