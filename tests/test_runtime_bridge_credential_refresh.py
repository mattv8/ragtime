from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from runtime.worker.sandbox import SandboxSpec
from runtime.worker.service import WorkerService, WorkerSession


def _token(session_id: str, exp: int = 1735693200) -> str:
    encoded = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()
    return f"{encoded({'alg': 'HS256'})}.{encoded({'kind': 'userspace_runtime_bridge', 'workspace_id': 'ws', 'session_id': session_id, 'iat': 1735689600, 'exp': exp})}.signature"


class RuntimeBridgeCredentialRefreshTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, root: Path) -> tuple[WorkerService, WorkerSession]:
        files = root / "files"
        files.mkdir()
        session = WorkerSession(
            id="worker", workspace_id="ws", provider_session_id="provider", workspace_root=root,
            workspace_files_path=files, sandbox_spec=SandboxSpec(workspace_id="ws", workspace_files_path=files, rootfs_path=root / "rootfs"),
            pty_access_token="pty", workspace_env={"RAGTIME_BRIDGE_URL": "http://bridge"}, workspace_env_visibility={}, workspace_mounts=[], mount_targets_to_clear=set(),
            state="running", devserver_running=True, devserver_port=None, devserver_command=None, launch_framework=None, launch_cwd=None, last_error=None,
            runtime_operation_id=None, runtime_operation_phase=None, runtime_operation_started_at=None, runtime_operation_updated_at=None, updated_at=datetime.now(timezone.utc),
            bridge_credential_mode="worker_file", bridge_session_id="db-session", bridge_token_file_initial_token=_token("db-session"),
        )
        service = WorkerService()
        service._sessions[session.id] = session
        return service, session

    async def test_refresh_cas_duplicate_and_stale_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, session = self._session(Path(directory))
            session.sandbox_spec.rootfs_path.mkdir()
            service._write_bridge_token_file(session, session.bridge_token_file_initial_token or "")
            first = await service.refresh_bridge_credential(session.id, token=_token("db-session", 1735693300), expected_session_id="db-session", expected_revision=0, request_id="request-1")
            duplicate = await service.refresh_bridge_credential(session.id, token=_token("db-session", 1735693300), expected_session_id="db-session", expected_revision=0, request_id="request-1")
            self.assertEqual(first.revision, 1)
            self.assertEqual(duplicate.revision, 1)
            with self.assertRaises(HTTPException) as stale:
                await service.refresh_bridge_credential(session.id, token=_token("db-session", 1735693400), expected_session_id="db-session", expected_revision=0, request_id="request-2")
            self.assertEqual(stale.exception.status_code, 409)
            with self.assertRaises(HTTPException) as conflict:
                await service.refresh_bridge_credential(session.id, token=_token("db-session", 1735693500), expected_session_id="db-session", expected_revision=1, request_id="request-1")
            self.assertEqual(conflict.exception.status_code, 409)

    async def test_refresh_history_evicts_stale_requests_without_reapplying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, session = self._session(Path(directory))
            session.sandbox_spec.rootfs_path.mkdir()
            service._write_bridge_token_file(session, session.bridge_token_file_initial_token or "")

            for revision in range(33):
                refreshed = await service.refresh_bridge_credential(
                    session.id,
                    token=_token("db-session", 1735693300 + revision),
                    expected_session_id="db-session",
                    expected_revision=revision,
                    request_id=f"request-{revision}",
                )
                self.assertEqual(refreshed.revision, revision + 1)

            duplicate = await service.refresh_bridge_credential(
                session.id,
                token=_token("db-session", 1735693332),
                expected_session_id="db-session",
                expected_revision=32,
                request_id="request-32",
            )
            self.assertEqual(duplicate.revision, 33)
            self.assertEqual(session.bridge_credential_revision, 33)
            self.assertEqual(len(session.bridge_refresh_requests), 32)

            with self.assertRaises(HTTPException) as stale:
                await service.refresh_bridge_credential(
                    session.id,
                    token=_token("db-session", 1735693400),
                    expected_session_id="db-session",
                    expected_revision=0,
                    request_id="request-0",
                )
            self.assertEqual(stale.exception.status_code, 409)
            self.assertEqual(session.bridge_credential_revision, 33)
            self.assertEqual(service._read_bridge_token_file(session), _token("db-session", 1735693332))

    def test_private_file_rejects_malicious_run_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as target:
            service, session = self._session(Path(directory))
            rootfs = session.sandbox_spec.rootfs_path
            rootfs.mkdir()
            os.symlink(target, rootfs / "run")
            with self.assertRaises(HTTPException):
                service._write_bridge_token_file(session, _token("db-session"))
            self.assertFalse((Path(target) / ".ragtime-bridge" / "token").exists())

    async def test_recycle_retains_private_token_without_workspace_env_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service, session = self._session(Path(directory))
            with mock.patch.object(service, "_run_startup_pipeline", new=mock.AsyncMock()):
                response = await service.restart_app(session.id, "restart-1")
                await service._startup_tasks[session.id]
            self.assertEqual(response.state, "starting")
            self.assertNotIn("RAGTIME_BRIDGE_TOKEN", session.workspace_env)
            self.assertEqual(session.bridge_token_file_initial_token, _token("db-session"))
