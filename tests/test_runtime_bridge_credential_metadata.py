from __future__ import annotations

import base64
import importlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from fastapi import HTTPException


def _jwt_segment(payload: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).rstrip(b"=").decode("ascii")


def _unsigned_jwt(payload: dict[str, object]) -> str:
    header: dict[str, object] = {"alg": "HS256", "typ": "JWT"}
    return f"{_jwt_segment(header)}.{_jwt_segment(payload)}.signature"


class RuntimeBridgeCredentialMetadataTests(unittest.TestCase):
    def _load_modules(self):
        models_module = importlib.import_module("runtime.manager.models")
        worker_service_module = importlib.import_module("runtime.worker.service")
        return models_module, worker_service_module

    def _build_session(self, worker_service_module, *, token: str | None):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"RUNTIME_WORKSPACE_ROOT": tmpdir}, clear=False):
                service = worker_service_module.WorkerService()
                workspace_root, workspace_files_path, sandbox_spec = service._resolve_workspace_root("ws-1")
                now = datetime.now(timezone.utc)
                session = worker_service_module.WorkerSession(
                    id="worker-1",
                    workspace_id="ws-1",
                    provider_session_id="provider-1",
                    workspace_root=workspace_root,
                    workspace_files_path=workspace_files_path,
                    sandbox_spec=sandbox_spec,
                    pty_access_token="pty-token",
                    workspace_env={
                        "RAGTIME_BRIDGE_URL": "http://bridge.example/runtime-bridge",
                        **({"RAGTIME_BRIDGE_TOKEN": token} if token is not None else {}),
                    },
                    workspace_env_visibility={},
                    workspace_mounts=[],
                    mount_targets_to_clear=set(),
                    state="running",
                    devserver_running=True,
                    devserver_port=4173,
                    devserver_command=["npm", "run", "dev"],
                    launch_framework="vite",
                    launch_cwd=".",
                    last_error=None,
                    runtime_operation_id=None,
                    runtime_operation_phase=None,
                    runtime_operation_started_at=None,
                    runtime_operation_updated_at=None,
                    updated_at=now,
                )
                return service, session

    def test_session_response_decodes_safe_bridge_credential_metadata(self) -> None:
        _, worker_service_module = self._load_modules()
        token = _unsigned_jwt(
            {
                "kind": "userspace_runtime_bridge",
                "workspace_id": "ws-1",
                "session_id": "runtime-session-1",
                "iat": 1735689600,
                "exp": 1735693200,
            }
        )
        service, session = self._build_session(worker_service_module, token=token)

        response = service._session_response(session)
        dumped = response.model_dump_json()

        self.assertIsNotNone(response.bridge_credential)
        self.assertEqual(response.bridge_credential.bridge_url, "http://bridge.example/runtime-bridge")
        self.assertEqual(response.bridge_credential.token_kind, "userspace_runtime_bridge")
        self.assertEqual(response.bridge_credential.workspace_id, "ws-1")
        self.assertEqual(response.bridge_credential.session_id, "runtime-session-1")
        self.assertEqual(
            response.bridge_credential.issued_at,
            datetime.fromtimestamp(1735689600, tz=timezone.utc),
        )
        self.assertEqual(
            response.bridge_credential.expires_at,
            datetime.fromtimestamp(1735693200, tz=timezone.utc),
        )
        self.assertNotIn(token, dumped)
        self.assertNotIn("RAGTIME_BRIDGE_TOKEN", dumped)

    def test_session_response_returns_none_for_malformed_bridge_token(self) -> None:
        _, worker_service_module = self._load_modules()
        service, session = self._build_session(worker_service_module, token="not-a-jwt")

        response = service._session_response(session)

        self.assertIsNone(response.bridge_credential)


if __name__ == "__main__":
    unittest.main()


class RuntimeWorkerWorkspaceIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_reuse_rejects_cross_workspace_identity_mismatch(self) -> None:
        worker_service_module = importlib.import_module("runtime.worker.service")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"RUNTIME_WORKSPACE_ROOT": tmpdir}, clear=False):
                service = worker_service_module.WorkerService()
                with mock.patch.object(
                    service,
                    "_run_startup_pipeline",
                    mock.AsyncMock(return_value=None),
                ) as run_startup_pipeline:
                    first = await service.start_session(
                        worker_service_module.WorkerStartSessionRequest(
                            workspace_id="ws-1",
                            provider_session_id="provider-1",
                            pty_access_token="pty-token",
                            workspace_env={"FIRST": "one"},
                            workspace_env_visibility={"FIRST": True},
                            workspace_mounts=[{"target_path": "/workspace/one"}],
                        )
                    )
                    startup_task = service._startup_tasks[first.worker_session_id]
                    await startup_task
                    run_startup_pipeline.assert_awaited_once()

                stored_session = service._sessions[first.worker_session_id]
                original_updated_at = stored_session.updated_at

                with self.assertRaises(HTTPException) as mismatched:
                    await service.start_session(
                        worker_service_module.WorkerStartSessionRequest(
                            workspace_id="ws-2",
                            provider_session_id="provider-1",
                            pty_access_token="new-pty-token",
                            workspace_env={"SECOND": "two"},
                            workspace_env_visibility={"SECOND": False},
                            workspace_mounts=[{"target_path": "/workspace/two"}],
                        )
                    )

                self.assertEqual(mismatched.exception.status_code, 409)
                self.assertIn("workspace", str(mismatched.exception.detail).lower())

                unchanged_session = service._sessions[first.worker_session_id]
                self.assertEqual(unchanged_session.workspace_id, "ws-1")
                self.assertEqual(unchanged_session.pty_access_token, "pty-token")
                self.assertEqual(unchanged_session.workspace_env, {"FIRST": "one"})
                self.assertEqual(unchanged_session.workspace_env_visibility, {"FIRST": True})
                self.assertEqual(unchanged_session.workspace_mounts, [{"target_path": "/workspace/one"}])
                self.assertEqual(unchanged_session.updated_at, original_updated_at)
