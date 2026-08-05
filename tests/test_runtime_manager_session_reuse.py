from __future__ import annotations

import importlib
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException


def _bridge_credential(models_module, *, workspace_id: str, session_id: str):
    now = datetime.now(timezone.utc)
    return models_module.RuntimeBridgeCredentialMetadata(
        bridge_url="http://bridge.example/runtime-bridge",
        token_kind="userspace_runtime_bridge",
        workspace_id=workspace_id,
        session_id=session_id,
        issued_at=now,
        expires_at=now,
    )


def _worker_session_response(
    models_module,
    *,
    workspace_id: str,
    worker_session_id: str,
    launch_port: int | None = None,
    bridge_session_id: str = "runtime-session-1",
):
    now = datetime.now(timezone.utc)
    return models_module.WorkerSessionResponse(
        worker_session_id=worker_session_id,
        workspace_id=workspace_id,
        state="starting",
        preview_internal_url=f"http://runtime/{workspace_id}",
        launch_framework="vite",
        launch_command="npm run dev",
        launch_cwd=".",
        launch_port=launch_port,
        runtime_capabilities=None,
        devserver_running=False,
        last_error=None,
        runtime_operation_id="op-1",
        runtime_operation_phase="starting",
        runtime_operation_started_at=now,
        runtime_operation_updated_at=now,
        updated_at=now,
        bridge_credential=_bridge_credential(
            models_module,
            workspace_id=workspace_id,
            session_id=bridge_session_id,
        ),
    )


class RuntimeManagerSessionReuseTests(unittest.IsolatedAsyncioTestCase):
    def _load_modules(self):
        service_module = importlib.import_module("runtime.manager.service")
        models_module = importlib.import_module("runtime.manager.models")
        return service_module, models_module

    async def test_reused_provider_session_restarts_worker_with_fresh_runtime_inputs(self) -> None:
        service_module, models_module = self._load_modules()
        start_requests = []

        async def start_session(request):
            start_requests.append(request)
            if len(start_requests) == 1:
                return _worker_session_response(
                    models_module,
                    workspace_id=request.workspace_id,
                    worker_session_id="worker-1",
                    launch_port=3000,
                    bridge_session_id="runtime-session-1",
                )
            return _worker_session_response(
                models_module,
                workspace_id=request.workspace_id,
                worker_session_id="worker-1",
                launch_port=4173,
                bridge_session_id="runtime-session-2",
            )

        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=start_session),
            get_session=mock.AsyncMock(),
            stop_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            first = await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-1",
                    workspace_env={"FIRST": "one"},
                    workspace_env_visibility={"FIRST": True},
                    workspace_mounts=[{"target_path": "/workspace/one"}],
                )
            )
            second = await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-2",
                    provider_session_id=first.provider_session_id,
                    workspace_env={"SECOND": "two"},
                    workspace_env_visibility={"SECOND": False},
                    workspace_mounts=[{"target_path": "/workspace/two"}],
                )
            )

        self.assertEqual(worker_service.start_session.await_count, 2)
        worker_service.get_session.assert_not_awaited()
        reused_request = start_requests[1]
        self.assertEqual(reused_request.provider_session_id, first.provider_session_id)
        self.assertEqual(reused_request.workspace_env, {"SECOND": "two"})
        self.assertEqual(reused_request.workspace_env_visibility, {"SECOND": False})
        self.assertEqual(reused_request.workspace_mounts, [{"target_path": "/workspace/two"}])
        self.assertEqual(second.provider_session_id, first.provider_session_id)
        self.assertEqual(second.launch_port, 4173)
        self.assertIsNotNone(second.bridge_credential)
        self.assertEqual(second.bridge_credential.session_id, "runtime-session-2")

    async def test_reused_provider_session_rejects_cross_workspace_identity_mismatch(self) -> None:
        service_module, models_module = self._load_modules()

        async def start_session(request):
            return _worker_session_response(
                models_module,
                workspace_id=request.workspace_id,
                worker_session_id="worker-1",
                launch_port=3000,
                bridge_session_id="runtime-session-1",
            )

        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=start_session),
            get_session=mock.AsyncMock(),
            stop_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            first = await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-1",
                    workspace_env={"FIRST": "one"},
                )
            )

            with self.assertRaises(HTTPException) as mismatched:
                await manager.start_session(
                    models_module.StartSessionRequest(
                        workspace_id="workspace-2",
                        leased_by_user_id="user-2",
                        provider_session_id=first.provider_session_id,
                        workspace_env={"SECOND": "two"},
                        workspace_env_visibility={"SECOND": False},
                        workspace_mounts=[{"target_path": "/workspace/two"}],
                    )
                )

        self.assertEqual(mismatched.exception.status_code, 409)
        self.assertIn("workspace", str(mismatched.exception.detail).lower())
        self.assertEqual(worker_service.start_session.await_count, 1)
        worker_service.get_session.assert_not_awaited()

        session = manager._sessions[first.provider_session_id]
        self.assertEqual(session.workspace_id, "workspace-1")
        self.assertEqual(session.leased_by_user_id, "user-1")


if __name__ == "__main__":
    unittest.main()
