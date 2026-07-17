from __future__ import annotations

import asyncio
import importlib
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import HTTPException


def _worker_session_response(models_module, *, workspace_id: str, state: str = "running"):
    now = datetime.now(timezone.utc)
    return models_module.WorkerSessionResponse(
        worker_session_id=f"worker-{workspace_id}",
        workspace_id=workspace_id,
        state=state,
        preview_internal_url=f"http://runtime/{workspace_id}",
        launch_framework=None,
        launch_command=None,
        launch_cwd=None,
        launch_port=None,
        runtime_capabilities=None,
        devserver_running=(state != "stopped"),
        last_error=None,
        runtime_operation_id=None,
        runtime_operation_phase=None,
        runtime_operation_started_at=None,
        runtime_operation_updated_at=None,
        updated_at=now,
    )


class RuntimeManagerMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    def _load_modules(self):
        service_module = importlib.import_module("runtime.manager.service")
        models_module = importlib.import_module("runtime.manager.models")
        api_module = importlib.import_module("runtime.manager.api")
        return service_module, models_module, api_module

    async def test_start_session_cleans_up_if_maintenance_lease_activates_mid_launch(self) -> None:
        service_module, models_module, _ = self._load_modules()
        start_entered = asyncio.Event()
        allow_start_finish = asyncio.Event()

        async def delayed_start(request):
            start_entered.set()
            await allow_start_finish.wait()
            return _worker_session_response(models_module, workspace_id=request.workspace_id)

        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=delayed_start),
            stop_session=mock.AsyncMock(
                side_effect=lambda worker_session_id: _worker_session_response(
                    models_module, workspace_id=worker_session_id.replace("worker-", ""), state="stopped"
                )
            ),
            get_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            start_task = asyncio.create_task(
                manager.start_session(
                    models_module.StartSessionRequest(
                        workspace_id="workspace-1",
                        leased_by_user_id="user-1",
                    )
                )
            )
            await start_entered.wait()

            await manager.acquire_maintenance_lease(
                models_module.RuntimeManagerMaintenanceAcquireRequest(
                    lease_id="lease-1",
                    reason="restore",
                )
            )
            allow_start_finish.set()

            with self.assertRaises(HTTPException) as blocked:
                await start_task

        self.assertEqual(blocked.exception.status_code, 503)
        worker_service.stop_session.assert_awaited_once()

    async def test_acquire_reports_conflict_when_active_sessions_remain(self) -> None:
        service_module, models_module, _ = self._load_modules()
        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=lambda request: _worker_session_response(models_module, workspace_id=request.workspace_id)),
            stop_session=mock.AsyncMock(side_effect=TimeoutError),
            get_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-1",
                )
            )

            with self.assertRaises(HTTPException) as conflicting:
                await manager.acquire_maintenance_lease(
                    models_module.RuntimeManagerMaintenanceAcquireRequest(
                        lease_id="lease-1",
                        reason="restore",
                    )
                )

            with self.assertRaises(HTTPException) as conflicting_again:
                await manager.acquire_maintenance_lease(
                    models_module.RuntimeManagerMaintenanceAcquireRequest(
                        lease_id="lease-1",
                        reason="restore",
                    )
                )

        self.assertEqual(conflicting.exception.status_code, 409)
        self.assertEqual(conflicting_again.exception.status_code, 409)

    async def test_maintenance_lease_expires_and_allows_launches_again(self) -> None:
        service_module, models_module, _ = self._load_modules()
        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=lambda request: _worker_session_response(models_module, workspace_id=request.workspace_id)),
            stop_session=mock.AsyncMock(),
            get_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            manager._maintenance_lease_ttl_seconds = 1
            acquired = await manager.acquire_maintenance_lease(
                models_module.RuntimeManagerMaintenanceAcquireRequest(
                    lease_id="lease-1",
                    reason="restore",
                )
            )
            manager._maintenance_lease = acquired.model_copy(update={"expires_at": acquired.acquired_at})

            response = await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-1",
                )
            )

        self.assertEqual(response.workspace_id, "workspace-1")

    async def test_runtime_manager_renew_extends_current_lease(self) -> None:
        service_module, models_module, _ = self._load_modules()
        with mock.patch.object(service_module, "get_worker_service", return_value=SimpleNamespace()):
            manager = service_module.SessionManager()

        acquired = await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-1",
                reason="restore",
            )
        )
        renewed = await manager.renew_maintenance_lease(
            "lease-1",
            models_module.RuntimeManagerMaintenanceRenewRequest(ttl_seconds=30),
        )

        self.assertEqual(renewed.lease_id, "lease-1")
        self.assertEqual(renewed.reason, acquired.reason)
        self.assertGreater(renewed.expires_at, acquired.expires_at)

    async def test_runtime_manager_renew_rejects_inactive_or_wrong_lease(self) -> None:
        service_module, models_module, _ = self._load_modules()
        with mock.patch.object(service_module, "get_worker_service", return_value=SimpleNamespace()):
            manager = service_module.SessionManager()

        with self.assertRaises(HTTPException) as inactive:
            await manager.renew_maintenance_lease(
                "lease-1",
                models_module.RuntimeManagerMaintenanceRenewRequest(ttl_seconds=30),
            )

        await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-1",
                reason="restore",
            )
        )

        with self.assertRaises(HTTPException) as wrong_owner:
            await manager.renew_maintenance_lease(
                "wrong-lease",
                models_module.RuntimeManagerMaintenanceRenewRequest(ttl_seconds=30),
            )

        self.assertEqual(inactive.exception.status_code, 404)
        self.assertEqual(wrong_owner.exception.status_code, 409)

    async def test_maintenance_renew_endpoint_keeps_long_running_lease_active(self) -> None:
        env = {"RUNTIME_AUTH_TOKEN": "runtime-test-token"}
        with mock.patch.dict(os.environ, env, clear=True):
            runtime_auth = importlib.import_module("runtime.auth")
            runtime_auth = importlib.reload(runtime_auth)
            api_module = importlib.import_module("runtime.manager.api")
            api_module = importlib.reload(api_module)
            models_module = importlib.import_module("runtime.manager.models")

            renewed_response = models_module.RuntimeManagerMaintenanceLeaseResponse(
                active=True,
                lease_id="lease-1",
                reason="restore",
                retry_after_seconds=60,
                acquired_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc),
                active_session_count=0,
            )
            fake_manager = SimpleNamespace(
                startup=mock.AsyncMock(),
                shutdown=mock.AsyncMock(),
                acquire_maintenance_lease=mock.AsyncMock(return_value=renewed_response),
                renew_maintenance_lease=mock.AsyncMock(return_value=renewed_response),
                release_maintenance_lease=mock.AsyncMock(return_value=models_module.RuntimeManagerMaintenanceLeaseResponse.inactive()),
            )

            with mock.patch.object(api_module, "SessionManager", return_value=fake_manager):
                app = api_module.create_app()

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                renewed = await client.put(
                    "/maintenance/lease/lease-1",
                    json={"ttl_seconds": 120},
                    headers={"Authorization": "Bearer runtime-test-token"},
                )

        self.assertEqual(renewed.status_code, 200)
        fake_manager.renew_maintenance_lease.assert_awaited_once()

    async def test_acquire_lease_stops_sessions_and_blocks_new_launches(self) -> None:
        service_module, models_module, _ = self._load_modules()
        worker_service = SimpleNamespace(
            start_session=mock.AsyncMock(side_effect=lambda request: _worker_session_response(models_module, workspace_id=request.workspace_id)),
            stop_session=mock.AsyncMock(
                side_effect=lambda worker_session_id: _worker_session_response(
                    models_module, workspace_id=worker_session_id.replace("worker-", ""), state="stopped"
                )
            ),
            get_session=mock.AsyncMock(),
        )

        with mock.patch.object(service_module, "get_worker_service", return_value=worker_service):
            manager = service_module.SessionManager()
            await manager.start_session(
                models_module.StartSessionRequest(
                    workspace_id="workspace-1",
                    leased_by_user_id="user-1",
                )
            )

            acquired = await manager.acquire_maintenance_lease(
                models_module.RuntimeManagerMaintenanceAcquireRequest(
                    lease_id="lease-1",
                    reason="restore",
                    retry_after_seconds=23,
                )
            )

            self.assertTrue(acquired.active)
            self.assertEqual(acquired.lease_id, "lease-1")
            self.assertEqual(acquired.active_session_count, 0)
            worker_service.stop_session.assert_awaited_once()

            with self.assertRaises(HTTPException) as blocked:
                await manager.start_session(
                    models_module.StartSessionRequest(
                        workspace_id="workspace-2",
                        leased_by_user_id="user-2",
                    )
                )

        self.assertEqual(blocked.exception.status_code, 503)
        self.assertEqual(blocked.exception.detail, "Runtime maintenance lease is active. Retry after maintenance completes.")

    async def test_acquire_is_idempotent_for_same_lease_and_rejects_conflicts(self) -> None:
        service_module, models_module, _ = self._load_modules()
        with mock.patch.object(service_module, "get_worker_service", return_value=SimpleNamespace()):
            manager = service_module.SessionManager()

        first = await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-1",
                reason="restore",
                retry_after_seconds=31,
            )
        )
        second = await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-1",
                reason="restore",
                retry_after_seconds=31,
            )
        )

        self.assertEqual(first.lease_id, second.lease_id)

        with self.assertRaises(HTTPException) as conflicting:
            await manager.acquire_maintenance_lease(
                models_module.RuntimeManagerMaintenanceAcquireRequest(
                    lease_id="lease-2",
                    reason="other",
                    retry_after_seconds=31,
                )
            )

        self.assertEqual(conflicting.exception.status_code, 409)

    async def test_release_is_idempotent_for_same_lease_and_rejects_other_holders(self) -> None:
        service_module, models_module, _ = self._load_modules()
        with mock.patch.object(service_module, "get_worker_service", return_value=SimpleNamespace()):
            manager = service_module.SessionManager()

        await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-1",
                reason="restore",
            )
        )

        released = await manager.release_maintenance_lease("lease-1")
        self.assertFalse(released.active)
        released_again = await manager.release_maintenance_lease("lease-1")
        self.assertFalse(released_again.active)

        await manager.acquire_maintenance_lease(
            models_module.RuntimeManagerMaintenanceAcquireRequest(
                lease_id="lease-2",
                reason="restore",
            )
        )
        with self.assertRaises(HTTPException) as conflicting:
            await manager.release_maintenance_lease("wrong-lease")

        self.assertEqual(conflicting.exception.status_code, 409)

    async def test_maintenance_endpoints_require_auth_and_delegate_to_manager(self) -> None:
        env = {"RUNTIME_AUTH_TOKEN": "runtime-test-token"}
        with mock.patch.dict(os.environ, env, clear=True):
            runtime_auth = importlib.import_module("runtime.auth")
            runtime_auth = importlib.reload(runtime_auth)
            api_module = importlib.import_module("runtime.manager.api")
            api_module = importlib.reload(api_module)
            models_module = importlib.import_module("runtime.manager.models")

            response_model = models_module.RuntimeManagerMaintenanceLeaseResponse(
                active=True,
                lease_id="lease-1",
                reason="restore",
                retry_after_seconds=29,
                acquired_at=datetime.now(timezone.utc),
                active_session_count=0,
            )
            fake_manager = SimpleNamespace(
                startup=mock.AsyncMock(),
                shutdown=mock.AsyncMock(),
                acquire_maintenance_lease=mock.AsyncMock(return_value=response_model),
                release_maintenance_lease=mock.AsyncMock(return_value=models_module.RuntimeManagerMaintenanceLeaseResponse.inactive()),
            )

            with mock.patch.object(api_module, "SessionManager", return_value=fake_manager):
                app = api_module.create_app()

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                unauthorized = await client.post("/maintenance/lease", json={"lease_id": "lease-1", "reason": "restore"})
                authorized = await client.post(
                    "/maintenance/lease",
                    json={"lease_id": "lease-1", "reason": "restore"},
                    headers={"Authorization": "Bearer runtime-test-token"},
                )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        fake_manager.acquire_maintenance_lease.assert_awaited_once()

    async def test_maintenance_endpoints_reject_wrong_token(self) -> None:
        env = {"RUNTIME_AUTH_TOKEN": "runtime-test-token"}
        with mock.patch.dict(os.environ, env, clear=True):
            runtime_auth = importlib.import_module("runtime.auth")
            runtime_auth = importlib.reload(runtime_auth)
            api_module = importlib.import_module("runtime.manager.api")
            api_module = importlib.reload(api_module)
            models_module = importlib.import_module("runtime.manager.models")
            fake_manager = SimpleNamespace(
                startup=mock.AsyncMock(),
                shutdown=mock.AsyncMock(),
                acquire_maintenance_lease=mock.AsyncMock(return_value=models_module.RuntimeManagerMaintenanceLeaseResponse.inactive()),
                release_maintenance_lease=mock.AsyncMock(return_value=models_module.RuntimeManagerMaintenanceLeaseResponse.inactive()),
            )

            with mock.patch.object(api_module, "SessionManager", return_value=fake_manager):
                app = api_module.create_app()

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
                wrong = await client.post(
                    "/maintenance/lease",
                    json={"lease_id": "lease-1", "reason": "restore"},
                    headers={"Authorization": "Bearer wrong-token"},
                )

        self.assertEqual(wrong.status_code, 403)
        fake_manager.acquire_maintenance_lease.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
