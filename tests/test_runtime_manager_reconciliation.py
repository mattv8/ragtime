from __future__ import annotations

import asyncio
import importlib
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock


def _worker_response(models_module, worker_session_id: str, *, state: str = "running"):
    now = datetime.now(timezone.utc)
    return models_module.WorkerSessionResponse(
        worker_session_id=worker_session_id,
        workspace_id=f"workspace-{worker_session_id}",
        state=state,
        preview_internal_url="http://runtime/preview",
        launch_framework=None,
        launch_command=None,
        launch_cwd=None,
        launch_port=None,
        runtime_capabilities=None,
        devserver_running=state != "stopped",
        last_error=None,
        runtime_operation_id=None,
        runtime_operation_phase=None,
        runtime_operation_started_at=None,
        runtime_operation_updated_at=None,
        updated_at=now,
    )


class RuntimeManagerReconciliationTests(unittest.IsolatedAsyncioTestCase):
    def _load_modules(self):
        return importlib.import_module("runtime.manager.service"), importlib.import_module("runtime.manager.models")

    def _manager(self, service_module, worker_service):
        patcher = mock.patch.object(service_module, "get_worker_service", return_value=worker_service)
        patcher.start()
        self.addCleanup(patcher.stop)
        manager = service_module.SessionManager()
        manager._max_sessions = 8
        return manager

    def _add_session(self, manager, models_module, provider_id: str, worker_id: str):
        session = manager._create_session(
            provider_id,
            f"workspace-{provider_id}",
            "user-1",
            _worker_response(models_module, worker_id),
            "pty-token",
        )
        manager._sessions[provider_id] = session
        manager._workspace_index[session.workspace_id] = provider_id
        return session

    async def test_reconciliation_caps_concurrent_heartbeats_at_four(self) -> None:
        service_module, models_module = self._load_modules()
        entered = asyncio.Event()
        release = asyncio.Event()
        active = 0
        peak = 0

        async def heartbeat(worker_id):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 4:
                entered.set()
            await release.wait()
            active -= 1
            return _worker_response(models_module, worker_id)

        worker_service = SimpleNamespace(get_session=mock.AsyncMock(side_effect=heartbeat))
        manager = self._manager(service_module, worker_service)
        for number in range(6):
            self._add_session(manager, models_module, f"provider-{number}", f"worker-{number}")

        task = asyncio.create_task(manager._reconcile_active_sessions())
        await entered.wait()
        self.assertEqual(peak, 4)
        self.assertEqual(worker_service.get_session.await_count, 4)
        release.set()
        await task
        self.assertEqual(peak, 4)

    async def test_slow_heartbeat_does_not_block_other_session_progress(self) -> None:
        service_module, models_module = self._load_modules()
        blocked = asyncio.Event()
        complete = asyncio.Event()
        release = asyncio.Event()

        async def heartbeat(worker_id):
            if worker_id == "worker-slow":
                blocked.set()
                await release.wait()
            else:
                complete.set()
            return _worker_response(models_module, worker_id)

        worker_service = SimpleNamespace(get_session=mock.AsyncMock(side_effect=heartbeat))
        manager = self._manager(service_module, worker_service)
        self._add_session(manager, models_module, "provider-slow", "worker-slow")
        fast = self._add_session(manager, models_module, "provider-fast", "worker-fast")

        task = asyncio.create_task(manager._reconcile_active_sessions())
        await blocked.wait()
        await complete.wait()
        self.assertEqual(fast.state, "running")
        release.set()
        await task

    async def test_stale_heartbeat_failure_does_not_mark_replacement_error(self) -> None:
        service_module, models_module = self._load_modules()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def heartbeat(_worker_id):
            entered.set()
            await release.wait()
            raise RuntimeError("worker unavailable")

        worker_service = SimpleNamespace(get_session=mock.AsyncMock(side_effect=heartbeat))
        manager = self._manager(service_module, worker_service)
        original = self._add_session(manager, models_module, "provider-1", "worker-old")

        task = asyncio.create_task(manager._reconcile_active_sessions())
        await entered.wait()
        replacement = self._add_session(manager, models_module, "provider-1", "worker-old")
        self.assertIsNot(replacement, original)
        release.set()
        await task
        self.assertEqual(manager._sessions["provider-1"].worker_session_id, "worker-old")
        self.assertEqual(manager._sessions["provider-1"].state, "running")
        self.assertIsNone(manager._sessions["provider-1"].last_error)

    async def test_heartbeat_and_stop_are_ordered_per_provider(self) -> None:
        service_module, models_module = self._load_modules()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def heartbeat(worker_id):
            entered.set()
            await release.wait()
            return _worker_response(models_module, worker_id)

        worker_service = SimpleNamespace(
            get_session=mock.AsyncMock(side_effect=heartbeat),
            stop_session=mock.AsyncMock(return_value=_worker_response(models_module, "worker-1", state="stopped")),
        )
        manager = self._manager(service_module, worker_service)
        self._add_session(manager, models_module, "provider-1", "worker-1")

        heartbeat_task = asyncio.create_task(manager._reconcile_active_sessions())
        await entered.wait()
        stop_task = asyncio.create_task(manager.stop_session("provider-1"))
        await asyncio.sleep(0)
        worker_service.stop_session.assert_not_awaited()
        release.set()
        await heartbeat_task
        await stop_task
        worker_service.stop_session.assert_awaited_once_with("worker-1")

    async def test_shutdown_cancels_and_drains_reconciliation_children(self) -> None:
        service_module, models_module = self._load_modules()
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def heartbeat(_worker_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        worker_service = SimpleNamespace(get_session=mock.AsyncMock(side_effect=heartbeat))
        manager = self._manager(service_module, worker_service)
        self._add_session(manager, models_module, "provider-1", "worker-1")
        manager._reconcile_task = asyncio.create_task(manager._reconcile_active_sessions())
        await entered.wait()

        await manager.shutdown()
        self.assertTrue(cancelled.is_set())
        self.assertFalse(manager._reconcile_children)


if __name__ == "__main__":
    unittest.main()
