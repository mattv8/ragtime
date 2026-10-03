from __future__ import annotations

import asyncio
import importlib
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

import httpx

from ragtime.core.mount_health import MountHealthEntry


class RuntimeMountHealthApiTests(unittest.IsolatedAsyncioTestCase):
    async def _client(self):
        with mock.patch.dict(os.environ, {"RUNTIME_AUTH_TOKEN": "test-token"}, clear=False):
            auth_module = importlib.import_module("runtime.auth")
            importlib.reload(auth_module)
            api_module = importlib.import_module("runtime.manager.api")
            api_module = importlib.reload(api_module)
            return api_module, httpx.AsyncClient(transport=httpx.ASGITransport(app=api_module.create_app()), base_url="http://test")

    async def test_mount_health_requires_authentication(self) -> None:
        _, client = await self._client()
        async with client:
            response = await client.get("/mounts/health")
        self.assertEqual(response.status_code, 401)

    async def test_get_returns_cached_snapshot_without_probe(self) -> None:
        api_module, client = await self._client()
        checker = mock.Mock()
        checker.last_checked_at.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
        checker.snapshot.return_value = []
        with mock.patch.object(api_module, "_mount_health_checker", checker):
            async with client:
                response = await client.get("/mounts/health", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.json(), {"checked_at": "2026-01-01T00:00:00+00:00", "mounts": []})
        checker.check_async.assert_not_called()

    async def test_post_rechecks_before_returning_snapshot(self) -> None:
        api_module, client = await self._client()
        checker = mock.Mock()
        checker.check_async = mock.AsyncMock()
        checker.last_checked_at.return_value = None
        checker.snapshot.return_value = [
            MountHealthEntry("/share", "nfs", "host:/share", "ok", None, None, 0, False, False, datetime(2026, 1, 1, tzinfo=timezone.utc))
        ]
        with mock.patch.object(api_module, "_mount_health_checker", checker):
            async with client:
                response = await client.post("/mounts/health/recheck", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status_code, 200)
        checker.check_async.assert_awaited_once()
        self.assertEqual(response.json()["mounts"][0]["mount_point"], "/share")

    async def test_lifespan_starts_and_cancels_mount_health_loop(self) -> None:
        api_module, _ = await self._client()
        checker = mock.Mock()
        checker.check_async = mock.AsyncMock()
        manager = mock.Mock()
        manager.startup = mock.AsyncMock()
        manager.shutdown = mock.AsyncMock()
        worker = mock.Mock()
        worker.workspace_root = "/workspace"
        worker.sqlite_history_coordinator.return_value.start = mock.AsyncMock()

        with (
            mock.patch.object(api_module, "SessionManager", return_value=manager),
            mock.patch.object(api_module, "get_worker_service", return_value=worker),
            mock.patch.object(api_module, "MountHealthChecker", return_value=checker),
            mock.patch("runtime.worker.api.shutdown_worker_resources", new=mock.AsyncMock()),
        ):
            app = api_module.create_app()
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0)
                checker.check_async.assert_awaited_once()

        manager.startup.assert_awaited_once()
        manager.shutdown.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
