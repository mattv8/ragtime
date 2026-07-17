import asyncio
import importlib
import unittest
from datetime import timedelta

import httpx
from fastapi import FastAPI


class ServerMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    def _load_module(self):
        try:
            return importlib.import_module("ragtime.core.maintenance")
        except ModuleNotFoundError as exc:
            self.fail(f"maintenance module missing: {exc}")

    async def test_allows_health_and_backup_status_paths(self) -> None:
        maintenance = self._load_module()

        allowed_cases = [
            ("GET", "/health"),
            ("HEAD", "/health"),
            ("GET", "/indexes/server-backups/jobs/job-123"),
            ("GET", "/indexes/server-backups/restore-jobs/job-123"),
        ]
        blocked_cases = [
            ("POST", "/indexes/server-backups/jobs"),
            ("GET", "/indexes/server-backups/jobs/job-123/download"),
            ("GET", "/api/conversations"),
        ]

        for method, path in allowed_cases:
            with self.subTest(method=method, path=path):
                self.assertTrue(maintenance.is_maintenance_bypass_path(method, path))

        for method, path in blocked_cases:
            with self.subTest(method=method, path=path):
                self.assertFalse(maintenance.is_maintenance_bypass_path(method, path))

    async def test_middleware_rejects_mutations_with_retry_after(self) -> None:
        maintenance = self._load_module()
        state = maintenance.ProcessMaintenanceState()

        app = FastAPI()
        app.add_middleware(maintenance.MaintenanceModeMiddleware, state=state)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/mutate")
        async def mutate() -> dict[str, str]:
            return {"status": "mutated"}

        await state.acquire("lease-1", reason="restore", retry_after_seconds=17)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            health_response = await client.get("/health")
            mutate_response = await client.post("/mutate")

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(mutate_response.status_code, 503)
        self.assertEqual(mutate_response.headers.get("Retry-After"), "17")
        self.assertEqual(mutate_response.json()["detail"], "Server maintenance is active. Retry later.")

    async def test_download_bypass_matching_is_precise(self) -> None:
        maintenance = self._load_module()

        self.assertFalse(
            maintenance.is_maintenance_bypass_path(
                "GET",
                "/indexes/server-backups/jobs/job-123/download",
            )
        )
        self.assertTrue(
            maintenance.is_maintenance_bypass_path(
                "GET",
                "/indexes/server-backups/jobs/job-123/download-status",
            )
        )

    async def test_process_state_release_conflict_and_wait_behaviors(self) -> None:
        maintenance = self._load_module()
        state = maintenance.ProcessMaintenanceState(default_ttl_seconds=30)

        waiter = asyncio.create_task(state.wait_until_inactive())
        await state.acquire("lease-1", reason="restore")
        self.assertFalse(waiter.done())

        with self.assertRaises(RuntimeError):
            await state.release("wrong-lease")

        await state.release("lease-1")
        await waiter

    async def test_process_state_expires_lease_after_ttl(self) -> None:
        maintenance = self._load_module()
        state = maintenance.ProcessMaintenanceState(default_ttl_seconds=1)

        acquired = await state.acquire("lease-1", reason="restore")
        expired_status = maintenance.MaintenanceStatus(
            active=True,
            lease_id=acquired.lease_id,
            reason=acquired.reason,
            retry_after_seconds=acquired.retry_after_seconds,
            acquired_at=acquired.acquired_at,
            expires_at=acquired.acquired_at - timedelta(seconds=1),
        )
        state._status = expired_status

        snapshot = await state.snapshot()
        self.assertFalse(snapshot.active)

    async def test_process_state_renew_extends_current_lease(self) -> None:
        maintenance = self._load_module()
        state = maintenance.ProcessMaintenanceState(default_ttl_seconds=10)

        acquired = await state.acquire("lease-1", reason="restore")
        renewed = await state.renew("lease-1", ttl_seconds=30)

        self.assertEqual(renewed.lease_id, "lease-1")
        self.assertGreater(renewed.expires_at, acquired.expires_at)

    async def test_process_state_renew_rejects_inactive_or_wrong_lease(self) -> None:
        maintenance = self._load_module()
        state = maintenance.ProcessMaintenanceState(default_ttl_seconds=10)

        with self.assertRaises(RuntimeError):
            await state.renew("lease-1")

        await state.acquire("lease-1", reason="restore")

        with self.assertRaises(RuntimeError):
            await state.renew("wrong-lease")


if __name__ == "__main__":
    unittest.main()
