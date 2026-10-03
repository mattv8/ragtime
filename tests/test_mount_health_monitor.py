import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException

from ragtime.core.mount_health import MountHealthEntry
from ragtime.indexer import routes
from ragtime.indexer.models import MountHealthStatus, MountProblem
from ragtime.indexer.mount_health_monitor import MountHealthMonitor


def _entry(*, reported: bool, state: str = "failed") -> MountHealthEntry:
    return MountHealthEntry(
        mount_point="/mnt/share",
        fstype="cifs",
        source="//host/share",
        state=state,  # type: ignore[arg-type]
        error="No such device" if state != "ok" else None,
        failing_since=datetime.now(timezone.utc) if state != "ok" else None,
        consecutive_failures=2 if state != "ok" else 0,
        reported=reported,
        recovered=False,
        checked_at=datetime.now(timezone.utc),
    )


class _Checker:
    def __init__(self, entries: list[MountHealthEntry]) -> None:
        self.entries = entries

    async def check_async(self) -> list[MountHealthEntry]:
        return self.entries


class MountHealthMonitorTests(unittest.IsolatedAsyncioTestCase):
    def _monitor(self, entries: list[MountHealthEntry], **kwargs: object) -> MountHealthMonitor:
        return MountHealthMonitor(checker=_Checker(entries), **kwargs)  # type: ignore[arg-type]

    async def test_status_is_unknown_before_first_cycle(self) -> None:
        monitor = self._monitor([], runtime_enabled=lambda: False)

        self.assertEqual(monitor.snapshot().status, "unknown")

    async def test_degraded_aggregates_reported_local_and_runtime_entries(self) -> None:
        async def runtime_request(*args: object, **kwargs: object) -> dict[str, object]:
            return {
                "checked_at": "2026-01-01T00:00:00+00:00",
                "mounts": [
                    {
                        "mount_point": "/runtime/share",
                        "fstype": "nfs",
                        "source": "server:/share",
                        "state": "unresponsive",
                        "error": "No response",
                        "failing_since": "2026-01-01T00:00:00+00:00",
                        "reported": True,
                        "checked_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }

        monitor = self._monitor([_entry(reported=True)], runtime_enabled=lambda: True, runtime_request=runtime_request)

        status = await monitor.recheck()

        self.assertEqual(status.status, "degraded")
        self.assertTrue(status.runtime_checked)
        self.assertEqual({problem.container for problem in status.problems}, {"ragtime", "runtime"})

    async def test_non_reported_entries_are_excluded(self) -> None:
        monitor = self._monitor([_entry(reported=False)], runtime_enabled=lambda: False)

        status = await monitor.recheck()

        self.assertEqual(status.status, "ok")
        self.assertEqual(status.problems, [])

    async def test_runtime_disabled_404_and_error_are_not_fatal(self) -> None:
        disabled = self._monitor([], runtime_enabled=lambda: False)
        self.assertFalse((await disabled.recheck()).runtime_checked)

        async def missing(*args: object, **kwargs: object) -> None:
            raise HTTPException(status_code=404)

        async def broken(*args: object, **kwargs: object) -> None:
            raise RuntimeError("offline")

        for request in (missing, broken):
            monitor = self._monitor([], runtime_enabled=lambda: True, runtime_request=request)
            self.assertFalse((await monitor.recheck()).runtime_checked)

    async def test_recheck_uses_runtime_post(self) -> None:
        request = mock.AsyncMock(return_value={"mounts": []})
        monitor = self._monitor([], runtime_enabled=lambda: True, runtime_request=request)

        await monitor.recheck()

        request.assert_awaited_once_with(
            "POST",
            "/mounts/health/recheck",
            timeout_override_seconds=30,
            retry_safe=False,
            surface_error_status=True,
        )

    async def test_background_check_uses_single_runtime_attempt(self) -> None:
        request = mock.AsyncMock(return_value={"checked_at": "2026-01-01T00:00:00+00:00", "mounts": []})
        monitor = self._monitor([], runtime_enabled=lambda: True, runtime_request=request)

        await monitor._check_once(runtime_recheck=False)

        request.assert_awaited_once_with(
            "GET",
            "/mounts/health",
            timeout_override_seconds=15,
            retry_safe=False,
            surface_error_status=True,
        )

    async def test_recheck_joins_an_inflight_cycle(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        request = mock.AsyncMock()

        async def slow_request(*args: object, **kwargs: object) -> dict[str, object]:
            started.set()
            await release.wait()
            return {"checked_at": "2026-01-01T00:00:00+00:00", "mounts": []}

        request.side_effect = slow_request
        monitor = self._monitor([], runtime_enabled=lambda: True, runtime_request=request)
        cycle = asyncio.create_task(monitor._check_once(runtime_recheck=False))
        await started.wait()
        recheck = asyncio.create_task(monitor.recheck())
        await asyncio.sleep(0)
        self.assertFalse(recheck.done())

        release.set()
        await asyncio.wait_for(recheck, timeout=1)
        await cycle
        request.assert_awaited_once()

    async def test_runtime_failure_retains_last_successful_problems(self) -> None:
        responses: list[dict[str, object] | Exception] = [
            {
                "checked_at": "2026-01-01T00:00:00+00:00",
                "mounts": [
                    {
                        "mount_point": "/runtime/share",
                        "fstype": "nfs",
                        "source": "server:/share",
                        "state": "failed",
                        "error": "offline",
                        "failing_since": "2026-01-01T00:00:00+00:00",
                        "reported": True,
                    }
                ],
            },
            RuntimeError("offline"),
        ]

        async def runtime_request(*args: object, **kwargs: object) -> dict[str, object]:
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        monitor = self._monitor([], runtime_enabled=lambda: True, runtime_request=runtime_request)
        self.assertTrue((await monitor.recheck()).runtime_checked)

        status = await monitor.recheck()

        self.assertFalse(status.runtime_checked)
        self.assertEqual(status.status, "degraded")
        self.assertEqual([problem.container for problem in status.problems], ["runtime"])

    async def test_runtime_response_without_checked_at_is_not_checked(self) -> None:
        monitor = self._monitor(
            [],
            runtime_enabled=lambda: True,
            runtime_request=mock.AsyncMock(return_value={"checked_at": None, "mounts": []}),
        )

        self.assertFalse((await monitor.recheck()).runtime_checked)

    async def test_logs_problem_and_recovery_transitions(self) -> None:
        checker = _Checker([_entry(reported=True)])
        monitor = MountHealthMonitor(checker=checker, runtime_enabled=lambda: False)  # type: ignore[arg-type]

        with self.assertLogs("ragtime.indexer.mount_health_monitor", level="INFO") as logs:
            await monitor.recheck()
            checker.entries = [_entry(reported=False, state="ok")]
            await monitor.recheck()

        self.assertIn("Mount health problem", "\n".join(logs.output))
        self.assertIn("Mount health recovered", "\n".join(logs.output))

    async def test_start_and_stop_lifecycle(self) -> None:
        monitor = self._monitor([], runtime_enabled=lambda: False, interval_seconds=3600)

        monitor.start()
        await asyncio.sleep(0)
        self.assertIsNotNone(monitor._task)
        await monitor.stop()

        self.assertIsNone(monitor._task)


class MountHealthRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(routes.router)
        self.status = MountHealthStatus(
            status="degraded",
            checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            runtime_checked=True,
            problems=[
                MountProblem(
                    container="ragtime",
                    mount_point="/mnt/share",
                    fstype="cifs",
                    source="//host/share",
                    state="failed",
                    error=None,
                    failing_since=None,
                )
            ],
        )
        self.snapshot = mock.Mock(return_value=self.status)
        self.recheck = mock.AsyncMock(return_value=self.status)
        self.monitor = SimpleNamespace(snapshot=self.snapshot, recheck=self.recheck)
        self.monitor_patch = mock.patch.object(routes, "mount_health_monitor", self.monitor)
        self.monitor_patch.start()

    async def asyncTearDown(self) -> None:
        self.monitor_patch.stop()
        self.app.dependency_overrides.clear()

    async def test_get_shapes_non_admin_and_admin_responses(self) -> None:
        self.app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id="user", role="user")
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            user_response = await client.get("/indexes/system/mount-health")
        self.assertEqual(user_response.status_code, 200)
        self.assertEqual(user_response.json()["problems"], [])
        self.assertFalse(user_response.json()["runtime_checked"])

        self.app.dependency_overrides[routes.get_current_user] = lambda: SimpleNamespace(id="admin", role="admin")
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            admin_response = await client.get("/indexes/system/mount-health")
        self.assertEqual(admin_response.json()["problems"][0]["mount_point"], "/mnt/share")

    async def test_recheck_requires_admin(self) -> None:
        self.app.dependency_overrides[routes.require_admin] = lambda: (_ for _ in ()).throw(HTTPException(status_code=403))
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            forbidden = await client.post("/indexes/system/mount-health/recheck")
        self.assertEqual(forbidden.status_code, 403)

        self.app.dependency_overrides[routes.require_admin] = lambda: SimpleNamespace(id="admin", role="admin")
        async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
            allowed = await client.post("/indexes/system/mount-health/recheck")
        self.assertEqual(allowed.status_code, 200)
        self.recheck.assert_awaited_once_with()
