import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

from fastapi import HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class ServicePreviewRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_service_preview_upstream_url_single_flights_runtime_start(self) -> None:
        service = UserSpaceRuntimeService()
        session = SimpleNamespace(preview_internal_url="http://runtime-service", workspace_id="workspace-1", provider_session_id=None)
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_ensure(workspace_id: str) -> Any:
            self.assertEqual(workspace_id, "workspace-1")
            started.set()
            await release.wait()
            return session

        service.ensure_service_preview_session = AsyncMock(side_effect=fake_ensure)  # type: ignore[method-assign]

        first = asyncio.create_task(service.build_service_preview_upstream_url("workspace-1", "/api/periods", query="a=1"))
        await started.wait()
        second = asyncio.create_task(service.build_service_preview_upstream_url("workspace-1", "/api/periods", query="a=1"))
        release.set()

        url_one, url_two = await asyncio.gather(first, second)

        self.assertEqual(url_one, "http://runtime-service/api/periods?a=1")
        self.assertEqual(url_two, url_one)
        service.ensure_service_preview_session.assert_awaited_once_with("workspace-1")

    async def test_service_preview_runtime_failures_throttle_after_three_attempts(self) -> None:
        service = UserSpaceRuntimeService()
        service.ensure_shared_preview_session = AsyncMock(side_effect=HTTPException(status_code=503, detail="boot failed"))  # type: ignore[method-assign]

        for _ in range(3):
            with self.assertRaises(HTTPException) as raised:
                await service.build_service_preview_upstream_url("workspace-1", "/api/periods")
            self.assertEqual(raised.exception.status_code, 503)

        with self.assertRaises(HTTPException) as raised:
            await service.build_service_preview_upstream_url("workspace-1", "/api/periods")

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers, {"Retry-After": "300"})

    async def test_service_preview_runtime_success_clears_failure_throttle(self) -> None:
        service = UserSpaceRuntimeService()
        session = SimpleNamespace(preview_internal_url="http://runtime-service", workspace_id="workspace-1", provider_session_id=None)
        with mock.patch.object(
            service,
            "ensure_shared_preview_session",
            AsyncMock(
                side_effect=[
                    HTTPException(status_code=503, detail="boot failed"),
                    session,
                    session,
                ]
            ),
        ):
            with self.assertRaises(HTTPException):
                await service.build_service_preview_upstream_url("workspace-1", "/api/periods")

            recovered = await service.build_service_preview_upstream_url("workspace-1", "/api/periods")
            self.assertEqual(recovered, "http://runtime-service/api/periods")

            again = await service.build_service_preview_upstream_url("workspace-1", "/api/periods")
            self.assertEqual(again, "http://runtime-service/api/periods")

    async def test_workspace_service_state_prunes_failures_and_bounds_unlocked_locks(self) -> None:
        service = UserSpaceRuntimeService()
        old_ts = 100.0
        now_ts = old_ts + 301.0

        service._workspace_service_start_failures = {
            "expired-a": [old_ts],
            "active": [now_ts - 1.0],
            "expired-b": [old_ts + 1.0],
        }

        pruned = service._prune_workspace_service_start_failures("active", now_ts)

        self.assertEqual(pruned, [now_ts - 1.0])
        self.assertEqual(service._workspace_service_start_failures, {"active": [now_ts - 1.0]})

        service._workspace_service_start_lock_max = 2  # type: ignore[attr-defined]
        first = await service._get_workspace_service_start_lock("workspace-1")
        second = await service._get_workspace_service_start_lock("workspace-2")
        third = await service._get_workspace_service_start_lock("workspace-3")

        self.assertIs(second, service._workspace_service_start_locks["workspace-2"])
        self.assertIs(third, service._workspace_service_start_locks["workspace-3"])
        self.assertNotIn("workspace-1", service._workspace_service_start_locks)
        self.assertLessEqual(len(service._workspace_service_start_locks), 2)
        self.assertIsNot(first, second)

    async def test_workspace_service_start_lock_keeps_active_lock_and_prunes_other_unlocked_lock(self) -> None:
        service = UserSpaceRuntimeService()
        service._workspace_service_start_lock_max = 1  # type: ignore[attr-defined]
        first = await service._get_workspace_service_start_lock("workspace-1")

        async with first:
            second = await service._get_workspace_service_start_lock("workspace-2")
            third = await service._get_workspace_service_start_lock("workspace-3")
            self.assertIn("workspace-1", service._workspace_service_start_locks)
            self.assertNotIn("workspace-2", service._workspace_service_start_locks)
            self.assertIn("workspace-3", service._workspace_service_start_locks)
            self.assertIs(first, service._workspace_service_start_locks["workspace-1"])
            self.assertIsNot(second, third)


if __name__ == "__main__":
    unittest.main()
