# ruff: noqa: I001

import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import _ProviderStatusCacheEntry, UserSpaceRuntimeService, utc_now


_NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)
_STATUS = {
    "state": "running",
    "provider_session_id": "provider-1",
    "preview_internal_url": "http://runtime/preview",
    "devserver_running": True,
    "runtime_operation_id": "operation-1",
    "runtime_operation_phase": "ready",
    "runtime_operation_started_at": "2026-09-22T00:00:00Z",
    "runtime_operation_updated_at": "2026-09-22T00:00:01Z",
}


def _row(*, state: str = "running") -> SimpleNamespace:
    return SimpleNamespace(
        id="session-1",
        workspaceId="workspace-1",
        leasedByUserId="owner-1",
        state=state,
        runtimeProvider="microvm_pool_v1",
        providerSessionId="provider-1",
        previewInternalUrl="http://runtime/preview",
        launchFramework="vite",
        launchCommand="npm run dev",
        launchCwd=".",
        launchPort=5173,
        createdAt=_NOW,
        updatedAt=_NOW,
        lastHeartbeatAt=_NOW,
        idleExpiresAt=None,
        ttlExpiresAt=None,
        lastError=None,
    )


class UserSpaceRuntimeStartPerformanceTests(unittest.IsolatedAsyncioTestCase):
    async def _start_active_session(
        self,
        service: UserSpaceRuntimeService,
        row: SimpleNamespace,
    ) -> Any:
        db = SimpleNamespace(userspaceruntimesession=SimpleNamespace())
        with (
            patch("ragtime.userspace.runtime_service.get_db", new=AsyncMock(return_value=db)),
            patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", new=AsyncMock()),
            patch(
                "ragtime.userspace.preview_host.invalidate_preview_sessions_for_workspace",
                new=AsyncMock(),
            ),
        ):
            return await service.start_runtime_session("workspace-1", "editor-1")

    async def test_start_uses_accepted_cache_miss_observation_then_next_poll_is_fresh(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row())  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        events: list[str] = []

        async def manager_request(*args: Any, **kwargs: Any) -> dict[str, Any]:
            events.append("provider-get")
            return dict(_STATUS)

        actual_invalidate = service.invalidate_preview_session_cache

        async def observe_invalidation(workspace_id: str) -> None:
            events.append("invalidate")
            await actual_invalidate(workspace_id)

        service._runtime_manager_request = AsyncMock(side_effect=manager_request)  # type: ignore[method-assign]
        service.invalidate_preview_session_cache = observe_invalidation  # type: ignore[method-assign]

        result = await self._start_active_session(service, _row())

        self.assertEqual(events, ["provider-get", "invalidate"])
        self.assertEqual(result.runtime_operation_id, "operation-1")
        self.assertEqual(result.runtime_operation_phase, "ready")
        self.assertEqual(result.runtime_operation_started_at, _NOW)
        self.assertEqual(result.runtime_operation_updated_at, _NOW.replace(second=1))
        self.assertIsNone(await service._get_cached_provider_status("provider-1", max_age_seconds=8))
        service._audit.assert_awaited_once_with(  # type: ignore[attr-defined]
            "workspace-1",
            "session_start",
            user_id="editor-1",
            session_id="session-1",
            payload={"provider_session_id": "provider-1"},
        )

        await service._runtime_provider_get_status("provider-1", max_age_seconds=8)

        self.assertEqual(service._runtime_manager_request.await_count, 2)

    async def test_start_uses_accepted_cache_hit_observation_then_invalidates_it(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row())  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        await service._cache_provider_status("provider-1", _STATUS)
        service._runtime_manager_request = AsyncMock(return_value=dict(_STATUS))  # type: ignore[method-assign]

        result = await self._start_active_session(service, _row())

        self.assertEqual(result.runtime_operation_id, "operation-1")
        service._runtime_manager_request.assert_not_awaited()
        self.assertIsNone(await service._get_cached_provider_status("provider-1", max_age_seconds=8))

        await service._runtime_provider_get_status("provider-1", max_age_seconds=8)

        service._runtime_manager_request.assert_awaited_once()

    async def test_starting_session_reuses_accepted_provider_observation(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row(state="starting"))  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        service._runtime_provider_start_session = AsyncMock()  # type: ignore[method-assign]
        queued = {**_STATUS, "state": "starting", "devserver_running": False, "runtime_operation_phase": "queued"}
        service._runtime_manager_request = AsyncMock(return_value=queued)  # type: ignore[method-assign]

        result = await self._start_active_session(service, _row(state="starting"))

        self.assertEqual(result.state, "starting")
        self.assertEqual(result.runtime_operation_phase, "queued")
        service._runtime_provider_start_session.assert_not_awaited()
        service._runtime_manager_request.assert_awaited_once()

    async def test_missing_provider_is_recreated_and_its_operation_is_returned(self) -> None:
        row = _row()
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=row)  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        service._runtime_manager_request = AsyncMock(side_effect=HTTPException(status_code=404, detail="(404) missing"))  # type: ignore[method-assign]
        recreated = {**_STATUS, "runtime_operation_id": "operation-recreated", "runtime_operation_phase": "queued"}
        service._runtime_provider_start_session = AsyncMock(return_value=recreated)  # type: ignore[method-assign]

        async def update_row(model: Any, row_id: str, data: dict[str, Any]) -> SimpleNamespace:
            self.assertEqual(row_id, "session-1")
            for key, value in data.items():
                setattr(row, key, value)
            return row

        service._runtime_session_update_row = update_row  # type: ignore[method-assign]

        result = await self._start_active_session(service, row)

        service._runtime_provider_start_session.assert_awaited_once_with("workspace-1", "owner-1", "session-1", existing_provider_session_id="provider-1")
        self.assertEqual(result.runtime_operation_id, "operation-recreated")
        self.assertEqual(result.runtime_operation_phase, "queued")

    async def test_start_returns_stale_on_error_observation_before_invalidation(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row())  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        await service._cache_provider_status("provider-1", _STATUS)
        service._provider_status_cache["provider-1"] = _ProviderStatusCacheEntry(
            payload=dict(_STATUS),
            cached_at=utc_now() - timedelta(seconds=3),
        )
        service._runtime_manager_request = AsyncMock(side_effect=HTTPException(status_code=503, detail="manager unavailable"))  # type: ignore[method-assign]

        result = await self._start_active_session(service, _row())

        self.assertEqual(result.runtime_operation_id, "operation-1")
        self.assertEqual(result.runtime_operation_phase, "ready")
        self.assertEqual(service._runtime_manager_request.await_count, 2)
        self.assertIsNone(await service._get_cached_provider_status("provider-1", max_age_seconds=8))

    async def test_start_propagates_ensure_failure_without_accepted_observation(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row())  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        service._runtime_manager_request = AsyncMock(side_effect=HTTPException(status_code=503, detail="manager unavailable"))  # type: ignore[method-assign]

        with self.assertRaises(HTTPException) as error:
            await self._start_active_session(service, _row())

        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(service._runtime_manager_request.await_count, 1)
        service._audit.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_start_accepts_bounded_observation_when_manager_fails_afterward(self) -> None:
        service = UserSpaceRuntimeService()
        service._require_runtime_manager = lambda: None  # type: ignore[method-assign]
        service._get_active_session_row = AsyncMock(return_value=_row())  # type: ignore[method-assign]
        service._audit = AsyncMock()  # type: ignore[method-assign]
        service._runtime_manager_request = AsyncMock(side_effect=[dict(_STATUS), HTTPException(status_code=503, detail="manager unavailable")])  # type: ignore[method-assign]

        result = await self._start_active_session(service, _row())

        self.assertEqual(result.runtime_operation_phase, "ready")
        self.assertEqual(service._runtime_manager_request.await_count, 1)
        with self.assertRaises(HTTPException) as poll_error:
            await service._runtime_provider_get_status("provider-1", max_age_seconds=8)
        self.assertEqual(poll_error.exception.status_code, 503)
        self.assertEqual(service._runtime_manager_request.await_count, 2)
