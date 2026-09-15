# ruff: noqa: I001

import sys
import types
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


_NOW = datetime(2026, 9, 14, tzinfo=timezone.utc)


class _RuntimeSessionTable:
    def __init__(self, row: SimpleNamespace) -> None:
        self.row = row
        self.updates: list[dict[str, Any]] = []

    async def update(self, *, where: dict[str, str], data: dict[str, Any]) -> SimpleNamespace:
        if where != {"id": "session-1"}:
            raise AssertionError(where)
        self.updates.append(data)
        for key, value in data.items():
            setattr(self.row, key, value)
        return self.row


def _starting_row() -> SimpleNamespace:
    return SimpleNamespace(
        id="session-1",
        workspaceId="workspace-1",
        leasedByUserId="owner-1",
        state="starting",
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


class UserSpaceRuntimeStartingSessionReuseTests(unittest.IsolatedAsyncioTestCase):
    async def _ensure_with_provider_status(
        self,
        provider_status: dict[str, Any] | None,
    ) -> tuple[UserSpaceRuntimeService, _RuntimeSessionTable, Any, Any]:
        row = _starting_row()
        table = _RuntimeSessionTable(row)
        service = UserSpaceRuntimeService()
        service._get_active_session_row = AsyncMock(return_value=row)  # type: ignore[method-assign]
        service._runtime_provider_get_status = AsyncMock(return_value=provider_status)  # type: ignore[method-assign]
        start_session = AsyncMock(
            return_value={
                "state": "starting",
                "provider_session_id": "provider-1",
                "preview_internal_url": "http://runtime/preview",
                "runtime_operation_phase": "queued",
            }
        )
        service._runtime_provider_start_session = start_session  # type: ignore[method-assign]
        db = SimpleNamespace(userspaceruntimesession=table)
        with patch("ragtime.userspace.runtime_service.get_db", AsyncMock(return_value=db)):
            session = await service._ensure_session_row("workspace-1", "viewer-1", auto_start=True)
        return service, table, session, start_session

    async def test_starting_session_with_ready_provider_reuses_session_and_persists_running(self) -> None:
        _, table, session, start_session = await self._ensure_with_provider_status(
            {
                "state": "running",
                "provider_session_id": "provider-1",
                "preview_internal_url": "http://runtime/preview",
                "devserver_running": True,
                "runtime_operation_phase": "ready",
            }
        )

        start_session.assert_not_awaited()
        self.assertEqual(session.state, "running")
        self.assertEqual(table.updates[0]["state"], "running")

    async def test_starting_session_with_queued_provider_does_not_start_duplicate(self) -> None:
        _, table, session, start_session = await self._ensure_with_provider_status(
            {
                "state": "starting",
                "provider_session_id": "provider-1",
                "preview_internal_url": "http://runtime/preview",
                "devserver_running": False,
                "runtime_operation_phase": "queued",
            }
        )

        start_session.assert_not_awaited()
        self.assertEqual(session.state, "starting")
        self.assertEqual(table.updates, [])

    async def test_starting_session_recovers_when_provider_is_failed_stopped_or_missing(self) -> None:
        for provider_status in (
            {
                "state": "running",
                "provider_session_id": "provider-1",
                "preview_internal_url": "http://runtime/preview",
                "devserver_running": False,
                "runtime_operation_phase": "failed",
            },
            {
                "state": "stopped",
                "provider_session_id": "provider-1",
                "preview_internal_url": "http://runtime/preview",
                "devserver_running": False,
                "runtime_operation_phase": "stopped",
            },
            None,
        ):
            with self.subTest(provider_status=provider_status):
                _, _, _, start_session = await self._ensure_with_provider_status(provider_status)
                start_session.assert_awaited_once_with(
                    "workspace-1",
                    "owner-1",
                    "session-1",
                    existing_provider_session_id="provider-1",
                )
