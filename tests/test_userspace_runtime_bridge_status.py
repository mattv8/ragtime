from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.models import UserSpaceRuntimeBridgeStatus
from ragtime.userspace.runtime_service import UserSpaceRuntimeService


def _session_row(*, session_id: str, state: str = "running") -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=session_id,
        workspaceId="ws-1",
        leasedByUserId="user-1",
        state=state,
        runtimeProvider="microvm_pool_v1",
        providerSessionId="provider-1",
        previewInternalUrl="http://preview",
        launchFramework=None,
        launchCommand=None,
        launchCwd=None,
        launchPort=5173,
        createdAt=now,
        updatedAt=now,
        lastHeartbeatAt=None,
        idleExpiresAt=None,
        ttlExpiresAt=None,
        lastError=None,
    )


class RuntimeBridgeStatusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceRuntimeService()

    async def test_get_bridge_status_reports_session_mismatch_and_last_success(self) -> None:
        last_success = datetime(2026, 8, 5, 12, 30, tzinfo=UTC)
        current_row = _session_row(session_id="sess-current")
        enforce_role = mock.AsyncMock()

        with (
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", enforce_role),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current_row),
            ),
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(
                    return_value={
                        "bridge_credential": {
                            "bridge_url": "https://ragtime.example/indexes/userspace/runtime-bridge",
                            "token_kind": "userspace_runtime_bridge",
                            "workspace_id": "ws-1",
                            "session_id": "sess-stale",
                            "issued_at": "2026-08-05T12:00:00+00:00",
                            "expires_at": "2099-08-05T16:00:00+00:00",
                        }
                    }
                ),
            ),
            mock.patch.object(
                self.service,
                "_get_latest_runtime_bridge_success_at",
                mock.AsyncMock(return_value=last_success),
            ),
        ):
            status = await self.service.get_runtime_bridge_status("ws-1", "user-1")

        self.assertEqual(status.state, "session_mismatch")
        self.assertEqual(status.bridge_url, "https://ragtime.example/indexes/userspace/runtime-bridge")
        self.assertEqual(status.token_session_id, "sess-stale")
        self.assertEqual(status.current_session_id, "sess-current")
        self.assertEqual(status.last_success_at, last_success)
        enforce_role.assert_awaited_once_with("ws-1", "user-1", "viewer")

    async def test_get_devserver_status_includes_bridge_status(self) -> None:
        current_row = _session_row(session_id="sess-current")
        bridge_status = UserSpaceRuntimeBridgeStatus(
            state="healthy",
            bridge_url="https://ragtime.example/indexes/userspace/runtime-bridge",
            token_session_id="sess-current",
            current_session_id="sess-current",
        )

        with (
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current_row),
            ),
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"devserver_running": True}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(return_value=bridge_status),
            ),
        ):
            status = await self.service._get_devserver_status_authorized("ws-1", "user-1")

        self.assertEqual(status.bridge_status, bridge_status)

    async def test_latest_runtime_bridge_success_skips_failed_newest_row(self) -> None:
        successful_at = datetime(2026, 8, 5, 11, 0, tzinfo=UTC)
        rows = [
            SimpleNamespace(createdAt=datetime(2026, 8, 5, 12, 0, tzinfo=UTC), eventPayload={"error": "boom"}),
            SimpleNamespace(createdAt=successful_at, eventPayload={"error": None}),
        ]
        db = SimpleNamespace(userspaceruntimeauditevent=SimpleNamespace(find_many=mock.AsyncMock(return_value=rows)))

        with mock.patch("ragtime.userspace.runtime_service.get_db", mock.AsyncMock(return_value=db)):
            last_success = await self.service._get_latest_runtime_bridge_success_at("ws-1", "sess-1")

        self.assertEqual(last_success, successful_at)
        db.userspaceruntimeauditevent.find_many.assert_awaited_once()
        self.assertEqual(db.userspaceruntimeauditevent.find_many.await_args.kwargs["take"], 10)

    async def test_get_runtime_bridge_status_maps_provider_failure_to_unavailable(self) -> None:
        current_row = _session_row(session_id="sess-current")

        with (
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=current_row),
            ),
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(side_effect=HTTPException(status_code=502, detail="upstream unavailable")),
            ),
        ):
            status = await self.service.get_runtime_bridge_status("ws-1", "user-1")

        self.assertEqual(status.state, "unavailable")
        self.assertEqual(status.current_session_id, "sess-current")

    async def test_runtime_bridge_status_derives_not_running(self) -> None:
        status = await self.service._get_runtime_bridge_status_for_session(None, None)
        self.assertEqual(status.state, "not_running")

    async def test_runtime_bridge_status_derives_missing(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        status = await self.service._get_runtime_bridge_status_for_session(session, {"devserver_running": True})
        self.assertEqual(status.state, "missing")

    async def test_runtime_bridge_status_derives_expired(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        with mock.patch.object(self.service, "_get_latest_runtime_bridge_success_at", mock.AsyncMock(return_value=None)):
            status = await self.service._get_runtime_bridge_status_for_session(
                session,
                {
                    "bridge_credential": {
                        "bridge_url": "https://ragtime.example/indexes/userspace/runtime-bridge",
                        "token_kind": "userspace_runtime_bridge",
                        "workspace_id": "ws-1",
                        "session_id": "sess-1",
                        "issued_at": "2026-08-05T12:00:00+00:00",
                        "expires_at": "2000-08-05T16:00:00+00:00",
                    }
                },
            )
        self.assertEqual(status.state, "expired")

    async def test_runtime_bridge_status_derives_invalid(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        with mock.patch.object(self.service, "_get_latest_runtime_bridge_success_at", mock.AsyncMock(return_value=None)):
            status = await self.service._get_runtime_bridge_status_for_session(
                session,
                {
                    "bridge_credential": {
                        "bridge_url": "",
                        "token_kind": "wrong-kind",
                        "workspace_id": "ws-2",
                        "session_id": "sess-1",
                    }
                },
            )
        self.assertEqual(status.state, "invalid")

    async def test_refresh_runtime_bridge_credentials_waits_then_fetches_status(self) -> None:
        expected = UserSpaceRuntimeBridgeStatus(state="healthy")
        enforce_role = mock.AsyncMock()

        with (
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", enforce_role),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "get_runtime_bridge_status",
                mock.AsyncMock(return_value=expected),
            ) as get_status,
        ):
            result = await self.service.refresh_runtime_bridge_credentials("ws-1", "user-1")

        self.assertIs(result, expected)
        enforce_role.assert_awaited_once_with("ws-1", "user-1", "editor")
        restart_wait.assert_awaited_once_with("ws-1", timeout_seconds=60.0)
        get_status.assert_awaited_once_with("ws-1", "user-1")
