from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.models import UserSpaceRuntimeStatusResponse
from ragtime.userspace.runtime_service import UserSpaceRuntimeService, _ProviderStatusCacheEntry

UTC = timezone.utc


def _session_row(*, provider_session_id: str | None = "provider-1") -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id="session-1",
        workspaceId="workspace-1",
        leasedByUserId="owner-1",
        state="running",
        runtimeProvider="microvm_pool_v1",
        providerSessionId=provider_session_id,
        previewInternalUrl="http://preview",
        launchFramework="vite",
        launchCommand="npm run dev",
        launchCwd="/workspace",
        launchPort=5173,
        createdAt=now,
        updatedAt=now,
        lastHeartbeatAt=now - timedelta(days=1),
        idleExpiresAt=now - timedelta(hours=1),
        ttlExpiresAt=now - timedelta(minutes=1),
        lastError=None,
    )


class RuntimeObservationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceRuntimeService()

    async def test_nonrecovering_status_uses_fresh_provider_observation_without_writes_or_probes(self) -> None:
        row = _session_row()
        provider_status = {
            "state": "running",
            "devserver_running": True,
            "updated_at": "2020-01-01T00:00:00Z",
        }
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=provider_status)) as get_status,
            mock.patch.object(self.service, "_runtime_provider_start_session", mock.AsyncMock()) as start,
            mock.patch.object(self.service, "_runtime_session_update_row", mock.AsyncMock()) as update,
            mock.patch.object(self.service, "_probe_preview_base_url_cached", mock.AsyncMock()) as preview_probe,
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        get_status.assert_awaited_once_with("provider-1", max_age_seconds=0, allow_stale_on_error=False)
        start.assert_not_awaited()
        update.assert_not_awaited()
        preview_probe.assert_not_awaited()
        assert status.observation is not None
        self.assertEqual(status.observation.state, "observed")
        self.assertTrue(status.devserver_running)
        self.assertEqual(status.observation.provider_updated_at, datetime(2020, 1, 1, tzinfo=UTC))
        self.assertEqual(status.recorded_last_heartbeat_at, row.lastHeartbeatAt)

    async def test_nonrecovering_status_preserves_unknown_liveness_and_false_legacy_value(self) -> None:
        row = _session_row()
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value={"state": "running"})),
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "observed")
        self.assertIsNone(status.observation.devserver_running)
        self.assertFalse(status.devserver_running)
        self.assertIsNotNone(status.observation.warning)

    async def test_nonrecovering_status_reports_missing_provider_without_recovery(self) -> None:
        row = _session_row()
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=None)),
            mock.patch.object(self.service, "_runtime_provider_start_session", mock.AsyncMock()) as start,
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "not_found")
        self.assertIsNone(status.observation.devserver_running)
        self.assertFalse(status.devserver_running)
        self.assertIsNotNone(status.observation.warning)
        start.assert_not_awaited()

    async def test_nonrecovering_status_reports_provider_error_as_unavailable(self) -> None:
        row = _session_row()
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(side_effect=RuntimeError("unavailable"))),
            mock.patch.object(self.service, "_runtime_session_update_row", mock.AsyncMock()) as update,
            mock.patch.object(self.service, "_probe_preview_base_url_cached", mock.AsyncMock()) as preview_probe,
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "unavailable")
        self.assertIsNone(status.observation.devserver_running)
        self.assertFalse(status.devserver_running)
        update.assert_not_awaited()
        preview_probe.assert_not_awaited()

    async def test_nonrecovering_status_applies_provider_delta_without_persisting_it(self) -> None:
        row = _session_row()
        provider_status = {
            "state": "error",
            "devserver_running": False,
            "launch_framework": "nextjs",
            "launch_command": "npm run start",
            "launch_cwd": "/new-workspace",
            "launch_port": 3000,
            "last_error": "npm crashed",
            "updated_at": "not-a-datetime",
            "runtime_operation_started_at": "invalid",
            "runtime_operation_updated_at": "also-invalid",
        }
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=provider_status)),
            mock.patch.object(self.service, "_runtime_session_update_row", mock.AsyncMock()) as update,
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        self.assertEqual(status.session_state, "error")
        self.assertEqual(status.launch_framework, "nextjs")
        self.assertEqual(status.launch_command, "npm run start")
        self.assertEqual(status.launch_cwd, "/new-workspace")
        self.assertEqual(status.devserver_port, 3000)
        self.assertEqual(status.last_error, "npm crashed")
        assert status.bridge_status is not None
        self.assertEqual(status.bridge_status.state, "not_running")
        assert status.observation is not None
        self.assertIsNone(status.observation.provider_updated_at)
        self.assertIsNone(status.runtime_operation_started_at)
        self.assertIsNone(status.runtime_operation_updated_at)
        update.assert_not_awaited()

    async def test_nonrecovering_status_reports_empty_provider_response_as_unavailable(self) -> None:
        row = _session_row()
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value={})),
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "unavailable")
        self.assertIsNone(status.observation.devserver_running)
        self.assertFalse(status.devserver_running)

    async def test_nonrecovering_status_reports_non_mapping_provider_response_as_unavailable(self) -> None:
        row = _session_row()
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=object())),
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "unavailable")

    async def test_nonrecovering_status_without_a_session_is_not_applicable(self) -> None:
        with mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=None)):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "not_applicable")
        self.assertFalse(status.observation.devserver_running)
        self.assertFalse(status.devserver_running)

    async def test_nonrecovering_status_with_missing_provider_id_is_not_found(self) -> None:
        row = _session_row(provider_session_id=None)
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=None)),
            mock.patch.object(self.service, "_runtime_session_update_row", mock.AsyncMock()) as update,
            mock.patch.object(self.service, "_probe_preview_base_url_cached", mock.AsyncMock()) as preview_probe,
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized(
                "workspace-1",
                "owner-1",
                allow_recovery=False,
            )

        assert status.observation is not None
        self.assertEqual(status.observation.state, "not_found")
        update.assert_not_awaited()
        preview_probe.assert_not_awaited()

    async def test_zero_age_bypasses_future_dated_provider_cache(self) -> None:
        self.service._provider_status_cache["provider-1"] = _ProviderStatusCacheEntry(
            payload={"devserver_running": False},
            cached_at=datetime.now(UTC) + timedelta(days=1),
        )
        with mock.patch.object(
            self.service,
            "_runtime_manager_request",
            mock.AsyncMock(return_value={"devserver_running": True}),
        ) as request:
            status = await self.service._runtime_provider_get_status("provider-1", max_age_seconds=0)

        request.assert_awaited_once_with("GET", "/sessions/provider-1")
        self.assertEqual(status, {"devserver_running": True})

    async def test_zero_age_does_not_fall_back_to_cache_after_provider_failure(self) -> None:
        await self.service._cache_provider_status("provider-1", {"devserver_running": True})
        with mock.patch.object(
            self.service,
            "_runtime_manager_request",
            mock.AsyncMock(side_effect=HTTPException(status_code=502, detail="provider unavailable")),
        ):
            with self.assertRaises(HTTPException) as error:
                await self.service._runtime_provider_get_status(
                    "provider-1",
                    max_age_seconds=0,
                    allow_stale_on_error=False,
                )

        self.assertEqual(error.exception.status_code, 502)

    async def test_public_status_forwards_allow_recovery(self) -> None:
        expected = UserSpaceRuntimeStatusResponse(workspace_id="workspace-1", session_state="stopped")
        with (
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role",
                mock.AsyncMock(),
            ) as enforce_role,
            mock.patch.object(
                self.service,
                "_get_devserver_status_authorized",
                mock.AsyncMock(return_value=expected),
            ) as status,
        ):
            result = await self.service.get_devserver_status("workspace-1", "owner-1", allow_recovery=False)

        self.assertIs(result, expected)
        enforce_role.assert_awaited_once_with("workspace-1", "owner-1", "viewer")
        status.assert_awaited_once_with("workspace-1", "owner-1", allow_recovery=False)

    async def test_default_status_still_recovers_a_missing_provider_session(self) -> None:
        row = _session_row()
        recovered = {"devserver_running": True}
        with (
            mock.patch.object(self.service, "_get_active_session_row", mock.AsyncMock(side_effect=[row, row])),
            mock.patch.object(self.service, "_runtime_provider_get_status", mock.AsyncMock(return_value=None)),
            mock.patch.object(self.service, "_runtime_provider_start_session", mock.AsyncMock(return_value=recovered)) as start,
            mock.patch.object(self.service, "_runtime_session_update_row", mock.AsyncMock(return_value=row)),
            mock.patch.object(self.service, "_get_runtime_bridge_status_for_session", mock.AsyncMock(return_value=None)),
        ):
            status = await self.service._get_devserver_status_authorized("workspace-1", "owner-1")

        start.assert_awaited_once()
        self.assertIsNone(status.observation)
        self.assertTrue(status.devserver_running)
