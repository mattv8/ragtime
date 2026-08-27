from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Literal
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.models import UserSpaceRuntimeBridgeStatus
from ragtime.userspace.runtime_service import _RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS, UserSpaceRuntimeService

UTC = timezone.utc


def _session_row(
    *,
    session_id: str,
    state: str = "running",
    workspace_id: str = "ws-1",
    provider_session_id: str = "provider-1",
) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=session_id,
        workspaceId=workspace_id,
        leasedByUserId="user-1",
        state=state,
        runtimeProvider="microvm_pool_v1",
        providerSessionId=provider_session_id,
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


def _bridge_status(
    *,
    state: Literal[
        "healthy",
        "not_running",
        "missing",
        "expired",
        "invalid",
        "session_mismatch",
        "unavailable",
    ] = "healthy",
    expires_at: datetime | None = None,
    detail: str | None = None,
    session_id: str = "sess-1",
) -> UserSpaceRuntimeBridgeStatus:
    return UserSpaceRuntimeBridgeStatus(
        state=state,
        current_session_id=session_id,
        token_session_id=session_id,
        bridge_url="https://ragtime.example/indexes/userspace/runtime-bridge",
        expires_at=expires_at,
        detail=detail,
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
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        enforce_role = mock.AsyncMock()

        with (
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", enforce_role),
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ),
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ) as ensure_ready,
            mock.patch.object(
                self.service,
                "get_runtime_bridge_status",
                mock.AsyncMock(return_value=expected),
            ) as get_status,
        ):
            result = await self.service.refresh_runtime_bridge_credentials("ws-1", "user-1")

        self.assertIs(result, expected)
        enforce_role.assert_awaited_once_with("ws-1", "user-1", "editor")
        ensure_ready.assert_awaited_once_with(session)
        get_status.assert_awaited_once_with("ws-1", "user-1")

    async def test_refresh_runtime_bridge_credentials_routes_through_preview_readiness_flow(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        expected = UserSpaceRuntimeBridgeStatus(state="healthy")
        enforce_role = mock.AsyncMock()

        with (
            mock.patch("ragtime.userspace.runtime_service.userspace_service.enforce_workspace_role", enforce_role),
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ) as ensure_session,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ) as ensure_ready,
            mock.patch.object(
                self.service,
                "get_runtime_bridge_status",
                mock.AsyncMock(return_value=expected),
            ) as get_status,
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
        ):
            result = await self.service.refresh_runtime_bridge_credentials("ws-1", "user-1")

        self.assertIs(result, expected)
        enforce_role.assert_awaited_once_with("ws-1", "user-1", "editor")
        ensure_session.assert_awaited_once_with("ws-1", "user-1")
        ensure_ready.assert_awaited_once_with(session)
        get_status.assert_awaited_once_with("ws-1", "user-1")
        restart_wait.assert_not_awaited()

    async def test_preview_bridge_readiness_skips_restart_when_healthy_and_not_near_expiry(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        status = _bridge_status(expires_at=datetime.now(UTC) + timedelta(seconds=900))

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ) as get_status,
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(return_value=status),
            ) as get_bridge_status,
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
        ):
            await self.service._ensure_workspace_preview_bridge_ready(session)

        get_status.assert_awaited_once_with(
            session.provider_session_id,
            max_age_seconds=_RUNTIME_PROVIDER_STATUS_CACHE_TTL_SECONDS,
            allow_stale_on_error=False,
        )
        get_bridge_status.assert_awaited_once_with(
            session,
            {"provider": "status"},
            include_last_success=False,
        )
        restart_wait.assert_not_awaited()

    async def test_preview_bridge_readiness_propagates_provider_status_error_without_restart(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(side_effect=HTTPException(status_code=502, detail="provider down")),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service._ensure_workspace_preview_bridge_ready(session)

        self.assertEqual(exc_info.exception.status_code, 502)
        self.assertEqual(exc_info.exception.detail, "provider down")
        restart_wait.assert_not_awaited()

    async def test_preview_bridge_readiness_restarts_when_expired_before_preview_build(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        call_order: list[str] = []

        async def ensure_ready(arg_session):
            self.assertIs(arg_session, session)
            call_order.append("ready")

        async def describe_preview(*args, **kwargs):
            call_order.append("describe")
            return ("https://preview", None)

        describe_preview_mock = mock.AsyncMock(side_effect=describe_preview)

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ),
            mock.patch.object(
                self.service,
                "_describe_preview_launch",
                describe_preview_mock,
            ),
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(side_effect=ensure_ready),
            ) as ensure_ready,
        ):
            preview_origin, expires_at, preview_warning = await self.service.describe_workspace_preview_launch(
                "ws-1",
                "user-1",
                control_plane_origin="https://ragtime.dev.visnovsky.us",
            )

        ensure_ready.assert_awaited_once_with(session)
        describe_preview_mock.assert_awaited_once()
        self.assertEqual(preview_origin, "https://preview")
        self.assertIsNone(preview_warning)
        self.assertIsNotNone(expires_at)
        self.assertEqual(call_order, ["ready", "describe"])

    async def test_describe_workspace_preview_launch_stops_before_build_when_readiness_fails(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ),
            mock.patch.object(
                self.service,
                "_describe_preview_launch",
                mock.AsyncMock(),
            ) as describe_preview,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(side_effect=HTTPException(status_code=502, detail="bridge not ready")),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service.describe_workspace_preview_launch(
                    "ws-1",
                    "user-1",
                    control_plane_origin="https://ragtime.dev.visnovsky.us",
                )

        self.assertEqual(exc_info.exception.detail, "bridge not ready")
        describe_preview.assert_not_awaited()

    async def test_preview_bridge_readiness_restarts_when_near_expiry(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        initial_status = _bridge_status(expires_at=datetime.now(UTC) + timedelta(seconds=300))
        refreshed_status = _bridge_status(expires_at=datetime.now(UTC) + timedelta(seconds=1200))

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(side_effect=[{"version": 1}, {"version": 2}, {"version": 3}]),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=[initial_status, initial_status, refreshed_status]),
            ) as get_bridge_status,
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
        ):
            await self.service._ensure_workspace_preview_bridge_ready(session)

        restart_wait.assert_awaited_once_with("ws-1", timeout_seconds=60.0)
        self.assertEqual(get_bridge_status.await_count, 3)

    async def test_preview_bridge_readiness_raises_when_refresh_does_not_restore_health(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        initial_status = _bridge_status(state="expired", detail="expired bridge token")
        refreshed_status = _bridge_status(state="missing", detail="metadata unavailable")

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(side_effect=[{"version": 1}, {"version": 2}, {"version": 3}]),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=[initial_status, initial_status, refreshed_status]),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service._ensure_workspace_preview_bridge_ready(session)

        restart_wait.assert_awaited_once_with("ws-1", timeout_seconds=60.0)
        self.assertEqual(exc_info.exception.status_code, 502)
        self.assertIn("metadata unavailable", str(exc_info.exception.detail))

    async def test_preview_bridge_readiness_fails_closed_when_active_session_disappears_after_restart(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        initial_status = _bridge_status(state="expired", detail="expired bridge token")

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=[initial_status, initial_status]),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=None),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service._ensure_workspace_preview_bridge_ready(session)

        self.assertEqual(exc_info.exception.status_code, 502)
        self.assertIn("no longer active", str(exc_info.exception.detail))

    async def test_preview_bridge_readiness_fails_closed_when_active_session_changes_after_restart(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        initial_status = _bridge_status(state="expired", detail="expired bridge token")

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=[initial_status, initial_status]),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ),
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-2")),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service._ensure_workspace_preview_bridge_ready(session)

        self.assertEqual(exc_info.exception.status_code, 502)
        self.assertIn("changed", str(exc_info.exception.detail))

    async def test_preview_bridge_readiness_cooldown_blocks_second_failed_recovery(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        expired_status = _bridge_status(state="expired", detail="expired bridge token")
        failed_status = _bridge_status(state="missing", detail="metadata unavailable")

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(
                    side_effect=[
                        expired_status,
                        expired_status,
                        failed_status,
                        expired_status,
                        expired_status,
                    ]
                ),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
            mock.patch.object(self.service, "_bridge_recovery_monotonic", side_effect=[100.0, 140.0]),
        ):
            with self.assertRaises(HTTPException) as first_exc:
                await self.service._ensure_workspace_preview_bridge_ready(session)
            with self.assertRaises(HTTPException) as second_exc:
                await self.service._ensure_workspace_preview_bridge_ready(session)

        self.assertEqual(first_exc.exception.status_code, 502)
        self.assertEqual(second_exc.exception.status_code, 503)
        self.assertEqual(second_exc.exception.headers, {"Retry-After": "80"})
        restart_wait.assert_awaited_once_with("ws-1", timeout_seconds=60.0)

    async def test_preview_bridge_readiness_can_retry_after_cooldown_expires(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        expired_status = _bridge_status(state="expired", detail="expired bridge token")
        failed_status = _bridge_status(state="missing", detail="metadata unavailable")

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(
                    side_effect=[
                        expired_status,
                        expired_status,
                        failed_status,
                        expired_status,
                        expired_status,
                        failed_status,
                    ]
                ),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
            mock.patch.object(self.service, "_bridge_recovery_monotonic", side_effect=[100.0, 221.0]),
        ):
            with self.assertRaises(HTTPException):
                await self.service._ensure_workspace_preview_bridge_ready(session)
            with self.assertRaises(HTTPException):
                await self.service._ensure_workspace_preview_bridge_ready(session)

        self.assertEqual(restart_wait.await_count, 2)

    async def test_preview_bridge_readiness_concurrent_waiters_after_failure_share_cooldown(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        expired_status = _bridge_status(state="expired", detail="expired bridge token")
        failed_status = _bridge_status(state="missing", detail="metadata unavailable")
        restart_started = asyncio.Event()
        allow_restart_finish = asyncio.Event()
        second_prelock_seen = asyncio.Event()

        status_calls = 0

        async def get_bridge_status(*args, **kwargs):
            nonlocal status_calls
            status_calls += 1
            if status_calls == 3:
                second_prelock_seen.set()
            if status_calls <= 3:
                return expired_status
            return failed_status

        async def restart(*args, **kwargs):
            restart_started.set()
            await allow_restart_finish.wait()

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(return_value={"provider": "status"}),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=get_bridge_status),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(side_effect=restart),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
            mock.patch.object(self.service, "_bridge_recovery_monotonic", side_effect=[100.0, 101.0]),
        ):
            first = asyncio.create_task(self.service._ensure_workspace_preview_bridge_ready(session))
            await restart_started.wait()
            second = asyncio.create_task(self.service._ensure_workspace_preview_bridge_ready(session))
            await second_prelock_seen.wait()
            allow_restart_finish.set()
            results = await asyncio.gather(first, second, return_exceptions=True)

        self.assertEqual(restart_wait.await_count, 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, HTTPException) for result in results))
        statuses = sorted(result.status_code for result in results if isinstance(result, HTTPException))
        self.assertEqual(statuses, [502, 503])

    async def test_preview_bridge_readiness_restarts_once_for_two_concurrent_calls(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        expired_status = _bridge_status(state="expired", detail="expired bridge token")
        healthy_status = _bridge_status(expires_at=datetime.now(UTC) + timedelta(seconds=1200))
        restart_started = asyncio.Event()
        second_prelock_seen = asyncio.Event()
        allow_restart_finish = asyncio.Event()
        second_inlock_seen = asyncio.Event()
        status_calls = 0

        async def get_status(*args, **kwargs):
            return {"provider": "status", "call": kwargs.get("max_age_seconds")}

        async def get_bridge_status(*args, **kwargs):
            nonlocal status_calls
            status_calls += 1
            if status_calls == 3:
                second_prelock_seen.set()
            if status_calls == 5:
                second_inlock_seen.set()
            if status_calls <= 3:
                return expired_status
            return healthy_status

        async def restart(*args, **kwargs):
            restart_started.set()
            await allow_restart_finish.wait()

        with (
            mock.patch.object(
                self.service,
                "_runtime_provider_get_status",
                mock.AsyncMock(side_effect=get_status),
            ),
            mock.patch.object(
                self.service,
                "_get_runtime_bridge_status_for_session",
                mock.AsyncMock(side_effect=get_bridge_status),
            ),
            mock.patch.object(
                self.service,
                "restart_runtime_env_vars_and_wait",
                mock.AsyncMock(side_effect=restart),
            ) as restart_wait,
            mock.patch.object(
                self.service,
                "_get_active_session_row",
                mock.AsyncMock(return_value=_session_row(session_id="sess-1")),
            ),
        ):
            first = asyncio.create_task(self.service._ensure_workspace_preview_bridge_ready(session))
            await restart_started.wait()
            second = asyncio.create_task(self.service._ensure_workspace_preview_bridge_ready(session))
            await second_prelock_seen.wait()
            allow_restart_finish.set()
            await asyncio.gather(first, second)

        restart_wait.assert_awaited_once_with("ws-1", timeout_seconds=60.0)
        self.assertTrue(second_inlock_seen.is_set())

    async def test_issue_workspace_preview_launch_waits_for_readiness_before_building_response(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))
        call_order: list[str] = []
        expected = SimpleNamespace(preview_url="https://preview")

        async def ensure_ready(arg_session):
            self.assertIs(arg_session, session)
            call_order.append("ready")

        async def build_response(**kwargs):
            call_order.append("build")
            return expected

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ),
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(side_effect=ensure_ready),
            ),
            mock.patch.object(
                self.service,
                "_build_workspace_preview_launch_response",
                mock.AsyncMock(side_effect=build_response),
            ) as build_launch,
        ):
            result = await self.service.issue_workspace_preview_launch(
                "ws-1",
                "user-1",
                control_plane_origin="https://ragtime.dev.visnovsky.us",
                path="/preview",
            )

        self.assertIs(result, expected)
        build_launch.assert_awaited_once()
        self.assertEqual(call_order, ["ready", "build"])

    async def test_issue_workspace_preview_launch_forwards_auto_start_true(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ) as ensure_session,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ),
            mock.patch.object(
                self.service,
                "_build_workspace_preview_launch_response",
                mock.AsyncMock(return_value=SimpleNamespace(preview_url="https://preview")),
            ),
        ):
            await self.service.issue_workspace_preview_launch(
                "ws-1",
                "user-1",
                control_plane_origin="https://ragtime.dev.visnovsky.us",
                auto_start=True,
            )

        ensure_session.assert_awaited_once_with("ws-1", "user-1", auto_start=True)

    async def test_issue_workspace_preview_launch_does_not_build_response_when_readiness_fails(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ),
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(side_effect=HTTPException(status_code=502, detail="bridge not ready")),
            ),
            mock.patch.object(
                self.service,
                "_build_workspace_preview_launch_response",
                mock.AsyncMock(),
            ) as build_launch,
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await self.service.issue_workspace_preview_launch(
                    "ws-1",
                    "user-1",
                    control_plane_origin="https://ragtime.dev.visnovsky.us",
                )

        self.assertEqual(exc_info.exception.detail, "bridge not ready")
        build_launch.assert_not_awaited()

    async def test_issue_workspace_preview_launch_defaults_auto_start_false(self) -> None:
        session = self.service._to_runtime_session(_session_row(session_id="sess-1"))

        with (
            mock.patch.object(
                self.service,
                "ensure_workspace_preview_session",
                mock.AsyncMock(return_value=session),
            ) as ensure_session,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ),
            mock.patch.object(
                self.service,
                "_build_workspace_preview_launch_response",
                mock.AsyncMock(return_value=SimpleNamespace(preview_url="https://preview")),
            ),
        ):
            await self.service.issue_workspace_preview_launch(
                "ws-1",
                "user-1",
                control_plane_origin="https://ragtime.dev.visnovsky.us",
            )

        ensure_session.assert_awaited_once_with("ws-1", "user-1", auto_start=False)

    async def test_bridge_refresh_watch_only_recovers_active_near_expiry_workspaces(self) -> None:
        healthy_status = _bridge_status(expires_at=datetime.now(UTC) + timedelta(seconds=900), session_id="sess-healthy")
        expired_status = _bridge_status(
            state="expired",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            detail="expired bridge token",
            session_id="sess-expired",
        )
        rows = [
            _session_row(session_id="sess-healthy", workspace_id="ws-healthy", provider_session_id="provider-healthy"),
            _session_row(session_id="sess-older", workspace_id="ws-healthy", provider_session_id="provider-older"),
            _session_row(session_id="sess-expired", workspace_id="ws-expired", provider_session_id="provider-expired"),
            _session_row(session_id="sess-empty", workspace_id="", provider_session_id="provider-empty"),
        ]
        db = SimpleNamespace(userspaceruntimesession=SimpleNamespace(find_many=mock.AsyncMock(return_value=rows)))

        async def get_status(session):
            if session.workspace_id == "ws-healthy":
                return healthy_status
            if session.workspace_id == "ws-expired":
                return expired_status
            raise AssertionError(f"unexpected workspace {session.workspace_id}")

        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                self.service,
                "_get_workspace_preview_bridge_status",
                mock.AsyncMock(side_effect=get_status),
            ) as get_status_mock,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ) as ensure_ready,
        ):
            await self.service._refresh_runtime_bridge_credentials_for_active_workspaces()

        db.userspaceruntimesession.find_many.assert_awaited_once_with(
            where={"state": {"in": ["starting", "running"]}},
            order={"updatedAt": "desc"},
        )
        self.assertEqual(
            [call.args[0].workspace_id for call in get_status_mock.await_args_list],
            ["ws-healthy", "ws-expired"],
        )
        ensure_ready.assert_awaited_once()
        ensure_ready_args = ensure_ready.await_args
        assert ensure_ready_args is not None
        self.assertEqual(ensure_ready_args.args[0].workspace_id, "ws-expired")

    async def test_bridge_refresh_watch_continues_after_workspace_failure(self) -> None:
        rows = [
            _session_row(session_id="sess-failing", workspace_id="ws-failing", provider_session_id="provider-failing"),
            _session_row(session_id="sess-recover", workspace_id="ws-recover", provider_session_id="provider-recover"),
        ]
        db = SimpleNamespace(userspaceruntimesession=SimpleNamespace(find_many=mock.AsyncMock(return_value=rows)))
        expired_status = _bridge_status(
            state="expired",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
            detail="expired bridge token",
            session_id="sess-recover",
        )

        async def get_status(session):
            if session.workspace_id == "ws-failing":
                raise HTTPException(status_code=502, detail="provider down")
            if session.workspace_id == "ws-recover":
                return expired_status
            raise AssertionError(f"unexpected workspace {session.workspace_id}")

        with (
            mock.patch("ragtime.userspace.runtime_service.get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                self.service,
                "_get_workspace_preview_bridge_status",
                mock.AsyncMock(side_effect=get_status),
            ) as get_status_mock,
            mock.patch.object(
                self.service,
                "_ensure_workspace_preview_bridge_ready",
                mock.AsyncMock(),
            ) as ensure_ready,
        ):
            await self.service._refresh_runtime_bridge_credentials_for_active_workspaces()

        self.assertEqual(
            [call.args[0].workspace_id for call in get_status_mock.await_args_list],
            ["ws-failing", "ws-recover"],
        )
        ensure_ready.assert_awaited_once()
        ensure_ready_args = ensure_ready.await_args
        assert ensure_ready_args is not None
        self.assertEqual(ensure_ready_args.args[0].workspace_id, "ws-recover")

    async def test_schedule_bridge_refresh_watch_is_idempotent(self) -> None:
        release_task = asyncio.Event()

        async def wait_forever() -> None:
            await release_task.wait()

        with mock.patch.object(self.service, "_runtime_bridge_refresh_watch_loop", side_effect=wait_forever):
            self.service.schedule_runtime_bridge_refresh_watch()
            first_task = self.service._runtime_bridge_refresh_watch_task
            self.service.schedule_runtime_bridge_refresh_watch()

            assert first_task is not None
            self.assertIs(self.service._runtime_bridge_refresh_watch_task, first_task)
            self.assertEqual(first_task.get_name(), "userspace-runtime-bridge-refresh-watch")
            await self.service.shutdown_runtime_bridge_refresh_watch()

        self.assertIsNone(self.service._runtime_bridge_refresh_watch_task)

    async def test_shutdown_bridge_refresh_watch_cancels_task(self) -> None:
        release_task = asyncio.Event()

        async def wait_forever() -> None:
            await release_task.wait()

        task = asyncio.create_task(wait_forever())
        self.service._runtime_bridge_refresh_watch_task = task

        await self.service.shutdown_runtime_bridge_refresh_watch()

        self.assertTrue(task.cancelled())
        self.assertIsNone(self.service._runtime_bridge_refresh_watch_task)

    async def test_shutdown_bridge_refresh_watch_logs_completed_task_failure_and_retrieves_exception(self) -> None:
        async def fail_watch() -> None:
            raise RuntimeError("bridge-watch boom secret-token")

        task = asyncio.create_task(fail_watch())
        await asyncio.sleep(0)
        self.service._runtime_bridge_refresh_watch_task = task

        with mock.patch("ragtime.userspace.runtime_service.logger.warning") as log_warning:
            await self.service.shutdown_runtime_bridge_refresh_watch()

        self.assertIsNone(self.service._runtime_bridge_refresh_watch_task)
        self.assertFalse(getattr(task, "_log_traceback", True))
        log_warning.assert_called_once()
        self.assertNotIn("secret-token", str(log_warning.call_args))
