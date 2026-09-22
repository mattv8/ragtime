import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from ragtime.userspace.runtime_service import UserSpaceRuntimeService

_NOW = datetime.now(timezone.utc)
_UNSET = object()


class _RuntimeSessionTable:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows

    async def find_many(self, **kwargs: object) -> list[SimpleNamespace]:
        return self.rows


def _row(
    *,
    state: str = "starting",
    provider: str = "microvm_pool_v1",
    provider_session_id: str | None = "provider-1",
    updated_at: datetime | None | object = _UNSET,
    last_heartbeat_at: datetime | None | object = _UNSET,
) -> SimpleNamespace:
    old = _NOW - timedelta(minutes=10)
    return SimpleNamespace(
        state=state,
        runtimeProvider=provider,
        providerSessionId=provider_session_id,
        updatedAt=old if updated_at is _UNSET else updated_at,
        lastHeartbeatAt=old if last_heartbeat_at is _UNSET else last_heartbeat_at,
    )


class LegacyRuntimeLivenessTests(unittest.IsolatedAsyncioTestCase):
    async def _has_active(self, rows: list[SimpleNamespace], status: AsyncMock) -> bool:
        service = UserSpaceRuntimeService()
        service._runtime_provider_get_status = status  # type: ignore[method-assign]
        db = SimpleNamespace(userspaceruntimesession=_RuntimeSessionTable(rows))
        with patch("ragtime.userspace.runtime_service.get_db", AsyncMock(return_value=db)):
            return await service.has_active_or_stopping_workspace_session("workspace-1")

    async def test_no_candidate_sessions_are_inactive(self) -> None:
        status = AsyncMock()

        self.assertFalse(await self._has_active([], status))
        status.assert_not_awaited()

    async def test_all_old_matching_sessions_confirmed_missing_are_inactive(self) -> None:
        status = AsyncMock(return_value=None)

        self.assertFalse(
            await self._has_active(
                [
                    _row(state="starting", provider_session_id="one"),
                    _row(state="running", provider_session_id="two"),
                    _row(state="stopping", provider_session_id="three"),
                ],
                status,
            )
        )
        self.assertEqual([call.args[0] for call in status.await_args_list], ["one", "two", "three"])
        for call in status.await_args_list:
            self.assertEqual(call.kwargs, {"max_age_seconds": 0, "allow_stale_on_error": False})

    async def test_mixed_confirmed_missing_and_live_sessions_remain_active_after_scanning_all(self) -> None:
        status = AsyncMock(side_effect=[None, {"state": "running"}])

        self.assertTrue(await self._has_active([_row(provider_session_id="missing"), _row(provider_session_id="live")], status))
        self.assertEqual([call.args[0] for call in status.await_args_list], ["missing", "live"])

    async def test_provider_mismatch_empty_id_recent_or_missing_timestamps_remain_active(self) -> None:
        status = AsyncMock(return_value=None)
        recent = _NOW - timedelta(seconds=30)

        self.assertTrue(
            await self._has_active(
                [
                    _row(provider="other", provider_session_id="other-provider"),
                    _row(provider_session_id=""),
                    _row(provider_session_id="recent", updated_at=recent, last_heartbeat_at=recent),
                    _row(provider_session_id="missing-timestamp", last_heartbeat_at=None),
                ],
                status,
            )
        )
        self.assertEqual([call.args[0] for call in status.await_args_list], ["recent", "missing-timestamp"])

    async def test_manager_error_or_timeout_remains_active(self) -> None:
        status = AsyncMock(side_effect=HTTPException(status_code=500, detail="manager failure"))
        self.assertTrue(await self._has_active([_row()], status))

        service = UserSpaceRuntimeService()
        service._runtime_provider_get_status = AsyncMock()  # type: ignore[method-assign]
        db = SimpleNamespace(userspaceruntimesession=_RuntimeSessionTable([_row()]))

        async def timeout_wait_for(coro: object, *, timeout: float) -> object:
            self.assertEqual(timeout, 5.0)
            cast_coro = coro
            close = getattr(cast_coro, "close", None)
            if close is not None:
                close()
            raise asyncio.TimeoutError

        with (
            patch("ragtime.userspace.runtime_service.get_db", AsyncMock(return_value=db)),
            patch("ragtime.userspace.runtime_service.asyncio.wait_for", new=timeout_wait_for),
        ):
            self.assertTrue(await service.has_active_or_stopping_workspace_session("workspace-1"))
