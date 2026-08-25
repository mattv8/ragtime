from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

import ragtime.main as _MAIN
from ragtime.userspace.service import UserSpaceService


def _build_request(path: str = "/shared/tok-1") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"host", b"ragtime.test")],
            "scheme": "https",
            "server": ("ragtime.test", 443),
            "client": ("127.0.0.1", 1234),
        }
    )


class ShareRecordResolutionParallelismTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_resolution_runs_both_lookups_concurrently(self) -> None:
        service = UserSpaceService()
        started: list[str] = []
        both_started = asyncio.Event()

        async def fake_workspace_lookup(*args: Any, **kwargs: Any) -> Any:
            started.append("workspace")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return SimpleNamespace(id="ws-share-1", workspaceId="workspace-1")

        async def fake_conversation_lookup(*args: Any, **kwargs: Any) -> Any:
            started.append("conversation")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return None

        with (
            mock.patch.object(service, "_find_workspace_share_by_token", new=fake_workspace_lookup),
            mock.patch.object(service, "_find_conversation_share_by_token", new=fake_conversation_lookup),
        ):
            target_type, record = await service._resolve_public_share_record_by_token("tok-1")

        self.assertEqual(target_type, "workspace")
        self.assertEqual(record.id, "ws-share-1")
        self.assertEqual(sorted(started), ["conversation", "workspace"])

    async def test_slug_resolution_runs_both_lookups_concurrently(self) -> None:
        service = UserSpaceService()
        started: list[str] = []
        both_started = asyncio.Event()

        async def fake_workspace_lookup(*args: Any, **kwargs: Any) -> Any:
            started.append("workspace")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return None

        async def fake_conversation_lookup(*args: Any, **kwargs: Any) -> Any:
            started.append("conversation")
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
            return SimpleNamespace(id="conv-share-1")

        with (
            mock.patch.object(service, "_resolve_share_owner_ids", new=mock.AsyncMock(return_value=["owner-1"])),
            mock.patch.object(service, "_find_workspace_share_by_slug", new=fake_workspace_lookup),
            mock.patch.object(service, "_find_conversation_share_by_slug", new=fake_conversation_lookup),
        ):
            target_type, record = await service._resolve_public_share_record_by_slug("alice", "dash")

        self.assertEqual(target_type, "conversation")
        self.assertEqual(record.id, "conv-share-1")
        self.assertEqual(sorted(started), ["conversation", "workspace"])

    async def test_ambiguous_token_still_raises_409(self) -> None:
        service = UserSpaceService()
        with (
            mock.patch.object(service, "_find_workspace_share_by_token", new=mock.AsyncMock(return_value=SimpleNamespace(id="a"))),
            mock.patch.object(service, "_find_conversation_share_by_token", new=mock.AsyncMock(return_value=SimpleNamespace(id="b"))),
        ):
            with self.assertRaises(HTTPException) as raised:
                await service._resolve_public_share_record_by_token("tok-1")
        self.assertEqual(raised.exception.status_code, 409)


class DeferredShareAnalyticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_entry_hit_is_recorded_off_the_critical_path(self) -> None:
        release = asyncio.Event()
        recorded = mock.AsyncMock()

        async def slow_record(*args: Any, **kwargs: Any) -> None:
            await release.wait()
            await recorded(*args, **kwargs)

        with (
            mock.patch.object(_MAIN.userspace_service, "record_public_share_hit", new=slow_record),
            mock.patch.object(_MAIN, "_should_record_public_share_entry", return_value=True),
        ):
            await asyncio.wait_for(
                _MAIN._record_public_share_entry_hit(
                    _build_request(),
                    route_kind="token",
                    path="",
                    target_type="workspace",
                    share_id="share-1",
                    outcome="redirect",
                    current_user=None,
                ),
                timeout=0.5,
            )
            recorded.assert_not_awaited()
            release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        recorded.assert_awaited_once()
        await_args = recorded.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.args[1], "workspace")
        self.assertEqual(await_args.args[2], "share-1")
        self.assertEqual(await_args.args[3], "redirect")

    async def test_entry_hit_task_failure_is_swallowed_and_logged(self) -> None:
        failing = mock.AsyncMock(side_effect=RuntimeError("db down"))
        with (
            mock.patch.object(_MAIN.userspace_service, "record_public_share_hit", new=failing),
            mock.patch.object(_MAIN, "_should_record_public_share_entry", return_value=True),
            mock.patch.object(_MAIN.logger, "exception") as log_exc,
        ):
            await _MAIN._record_public_share_entry_hit(
                _build_request(),
                route_kind="token",
                path="",
                target_type="workspace",
                share_id="share-1",
                outcome="redirect",
                current_user=None,
            )
            for _ in range(4):
                await asyncio.sleep(0)
        failing.assert_awaited_once()
        log_exc.assert_called_once()


class SingleShareResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_route_resolves_share_record_exactly_once(self) -> None:
        share_record = SimpleNamespace(
            id="share-1",
            workspaceId="workspace-1",
            workspace=SimpleNamespace(name="Sales", owner=SimpleNamespace(username="alice", displayName="Alice")),
            ownerUser=SimpleNamespace(username="alice", displayName="Alice"),
            conversation=None,
        )
        resolve = mock.AsyncMock(return_value=("workspace", share_record))
        launch = SimpleNamespace(preview_url="https://workspace-1.ragtime.test/__ragtime/bootstrap?grant=g")

        with (
            mock.patch.object(_MAIN, "_share_current_user_from_request", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(_MAIN, "share_auth_token_from_request", return_value=None),
            mock.patch.object(_MAIN.userspace_service, "_resolve_public_share_record_by_token", new=resolve),
            mock.patch.object(_MAIN.userspace_service, "resolve_shared_workspace_id", new=mock.AsyncMock(return_value="workspace-1")),
            mock.patch.object(_MAIN.userspace_runtime_service, "issue_shared_preview_launch", new=mock.AsyncMock(return_value=launch)),
            mock.patch.object(_MAIN, "_record_public_share_entry_hit", new=mock.AsyncMock()),
        ):
            response = await _MAIN._shared_launch_redirect_by_token("tok-1", _build_request(), "")

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["location"], launch.preview_url)
        resolve.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
