from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException, Request

from ragtime.userspace import sqlite_history_routes as routes
from ragtime.userspace.sqlite_history_transport import ClosingStreamingResponse


def _backup(**changes):
    backup = {
        "id": "backup-1",
        "workspace_id": "workspace-1",
        "database_name": "app.sqlite3",
        "created_at": "2026-01-01T00:00:00+00:00",
        "trigger": "manual",
        "snapshot_id": "snapshot-1",
        "snapshot_git_commit_hash": None,
        "status": "ready",
        "size_bytes": 1,
        "sha256": None,
        "error": None,
        "can_restore": True,
        "can_delete": True,
        "capture_job_id": None,
    }
    backup.update(changes)
    return backup


def _job(**changes):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    job = {
        "id": "job-1",
        "workspace_id": "workspace-1",
        "trigger": "manual",
        "database_names": ["app.sqlite3"],
        "snapshot_id": "snapshot-1",
        "snapshot_git_commit_hash": None,
        "status": "completed",
        "created_at": now,
        "available_at": now,
        "started_at": now,
        "finished_at": now,
        "updated_at": now,
        "completed_databases": 1,
        "total_databases": 1,
        "backup_ids": ["backup-1"],
        "error_message": None,
        "cancel_requested": False,
        "owner_token": "private-owner-token",
        "request_hash": "private-request-hash",
    }
    job.update(changes)
    return job


class SqliteHistoryEventRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.history = SimpleNamespace(
            list_backups=mock.AsyncMock(return_value=[_backup()]),
            interrupted_maintenance=mock.AsyncMock(return_value=None),
            history_state=mock.AsyncMock(return_value={"backups": [_backup()], "interrupted_maintenance": None}),
        )
        self.queue = SimpleNamespace(list_jobs=mock.AsyncMock(return_value=[_job()]))
        self.manage = mock.patch.object(routes, "_manage", new_callable=mock.AsyncMock)
        self.history_service = mock.patch.object(routes, "get_sqlite_history_service", return_value=self.history)
        self.queue_service = mock.patch.object(routes, "get_sqlite_backup_queue_service", return_value=self.queue)
        self.manage.start()
        self.history_service.start()
        self.queue_service.start()
        self.request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
        self.request_is_disconnected = mock.patch.object(
            self.request,
            "is_disconnected",
            new=mock.AsyncMock(return_value=False),
        )
        self.is_disconnected = self.request_is_disconnected.start()
        self.user = SimpleNamespace(id="owner-1", role="user")

    async def asyncTearDown(self) -> None:
        self.request_is_disconnected.stop()
        self.queue_service.stop()
        self.history_service.stop()
        self.manage.stop()

    async def _stream(self):
        response = await routes.stream_sqlite_history_events("workspace-1", self.request, "app.sqlite3", "snapshot-1", self.user)
        return response, aiter(response.body_iterator)

    async def test_streams_initial_notification_with_no_cache_headers(self) -> None:
        response, events = await self._stream()

        self.assertEqual("text/event-stream; charset=utf-8", response.headers["content-type"])
        self.assertEqual("no-store", response.headers["cache-control"])
        self.assertEqual("no", response.headers["x-accel-buffering"])
        self.assertEqual("event: history_changed\ndata: {}\n\n", await anext(events))
        self.history.history_state.assert_awaited_once_with("workspace-1", database_name="app.sqlite3", snapshot_id="snapshot-1")
        self.queue.list_jobs.assert_awaited_once_with("workspace-1", database_name="app.sqlite3", snapshot_id="snapshot-1", limit=50)

    async def test_streams_change_then_keepalive_using_active_and_idle_intervals(self) -> None:
        response, events = await self._stream()
        self.assertEqual("event: history_changed\ndata: {}\n\n", await anext(events))
        self.queue.list_jobs.return_value = [_job(status="pending")]

        with mock.patch.object(routes.asyncio, "sleep", new_callable=mock.AsyncMock) as sleep:
            self.assertEqual("event: history_changed\ndata: {}\n\n", await anext(events))
            sleep.assert_awaited_once_with(5)
            self.assertEqual(": keepalive\n\n", await anext(events))
            self.assertEqual([mock.call(5), mock.call(2)], sleep.await_args_list)

    async def test_stream_compares_only_public_filtered_history_and_jobs(self) -> None:
        response, events = await self._stream()
        await anext(events)
        self.queue.list_jobs.return_value = [_job(owner_token="changed-secret", request_hash="changed-private")]

        with mock.patch.object(routes.asyncio, "sleep", new_callable=mock.AsyncMock):
            self.assertEqual(": keepalive\n\n", await anext(events))

    async def test_denies_before_collecting_the_initial_stream_payload(self) -> None:
        self.manage.stop()
        with mock.patch.object(
            routes,
            "_manage",
            new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Forbidden")),
        ):
            with self.assertRaises(HTTPException) as error:
                await self._stream()

        self.assertEqual(403, error.exception.status_code)
        self.history.history_state.assert_not_awaited()
        self.queue.list_jobs.assert_not_awaited()

    async def test_denies_before_creating_stream_and_revokes_later_access(self) -> None:
        self.manage.stop()
        with mock.patch.object(
            routes,
            "_manage",
            new=mock.AsyncMock(side_effect=[None, HTTPException(status_code=403, detail="Forbidden")]),
        ):
            response, events = await self._stream()
            self.assertEqual("event: history_changed\ndata: {}\n\n", await anext(events))

            with mock.patch.object(routes.asyncio, "sleep", new_callable=mock.AsyncMock):
                self.assertEqual("event: access_revoked\ndata: {}\n\n", await anext(events))
                with self.assertRaises(StopAsyncIteration):
                    await anext(events)

    async def test_does_not_observe_after_disconnect(self) -> None:
        self.is_disconnected.return_value = True
        response, events = await self._stream()

        with self.assertRaises(StopAsyncIteration):
            await anext(events)
        self.history.history_state.assert_awaited_once()

    async def test_streaming_cleanup_runs_when_asgi_send_fails_before_body_iteration(self) -> None:
        cleanup = mock.AsyncMock()

        async def body():
            yield b"unreachable"

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            raise RuntimeError("client disconnected")

        response = ClosingStreamingResponse(body(), cleanup=cleanup)
        with self.assertRaisesRegex(RuntimeError, "client disconnected"):
            await response({"type": "http", "method": "GET", "headers": []}, receive, send)

        cleanup.assert_awaited_once()
