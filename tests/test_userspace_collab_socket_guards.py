from __future__ import annotations

import json
import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import WebSocketDisconnect

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.userspace.models import UserSpaceCollabSnapshotResponse


class _FakeWebSocket:
    def __init__(self, messages: Sequence[dict[str, Any]]) -> None:
        self.cookies = {"userspace_collab_capability": "capability-token"}
        self._messages = iter(messages)
        self.sent_texts: list[str] = []
        self.closed_codes: list[int] = []

    async def accept(self) -> None:
        return None

    async def close(self, *, code: int) -> None:
        self.closed_codes.append(code)

    async def receive_text(self) -> str:
        try:
            return json.dumps(next(self._messages))
        except StopIteration as exc:
            raise WebSocketDisconnect(code=1000) from exc

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)


def _snapshot(*, version: int = 1, content: str = "original") -> UserSpaceCollabSnapshotResponse:
    return UserSpaceCollabSnapshotResponse(
        workspace_id="workspace-1",
        file_path="src/app.py",
        version=version,
        content=content,
        read_only=False,
    )


def _runtime_service(snapshot: UserSpaceCollabSnapshotResponse) -> SimpleNamespace:
    return SimpleNamespace(
        get_collab_snapshot=mock.AsyncMock(return_value=snapshot),
        register_collab_client=mock.AsyncMock(return_value=snapshot),
        clear_collab_presence=mock.AsyncMock(return_value=[]),
        unregister_collab_client=mock.AsyncMock(),
        get_collab_presence=mock.AsyncMock(return_value=[]),
        get_collab_clients=mock.AsyncMock(return_value=[]),
        _normalize_file_path=mock.Mock(side_effect=lambda path: path.strip().lstrip("/")),
        apply_collab_update=mock.AsyncMock(),
    )


class CollabSocketGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_mismatched_update_path_is_rejected_without_applying(self) -> None:
        snapshot = _snapshot()
        service = _runtime_service(snapshot)
        websocket = _FakeWebSocket([{"type": "update", "file_path": "other.py", "version": 1, "content": "wrong file"}])

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=service),
            mock.patch.object(_RUNTIME_ROUTES, "_require_workspace_capability", return_value=({}, "user-1")),
        ):
            await _RUNTIME_ROUTES.collab_file_socket("workspace-1", "src/app.py", cast(Any, websocket))

        service.apply_collab_update.assert_not_awaited()
        self.assertEqual(json.loads(websocket.sent_texts[-1])["type"], "error")

    async def test_missing_or_zero_update_version_is_rejected_with_fresh_snapshot(self) -> None:
        for payload in (
            {"type": "update", "file_path": "src/app.py", "content": "missing version"},
            {"type": "update", "file_path": "src/app.py", "version": 0, "content": "zero version"},
        ):
            with self.subTest(payload=payload):
                snapshot = _snapshot()
                latest = _snapshot(version=2, content="latest")
                service = _runtime_service(snapshot)
                service.get_collab_snapshot.side_effect = [snapshot, latest]
                websocket = _FakeWebSocket([payload])

                with (
                    mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=service),
                    mock.patch.object(_RUNTIME_ROUTES, "_require_workspace_capability", return_value=({}, "user-1")),
                ):
                    await _RUNTIME_ROUTES.collab_file_socket("workspace-1", "src/app.py", cast(Any, websocket))

                service.apply_collab_update.assert_not_awaited()
                self.assertIn("current 2", json.loads(websocket.sent_texts[-2])["message"])
                self.assertEqual(json.loads(websocket.sent_texts[-1]), {"type": "snapshot", **latest.model_dump()})

    async def test_missing_path_or_non_string_content_is_rejected_without_applying(self) -> None:
        for payload in (
            {"type": "update", "version": 1, "content": "missing path"},
            {"type": "update", "file_path": "src/app.py", "version": 1},
            {"type": "update", "file_path": "src/app.py", "version": 1, "content": 42},
        ):
            with self.subTest(payload=payload):
                snapshot = _snapshot()
                service = _runtime_service(snapshot)
                websocket = _FakeWebSocket([payload])

                with (
                    mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=service),
                    mock.patch.object(_RUNTIME_ROUTES, "_require_workspace_capability", return_value=({}, "user-1")),
                ):
                    await _RUNTIME_ROUTES.collab_file_socket("workspace-1", "src/app.py", cast(Any, websocket))

                service.apply_collab_update.assert_not_awaited()
                self.assertEqual(json.loads(websocket.sent_texts[-1])["type"], "error")

    async def test_valid_update_is_applied_and_acked(self) -> None:
        snapshot = _snapshot()
        updated = _snapshot(version=2, content="updated")
        service = _runtime_service(snapshot)
        service.apply_collab_update.return_value = updated
        websocket = _FakeWebSocket([{"type": "update", "file_path": "/src/app.py", "version": 1, "content": "updated"}])

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=service),
            mock.patch.object(_RUNTIME_ROUTES, "_require_workspace_capability", return_value=({}, "user-1")),
        ):
            await _RUNTIME_ROUTES.collab_file_socket("workspace-1", "src/app.py", cast(Any, websocket))

        service.apply_collab_update.assert_awaited_once_with(
            "workspace-1",
            "src/app.py",
            "updated",
            "user-1",
            expected_version=1,
        )
        self.assertEqual(json.loads(websocket.sent_texts[-1]), {"type": "ack", "workspace_id": "workspace-1", "file_path": "src/app.py", "version": 2})

    async def test_register_failure_cleans_up_and_closes_socket(self) -> None:
        snapshot = _snapshot()
        service = _runtime_service(snapshot)
        service.register_collab_client.side_effect = RuntimeError("runtime unavailable")
        websocket = _FakeWebSocket([])

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=service),
            mock.patch.object(_RUNTIME_ROUTES, "_require_workspace_capability", return_value=({}, "user-1")),
        ):
            await _RUNTIME_ROUTES.collab_file_socket("workspace-1", "src/app.py", cast(Any, websocket))

        service.unregister_collab_client.assert_awaited_once()
        self.assertEqual(websocket.closed_codes, [1011])
