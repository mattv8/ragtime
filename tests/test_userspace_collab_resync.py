from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from ragtime.userspace.runtime_service import UserSpaceRuntimeService, _CollabDocState


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_texts: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)


class CollabResyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = UserSpaceRuntimeService()
        self.workspace_id = "workspace-1"
        self.file_path = "src/app.py"
        self.user_id = "user-1"

    def _set_state(self, *, content: str, disk_content: str, version: int = 1) -> _CollabDocState:
        state = _CollabDocState(
            workspace_id=self.workspace_id,
            file_path=self.file_path,
            content=content,
            disk_content=disk_content,
            version=version,
        )
        self.service._collab_docs[(self.workspace_id, self.file_path)] = state
        return state

    def _service_dependencies(self) -> Any:
        return mock.patch.multiple(
            "ragtime.userspace.runtime_service.userspace_service",
            enforce_workspace_role=mock.AsyncMock(),
            ensure_workspace_path_not_in_disabled_mount=mock.AsyncMock(side_effect=lambda _workspace_id, path: path),
        )

    async def test_snapshot_adopts_external_disk_change_and_broadcasts(self) -> None:
        state = self._set_state(content="old", disk_content="old")
        client = _FakeWebSocket()
        state.clients.add(cast(Any, client))
        self.service._load_collab_disk_content = mock.AsyncMock(return_value=("external", True, True))  # type: ignore[method-assign]

        with self._service_dependencies():
            snapshot = await self.service.get_collab_snapshot(self.workspace_id, self.file_path, self.user_id)

        self.assertEqual((snapshot.content, snapshot.version), ("external", 2))
        self.assertEqual(state.disk_content, "external")
        self.assertEqual(
            json.loads(client.sent_texts[0]),
            {
                "type": "update",
                "workspace_id": self.workspace_id,
                "file_path": self.file_path,
                "version": 2,
                "content": "external",
            },
        )

    async def test_snapshot_does_not_adopt_persisting_or_known_disk_content(self) -> None:
        for disk_content in ("pending", "old"):
            with self.subTest(disk_content=disk_content):
                state = self._set_state(content="pending", disk_content="old", version=3)
                self.service._load_collab_disk_content = mock.AsyncMock(return_value=(disk_content, True, True))  # type: ignore[method-assign]

                with self._service_dependencies():
                    snapshot = await self.service.get_collab_snapshot(self.workspace_id, self.file_path, self.user_id)

                self.assertEqual((snapshot.content, snapshot.version), ("pending", 3))
                self.assertEqual(state.disk_content, "old")

    async def test_apply_update_records_persisted_disk_content(self) -> None:
        state = self._set_state(content="old", disk_content="old")
        self.service._persist_file_content = mock.AsyncMock()  # type: ignore[method-assign]
        self.service._store_collab_checkpoint = mock.AsyncMock()  # type: ignore[method-assign]
        self.service.bump_workspace_generation = mock.AsyncMock()  # type: ignore[method-assign]
        self.service._audit = mock.AsyncMock()  # type: ignore[method-assign]

        with self._service_dependencies():
            await self.service.apply_collab_update(self.workspace_id, self.file_path, "updated", self.user_id, expected_version=1)

        self.assertEqual(state.disk_content, "updated")

    async def test_snapshot_skips_adoption_when_collab_write_races_disk_read(self) -> None:
        state = self._set_state(content="old", disk_content="old", version=1)

        async def _read_during_collab_write(*_args: Any) -> tuple[str, bool, bool]:
            state.content = "collab"
            state.disk_content = "collab"
            state.version += 1
            return "old-read", True, True

        self.service._load_collab_disk_content = mock.AsyncMock(side_effect=_read_during_collab_write)  # type: ignore[method-assign]

        with self._service_dependencies():
            snapshot = await self.service.get_collab_snapshot(self.workspace_id, self.file_path, self.user_id)

        self.assertEqual((snapshot.content, snapshot.version), ("collab", 2))

    async def test_snapshot_skips_resync_while_persist_in_flight(self) -> None:
        state = self._set_state(content="new", disk_content="old", version=2)
        state.pending_persists = 1
        self.service._load_collab_disk_content = mock.AsyncMock(return_value=("external", True, True))  # type: ignore[method-assign]

        with self._service_dependencies():
            snapshot = await self.service.get_collab_snapshot(self.workspace_id, self.file_path, self.user_id)

        self.assertEqual((snapshot.content, snapshot.version), ("new", 2))
        self.service._load_collab_disk_content.assert_not_awaited()

    async def test_snapshot_does_not_adopt_missing_or_non_text_disk_content(self) -> None:
        for exists, is_utf8_text in ((False, False), (True, False)):
            with self.subTest(exists=exists, is_utf8_text=is_utf8_text):
                state = self._set_state(content="old", disk_content="old", version=1)
                self.service.ensure_workspace_preview_session = mock.AsyncMock(  # type: ignore[method-assign]
                    return_value=SimpleNamespace(provider_session_id="session-1")
                )
                self.service._runtime_provider_read_file = mock.AsyncMock(  # type: ignore[method-assign]
                    return_value={"content": "", "exists": exists, "is_utf8_text": is_utf8_text}
                )

                with self._service_dependencies():
                    snapshot = await self.service.get_collab_snapshot(self.workspace_id, self.file_path, self.user_id)

                self.assertEqual((snapshot.content, snapshot.version), ("old", 1))
                self.assertEqual(state.disk_content, "old")

    async def test_register_can_skip_disk_resync_after_snapshot(self) -> None:
        self._set_state(content="old", disk_content="old")
        client = _FakeWebSocket()
        self.service._load_collab_disk_content = mock.AsyncMock()  # type: ignore[method-assign]

        with self._service_dependencies():
            snapshot = await self.service.register_collab_client(
                self.workspace_id,
                self.file_path,
                cast(Any, client),
                self.user_id,
                skip_disk_resync=True,
            )

        self.assertEqual(snapshot.content, "old")
        self.service._load_collab_disk_content.assert_not_awaited()
