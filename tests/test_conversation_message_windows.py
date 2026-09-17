import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.indexer.models import Conversation
from ragtime.indexer.repository import ConversationWindowStaleError, repository
from ragtime.indexer.routes import get_conversation_latest_exchange

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def _user() -> User:
    return User(
        id="user-1",
        username="user",
        authProvider=AuthProvider.local,
        cachedGroups=cast(Json, "[]"),
        role=UserRole.user,
        roleManuallySet=False,
        createdAt=NOW,
        updatedAt=NOW,
        securityGeneration=0,
    )


class ConversationMessageWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_latest_exchange_returns_latest_user_and_final_message_at_absolute_indexes(self) -> None:
        rows = [
            {
                "id": "conversation-1",
                "title": "Windowed chat",
                "model": "model-1",
                "user_id": "user-1",
                "workspace_id": None,
                "total_tokens": 42,
                "active_task_id": "task-1",
                "active_branch_id": "branch-1",
                "disabled_builtin_tool_ids": ["builtin-1"],
                "parent_conversation_id": "parent-1",
                "created_at": NOW,
                "updated_at": NOW,
                "revision": "revision-1",
                "total_message_count": 4,
                "entries": [
                    {"index": 1, "message": {"role": "user", "content": "latest question", "timestamp": NOW.isoformat(), "message_id": "u-1"}},
                    {
                        "index": 3,
                        "key": "a-1",
                        "has_details": True,
                        "message": {
                            "role": "assistant",
                            "content": '[{"type":"image_url","image_url":"data:image/png;base64,secret"},{"type":"text","text":"latest answer"}]',
                            "timestamp": NOW.isoformat(),
                            "message_id": "a-1",
                        },
                    },
                ],
            }
        ]
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=rows))

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            window = await repository.get_latest_conversation_exchange("conversation-1")

        self.assertIsNotNone(window)
        assert window is not None
        self.assertEqual([entry.index for entry in window.entries], [1, 3])
        self.assertTrue(all(entry.state == "deferred" and entry.preview is not None for entry in window.entries))
        self.assertEqual([entry.preview.content for entry in window.entries if entry.preview], ["latest question", "latest answer"])
        self.assertTrue(window.entries[-1].preview.has_details if window.entries[-1].preview else False)
        self.assertNotIn("data:image", window.entries[-1].preview.content if window.entries[-1].preview else "")
        self.assertIsInstance(window.next_cursor, str)
        self.assertEqual(repository.decode_conversation_window_cursor(window.next_cursor or "")["before_index"], 4)
        self.assertTrue(window.has_more)
        self.assertEqual(window.conversation.total_tokens, 42)
        self.assertEqual(window.conversation.active_task_id, "task-1")
        self.assertEqual(window.conversation.disabled_builtin_tool_ids, ["builtin-1"])
        self.assertTrue(window.conversation.is_subagent)
        self.assertTrue(window.conversation.read_only)
        sql = db.query_raw.await_args.args[0]
        self.assertIn("jsonb_array_elements", sql)
        self.assertIn("sha256", sql)
        self.assertNotIn("digest", sql)
        self.assertIn("SELECT c.id, c.title", sql)

    async def test_latest_exchange_route_reuses_access_gate_before_window_query(self) -> None:
        expected = object()
        user = _user()
        with (
            mock.patch.object(repository, "check_conversation_access", mock.AsyncMock(return_value=True)) as access_mock,
            mock.patch.object(repository, "get_latest_conversation_exchange", mock.AsyncMock(return_value=expected)) as window_mock,
        ):
            result = await get_conversation_latest_exchange("conversation-1", user=user)

        self.assertIs(result, expected)
        access_mock.assert_awaited_once_with("conversation-1", "user-1", is_admin=False, workspace_id=None)
        window_mock.assert_awaited_once_with("conversation-1")

    async def test_stale_cursor_rejects_canonical_to_legacy_transition_before_hydration(self) -> None:
        stale_cursor = repository._encode_conversation_window_cursor("conversation-1", None, "old-revision", 2)
        db = SimpleNamespace(
            query_raw=mock.AsyncMock(
                return_value=[
                    {
                        "id": "conversation-1",
                        "revision": "new-revision",
                        "active_branch_id": None,
                        "is_canonical": False,
                    }
                ]
            )
        )

        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(repository, "get_conversation", mock.AsyncMock(side_effect=AssertionError("stale reads must not hydrate legacy data"))),
        ):
            with self.assertRaises(ConversationWindowStaleError):
                await repository.get_conversation_message_window("conversation-1", stale_cursor, 20)

    async def test_cursor_beyond_current_message_count_is_rejected(self) -> None:
        cursor = repository._encode_conversation_window_cursor("conversation-1", None, "revision-1", 3)
        db = SimpleNamespace(
            query_raw=mock.AsyncMock(
                return_value=[
                    {
                        "id": "conversation-1",
                        "revision": "revision-1",
                        "active_branch_id": None,
                        "is_canonical": True,
                        "total_message_count": 2,
                    }
                ]
            )
        )

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaisesRegex(ValueError, "Invalid conversation message cursor"):
                await repository.get_conversation_message_window("conversation-1", cursor, 20)

    async def test_legacy_window_marks_subagent_metadata_read_only(self) -> None:
        legacy = Conversation(
            id="conversation-1",
            title="Legacy child",
            model="model-1",
            parent_conversation_id="parent-1",
            created_at=NOW,
            updated_at=NOW,
        )
        with mock.patch.object(repository, "get_conversation", mock.AsyncMock(return_value=legacy)):
            window = await repository._legacy_conversation_window("conversation-1", "revision-1")

        self.assertIsNotNone(window)
        assert window is not None
        self.assertTrue(window.conversation.is_subagent)
        self.assertTrue(window.conversation.read_only)

    def test_preview_removes_multimodal_data_urls_and_keeps_detail_signal(self) -> None:
        preview = repository._window_preview(
            {
                "role": "user",
                "content": '[{"type":"image_url","image_url":"data:image/png;base64,secret"},{"type":"text","text":"visible text"}]',
                "timestamp": NOW.isoformat(),
            },
            max_bytes=16 * 1024,
            has_details=True,
        )

        self.assertEqual(preview.content, "visible text")
        self.assertNotIn("data:image", preview.content)
        self.assertTrue(preview.has_details)

    async def test_latest_exchange_uses_projected_content_event_preview(self) -> None:
        db = SimpleNamespace(
            query_raw=mock.AsyncMock(
                return_value=[
                    {
                        "id": "conversation-1",
                        "title": "Event-only answer",
                        "model": "model-1",
                        "created_at": NOW,
                        "updated_at": NOW,
                        "revision": "revision-1",
                        "total_message_count": 1,
                        "entries": [
                            {
                                "index": 0,
                                "key": "a-1",
                                "has_details": True,
                                # This is the SQL-projected readable content;
                                # raw events/tool output are intentionally absent.
                                "message": {
                                    "role": "assistant",
                                    "content": "event-only final",
                                    "timestamp": NOW.isoformat(),
                                    "message_id": "a-1",
                                },
                            }
                        ],
                    }
                ]
            )
        )

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            window = await repository.get_latest_conversation_exchange("conversation-1")

        self.assertIsNotNone(window)
        assert window is not None
        preview = window.entries[0].preview
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.content, "event-only final")
        self.assertTrue(preview.has_details)


if __name__ == "__main__":
    unittest.main()
