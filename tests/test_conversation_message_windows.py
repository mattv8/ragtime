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
from tests.content_protection_support import use_disabled_content_protection

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
    def setUp(self) -> None:
        use_disabled_content_protection(self)

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
        sql = next(call.args[0] for call in db.query_raw.await_args_list if "WITH c AS" in call.args[0])
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
            mock.patch.object(repository, "get_conversation", mock.AsyncMock(return_value=SimpleNamespace(user_id="user-1"))) as conversation_mock,
        ):
            result = await get_conversation_latest_exchange("conversation-1", user=user)

        self.assertIs(result, expected)
        access_mock.assert_awaited_once_with("conversation-1", "user-1", is_admin=False, workspace_id=None)
        window_mock.assert_awaited_once_with("conversation-1")
        conversation_mock.assert_awaited_once_with("conversation-1")

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

    def test_preview_uses_content_event_when_content_is_null(self) -> None:
        preview = repository._window_preview({"role": "assistant", "content": None, "events": [{"type": "content", "content": "event text"}]}, max_bytes=1024)

        self.assertEqual(preview.content, "event text")
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

    async def test_oversized_sql_entry_is_deferred_without_parsing_raw_details(self) -> None:
        row = {
            "id": "conversation-1",
            "title": "Windowed chat",
            "model": "model-1",
            "created_at": NOW,
            "updated_at": NOW,
            "revision": "revision-1",
            "is_canonical": True,
            "total_message_count": 1,
            "entries": [
                {
                    "index": 0,
                    "key": "assistant-1",
                    "has_details": True,
                    "force_deferred": True,
                    "message": {"role": "assistant", "content": "readable preview", "timestamp": NOW.isoformat(), "message_id": "assistant-1"},
                }
            ],
        }
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[row]))
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            window = await repository._build_conversation_window("conversation-1", before_index=None, latest_exchange=False)

        assert window is not None
        self.assertEqual(window.entries[0].state, "deferred")
        self.assertEqual(window.entries[0].preview.content if window.entries[0].preview else None, "readable preview")
        self.assertTrue(window.entries[0].preview.has_details if window.entries[0].preview else False)
        sql = next(call.args[0] for call in db.query_raw.await_args_list if "WITH c AS" in call.args[0])
        self.assertIn("force_deferred", sql)
        self.assertIn("octet_length(e.message::text) > 262144", sql)

    async def test_full_message_read_does_not_thin_oversized_entry(self) -> None:
        row = {
            "id": "conversation-1",
            "title": "Windowed chat",
            "model": "model-1",
            "created_at": NOW,
            "updated_at": NOW,
            "revision": "revision-1",
            "is_canonical": True,
            "total_message_count": 1,
            "entries": [
                {
                    "index": 0,
                    "key": "assistant-1",
                    "has_details": True,
                    "message": {
                        "role": "assistant",
                        "content": "full detail",
                        "events": [{"type": "tool", "payload": "full"}],
                        "timestamp": NOW.isoformat(),
                        "message_id": "assistant-1",
                    },
                }
            ],
        }
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[row]))
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            entry = await repository.get_conversation_window_message("conversation-1", 0, "revision-1")

        assert entry is not None
        assert entry.message is not None
        assert entry.message.events is not None
        self.assertEqual(entry.state, "ready")
        self.assertEqual(entry.message.events[0].get("payload"), "full")
        sql = next(call.args[0] for call in db.query_raw.await_args_list if "WITH c AS" in call.args[0])
        self.assertIn("FALSE AS force_deferred", sql)

    async def test_snapshot_links_are_limited_to_ready_page_message_ids(self) -> None:
        links = [SimpleNamespace(messageId="ready-1", snapshotId="snapshot-1", restoreMessageCount=2, createdAt=NOW, updatedAt=NOW)]
        db = SimpleNamespace(conversationmessagesnapshotlink=SimpleNamespace(find_many=mock.AsyncMock(return_value=links)))
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            found = await repository.get_message_snapshot_links_for_conversation("conversation-1", message_ids=["ready-1"])
            empty = await repository.get_message_snapshot_links_for_conversation("conversation-1", message_ids=[])

        self.assertIn("ready-1", found)
        self.assertEqual(empty, {})
        db.conversationmessagesnapshotlink.find_many.assert_awaited_once_with(where={"conversationId": "conversation-1", "messageId": {"in": ["ready-1"]}})

    async def test_window_query_uses_bounded_subscripts_and_single_validation_scan(self) -> None:
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[]))
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)):
            await repository._query_conversation_window("conversation-1", before_index=100, latest_exchange=False, limit=20)

        sql = next(call.args[0] for call in db.query_raw.await_args_list if "WITH c AS" in call.args[0])
        self.assertIn("WITH ORDINALITY AS item", sql)
        self.assertIn("ORDER BY message_index DESC", sql)
        self.assertNotIn("generate_series", sql)
        # Canonicality and ordinary page selection each scan once; the third
        # occurrence is the selected-entry content-event preview fallback.
        self.assertEqual(sql.count("jsonb_array_elements"), 3)

    async def test_empty_canonical_window_has_no_entries_or_snapshot_query(self) -> None:
        row = {
            "id": "conversation-1",
            "title": "Empty",
            "model": "model-1",
            "created_at": NOW,
            "updated_at": NOW,
            "revision": "revision-1",
            "is_canonical": True,
            "total_message_count": 0,
            "entries": [],
        }
        db = SimpleNamespace(query_raw=mock.AsyncMock(return_value=[row]))
        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(repository, "get_message_snapshot_links_for_conversation", mock.AsyncMock()) as links,
        ):
            window = await repository._build_conversation_window("conversation-1", before_index=None, latest_exchange=False)

        assert window is not None
        self.assertEqual(window.entries, [])
        self.assertFalse(window.has_more)
        links.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
