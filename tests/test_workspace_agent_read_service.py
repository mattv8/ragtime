import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer.models import ChatMessage

NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def _message(role: str, content: str = "", *, events=None, message_id: str | None = None) -> ChatMessage:
    return ChatMessage(
        role=role,
        content=content,
        timestamp=NOW,
        events=events,
        message_id=message_id,
    )


class AgentTranscriptProjectionTests(unittest.TestCase):
    def test_projects_multimodal_text_and_safe_attachment_labels(self) -> None:
        from ragtime.userspace.agent_read_service import _project_user_content

        content = (
            '[{"type":"text","text":"Review this"},'
            '{"type":"image_url","image_url":{"url":"data:image/png;base64,SECRET"}},'
            '{"type":"file","filename":"../../reports\\nsecret.csv","file_id":"private-id"}]'
        )
        projected = _project_user_content(content)
        self.assertIn("Review this", projected)
        self.assertIn("[Image attachment]", projected)
        self.assertIn("secret.csv", projected)
        self.assertNotIn("SECRET", projected)
        self.assertNotIn("private-id", projected)
        self.assertNotIn("../", projected)

    def test_assistant_events_return_text_and_tool_inputs_only(self) -> None:
        from ragtime.userspace.agent_read_service import _project_visible_messages

        messages = [
            _message(
                "assistant",
                "duplicate final text",
                events=[
                    {"type": "content", "content": "I searched."},
                    {
                        "type": "tool",
                        "tool": "search_userspace_code",
                        "input": {"query": "revenue"},
                        "output": "SECRET OUTPUT",
                        "connection": {"password": "SECRET"},
                        "presentation": {"kind": "table"},
                        "mcp": {"response": "SECRET"},
                    },
                    {"type": "reasoning", "content": "private reasoning"},
                    {"type": "error", "content": "private error"},
                ],
            )
        ]
        projected = _project_visible_messages(messages)
        self.assertEqual(projected[0]["content"], "I searched.")
        self.assertEqual(
            projected[0]["tool_calls"],
            [{"tool": "search_userspace_code", "input": {"query": "revenue"}}],
        )
        dumped = str(projected)
        self.assertNotIn("duplicate final text", dumped)
        self.assertNotIn("SECRET", dumped)
        self.assertNotIn("private reasoning", dumped)
        self.assertNotIn("private error", dumped)

    def test_filters_internal_roles_omits_empty_assistant_and_assigns_visible_indexes(self) -> None:
        from ragtime.userspace.agent_read_service import _project_visible_messages

        projected = _project_visible_messages(
            [
                _message("system", "secret"),
                _message("user", "first", message_id="m1"),
                _message("compaction", "summary"),
                _message("assistant", "", message_id="m-empty"),
                _message("assistant", "second", message_id="m2"),
            ]
        )
        self.assertEqual([item["message_index"] for item in projected], [0, 1])
        self.assertEqual([item["content"] for item in projected], ["first", "second"])


class AgentTranscriptPagingTests(unittest.TestCase):
    def setUp(self) -> None:
        from ragtime.userspace.agent_read_service import _page_visible_messages

        self.page = _page_visible_messages
        self.messages = [{"message_index": i, "content": str(i)} for i in range(6)]

    def test_forward_pages_after_cursor(self) -> None:
        first = self.page(self.messages, direction="forward", cursor=None, limit=2)
        second = self.page(
            self.messages,
            direction="forward",
            cursor=first["next_cursor"],
            limit=2,
        )
        self.assertEqual([m["message_index"] for m in first["messages"]], [0, 1])
        self.assertEqual([m["message_index"] for m in second["messages"]], [2, 3])

    def test_backward_pages_before_cursor_but_returns_chronologically(self) -> None:
        first = self.page(self.messages, direction="backward", cursor=None, limit=2)
        second = self.page(
            self.messages,
            direction="backward",
            cursor=first["previous_cursor"],
            limit=2,
        )
        self.assertEqual([m["message_index"] for m in first["messages"]], [4, 5])
        self.assertEqual([m["message_index"] for m in second["messages"]], [2, 3])

    def test_out_of_range_cursor_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self.page(self.messages, direction="forward", cursor=99, limit=2)
        self.assertEqual(ctx.exception.status_code, 400)


class AgentReadServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_enforces_viewer_and_returns_bounded_top_level_summaries(self) -> None:
        from ragtime.userspace import agent_read_service as module

        summaries = [
            SimpleNamespace(
                id="c1",
                title="Chat 1",
                message_count=2,
                created_at=NOW,
                updated_at=NOW,
                subagent_conversation_ids=["child-1"],
            ),
            SimpleNamespace(
                id="c2",
                title="Chat 2",
                message_count=3,
                created_at=NOW,
                updated_at=NOW,
                subagent_conversation_ids=[],
            ),
        ]
        enforce = mock.AsyncMock()
        with (
            mock.patch.object(module.agent_read_service, "_enforce_viewer", enforce),
            mock.patch.object(
                module.repository,
                "list_conversation_summaries",
                mock.AsyncMock(return_value=summaries),
            ),
        ):
            result = await module.agent_read_service.list_conversations("ws-1", "user-1", is_admin=False, offset=1, limit=1)
        enforce.assert_awaited_once_with("ws-1", "user-1", is_admin=False)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["offset"], 1)
        self.assertEqual(result["limit"], 1)
        self.assertEqual(len(result["conversations"]), 1)
        self.assertEqual(result["conversations"][0]["id"], "c2")
        self.assertEqual(result["conversations"][0]["subagent_conversation_ids"], [])

    async def test_transcript_hides_inaccessible_conversation_as_not_found(self) -> None:
        from ragtime.userspace import agent_read_service as module

        with (
            mock.patch.object(module.agent_read_service, "_enforce_viewer", mock.AsyncMock()),
            mock.patch.object(
                module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=False),
            ),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock()) as get_conversation,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await module.agent_read_service.get_conversation_transcript(
                    "ws-1",
                    "user-1",
                    "other-workspace-chat",
                    is_admin=False,
                    direction="forward",
                    cursor=None,
                    limit=20,
                )
        self.assertEqual(ctx.exception.status_code, 404)
        get_conversation.assert_not_awaited()

    async def test_child_transcript_is_read_only(self) -> None:
        from ragtime.userspace import agent_read_service as module

        conversation = SimpleNamespace(
            id="child-1",
            title="Subagent",
            created_at=NOW,
            updated_at=NOW,
            parent_conversation_id="parent-1",
            subagent_conversation_ids=[],
            messages=[_message("user", "assay")],
        )
        with (
            mock.patch.object(module.agent_read_service, "_enforce_viewer", mock.AsyncMock()),
            mock.patch.object(
                module.repository,
                "check_conversation_access",
                mock.AsyncMock(return_value=True),
            ),
            mock.patch.object(module.repository, "get_conversation", mock.AsyncMock(return_value=conversation)),
        ):
            result = await module.agent_read_service.get_conversation_transcript(
                "ws-1",
                "user-1",
                "child-1",
                is_admin=False,
                direction="forward",
                cursor=None,
                limit=20,
            )
        self.assertTrue(result["conversation"]["read_only"])

    async def test_code_search_enforces_acl_and_sanitizes_results(self) -> None:
        from ragtime.userspace import agent_read_service as module

        raw = {
            "status": "ready",
            "workspace_id": "ws-1",
            "mode": "hybrid",
            "query": "revenue",
            "result_count": 2,
            "results": [
                {
                    "path": "app.py",
                    "chunk_index": 4,
                    "snippet": "def revenue():\n" + ("x" * 2000),
                    "score": 0.9,
                    "source": "semantic",
                    "metadata": {"secret": True},
                },
                {
                    "path": "symbols.py",
                    "snippet": "def revenue_symbol():",
                    "score": 1.0,
                    "source": "symbol",
                    "kind": "function",
                    "symbol": "revenue_symbol",
                    "workspace_id": "ws-1",
                    "connection": {"token": "secret"},
                },
            ],
            "error": "raw backend failure",
            "last_error": "do not leak",
        }
        with (
            mock.patch.object(module.agent_read_service, "_enforce_viewer", mock.AsyncMock()) as enforce,
            mock.patch.object(
                module.workspace_code_index_service,
                "search_workspace_code_read_only",
                mock.AsyncMock(return_value=raw),
            ) as search,
        ):
            result = await module.agent_read_service.search_code(
                "ws-1",
                "user-1",
                is_admin=False,
                query=" revenue ",
                mode="hybrid",
                max_results=8,
                max_chars_per_result=20,
            )
        enforce.assert_awaited_once_with("ws-1", "user-1", is_admin=False)
        search.assert_awaited_once_with(
            workspace_id="ws-1",
            query="revenue",
            mode="hybrid",
            max_results=8,
            max_chars_per_result=20,
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["query"], "revenue")
        self.assertEqual(result["result_count"], 2)
        self.assertEqual(
            result["results"][0],
            {
                "path": "app.py",
                "snippet": "def revenue():\n" + ("x" * 5),
                "score": 0.9,
                "source": "semantic",
            },
        )
        self.assertEqual(
            result["results"][1],
            {
                "path": "symbols.py",
                "snippet": "def revenue_symbol()",
                "score": 1.0,
                "source": "symbol",
                "kind": "function",
                "symbol": "revenue_symbol",
            },
        )
        self.assertNotIn("workspace_id", result)
        self.assertNotIn("error", result)

    async def test_code_search_rejects_blank_query_without_calling_index_search(self) -> None:
        from ragtime.userspace import agent_read_service as module

        with (
            mock.patch.object(module.agent_read_service, "_enforce_viewer", mock.AsyncMock()),
            mock.patch.object(
                module.workspace_code_index_service,
                "search_workspace_code_read_only",
                mock.AsyncMock(),
            ) as search,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await module.agent_read_service.search_code(
                    "ws-1",
                    "user-1",
                    is_admin=False,
                    query="   ",
                    mode="hybrid",
                    max_results=8,
                    max_chars_per_result=1200,
                )
        self.assertEqual(ctx.exception.status_code, 400)
        search.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
