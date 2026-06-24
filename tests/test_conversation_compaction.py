import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ragtime.indexer.models import ChatMessage, ConversationBranchKind
from ragtime.indexer.background_tasks import _find_compaction_split_index_with_fallback
from ragtime.indexer.repository import (
    IndexerRepository,
    _estimate_conversation_tokens,
    _estimate_effective_conversation_tokens,
)
from ragtime.indexer.routes import (
    _build_chat_history_for_conversation,
    _find_compaction_split_index,
)
from ragtime.rag.components import RAGComponents


def _message(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class _ChannelContentLLM:
    def __init__(self, channel: str, text: str) -> None:
        self._channel = channel
        self._text = text

    async def ainvoke(self, _messages: Any) -> AIMessage:
        return AIMessage(
            content=[
                {"type": "reasoning", "channel": "analysis", "text": "private chain of thought"},
                {"type": "text", "channel": self._channel, "text": self._text},
            ]
        )


class _FakeConversationDelegate:
    def __init__(self, initial: Any, updated: Any) -> None:
        self._initial = initial
        self._updated = updated
        self.find_unique_calls = 0

    async def find_unique(self, **_kwargs: Any) -> Any:
        self.find_unique_calls += 1
        return self._initial if self.find_unique_calls == 1 else self._updated


class _FakeCompactionTransaction:
    def __init__(self, initial: Any, updated: Any) -> None:
        self.conversation = _FakeConversationDelegate(initial, updated)
        self.executed_sql: list[str] = []

    async def execute_raw(self, sql: str) -> int:
        self.executed_sql.append(sql)
        return 1


class _FakeTransactionContext:
    def __init__(self, tx: _FakeCompactionTransaction) -> None:
        self._tx = tx

    async def __aenter__(self) -> _FakeCompactionTransaction:
        return self._tx

    async def __aexit__(self, *_args: Any) -> bool:
        return False


class _FakeCompactionDb:
    def __init__(self, tx: _FakeCompactionTransaction) -> None:
        self._tx = tx

    def tx(self) -> _FakeTransactionContext:
        return _FakeTransactionContext(self._tx)


class ConversationCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def _summarize_with_channel_content(self, channel: str, text: str) -> str:
        components = RAGComponents()
        resolution = SimpleNamespace(llm=_ChannelContentLLM(channel, text), provider=None, model="test-model", max_tokens=1200)
        messages = [_message("user", "Need a summary"), _message("assistant", "Working on it")]

        with (
            mock.patch.object(components, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(components, "_cap_request_llm_output_tokens", new=mock.AsyncMock(return_value=resolution)),
        ):
            return await components.summarize_for_compaction(messages, "test-model")

    @staticmethod
    def _repo_with_fake_compaction(
        messages: list[dict[str, Any]],
        active_task_id: str | None,
    ) -> tuple[IndexerRepository, _FakeCompactionTransaction]:
        initial = SimpleNamespace(messages=messages, activeTaskId=active_task_id)
        updated = SimpleNamespace(messages=[*messages], activeTaskId=active_task_id)
        fake_tx = _FakeCompactionTransaction(initial, updated)
        return IndexerRepository(), fake_tx

    def test_effective_token_count_starts_at_latest_compaction_marker(self) -> None:
        messages = [
            {"role": "user", "content": "old raw text " * 100},
            {"role": "assistant", "content": "old assistant text " * 100},
            {"role": "compaction", "content": "compact summary"},
            {"role": "user", "content": "recent user"},
            {"role": "assistant", "content": "recent assistant"},
        ]

        full_tokens = _estimate_conversation_tokens(messages)
        effective_tokens = _estimate_effective_conversation_tokens(messages)

        self.assertLess(effective_tokens, full_tokens)
        self.assertEqual(effective_tokens, _estimate_conversation_tokens(messages[2:]))

    def test_find_compaction_split_keeps_recent_user_turns(self) -> None:
        messages = []
        for idx in range(6):
            messages.append(_message("user", f"user {idx}"))
            messages.append(_message("assistant", f"assistant {idx}"))

        split_index, summarized = _find_compaction_split_index(messages, keep_recent_pairs=2)

        self.assertEqual(split_index, 8)
        self.assertEqual(
            [message.content for message in summarized],
            [
                "user 0",
                "assistant 0",
                "user 1",
                "assistant 1",
                "user 2",
                "assistant 2",
                "user 3",
                "assistant 3",
            ],
        )

    def test_find_compaction_split_includes_previous_marker_summary(self) -> None:
        messages = [
            _message("user", "old user"),
            _message("assistant", "old assistant"),
            _message("compaction", "first summary"),
            _message("user", "fresh 1"),
            _message("assistant", "fresh reply 1"),
            _message("user", "fresh 2"),
            _message("assistant", "fresh reply 2"),
            _message("user", "fresh 3"),
            _message("assistant", "fresh reply 3"),
        ]

        split_index, summarized = _find_compaction_split_index(messages, keep_recent_pairs=1)

        self.assertEqual(split_index, 7)
        self.assertEqual(
            [message.content for message in summarized],
            [
                "first summary",
                "fresh 1",
                "fresh reply 1",
                "fresh 2",
                "fresh reply 2",
            ],
        )

    def test_find_compaction_split_requires_history_before_recent_tail(self) -> None:
        messages = [
            _message("user", "recent 1"),
            _message("assistant", "reply 1"),
            _message("user", "recent 2"),
            _message("assistant", "reply 2"),
        ]

        split_index, summarized = _find_compaction_split_index(messages, keep_recent_pairs=4)

        self.assertEqual(split_index, len(messages))
        self.assertEqual(summarized, [])

    def test_find_compaction_split_fallback_reduces_recent_tail(self) -> None:
        messages = [_message("compaction", "previous summary")]
        for idx in range(4):
            messages.append(_message("user", f"user {idx}"))
            messages.append(_message("assistant", f"assistant {idx}"))

        split_index, summarized = _find_compaction_split_index_with_fallback(messages, preferred_keep_recent_pairs=4)

        self.assertEqual(split_index, 3)
        self.assertEqual(
            [message.content for message in summarized],
            ["previous summary", "user 0", "assistant 0"],
        )

    def test_find_compaction_split_fallback_can_preserve_no_recent_tail(self) -> None:
        messages = [
            _message("compaction", "previous summary"),
            _message("user", "only new user"),
            _message("assistant", "only new assistant"),
        ]

        split_index, summarized = _find_compaction_split_index_with_fallback(messages, preferred_keep_recent_pairs=4)

        self.assertEqual(split_index, len(messages))
        self.assertEqual(
            [message.content for message in summarized],
            ["previous summary", "only new user", "only new assistant"],
        )

    def test_effective_token_count_matches_truncated_tool_history(self) -> None:
        full_output = "tool output " * 1000
        messages = [
            {
                "role": "assistant",
                "content": "",
                "events": [
                    {"type": "tool", "tool": "example", "input": {"query": "status"}, "output": full_output},
                ],
            }
        ]

        provider_history_tokens = _estimate_conversation_tokens(messages)
        raw_output_tokens = _estimate_conversation_tokens([{"role": "assistant", "content": full_output}])

        self.assertLess(provider_history_tokens, raw_output_tokens)

    async def test_build_chat_history_resets_at_compaction_marker(self) -> None:
        messages = [
            _message("user", "old user should be omitted"),
            _message("assistant", "old assistant should be omitted"),
            _message("compaction", "summary of old thread"),
            _message("user", "recent user"),
            _message("assistant", "recent assistant"),
        ]

        history = await _build_chat_history_for_conversation(
            messages,
            conversation_id="conv-1",
            user_id="user-1",
            workspace_id=None,
            model_id="test-model",
        )

        self.assertEqual(len(history), 3)
        self.assertIsInstance(history[0], SystemMessage)
        self.assertIn("summary of old thread", str(history[0].content))
        self.assertNotIn("old user should be omitted", "\n".join(str(message.content) for message in history))
        self.assertIsInstance(history[1], HumanMessage)
        self.assertEqual(history[1].content, "recent user")
        self.assertIsInstance(history[2], AIMessage)
        self.assertEqual(history[2].content, "recent assistant")

    async def test_compaction_formatter_replaces_inline_image_data_url(self) -> None:
        components = RAGComponents()
        image_payload = "data:image/png;base64," + ("a" * 120)
        message = _message(
            "user",
            '[{"type":"text","text":"please inspect this"},{"type":"image_url","image_url":{"url":"' + image_payload + '","detail":"auto"}}]',
        )

        formatted = await components._format_message_for_compaction(message, 0, None, "test-model")

        self.assertIn("please inspect this", formatted)
        self.assertIn("Image attachment", formatted)
        self.assertIn("image/png", formatted)
        self.assertNotIn("data:image", formatted)
        self.assertNotIn("base64", formatted.lower())
        self.assertNotIn("a" * 80, formatted)

    async def test_compaction_formatter_includes_vision_image_summary(self) -> None:
        class FakeVisionModel:
            async def ainvoke(self, messages):
                return SimpleNamespace(content="Image description: screenshot of settings.\nVisible text: Save changes")

        components = RAGComponents()
        message = _message(
            "user",
            '[{"type":"image_url","image_url":{"url":"data:image/png;base64,abc123","detail":"auto"}}]',
        )

        formatted = await components._format_message_for_compaction(message, 0, FakeVisionModel(), "test-model")

        self.assertIn("Image description: screenshot of settings", formatted)
        self.assertIn("Visible text: Save changes", formatted)
        self.assertNotIn("data:image", formatted)
        self.assertNotIn("abc123", formatted)

    async def test_summarize_for_compaction_prefers_final_over_analysis_blocks(self) -> None:
        summary = await self._summarize_with_channel_content("final", "Continuity summary from final channel.")

        self.assertEqual(summary, "Continuity summary from final channel.")

    async def test_summarize_for_compaction_accepts_commentary_when_final_missing(self) -> None:
        summary = await self._summarize_with_channel_content("commentary", "Continuity summary from commentary channel.")

        self.assertEqual(summary, "Continuity summary from commentary channel.")

    async def test_compaction_persistence_uses_postgres_side_json_updates(self) -> None:
        large_payload = "large-payload-" + ("x" * 1000)
        messages = [
            {"role": "user", "content": large_payload, "message_id": "m1"},
            {"role": "assistant", "content": "reply", "message_id": "tail-id"},
        ]
        repo, fake_tx = self._repo_with_fake_compaction(messages, "task-1")

        with (
            mock.patch.object(
                repo,
                "_get_db",
                new=mock.AsyncMock(return_value=_FakeCompactionDb(fake_tx)),
            ),
            mock.patch.object(repo, "_prisma_conversation_to_model", return_value="converted") as convert,
        ):
            result = await repo.compact_conversation(
                "conv-1",
                len(messages),
                "summary",
                expected_message_count=len(messages),
                expected_tail_message_id="tail-id",
                expected_active_task_id="task-1",
                snapshot_branch_kind=ConversationBranchKind.REPLAY,
                snapshot_user_id="user-1",
            )

        self.assertEqual(result, "converted")
        self.assertEqual(convert.call_count, 1)
        self.assertEqual(len(fake_tx.executed_sql), 2)
        branch_sql, update_sql = fake_tx.executed_sql
        self.assertIn("INSERT INTO conversation_branches", branch_sql)
        self.assertIn("SELECT", branch_sql)
        self.assertIn("messages,", branch_sql)
        self.assertIn("messages || jsonb_build_array", update_sql)
        self.assertNotIn(large_payload, branch_sql + update_sql)

    async def test_compaction_marker_edit_uses_jsonb_set(self) -> None:
        large_payload = "large-marker-edit-payload-" + ("x" * 1000)
        messages = [
            {"role": "user", "content": large_payload, "message_id": "m1"},
            {"role": "compaction", "content": "old summary", "message_id": "marker-1"},
            {"role": "assistant", "content": "reply", "message_id": "m2"},
        ]
        repo, fake_tx = self._repo_with_fake_compaction(messages, None)

        with (
            mock.patch.object(
                repo,
                "_get_db",
                new=mock.AsyncMock(return_value=_FakeCompactionDb(fake_tx)),
            ),
            mock.patch.object(repo, "_prisma_conversation_to_model", return_value="converted"),
        ):
            result = await repo.update_compaction_marker_summary(
                "conv-1",
                "new summary",
                message_id="marker-1",
            )

        self.assertEqual(result, "converted")
        self.assertEqual(len(fake_tx.executed_sql), 1)
        update_sql = fake_tx.executed_sql[0]
        self.assertIn("jsonb_set", update_sql)
        self.assertIn("marker-1", update_sql)
        self.assertNotIn(large_payload, update_sql)

    async def test_compaction_retry_can_move_tail_marker_before_recent_turns(self) -> None:
        messages = [
            {"role": "compaction", "content": "previous summary", "message_id": "marker-old"},
            {"role": "user", "content": "older user", "message_id": "m1"},
            {"role": "assistant", "content": "older reply", "message_id": "m2"},
            {"role": "user", "content": "recent user", "message_id": "m3"},
            {"role": "assistant", "content": "recent reply", "message_id": "m4"},
            {"role": "compaction", "content": "bad tail summary", "message_id": "marker-bad"},
        ]
        repo, fake_tx = self._repo_with_fake_compaction(messages, "task-1")

        with (
            mock.patch.object(
                repo,
                "_get_db",
                new=mock.AsyncMock(return_value=_FakeCompactionDb(fake_tx)),
            ),
            mock.patch.object(repo, "_prisma_conversation_to_model", return_value="converted"),
        ):
            result = await repo.compact_conversation(
                "conv-1",
                3,
                "stitched summary",
                expected_message_count=len(messages),
                expected_tail_message_id="marker-bad",
                expected_active_task_id="task-1",
                replace_message_id="marker-bad",
            )

        self.assertEqual(result, "converted")
        self.assertEqual(len(fake_tx.executed_sql), 1)
        update_sql = fake_tx.executed_sql[0]
        self.assertIn("messages - 5", update_sql)
        self.assertIn("jsonb_insert", update_sql)
        self.assertIn("ARRAY['3']", update_sql)
        self.assertIn("marker-bad", update_sql)


if __name__ == "__main__":
    unittest.main()
