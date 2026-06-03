import unittest
from types import SimpleNamespace
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ragtime.indexer.models import ChatMessage
from ragtime.indexer.repository import _estimate_conversation_tokens, _estimate_effective_conversation_tokens
from ragtime.indexer.routes import _build_chat_history_for_conversation, _find_compaction_split_index
from ragtime.rag.components import RAGComponents


def _message(role: str, content: str) -> ChatMessage:
    return ChatMessage(role=role, content=content)


class ConversationCompactionTests(unittest.IsolatedAsyncioTestCase):
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

    def test_find_compaction_split_anchors_at_tail(self) -> None:
        messages = []
        for idx in range(6):
            messages.append(_message("user", f"user {idx}"))
            messages.append(_message("assistant", f"assistant {idx}"))

        split_index, summarized = _find_compaction_split_index(messages, keep_recent_pairs=2)

        self.assertEqual(split_index, len(messages))
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
                "user 4",
                "assistant 4",
                "user 5",
                "assistant 5",
            ],
        )

    def test_find_compaction_split_only_summarizes_after_last_marker(self) -> None:
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

        self.assertEqual(split_index, len(messages))
        self.assertEqual(
            [message.content for message in summarized],
            [
                "fresh 1",
                "fresh reply 1",
                "fresh 2",
                "fresh reply 2",
                "fresh 3",
                "fresh reply 3",
            ],
        )

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
        class FakeLLM:
            async def ainvoke(self, messages):
                return AIMessage(
                    content=[
                        {"type": "reasoning", "channel": "analysis", "text": "private chain of thought"},
                        {"type": "text", "channel": "final", "text": "Continuity summary from final channel."},
                    ]
                )

        components = RAGComponents()
        resolution = SimpleNamespace(llm=FakeLLM(), provider=None, model="test-model", max_tokens=1200)
        messages = [_message("user", "Need a summary"), _message("assistant", "Working on it")]

        with (
            mock.patch.object(components, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(components, "_cap_request_llm_output_tokens", new=mock.AsyncMock(return_value=resolution)),
        ):
            summary = await components.summarize_for_compaction(messages, "test-model")

        self.assertEqual(summary, "Continuity summary from final channel.")

    async def test_summarize_for_compaction_accepts_commentary_when_final_missing(self) -> None:
        class FakeLLM:
            async def ainvoke(self, messages):
                return AIMessage(
                    content=[
                        {"type": "reasoning", "channel": "analysis", "text": "private chain of thought"},
                        {"type": "text", "channel": "commentary", "text": "Continuity summary from commentary channel."},
                    ]
                )

        components = RAGComponents()
        resolution = SimpleNamespace(llm=FakeLLM(), provider=None, model="test-model", max_tokens=1200)
        messages = [_message("user", "Need a summary"), _message("assistant", "Working on it")]

        with (
            mock.patch.object(components, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=resolution)),
            mock.patch.object(components, "_cap_request_llm_output_tokens", new=mock.AsyncMock(return_value=resolution)),
        ):
            summary = await components.summarize_for_compaction(messages, "test-model")

        self.assertEqual(summary, "Continuity summary from commentary channel.")


if __name__ == "__main__":
    unittest.main()
