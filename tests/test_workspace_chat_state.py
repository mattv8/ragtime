"""Unit tests for build_workspace_chat_state."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest import mock

from ragtime.indexer.models import ChatMessage, ChatTask, ChatTaskStatus, Conversation, ConversationSummaryResponse
from ragtime.indexer.repository import repository
from ragtime.indexer.workspace_state import build_workspace_chat_state

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _make_summary(
    conversation_id: str = "conv-1",
    title: str = "Test Chat",
    message_count: int = 5,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation_id,
        title=title,
        model="gpt-4",
        message_count=message_count,
        total_tokens=100,
        created_at=NOW,
        updated_at=NOW,
    )


def _make_conversation(
    conversation_id: str = "conv-1",
    title: str = "Test Chat",
    messages: list[ChatMessage] | None = None,
    workspace_id: str = "ws-1",
) -> Conversation:
    return Conversation(
        id=conversation_id,
        title=title,
        model="gpt-4",
        workspace_id=workspace_id,
        messages=messages or [],
        total_tokens=100,
        created_at=NOW,
        updated_at=NOW,
    )


class BuildWorkspaceChatStateTests(unittest.IsolatedAsyncioTestCase):
    async def test_include_selected_false_skips_full_fetch(self) -> None:
        summary = _make_summary(conversation_id="conv-1", message_count=5)
        active = ChatTask(
            id="task-1",
            conversation_id="conv-1",
            status=ChatTaskStatus.running,
            user_message="hello",
            created_at=NOW,
            last_update_at=NOW,
        )

        with (
            mock.patch.object(repository, "list_conversation_summaries", mock.AsyncMock(return_value=[summary])),
            mock.patch.object(repository, "get_interrupted_conversation_ids_for_workspace", mock.AsyncMock(return_value=[])),
            mock.patch.object(repository, "get_conversation", mock.AsyncMock(return_value=None)) as get_conv_mock,
            mock.patch.object(repository, "get_active_task_for_conversation", mock.AsyncMock(return_value=active)),
            mock.patch.object(repository, "get_last_interrupted_task_for_conversation", mock.AsyncMock(return_value=None)),
        ):
            state = await build_workspace_chat_state(
                workspace_id="ws-1",
                user_id="user-1",
                is_admin=False,
                selected_conversation_id="conv-1",
                include_selected_conversation=False,
            )

        self.assertEqual(state.selected_conversation_id, "conv-1")
        self.assertIsNotNone(state.active_task)
        assert state.active_task is not None
        self.assertEqual(state.active_task.id, "task-1")
        self.assertEqual(len(state.conversations), 1)
        self.assertEqual(state.conversations[0].id, "conv-1")
        self.assertEqual(state.conversations[0].messages, [])
        get_conv_mock.assert_not_awaited()

    async def test_include_selected_true_embeds_messages(self) -> None:
        summary = _make_summary(conversation_id="conv-1", message_count=2)
        full = _make_conversation(
            conversation_id="conv-1",
            messages=[
                ChatMessage(role="user", content="hi"),
                ChatMessage(role="assistant", content="hello"),
            ],
        )

        with (
            mock.patch.object(repository, "list_conversation_summaries", mock.AsyncMock(return_value=[summary])),
            mock.patch.object(repository, "get_interrupted_conversation_ids_for_workspace", mock.AsyncMock(return_value=[])),
            mock.patch.object(repository, "get_conversation", mock.AsyncMock(return_value=full)) as get_conv_mock,
            mock.patch.object(repository, "get_active_task_for_conversation", mock.AsyncMock(return_value=None)),
            mock.patch.object(repository, "get_last_interrupted_task_for_conversation", mock.AsyncMock(return_value=None)),
        ):
            state = await build_workspace_chat_state(
                workspace_id="ws-1",
                user_id="user-1",
                is_admin=False,
                selected_conversation_id="conv-1",
                include_selected_conversation=True,
            )

        self.assertEqual(state.selected_conversation_id, "conv-1")
        self.assertEqual(len(state.conversations), 1)
        self.assertEqual(len(state.conversations[0].messages), 2)
        get_conv_mock.assert_awaited_once_with("conv-1")

    async def _build_state_for_candidate(
        self,
        candidate: Conversation,
        *,
        include_selected_conversation: bool,
    ) -> tuple[Any, mock.AsyncMock]:
        summary = _make_summary(conversation_id="conv-1")

        with (
            mock.patch.object(repository, "list_conversation_summaries", mock.AsyncMock(return_value=[summary])),
            mock.patch.object(repository, "get_interrupted_conversation_ids_for_workspace", mock.AsyncMock(return_value=[])),
            mock.patch.object(repository, "get_conversation", mock.AsyncMock(return_value=candidate)) as get_conv_mock,
            mock.patch.object(repository, "get_active_task_for_conversation", mock.AsyncMock(return_value=None)),
            mock.patch.object(repository, "get_last_interrupted_task_for_conversation", mock.AsyncMock(return_value=None)),
        ):
            state = await build_workspace_chat_state(
                workspace_id="ws-1",
                user_id="user-1",
                is_admin=False,
                selected_conversation_id="conv-2",
                include_selected_conversation=include_selected_conversation,
            )

        return state, get_conv_mock

    async def test_candidate_validation_reuses_get_conversation(self) -> None:
        candidate = _make_conversation(conversation_id="conv-2", workspace_id="ws-1")

        state, get_conv_mock = await self._build_state_for_candidate(candidate, include_selected_conversation=True)

        self.assertEqual(state.selected_conversation_id, "conv-2")
        self.assertEqual(len(state.conversations), 2)
        get_conv_mock.assert_awaited_once_with("conv-2")

    async def test_include_selected_false_validates_candidate_without_embedding_messages(self) -> None:
        candidate = _make_conversation(
            conversation_id="conv-2",
            workspace_id="ws-1",
            messages=[
                ChatMessage(role="user", content="heavy user message"),
                ChatMessage(role="assistant", content="heavy assistant message"),
            ],
        )

        state, get_conv_mock = await self._build_state_for_candidate(candidate, include_selected_conversation=False)

        self.assertEqual(state.selected_conversation_id, "conv-2")
        self.assertEqual([conversation.id for conversation in state.conversations], ["conv-1", "conv-2"])
        self.assertEqual(state.conversations[1].messages, [])
        get_conv_mock.assert_awaited_once_with("conv-2")


if __name__ == "__main__":
    unittest.main()
