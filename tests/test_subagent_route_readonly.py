import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer.models import (
    CompactConversationRequest,
    Conversation,
    RetryVisualizationRequest,
)
from ragtime.indexer.routes import (
    compact_conversation,
    retry_visualization,
    truncate_conversation,
)


def _make_subagent_conversation(parent_conversation_id: str = "parent-1") -> Conversation:
    return Conversation(
        id="sub-1",
        title="Subagent",
        model="openai::gpt-4.1",
        user_id="user-1",
        workspace_id="workspace-1",
        parent_conversation_id=parent_conversation_id,
        username="user-1",
        messages=[],
        total_tokens=0,
        disabled_builtin_tool_ids=[],
        tool_selection_mode="default_all",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class SubAgentRouteReadOnlyTests(unittest.IsolatedAsyncioTestCase):
    def _make_user(self, role: str = "editor") -> SimpleNamespace:
        return SimpleNamespace(id="user-1", role=role)

    def _mock_access(self, conv: Conversation) -> SimpleNamespace:
        return SimpleNamespace(
            enforce_workspace_role=mock.AsyncMock(),
        )

    def _mock_repository(self, conv: Conversation) -> SimpleNamespace:
        return SimpleNamespace(
            check_conversation_access=mock.AsyncMock(return_value=True),
            get_conversation=mock.AsyncMock(return_value=conv),
        )

    async def test_truncate_conversation_rejects_subagent(self) -> None:
        conv = _make_subagent_conversation()
        with (
            mock.patch("ragtime.indexer.routes.repository", self._mock_repository(conv)),
            mock.patch("ragtime.indexer.routes.userspace_service", self._mock_access(conv)),
        ):
            with self.assertRaisesRegex(HTTPException, "read-only"):
                await truncate_conversation(
                    conversation_id="sub-1",
                    keep_count=1,
                    workspace_id="workspace-1",
                    user=cast(Any, self._make_user()),
                )

    async def test_compact_conversation_rejects_subagent(self) -> None:
        conv = _make_subagent_conversation()
        with (
            mock.patch("ragtime.indexer.routes.repository", self._mock_repository(conv)),
            mock.patch("ragtime.indexer.routes.userspace_service", self._mock_access(conv)),
        ):
            with self.assertRaisesRegex(HTTPException, "read-only"):
                await compact_conversation(
                    conversation_id="sub-1",
                    request=CompactConversationRequest(
                        replace_message_id=None,
                        replace_message_index=None,
                        create_revision_branch=False,
                    ),
                    workspace_id="workspace-1",
                    user=cast(Any, self._make_user()),
                )

    async def test_retry_visualization_rejects_subagent(self) -> None:
        conv = _make_subagent_conversation()
        with (
            mock.patch("ragtime.indexer.routes.repository", self._mock_repository(conv)),
            mock.patch("ragtime.indexer.routes.userspace_service", self._mock_access(conv)),
        ):
            with self.assertRaisesRegex(HTTPException, "read-only"):
                await retry_visualization(
                    conversation_id="sub-1",
                    request=RetryVisualizationRequest(
                        tool_type="datatable",
                        message_id=None,
                        message_index=0,
                        event_index=0,
                    ),
                    workspace_id="workspace-1",
                    user=cast(Any, self._make_user()),
                )


if __name__ == "__main__":
    unittest.main()
