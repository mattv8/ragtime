"""Route contract tests for the subagent conversation summaries endpoint."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma.models import User

from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import ConversationSummaryResponse
from ragtime.indexer.repository import repository

NOW = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)


def _make_user(user_id: str = "user-1", role: str = "user") -> User:
    """Route only reads user.id and user.role; a namespace stands in for prisma User."""
    return cast(User, SimpleNamespace(id=user_id, role=role))


def _make_summary(conversation_id: str = "child-1") -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=conversation_id,
        title="Subagent child",
        model="gpt-4",
        message_count=3,
        total_tokens=50,
        created_at=NOW,
        updated_at=NOW,
    )


class SubagentSummariesRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_id_is_passed_to_access_check(self) -> None:
        """Workspace-linked parents must be resolvable when workspace_id is provided.

        check_conversation_access rejects workspace-linked conversations unless the
        matching workspace_id is passed through, so the route must forward it.
        """
        user = _make_user()
        summaries = [_make_summary()]

        with (
            mock.patch.object(indexer_routes, "_assert_workspace_access", mock.AsyncMock()) as assert_ws_mock,
            mock.patch.object(repository, "check_conversation_access", mock.AsyncMock(return_value=True)) as access_mock,
            mock.patch.object(
                repository,
                "list_subagent_conversation_summaries",
                mock.AsyncMock(return_value=summaries),
            ) as list_mock,
        ):
            result = await indexer_routes.list_subagent_conversation_summaries_route(
                conversation_id="parent-1",
                workspace_id="ws-1",
                user=user,
            )

        assert_ws_mock.assert_awaited_once_with("ws-1", user, "viewer")
        access_mock.assert_awaited_once_with(
            "parent-1",
            user.id,
            is_admin=False,
            workspace_id="ws-1",
        )
        list_mock.assert_awaited_once_with("parent-1")
        self.assertEqual([summary.id for summary in result], ["child-1"])

    async def test_denied_access_returns_404(self) -> None:
        user = _make_user()

        with (
            mock.patch.object(indexer_routes, "_assert_workspace_access", mock.AsyncMock()),
            mock.patch.object(repository, "check_conversation_access", mock.AsyncMock(return_value=False)),
            mock.patch.object(
                repository,
                "list_subagent_conversation_summaries",
                mock.AsyncMock(return_value=[]),
            ) as list_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                await indexer_routes.list_subagent_conversation_summaries_route(
                    conversation_id="parent-1",
                    workspace_id="ws-other",
                    user=user,
                )

        self.assertEqual(ctx.exception.status_code, 404)
        list_mock.assert_not_awaited()

    async def test_non_workspace_conversation_still_supported(self) -> None:
        """workspace_id defaults to None for standalone (non-workspace) parents."""
        user = _make_user()

        with (
            mock.patch.object(indexer_routes, "_assert_workspace_access", mock.AsyncMock()) as assert_ws_mock,
            mock.patch.object(repository, "check_conversation_access", mock.AsyncMock(return_value=True)) as access_mock,
            mock.patch.object(
                repository,
                "list_subagent_conversation_summaries",
                mock.AsyncMock(return_value=[]),
            ),
        ):
            result = await indexer_routes.list_subagent_conversation_summaries_route(
                conversation_id="parent-1",
                user=user,
            )

        assert_ws_mock.assert_awaited_once_with(None, user, "viewer")
        access_mock.assert_awaited_once_with(
            "parent-1",
            user.id,
            is_admin=False,
            workspace_id=None,
        )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
