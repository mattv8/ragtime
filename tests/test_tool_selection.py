from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from prisma.models import User

from ragtime.indexer.models import Conversation
from ragtime.indexer.routes import _resolve_selected_tool_ids_for_request
from ragtime.indexer.tool_selection import resolve_effective_tool_ids


class ResolveEffectiveToolIdsTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_all_returns_all_healthy_enabled_tools(self) -> None:
        healthy_loader = mock.AsyncMock(return_value=["tool-2", "tool-1"])
        group_loader = mock.AsyncMock(return_value=["tool-3"])

        resolved = await resolve_effective_tool_ids(
            tool_selection_mode="default_all",
            selected_tool_ids=["tool-9"],
            selected_tool_group_ids=["group-1"],
            list_healthy_enabled_tool_ids=healthy_loader,
            get_tool_ids_for_groups=group_loader,
        )

        self.assertEqual(resolved, ["tool-2", "tool-1"])
        healthy_loader.assert_awaited_once_with()
        group_loader.assert_not_awaited()

    async def test_custom_without_enabled_callback_preserves_legacy_healthy_filtering(self) -> None:
        resolved = await resolve_effective_tool_ids(
            tool_selection_mode="custom",
            selected_tool_ids=["tool-3", "tool-1", "tool-3", ""],
            selected_tool_group_ids=["group-1", "group-2"],
            list_healthy_enabled_tool_ids=mock.AsyncMock(return_value=["tool-1", "tool-2", "tool-4"]),
            get_tool_ids_for_groups=mock.AsyncMock(return_value=["tool-2", "tool-1", "tool-4", "tool-2"]),
        )

        self.assertEqual(resolved, ["tool-1", "tool-2", "tool-4"])

    async def test_custom_with_enabled_callback_preserves_unhealthy_enabled_tools_and_filters_disabled(self) -> None:
        healthy_loader = mock.AsyncMock(return_value=["tool-1", "tool-2"])
        enabled_loader = mock.AsyncMock(return_value=["tool-1", "tool-2", "tool-3", "tool-4"])
        group_loader = mock.AsyncMock(return_value=["tool-4", "tool-2", "tool-disabled", "tool-4"])

        resolved = await resolve_effective_tool_ids(
            tool_selection_mode="custom",
            selected_tool_ids=["tool-3", "tool-1", "tool-disabled", "tool-3", ""],
            selected_tool_group_ids=["group-1"],
            list_healthy_enabled_tool_ids=healthy_loader,
            list_enabled_tool_ids=enabled_loader,
            get_tool_ids_for_groups=group_loader,
        )

        self.assertEqual(resolved, ["tool-3", "tool-1", "tool-4", "tool-2"])
        healthy_loader.assert_not_awaited()
        enabled_loader.assert_awaited_once_with()
        group_loader.assert_awaited_once_with(["group-1"])

    async def test_invalid_mode_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "tool_selection_mode"):
            await resolve_effective_tool_ids(
                tool_selection_mode="",
                selected_tool_ids=[],
                selected_tool_group_ids=[],
                list_healthy_enabled_tool_ids=mock.AsyncMock(return_value=[]),
                get_tool_ids_for_groups=mock.AsyncMock(return_value=[]),
            )


class ResolveSelectedToolIdsForRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_scope_intersects_conversation_and_workspace_effective_tools(self) -> None:
        conversation = Conversation(
            id="conv-1",
            title="Test",
            model="gpt-4.1",
            user_id="user-1",
            workspace_id="workspace-1",
            messages=[],
            total_tokens=0,
            tool_selection_mode="custom",
        )

        fake_db = SimpleNamespace(
            connect=mock.AsyncMock(),
            disconnect=mock.AsyncMock(),
            conversationtoolselection=SimpleNamespace(
                find_many=mock.AsyncMock(return_value=[SimpleNamespace(toolConfigId="tool-1"), SimpleNamespace(toolConfigId="tool-2")])
            ),
            conversationtoolgroupselection=SimpleNamespace(find_many=mock.AsyncMock(return_value=[])),
        )
        workspace = SimpleNamespace(
            tool_selection_mode="custom",
            selected_tool_ids=["tool-2", "tool-3"],
            selected_tool_group_ids=[],
        )
        user = cast(User, SimpleNamespace(id="user-1", role="editor"))

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.enforce_workspace_role",
                mock.AsyncMock(return_value=workspace),
            ),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.get_active_cross_workspace_grant_modes",
                mock.AsyncMock(return_value={}),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-1", "tool-2", "tool-3"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(return_value=["tool-1", "tool-2"]),
                create=True,
            ),
            mock.patch.object(
                __import__("ragtime.indexer.routes", fromlist=["userspace_service"]).userspace_service,
                "filter_tool_ids_for_workspace_owner",
                create=True,
                new=mock.AsyncMock(return_value=["tool-2", "tool-3"]),
            ),
        ):
            _, selected_tool_ids, _ = await _resolve_selected_tool_ids_for_request(
                conversation,
                user,
                "workspace-1",
                "editor",
            )

        self.assertEqual(selected_tool_ids, {"tool-2"})


if __name__ == "__main__":
    unittest.main()
