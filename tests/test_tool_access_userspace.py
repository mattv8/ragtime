from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException

from tests.rag_prompts_stub import install_fake_rag_prompts, remove_fake_rag_prompts

inserted_fake_rag_prompts = install_fake_rag_prompts()

from ragtime.userspace import routes as userspace_routes
from ragtime.userspace.models import ExecuteComponentRequest, UserSpaceWorkspace
from ragtime.userspace.routes import list_userspace_tool_groups, list_userspace_tools
from ragtime.userspace.service import UserSpaceService

remove_fake_rag_prompts(inserted_fake_rag_prompts)


def _tool(
    tool_id: str,
    *,
    group_id: str | None = None,
    tool_type: str = "postgres",
    allow_write: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        name=tool_id,
        tool_type=SimpleNamespace(value=tool_type),
        description=f"desc:{tool_id}",
        allow_write=allow_write,
        group_id=group_id,
        group_name=(f"Group {group_id}" if group_id else None),
        enabled=True,
    )


def _workspace() -> UserSpaceWorkspace:
    now = datetime.now(timezone.utc)
    return UserSpaceWorkspace(
        id="ws-1",
        name="Workspace",
        owner_user_id="owner-1",
        tool_selection_mode="custom",
        selected_tool_ids=["tool-1"],
        created_at=now,
        updated_at=now,
    )


class UserSpaceToolCatalogAclTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_union_uses_strongest_access_level_and_preserves_catalog_order(self) -> None:
        user = SimpleNamespace(id="user-1", role="user")
        resolve_mock = mock.AsyncMock(
            side_effect=[
                {"tool-a": "deny", "tool-b": "read", "tool-c": "read_write"},
                {"tool-a": "read", "tool-b": "deny", "tool-c": "read"},
            ]
        )

        with (
            mock.patch.object(
                userspace_routes.repository,
                "list_tool_configs",
                mock.AsyncMock(return_value=[_tool("tool-a"), _tool("tool-b"), _tool("tool-c")]),
            ),
            mock.patch.object(userspace_routes.tool_health_monitor, "is_tool_healthy", side_effect=lambda _tool_id: True),
            mock.patch("ragtime.userspace.routes.resolve_tool_access", resolve_mock),
        ):
            result = await list_userspace_tools(user=user)

        self.assertEqual([tool.id for tool in result], ["tool-a", "tool-b", "tool-c"])
        self.assertEqual([tool.access_level for tool in result], ["read", "read", "read_write"])
        self.assertEqual(resolve_mock.await_count, 2)
        self.assertEqual(resolve_mock.await_args_list[0].kwargs["surface"], "chat")
        self.assertEqual(resolve_mock.await_args_list[1].kwargs["surface"], "workspace")

    async def test_tools_single_surface_omits_denied_tools_and_sets_access_level(self) -> None:
        user = SimpleNamespace(id="user-1", role="user")
        resolve_mock = mock.AsyncMock(return_value={"tool-a": "deny", "tool-b": "read"})

        with (
            mock.patch.object(
                userspace_routes.repository,
                "list_tool_configs",
                mock.AsyncMock(return_value=[_tool("tool-a"), _tool("tool-b")]),
            ),
            mock.patch.object(userspace_routes.tool_health_monitor, "is_tool_healthy", side_effect=lambda _tool_id: True),
            mock.patch("ragtime.userspace.routes.resolve_tool_access", resolve_mock),
        ):
            result = await list_userspace_tools(surface="workspace", user=user)

        self.assertEqual([tool.id for tool in result], ["tool-b"])
        self.assertEqual([tool.access_level for tool in result], ["read"])
        resolve_mock.assert_awaited_once()
        await_args = resolve_mock.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(await_args.kwargs["surface"], "workspace")

    async def test_tools_admin_bypass_returns_all_without_acl_filter(self) -> None:
        user = SimpleNamespace(id="admin-1", role="admin")
        resolve_mock = mock.AsyncMock()

        with (
            mock.patch.object(
                userspace_routes.repository,
                "list_tool_configs",
                mock.AsyncMock(return_value=[_tool("tool-a"), _tool("tool-b")]),
            ),
            mock.patch.object(userspace_routes.tool_health_monitor, "is_tool_healthy", side_effect=lambda _tool_id: True),
            mock.patch("ragtime.userspace.routes.resolve_tool_access", resolve_mock),
        ):
            result = await list_userspace_tools(surface="chat", user=user)

        self.assertEqual([tool.id for tool in result], ["tool-a", "tool-b"])
        self.assertEqual([tool.access_level for tool in result], ["read_write", "read_write"])
        resolve_mock.assert_not_awaited()

    async def test_tool_groups_omit_empty_groups_after_acl_filtering(self) -> None:
        user = SimpleNamespace(id="user-1", role="user")

        with (
            mock.patch.object(
                userspace_routes.repository,
                "list_tool_groups",
                mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(id="group-a", name="A", description="", sort_order=0, created_at=None, updated_at=None),
                        SimpleNamespace(id="group-b", name="B", description="", sort_order=1, created_at=None, updated_at=None),
                    ]
                ),
            ),
            mock.patch.object(
                userspace_routes.repository,
                "list_tool_configs",
                mock.AsyncMock(return_value=[_tool("tool-a", group_id="group-a"), _tool("tool-b", group_id="group-b")]),
            ),
            mock.patch(
                "ragtime.userspace.routes.resolve_tool_access",
                mock.AsyncMock(side_effect=[{"tool-a": "read", "tool-b": "deny"}, {"tool-a": "deny", "tool-b": "deny"}]),
            ),
        ):
            result = await list_userspace_tool_groups(user=user)

        self.assertEqual([group["id"] for group in result], ["group-a"])


class WorkspaceOwnerAclHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_filter_tool_ids_for_workspace_owner_uses_owner_identity(self) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(id="owner-1", role="admin"))))
        filter_mock = mock.AsyncMock(return_value=["tool-2"])

        with (
            mock.patch("ragtime.userspace.service.get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch("ragtime.userspace.service.filter_tool_ids_by_access", filter_mock),
        ):
            result = await service.filter_tool_ids_for_workspace_owner(
                cast(UserSpaceWorkspace, SimpleNamespace(owner_user_id="owner-1")),
                ["tool-1", "tool-2"],
            )

        self.assertEqual(result, ["tool-2"])
        filter_mock.assert_awaited_once_with(
            user_id="owner-1",
            is_admin=True,
            surface="workspace",
            tool_config_ids=["tool-1", "tool-2"],
        )

    async def test_filter_tool_ids_for_workspace_owner_missing_owner_fails_closed(self) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=None)))

        with (
            mock.patch("ragtime.userspace.service.get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch("ragtime.userspace.service.filter_tool_ids_by_access", mock.AsyncMock()) as filter_mock,
        ):
            result = await service.filter_tool_ids_for_workspace_owner(
                cast(UserSpaceWorkspace, SimpleNamespace(owner_user_id="owner-1")),
                ["tool-1"],
            )

        self.assertEqual(result, [])
        filter_mock.assert_not_awaited()

    async def test_resolve_effective_workspace_tool_ids_post_filters_owner_acl(self) -> None:
        service = UserSpaceService()
        workspace = _workspace()

        with (
            mock.patch("ragtime.userspace.service.resolve_effective_tool_ids", mock.AsyncMock(return_value=["tool-1", "tool-2"])),
            mock.patch.object(service, "filter_tool_ids_for_workspace_owner", mock.AsyncMock(return_value=["tool-2"])) as filter_mock,
        ):
            result = await service._resolve_effective_workspace_tool_ids(workspace)

        self.assertEqual(result, ["tool-2"])
        filter_mock.assert_awaited_once_with(workspace, ["tool-1", "tool-2"])


class WorkspaceExecutionAclParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_component_denied_by_owner_acl_returns_not_selected_403(self) -> None:
        service = UserSpaceService()
        workspace = _workspace()

        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=workspace)),
            mock.patch.object(service, "_resolve_effective_workspace_tool_ids", mock.AsyncMock(return_value=[])),
            mock.patch.object(userspace_routes.repository, "get_tool_config", mock.AsyncMock()) as get_tool_config,
        ):
            with self.assertRaises(HTTPException) as raised:
                await service.execute_component(
                    "ws-1",
                    ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"}),
                    user_id="viewer-1",
                )

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(raised.exception.detail, "Component tool-1 is not selected for this request.")
        get_tool_config.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
