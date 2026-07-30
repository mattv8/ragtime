from __future__ import annotations

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from prisma.models import User

from ragtime.indexer.models import Conversation
from ragtime.indexer.routes import (
    _create_background_chat_task_after_user_message,
    _resolve_selected_tool_ids_for_request,
    _shared_conversation_request_actor,
    get_conversation_tools,
    update_conversation_tools,
)
from ragtime.indexer.tool_selection import resolve_effective_tool_ids
from ragtime.rag import components as rag_components


class _FakeConversationToolDb:
    def __init__(
        self,
        *,
        conversation: Any,
        selections: list[Any] | None = None,
        group_selections: list[Any] | None = None,
        option_rows: list[Any] | None = None,
    ) -> None:
        self.conversation = conversation
        self._selections = list(selections or [])
        self._group_selections = list(group_selections or [])
        self._option_rows = list(option_rows or [])
        self.connect = mock.AsyncMock()
        self.disconnect = mock.AsyncMock()
        self.conversationtoolselection = SimpleNamespace(
            find_many=mock.AsyncMock(return_value=list(self._selections)),
            delete_many=mock.AsyncMock(),
            create=mock.AsyncMock(),
        )
        self.conversationtoolgroupselection = SimpleNamespace(
            find_many=mock.AsyncMock(return_value=list(self._group_selections)),
            delete_many=mock.AsyncMock(),
            create=mock.AsyncMock(),
        )
        self.conversationtooloption = SimpleNamespace(
            find_many=mock.AsyncMock(return_value=list(self._option_rows)),
            delete_many=mock.AsyncMock(),
            create=mock.AsyncMock(),
        )
        self.conversation = SimpleNamespace(
            find_unique=mock.AsyncMock(return_value=conversation),
            update=mock.AsyncMock(),
        )
        self.user = SimpleNamespace(find_unique=mock.AsyncMock(return_value=None))


class ResolveSelectedToolIdsAclTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_acl_filters_conversation_selected_ids_for_regular_user(self) -> None:
        conversation = Conversation(
            id="conv-1",
            title="Test",
            model="gpt-4.1",
            user_id="user-1",
            workspace_id=None,
            messages=[],
            total_tokens=0,
            tool_selection_mode="custom",
        )
        fake_db = _FakeConversationToolDb(
            conversation=SimpleNamespace(),
            selections=[
                SimpleNamespace(toolConfigId="tool-1"),
                SimpleNamespace(toolConfigId="tool-2"),
            ],
        )
        user = cast(User, SimpleNamespace(id="user-1", role="user"))

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-1", "tool-2"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(return_value=["tool-2"]),
                create=True,
            ) as filter_access,
        ):
            _, selected_tool_ids, _ = await _resolve_selected_tool_ids_for_request(
                conversation,
                user,
                None,
                "editor",
            )

        self.assertEqual(selected_tool_ids, {"tool-2"})
        filter_access.assert_awaited_once_with(
            user_id="user-1",
            is_admin=False,
            surface="chat",
            tool_config_ids=["tool-1", "tool-2"],
        )

    async def test_chat_acl_preserves_admin_bypass_and_workspace_owner_intersection(self) -> None:
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
        fake_db = _FakeConversationToolDb(
            conversation=SimpleNamespace(),
            selections=[
                SimpleNamespace(toolConfigId="tool-1"),
                SimpleNamespace(toolConfigId="tool-2"),
            ],
        )
        workspace = SimpleNamespace(
            tool_selection_mode="custom",
            selected_tool_ids=["tool-1", "tool-2", "tool-3"],
            selected_tool_group_ids=[],
            owner_user_id="owner-1",
        )
        user = cast(User, SimpleNamespace(id="admin-1", role="admin"))

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-1", "tool-2", "tool-3"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.enforce_workspace_role",
                mock.AsyncMock(return_value=workspace),
            ),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.get_active_cross_workspace_grant_modes",
                mock.AsyncMock(return_value={}),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=lambda **kwargs: list(kwargs["tool_config_ids"])),
                create=True,
            ) as filter_access,
            mock.patch.object(
                __import__("ragtime.indexer.routes", fromlist=["userspace_service"]).userspace_service,
                "filter_tool_ids_for_workspace_owner",
                create=True,
                new=mock.AsyncMock(return_value=["tool-2", "tool-3"]),
            ) as filter_owner,
        ):
            _, selected_tool_ids, _ = await _resolve_selected_tool_ids_for_request(
                conversation,
                user,
                "workspace-1",
                "editor",
            )

        self.assertEqual(selected_tool_ids, {"tool-2"})
        filter_access.assert_awaited_once_with(
            user_id="admin-1",
            is_admin=True,
            surface="chat",
            tool_config_ids=["tool-1", "tool-2"],
        )
        filter_owner.assert_awaited_once_with(workspace, ["tool-1", "tool-2", "tool-3"])

    async def test_workspace_owner_filter_does_not_fall_back_to_original_ids_when_helper_is_not_callable(self) -> None:
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
        fake_db = _FakeConversationToolDb(
            conversation=SimpleNamespace(),
            selections=[
                SimpleNamespace(toolConfigId="tool-1"),
                SimpleNamespace(toolConfigId="tool-2"),
            ],
        )
        workspace = SimpleNamespace(
            tool_selection_mode="custom",
            selected_tool_ids=["tool-1", "tool-2"],
            selected_tool_group_ids=[],
            owner_user_id="owner-1",
        )
        user = cast(User, SimpleNamespace(id="admin-1", role="admin"))

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-1", "tool-2"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=[]),
            ),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.enforce_workspace_role",
                mock.AsyncMock(return_value=workspace),
            ),
            mock.patch(
                "ragtime.indexer.routes.userspace_service.get_active_cross_workspace_grant_modes",
                mock.AsyncMock(return_value={}),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=lambda **kwargs: list(kwargs["tool_config_ids"])),
                create=True,
            ),
            mock.patch.object(
                __import__("ragtime.indexer.routes", fromlist=["userspace_service"]).userspace_service,
                "filter_tool_ids_for_workspace_owner",
                new=None,
            ),
        ):
            with self.assertRaises(TypeError):
                await _resolve_selected_tool_ids_for_request(
                    conversation,
                    user,
                    "workspace-1",
                    "editor",
                )


class SharedConversationActorTests(unittest.IsolatedAsyncioTestCase):
    async def test_anonymous_shared_actor_is_safe_non_admin_prompt_identity(self) -> None:
        fake_db = _FakeConversationToolDb(conversation=SimpleNamespace())
        fake_db.user.find_unique = mock.AsyncMock(return_value=SimpleNamespace(id="owner-1", role="admin"))

        with (
            mock.patch("ragtime.indexer.routes.repository._get_db", mock.AsyncMock(return_value=fake_db)) as get_db,
            mock.patch("ragtime.indexer.routes.Prisma") as prisma_cls,
        ):
            actor = await _shared_conversation_request_actor(
                Conversation(
                    id="conv-1",
                    title="Shared",
                    model="gpt-4.1",
                    user_id="owner-1",
                    workspace_id=None,
                    messages=[],
                    total_tokens=0,
                    tool_selection_mode="custom",
                ),
                SimpleNamespace(ownerUserId="owner-1"),
                None,
            )

        get_db.assert_not_awaited()
        prisma_cls.assert_not_called()
        fake_db.connect.assert_not_awaited()
        fake_db.disconnect.assert_not_awaited()
        self.assertEqual(actor.id, "owner-1")
        self.assertEqual(actor.username, "")
        self.assertEqual(actor.displayName, "")
        self.assertEqual(actor.role, "user")


class ConversationToolEndpointAclTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_conversation_tools_preserves_visible_group_selections_without_flattening(self) -> None:
        conversation = SimpleNamespace(
            id="conv-1",
            workspaceId=None,
            userId="owner-1",
            toolSelectionMode="custom",
            disabledBuiltinToolIds=[],
            subagentsEnabled=True,
            members=[],
        )
        fake_db = _FakeConversationToolDb(
            conversation=conversation,
            selections=[SimpleNamespace(toolConfigId="tool-direct")],
            group_selections=[SimpleNamespace(toolGroupId="group-1")],
            option_rows=[
                SimpleNamespace(toolConfigId="tool-group", options={"write_access_enabled": True}),
                SimpleNamespace(toolConfigId="tool-hidden", options={"write_access_enabled": True}),
            ],
        )
        user = cast(User, SimpleNamespace(id="owner-1", role="user"))

        async def _filter_visible(**kwargs: Any) -> list[str]:
            allowed_ids = {"tool-group"}
            return [tool_id for tool_id in kwargs["tool_config_ids"] if tool_id in allowed_ids]

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-direct", "tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=["tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=_filter_visible),
                create=True,
            ),
        ):
            result = await get_conversation_tools("conv-1", user)

        self.assertEqual(result["tool_config_ids"], [])
        self.assertEqual(result["tool_group_ids"], ["group-1"])
        self.assertEqual(result["tool_options"], {"tool-group": {"write_access_enabled": True}})

    async def test_get_conversation_tools_default_all_returns_live_acl_visible_effective_ids(self) -> None:
        conversation = SimpleNamespace(
            id="conv-1",
            workspaceId=None,
            userId="owner-1",
            toolSelectionMode="default_all",
            disabledBuiltinToolIds=[],
            subagentsEnabled=True,
            members=[],
        )
        fake_db = _FakeConversationToolDb(
            conversation=conversation,
            selections=[SimpleNamespace(toolConfigId="tool-direct")],
            group_selections=[SimpleNamespace(toolGroupId="group-1")],
            option_rows=[
                SimpleNamespace(toolConfigId="tool-direct", options={"write_access_enabled": True}),
                SimpleNamespace(toolConfigId="tool-group", options={"write_access_enabled": True}),
                SimpleNamespace(toolConfigId="tool-hidden", options={"write_access_enabled": True}),
            ],
        )
        user = cast(User, SimpleNamespace(id="owner-1", role="user"))

        async def _filter_visible(**kwargs: Any) -> list[str]:
            allowed_ids = {"tool-direct", "tool-group"}
            return [tool_id for tool_id in kwargs["tool_config_ids"] if tool_id in allowed_ids]

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-direct", "tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=["tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=_filter_visible),
                create=True,
            ),
        ):
            result = await get_conversation_tools("conv-1", user)

        self.assertEqual(result["tool_selection_mode"], "default_all")
        self.assertEqual(result["tool_config_ids"], ["tool-direct", "tool-group"])
        self.assertEqual(result["tool_group_ids"], [])
        self.assertEqual(
            result["tool_options"],
            {
                "tool-direct": {"write_access_enabled": True},
                "tool-group": {"write_access_enabled": True},
            },
        )

    async def test_update_conversation_tools_persists_direct_and_group_selections_separately(self) -> None:
        conversation = SimpleNamespace(
            id="conv-1",
            parentConversationId=None,
            workspaceId=None,
            userId="owner-1",
            members=[],
        )
        fake_db = _FakeConversationToolDb(conversation=conversation)
        user = cast(User, SimpleNamespace(id="owner-1", role="user"))

        async def _filter_visible(**kwargs: Any) -> list[str]:
            allowed_ids = {"tool-direct", "tool-group"}
            return [tool_id for tool_id in kwargs["tool_config_ids"] if tool_id in allowed_ids]

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-direct", "tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=["tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=_filter_visible),
                create=True,
            ),
        ):
            await update_conversation_tools(
                "conv-1",
                {
                    "tool_selection_mode": "custom",
                    "tool_config_ids": ["tool-direct", "tool-hidden"],
                    "tool_group_ids": ["group-1"],
                    "tool_options": {
                        "tool-direct": {"write_access_enabled": True},
                        "tool-group": {"write_access_enabled": True},
                        "tool-hidden": {"write_access_enabled": True},
                    },
                },
                user,
            )

        created_ids = [call.kwargs["data"]["toolConfigId"] for call in fake_db.conversationtoolselection.create.await_args_list]
        self.assertEqual(created_ids, ["tool-direct"])
        created_group_ids = [call.kwargs["data"]["toolGroupId"] for call in fake_db.conversationtoolgroupselection.create.await_args_list]
        self.assertEqual(created_group_ids, ["group-1"])
        option_ids = [call.kwargs["data"]["toolConfigId"] for call in fake_db.conversationtooloption.create.await_args_list]
        self.assertEqual(option_ids, ["tool-direct", "tool-group"])

    async def test_update_conversation_tools_default_all_does_not_persist_tool_snapshot(self) -> None:
        conversation = SimpleNamespace(
            id="conv-1",
            parentConversationId=None,
            workspaceId=None,
            userId="owner-1",
            members=[],
        )
        fake_db = _FakeConversationToolDb(conversation=conversation)
        user = cast(User, SimpleNamespace(id="owner-1", role="user"))

        async def _filter_visible(**kwargs: Any) -> list[str]:
            allowed_ids = {"tool-direct", "tool-group"}
            return [tool_id for tool_id in kwargs["tool_config_ids"] if tool_id in allowed_ids]

        with (
            mock.patch("ragtime.indexer.routes.Prisma", return_value=fake_db),
            mock.patch(
                "ragtime.indexer.routes.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=["tool-direct", "tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.repository.get_tool_ids_for_groups",
                mock.AsyncMock(return_value=["tool-group", "tool-hidden"]),
            ),
            mock.patch(
                "ragtime.indexer.routes.filter_tool_ids_by_access",
                mock.AsyncMock(side_effect=_filter_visible),
                create=True,
            ),
        ):
            await update_conversation_tools(
                "conv-1",
                {
                    "tool_selection_mode": "default_all",
                    "tool_config_ids": ["tool-direct"],
                    "tool_group_ids": ["group-1"],
                    "tool_options": {
                        "tool-direct": {"write_access_enabled": True},
                        "tool-group": {"write_access_enabled": True},
                        "tool-hidden": {"write_access_enabled": True},
                    },
                },
                user,
            )

        fake_db.conversationtoolselection.create.assert_not_awaited()
        fake_db.conversationtoolgroupselection.create.assert_not_awaited()
        option_ids = [call.kwargs["data"]["toolConfigId"] for call in fake_db.conversationtooloption.create.await_args_list]
        self.assertEqual(option_ids, ["tool-direct", "tool-group"])


class ConversationToolPromptAclTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_write_override_fails_closed_without_user_identity(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {
                "id": "tool-1",
                "name": "Demo SSH",
                "tool_type": "ssh_shell",
                "allow_write": False,
            }
        ]
        rag._app_settings = {"max_tool_output_chars": 0}
        option_rows = [SimpleNamespace(toolConfigId="tool-1", options={"write_access_enabled": True})]
        fake_db = SimpleNamespace(
            conversationtooloption=SimpleNamespace(find_many=mock.AsyncMock(return_value=option_rows)),
        )

        with (
            mock.patch.object(rag_components, "get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(
                rag,
                "build_tools_from_runtime_config",
                new=mock.AsyncMock(return_value=[SimpleNamespace(name="ssh_demo_ssh", description="writable")]),
            ) as build_tools,
        ):
            runtime_tools = await rag._apply_conversation_tool_overrides(
                "conversation-1",
                [SimpleNamespace(name="ssh_demo_ssh", description="original")],
                user_id=None,
                is_admin=False,
            )

        self.assertEqual([tool.name for tool in runtime_tools], ["ssh_demo_ssh"])
        build_tools.assert_not_awaited()

    async def test_background_chat_task_receives_authenticated_actor_context(self) -> None:
        user = cast(
            User,
            SimpleNamespace(
                id="admin-1",
                username="local:admin",
                displayName="Admin",
                role="admin",
            ),
        )
        conversation = Conversation(
            id="conversation-1",
            title="Test",
            model="gpt-4.1",
            user_id="user-1",
            workspace_id=None,
            messages=[],
            total_tokens=0,
            tool_selection_mode="all",
        )
        persisted_task = SimpleNamespace(id="task-1")

        with (
            mock.patch("ragtime.indexer.routes.create_usage_attempt", mock.AsyncMock(return_value="attempt-1")),
            mock.patch("ragtime.indexer.routes.background_task_service.start_task", return_value="task-1") as start_task,
            mock.patch("ragtime.indexer.routes.repository.get_chat_task", mock.AsyncMock(return_value=persisted_task)),
        ):
            result = await _create_background_chat_task_after_user_message(
                conversation_id="conversation-1",
                user_message="run the write operation",
                user=user,
                conv=conversation,
                blocked_tool_names=set(),
                workspace_context=None,
                existing_task_id="task-1",
            )

        self.assertIs(result, persisted_task)
        self.assertEqual(
            start_task.call_args.kwargs["current_user_context"],
            {
                "user_id": "admin-1",
                "username": "admin",
                "display_name": "Admin",
                "is_admin": True,
            },
        )

    async def test_conversation_write_override_is_capped_by_chat_acl_read(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {
                "id": "tool-1",
                "name": "Demo SSH",
                "tool_type": "ssh_shell",
                "allow_write": False,
            }
        ]
        rag._app_settings = {"max_tool_output_chars": 0}

        option_rows = [SimpleNamespace(toolConfigId="tool-1", options={"write_access_enabled": True})]
        fake_db = SimpleNamespace(
            conversationtooloption=SimpleNamespace(find_many=mock.AsyncMock(return_value=option_rows)),
        )

        with (
            mock.patch.object(rag_components, "get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch(
                "ragtime.rag.components.resolve_tool_access",
                mock.AsyncMock(return_value={"tool-1": "read"}),
                create=True,
            ),
            mock.patch.object(
                rag,
                "build_tools_from_runtime_config",
                new=mock.AsyncMock(return_value=[SimpleNamespace(name="ssh_demo_ssh", description="readonly")]),
            ) as build_tools,
        ):
            runtime_tools = await rag._apply_conversation_tool_overrides(
                "conversation-1",
                [SimpleNamespace(name="ssh_demo_ssh", description="original")],
                user_id="user-1",
                is_admin=False,
            )

        self.assertEqual([tool.name for tool in runtime_tools], ["ssh_demo_ssh"])
        build_tools.assert_not_awaited()

    def test_request_system_prompt_omits_denied_tool_names_from_unavailable_prose(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {"id": "tool-1", "name": "Allowed Tool", "tool_type": "ssh_shell"},
            {"id": "tool-2", "name": "Denied Tool", "tool_type": "ssh_shell"},
        ]
        rag._app_settings = {}
        rag._index_metadata = []
        rag._request_prompt_cache = {}

        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=[],
        )

        self.assertIn("Allowed Tool", prompt)
        self.assertNotIn("Denied Tool", prompt)

    async def test_userspace_runtime_context_filters_owner_denied_tool_names_before_prompt_materialization(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {"id": "tool-1", "name": "Healthy Tool", "tool_type": "postgres"},
            {"id": "tool-3", "name": "Flapping Tool", "tool_type": "postgres"},
        ]
        rag._app_settings = {}
        rag._index_metadata = []
        rag._request_prompt_cache = {}

        workspace = SimpleNamespace(
            id="workspace-1",
            owner_user_id="owner-1",
            tool_selection_mode="custom",
            selected_tool_ids=["tool-3"],
            selected_tool_group_ids=[],
            sqlite_persistence_mode="exclude",
        )
        runtime_tools = [SimpleNamespace(name="query_flapping_tool")]

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    rag,
                    "_apply_conversation_tool_overrides",
                    mock.AsyncMock(return_value=list(runtime_tools)),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.get_workspace",
                    mock.AsyncMock(return_value=workspace),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.resolve_effective_tool_ids",
                    mock.AsyncMock(side_effect=resolve_effective_tool_ids),
                )
            )
            filter_owner = stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.filter_tool_ids_for_workspace_owner",
                    mock.AsyncMock(return_value=["tool-3"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.repository.list_healthy_enabled_tool_ids",
                    mock.AsyncMock(return_value=["tool-1"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.repository.list_enabled_tool_ids",
                    mock.AsyncMock(return_value=["tool-1", "tool-3"]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.repository.get_tool_ids_for_groups",
                    mock.AsyncMock(return_value=[]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.list_workspace_env_var_summaries",
                    mock.AsyncMock(return_value=[]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.list_workspace_mounts",
                    mock.AsyncMock(return_value=[]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.list_mountable_sources",
                    mock.AsyncMock(return_value=[]),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.get_workspace_object_storage_summary",
                    mock.AsyncMock(return_value=None),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_service.list_workspace_preview_diagnostic_summary",
                    mock.AsyncMock(return_value=None),
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.userspace_runtime_service.get_devserver_status",
                    mock.AsyncMock(return_value=None),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    rag,
                    "_create_userspace_file_tools",
                    mock.AsyncMock(return_value=[]),
                )
            )
            stack.enter_context(mock.patch.object(rag, "_create_spawn_subagents_tool", mock.AsyncMock(return_value=None)))
            stack.enter_context(
                mock.patch.object(
                    rag,
                    "_apply_mode_specific_tool_description_overrides",
                    side_effect=lambda tools, **_: tools,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    rag,
                    "_wrap_userspace_runtime_tools_for_execution_proofs",
                    side_effect=lambda tools, *_: tools,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    rag,
                    "_wrap_runtime_tools_with_request_state",
                    side_effect=lambda tools, **_: (tools, {}),
                )
            )
            stack.enter_context(mock.patch.object(rag, "_build_userspace_continuity_prompt", mock.AsyncMock(return_value="")))
            stack.enter_context(mock.patch.object(rag, "_build_userspace_env_var_turn_hint", return_value=""))
            stack.enter_context(mock.patch.object(rag, "_build_userspace_runtime_status_turn_hint", return_value=""))
            stack.enter_context(mock.patch.object(rag, "_build_userspace_env_var_prompt_fragment", return_value=""))
            stack.enter_context(mock.patch.object(rag, "_build_userspace_mount_prompt_fragment", return_value=""))
            stack.enter_context(mock.patch.object(rag, "_build_userspace_object_storage_prompt_fragment", return_value=""))
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.build_userspace_mode_prompt_addition",
                    return_value="",
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.build_userspace_entrypoint_nudge",
                    return_value="",
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.rag.components.build_userspace_diagnostics_turn_reminder_line",
                    return_value="",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    rag_components.userspace_service,
                    "get_workspace_entrypoint_status",
                    return_value=SimpleNamespace(state="valid", framework="react", command="npm run dev", cwd="."),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    rag_components.userspace_service,
                    "is_default_static_entrypoint",
                    return_value=False,
                )
            )
            request_context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=cast(Any, SimpleNamespace(tools=list(runtime_tools))),
                blocked_tool_names=None,
                workspace_context={"workspace_id": "workspace-1", "user_id": "viewer-1"},
                add_chat_visualization_prompt=False,
                user_id="viewer-1",
                current_user_context={
                    "user_id": "viewer-1",
                    "username": "viewer",
                    "display_name": "Viewer",
                    "is_admin": False,
                },
            )

        filter_owner.assert_awaited_once_with(workspace, ["tool-3"])
        self.assertEqual(request_context["allowed_tool_config_ids"], ["tool-3"])
        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="userspace",
            allowed_tool_config_ids=request_context["allowed_tool_config_ids"],
            runtime_tools=[],
        )
        self.assertIn("Flapping Tool", prompt)
        self.assertNotIn("Healthy Tool", prompt)


if __name__ == "__main__":
    unittest.main()
