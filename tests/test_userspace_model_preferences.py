from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Iterator, Protocol, cast
from unittest import mock

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.build_task_service import build_task_service
from ragtime.userspace.models import CreateWorkspaceRequest, DuplicateWorkspaceRequest, UserSpaceWorkspace
from ragtime.userspace.service import UserSpaceService

_NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


class _CapturedTask:
    def add_done_callback(self, callback) -> None:
        self.callback = callback


class _ModelPreferencesModule(Protocol):
    resolve_new_conversation_model: mock.AsyncMock
    get_workspace_user_default_model: mock.AsyncMock
    set_workspace_user_default_model: mock.AsyncMock


@contextmanager
def _fake_model_preferences_module(
    *,
    general_model: str = "general-model",
    workspace_model: str = "workspace-model",
    workspace_override: str | None = "workspace-override",
) -> Iterator[_ModelPreferencesModule]:
    module = cast(_ModelPreferencesModule, types.ModuleType("ragtime.indexer.model_preferences"))

    async def _resolve_new_conversation_model(app_settings, *, user_id, workspace_id=None):
        _ = app_settings, user_id
        return workspace_model if workspace_id else general_model

    module.resolve_new_conversation_model = mock.AsyncMock(side_effect=_resolve_new_conversation_model)
    module.get_workspace_user_default_model = mock.AsyncMock(return_value=workspace_override)
    module.set_workspace_user_default_model = mock.AsyncMock()
    with mock.patch.dict(sys.modules, {"ragtime.indexer.model_preferences": module}):
        yield module


class WorkspaceCreateTaskModelPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_workspace_create_task_resolves_model_for_creator_preference(self) -> None:
        service = UserSpaceService()
        app_settings = SimpleNamespace(llm_model="global-model", default_chat_model=None)
        captured_coroutines = []

        def _capture_task(coro, *, name=None):
            captured_coroutines.append((coro, name))
            return _CapturedTask()

        with (
            _fake_model_preferences_module(general_model="creator-general-model") as preferences,
            mock.patch("ragtime.userspace.service.repository.get_settings", mock.AsyncMock(return_value=app_settings)),
            mock.patch("ragtime.userspace.service.asyncio.create_task", side_effect=_capture_task),
        ):
            await service.enqueue_workspace_create_task(
                CreateWorkspaceRequest(name="Workspace"),
                "creator-1",
            )

        self.assertEqual(len(captured_coroutines), 1)
        captured_coroutines[0][0].close()
        preferences.resolve_new_conversation_model.assert_awaited_once_with(
            app_settings,
            user_id="creator-1",
            workspace_id=None,
        )


class WorkspaceDuplicateTaskModelPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_runner_uses_enqueued_preference_snapshot_before_creating_fallback_chat(self) -> None:
        service = UserSpaceService()
        source_workspace = UserSpaceWorkspace(
            id="source-ws",
            name="Source",
            owner_user_id="owner-1",
            created_at=_NOW,
            updated_at=_NOW,
        )
        target_workspace = UserSpaceWorkspace(
            id="target-ws",
            name="Copy",
            owner_user_id="user-1",
            created_at=_NOW,
            updated_at=_NOW,
        )
        app_settings = SimpleNamespace(llm_model="global-model", default_chat_model=None)
        captured_coroutines = []
        calls: list[str] = []

        def _capture_task(coro, *, name=None):
            captured_coroutines.append((coro, name))
            return _CapturedTask()

        async def _record_set_workspace_default(user_id: str, workspace_id: str, model: str | None) -> None:
            calls.append(f"set:{user_id}:{workspace_id}:{model}")

        async def _record_create_conversation(**kwargs):
            calls.append(f"create:{kwargs['model']}")
            return SimpleNamespace(id="conv-1")

        with (
            _fake_model_preferences_module(
                workspace_model="resolved-workspace-model",
                workspace_override="snapshotted-override",
            ) as preferences,
            mock.patch.object(service, "_enforce_workspace_access", mock.AsyncMock(return_value=source_workspace)),
            mock.patch.object(service, "_allocate_next_duplicate_workspace_name", mock.AsyncMock(return_value="Copy")),
            mock.patch("ragtime.userspace.service.repository.get_settings", mock.AsyncMock(return_value=app_settings)),
            mock.patch("ragtime.userspace.service.asyncio.create_task", side_effect=_capture_task),
            mock.patch.object(service, "_is_admin_user", mock.AsyncMock(return_value=False)),
            mock.patch.object(service, "create_workspace", mock.AsyncMock(return_value=target_workspace)),
            mock.patch.object(service, "_copy_workspace_files_for_duplicate", mock.AsyncMock()),
            mock.patch.object(service, "_copy_workspace_chats_for_duplicate", mock.AsyncMock(return_value=0)),
            mock.patch.object(service, "_copy_workspace_mounts_for_duplicate", mock.AsyncMock()),
            mock.patch.object(service, "upsert_workspace_file", mock.AsyncMock()),
            mock.patch.object(service, "_ensure_object_storage_config"),
            mock.patch.object(service, "_seed_runtime_bootstrap_config"),
            mock.patch.object(service, "_seed_runtime_entrypoint_config"),
            mock.patch.object(service, "_ensure_workspace_git_repo", mock.AsyncMock()),
            mock.patch.object(service, "_copy_workspace_env_vars_for_duplicate", mock.AsyncMock()),
            mock.patch.object(service, "_mark_workspace_code_index_dirty", mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.service.repository.create_conversation",
                mock.AsyncMock(side_effect=_record_create_conversation),
            ),
        ):
            await service.enqueue_workspace_duplicate_task(
                "source-ws",
                DuplicateWorkspaceRequest(copy_metadata=False, copy_files=True, copy_chats=False, copy_mounts=False),
                "user-1",
            )

            self.assertEqual(len(captured_coroutines), 1)
            preferences.get_workspace_user_default_model.return_value = "changed-after-enqueue"
            preferences.set_workspace_user_default_model.side_effect = _record_set_workspace_default
            await captured_coroutines[0][0]

        preferences.get_workspace_user_default_model.assert_awaited_once_with("user-1", "source-ws")
        preferences.set_workspace_user_default_model.assert_awaited_once_with(
            "user-1",
            "target-ws",
            "snapshotted-override",
        )
        preferences.resolve_new_conversation_model.assert_awaited_once_with(
            app_settings,
            user_id="user-1",
            workspace_id="source-ws",
        )
        self.assertEqual(
            calls,
            [
                "set:user-1:target-ws:snapshotted-override",
                "create:resolved-workspace-model",
            ],
        )


class BuildTaskModelPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_build_task_resolves_workspace_model_before_creating_conversation(self) -> None:
        app_settings = SimpleNamespace(llm_model="global-model", default_chat_model=None)
        workspace = SimpleNamespace(
            id="ws-1",
            name="Sales",
            tool_selection_mode="custom",
            selected_tool_ids=[],
            selected_tool_group_ids=[],
        )
        conversation = SimpleNamespace(id="conv-1")
        ledger = SimpleNamespace(id="ebr-1")
        task = SimpleNamespace(id="task-1", status="pending")

        with (
            _fake_model_preferences_module(workspace_model="workspace-preference-model") as preferences,
            mock.patch.object(type(build_task_service), "_load_prisma_user", mock.AsyncMock(return_value=SimpleNamespace(id="user-1", role="user"))),
            mock.patch.object(type(build_task_service), "_enforce_editor", mock.AsyncMock(return_value=workspace)),
            mock.patch.object(type(build_task_service), "_validate_brief", mock.AsyncMock()),
            mock.patch("ragtime.userspace.build_task_service.find_external_build_request", mock.AsyncMock(return_value=None)),
            mock.patch("ragtime.userspace.build_task_service.create_external_build_request", mock.AsyncMock(return_value=ledger)),
            mock.patch("ragtime.userspace.build_task_service.bind_external_build_request_conversation", mock.AsyncMock()),
            mock.patch("ragtime.userspace.build_task_service.finalize_external_build_request", mock.AsyncMock()),
            mock.patch("ragtime.userspace.build_task_service.repository.get_settings", mock.AsyncMock(return_value=app_settings)),
            mock.patch("ragtime.userspace.build_task_service.repository.create_conversation", mock.AsyncMock(return_value=conversation)) as create_conversation,
            mock.patch(
                "ragtime.indexer.routes._send_background_message_to_loaded_conversation",
                mock.AsyncMock(return_value={"task": task}),
            ),
        ):
            from ragtime.userspace.agent_briefs import BuildBriefInput

            await build_task_service.start_build_task(
                "ws-1",
                "user-1",
                BuildBriefInput(
                    idempotency_key="request-1",
                    title="Build it",
                    objective="Do the work",
                    requirements=["One"],
                    acceptance_criteria=["Done"],
                ),
            )

        preferences.resolve_new_conversation_model.assert_awaited_once_with(
            app_settings,
            user_id="user-1",
            workspace_id="ws-1",
        )
        create_conversation.assert_awaited_once_with(
            title="Build it",
            user_id="user-1",
            workspace_id="ws-1",
            model="workspace-preference-model",
        )


if __name__ == "__main__":
    unittest.main()
