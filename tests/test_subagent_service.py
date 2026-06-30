import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException
from langchain_core.tools import StructuredTool, ToolException

from ragtime.indexer.models import ChatTaskStatus
from ragtime.indexer.routes import _assert_conversation_mutable
from ragtime.rag.components import RAGComponents
from ragtime.rag.prompts import USERSPACE_SUBAGENT_GUIDANCE_PROMPT
from ragtime.userspace.subagent_service import MAX_SUBAGENT_FANOUT, SUBAGENT_HANDOFF_TOOL_NAME, SUBAGENT_PRIVATE_PROMPT_CONTEXT_KEY, SubAgentService


class SubAgentServiceSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SubAgentService()

    def test_normalize_specs_accepts_non_overlapping_scopes(self) -> None:
        specs = self.service._normalize_specs(
            [
                {
                    "name": "UI",
                    "role": "implement",
                    "instructions": "Update interface",
                    "file_scope": ["dashboard/ui"],
                },
                {
                    "name": "Review",
                    "role": "review",
                    "instructions": "Review changes",
                    "file_scope": [],
                },
            ]
        )

        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].file_scope, ["dashboard/ui"])
        self.assertEqual(specs[1].role, "review")

    def test_normalize_specs_rejects_overlapping_scopes(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.service._normalize_specs(
                [
                    {
                        "name": "A",
                        "role": "implement",
                        "instructions": "Work A",
                        "file_scope": ["dashboard"],
                    },
                    {
                        "name": "B",
                        "role": "implement",
                        "instructions": "Work B",
                        "file_scope": ["dashboard/main.ts"],
                    },
                ]
            )

    def test_normalize_specs_rejects_traversal_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid subagent file scope"):
            self.service._normalize_specs(
                [
                    {
                        "name": "Bad",
                        "role": "implement",
                        "instructions": "Escape scope",
                        "file_scope": ["../secrets"],
                    }
                ]
            )

    def test_normalize_specs_rejects_too_many_subagents(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 6"):
            self.service._normalize_specs(
                [
                    {
                        "name": f"Agent {index}",
                        "role": "implement",
                        "instructions": "Work",
                        "file_scope": [f"file-{index}.ts"],
                    }
                    for index in range(MAX_SUBAGENT_FANOUT + 1)
                ]
            )

    def test_extract_handoff_requires_structured_tool_payload(self) -> None:
        task = SimpleNamespace(
            streaming_state=SimpleNamespace(
                tool_calls=[
                    {
                        "tool": SUBAGENT_HANDOFF_TOOL_NAME,
                        "input": {
                            "final_output": "## Work complete\n\nChanged the dashboard.",
                            "summary": "Changed the dashboard",
                            "files_changed": ["dashboard/main.ts"],
                            "validation_performed": ["validate_userspace_code dashboard/main.ts"],
                            "remaining_risks": ["Parent still needs final smoke test"],
                        },
                        "output": '{"status":"accepted"}',
                    }
                ]
            ),
            response_content="freeform child text should not be used",
        )

        handoff, error = self.service._extract_subagent_handoff(task)

        self.assertIsNone(error)
        self.assertIsNotNone(handoff)
        assert handoff is not None
        self.assertEqual(handoff["final_output"], "## Work complete\n\nChanged the dashboard.")
        self.assertEqual(handoff["summary"], "Changed the dashboard")
        self.assertEqual(handoff["files_changed"], ["dashboard/main.ts"])
        self.assertEqual(handoff["validation_performed"], ["validate_userspace_code dashboard/main.ts"])
        self.assertEqual(handoff["remaining_risks"], ["Parent still needs final smoke test"])

    def test_extract_handoff_rejects_missing_tool_payload(self) -> None:
        task = SimpleNamespace(
            streaming_state=SimpleNamespace(tool_calls=[]),
            response_content="I did work but forgot the handoff tool",
        )

        handoff, error = self.service._extract_subagent_handoff(task)

        self.assertIsNone(handoff)
        self.assertIn(SUBAGENT_HANDOFF_TOOL_NAME, error or "")

    def test_private_prompt_is_concise_and_excludes_parent_linking_noise(self) -> None:
        spec = self.service._normalize_specs(
            [
                {
                    "name": "Alpha",
                    "role": "worker",
                    "instructions": "Create alpha/alpha.ts",
                    "file_scope": ["alpha"],
                }
            ]
        )[0]

        prompt = self.service._build_child_private_prompt(spec)

        self.assertIn("Role: worker", prompt)
        self.assertIn("Declared file scope: alpha", prompt)
        self.assertIn(SUBAGENT_HANDOFF_TOOL_NAME, prompt)
        self.assertNotIn("read-linked subagent", prompt)
        self.assertNotIn("parent conversation", prompt)
        self.assertNotIn("Create alpha/alpha.ts", prompt)


class SubAgentServiceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_registers_child_before_publish(self) -> None:
        service = SubAgentService()
        repo = SimpleNamespace()
        background = SimpleNamespace(
            started=[],
            registered=[],
        )

        async def create_conversation(**_: object) -> SimpleNamespace:
            return SimpleNamespace(id="child-conv-1")

        async def add_user_message_and_create_chat_task_if_idle(*_: object) -> tuple[None, SimpleNamespace, bool]:
            return None, SimpleNamespace(id="child-task-1"), True

        async def get_chat_task(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(id=task_id, status=ChatTaskStatus.completed, response_content="child done")

        async def get_conversation(_conversation_id: str) -> SimpleNamespace:
            return SimpleNamespace(messages=[])

        def start_task(*_: object, existing_task_id: str, **__: object) -> None:
            self.assertEqual(background.registered, [("parent-task-1", ["child-task-1"])])
            background.started.append(existing_task_id)

        def register_subagent_children(parent_task_id: str, child_task_ids: list[str]) -> None:
            background.registered.append((parent_task_id, list(child_task_ids)))

        async def await_task(_task_id: str) -> None:
            return None

        async def publish(_channel: str, payload: dict[str, object]) -> None:
            if payload.get("event") == "subagent_spawned":
                self.assertEqual(background.started, ["child-task-1"])
                self.assertEqual(background.registered, [("parent-task-1", ["child-task-1"])])

        repo.create_conversation = create_conversation
        repo.add_user_message_and_create_chat_task_if_idle = add_user_message_and_create_chat_task_if_idle
        repo.get_chat_task = get_chat_task
        repo.get_conversation = get_conversation
        background.start_task = start_task
        background.register_subagent_children = register_subagent_children
        background.await_task = await_task

        with (
            mock.patch("ragtime.userspace.subagent_service.repository", repo),
            mock.patch(
                "ragtime.userspace.subagent_service.task_event_bus",
                SimpleNamespace(publish=publish),
            ),
            mock.patch("ragtime.indexer.background_tasks.background_task_service", background),
        ):
            result = await service.spawn_subagents(
                parent_conversation_id="parent-conv-1",
                parent_task_id="parent-task-1",
                workspace_id="workspace-1",
                user_id="user-1",
                parent_model="openai::gpt-4.1",
                specs=[{"name": "Child", "role": "worker", "instructions": "Do work", "file_scope": ["dashboard"]}],
                workspace_context={"workspace_id": "workspace-1"},
                blocked_tool_names=set(),
            )

        self.assertIn("child-task-1", result)

    async def test_spawn_returns_structured_handoff_instead_of_freeform_response(self) -> None:
        service = SubAgentService()
        repo = SimpleNamespace()
        background = SimpleNamespace(started=[], registered=[])

        async def create_conversation(**_: object) -> SimpleNamespace:
            return SimpleNamespace(id="child-conv-1")

        async def add_user_message_and_create_chat_task_if_idle(*_: object) -> tuple[None, SimpleNamespace, bool]:
            return None, SimpleNamespace(id="child-task-1"), True

        async def get_chat_task(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=task_id,
                status=ChatTaskStatus.completed,
                response_content="freeform child text should not be returned",
                streaming_state=SimpleNamespace(
                    tool_calls=[
                        {
                            "tool": SUBAGENT_HANDOFF_TOOL_NAME,
                            "input": {
                                "final_output": "## Done\n\nParent-visible handoff.",
                                "summary": "Parent-visible handoff",
                                "files_changed": ["dashboard/main.ts"],
                                "validation_performed": ["pytest tests/test_subagent_service.py"],
                                "remaining_risks": [],
                            },
                            "output": '{"status":"accepted"}',
                        }
                    ]
                ),
            )

        async def get_conversation(_conversation_id: str) -> SimpleNamespace:
            return SimpleNamespace(messages=[])

        def start_task(*_: object, existing_task_id: str, **__: object) -> None:
            background.started.append(existing_task_id)

        def register_subagent_children(parent_task_id: str, child_task_ids: list[str]) -> None:
            background.registered.append((parent_task_id, list(child_task_ids)))

        async def await_task(_task_id: str) -> None:
            return None

        repo.create_conversation = create_conversation
        repo.add_user_message_and_create_chat_task_if_idle = add_user_message_and_create_chat_task_if_idle
        repo.get_chat_task = get_chat_task
        repo.get_conversation = get_conversation
        background.start_task = start_task
        background.register_subagent_children = register_subagent_children
        background.await_task = await_task

        with (
            mock.patch("ragtime.userspace.subagent_service.repository", repo),
            mock.patch(
                "ragtime.userspace.subagent_service.task_event_bus",
                SimpleNamespace(publish=mock.AsyncMock()),
            ),
            mock.patch("ragtime.indexer.background_tasks.background_task_service", background),
        ):
            raw_result = await service.spawn_subagents(
                parent_conversation_id="parent-conv-1",
                parent_task_id="parent-task-1",
                workspace_id="workspace-1",
                user_id="user-1",
                parent_model="openai::gpt-4.1",
                specs=[{"name": "Child", "role": "worker", "instructions": "Do work", "file_scope": ["dashboard"]}],
                workspace_context={"workspace_id": "workspace-1"},
                blocked_tool_names=set(),
            )

        result = json.loads(raw_result)
        child_result = result["subagents"][0]
        self.assertEqual(child_result["status"], "completed")
        self.assertEqual(child_result["final_output"], "## Done\n\nParent-visible handoff.")
        self.assertEqual(child_result["handoff"]["summary"], "Parent-visible handoff")
        self.assertNotIn("freeform child text", child_result["final_output"])

    async def test_spawn_persists_only_visible_assignment_not_private_subagent_context(self) -> None:
        service = SubAgentService()
        repo = SimpleNamespace()
        background = SimpleNamespace(started=[])
        stored_messages: list[str] = []
        runtime_messages: list[str] = []
        runtime_contexts: list[dict[str, Any]] = []

        async def create_conversation(**_: object) -> SimpleNamespace:
            return SimpleNamespace(id="child-conv-1")

        async def add_user_message_and_create_chat_task_if_idle(
            _conversation_id: str,
            user_message: str,
        ) -> tuple[None, SimpleNamespace, bool]:
            stored_messages.append(user_message)
            return None, SimpleNamespace(id="child-task-1"), True

        async def get_chat_task(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=task_id,
                status=ChatTaskStatus.completed,
                streaming_state=SimpleNamespace(
                    tool_calls=[
                        {
                            "tool": SUBAGENT_HANDOFF_TOOL_NAME,
                            "input": {"final_output": "Child done"},
                        }
                    ]
                ),
            )

        def start_task(
            _conversation_id: str,
            user_message: str,
            *_: object,
            workspace_context: dict[str, Any] | None = None,
            **__: object,
        ) -> None:
            runtime_messages.append(user_message)
            runtime_contexts.append(dict(workspace_context or {}))

        repo.create_conversation = create_conversation
        repo.add_user_message_and_create_chat_task_if_idle = add_user_message_and_create_chat_task_if_idle
        repo.get_chat_task = get_chat_task
        background.start_task = start_task
        background.register_subagent_children = mock.Mock()
        background.await_task = mock.AsyncMock()

        with (
            mock.patch("ragtime.userspace.subagent_service.repository", repo),
            mock.patch("ragtime.userspace.subagent_service.task_event_bus", SimpleNamespace(publish=mock.AsyncMock())),
            mock.patch("ragtime.indexer.background_tasks.background_task_service", background),
        ):
            await service.spawn_subagents(
                parent_conversation_id="parent-conv-1",
                parent_task_id="parent-task-1",
                workspace_id="workspace-1",
                user_id="user-1",
                parent_model="openai::gpt-4.1",
                specs=[
                    {
                        "name": "Child",
                        "role": "worker",
                        "instructions": "Create alpha/alpha.ts",
                        "file_scope": ["alpha"],
                    }
                ],
                workspace_context={"workspace_id": "workspace-1"},
                blocked_tool_names=set(),
            )

        self.assertEqual(stored_messages, ["Create alpha/alpha.ts"])
        self.assertEqual(runtime_messages, ["Create alpha/alpha.ts"])
        self.assertIn("alpha", runtime_contexts[0][SUBAGENT_PRIVATE_PROMPT_CONTEXT_KEY])
        self.assertNotIn("parent-conv-1", stored_messages[0])
        self.assertNotIn("read-linked subagent", stored_messages[0])
        self.assertNotIn("read-linked subagent", runtime_contexts[0][SUBAGENT_PRIVATE_PROMPT_CONTEXT_KEY])

    async def test_spawn_marks_child_failed_when_handoff_tool_missing(self) -> None:
        service = SubAgentService()
        repo = SimpleNamespace()
        background = SimpleNamespace(started=[], registered=[])

        async def create_conversation(**_: object) -> SimpleNamespace:
            return SimpleNamespace(id="child-conv-1")

        async def add_user_message_and_create_chat_task_if_idle(*_: object) -> tuple[None, SimpleNamespace, bool]:
            return None, SimpleNamespace(id="child-task-1"), True

        async def get_chat_task(task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                id=task_id,
                status=ChatTaskStatus.completed,
                response_content="freeform child text",
                streaming_state=SimpleNamespace(tool_calls=[]),
            )

        async def get_conversation(_conversation_id: str) -> SimpleNamespace:
            return SimpleNamespace(messages=[])

        def start_task(*_: object, existing_task_id: str, **__: object) -> None:
            background.started.append(existing_task_id)

        def register_subagent_children(parent_task_id: str, child_task_ids: list[str]) -> None:
            background.registered.append((parent_task_id, list(child_task_ids)))

        async def await_task(_task_id: str) -> None:
            return None

        repo.create_conversation = create_conversation
        repo.add_user_message_and_create_chat_task_if_idle = add_user_message_and_create_chat_task_if_idle
        repo.get_chat_task = get_chat_task
        repo.get_conversation = get_conversation
        background.start_task = start_task
        background.register_subagent_children = register_subagent_children
        background.await_task = await_task

        with (
            mock.patch("ragtime.userspace.subagent_service.repository", repo),
            mock.patch(
                "ragtime.userspace.subagent_service.task_event_bus",
                SimpleNamespace(publish=mock.AsyncMock()),
            ),
            mock.patch("ragtime.indexer.background_tasks.background_task_service", background),
        ):
            raw_result = await service.spawn_subagents(
                parent_conversation_id="parent-conv-1",
                parent_task_id="parent-task-1",
                workspace_id="workspace-1",
                user_id="user-1",
                parent_model="openai::gpt-4.1",
                specs=[{"name": "Child", "role": "worker", "instructions": "Do work", "file_scope": ["dashboard"]}],
                workspace_context={"workspace_id": "workspace-1"},
                blocked_tool_names=set(),
            )

        result = json.loads(raw_result)
        child_result = result["subagents"][0]
        self.assertEqual(child_result["status"], "failed")
        self.assertIn(SUBAGENT_HANDOFF_TOOL_NAME, child_result["final_output"])

    async def test_spawn_cancels_siblings_when_one_child_wait_fails(self) -> None:
        service = SubAgentService()
        child_conversations = iter(
            [
                SimpleNamespace(id="child-conv-1"),
                SimpleNamespace(id="child-conv-2"),
            ]
        )
        child_tasks = iter(
            [
                SimpleNamespace(id="child-task-1"),
                SimpleNamespace(id="child-task-2"),
            ]
        )
        background = SimpleNamespace(
            started=[],
            registered=[],
            cancelled=[],
            db_cancelled=[],
        )
        sibling_cancelled = asyncio.Event()

        async def create_conversation(**_: object) -> SimpleNamespace:
            return next(child_conversations)

        async def add_user_message_and_create_chat_task_if_idle(*_: object) -> tuple[None, SimpleNamespace, bool]:
            return None, next(child_tasks), True

        task_rows = {
            "child-task-1": SimpleNamespace(
                id="child-task-1",
                status=ChatTaskStatus.failed,
                response_content="",
            ),
            "child-task-2": SimpleNamespace(
                id="child-task-2",
                status=ChatTaskStatus.running,
                response_content="",
            ),
        }

        async def get_chat_task(task_id: str) -> SimpleNamespace:
            return task_rows[task_id]

        async def get_conversation(_conversation_id: str) -> SimpleNamespace:
            return SimpleNamespace(messages=[])

        async def cancel_chat_task(task_id: str) -> None:
            background.db_cancelled.append(task_id)
            task_rows[task_id].status = ChatTaskStatus.cancelled

        def start_task(*_: object, existing_task_id: str, **__: object) -> None:
            background.started.append(existing_task_id)

        def register_subagent_children(parent_task_id: str, child_task_ids: list[str]) -> None:
            background.registered.append((parent_task_id, list(child_task_ids)))

        def cancel_task(task_id: str) -> None:
            background.cancelled.append(task_id)
            if task_id == "child-task-2":
                sibling_cancelled.set()

        async def await_task(task_id: str) -> None:
            if task_id == "child-task-1":
                raise RuntimeError("boom")
            await sibling_cancelled.wait()
            raise asyncio.CancelledError()

        repo = SimpleNamespace(
            create_conversation=create_conversation,
            add_user_message_and_create_chat_task_if_idle=add_user_message_and_create_chat_task_if_idle,
            get_chat_task=get_chat_task,
            get_conversation=get_conversation,
            cancel_chat_task=cancel_chat_task,
        )
        background.start_task = start_task
        background.register_subagent_children = register_subagent_children
        background.cancel_task = cancel_task
        background.await_task = await_task

        with (
            mock.patch("ragtime.userspace.subagent_service.repository", repo),
            mock.patch(
                "ragtime.userspace.subagent_service.task_event_bus",
                SimpleNamespace(publish=mock.AsyncMock()),
            ),
            mock.patch("ragtime.indexer.background_tasks.background_task_service", background),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                await service.spawn_subagents(
                    parent_conversation_id="parent-conv-1",
                    parent_task_id="parent-task-1",
                    workspace_id="workspace-1",
                    user_id="user-1",
                    parent_model="openai::gpt-4.1",
                    specs=[
                        {
                            "name": "Child 1",
                            "role": "worker",
                            "instructions": "Do work 1",
                            "file_scope": ["dashboard/a"],
                        },
                        {
                            "name": "Child 2",
                            "role": "worker",
                            "instructions": "Do work 2",
                            "file_scope": ["dashboard/b"],
                        },
                    ],
                    workspace_context={"workspace_id": "workspace-1"},
                    blocked_tool_names=set(),
                )

        self.assertEqual(background.cancelled, ["child-task-2"])
        self.assertEqual(background.db_cancelled, ["child-task-2"])


class SubAgentModelOverrideTests(unittest.TestCase):
    def test_model_key_variants_match_scoped_bare_and_publisher_prefixed_forms(self) -> None:
        components = RAGComponents()

        self.assertFalse(
            components._model_key_variants_for_subagent_override("anthropic::claude-opus-4").isdisjoint(
                components._model_key_variants_for_subagent_override("claude-opus-4")
            )
        )
        self.assertFalse(
            components._model_key_variants_for_subagent_override("openrouter::anthropic/claude-opus-4").isdisjoint(
                components._model_key_variants_for_subagent_override("claude-opus-4")
            )
        )
        self.assertTrue(
            components._model_key_variants_for_subagent_override("openai::gpt-4.1").isdisjoint(
                components._model_key_variants_for_subagent_override("claude-opus-4")
            )
        )


class SubAgentToolScopeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_scoped_tools(
        self,
        *,
        workspace_context: dict[str, object] | None = None,
    ) -> list[StructuredTool]:
        components = RAGComponents()
        runtime_service = SimpleNamespace(
            get_global_mcp_tools=mock.AsyncMock(return_value=[]),
            list_workspace_mcp_tools=mock.AsyncMock(return_value=[]),
        )
        patcher = mock.patch("ragtime.rag.components.userspace_runtime_service", runtime_service)
        patcher.start()
        self.addCleanup(patcher.stop)
        return await components._create_userspace_file_tools(
            "workspace-1",
            "user-1",
            accessible_workspace_modes={"workspace-2": "read_write"},
            is_admin=False,
            subagent_file_scope=["dashboard"],
            workspace_context=workspace_context,
        )

    async def test_handoff_tool_is_available_for_subagent_child(self) -> None:
        tools = await self._build_scoped_tools(
            workspace_context={
                "subagent_parent_conversation_id": "parent-1",
                "subagent_depth": 1,
            }
        )

        self.assertIn(SUBAGENT_HANDOFF_TOOL_NAME, {tool.name for tool in tools})

    async def test_handoff_tool_is_omitted_for_parent_agent(self) -> None:
        tools = await self._build_scoped_tools(workspace_context={})

        self.assertNotIn(SUBAGENT_HANDOFF_TOOL_NAME, {tool.name for tool in tools})

    async def test_scoped_subagent_cannot_snapshot_full_workspace(self) -> None:
        tools = await self._build_scoped_tools()
        snapshot_tool = next(tool for tool in tools if tool.name == "create_userspace_snapshot")
        assert snapshot_tool.coroutine is not None

        with self.assertRaisesRegex(ToolException, "full workspace"):
            await snapshot_tool.coroutine(message="checkpoint")

    async def test_scoped_subagent_patch_rejects_cross_workspace_target(self) -> None:
        tools = await self._build_scoped_tools()
        patch_tool = next(tool for tool in tools if tool.name == "patch_userspace_file")
        assert patch_tool.coroutine is not None
        userspace = SimpleNamespace(
            resolve_cross_workspace_target=mock.AsyncMock(return_value=("workspace-2", "user-2")),
            enforce_workspace_role=mock.AsyncMock(),
        )

        with mock.patch("ragtime.rag.components.userspace_service", userspace):
            with self.assertRaisesRegex(ToolException, "active parent workspace"):
                await patch_tool.coroutine(
                    path="dashboard/main.ts",
                    replacements=[{"old_text": "a", "new_text": "b"}],
                    workspace_id="workspace-2",
                )

            with self.assertRaisesRegex(ToolException, "active parent workspace"):
                await patch_tool.coroutine(
                    patches=[{"path": "dashboard/main.ts", "replacements": [{"old_text": "a", "new_text": "b"}]}],
                    workspace_id="workspace-2",
                )


class SubAgentPromptTests(unittest.TestCase):
    def test_guidance_mentions_fanout_limit(self) -> None:
        self.assertIn("1 and 6", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("Never spawn more than 6", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)

    def test_guidance_mentions_empty_scope_for_reviewers(self) -> None:
        self.assertIn("pass an empty `file_scope`", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("`role=review`", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)

    def test_guidance_keeps_private_boilerplate_out_of_child_instructions(self) -> None:
        self.assertIn("user-visible task instruction", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("Do not include parent conversation IDs", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("runtime injects those privately", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)

    def test_guidance_forbids_recursive_spawning(self) -> None:
        self.assertIn("depth 1", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("cannot spawn additional subagents", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)

    def test_guidance_assigns_synthesis_to_parent(self) -> None:
        self.assertIn("You remain the parent agent", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("Do not delegate final validation or snapshot creation", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)

    def test_guidance_requires_non_overlapping_write_scopes(self) -> None:
        self.assertIn("non-overlapping `file_scope`", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)
        self.assertIn("worker may only modify files under its declared scope", USERSPACE_SUBAGENT_GUIDANCE_PROMPT)


class SubAgentRouteReadOnlyTests(unittest.TestCase):
    def test_subagent_conversation_is_read_only(self) -> None:
        conv = SimpleNamespace(parent_conversation_id="parent-1")
        with self.assertRaisesRegex(HTTPException, "read-only"):
            _assert_conversation_mutable(conv)

    def test_prisma_conversation_read_only(self) -> None:
        conv = SimpleNamespace(parentConversationId="parent-1")
        with self.assertRaisesRegex(HTTPException, "read-only"):
            _assert_conversation_mutable(conv)

    def test_parent_conversation_is_mutable(self) -> None:
        conv = SimpleNamespace(parent_conversation_id=None)
        try:
            _assert_conversation_mutable(conv)
        except HTTPException:
            self.fail("Parent conversation should be mutable")


class SubAgentToolGatingTests(unittest.IsolatedAsyncioTestCase):
    async def _create_spawn_tool(self, *, parent: Any, workspace_context: dict[str, object] | None = None) -> StructuredTool | None:
        components = RAGComponents()
        with mock.patch(
            "ragtime.rag.components.repository",
            new=SimpleNamespace(get_conversation=mock.AsyncMock(return_value=parent)),
        ):
            return await components._create_spawn_subagents_tool(
                workspace_id="workspace-1",
                user_id="user-1",
                parent_conversation_id="parent-1",
                parent_task_id="task-1",
                parent_model="openai::gpt-4.1",
                workspace_context=workspace_context or {},
                blocked_tool_names=set(),
                disabled_builtin_tool_ids=None,
                current_time_context=None,
            )

    async def test_spawn_tool_created_for_parent_conversation(self) -> None:
        parent = SimpleNamespace(
            id="parent-1",
            parent_conversation_id=None,
            subagents_enabled=True,
            model="openai::gpt-4.1",
        )
        tool = await self._create_spawn_tool(parent=parent)
        self.assertIsNotNone(tool)
        assert tool is not None
        self.assertEqual(tool.name, "spawn_subagents")

    async def test_spawn_tool_omitted_for_subagent_child(self) -> None:
        parent = SimpleNamespace(
            id="parent-1",
            parent_conversation_id="grandparent-1",
            subagents_enabled=True,
            model="openai::gpt-4.1",
        )
        tool = await self._create_spawn_tool(parent=parent)
        self.assertIsNone(tool)

    async def test_spawn_tool_omitted_when_subagents_disabled(self) -> None:
        parent = SimpleNamespace(
            id="parent-1",
            parent_conversation_id=None,
            subagents_enabled=False,
            model="openai::gpt-4.1",
        )
        tool = await self._create_spawn_tool(parent=parent)
        self.assertIsNone(tool)

    async def test_spawn_tool_omitted_at_max_depth(self) -> None:
        parent = SimpleNamespace(
            id="parent-1",
            parent_conversation_id=None,
            subagents_enabled=True,
            model="openai::gpt-4.1",
        )
        tool = await self._create_spawn_tool(
            parent=parent,
            workspace_context={"subagent_depth": 1},
        )
        self.assertIsNone(tool)

    async def test_spawn_tool_omitted_when_blocked(self) -> None:
        parent = SimpleNamespace(
            id="parent-1",
            parent_conversation_id=None,
            subagents_enabled=True,
            model="openai::gpt-4.1",
        )
        components = RAGComponents()
        with mock.patch(
            "ragtime.rag.components.repository",
            new=SimpleNamespace(get_conversation=mock.AsyncMock(return_value=parent)),
        ):
            tool = await components._create_spawn_subagents_tool(
                workspace_id="workspace-1",
                user_id="user-1",
                parent_conversation_id="parent-1",
                parent_task_id="task-1",
                parent_model="openai::gpt-4.1",
                workspace_context={},
                blocked_tool_names={"spawn_subagents"},
                disabled_builtin_tool_ids=None,
                current_time_context=None,
            )
        self.assertIsNone(tool)


class SubAgentCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_parent_cancels_registered_children(self) -> None:
        from ragtime.indexer.background_tasks import BackgroundTaskService

        service = BackgroundTaskService()

        async def do_nothing() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass

        parent_task = asyncio.create_task(do_nothing())
        child_task = asyncio.create_task(do_nothing())
        service._running_tasks["parent-1"] = parent_task
        service._running_tasks["child-1"] = child_task
        service.register_subagent_children("parent-1", ["child-1"])

        cancelled = service.cancel_task("parent-1")
        # Allow the event loop to process cancellation.
        await asyncio.sleep(0)

        self.assertTrue(cancelled)
        self.assertTrue(child_task.cancelled() or child_task.done())


if __name__ == "__main__":
    unittest.main()
