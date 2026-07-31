from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ragtime.rag import components as rag_components
from ragtime.rag.tool_skills import ToolSkillBindingState, ToolSkillDefinition, build_tool_skill_control_tools


def _make_tool(name: str, *, description: str | None = None) -> StructuredTool:
    async def _tool(**_kwargs: Any) -> str:
        return name

    return StructuredTool.from_function(
        coroutine=_tool,
        name=name,
        description=description or f"Tool {name}",
    )


class _FakeAction:
    def __init__(self, tool: str, tool_input: dict[str, Any], tool_call_id: str) -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.tool_call_id = tool_call_id

    @property
    def message_log(self) -> list[AIMessage]:
        return [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": self.tool,
                        "args": self.tool_input,
                        "id": self.tool_call_id,
                        "type": "tool_call",
                    }
                ],
            )
        ]


class _FakeExecutor:
    def __init__(self, tools: list[Any], results: list[dict[str, Any]]) -> None:
        self.tools = tools
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self._results:
            raise AssertionError("No more fake executor results")
        return self._results.pop(0)


class _FakeStreamExecutor:
    def __init__(self, tools: list[Any], streams: list[list[dict[str, Any]]]) -> None:
        self.tools = tools
        self._streams = list(streams)

    def astream_events(self, _payload: dict[str, Any], version: str = "v2"):
        async def _gen():
            if not self._streams:
                raise AssertionError("No more fake stream stages")
            for event in self._streams.pop(0):
                yield event

        return _gen()


class ToolSkillLoadingTests(unittest.IsolatedAsyncioTestCase):
    def _make_rag(self) -> rag_components.RAGComponents:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {
                "id": "tool-1",
                "name": "Demo SQL",
                "tool_type": "postgres",
                "description": "Reads demo SQL rows.",
                "connection_config": {},
            }
        ]
        rag._index_metadata = []
        rag._request_prompt_cache = {}
        rag._app_settings = {"tool_skills_enabled": True, "max_iterations": 4}
        return rag

    def test_request_system_prompt_hides_unloaded_optional_tool_configs(self) -> None:
        rag = self._make_rag()

        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=[_make_tool("search_tool_skills")],
            hidden_tool_config_ids=["tool-1"],
            tool_skill_mode="enabled",
            loaded_tool_skill_ids=[],
            tool_skill_has_loadable=True,
        )

        self.assertIn("No optional system tools are currently loaded", prompt)
        self.assertIn("search_tool_skills", prompt)
        self.assertIn("load_tool_skills", prompt)
        self.assertNotIn("Demo SQL", prompt)
        self.assertNotIn("CONFIGURED BUT UNAVAILABLE", prompt)

    async def test_feature_off_preserves_eager_runtime_tools_without_controls(self) -> None:
        rag = self._make_rag()
        cast(dict[str, Any], rag._app_settings)["tool_skills_enabled"] = False
        query_tool = _make_tool("query_demo_sql")

        binding = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=[query_tool],
            mode="chat",
            conversation_id="conversation-1",
        )

        self.assertEqual([tool.name for tool in binding["runtime_tools"]], ["query_demo_sql"])
        self.assertEqual(binding["tool_skill_mode"], "disabled")

    async def test_persisted_requested_ids_survive_when_tool_becomes_ineligible(self) -> None:
        rag = self._make_rag()
        query_tool = _make_tool("query_demo_sql")

        with mock.patch.object(
            rag_components.repository,
            "get_conversation_loaded_tool_skill_ids",
            new=mock.AsyncMock(return_value=["tool_config:tool-1"]),
        ):
            binding = await rag._resolve_request_tool_skill_bindings(
                runtime_tools=[query_tool],
                mode="chat",
                conversation_id="conversation-1",
                allowed_tool_config_ids=[],
            )

        self.assertEqual(binding["tool_skill_binding_state"].requested_ids, ["tool_config:tool-1"])
        self.assertEqual(binding["tool_skill_binding_state"].effective_ids, [])
        self.assertEqual({tool.name for tool in binding["runtime_tools"]}, {"search_tool_skills", "load_tool_skills", "unload_tool_skills"})

    async def test_disallowed_config_derived_tool_names_never_fall_through_as_builtins(self) -> None:
        rag = self._make_rag()
        query_tool = _make_tool("query_demo_sql")

        binding = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=[query_tool],
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=[],
        )

        self.assertFalse(binding["tool_skill_has_loadable"])
        self.assertEqual(binding["tool_skill_hidden_ids"], [])
        self.assertEqual(binding["tool_skill_catalog"], [])
        self.assertEqual({tool.name for tool in binding["runtime_tools"]}, {"search_tool_skills", "load_tool_skills", "unload_tool_skills"})

    def test_request_system_prompt_cache_isolates_loaded_and_hidden_skill_state(self) -> None:
        rag = self._make_rag()

        unloaded_prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=[_make_tool("search_tool_skills")],
            hidden_tool_config_ids=["tool-1"],
            tool_skill_mode="enabled",
            loaded_tool_skill_ids=[],
            tool_skill_has_loadable=True,
        )
        loaded_prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=[_make_tool("query_demo_sql")],
            hidden_tool_config_ids=[],
            tool_skill_mode="enabled",
            loaded_tool_skill_ids=["tool_config:tool-1"],
            tool_skill_has_loadable=True,
        )

        self.assertNotEqual(unloaded_prompt, loaded_prompt)
        self.assertIn("Demo SQL", loaded_prompt)
        self.assertNotIn("Demo SQL", unloaded_prompt)

    async def test_resolve_request_tool_skill_bindings_keeps_stateless_state_in_memory_only(self) -> None:
        rag = self._make_rag()
        query_tool = _make_tool("query_demo_sql")

        with mock.patch.object(
            rag_components.repository,
            "get_conversation_loaded_tool_skill_ids",
            new=mock.AsyncMock(),
        ) as get_loaded:
            binding = await rag._resolve_request_tool_skill_bindings(
                runtime_tools=[query_tool],
                mode="chat",
                conversation_id=None,
            )

        get_loaded.assert_not_awaited()
        self.assertEqual(binding["tool_skill_binding_state"].requested_ids, [])
        control_names = {tool.name for tool in binding["runtime_tools"]}
        self.assertTrue({"search_tool_skills", "load_tool_skills", "unload_tool_skills"}.issubset(control_names))
        self.assertNotIn("query_demo_sql", control_names)

    async def test_nonstream_stage_transition_rebuilds_executor_with_loaded_tool(self) -> None:
        rag = self._make_rag()
        binding_state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        original_history: list[BaseMessage] = [HumanMessage(content="earlier")]
        user_input = "load then query"

        load_step = (
            _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
            json.dumps(
                {
                    "status": "ok",
                    "requested_ids": ["tool_config:tool-1"],
                    "effective_ids": ["tool_config:tool-1"],
                    "bindings_changed": True,
                    "transition_kind": "load",
                }
            ),
        )
        query_step = (
            _FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"),
            "demo rows",
        )
        first_executor = _FakeExecutor([_make_tool("search_tool_skills")], [{"output": "ignored", "intermediate_steps": [load_step]}])
        second_executor = _FakeExecutor([_make_tool("query_demo_sql")], [{"output": "final answer", "intermediate_steps": [query_step]}])

        rebuilt_history: list[list[Any]] = []

        async def _rebuild_stage(*, tool_skill_binding_state_override: ToolSkillBindingState, **_kwargs: Any) -> dict[str, Any]:
            self.assertIs(tool_skill_binding_state_override, binding_state)
            return {
                "runtime_tools": [_make_tool("query_demo_sql")],
                "tool_skill_binding_state": tool_skill_binding_state_override,
                "tool_skill_hidden_ids": [],
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": ["tool_config:tool-1"],
                "allowed_tool_config_ids": ["tool-1"],
                "mode": "chat",
                "prompt_is_ui": False,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": {},
                "export_context": {},
                "workspace_id": None,
            }

        def _build_executor(tools: list[Any], *_args: Any, **_kwargs: Any) -> _FakeExecutor:
            return second_executor if any(tool.name == "query_demo_sql" for tool in tools) else first_executor

        async def _prepare_context_window(**kwargs: Any):
            rebuilt_history.append(list(kwargs["chat_history"]))
            return SimpleNamespace(llm=None, provider="openai", model="gpt-test"), kwargs["chat_history"], kwargs["turn_system_content"]

        binding_state.requested_ids = ["tool_config:tool-1"]
        binding_state.effective_ids = ["tool_config:tool-1"]

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=_build_executor),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(side_effect=_prepare_context_window)),
        ):
            output, rebuilt_executor, rebuilt_chat_history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=cast(list[BaseMessage], original_history),
                user_content=user_input,
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_tool("search_tool_skills")],
                    "prompt_additions": "",
                    "request_tool_state": {},
                    "tool_skill_binding_state": binding_state,
                    "tool_skill_hidden_ids": ["tool-1"],
                    "tool_skill_mode": "enabled",
                    "tool_skill_loaded_ids": [],
                    "include_sqlite_persistence": False,
                    "userspace_env_var_turn_hint": "",
                    "userspace_runtime_status_turn_hint": "",
                    "userspace_diagnostics_turn_hint": "",
                    "user_identity_turn_line": "",
                    "current_time_turn_line": "",
                    "workspace_id": None,
                },
                system_prompt="system",
                tool_scope_prompt="",
                turn_system_content="",
                conversation_id=None,
                conversation_model=None,
                user_id=None,
                blocked_tool_names=None,
                workspace_context=None,
                current_user_context=None,
                current_time_context=None,
                chat_task_id=None,
                message_index=None,
                disabled_builtin_tool_ids=None,
                max_iterations=4,
            )

        self.assertEqual(output, "final answer")
        self.assertIs(rebuilt_executor, second_executor)
        self.assertEqual([tool.name for tool in rebuilt_executor.tools], ["query_demo_sql"])
        self.assertIsInstance(rebuilt_chat_history[-2], AIMessage)
        self.assertIsInstance(rebuilt_chat_history[-1], ToolMessage)
        last_message = rebuilt_chat_history[-1]
        assert isinstance(last_message, ToolMessage)
        self.assertEqual(last_message.tool_call_id, "call-load")
        self.assertGreaterEqual(len(rebuilt_history), 1)

    async def test_nonstream_stage_loop_rebuilds_from_original_history_across_two_transitions(self) -> None:
        rag = self._make_rag()
        binding_state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        original_history: list[BaseMessage] = [HumanMessage(content="earlier")]
        load_step = (
            _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
            json.dumps(
                {
                    "status": "ok",
                    "bindings_changed": True,
                    "transition_kind": "load",
                    "requested_ids": ["tool_config:tool-1"],
                    "effective_ids": ["tool_config:tool-1"],
                }
            ),
        )
        unload_step = (
            _FakeAction("unload_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-unload"),
            json.dumps({"status": "ok", "bindings_changed": True, "transition_kind": "unload", "requested_ids": [], "effective_ids": []}),
        )
        first_executor = _FakeExecutor([_make_tool("search_tool_skills")], [{"output": "ignored", "intermediate_steps": [load_step]}])
        second_executor = _FakeExecutor([_make_tool("query_demo_sql")], [{"output": "ignored", "intermediate_steps": [unload_step]}])
        third_executor = _FakeExecutor([_make_tool("search_tool_skills")], [{"output": "done", "intermediate_steps": []}])
        rebuilt_histories: list[list[Any]] = []

        async def _rebuild_stage(*, tool_skill_binding_state_override: ToolSkillBindingState, **_kwargs: Any) -> dict[str, Any]:
            loaded = list(tool_skill_binding_state_override.effective_ids)
            runtime_tools = [_make_tool("query_demo_sql")] if loaded else [_make_tool("search_tool_skills")]
            return {
                "runtime_tools": runtime_tools,
                "tool_skill_binding_state": tool_skill_binding_state_override,
                "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": loaded,
                "allowed_tool_config_ids": ["tool-1"],
                "mode": "chat",
                "prompt_is_ui": False,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": {},
                "export_context": {},
                "workspace_id": None,
            }

        def _build_executor(tools: list[Any], *_args: Any, **_kwargs: Any) -> _FakeExecutor:
            names = {tool.name for tool in tools}
            if "query_demo_sql" in names:
                return second_executor
            if third_executor._results:
                return third_executor
            return first_executor

        async def _prepare_context_window(**kwargs: Any):
            rebuilt_histories.append(list(kwargs["chat_history"]))
            return SimpleNamespace(llm=None, provider="openai", model="gpt-test"), kwargs["chat_history"], kwargs["turn_system_content"]

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=_build_executor),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(side_effect=_prepare_context_window)),
        ):
            output, _rebuilt_executor, rebuilt_chat_history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=cast(list[BaseMessage], original_history),
                user_content="refit",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_tool("search_tool_skills")],
                    "prompt_additions": "",
                    "request_tool_state": {},
                    "tool_skill_binding_state": binding_state,
                    "tool_skill_hidden_ids": ["tool-1"],
                    "tool_skill_mode": "enabled",
                    "tool_skill_loaded_ids": [],
                    "include_sqlite_persistence": False,
                    "userspace_env_var_turn_hint": "",
                    "userspace_runtime_status_turn_hint": "",
                    "userspace_diagnostics_turn_hint": "",
                    "user_identity_turn_line": "",
                    "current_time_turn_line": "",
                    "workspace_id": None,
                },
                system_prompt="system",
                tool_scope_prompt="",
                turn_system_content="",
                conversation_id=None,
                conversation_model=None,
                user_id=None,
                blocked_tool_names=None,
                workspace_context=None,
                current_user_context=None,
                current_time_context=None,
                chat_task_id=None,
                message_index=None,
                disabled_builtin_tool_ids=None,
                max_iterations=6,
            )

        self.assertEqual(output, "done")
        self.assertEqual(sum(isinstance(message, HumanMessage) and message.content == "refit" for message in rebuilt_chat_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-load" for message in rebuilt_chat_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-unload" for message in rebuilt_chat_history), 1)
        self.assertGreaterEqual(len(rebuilt_histories), 2)

    async def test_nonstream_stage_loop_caps_binding_transitions(self) -> None:
        rag = self._make_rag()
        binding_state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        transition_step = (
            _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
            json.dumps(
                {
                    "status": "ok",
                    "bindings_changed": True,
                    "transition_kind": "load",
                    "requested_ids": ["tool_config:tool-1"],
                    "effective_ids": ["tool_config:tool-1"],
                }
            ),
        )
        executors = [_FakeExecutor([_make_tool("search_tool_skills")], [{"output": "ignored", "intermediate_steps": [transition_step]}]) for _ in range(6)]

        async def _rebuild_stage(*, tool_skill_binding_state_override: ToolSkillBindingState, **_kwargs: Any) -> dict[str, Any]:
            tool_skill_binding_state_override.effective_ids = ["tool_config:tool-1"]
            return {
                "runtime_tools": [_make_tool("query_demo_sql")],
                "tool_skill_binding_state": tool_skill_binding_state_override,
                "tool_skill_hidden_ids": [],
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": ["tool_config:tool-1"],
                "allowed_tool_config_ids": ["tool-1"],
                "mode": "chat",
                "prompt_is_ui": False,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": {},
                "export_context": {},
                "workspace_id": None,
            }

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag, "_prepare_chat_context_window", new=mock.AsyncMock(return_value=(SimpleNamespace(llm=None, provider="openai", model="gpt-test"), [], ""))
            ),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=lambda *_args, **_kwargs: executors.pop(0)),
        ):
            output, _executor, _chat_history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=executors.pop(0),
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[],
                user_content="cap",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_tool("search_tool_skills")],
                    "prompt_additions": "",
                    "request_tool_state": {},
                    "tool_skill_binding_state": binding_state,
                    "tool_skill_hidden_ids": ["tool-1"],
                    "tool_skill_mode": "enabled",
                    "tool_skill_loaded_ids": [],
                    "include_sqlite_persistence": False,
                    "userspace_env_var_turn_hint": "",
                    "userspace_runtime_status_turn_hint": "",
                    "userspace_diagnostics_turn_hint": "",
                    "user_identity_turn_line": "",
                    "current_time_turn_line": "",
                    "workspace_id": None,
                },
                system_prompt="system",
                tool_scope_prompt="",
                turn_system_content="",
                conversation_id=None,
                conversation_model=None,
                user_id=None,
                blocked_tool_names=None,
                workspace_context=None,
                current_user_context=None,
                current_time_context=None,
                chat_task_id=None,
                message_index=None,
                disabled_builtin_tool_ids=None,
                max_iterations=8,
            )

        self.assertIn("changed too many times", output)

    async def test_nonstream_search_control_replays_and_continues_without_transition(self) -> None:
        rag = self._make_rag()
        request_state: dict[str, Any] = {
            "tool_calls": [],
            "signature_counts": {"same": 2},
            "blocked_repeat_calls": 1,
            "max_iterations_reached": False,
            "internal_continue_attempts": 0,
            "internal_continue_stop_reason": "",
            "tool_free_synthesis_used": False,
        }
        first_executor = _FakeExecutor(
            [_make_tool("search_tool_skills")],
            [
                {
                    "output": json.dumps({"status": "ok", "results": [{"id": "tool_config:tool-1"}]}),
                    "intermediate_steps": [
                        (
                            _FakeAction("search_tool_skills", {"query": "demo"}, "call-search"),
                            json.dumps({"status": "ok", "results": [{"id": "tool_config:tool-1"}]}),
                        )
                    ],
                }
            ],
        )
        second_executor = _FakeExecutor(
            [_make_tool("search_tool_skills")],
            [{"output": "final answer", "intermediate_steps": []}],
        )
        rebuilt_histories: list[list[Any]] = []

        async def _rebuild_stage(**_kwargs: Any) -> dict[str, Any]:
            return {
                "runtime_tools": [_make_tool("search_tool_skills")],
                "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
                "tool_skill_hidden_ids": ["tool-1"],
                "tool_skill_has_loadable": True,
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": [],
                "allowed_tool_config_ids": ["tool-1"],
                "mode": "chat",
                "prompt_is_ui": False,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": request_state,
                "export_context": {},
                "workspace_id": None,
            }

        async def _prepare_context_window(**kwargs: Any):
            rebuilt_histories.append(list(kwargs["chat_history"]))
            return SimpleNamespace(llm=None, provider="openai", model="gpt-test"), kwargs["chat_history"], kwargs["turn_system_content"]

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[second_executor]),
            mock.patch.object(rag, "_prepare_chat_context_window", new=mock.AsyncMock(side_effect=_prepare_context_window)),
        ):
            output, _executor, rebuilt_history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[HumanMessage(content="earlier")],
                user_content="search first",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_tool("search_tool_skills")],
                    "prompt_additions": "",
                    "request_tool_state": request_state,
                    "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
                    "tool_skill_hidden_ids": ["tool-1"],
                    "tool_skill_has_loadable": True,
                    "tool_skill_mode": "enabled",
                    "tool_skill_loaded_ids": [],
                    "include_sqlite_persistence": False,
                    "userspace_env_var_turn_hint": "",
                    "userspace_runtime_status_turn_hint": "",
                    "userspace_diagnostics_turn_hint": "",
                    "user_identity_turn_line": "",
                    "current_time_turn_line": "",
                    "workspace_id": None,
                },
                system_prompt="system",
                tool_scope_prompt="",
                turn_system_content="",
                conversation_id=None,
                conversation_model=None,
                user_id=None,
                blocked_tool_names=None,
                workspace_context=None,
                current_user_context=None,
                current_time_context=None,
                chat_task_id=None,
                message_index=None,
                disabled_builtin_tool_ids=None,
                max_iterations=4,
            )

        self.assertEqual(output, "final answer")
        self.assertEqual(request_state["tool_skill_transition_count"], 0)
        self.assertEqual(request_state["signature_counts"]["same"], 2)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-search" for message in rebuilt_history), 1)
        self.assertGreaterEqual(len(rebuilt_histories), 1)

    async def test_search_control_tool_returns_ukg_skill_for_production_query(self) -> None:
        tools = build_tool_skill_control_tools(
            eligible_catalog=[
                ToolSkillDefinition(
                    id="tool_config:ukg-id",
                    label="UKG API",
                    description="Configured HTTP API tool.",
                    tool_names=["request_ukg_api"],
                    tool_config_ids=["ukg-id"],
                    kind="http_api",
                ),
                ToolSkillDefinition(
                    id="tool_config:unrelated-id",
                    label="Unrelated ERP Query",
                    description="Configured Odoo query tool.",
                    tool_names=["query_odoo"],
                    tool_config_ids=["unrelated-id"],
                    kind="odoo",
                ),
            ],
            binding_state=ToolSkillBindingState(requested_ids=[], effective_ids=[]),
            persist_requested_ids=mock.AsyncMock(),
        )

        payload = json.loads(await tools["search_tool_skills"].ainvoke({"query": "UKG employees pay-info cost-centers cost-center-jobs API GET"}))

        result_ids = [result["id"] for result in payload["results"]]
        self.assertEqual(result_ids, ["tool_config:ukg-id"])
        self.assertNotIn("tool_config:unrelated-id", result_ids)

    async def test_nonstream_mixed_control_and_real_steps_preserve_pairs_once_and_continue(self) -> None:
        rag = self._make_rag()
        first_executor = _FakeExecutor(
            [_make_tool("search_tool_skills"), _make_tool("query_demo_sql")],
            [
                {
                    "output": "ignored",
                    "intermediate_steps": [
                        (
                            _FakeAction("search_tool_skills", {"query": "demo"}, "call-search"),
                            json.dumps({"status": "ok", "results": [{"id": "tool_config:tool-1"}]}),
                        ),
                        (_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"), "rows"),
                    ],
                }
            ],
        )
        second_executor = _FakeExecutor([_make_tool("query_demo_sql")], [{"output": "recovered guidance", "intermediate_steps": []}])
        with (
            mock.patch.object(
                rag,
                "_build_request_runtime_context",
                new=mock.AsyncMock(
                    return_value={
                        "runtime_tools": [_make_tool("search_tool_skills"), _make_tool("query_demo_sql")],
                        "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
                        "tool_skill_hidden_ids": ["tool-1"],
                        "tool_skill_has_loadable": True,
                        "tool_skill_mode": "enabled",
                        "tool_skill_loaded_ids": [],
                        "allowed_tool_config_ids": ["tool-1"],
                        "mode": "chat",
                        "prompt_is_ui": False,
                        "prompt_additions": "",
                        "include_sqlite_persistence": False,
                        "userspace_env_var_turn_hint": "",
                        "userspace_runtime_status_turn_hint": "",
                        "userspace_diagnostics_turn_hint": "",
                        "user_identity_turn_line": "",
                        "current_time_turn_line": "",
                        "request_tool_state": {
                            "tool_calls": [],
                            "signature_counts": {},
                            "blocked_repeat_calls": 0,
                            "max_iterations_reached": False,
                            "internal_continue_attempts": 0,
                            "internal_continue_stop_reason": "",
                            "tool_free_synthesis_used": False,
                        },
                        "export_context": {},
                        "workspace_id": None,
                    }
                ),
            ),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[second_executor]),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
        ):
            output, _executor, rebuilt_history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[],
                user_content="mixed",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_tool("search_tool_skills"), _make_tool("query_demo_sql")],
                    "prompt_additions": "",
                    "request_tool_state": {
                        "tool_calls": [],
                        "signature_counts": {},
                        "blocked_repeat_calls": 0,
                        "max_iterations_reached": False,
                        "internal_continue_attempts": 0,
                        "internal_continue_stop_reason": "",
                        "tool_free_synthesis_used": False,
                    },
                    "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
                    "tool_skill_hidden_ids": ["tool-1"],
                    "tool_skill_has_loadable": True,
                    "tool_skill_mode": "enabled",
                    "tool_skill_loaded_ids": [],
                    "include_sqlite_persistence": False,
                    "userspace_env_var_turn_hint": "",
                    "userspace_runtime_status_turn_hint": "",
                    "userspace_diagnostics_turn_hint": "",
                    "user_identity_turn_line": "",
                    "current_time_turn_line": "",
                    "workspace_id": None,
                },
                system_prompt="system",
                tool_scope_prompt="",
                turn_system_content="",
                conversation_id=None,
                conversation_model=None,
                user_id=None,
                blocked_tool_names=None,
                workspace_context=None,
                current_user_context=None,
                current_time_context=None,
                chat_task_id=None,
                message_index=None,
                disabled_builtin_tool_ids=None,
                max_iterations=4,
            )

        self.assertEqual(output, "recovered guidance")
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-search" for message in rebuilt_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-query" for message in rebuilt_history), 1)

    async def test_streaming_transition_preempts_internal_continue_and_replays_once(self) -> None:
        rag = self._make_rag()
        first_executor = _FakeStreamExecutor(
            [_make_tool("search_tool_skills")],
            [
                [
                    {"event": "on_tool_start", "run_id": "call-load", "name": "load_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                    {
                        "event": "on_tool_end",
                        "run_id": "call-load",
                        "name": "load_tool_skills",
                        "data": {
                            "output": json.dumps(
                                {
                                    "status": "ok",
                                    "bindings_changed": True,
                                    "transition_kind": "load",
                                    "requested_ids": ["tool_config:tool-1"],
                                    "effective_ids": ["tool_config:tool-1"],
                                }
                            )
                        },
                    },
                    {
                        "event": "on_chain_end",
                        "data": {
                            "output": {
                                "intermediate_steps": [
                                    (
                                        _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
                                        json.dumps(
                                            {
                                                "status": "ok",
                                                "bindings_changed": True,
                                                "transition_kind": "load",
                                                "requested_ids": ["tool_config:tool-1"],
                                                "effective_ids": ["tool_config:tool-1"],
                                            }
                                        ),
                                    )
                                ],
                                "output": "ignored",
                            }
                        },
                    },
                ]
            ],
        )
        setattr(rag, "agent_executor", first_executor)
        second_executor = _FakeStreamExecutor(
            [_make_tool("query_demo_sql")],
            [
                [
                    {"event": "on_tool_start", "run_id": "call-query", "name": "query_demo_sql", "data": {"input": {"query": "select 1 limit 1"}}},
                    {"event": "on_tool_end", "run_id": "call-query", "name": "query_demo_sql", "data": {"output": "rows"}},
                    {"event": "on_chat_model_stream", "run_id": "chat-2", "data": {"chunk": SimpleNamespace(content="final stream")}},
                    {
                        "event": "on_chain_end",
                        "data": {
                            "output": {
                                "intermediate_steps": [(_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"), "rows")],
                                "output": "final stream",
                            }
                        },
                    },
                ]
            ],
        )

        async def _build_runtime_context(*, tool_skill_binding_state_override: ToolSkillBindingState | None = None, **_kwargs: Any) -> dict[str, Any]:
            loaded = list((tool_skill_binding_state_override or ToolSkillBindingState([], [])).effective_ids)
            runtime_tools = [_make_tool("query_demo_sql")] if loaded else [_make_tool("search_tool_skills")]
            return {
                "mode": "chat",
                "prompt_is_ui": False,
                "allowed_tool_config_ids": ["tool-1"],
                "runtime_tools": runtime_tools,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": {
                    "tool_calls": [],
                    "signature_counts": {},
                    "blocked_repeat_calls": 0,
                    "max_iterations_reached": False,
                    "internal_continue_attempts": 0,
                    "internal_continue_stop_reason": "",
                    "tool_free_synthesis_used": False,
                },
                "export_context": {},
                "workspace_id": None,
                "tool_skill_binding_state": tool_skill_binding_state_override or ToolSkillBindingState([], []),
                "tool_skill_catalog": [],
                "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": loaded,
            }

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="load then stream")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_build_runtime_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_seed_latest_export_context_from_chat_history", return_value=None),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[getattr(rag, "agent_executor"), second_executor]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()),
        ):
            events = [event async for event in rag.process_query_stream("hello", chat_history=[])]

        tool_end_names = [event["tool"] for event in events if isinstance(event, dict) and event.get("type") == "tool_end"]
        self.assertEqual(tool_end_names, ["load_tool_skills", "query_demo_sql"])
        self.assertIn("final stream", "".join(event for event in events if isinstance(event, str)))

    async def test_streaming_two_transitions_replay_completed_pairs_once(self) -> None:
        rag = self._make_rag()
        first_executor = _FakeStreamExecutor(
            [_make_tool("search_tool_skills")],
            [
                [
                    {"event": "on_tool_start", "run_id": "call-load", "name": "load_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                    {
                        "event": "on_tool_end",
                        "run_id": "call-load",
                        "name": "load_tool_skills",
                        "data": {
                            "output": json.dumps(
                                {
                                    "status": "ok",
                                    "bindings_changed": True,
                                    "transition_kind": "load",
                                    "requested_ids": ["tool_config:tool-1"],
                                    "effective_ids": ["tool_config:tool-1"],
                                }
                            )
                        },
                    },
                    {
                        "event": "on_chain_end",
                        "data": {
                            "output": {
                                "intermediate_steps": [
                                    (
                                        _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
                                        json.dumps(
                                            {
                                                "status": "ok",
                                                "bindings_changed": True,
                                                "transition_kind": "load",
                                                "requested_ids": ["tool_config:tool-1"],
                                                "effective_ids": ["tool_config:tool-1"],
                                            }
                                        ),
                                    )
                                ],
                                "output": "ignored",
                            }
                        },
                    },
                ]
            ],
        )
        setattr(rag, "agent_executor", first_executor)
        second_executor = _FakeStreamExecutor(
            [_make_tool("query_demo_sql")],
            [
                [
                    {"event": "on_tool_start", "run_id": "call-unload", "name": "unload_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                    {
                        "event": "on_tool_end",
                        "run_id": "call-unload",
                        "name": "unload_tool_skills",
                        "data": {
                            "output": json.dumps(
                                {"status": "ok", "bindings_changed": True, "transition_kind": "unload", "requested_ids": [], "effective_ids": []}
                            )
                        },
                    },
                    {
                        "event": "on_chain_end",
                        "data": {
                            "output": {
                                "intermediate_steps": [
                                    (
                                        _FakeAction("unload_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-unload"),
                                        json.dumps(
                                            {"status": "ok", "bindings_changed": True, "transition_kind": "unload", "requested_ids": [], "effective_ids": []}
                                        ),
                                    )
                                ],
                                "output": "ignored",
                            }
                        },
                    },
                ]
            ],
        )
        third_executor = _FakeStreamExecutor(
            [_make_tool("search_tool_skills")],
            [
                [
                    {"event": "on_chat_model_stream", "run_id": "chat-3", "data": {"chunk": SimpleNamespace(content="done")}},
                    {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [], "output": "done"}}},
                ]
            ],
        )
        prepared_histories: list[list[Any]] = []

        async def _build_runtime_context(
            *, tool_skill_binding_state_override: ToolSkillBindingState | None = None, request_tool_state_override: dict[str, Any] | None = None, **_kwargs: Any
        ) -> dict[str, Any]:
            loaded = list((tool_skill_binding_state_override or ToolSkillBindingState([], [])).effective_ids)
            runtime_tools = [_make_tool("query_demo_sql")] if loaded else [_make_tool("search_tool_skills")]
            return {
                "mode": "chat",
                "prompt_is_ui": False,
                "allowed_tool_config_ids": ["tool-1"],
                "runtime_tools": runtime_tools,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": request_tool_state_override
                or {
                    "tool_calls": [],
                    "signature_counts": {},
                    "blocked_repeat_calls": 0,
                    "max_iterations_reached": False,
                    "internal_continue_attempts": 0,
                    "internal_continue_stop_reason": "",
                    "tool_free_synthesis_used": False,
                },
                "export_context": {},
                "workspace_id": None,
                "tool_skill_binding_state": tool_skill_binding_state_override or ToolSkillBindingState([], []),
                "tool_skill_catalog": [],
                "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                "tool_skill_has_loadable": not bool(loaded),
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": loaded,
            }

        async def _prepare_context_window(**kwargs: Any) -> tuple[Any, Any, Any]:
            prepared_histories.append(list(kwargs["chat_history"]))
            return SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"), kwargs["chat_history"], kwargs["turn_system_content"]

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="multi")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_build_runtime_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(side_effect=_prepare_context_window),
            ),
            mock.patch.object(rag, "_seed_latest_export_context_from_chat_history", return_value=None),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[getattr(rag, "agent_executor"), second_executor, third_executor]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()),
        ):
            events = [event async for event in rag.process_query_stream("hello", chat_history=[])]

        self.assertIn("done", "".join(event for event in events if isinstance(event, str)))
        last_history = prepared_histories[-1]
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-load" for message in last_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-unload" for message in last_history), 1)

    async def test_feature_off_streaming_debug_persists_once_with_original_user_input(self) -> None:
        rag = self._make_rag()
        cast(dict[str, Any], rag._app_settings)["tool_skills_enabled"] = False
        first_executor = _FakeStreamExecutor(
            [_make_tool("query_demo_sql")],
            [
                [
                    {"event": "on_chat_model_stream", "run_id": "chat-1", "data": {"chunk": SimpleNamespace(content="plain")}},
                    {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [], "output": "plain"}}},
                ]
            ],
        )
        setattr(rag, "agent_executor", first_executor)
        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello raw")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(
                rag,
                "_build_request_runtime_context",
                new=mock.AsyncMock(
                    return_value={
                        "mode": "chat",
                        "prompt_is_ui": False,
                        "allowed_tool_config_ids": ["tool-1"],
                        "runtime_tools": [_make_tool("query_demo_sql")],
                        "prompt_additions": "",
                        "include_sqlite_persistence": False,
                        "userspace_env_var_turn_hint": "",
                        "userspace_runtime_status_turn_hint": "",
                        "userspace_diagnostics_turn_hint": "",
                        "user_identity_turn_line": "",
                        "current_time_turn_line": "",
                        "request_tool_state": {
                            "tool_calls": [],
                            "signature_counts": {},
                            "blocked_repeat_calls": 0,
                            "max_iterations_reached": False,
                            "internal_continue_attempts": 0,
                            "internal_continue_stop_reason": "",
                            "tool_free_synthesis_used": False,
                        },
                        "export_context": {},
                        "workspace_id": None,
                        "tool_skill_binding_state": None,
                        "tool_skill_catalog": [],
                        "tool_skill_hidden_ids": [],
                        "tool_skill_has_loadable": False,
                        "tool_skill_mode": "disabled",
                        "tool_skill_loaded_ids": [],
                    }
                ),
            ),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_seed_latest_export_context_from_chat_history", return_value=None),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[getattr(rag, "agent_executor")]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()) as persist_debug,
        ):
            _ = [event async for event in rag.process_query_stream("hello", chat_history=[])]

        persist_debug.assert_awaited_once()
        await_args = persist_debug.await_args
        assert await_args is not None
        self.assertEqual(await_args.kwargs["rendered_user_input"], "hello raw")

    async def test_enabled_nonstream_process_query_does_not_outer_retry_tool_skill_stage_loop_after_ocr_error(self) -> None:
        rag = self._make_rag()
        setattr(rag, "agent_executor", SimpleNamespace(tools=[_make_tool("search_tool_skills")]))
        ocr_error = RuntimeError("image input not supported")
        request_context = {
            "mode": "chat",
            "prompt_is_ui": False,
            "allowed_tool_config_ids": ["tool-1"],
            "runtime_tools": [_make_tool("search_tool_skills")],
            "prompt_additions": "",
            "include_sqlite_persistence": False,
            "userspace_env_var_turn_hint": "",
            "userspace_runtime_status_turn_hint": "",
            "userspace_diagnostics_turn_hint": "",
            "user_identity_turn_line": "",
            "current_time_turn_line": "",
            "request_tool_state": {
                "tool_calls": [],
                "signature_counts": {},
                "blocked_repeat_calls": 0,
                "max_iterations_reached": False,
                "internal_continue_attempts": 0,
                "internal_continue_stop_reason": "",
                "tool_free_synthesis_used": False,
            },
            "export_context": {},
            "workspace_id": None,
            "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
            "tool_skill_catalog": [],
            "tool_skill_hidden_ids": ["tool-1"],
            "tool_skill_has_loadable": True,
            "tool_skill_mode": "enabled",
            "tool_skill_loaded_ids": [],
        }

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello raw")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(return_value=request_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_seed_latest_export_context_from_chat_history", return_value=None),
            mock.patch.object(rag, "_build_runtime_executor", return_value=SimpleNamespace()),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_run_nonstream_tool_skill_stage_loop", new=mock.AsyncMock(side_effect=ocr_error)) as stage_loop,
            mock.patch.object(rag, "_ocr_images_after_no_image_support_error", new=mock.AsyncMock(return_value="hello ocr")) as ocr_retry,
            mock.patch.object(rag, "_chat_runtime_error_message", return_value="runtime-error"),
        ):
            output = await rag.process_query("hello", chat_history=[])

        self.assertEqual(output, "runtime-error")
        stage_loop.assert_awaited_once()
        ocr_retry.assert_not_awaited()

    async def test_feature_off_nonstream_ocr_retry_preserves_debug_record_behavior(self) -> None:
        rag = self._make_rag()
        cast(dict[str, Any], rag._app_settings)["tool_skills_enabled"] = False
        executor = _FakeExecutor(
            [_make_tool("query_demo_sql")],
            [{"output": "plain", "intermediate_steps": []}],
        )
        setattr(rag, "agent_executor", SimpleNamespace(tools=[_make_tool("query_demo_sql")]))
        setattr(executor, "ainvoke", mock.AsyncMock(side_effect=[RuntimeError("image input not supported"), {"output": "plain", "intermediate_steps": []}]))

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello raw")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(
                rag,
                "_build_request_runtime_context",
                new=mock.AsyncMock(
                    return_value={
                        "mode": "chat",
                        "prompt_is_ui": False,
                        "allowed_tool_config_ids": ["tool-1"],
                        "runtime_tools": [_make_tool("query_demo_sql")],
                        "prompt_additions": "",
                        "include_sqlite_persistence": False,
                        "userspace_env_var_turn_hint": "",
                        "userspace_runtime_status_turn_hint": "",
                        "userspace_diagnostics_turn_hint": "",
                        "user_identity_turn_line": "",
                        "current_time_turn_line": "",
                        "request_tool_state": {
                            "tool_calls": [],
                            "signature_counts": {},
                            "blocked_repeat_calls": 0,
                            "max_iterations_reached": False,
                            "internal_continue_attempts": 0,
                            "internal_continue_stop_reason": "",
                            "tool_free_synthesis_used": False,
                        },
                        "export_context": {},
                        "workspace_id": None,
                        "tool_skill_binding_state": None,
                        "tool_skill_catalog": [],
                        "tool_skill_hidden_ids": [],
                        "tool_skill_has_loadable": False,
                        "tool_skill_mode": "disabled",
                        "tool_skill_loaded_ids": [],
                    }
                ),
            ),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_seed_latest_export_context_from_chat_history", return_value=None),
            mock.patch.object(rag, "_build_runtime_executor", return_value=executor),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_ocr_images_after_no_image_support_error", new=mock.AsyncMock(return_value="hello ocr")) as ocr_retry,
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()) as persist_debug,
        ):
            output = await rag.process_query("hello", chat_history=[])

        self.assertEqual(output, "plain")
        self.assertEqual(getattr(executor, "ainvoke").await_count, 2)
        ocr_retry.assert_awaited_once()
        persist_debug.assert_awaited_once()
        await_args = persist_debug.await_args
        assert await_args is not None
        self.assertEqual(await_args.kwargs["rendered_user_input"], "hello ocr")

    def test_internal_synthesis_tool_context_includes_replay_and_final_steps_once(self) -> None:
        replay_messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "load_tool_skills", "args": {"ids": ["tool_config:tool-1"]}, "id": "call-load", "type": "tool_call"}],
            ),
            ToolMessage(content='{"status":"ok"}', tool_call_id="call-load"),
        ]
        intermediate_steps = [
            (_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"), "rows"),
        ]

        context = rag_components.RAGComponents._build_internal_synthesis_tool_context(
            intermediate_steps=intermediate_steps,
            replay_messages=replay_messages,
        )

        assert context is not None
        content = str(context.content)
        self.assertIn("load_tool_skills", content)
        self.assertIn("query_demo_sql", content)
        self.assertEqual(content.count("load_tool_skills"), 1)
        self.assertEqual(content.count("query_demo_sql"), 1)

    def test_reasoning_only_fallback_includes_replay_and_final_step_tool_names_once(self) -> None:
        replay_messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "load_tool_skills", "args": {"ids": ["tool_config:tool-1"]}, "id": "call-load", "type": "tool_call"}],
            ),
            ToolMessage(content='{"status":"ok"}', tool_call_id="call-load"),
        ]
        intermediate_steps = [
            (_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"), "rows"),
        ]

        fallback = rag_components.RAGComponents._build_synthesis_reasoning_only_fallback(
            intermediate_steps=intermediate_steps,
            replay_messages=replay_messages,
        )

        self.assertIn("`load_tool_skills`", fallback)
        self.assertIn("`query_demo_sql`", fallback)
        self.assertEqual(fallback.count("`load_tool_skills`"), 1)
        self.assertEqual(fallback.count("`query_demo_sql`"), 1)

    def test_internal_synthesis_tool_context_keeps_distinct_replay_pairs_when_fallback_ids_repeat(self) -> None:
        replay_messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_tool_skills", "args": {"query": "demo"}, "id": "tool_skill_stage_0", "type": "tool_call"}],
            ),
            ToolMessage(content='{"results":[{"id":"tool_config:tool-1"}]}', tool_call_id="tool_skill_stage_0"),
            AIMessage(
                content="",
                tool_calls=[{"name": "load_tool_skills", "args": {"ids": ["tool_config:tool-1"]}, "id": "tool_skill_stage_0", "type": "tool_call"}],
            ),
            ToolMessage(content='{"status":"ok"}', tool_call_id="tool_skill_stage_0"),
        ]

        context = rag_components.RAGComponents._build_internal_synthesis_tool_context(
            intermediate_steps=[],
            replay_messages=replay_messages,
        )

        assert context is not None
        content = str(context.content)
        self.assertIn("search_tool_skills", content)
        self.assertIn("load_tool_skills", content)
        self.assertEqual(content.count("search_tool_skills"), 1)
        self.assertEqual(content.count("load_tool_skills"), 1)

    def test_internal_synthesis_tool_context_includes_all_parallel_replay_tool_calls(self) -> None:
        replay_messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_tool_skills", "args": {"query": "demo"}, "id": "call-search", "type": "tool_call"},
                    {"name": "load_tool_skills", "args": {"ids": ["tool_config:tool-1"]}, "id": "call-load", "type": "tool_call"},
                ],
            ),
            ToolMessage(content='{"results":[{"id":"tool_config:tool-1"}]}', tool_call_id="call-search"),
            ToolMessage(content='{"status":"ok"}', tool_call_id="call-load"),
        ]

        context = rag_components.RAGComponents._build_internal_synthesis_tool_context(
            intermediate_steps=[],
            replay_messages=replay_messages,
        )

        assert context is not None
        content = str(context.content)
        self.assertIn("search_tool_skills", content)
        self.assertIn("load_tool_skills", content)
        self.assertEqual(content.count("search_tool_skills"), 1)
        self.assertEqual(content.count("load_tool_skills"), 1)

    def test_synthesis_helpers_do_not_double_count_final_attempt_when_replay_and_intermediate_ids_differ(self) -> None:
        replay_messages = [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_tool_skills", "args": {"query": "demo"}, "id": "prior-search", "type": "tool_call"}],
            ),
            ToolMessage(content='{"results":[{"id":"tool_config:tool-1"}]}', tool_call_id="prior-search"),
            AIMessage(
                content="",
                tool_calls=[{"name": "load_tool_skills", "args": {"ids": ["tool_config:tool-1"]}, "id": "prior-load", "type": "tool_call"}],
            ),
            ToolMessage(content='{"status":"ok"}', tool_call_id="prior-load"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_demo_sql_schema", "args": {"query": "orders"}, "id": "prior-schema", "type": "tool_call"}],
            ),
            ToolMessage(content='{"columns":["id"]}', tool_call_id="prior-schema"),
            AIMessage(
                content="",
                tool_calls=[{"name": "query_demo_sql", "args": {"query": "select 1 limit 1"}, "id": "run-final", "type": "tool_call"}],
            ),
            ToolMessage(content='[{"value":1}]', tool_call_id="run-final"),
        ]
        intermediate_steps = [
            (_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "provider-final"), '[{"value":1}]'),
        ]

        context = rag_components.RAGComponents._build_internal_synthesis_tool_context(
            intermediate_steps=intermediate_steps,
            replay_messages=replay_messages,
            max_items=4,
        )
        fallback = rag_components.RAGComponents._build_synthesis_reasoning_only_fallback(
            intermediate_steps=intermediate_steps,
            replay_messages=replay_messages,
            max_tools=4,
        )

        assert context is not None
        content = str(context.content)
        self.assertIn("search_tool_skills", content)
        self.assertIn("load_tool_skills", content)
        self.assertIn("search_demo_sql_schema", content)
        self.assertEqual(content.count("query_demo_sql"), 1)
        self.assertIn("`search_tool_skills`", fallback)
        self.assertIn("`load_tool_skills`", fallback)
        self.assertIn("`search_demo_sql_schema`", fallback)
        self.assertEqual(fallback.count("`query_demo_sql`"), 1)

    async def test_process_level_runtime_context_resolves_real_catalog_loadable_and_hidden_config_ids(self) -> None:
        rag = self._make_rag()
        runtime_tools = [_make_tool("query_demo_sql"), _make_tool("search_tool_skills_builtin")]
        binding_state = ToolSkillBindingState(requested_ids=["tool_config:tool-1"], effective_ids=["tool_config:tool-1"])
        result = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
            binding_state_override=binding_state,
        )

        self.assertEqual(result["tool_skill_hidden_ids"], [])
        self.assertEqual(result["tool_skill_loaded_ids"], ["tool_config:tool-1"])
        self.assertTrue(result["tool_skill_has_loadable"])
        self.assertNotIn("search_tool_skills_builtin", {tool.name for tool in result["runtime_tools"]})

    async def test_process_query_real_runtime_context_binds_only_controls_when_optional_tool_is_unloaded(self) -> None:
        rag = self._make_rag()
        setattr(rag, "agent_executor", SimpleNamespace(tools=[_make_tool("query_demo_sql")]))
        built_tool_names: list[list[str]] = []

        def _build_executor(tools: list[Any], *_args: Any, **_kwargs: Any) -> _FakeExecutor:
            built_tool_names.append([tool.name for tool in tools])
            return _FakeExecutor(tools, [{"output": "done", "intermediate_steps": []}])

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="hello")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(
                rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(side_effect=lambda *_args, **_kwargs: list(getattr(rag, "agent_executor").tools))
            ),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=_build_executor),
        ):
            output = await rag.process_query("hello", chat_history=[])

        self.assertEqual(output, "done")
        self.assertTrue({"search_tool_skills", "load_tool_skills", "unload_tool_skills"}.issubset(set(built_tool_names[0])))
        self.assertNotIn("query_demo_sql", built_tool_names[0])

    async def test_builtin_only_loadable_state_adds_generic_workflow_guidance_without_connection_sentence(self) -> None:
        rag = self._make_rag()
        request_state = {
            "tool_calls": [],
            "signature_counts": {},
            "blocked_repeat_calls": 0,
            "max_iterations_reached": False,
            "internal_continue_attempts": 0,
            "internal_continue_stop_reason": "",
            "tool_free_synthesis_used": False,
        }

        with (
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=[_make_tool("search_tool_skills_builtin")])),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(
                rag,
                "_resolve_request_tool_skill_bindings",
                new=mock.AsyncMock(
                    return_value={
                        "runtime_tools": [_make_tool("search_tool_skills"), _make_tool("load_tool_skills"), _make_tool("unload_tool_skills")],
                        "tool_skill_binding_state": ToolSkillBindingState(requested_ids=[], effective_ids=[]),
                        "tool_skill_catalog": [],
                        "tool_skill_hidden_ids": [],
                        "tool_skill_has_loadable": True,
                        "tool_skill_mode": "enabled",
                        "tool_skill_loaded_ids": [],
                    }
                ),
            ),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=cast(Any, SimpleNamespace(tools=[_make_tool("search_tool_skills_builtin")])),
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=False,
                conversation_id=None,
                request_tool_state_override=request_state,
            )

        self.assertIn("Optional tool-skill workflow", context["prompt_additions"])
        self.assertNotIn("missing eligible connection", context["prompt_additions"])

    async def test_real_builtin_only_loadable_state_adds_generic_workflow_guidance_without_connection_sentence(self) -> None:
        rag = self._make_rag()
        request_state = {
            "tool_calls": [],
            "signature_counts": {},
            "blocked_repeat_calls": 0,
            "max_iterations_reached": False,
            "internal_continue_attempts": 0,
            "internal_continue_stop_reason": "",
            "tool_free_synthesis_used": False,
        }

        with (
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=[_make_tool("builtin_optional_tool")])),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=cast(Any, SimpleNamespace(tools=[_make_tool("builtin_optional_tool")])),
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=False,
                conversation_id=None,
                request_tool_state_override=request_state,
            )

        self.assertTrue(context["tool_skill_has_loadable"])
        self.assertEqual(context["tool_skill_hidden_ids"], [])
        self.assertIn("Optional tool-skill workflow", context["prompt_additions"])
        self.assertNotIn("missing eligible connection", context["prompt_additions"])

    def test_request_debug_metadata_includes_tool_skill_stage_fields(self) -> None:
        rag = self._make_rag()
        metadata = rag._build_request_debug_metadata(
            mode="chat",
            request_tool_state={
                "tool_calls": [],
                "blocked_repeat_calls": 0,
                "max_iterations_reached": False,
                "internal_continue_attempts": 0,
                "internal_continue_stop_reason": "",
                "tool_free_synthesis_used": False,
                "tool_skill_mode": "enabled",
                "tool_skill_stage_index": 2,
                "tool_skill_transition_count": 1,
                "tool_skill_requested_ids": ["tool_config:tool-1"],
                "tool_skill_effective_ids": [],
                "tool_skill_hidden_ids": ["tool-1"],
                "tool_skill_transition_kind": "unload",
            },
            workspace_id=None,
        )

        self.assertEqual(metadata["tool_skill_mode"], "enabled")
        self.assertEqual(metadata["tool_skill_stage_index"], 2)
        self.assertEqual(metadata["tool_skill_transition_count"], 1)
        self.assertEqual(metadata["tool_skill_transition_kind"], "unload")

    def test_enabled_ui_request_system_prompt_omits_visualization_common_guidance_until_tools_are_loaded(self) -> None:
        rag = self._make_rag()

        prompt = rag._build_request_system_prompt(
            is_ui=True,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=[_make_tool("search_tool_skills")],
            hidden_tool_config_ids=["tool-1"],
            tool_skill_mode="enabled",
            loaded_tool_skill_ids=[],
            tool_skill_has_loadable=True,
        )

        self.assertNotIn("create_chart", prompt)
        self.assertNotIn("create_datatable", prompt)

    async def test_userspace_diagnostics_turn_hint_uses_generic_guidance_when_optional_tool_is_hidden(self) -> None:
        rag = self._make_rag()
        workspace = SimpleNamespace(
            tool_selection_mode="default_all",
            selected_tool_ids=[],
            selected_tool_group_ids=[],
            sqlite_persistence_mode="exclude",
        )

        with (
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag_components, "resolve_effective_tool_ids", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "get_workspace", new=mock.AsyncMock(return_value=workspace)),
            mock.patch.object(rag_components.userspace_service, "filter_tool_ids_for_workspace_owner", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_workspace_env_var_summaries", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_workspace_mounts", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_mountable_sources", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "get_workspace_object_storage_summary", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(
                rag_components.userspace_service,
                "list_workspace_preview_diagnostic_summary",
                new=mock.AsyncMock(return_value=[{"target_label": "dashboard", "max_ms": 1200, "last_error": "boom"}]),
            ),
            mock.patch.object(rag_components.userspace_runtime_service, "get_devserver_status", new=mock.AsyncMock(side_effect=RuntimeError("skip"))),
            mock.patch.object(
                rag,
                "_create_userspace_file_tools",
                new=mock.AsyncMock(return_value=[_make_tool("read_userspace_file"), _make_tool("userspace_diagnostics")]),
            ),
            mock.patch.object(rag, "_create_spawn_subagents_tool", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(rag, "_apply_mode_specific_tool_description_overrides", side_effect=lambda tools, **_kwargs: tools),
            mock.patch.object(rag, "_wrap_userspace_runtime_tools_for_execution_proofs", side_effect=lambda tools, *_args, **_kwargs: tools),
            mock.patch.object(
                rag,
                "_wrap_runtime_tools_with_request_state",
                side_effect=lambda tools, **_kwargs: (
                    tools,
                    {
                        "tool_calls": [],
                        "signature_counts": {},
                        "blocked_repeat_calls": 0,
                        "max_iterations_reached": False,
                        "internal_continue_attempts": 0,
                        "internal_continue_stop_reason": "",
                        "tool_free_synthesis_used": False,
                    },
                ),
            ),
            mock.patch.object(
                rag_components.userspace_service,
                "get_workspace_entrypoint_status",
                return_value=SimpleNamespace(state="valid", framework="react", command="npm run dev", cwd="."),
            ),
            mock.patch.object(rag_components.userspace_service, "is_default_static_entrypoint", return_value=False),
            mock.patch.object(rag, "_build_userspace_continuity_prompt", new=mock.AsyncMock(return_value="")),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=cast(Any, SimpleNamespace(tools=[])),
                blocked_tool_names=None,
                workspace_context={"workspace_id": "ws-1", "user_id": "user-1", "username": "alice", "display_name": "Alice", "accessible_workspace_modes": {}},
                add_chat_visualization_prompt=False,
                user_id="user-1",
                current_user_context={"user_id": "user-1", "username": "alice", "display_name": "Alice", "is_admin": False},
                conversation_id=None,
            )

        self.assertNotIn("userspace_diagnostics", context["userspace_diagnostics_turn_hint"])
        self.assertIn("Use available diagnostics tooling", context["userspace_diagnostics_turn_hint"])

    async def test_dynamic_workspace_mcp_tools_do_not_claim_tool_skill_control_names(self) -> None:
        rag = self._make_rag()

        with (
            mock.patch.object(
                rag_components.userspace_runtime_service,
                "get_global_mcp_tools",
                new=mock.AsyncMock(
                    return_value=[
                        {
                            "server_name": "Workspace Server",
                            "name": "search_tool_skills",
                            "description": "Collision",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ]
                ),
            ),
            mock.patch.object(rag_components.userspace_runtime_service, "list_workspace_mcp_tools", new=mock.AsyncMock(return_value=[])),
        ):
            tools = await rag._create_userspace_file_tools("ws-1", "user-1")

        tool_names = {tool.name for tool in tools}
        self.assertNotIn("search_tool_skills", tool_names)
        self.assertIn("Workspace_Server_search_tool_skills", tool_names)

    async def test_nonstream_process_query_persists_one_debug_record_per_stage(self) -> None:
        rag = self._make_rag()
        first_executor = _FakeExecutor(
            [_make_tool("search_tool_skills")],
            [
                {
                    "output": "ignored",
                    "intermediate_steps": [
                        (
                            _FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"),
                            json.dumps(
                                {
                                    "status": "ok",
                                    "bindings_changed": True,
                                    "transition_kind": "load",
                                    "requested_ids": ["tool_config:tool-1"],
                                    "effective_ids": ["tool_config:tool-1"],
                                }
                            ),
                        )
                    ],
                }
            ],
        )
        setattr(rag, "agent_executor", first_executor)
        binding_state = ToolSkillBindingState(requested_ids=[], effective_ids=[])
        second_executor = _FakeExecutor(
            [_make_tool("query_demo_sql")],
            [{"output": "final", "intermediate_steps": [(_FakeAction("query_demo_sql", {"query": "select 1 limit 1"}, "call-query"), "rows")]}],
        )

        async def _build_runtime_context(
            *, tool_skill_binding_state_override: ToolSkillBindingState | None = None, request_tool_state_override: dict[str, Any] | None = None, **_kwargs: Any
        ) -> dict[str, Any]:
            loaded = list((tool_skill_binding_state_override or binding_state).effective_ids)
            runtime_tools = [_make_tool("query_demo_sql")] if loaded else [_make_tool("search_tool_skills")]
            request_state = request_tool_state_override or {
                "tool_calls": [],
                "signature_counts": {},
                "blocked_repeat_calls": 0,
                "max_iterations_reached": False,
                "internal_continue_attempts": 0,
                "internal_continue_stop_reason": "",
                "tool_free_synthesis_used": False,
            }
            request_state.update(
                {
                    "tool_skill_mode": "enabled",
                    "tool_skill_requested_ids": list((tool_skill_binding_state_override or binding_state).requested_ids),
                    "tool_skill_effective_ids": loaded,
                    "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                    "tool_skill_has_loadable": not bool(loaded),
                    "tool_skill_stage_index": int(request_state.get("tool_skill_stage_index", 1)),
                    "tool_skill_transition_count": int(request_state.get("tool_skill_transition_count", 0)),
                    "tool_skill_transition_kind": request_state.get("tool_skill_transition_kind"),
                }
            )
            return {
                "mode": "chat",
                "prompt_is_ui": False,
                "allowed_tool_config_ids": ["tool-1"],
                "runtime_tools": runtime_tools,
                "prompt_additions": "",
                "include_sqlite_persistence": False,
                "userspace_env_var_turn_hint": "",
                "userspace_runtime_status_turn_hint": "",
                "userspace_diagnostics_turn_hint": "",
                "user_identity_turn_line": "",
                "current_time_turn_line": "",
                "request_tool_state": request_state,
                "export_context": {},
                "workspace_id": None,
                "tool_skill_binding_state": tool_skill_binding_state_override or binding_state,
                "tool_skill_catalog": [],
                "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                "tool_skill_has_loadable": not bool(loaded),
                "tool_skill_mode": "enabled",
                "tool_skill_loaded_ids": loaded,
            }

        with (
            mock.patch.object(
                rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))
            ),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="load then query")),
            mock.patch.object(rag, "_ocr_images_if_model_lacks_support", new=mock.AsyncMock(side_effect=lambda content, *_args, **_kwargs: content)),
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_build_runtime_context)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"),
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[first_executor, second_executor]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()) as persist_debug,
        ):
            await rag.process_query("hello", chat_history=[], conversation_id="conversation-1")

        self.assertEqual(persist_debug.await_count, 2)
        first_stage_kwargs = persist_debug.await_args_list[0].kwargs
        second_stage_kwargs = persist_debug.await_args_list[1].kwargs
        self.assertEqual(first_stage_kwargs["debug_metadata"]["tool_skill_effective_ids"], [])
        self.assertEqual(second_stage_kwargs["debug_metadata"]["tool_skill_effective_ids"], ["tool_config:tool-1"])

    async def test_userspace_wrappers_skip_tool_skill_controls(self) -> None:
        rag = self._make_rag()
        control_tool = _make_tool("load_tool_skills")
        query_tool = _make_tool("query_demo_sql")

        wrapped_tools, _request_state = rag._wrap_runtime_tools_with_request_state(
            [control_tool, query_tool],
            mode="userspace",
            workspace_id="workspace-1",
        )
        self.assertEqual([tool.name for tool in wrapped_tools], ["load_tool_skills", "query_demo_sql"])
        self.assertIs(getattr(wrapped_tools[0], "coroutine", None), getattr(control_tool, "coroutine", None))

        proofed_tools = rag._wrap_userspace_runtime_tools_for_execution_proofs(
            [control_tool, query_tool],
            "workspace-1",
            ["tool-1"],
        )
        self.assertEqual([tool.name for tool in proofed_tools], ["load_tool_skills", "query_demo_sql"])
        self.assertIs(getattr(proofed_tools[0], "coroutine", None), getattr(control_tool, "coroutine", None))


if __name__ == "__main__":
    unittest.main()
