from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from ragtime.rag import components as rag_components
from ragtime.rag.prompts import UI_VISUALIZATION_CHAT_PROMPT, UI_VISUALIZATION_COMMON_PROMPT
from ragtime.rag.tool_skills import ToolSkillBindingState


def _make_tool(name: str, *, description: str | None = None, coroutine: Any) -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description or f"Tool {name}",
    )


def _make_sql_bundle_tools() -> list[StructuredTool]:
    async def _query_demo_sql(query: str, limit: int = 100) -> str:
        return f"rows for {query} limit {limit}"

    async def _search_demo_sql_schema(query: str) -> str:
        return f"schema for {query}"

    return [
        _make_tool(
            "query_demo_sql",
            description="Query the demo SQL database.",
            coroutine=_query_demo_sql,
        ),
        _make_tool(
            "search_demo_sql_schema",
            description="Search the demo SQL schema.",
            coroutine=_search_demo_sql_schema,
        ),
    ]


def _make_builtin_tool() -> StructuredTool:
    async def _builtin_lookup(query: str, limit: int = 5) -> str:
        return f"builtin {query} {limit}"

    return _make_tool(
        "builtin_lookup",
        description="Optional built-in lookup tool.",
        coroutine=_builtin_lookup,
    )


def _make_noop_tool(name: str, *, description: str | None = None) -> StructuredTool:
    async def _tool(**_kwargs: Any) -> str:
        return name

    return _make_tool(name, description=description, coroutine=_tool)


def _make_chart_tool() -> StructuredTool:
    async def _create_chart(chart_type: str, title: str = "Chart") -> str:
        return f"{chart_type}:{title}"

    return _make_tool(
        "create_chart",
        description="Create a chart.",
        coroutine=_create_chart,
    )


def _make_datatable_tool() -> StructuredTool:
    async def _create_datatable(title: str = "Table") -> str:
        return title

    return _make_tool(
        "create_datatable",
        description="Create a datatable.",
        coroutine=_create_datatable,
    )


class _FakeAction:
    def __init__(self, tool: str, tool_input: dict[str, Any], tool_call_id: str) -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.tool_call_id = tool_call_id


class _FakeExecutor:
    def __init__(self, tools: list[Any], results: list[dict[str, Any]]) -> None:
        self.tools = tools
        self._results = list(results)
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.inputs.append(payload)
        if not self._results:
            raise AssertionError("No more fake executor results")
        return self._results.pop(0)


class _FakeStreamExecutor:
    def __init__(self, tools: list[Any], streams: list[list[dict[str, Any]]]) -> None:
        self.tools = tools
        self._streams = list(streams)
        self.inputs: list[dict[str, Any]] = []

    def astream_events(self, payload: dict[str, Any], version: str = "v2"):
        self.inputs.append(payload)
        async def _gen():
            if not self._streams:
                raise AssertionError("No more fake stream stages")
            for event in self._streams.pop(0):
                yield event

        return _gen()


def _tool_by_name(tools: list[StructuredTool], name: str) -> StructuredTool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Missing tool {name}")


class ToolSkillRuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def _make_rag(self, *, tool_name: str = "Demo SQL", tool_description: str = "Reads demo SQL rows.") -> rag_components.RAGComponents:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = [
            {
                "id": "tool-1",
                "name": tool_name,
                "tool_type": "postgres",
                "description": tool_description,
                "connection_config": {},
            }
        ]
        rag._index_metadata = []
        rag._request_prompt_cache = {}
        rag._app_settings = {"tool_skills_enabled": True, "max_iterations": 4}
        return rag

    async def test_tool_config_bundle_can_be_searched_and_loaded_request_locally(self) -> None:
        rag = self._make_rag()
        runtime_tools = _make_sql_bundle_tools()

        initial = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
        )

        initial_prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=initial["runtime_tools"],
            hidden_tool_config_ids=initial["tool_skill_hidden_ids"],
            tool_skill_has_loadable=initial["tool_skill_has_loadable"],
            tool_skill_mode=initial["tool_skill_mode"],
            loaded_tool_skill_ids=initial["tool_skill_loaded_ids"],
        )

        self.assertIn("No optional system tools are currently loaded", initial_prompt)
        self.assertNotIn("Demo SQL", initial_prompt)
        self.assertNotIn("CONFIGURED BUT UNAVAILABLE", initial_prompt)

        search_tool = _tool_by_name(initial["runtime_tools"], "search_tool_skills")
        search_payload = json.loads(await search_tool.ainvoke({"query": "demo sql", "limit": 8}))
        self.assertEqual(search_payload["results"][0]["id"], "tool_config:tool-1")
        self.assertEqual(search_payload["results"][0]["tool_names"], ["query_demo_sql", "search_demo_sql_schema"])

        load_tool = _tool_by_name(initial["runtime_tools"], "load_tool_skills")
        load_payload = json.loads(await load_tool.ainvoke({"ids": ["tool_config:tool-1"]}))
        self.assertEqual(load_payload["effective_ids"], ["tool_config:tool-1"])

        rebound = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
            binding_state_override=initial["tool_skill_binding_state"],
        )

        rebound_names = [tool.name for tool in rebound["runtime_tools"]]
        self.assertEqual(
            rebound_names,
            [
                "query_demo_sql",
                "search_demo_sql_schema",
                "search_tool_skills",
                "load_tool_skills",
                "unload_tool_skills",
            ],
        )
        self.assertEqual(
            set(_tool_by_name(rebound["runtime_tools"], "query_demo_sql").args_schema.model_fields),
            {"query", "limit"},
        )
        self.assertEqual(
            set(_tool_by_name(rebound["runtime_tools"], "search_demo_sql_schema").args_schema.model_fields),
            {"query"},
        )

        loaded_prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=rebound["runtime_tools"],
            hidden_tool_config_ids=rebound["tool_skill_hidden_ids"],
            tool_skill_has_loadable=rebound["tool_skill_has_loadable"],
            tool_skill_mode=rebound["tool_skill_mode"],
            loaded_tool_skill_ids=rebound["tool_skill_loaded_ids"],
        )
        tool_scope_prompt = rag._build_request_tool_scope_prompt(rebound["runtime_tools"], mode="chat")

        self.assertIn("Demo SQL", loaded_prompt)
        self.assertIn("query_demo_sql", tool_scope_prompt)
        self.assertIn("search_demo_sql_schema", tool_scope_prompt)

    async def test_tool_config_bundle_can_be_loaded_with_persisted_conversation_state(self) -> None:
        rag = self._make_rag()
        runtime_tools = _make_sql_bundle_tools()

        with (
            mock.patch.object(
                rag_components.repository,
                "get_conversation_loaded_tool_skill_ids",
                new=mock.AsyncMock(return_value=[]),
            ),
            mock.patch.object(
                rag_components.repository,
                "mutate_conversation_loaded_tool_skill_ids",
                new=mock.AsyncMock(return_value=["tool_config:tool-1"]),
            ) as mutate_loaded,
        ):
            initial = await rag._resolve_request_tool_skill_bindings(
                runtime_tools=runtime_tools,
                mode="chat",
                conversation_id="conversation-1",
                allowed_tool_config_ids=["tool-1"],
            )
            load_tool = _tool_by_name(initial["runtime_tools"], "load_tool_skills")
            load_payload = json.loads(await load_tool.ainvoke({"ids": ["tool_config:tool-1"]}))

        self.assertEqual(load_payload["requested_ids"], ["tool_config:tool-1"])
        mutate_loaded.assert_awaited_once_with(
            "conversation-1",
            add_ids=["tool_config:tool-1"],
            remove_ids=[],
        )

        with mock.patch.object(
            rag_components.repository,
            "get_conversation_loaded_tool_skill_ids",
            new=mock.AsyncMock(return_value=["tool_config:tool-1"]),
        ):
            rebound = await rag._resolve_request_tool_skill_bindings(
                runtime_tools=runtime_tools,
                mode="chat",
                conversation_id="conversation-1",
                allowed_tool_config_ids=["tool-1"],
            )

        self.assertIn("query_demo_sql", [tool.name for tool in rebound["runtime_tools"]])
        self.assertEqual(rebound["tool_skill_loaded_ids"], ["tool_config:tool-1"])

    async def test_builtin_only_catalog_still_reports_loadable_state(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = []
        rag._index_metadata = []
        rag._request_prompt_cache = {}
        rag._app_settings = {"tool_skills_enabled": True, "max_iterations": 4}
        runtime_tools = [_make_builtin_tool()]

        binding = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
        )

        search_tool = _tool_by_name(binding["runtime_tools"], "search_tool_skills")
        search_payload = json.loads(await search_tool.ainvoke({"query": "builtin", "limit": 8}))
        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=None,
            runtime_tools=binding["runtime_tools"],
            hidden_tool_config_ids=binding["tool_skill_hidden_ids"],
            tool_skill_has_loadable=binding["tool_skill_has_loadable"],
            tool_skill_mode=binding["tool_skill_mode"],
            loaded_tool_skill_ids=binding["tool_skill_loaded_ids"],
        )

        self.assertEqual(search_payload["results"][0]["id"], "builtin:builtin_lookup")
        self.assertIn("No optional system tools are currently loaded", prompt)

    async def test_search_only_rebuild_preserves_builtin_catalog_for_followup_load(self) -> None:
        rag = rag_components.RAGComponents.__new__(rag_components.RAGComponents)
        rag._tool_configs = []
        rag._index_metadata = []
        rag._request_prompt_cache = {}
        rag._app_settings = {"tool_skills_enabled": True, "max_iterations": 4}
        base_executor = SimpleNamespace(tools=[_make_builtin_tool()])

        async def _passthrough_conversation_tools(
            _conversation_id: str | None,
            runtime_tools: list[StructuredTool],
            **_kwargs: Any,
        ) -> list[StructuredTool]:
            return list(runtime_tools)

        with (
            mock.patch.object(
                rag,
                "_apply_conversation_tool_overrides",
                new=mock.AsyncMock(side_effect=_passthrough_conversation_tools),
            ),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(
                rag,
                "_apply_mode_specific_tool_description_overrides",
                side_effect=lambda tools, **_kwargs: list(tools),
            ),
            mock.patch.object(
                rag,
                "_prepare_chat_context_window",
                new=mock.AsyncMock(
                    side_effect=lambda **kwargs: (
                        kwargs["llm_resolution"],
                        kwargs["chat_history"],
                        kwargs["turn_system_content"],
                    )
                ),
            ),
            mock.patch.object(
                rag,
                "_build_runtime_executor",
                side_effect=lambda tools, *_args, **_kwargs: SimpleNamespace(tools=tools),
            ),
        ):
            initial = await rag._build_request_runtime_context(
                is_ui=False,
                executor=base_executor,
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=True,
                user_id=None,
                current_user_context=None,
                current_time_context=None,
                conversation_id=None,
                conversation_model=None,
                chat_task_id=None,
                disabled_builtin_tool_ids=None,
            )

            search_tool = _tool_by_name(initial["runtime_tools"], "search_tool_skills")
            search_output = await search_tool.ainvoke({"query": "builtin", "limit": 8})
            search_payload = json.loads(search_output)

            self.assertEqual(search_payload["results"][0]["id"], "builtin:builtin_lookup")

            rebuilt_executor, rebuilt_context, _rebuilt_history, _system_prompt, _tool_scope_prompt, _turn_system_content, _llm_resolution = await rag._rebuild_tool_skill_stage_after_control_activity(
                current_request_context=initial,
                current_executor=SimpleNamespace(tools=initial["runtime_tools"]),
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                original_chat_history=[HumanMessage(content="earlier")],
                original_user_content="search builtin then load",
                replay_messages=rag._format_tool_skill_stage_replay_messages(
                    [(_FakeAction("search_tool_skills", {"query": "builtin", "limit": 8}, "call-search"), search_output)]
                ),
                blocked_tool_names=None,
                workspace_context=None,
                user_id=None,
                current_user_context=None,
                current_time_context=None,
                conversation_id=None,
                conversation_model=None,
                chat_task_id=None,
                disabled_builtin_tool_ids=None,
                turn_system_content="",
                remaining_iterations=3,
                transition=None,
                next_transition_count=1,
                next_stage_index=2,
                add_chat_visualization_prompt=True,
            )

            self.assertIn("load_tool_skills", [tool.name for tool in rebuilt_executor.tools])

            load_tool = _tool_by_name(rebuilt_context["runtime_tools"], "load_tool_skills")
            load_payload = json.loads(await load_tool.ainvoke({"ids": [search_payload["results"][0]["id"]]}))

        self.assertEqual(load_payload["status"], "ok")
        self.assertEqual(load_payload["effective_ids"], ["builtin:builtin_lookup"])

        rebound = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=list(base_executor.tools),
            mode="chat",
            conversation_id=None,
            binding_state_override=rebuilt_context["tool_skill_binding_state"],
        )

        self.assertIn("builtin_lookup", [tool.name for tool in rebound["runtime_tools"]])

    async def test_persisted_ineligible_ids_stay_requested_but_unbound(self) -> None:
        rag = self._make_rag()

        with mock.patch.object(
            rag_components.repository,
            "get_conversation_loaded_tool_skill_ids",
            new=mock.AsyncMock(return_value=["tool_config:tool-1"]),
        ):
            binding = await rag._resolve_request_tool_skill_bindings(
                runtime_tools=_make_sql_bundle_tools(),
                mode="chat",
                conversation_id="conversation-1",
                allowed_tool_config_ids=[],
            )

        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=[],
            runtime_tools=binding["runtime_tools"],
            hidden_tool_config_ids=binding["tool_skill_hidden_ids"],
            tool_skill_has_loadable=binding["tool_skill_has_loadable"],
            tool_skill_mode=binding["tool_skill_mode"],
            loaded_tool_skill_ids=binding["tool_skill_loaded_ids"],
        )

        self.assertEqual(binding["tool_skill_binding_state"].requested_ids, ["tool_config:tool-1"])
        self.assertEqual(binding["tool_skill_binding_state"].effective_ids, [])
        self.assertNotIn("query_demo_sql", [tool.name for tool in binding["runtime_tools"]])
        self.assertIn("No system tools are selected for this request", prompt)

    async def test_separate_instances_do_not_cross_leak_prompt_caches(self) -> None:
        rag_alpha = self._make_rag(tool_name="Alpha SQL", tool_description="Reads alpha rows.")
        rag_beta = self._make_rag(tool_name="Beta SQL", tool_description="Reads beta rows.")
        runtime_tools = _make_sql_bundle_tools()

        alpha_binding = await rag_alpha._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
        )
        beta_binding = await rag_beta._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
        )
        await _tool_by_name(alpha_binding["runtime_tools"], "load_tool_skills").ainvoke({"ids": ["tool_config:tool-1"]})
        await _tool_by_name(beta_binding["runtime_tools"], "load_tool_skills").ainvoke({"ids": ["tool_config:tool-1"]})

        alpha_rebound = await rag_alpha._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
            binding_state_override=alpha_binding["tool_skill_binding_state"],
        )
        beta_rebound = await rag_beta._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id=None,
            allowed_tool_config_ids=["tool-1"],
            binding_state_override=beta_binding["tool_skill_binding_state"],
        )

        alpha_prompt = rag_alpha._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=alpha_rebound["runtime_tools"],
            hidden_tool_config_ids=alpha_rebound["tool_skill_hidden_ids"],
            tool_skill_has_loadable=alpha_rebound["tool_skill_has_loadable"],
            tool_skill_mode=alpha_rebound["tool_skill_mode"],
            loaded_tool_skill_ids=alpha_rebound["tool_skill_loaded_ids"],
        )
        beta_prompt = rag_beta._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=["tool-1"],
            runtime_tools=beta_rebound["runtime_tools"],
            hidden_tool_config_ids=beta_rebound["tool_skill_hidden_ids"],
            tool_skill_has_loadable=beta_rebound["tool_skill_has_loadable"],
            tool_skill_mode=beta_rebound["tool_skill_mode"],
            loaded_tool_skill_ids=beta_rebound["tool_skill_loaded_ids"],
        )

        self.assertIn("Alpha SQL", alpha_prompt)
        self.assertNotIn("Beta SQL", alpha_prompt)
        self.assertIn("Beta SQL", beta_prompt)
        self.assertNotIn("Alpha SQL", beta_prompt)

    async def test_feature_off_returns_original_runtime_tools_without_controls(self) -> None:
        rag = self._make_rag()
        rag._app_settings["tool_skills_enabled"] = False
        runtime_tools = _make_sql_bundle_tools()

        binding = await rag._resolve_request_tool_skill_bindings(
            runtime_tools=runtime_tools,
            mode="chat",
            conversation_id="conversation-1",
            allowed_tool_config_ids=["tool-1"],
        )

        self.assertIs(binding["runtime_tools"], runtime_tools)
        self.assertEqual([tool.name for tool in binding["runtime_tools"]], ["query_demo_sql", "search_demo_sql_schema"])
        self.assertEqual(binding["tool_skill_mode"], "disabled")
        self.assertEqual(binding["tool_skill_loaded_ids"], [])

    async def test_nonstream_search_then_answer_records_distinct_monotonic_stages(self) -> None:
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
        first_executor = _FakeExecutor(
            [_make_noop_tool("search_tool_skills")],
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
            [_make_noop_tool("search_tool_skills")],
            [{"output": "final answer", "intermediate_steps": []}],
        )
        stage_records: list[dict[str, Any]] = []

        async def _rebuild_stage(**_kwargs: Any) -> dict[str, Any]:
            return {
                "runtime_tools": [_make_builtin_tool()],
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

        async def _stage_debug(
            stage_index: int,
            stage_request_context: dict[str, Any],
            stage_chat_history: list[Any],
            stage_input: Any,
            _stage_system_prompt: str,
            _stage_tool_scope_prompt: str,
            _stage_turn_system_content: str,
        ) -> None:
            stage_records.append(
                {
                    "stage_index": stage_index,
                    "input": stage_input,
                    "tool_skill_stage_index": stage_request_context["request_tool_state"].get("tool_skill_stage_index"),
                    "history_len": len(stage_chat_history),
                }
            )

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
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
            output, _executor, _history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[HumanMessage(content="earlier")],
                user_content="search first",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_builtin_tool()],
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
                stage_debug_callback=_stage_debug,
            )

        self.assertEqual(output, "final answer")
        self.assertEqual([record["stage_index"] for record in stage_records], [1, 2])
        self.assertEqual([record["tool_skill_stage_index"] for record in stage_records], [1, 2])
        self.assertNotIn("Do not call additional tools", second_executor.inputs[0]["input"])

    async def test_nonstream_search_load_and_use_continue_in_one_request(self) -> None:
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
        search_output = json.dumps({"status": "ok", "results": [{"id": "tool_config:tool-1"}]})
        load_output = json.dumps(
            {
                "status": "ok",
                "bindings_changed": True,
                "transition_kind": "load",
                "requested_ids": ["tool_config:tool-1"],
                "effective_ids": ["tool_config:tool-1"],
            }
        )
        first_executor = _FakeExecutor(
            [_make_noop_tool("search_tool_skills")],
            [{
                "output": search_output,
                "intermediate_steps": [(_FakeAction("search_tool_skills", {"query": "demo"}, "call-search"), search_output)],
            }],
        )
        second_executor = _FakeExecutor(
            [_make_noop_tool("load_tool_skills")],
            [{
                "output": load_output,
                "intermediate_steps": [(_FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"), load_output)],
            }],
        )
        third_executor = _FakeExecutor(
            [_make_noop_tool("query_demo_sql")],
            [{
                "output": "final answer",
                "intermediate_steps": [(_FakeAction("query_demo_sql", {"query": "select 1"}, "call-query"), "rows for select 1")],
            }],
        )
        stage_records: list[dict[str, Any]] = []

        async def _rebuild_stage(
            *,
            tool_skill_binding_state_override: ToolSkillBindingState | None = None,
            request_tool_state_override: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            binding_state = tool_skill_binding_state_override or ToolSkillBindingState([], [])
            loaded = list(binding_state.effective_ids)
            return {
                "runtime_tools": [_make_noop_tool("query_demo_sql")] if loaded else [_make_builtin_tool()],
                "tool_skill_binding_state": binding_state,
                "tool_skill_hidden_ids": [] if loaded else ["tool-1"],
                "tool_skill_has_loadable": not bool(loaded),
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
                "request_tool_state": request_tool_state_override or request_state,
                "export_context": {},
                "workspace_id": None,
            }

        async def _stage_debug(
            stage_index: int,
            stage_request_context: dict[str, Any],
            _stage_chat_history: list[Any],
            stage_input: Any,
            _stage_system_prompt: str,
            _stage_tool_scope_prompt: str,
            _stage_turn_system_content: str,
        ) -> None:
            stage_records.append({"stage_index": stage_index, "input": stage_input})

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
            mock.patch.object(rag, "_build_request_system_prompt", return_value="system"),
            mock.patch.object(rag, "_build_request_tool_scope_prompt", return_value=""),
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[second_executor, third_executor]),
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
            output, _executor, _history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[HumanMessage(content="earlier")],
                user_content="search then load then use",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_builtin_tool()],
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
                stage_debug_callback=_stage_debug,
            )

        self.assertEqual(output, "final answer")
        self.assertEqual([record["stage_index"] for record in stage_records], [1, 2, 3])
        self.assertEqual(second_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        self.assertEqual(third_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        self.assertIn("Do not call additional tools", rag_components.INTERNAL_AGENT_CONTINUATION_PROMPT)

    async def test_nonstream_search_with_no_results_can_answer_without_more_controls(self) -> None:
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
        search_output = json.dumps({"status": "ok", "results": []})
        first_executor = _FakeExecutor(
            [_make_noop_tool("search_tool_skills")],
            [{
                "output": search_output,
                "intermediate_steps": [(_FakeAction("search_tool_skills", {"query": "demo"}, "call-search"), search_output)],
            }],
        )
        second_executor = _FakeExecutor([_make_noop_tool("search_tool_skills")], [{"output": "no matching skill", "intermediate_steps": []}])
        stage_records: list[dict[str, Any]] = []

        async def _rebuild_stage(**_kwargs: Any) -> dict[str, Any]:
            return {
                "runtime_tools": [_make_builtin_tool()],
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

        async def _stage_debug(
            stage_index: int,
            _stage_request_context: dict[str, Any],
            _stage_chat_history: list[Any],
            _stage_input: Any,
            _stage_system_prompt: str,
            _stage_tool_scope_prompt: str,
            _stage_turn_system_content: str,
        ) -> None:
            stage_records.append({"stage_index": stage_index})

        with (
            mock.patch.object(rag, "_build_request_runtime_context", new=mock.AsyncMock(side_effect=_rebuild_stage)),
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
            output, _executor, _history = await rag._run_nonstream_tool_skill_stage_loop(
                executor=first_executor,
                llm_resolution=SimpleNamespace(llm=None, provider="openai", model="gpt-test"),
                chat_history=[HumanMessage(content="earlier")],
                user_content="search first",
                request_context={
                    "mode": "chat",
                    "prompt_is_ui": False,
                    "allowed_tool_config_ids": ["tool-1"],
                    "runtime_tools": [_make_builtin_tool()],
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
                stage_debug_callback=_stage_debug,
            )

        self.assertEqual(output, "no matching skill")
        self.assertEqual([record["stage_index"] for record in stage_records], [1, 2])

    async def test_streaming_search_load_and_use_continue_in_one_request(self) -> None:
        rag = self._make_rag()
        search_output = json.dumps({"status": "ok", "results": [{"id": "tool_config:tool-1"}]})
        load_output = json.dumps(
            {
                "status": "ok",
                "bindings_changed": True,
                "transition_kind": "load",
                "requested_ids": ["tool_config:tool-1"],
                "effective_ids": ["tool_config:tool-1"],
            }
        )
        first_executor = _FakeStreamExecutor(
            [_make_noop_tool("search_tool_skills")],
            [[
                {"event": "on_tool_start", "run_id": "call-search", "name": "search_tool_skills", "data": {"input": {"query": "demo"}}},
                {"event": "on_tool_end", "run_id": "call-search", "name": "search_tool_skills", "data": {"output": search_output}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [(_FakeAction("search_tool_skills", {"query": "demo"}, "call-search"), search_output)], "output": "ignored"}}},
            ]],
        )
        rag.agent_executor = first_executor
        second_executor = _FakeStreamExecutor(
            [_make_noop_tool("load_tool_skills")],
            [[
                {"event": "on_tool_start", "run_id": "call-load", "name": "load_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                {"event": "on_tool_end", "run_id": "call-load", "name": "load_tool_skills", "data": {"output": load_output}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [(_FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"), load_output)], "output": "ignored"}}},
            ]],
        )
        third_executor = _FakeStreamExecutor(
            [_make_noop_tool("query_demo_sql")],
            [[
                {"event": "on_tool_start", "run_id": "call-query", "name": "query_demo_sql", "data": {"input": {"query": "select 1"}}},
                {"event": "on_tool_end", "run_id": "call-query", "name": "query_demo_sql", "data": {"output": "rows for select 1"}},
                {"event": "on_chat_model_stream", "run_id": "chat-3", "data": {"chunk": SimpleNamespace(content="done")}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [(_FakeAction("query_demo_sql", {"query": "select 1"}, "call-query"), "rows for select 1")], "output": "done"}}},
            ]],
        )

        async def _build_runtime_context(
            *,
            tool_skill_binding_state_override: ToolSkillBindingState | None = None,
            request_tool_state_override: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            loaded = list((tool_skill_binding_state_override or ToolSkillBindingState([], [])).effective_ids)
            runtime_tools = [_make_noop_tool("query_demo_sql")] if loaded else [_make_builtin_tool()]
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

        with (
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="multi")),
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
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[first_executor, second_executor, third_executor]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()) as persist_debug,
        ):
            events = [event async for event in rag.process_query_stream("hello", chat_history=[])]

        self.assertIn("done", "".join(event for event in events if isinstance(event, str)))
        self.assertEqual(
            [call.kwargs["debug_metadata"]["tool_skill_stage_index"] for call in persist_debug.await_args_list],
            [1, 2, 3],
        )
        self.assertEqual(second_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        self.assertEqual(third_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        final_history = persist_debug.await_args_list[-1].kwargs["chat_history"]
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-search" for message in final_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-load" for message in final_history), 1)

    async def test_streaming_two_transitions_retain_prior_pairs_and_record_each_stage(self) -> None:
        rag = self._make_rag()
        first_executor = _FakeStreamExecutor(
            [_make_builtin_tool()],
            [[
                {"event": "on_tool_start", "run_id": "call-load", "name": "load_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                {"event": "on_tool_end", "run_id": "call-load", "name": "load_tool_skills", "data": {"output": json.dumps({"status": "ok", "bindings_changed": True, "transition_kind": "load", "requested_ids": ["tool_config:tool-1"], "effective_ids": ["tool_config:tool-1"]})}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [(_FakeAction("load_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-load"), json.dumps({"status": "ok", "bindings_changed": True, "transition_kind": "load", "requested_ids": ["tool_config:tool-1"], "effective_ids": ["tool_config:tool-1"]}))], "output": "ignored"}}},
            ]],
        )
        rag.agent_executor = first_executor
        second_executor = _FakeStreamExecutor(
            [_make_noop_tool("query_demo_sql")],
            [[
                {"event": "on_tool_start", "run_id": "call-unload", "name": "unload_tool_skills", "data": {"input": {"ids": ["tool_config:tool-1"]}}},
                {"event": "on_tool_end", "run_id": "call-unload", "name": "unload_tool_skills", "data": {"output": json.dumps({"status": "ok", "bindings_changed": True, "transition_kind": "unload", "requested_ids": [], "effective_ids": []})}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [(_FakeAction("unload_tool_skills", {"ids": ["tool_config:tool-1"]}, "call-unload"), json.dumps({"status": "ok", "bindings_changed": True, "transition_kind": "unload", "requested_ids": [], "effective_ids": []}))], "output": "ignored"}}},
            ]],
        )
        third_executor = _FakeStreamExecutor(
            [_make_builtin_tool()],
            [[
                {"event": "on_chat_model_stream", "run_id": "chat-3", "data": {"chunk": SimpleNamespace(content="done")}},
                {"event": "on_chain_end", "data": {"output": {"intermediate_steps": [], "output": "done"}}},
            ]],
        )

        async def _build_runtime_context(
            *,
            tool_skill_binding_state_override: ToolSkillBindingState | None = None,
            request_tool_state_override: dict[str, Any] | None = None,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            loaded = list((tool_skill_binding_state_override or ToolSkillBindingState([], [])).effective_ids)
            runtime_tools = [_make_noop_tool("query_demo_sql")] if loaded else [_make_builtin_tool()]
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

        with (
            mock.patch.object(rag, "_get_request_scoped_llm", new=mock.AsyncMock(return_value=SimpleNamespace(llm=SimpleNamespace(), provider="openai", model="gpt-test"))),
            mock.patch.object(rag, "_convert_message_to_langchain_async", new=mock.AsyncMock(return_value="multi")),
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
            mock.patch.object(rag, "_build_runtime_executor", side_effect=[first_executor, second_executor, third_executor]),
            mock.patch.object(rag, "_build_turn_reminder_text", return_value=""),
            mock.patch.object(rag, "_build_context_headroom_prompt", new=mock.AsyncMock(return_value="")),
            mock.patch.object(rag, "_persist_provider_prompt_debug_record", new=mock.AsyncMock()) as persist_debug,
        ):
            events = [event async for event in rag.process_query_stream("hello", chat_history=[])]

        self.assertIn("done", "".join(event for event in events if isinstance(event, str)))
        self.assertEqual(
            [call.kwargs["debug_metadata"]["tool_skill_stage_index"] for call in persist_debug.await_args_list],
            [1, 2, 3],
        )
        self.assertEqual(second_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        self.assertEqual(third_executor.inputs[0]["input"], rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT)
        self.assertIn("do not reload a skill that was just unloaded", rag_components.INTERNAL_TOOL_SKILL_CONTINUATION_PROMPT.lower())
        final_history = persist_debug.await_args_list[-1].kwargs["chat_history"]
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-load" for message in final_history), 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) and message.tool_call_id == "call-unload" for message in final_history), 1)

    def test_feature_off_ui_prompt_includes_visualization_guidance_exactly_once(self) -> None:
        rag = self._make_rag()
        rag._app_settings["tool_skills_enabled"] = False
        prompt = rag._build_request_system_prompt(
            is_ui=True,
            mode="chat",
            allowed_tool_config_ids=None,
            runtime_tools=[_make_chart_tool(), _make_datatable_tool()],
            hidden_tool_config_ids=[],
            tool_skill_mode="disabled",
            loaded_tool_skill_ids=[],
            tool_skill_has_loadable=False,
        )

        unique_line = "- **create_chart** - Chart.js visualizations (bar, line, pie, doughnut)"
        self.assertIn(unique_line, UI_VISUALIZATION_COMMON_PROMPT)
        self.assertEqual(prompt.count(unique_line), 1)

    async def test_enabled_ui_with_unloaded_visualization_tools_does_not_name_chart_or_datatable(self) -> None:
        rag = self._make_rag()
        executor = SimpleNamespace(tools=[_make_chart_tool(), _make_datatable_tool()])

        with (
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=list(executor.tools))),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(rag, "_apply_mode_specific_tool_description_overrides", side_effect=lambda tools, **_kwargs: tools),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=True,
                executor=executor,
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=True,
                conversation_id=None,
            )

        self.assertNotIn("create_chart", {tool.name for tool in context["runtime_tools"]})
        self.assertNotIn("create_datatable", {tool.name for tool in context["runtime_tools"]})
        self.assertNotIn("create_chart", context["prompt_additions"])
        self.assertNotIn("create_datatable", context["prompt_additions"])
        self.assertNotIn(UI_VISUALIZATION_CHAT_PROMPT.strip(), context["prompt_additions"])

    async def test_userspace_prompt_with_unloaded_optional_runtime_tools_does_not_name_them(self) -> None:
        rag = self._make_rag()
        workspace = SimpleNamespace(
            tool_selection_mode="default_all",
            selected_tool_ids=[],
            selected_tool_group_ids=[],
            sqlite_persistence_mode="exclude",
        )
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
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag_components, "resolve_effective_tool_ids", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "get_workspace", new=mock.AsyncMock(return_value=workspace)),
            mock.patch.object(rag_components.userspace_service, "filter_tool_ids_for_workspace_owner", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_workspace_env_var_summaries", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_workspace_mounts", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "list_mountable_sources", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(rag_components.userspace_service, "get_workspace_object_storage_summary", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(rag_components.userspace_service, "list_workspace_preview_diagnostic_summary", new=mock.AsyncMock(side_effect=RuntimeError("skip"))),
            mock.patch.object(rag_components.userspace_runtime_service, "get_devserver_status", new=mock.AsyncMock(side_effect=RuntimeError("skip"))),
            mock.patch.object(
                rag,
                "_create_userspace_file_tools",
                new=mock.AsyncMock(
                    return_value=[
                        _make_noop_tool("read_userspace_file"),
                        _make_noop_tool("run_terminal_command"),
                        _make_noop_tool("discover_userspace_primitives"),
                        _make_noop_tool("list_userspace_env_vars"),
                    ]
                ),
            ),
            mock.patch.object(rag, "_create_spawn_subagents_tool", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[_make_noop_tool("userspace_diagnostics")]),
            mock.patch.object(rag, "_apply_mode_specific_tool_description_overrides", side_effect=lambda tools, **_kwargs: tools),
            mock.patch.object(rag, "_wrap_userspace_runtime_tools_for_execution_proofs", side_effect=lambda tools, *_args, **_kwargs: tools),
            mock.patch.object(rag, "_wrap_runtime_tools_with_request_state", side_effect=lambda tools, **_kwargs: (tools, request_state)),
            mock.patch.object(rag_components.userspace_service, "get_workspace_entrypoint_status", return_value=SimpleNamespace(state="valid", framework="react", command="npm run dev", cwd=".")),
            mock.patch.object(rag_components.userspace_service, "is_default_static_entrypoint", return_value=False),
            mock.patch.object(rag, "_build_userspace_continuity_prompt", new=mock.AsyncMock(return_value="")),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=SimpleNamespace(tools=[]),
                blocked_tool_names=None,
                workspace_context={"workspace_id": "ws-1", "user_id": "user-1", "username": "alice", "display_name": "Alice", "accessible_workspace_modes": {}},
                add_chat_visualization_prompt=False,
                user_id="user-1",
                current_user_context={"user_id": "user-1", "username": "alice", "display_name": "Alice", "is_admin": False},
                conversation_id=None,
            )

        runtime_names = {tool.name for tool in context["runtime_tools"]}
        self.assertNotIn("run_terminal_command", runtime_names)
        self.assertNotIn("discover_userspace_primitives", runtime_names)
        self.assertNotIn("list_userspace_env_vars", runtime_names)
        self.assertNotIn("userspace_diagnostics", runtime_names)
        self.assertNotIn("run_terminal_command", context["prompt_additions"])
        self.assertNotIn("discover_userspace_primitives", context["prompt_additions"])
        self.assertNotIn("list_userspace_env_vars", context["prompt_additions"])
        self.assertNotIn("userspace_diagnostics", context["prompt_additions"])

    async def test_enabled_ui_with_loaded_visualization_tools_includes_legacy_guidance(self) -> None:
        rag = self._make_rag()
        executor = SimpleNamespace(tools=[_make_chart_tool(), _make_datatable_tool()])

        with (
            mock.patch.object(rag, "_apply_conversation_tool_overrides", new=mock.AsyncMock(return_value=list(executor.tools))),
            mock.patch.object(rag, "_build_conversation_export_tool", return_value=None),
            mock.patch.object(rag, "_build_chat_diagnostic_tools", return_value=[]),
            mock.patch.object(rag, "_apply_mode_specific_tool_description_overrides", side_effect=lambda tools, **_kwargs: tools),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=True,
                executor=executor,
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=True,
                conversation_id=None,
                tool_skill_binding_state_override=ToolSkillBindingState(
                    requested_ids=["builtin:create_chart", "builtin:create_datatable"],
                    effective_ids=["builtin:create_chart", "builtin:create_datatable"],
                ),
            )

        runtime_names = {tool.name for tool in context["runtime_tools"]}
        self.assertIn("create_chart", runtime_names)
        self.assertIn("create_datatable", runtime_names)
        self.assertIn(UI_VISUALIZATION_CHAT_PROMPT.strip(), context["prompt_additions"])


if __name__ == "__main__":
    unittest.main()
