from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from ragtime.rag import components as rag_components


def make_tool(name: str, *, description: str | None = None, coroutine: Any = None) -> StructuredTool:
    """Create a StructuredTool for testing.

    Args:
        name: Tool name
        description: Tool description (defaults to f"Tool {name}")
        coroutine: Async function to use as the tool's coroutine.
                   If None, creates a simple coroutine that returns the tool name.
    """
    if coroutine is None:

        async def _tool(**_kwargs: Any) -> str:
            return name

        coroutine = _tool

    return StructuredTool.from_function(
        coroutine=coroutine,
        name=name,
        description=description or f"Tool {name}",
    )


def make_noop_tool(name: str, *, description: str | None = None) -> StructuredTool:
    """Create a no-op StructuredTool that returns its name."""

    async def _tool(**_kwargs: Any) -> str:
        return name

    return make_tool(name, description=description, coroutine=_tool)


def make_builtin_tool() -> StructuredTool:
    """Create a builtin lookup tool for testing."""

    async def _builtin_lookup(query: str, limit: int = 5) -> str:
        return f"builtin {query} {limit}"

    return make_tool(
        "builtin_lookup",
        description="Optional built-in lookup tool.",
        coroutine=_builtin_lookup,
    )


def make_sql_bundle_tools() -> list[StructuredTool]:
    """Create a bundle of SQL testing tools."""

    async def _query_demo_sql(query: str, limit: int = 100) -> str:
        return f"rows for {query} limit {limit}"

    async def _search_demo_sql_schema(query: str) -> str:
        return f"schema for {query}"

    return [
        make_tool(
            "query_demo_sql",
            description="Query the demo SQL database.",
            coroutine=_query_demo_sql,
        ),
        make_tool(
            "search_demo_sql_schema",
            description="Search the demo SQL schema.",
            coroutine=_search_demo_sql_schema,
        ),
    ]


def make_chart_tool() -> StructuredTool:
    """Create a chart creation tool for testing."""

    async def _create_chart(chart_type: str, title: str = "Chart") -> str:
        return f"{chart_type}:{title}"

    return make_tool(
        "create_chart",
        description="Create a chart.",
        coroutine=_create_chart,
    )


def make_datatable_tool() -> StructuredTool:
    """Create a datatable creation tool for testing."""

    async def _create_datatable(title: str = "Table") -> str:
        return title

    return make_tool(
        "create_datatable",
        description="Create a datatable.",
        coroutine=_create_datatable,
    )


def tool_by_name(tools: list[StructuredTool], name: str) -> StructuredTool:
    """Find a tool by name from a list."""
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Missing tool {name}")


def find_tool_names(tools: list[StructuredTool]) -> set[str]:
    """Get all tool names from a list."""
    return {tool.name for tool in tools}


def make_empty_request_state() -> dict[str, Any]:
    """Create an empty request tool state dict."""
    return {
        "tool_calls": [],
        "signature_counts": {},
        "blocked_repeat_calls": 0,
        "max_iterations_reached": False,
        "internal_continue_attempts": 0,
        "internal_continue_stop_reason": "",
        "tool_free_synthesis_used": False,
    }


def make_base_request_context(
    *,
    mode: str = "chat",
    prompt_is_ui: bool = False,
    allowed_tool_config_ids: list[str] | None = None,
    runtime_tools: list[StructuredTool] | None = None,
    prompt_additions: str = "",
    include_sqlite_persistence: bool = False,
    request_tool_state: dict[str, Any] | None = None,
    tool_skill_binding_state: Any = None,
    tool_skill_hidden_ids: list[str] | None = None,
    tool_skill_has_loadable: bool = False,
    tool_skill_mode: str = "disabled",
    tool_skill_loaded_ids: list[str] | None = None,
    tool_skill_catalog: list[Any] | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create a base request context dict for testing.

    Provides default values for all the userspace and tool-skill fields
    that appear in test request contexts.
    """
    if runtime_tools is None:
        runtime_tools = []
    if request_tool_state is None:
        request_tool_state = make_empty_request_state()
    if tool_skill_hidden_ids is None:
        tool_skill_hidden_ids = []
    if tool_skill_loaded_ids is None:
        tool_skill_loaded_ids = []
    if tool_skill_catalog is None:
        tool_skill_catalog = []
    if allowed_tool_config_ids is None:
        allowed_tool_config_ids = []

    return {
        "mode": mode,
        "prompt_is_ui": prompt_is_ui,
        "allowed_tool_config_ids": allowed_tool_config_ids,
        "runtime_tools": runtime_tools,
        "prompt_additions": prompt_additions,
        "include_sqlite_persistence": include_sqlite_persistence,
        "userspace_env_var_turn_hint": "",
        "userspace_runtime_status_turn_hint": "",
        "userspace_diagnostics_turn_hint": "",
        "user_identity_turn_line": "",
        "current_time_turn_line": "",
        "request_tool_state": request_tool_state,
        "export_context": {},
        "workspace_id": workspace_id,
        "tool_skill_binding_state": tool_skill_binding_state,
        "tool_skill_hidden_ids": tool_skill_hidden_ids,
        "tool_skill_has_loadable": tool_skill_has_loadable,
        "tool_skill_mode": tool_skill_mode,
        "tool_skill_loaded_ids": tool_skill_loaded_ids,
        "tool_skill_catalog": tool_skill_catalog,
    }


class FakeAction:
    """Fake LangChain action for testing tool execution."""

    def __init__(self, tool: str, tool_input: dict[str, Any], tool_call_id: str) -> None:
        self.tool = tool
        self.tool_input = tool_input
        self.tool_call_id = tool_call_id

    @property
    def message_log(self) -> list[AIMessage]:
        """Return a fake AIMessage with tool_call."""
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


class FakeExecutor:
    """Fake executor that returns pre-configured results."""

    def __init__(self, tools: list[Any], results: list[dict[str, Any]]) -> None:
        self.tools = tools
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.inputs: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke and return next result from queue."""
        self.calls.append(payload)
        self.inputs.append(payload)
        if not self._results:
            raise AssertionError("No more fake executor results")
        return self._results.pop(0)


class FakeStreamExecutor:
    """Fake executor that streams pre-configured event sequences."""

    def __init__(self, tools: list[Any], streams: list[list[dict[str, Any]]]) -> None:
        self.tools = tools
        self._streams = list(streams)
        self.inputs: list[dict[str, Any]] = []

    def astream_events(self, payload: dict[str, Any], version: str = "v2"):
        """Stream events from pre-configured sequence."""
        self.inputs.append(payload)

        async def _gen():
            if not self._streams:
                raise AssertionError("No more fake stream stages")
            for event in self._streams.pop(0):
                yield event

        return _gen()


def make_rag_components(
    *,
    tool_name: str = "Demo SQL",
    tool_description: str = "Reads demo SQL rows.",
    tool_skills_enabled: bool = True,
    max_iterations: int = 4,
) -> Any:
    """Create a minimal RAGComponents instance for testing.

    Args:
        tool_name: Name of the tool to include in configs
        tool_description: Description of the tool
        tool_skills_enabled: Whether tool skills are enabled
        max_iterations: Max iterations setting

    Returns:
        RAGComponents instance with mocked attributes
    """
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
    rag._app_settings = {"tool_skills_enabled": tool_skills_enabled, "max_iterations": max_iterations}
    return rag
