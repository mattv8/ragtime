import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.server import Server

from ragtime.mcp import server as mcp_server
from ragtime.mcp.skills import tool_skill_lines, visible_skills
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, ToolHealthStatus
from ragtime.rag.prompts import build_tool_system_prompt
from ragtime.rag.tool_skills import ToolSkillBindingState
from ragtime.skills.contract import SKILL_OWNER_TOOL, build_skill_uri
from ragtime.solidworks.linking import hidden_pdm_config_ids
from ragtime.solidworks.tool import build_solidworks_tool_description
from tests.test_instruction_skills_mcp import call_read_skill, first_text
from tests.test_tool_skill_shared import make_noop_tool, make_rag_components


def skill(name, mode="on_demand", content="instructions", skill_id="admin"):
    return {"id": skill_id, "name": name, "mode": mode, "description": "When needed", "content": content}


def test_hidden_ids_require_both_configs_and_enabled_solidworks():
    sw = {"id": "sw", "tool_type": "solidworks", "connection_config": {"pdm_tool_id": "pdm"}}
    pdm = {"id": "pdm", "tool_type": "solidworks_pdm"}
    assert hidden_pdm_config_ids([sw, pdm]) == {"pdm"}
    assert hidden_pdm_config_ids([pdm]) == set()
    assert hidden_pdm_config_ids([{**sw, "enabled": False}, pdm]) == set()


def test_linked_skills_are_visible_with_pdm_ownership():
    pdm = {"id": "pdm", "name": "Vault", "skills": [skill("vault")]}
    sw = {"id": "sw", "name": "Design", "tool_type": "solidworks", "pdm_link": pdm}
    uri = build_skill_uri(SKILL_OWNER_TOOL, "pdm", "vault")
    assert uri in tool_skill_lines(sw)
    assert visible_skills([SimpleNamespace(tool_config=sw)], None, [])[uri] == (pdm["skills"][0], "tool Vault")
    assert uri in build_tool_system_prompt([sw], instruction_skill_loader_available=True)
    assert "Skills:" in MCPToolAdapter()._build_tool_description(sw)


def test_builtin_always_inline_when_admin_exceeds_limit():
    sw = {
        "id": "sw",
        "skills": [
            skill("essentials", "always", "essential text", "builtin:essentials"),
            skill("admin", "always", "x" * 100000),
        ],
    }
    text = tool_skill_lines(sw)
    assert "essential text" in text
    assert "x" * 100000 not in text
    assert "before using this tool, call read_skill" in text


def test_description_linked_vault_and_admin_budget():
    config = {
        "name": "Design",
        "description": "Models",
        "pdm_link_name": "Vault",
        "pdm_link": {"description": "Index"},
        "allow_write": True,
        "connection_config": {"pdm_tool_id": "pdm", "allow_scripts": True, "pdm_approved_folders": ["A", "B"]},
    }
    description = build_solidworks_tool_description(config)
    assert "Models. Vault index: Index. Scripts: on. Vault changes: on in 2 approved folder(s)." in description
    assert "Linked to PDM vault 'Vault': find files with pdm.lookup and pdm.search, check state, get latest, check out and check in." in description
    config["pdm_link"]["description"] = "x" * 3000
    assert len(build_solidworks_tool_description(config)) == 1499


@pytest.mark.parametrize(
    ("selected", "sw_enabled", "healthy", "expected"),
    [
        (None, True, True, {"solidworks_design"}),
        (["pdm"], True, True, {"search_vault"}),
        (None, True, False, {"search_vault"}),
        (None, False, True, {"search_vault"}),
    ],
)
def test_mcp_hides_only_when_solidworks_survives(selected, sw_enabled, healthy, expected):
    sw = {
        "id": "sw",
        "name": "Design",
        "tool_type": "solidworks",
        "enabled": sw_enabled,
        "connection_config": {"pdm_tool_id": "pdm"},
    }
    pdm = {"id": "pdm", "name": "Vault", "tool_type": "solidworks_pdm", "enabled": True, "skills": [skill("vault")]}
    linked = SimpleNamespace(name="Vault", tool_type="solidworks_pdm", enabled=True)
    adapter = MCPToolAdapter()
    route = McpRouteFilter(selected, include_knowledge_search=False, include_git_history=False) if selected is not None else None
    with (
        patch("ragtime.mcp.tools.get_tool_configs", new=AsyncMock(return_value=[sw, pdm])),
        patch("ragtime.indexer.repository.repository.get_tool_config", new=AsyncMock(return_value=linked)),
        patch("ragtime.mcp.tools.get_app_settings", new=AsyncMock(return_value={})),
        patch.object(adapter, "_check_heartbeats", new=AsyncMock(return_value={"sw": ToolHealthStatus("sw", healthy)})),
        patch.object(adapter, "_create_executor", new=AsyncMock(return_value=AsyncMock())),
        patch.object(adapter, "_create_schema_search_tool_definition", new=AsyncMock(return_value=None)),
        patch.object(adapter, "_create_knowledge_search_tool", new=AsyncMock(return_value=None)),
        patch.object(adapter, "_create_git_history_tools", new=AsyncMock(return_value=[])),
        patch.object(adapter, "get_ssh_transfer_definition", new=AsyncMock(return_value=None)),
    ):
        definitions = asyncio.run(adapter.get_available_tools(route_filter=route))
    assert {definition.name for definition in definitions} == expected


@pytest.mark.parametrize(
    ("pdm_selected", "pdm_enabled", "pdm_healthy", "visible"),
    [(False, True, True, False), (True, False, True, False), (True, True, True, True), (True, True, False, True)],
)
def test_mcp_linked_skill_visibility_follows_candidate_set(pdm_selected, pdm_enabled, pdm_healthy, visible):
    async def run():
        sw = {
            "id": "sw",
            "name": "Design",
            "tool_type": "solidworks",
            "connection_config": {"pdm_tool_id": "pdm"},
            "skills": [skill("own")],
        }
        pdm = {
            "id": "pdm",
            "name": "Vault",
            "tool_type": "solidworks_pdm",
            "enabled": pdm_enabled,
            "description": "Unique vault index description",
            "skills": [skill("vault")],
        }
        route = McpRouteFilter(["sw", *(["pdm"] if pdm_selected else [])], include_knowledge_search=False, include_git_history=False)
        linked = SimpleNamespace(name="Vault", tool_type="solidworks_pdm", enabled=pdm_enabled)
        adapter = MCPToolAdapter()
        server = Server("linked-candidate-test")
        mcp_server._register_handlers(server, adapter, route, development_tools=False)
        with (
            patch("ragtime.mcp.tools.get_tool_configs", new=AsyncMock(return_value=[sw, pdm])),
            patch("ragtime.indexer.repository.repository.get_tool_config", new=AsyncMock(return_value=linked)),
            patch("ragtime.mcp.tools.get_app_settings", new=AsyncMock(return_value={})),
            patch.object(adapter, "_check_heartbeats", new=AsyncMock(return_value={"pdm": ToolHealthStatus("pdm", pdm_healthy)})),
            patch.object(adapter, "_create_executor", new=AsyncMock(return_value=AsyncMock())),
            patch.object(adapter, "_create_schema_search_tool_definition", new=AsyncMock(return_value=None)),
            patch.object(adapter, "get_ssh_transfer_definition", new=AsyncMock(return_value=None)),
            patch.object(adapter, "resolve_canonical_tool_id", new=AsyncMock(return_value=None)),
            patch.object(mcp_server, "authorize_external_content", new=AsyncMock()),
        ):
            definitions = await adapter.get_available_tools(route_filter=route)
            definition = next(item for item in definitions if item.tool_config["id"] == "sw")
            uri = build_skill_uri(SKILL_OWNER_TOOL, "pdm", "vault")
            assert ("Unique vault index description" in definition.description) is visible
            assert (uri in definition.description) is visible
            assert (uri in visible_skills(definitions, None, [])) is visible
            result = await call_read_skill(server, uri)
            assert bool(result.isError) is not visible
            if visible:
                assert "instructions" in first_text(result)

    asyncio.run(run())


def test_chat_hides_pdm_from_catalog_and_resolves_linked_skill():
    async def run():
        rag = make_rag_components(tool_skills_enabled=True)
        pdm = {"id": "pdm", "name": "Vault", "tool_type": "solidworks_pdm", "skills": [skill("vault")]}
        sw = {"id": "sw", "name": "Design", "tool_type": "solidworks", "connection_config": {"pdm_tool_id": "pdm"}}
        rag._tool_configs = [sw, pdm]
        tools = [make_noop_tool("solidworks_design"), make_noop_tool("search_vault"), make_noop_tool("lookup_vault")]
        state = ToolSkillBindingState(requested_ids=["tool_config:sw", "tool_config:pdm"], effective_ids=[])
        with (
            patch.object(rag, "_create_ssh_transfer_tool", new=AsyncMock(return_value=None)),
            patch("ragtime.rag.components.content_protection_service.access_guidance", new=AsyncMock(return_value=None)),
        ):
            context = await rag._build_request_runtime_context(
                is_ui=False,
                executor=None,
                blocked_tool_names=None,
                workspace_context=None,
                add_chat_visualization_prompt=False,
                runtime_tool_source_override=tools,
                tool_skill_binding_state_override=state,
            )
        runtime = context["runtime_tools"]
        names = {tool.name for tool in runtime}
        assert "solidworks_design" in names
        assert "search_vault" not in names and "lookup_vault" not in names
        assert "tool_config:pdm" not in state.effective_ids
        load_tools = next(tool for tool in runtime if tool.name == "load_tool_skills")
        assert "unknown_tool_skill_ids" in await load_tools.ainvoke({"ids": ["tool_config:pdm"]})
        loader = next(tool for tool in runtime if tool.name == "read_skill")
        uri = build_skill_uri(SKILL_OWNER_TOOL, "pdm", "vault")
        assert "instructions" in await loader.ainvoke({"uri": uri})
        prompt = rag._build_request_system_prompt(
            is_ui=False,
            mode="chat",
            allowed_tool_config_ids=None,
            runtime_tools=runtime,
            hidden_tool_config_ids=["pdm"],
        )
        assert uri in prompt
        assert "**Vault**" not in prompt

    asyncio.run(run())


def test_chat_solidworks_description_is_request_scoped():
    async def run():
        rag = make_rag_components(tool_skills_enabled=False)
        pdm = {"id": "pdm", "name": "Vault", "tool_type": "solidworks_pdm", "description": "Private vault description"}
        sw = {"id": "sw", "name": "Design", "tool_type": "solidworks", "connection_config": {"pdm_tool_id": "pdm"}}
        rag._tool_configs = [sw, pdm]
        linked = SimpleNamespace(name="Vault", tool_type=SimpleNamespace(value="solidworks_pdm"), enabled=True)
        with patch("ragtime.rag.components.repository.get_tool_config", new=AsyncMock(return_value=linked)):
            tool = await rag.build_primary_runtime_tool_from_config(sw)
        assert tool is not None
        assert "Private vault description" not in tool.description
        userspace_tool = rag._apply_mode_specific_tool_description_overrides([tool], mode="userspace")[0]
        assert userspace_tool.description == tool.description

        async def context(blocked):
            with (
                patch.object(rag, "_create_ssh_transfer_tool", new=AsyncMock(return_value=None)),
                patch("ragtime.rag.components.content_protection_service.access_guidance", new=AsyncMock(return_value=None)),
            ):
                return await rag._build_request_runtime_context(
                    is_ui=False,
                    executor=None,
                    blocked_tool_names=blocked,
                    workspace_context=None,
                    add_chat_visualization_prompt=False,
                    runtime_tool_source_override=[tool, make_noop_tool("search_vault"), make_noop_tool("lookup_vault")],
                )

        excluded = await context({"search_vault", "lookup_vault"})
        included = await context(set())
        excluded_tool = next(item for item in excluded["runtime_tools"] if item.name == "solidworks_design")
        included_tool = next(item for item in included["runtime_tools"] if item.name == "solidworks_design")
        assert excluded["allowed_tool_config_ids"] == ["sw"]
        assert "Private vault description" not in excluded_tool.description
        assert "Private vault description" in included_tool.description
        assert "Private vault description" not in tool.description

    asyncio.run(run())
