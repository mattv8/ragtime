import unittest
from types import SimpleNamespace
from unittest import mock

from mcp.server import Server
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    ReadResourceRequestParams,
)

import ragtime.mcp.server as mcp_server
from ragtime.content_protection import service as protection_service
from ragtime.content_protection.models import ClassifierConfig, ContentProtectionConfig, ContentProtectionError, Requirement
from ragtime.mcp.skills import read_skill_description, route_instructions
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, MCPToolDefinition
from ragtime.skills.contract import build_skill_uri


def skill(mode="on_demand", content="Follow these instructions."):
    return {"id": "s1", "name": "example", "description": "Helpful guidance", "content": content, "mode": mode, "sort_order": 0}


class InstructionSkillsMcpTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {"id": "tool-1", "name": "Example", "tool_type": "postgres", "skills": [skill()]}
        self.adapter = MCPToolAdapter()
        definition = MCPToolDefinition(
            name="query_example",
            description=self.adapter._build_tool_description(self.config),
            input_schema={"type": "object"},
            tool_config=self.config,
            execute_fn=mock.AsyncMock(),
        )
        self.adapter.get_available_tools = mock.AsyncMock(return_value=[definition])
        self.adapter.resolve_canonical_tool_id = mock.AsyncMock(return_value=None)
        self.adapter.is_tool_allowed_by_route_filter = mock.AsyncMock(return_value=False)
        self.server = Server("test")
        mcp_server._register_handlers(self.server, self.adapter, McpRouteFilter(tool_config_ids=["tool-1"]))
        self.uri = build_skill_uri("tool", "tool-1", "example")

    async def list_tools(self):
        return (await self.server.request_handlers[ListToolsRequest](ListToolsRequest())).root.tools

    async def call(self, uri):
        request = CallToolRequest(params=CallToolRequestParams(name="read_skill", arguments={"uri": uri}))
        return (await self.server.request_handlers[CallToolRequest](request)).root

    def test_description_inline_pointer_and_unchanged(self):
        config = dict(self.config, skills=[])
        self.assertEqual(self.adapter._build_tool_description(config), "Query the 'Example' PostgreSQL database using SQL.")
        self.assertIn(f"(load with read_skill, uri {self.uri})", self.adapter._build_tool_description(self.config))
        config["skills"] = [dict(skill(), description="First line\n  second line")]
        self.assertIn("- example: First line second line (load with read_skill", self.adapter._build_tool_description(config))
        config["skills"] = [skill("always", "- inline text"), dict(skill(), name="second")]
        inline_description = self.adapter._build_tool_description(config)
        self.assertIn("- example (always applied; text below)", inline_description)
        self.assertTrue(inline_description.endswith("Always follow these instructions when using this tool:\n### example\n- inline text"))
        config["skills"] = [skill("always", "x" * 1501)]
        self.assertIn("call read_skill", self.adapter._build_tool_description(config))

    async def test_list_call_filter_and_resources(self):
        with mock.patch.object(mcp_server, "authorize_external_content", new=mock.AsyncMock()) as guard:
            tools = await self.list_tools()
            self.assertIn("read_skill", [tool.name for tool in tools])
            result = await self.call(self.uri)
            self.assertIn("Follow these instructions.", result.content[0].text)
            denied = await self.call("skill://tools/other/example/SKILL.md")
            self.assertTrue(denied.isError)
            listed = await self.server.request_handlers[ListResourcesRequest](ListResourcesRequest())
            self.assertEqual(str(listed.root.resources[0].uri), self.uri)
            request = ReadResourceRequest(params=ReadResourceRequestParams(uri=self.uri))
            resource = await self.server.request_handlers[ReadResourceRequest](request)
            self.assertIn("Follow these instructions.", resource.root.contents[0].text)
            self.assertTrue(any(call.kwargs.get("operation") == "resources/read" for call in guard.await_args_list))

    async def test_tool_scoped_policy_denies_skill_outputs(self):
        policy = ContentProtectionConfig(
            enabled=True,
            classifier=ClassifierConfig(backend="llm", llm_model="openai::classifier"),
            coverage_mode="selected_scopes",
            requirements=[Requirement(scope_kind="tool", scope_key="tool-1", mode="require")],
        )
        classified = []

        async def detect(config, envelope):
            classified.append(envelope)
            probabilities = {category.id: 0.0 for category in config.categories if category.id != "rule_override"}
            probabilities["credentials"] = 1.0
            return {"probabilities": probabilities, "model": "test", "usage": {}, "transport": "test", "cache_hit": False}

        with (
            mock.patch.object(protection_service, "load_config", mock.AsyncMock(return_value=policy)),
            mock.patch.object(protection_service, "resolve_identities", mock.AsyncMock(return_value=(set(), {}, {}))),
            mock.patch.object(protection_service, "_provider_settings_identity", mock.AsyncMock(return_value="test")),
            mock.patch.object(protection_service, "detect", new=detect),
            mock.patch.object(protection_service, "_audit", mock.AsyncMock()),
        ):
            await self.list_tools()
            result = await self.call(self.uri)
            self.assertTrue(result.isError)
            self.assertIn("content_denied", result.content[0].text)
            request = ReadResourceRequest(params=ReadResourceRequestParams(uri=self.uri))
            with self.assertRaises(ContentProtectionError) as raised:
                await self.server.request_handlers[ReadResourceRequest](request)
            self.assertEqual(raised.exception.code, "content_denied")
        self.assertEqual([entry["tool_id"] for entry in classified], ["tool-1", "tool-1"])

    async def test_route_skill_outbound_has_no_tool_owner(self):
        server = Server("route")
        mcp_server._register_handlers(server, self.adapter, route_id="route-1", route_skills=[skill()])
        uri = build_skill_uri("route", "route-1", "example")
        with mock.patch.object(mcp_server, "authorize_external_content", new=mock.AsyncMock()) as guard:
            await server.request_handlers[ListToolsRequest](ListToolsRequest())
            await server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name="read_skill", arguments={"uri": uri})))
            await server.request_handlers[ReadResourceRequest](ReadResourceRequest(params=ReadResourceRequestParams(uri=uri)))
        outbound = [call.kwargs for call in guard.await_args_list if call.kwargs.get("direction") == "outbound"]
        self.assertEqual([call["tool_id"] for call in outbound], [None, None])

    async def test_no_skills_and_collision(self):
        self.config["skills"] = []
        self.assertNotIn("read_skill", [tool.name for tool in await self.list_tools()])
        self.config["skills"] = [skill()]
        self.adapter.get_available_tools.return_value[0].name = "read_skill"
        tools = await self.list_tools()
        self.assertEqual([tool.name for tool in tools].count("read_skill"), 1)
        self.assertNotIn("Skills:", tools[0].description)

    async def test_development_only_has_no_skill_surfaces(self):
        principal = SimpleNamespace(general_tools_allowed=False)
        with (
            mock.patch.object(mcp_server, "get_request_development_principal", return_value=principal),
            mock.patch.object(mcp_server, "authorize_external_content", new=mock.AsyncMock()),
        ):
            with mock.patch.object(mcp_server, "_development_tools_for_principal", mock.AsyncMock(return_value=[])):
                self.assertEqual(await self.list_tools(), [])
            listed = await self.server.request_handlers[ListResourcesRequest](ListResourcesRequest())
            self.assertEqual(listed.root.resources, [])
            self.assertTrue((await self.call(self.uri)).isError)

    def test_route_instructions_and_description(self):
        route = [skill("always", "Route rules."), dict(skill(), name="second")]
        instructions = route_instructions("route-1", route)
        self.assertIn("Route rules.", instructions)
        self.assertIn("with the read_skill tool when it is listed, otherwise resources/read:", instructions)
        self.assertIn("- second: Helpful guidance (uri skill://routes/route-1/second/SKILL.md)", instructions)
        self.assertLess(instructions.index("### example\nRoute rules."), instructions.index("- second:"))
        description = read_skill_description({self.uri: (skill(), "tool Example")})
        self.assertIn(self.uri, description)
        self.assertNotIn("Always applies", description)
        self.assertLessEqual(len(description), 8000)
        route_uri = build_skill_uri("route", "route-1", "example")
        inline_route = read_skill_description({route_uri: (skill("always", "Route rules."), "route")})
        self.assertIn("Always follow these instructions on this server:\n### example\nRoute rules.\n\nSkills:\n- example (route)", inline_route)
        self.assertIn("Always applies; text above.", inline_route)
        long_route = read_skill_description({route_uri: (skill("always", "x" * 1501), "route")})
        self.assertIn("Always applies; load it before using this server's tools.", long_route)
        self.assertNotIn("x" * 1501, long_route)
        self.assertIn("Always applies to that tool.", read_skill_description({self.uri: (skill("always"), "tool Example")}))

    async def test_custom_route_loads_enabled_sorted_skills(self):
        rows = [
            SimpleNamespace(id="b", name="second", description="Second", content="Second rules", mode="always", sortOrder=2, enabled=True),
            SimpleNamespace(id="a", name="first", description="First", content="First rules", mode="always", sortOrder=1, enabled=True),
            SimpleNamespace(id="c", name="hidden", description="Hidden", content="Hidden rules", mode="always", sortOrder=0, enabled=False),
        ]
        route = SimpleNamespace(
            id="route-1",
            enabled=True,
            toolSelections=[],
            instructionSkills=rows,
            includeKnowledgeSearch=False,
            includeGitHistory=False,
            selectedDocumentIndexes=None,
            selectedFilesystemIndexes=None,
            selectedSchemaIndexes=None,
            requireAuth=False,
            authPassword=None,
            authMethod=None,
            allowedLdapGroup=None,
            authClientId=None,
        )
        db = SimpleNamespace(mcprouteconfig=SimpleNamespace(find_unique=mock.AsyncMock(return_value=route)))
        try:
            with (
                mock.patch.object(mcp_server, "get_db", mock.AsyncMock(return_value=db)),
                mock.patch.object(mcp_server, "get_app_settings", mock.AsyncMock(return_value={"server_name": "Test"})),
            ):
                result = await mcp_server.get_custom_route_server("skills-test")
                self.assertIsNotNone(result)
                instructions = result[0].instructions
                self.assertLess(instructions.index("First rules"), instructions.index("Second rules"))
                self.assertNotIn("Hidden rules", instructions)
                self.assertEqual(db.mcprouteconfig.find_unique.await_args.kwargs["include"], {"toolSelections": True, "instructionSkills": True})
                self.assertIsNotNone(await mcp_server.get_custom_route_server("skills-test"))
        finally:
            mcp_server._custom_route_servers.pop("skills-test", None)
            mcp_server._custom_route_adapters.pop("skills-test", None)
