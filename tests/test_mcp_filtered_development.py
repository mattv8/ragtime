import unittest
from unittest import mock

from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ListToolsRequest, ListToolsResult, TextContent

from ragtime.mcp import server
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, MCPToolDefinition
from ragtime.userspace.development_access import DevelopmentPrincipal


class McpFilteredDevelopmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_filtered_default_exposes_and_calls_development_tools_but_custom_does_not(self) -> None:
        route_filter = McpRouteFilter(tool_config_ids=[])
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        default_adapter = MCPToolAdapter()
        custom_adapter = MCPToolAdapter()
        default_server = Server("filtered-default")
        custom_server = Server("custom")

        server._register_handlers(default_server, default_adapter, route_filter, development_tools=True)
        server._register_handlers(custom_server, custom_adapter, route_filter, development_tools=False)

        with (
            mock.patch.object(default_adapter, "get_available_tools", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(custom_adapter, "get_available_tools", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(default_adapter, "_get_configs_for_name_resolution", new=mock.AsyncMock(return_value=[])),
            mock.patch.object(custom_adapter, "_get_configs_for_name_resolution", new=mock.AsyncMock(return_value=[])),
            mock.patch(
                "ragtime.mcp.server._execute_development_tool",
                new=mock.AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="ok")])),
            ) as execute_development_tool,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()) as authorize_content,
        ):
            with server.mcp_request_context(principal):
                default_listed = await default_server.request_handlers[ListToolsRequest](ListToolsRequest())
                custom_listed = await custom_server.request_handlers[ListToolsRequest](ListToolsRequest())
                default_called = await default_server.request_handlers[CallToolRequest](
                    CallToolRequest(params=CallToolRequestParams(name="workspace_development_context", arguments={"workspace_id": "workspace-1"}))
                )
                custom_called = await custom_server.request_handlers[CallToolRequest](
                    CallToolRequest(params=CallToolRequestParams(name="workspace_development_context", arguments={"workspace_id": "workspace-1"}))
                )

        assert isinstance(default_listed.root, ListToolsResult)
        assert isinstance(custom_listed.root, ListToolsResult)
        assert isinstance(default_called.root, CallToolResult)
        assert isinstance(custom_called.root, CallToolResult)
        assert "workspace_development_context" in [tool.name for tool in default_listed.root.tools]
        assert "workspace_development_context" not in [tool.name for tool in custom_listed.root.tools]
        assert not default_called.root.isError
        assert custom_called.root.isError
        execute_development_tool.assert_awaited_once_with("workspace_development_context", {"workspace_id": "workspace-1"})
        self.assertIsNone(server.get_request_development_principal())
        authorize_content.assert_awaited_with(
            {"workspace_id": "workspace-1"},
            direction="inbound",
            principal=principal,
            surface="mcp",
            mcp_route="default",
            tool_id="workspace_development_context",
            operation="tools/call",
        )

    async def test_filtered_general_tools_use_the_filter_and_denied_calls_do_not_execute(self) -> None:
        route_filter = McpRouteFilter(tool_config_ids=["allowed-tool"])
        adapter = MCPToolAdapter()
        registered_server = Server("filtered-default")
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, general_tools_allowed=True)
        tool = MCPToolDefinition("allowed", "general", {}, {}, mock.AsyncMock(return_value="executed"))
        server._register_handlers(registered_server, adapter, route_filter, development_tools=True)

        with (
            mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=[tool])) as get_available_tools,
            mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="denied-tool")),
            mock.patch.object(adapter, "is_tool_allowed_by_route_filter", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(adapter, "execute_tool", new=mock.AsyncMock()) as execute_tool,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with server.mcp_request_context(principal):
                listed = await registered_server.request_handlers[ListToolsRequest](ListToolsRequest())
                denied = await registered_server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name="denied", arguments={})))

        assert isinstance(listed.root, ListToolsResult)
        assert isinstance(denied.root, CallToolResult)
        self.assertEqual([item.name for item in listed.root.tools][:1], ["allowed"])
        self.assertEqual(denied.root.content, [TextContent(type="text", text="Error: Tool 'denied' is not available")])
        self.assertGreaterEqual(get_available_tools.await_count, 1)
        self.assertTrue(all(call == mock.call(route_filter=route_filter) for call in get_available_tools.await_args_list))
        execute_tool.assert_not_awaited()

    async def test_default_and_custom_factories_set_development_purpose(self) -> None:
        default_filter = mock.Mock(
            enabled=True,
            toolSelections=[],
            includeKnowledgeSearch=True,
            includeGitHistory=True,
            selectedDocumentIndexes=None,
            selectedFilesystemIndexes=None,
            selectedSchemaIndexes=None,
            name="Default filter",
        )
        custom_route = mock.Mock(
            enabled=True,
            toolSelections=[],
            includeKnowledgeSearch=True,
            includeGitHistory=True,
            selectedDocumentIndexes=None,
            selectedFilesystemIndexes=None,
            selectedSchemaIndexes=None,
            requireAuth=False,
            authPassword=None,
            authMethod=None,
            allowedLdapGroup=None,
            authClientId=None,
        )
        db = mock.Mock()
        db.mcpdefaultroutefilter.find_unique = mock.AsyncMock(return_value=default_filter)
        db.mcprouteconfig.find_unique = mock.AsyncMock(return_value=custom_route)
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))

        server._default_filter_servers.clear()
        server._default_filter_adapters.clear()
        server._custom_route_servers.clear()
        server._custom_route_adapters.clear()
        try:
            with (
                mock.patch("ragtime.mcp.server.get_db", new=mock.AsyncMock(return_value=db)),
                mock.patch("ragtime.mcp.server.get_app_settings", new=mock.AsyncMock(return_value={"server_name": "Ragtime"})),
                mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
                mock.patch(
                    "ragtime.mcp.server._execute_development_tool",
                    new=mock.AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="ok")])),
                ) as execute_development_tool,
            ):
                default_result = await server.get_default_route_filtered_server("default-filter")
                custom_result = await server.get_custom_route_server("custom")
                assert default_result is not None
                assert custom_result is not None
                default_server, default_adapter = default_result
                custom_server, custom_adapter, *_ = custom_result
                with (
                    mock.patch.object(default_adapter, "get_available_tools", new=mock.AsyncMock(return_value=[])),
                    mock.patch.object(custom_adapter, "get_available_tools", new=mock.AsyncMock(return_value=[])),
                    mock.patch.object(default_adapter, "_get_configs_for_name_resolution", new=mock.AsyncMock(return_value=[])),
                    mock.patch.object(custom_adapter, "_get_configs_for_name_resolution", new=mock.AsyncMock(return_value=[])),
                ):
                    with server.mcp_request_context(principal):
                        default_listed = await default_server.request_handlers[ListToolsRequest](ListToolsRequest())
                        custom_listed = await custom_server.request_handlers[ListToolsRequest](ListToolsRequest())
                        default_called = await default_server.request_handlers[CallToolRequest](
                            CallToolRequest(params=CallToolRequestParams(name="workspace_development_context", arguments={"workspace_id": "workspace-1"}))
                        )
                        custom_called = await custom_server.request_handlers[CallToolRequest](
                            CallToolRequest(params=CallToolRequestParams(name="workspace_development_context", arguments={"workspace_id": "workspace-1"}))
                        )
        finally:
            server._default_filter_servers.clear()
            server._default_filter_adapters.clear()
            server._custom_route_servers.clear()
            server._custom_route_adapters.clear()

        assert isinstance(default_listed.root, ListToolsResult)
        assert isinstance(custom_listed.root, ListToolsResult)
        assert isinstance(default_called.root, CallToolResult)
        assert isinstance(custom_called.root, CallToolResult)
        self.assertIn("workspace_development_context", [item.name for item in default_listed.root.tools])
        self.assertNotIn("workspace_development_context", [item.name for item in custom_listed.root.tools])
        self.assertFalse(default_called.root.isError)
        self.assertTrue(custom_called.root.isError)
        execute_development_tool.assert_awaited_once_with("workspace_development_context", {"workspace_id": "workspace-1"})
