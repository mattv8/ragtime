import unittest
from unittest import mock

from mcp.server import Server
from mcp.types import ListToolsRequest, ListToolsResult, Tool, ToolAnnotations

from ragtime.mcp import server as mcp_server
from ragtime.mcp.tools import MCPToolAdapter, MCPToolDefinition
from ragtime.rag.components import RAGComponents
from ragtime.solidworks.tool import SOLIDWORKS_TOOL_INPUT_SCHEMA, build_solidworks_tool_description


class SolidworksWiringTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "id": "sw-1",
            "name": "  Design / Shop -- 2024! ",
            "tool_type": "solidworks",
            "description": "Engineering models",
            "connection_config": {},
            "timeout_max_seconds": 45,
        }

    async def test_runtime_factory_and_content_protection_alias(self):
        components = RAGComponents()
        components._tool_configs = [self.config]
        fake_tool = mock.Mock(name="tool")
        with mock.patch("ragtime.solidworks.tool.create_solidworks_tool", return_value=fake_tool) as factory:
            tools = await components.build_tools_from_runtime_config(self.config)

        self.assertEqual(tools, [fake_tool])
        factory.assert_called_once_with({**self.config, "pdm_link_name": None, "pdm_link": None}, "design_shop_2024")
        self.assertEqual(components._derive_config_tool_names(self.config), {"solidworks_design_shop_2024"})
        self.assertEqual(components._content_protection_tool_ids(), {"solidworks_design_shop_2024": "sw-1"})

    async def test_mcp_definition_and_executor_forward_arguments(self):
        adapter = MCPToolAdapter()
        fake_tool = mock.Mock()
        result = '{"ok":true,"data":"' + "x" * 20000 + '"}'
        fake_tool.ainvoke = mock.AsyncMock(return_value=result)
        with mock.patch.object(RAGComponents, "build_primary_runtime_tool_from_config", new_callable=mock.AsyncMock, return_value=fake_tool) as builder:
            definition = await adapter._create_tool_definition(self.config)

        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition.name, "solidworks_design_shop_2024")
        self.assertEqual(definition.description, build_solidworks_tool_description(self.config))
        self.assertEqual(definition.input_schema, SOLIDWORKS_TOOL_INPUT_SCHEMA)
        self.assertEqual(definition.input_schema["required"], ["operation"])
        arguments = {"operation": "describe", "arguments": {"page": 2}, "timeout": 17}
        self.assertEqual(await definition.execute_fn(**arguments), result)
        builder.assert_awaited_once_with(self.config)
        fake_tool.ainvoke.assert_awaited_once_with(arguments)

    async def test_mcp_outer_deadline_uses_solidworks_cap_when_maximum_is_zero(self):
        config = {**self.config, "timeout_max_seconds": 0}
        definition = MCPToolDefinition("solidworks_design", "SolidWorks", SOLIDWORKS_TOOL_INPUT_SCHEMA, config, mock.AsyncMock(return_value="ok"))
        adapter = MCPToolAdapter()
        with mock.patch.object(adapter, "_execute_with_timeout", new=mock.AsyncMock(return_value="ok")) as run:
            self.assertEqual(await adapter._execute_tool_definition(definition, {}), "ok")
        assert run.await_args is not None
        self.assertEqual(run.await_args.kwargs["timeout_max_seconds"], 100)

    async def test_runtime_factory_forwards_to_service(self):
        components = RAGComponents()
        response = '{"ok":true,"data":{"example":"value"}}'
        with mock.patch("ragtime.solidworks.tool.solidworks_service.execute", new_callable=mock.AsyncMock, return_value=response) as execute:
            tool = await components.build_primary_runtime_tool_from_config(self.config)
            self.assertIsNotNone(tool)
            assert tool is not None
            self.assertEqual(tool.name, MCPToolAdapter()._build_tool_name(self.config))
            result = await tool.ainvoke({"operation": "guide", "arguments": {"topic": "overview"}, "timeout": 12})
        self.assertEqual(result, response)
        execute.assert_awaited_once_with({**self.config, "pdm_link_name": None, "pdm_link": None}, "guide", {"topic": "overview"}, 12)

    async def test_runtime_factory_includes_enabled_pdm_name(self):
        from ragtime.indexer.models import ToolConfig, ToolType

        pdm = ToolConfig(id="pdm", name="Vault", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={})
        config = {**self.config, "connection_config": {"pdm_tool_id": "pdm"}}
        with (
            mock.patch("ragtime.rag.components.repository.get_tool_config", new=mock.AsyncMock(return_value=pdm)),
            mock.patch("ragtime.solidworks.tool.create_solidworks_tool") as factory,
        ):
            await RAGComponents()._create_solidworks_tool(config, "design", "sw-1")
        factory.assert_called_once_with({**config, "pdm_link_name": "Vault", "pdm_link": None}, "design")

    async def test_mcp_description_receives_usable_pdm_name(self):
        from ragtime.indexer.models import ToolConfig, ToolType

        adapter = MCPToolAdapter()
        config = {**self.config, "connection_config": {"pdm_tool_id": "pdm"}}
        cases = [
            (ToolConfig(id="pdm", name="Vault", tool_type=ToolType.SOLIDWORKS_PDM, connection_config={}), "Vault"),
            (None, None),
            (ToolConfig(id="pdm", name="Vault", enabled=False, tool_type=ToolType.SOLIDWORKS_PDM, connection_config={}), None),
            (ToolConfig(id="pdm", name="Wrong", tool_type=ToolType.POSTGRES, connection_config={}), None),
        ]
        for linked, expected in cases:
            with (
                self.subTest(linked=linked),
                mock.patch("ragtime.mcp.tools.get_tool_configs", new=mock.AsyncMock(return_value=[config])),
                mock.patch("ragtime.indexer.repository.repository.get_tool_config", new=mock.AsyncMock(return_value=linked)),
                mock.patch("ragtime.mcp.tools.get_app_settings", new=mock.AsyncMock(return_value={})),
                mock.patch("ragtime.mcp.tools.build_solidworks_tool_description", wraps=build_solidworks_tool_description) as description,
                mock.patch.object(adapter, "_check_heartbeats", new=mock.AsyncMock(return_value={})),
                mock.patch.object(adapter, "_create_executor", new=mock.AsyncMock(return_value=mock.AsyncMock())),
                mock.patch.object(adapter, "_create_schema_search_tool_definition", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(adapter, "_create_knowledge_search_tool", new=mock.AsyncMock(return_value=None)),
                mock.patch.object(adapter, "_create_git_history_tools", new=mock.AsyncMock(return_value=[])),
                mock.patch.object(adapter, "get_ssh_transfer_definition", new=mock.AsyncMock(return_value=None)),
            ):
                definitions = await adapter.get_available_tools()
            self.assertEqual(description.call_args.args[0]["pdm_link_name"], expected)
            self.assertEqual(definitions[0].tool_config["pdm_link_name"], expected)

    async def test_mcp_list_annotations_preserve_other_tool_types(self):
        adapter = MCPToolAdapter()
        server = Server("solidworks-annotations-test")
        definitions = [
            MCPToolDefinition("solidworks_design", "SolidWorks", {}, self.config, mock.AsyncMock()),
            MCPToolDefinition("request_api", "API", {}, {"tool_type": "http_api"}, mock.AsyncMock()),
            MCPToolDefinition("ssh_transfer", "Transfer", {}, {}, mock.AsyncMock(), is_synthetic=True),
        ]
        mcp_server._register_handlers(server, adapter, development_tools=False)
        with mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=definitions)):
            result = await server.request_handlers[ListToolsRequest](ListToolsRequest())

        self.assertIsInstance(result.root, ListToolsResult)
        assert isinstance(result.root, ListToolsResult)
        tools = {tool.name: tool for tool in result.root.tools}
        self.assertIs(_annotations(tools["solidworks_design"]).readOnlyHint, True)
        self.assertIs(_annotations(tools["solidworks_design"]).destructiveHint, False)
        self.assertIsNone(tools["request_api"].annotations)
        self.assertIs(_annotations(tools["ssh_transfer"]).readOnlyHint, False)
        self.assertIs(_annotations(tools["ssh_transfer"]).destructiveHint, True)

    async def test_mcp_solidworks_annotation_matrix(self):
        adapter = MCPToolAdapter()
        server = Server("solidworks-pdm-annotations")
        definitions = [
            MCPToolDefinition(
                f"sw_{allow_write}_{bool(folders)}",
                "SolidWorks",
                {},
                {**self.config, "allow_write": allow_write, "connection_config": {"pdm_approved_folders": folders}},
                mock.AsyncMock(),
            )
            for allow_write in (False, True)
            for folders in ([], [r"\A"])
        ]
        mcp_server._register_handlers(server, adapter, development_tools=False)
        with mock.patch.object(adapter, "get_available_tools", new=mock.AsyncMock(return_value=definitions)):
            result = await server.request_handlers[ListToolsRequest](ListToolsRequest())
        assert isinstance(result.root, ListToolsResult)
        for tool in result.root.tools:
            self.assertEqual(_annotations(tool).readOnlyHint, tool.name != "sw_True_True")
            self.assertEqual(_annotations(tool).destructiveHint, tool.name == "sw_True_True")


def _annotations(tool: Tool) -> ToolAnnotations:
    """Return a listed tool's annotations; the protocol types them as optional."""
    assert tool.annotations is not None
    return tool.annotations


if __name__ == "__main__":
    unittest.main()
