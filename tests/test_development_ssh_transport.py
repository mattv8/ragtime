import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ListToolsRequest, ListToolsResult, TextContent

from ragtime.mcp.server import _register_handlers, mcp_request_context
from ragtime.mcp.tools import MCPToolAdapter
from ragtime.userspace import development_routes
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service


class DevelopmentSshTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.workspace = SimpleNamespace(id="workspace-1")

    def _service_boundary_patches(self, stack: ExitStack) -> mock.AsyncMock:
        enforce_role = mock.AsyncMock(return_value=self.workspace)
        stack.enter_context(mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", new=enforce_role))
        stack.enter_context(mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", new=mock.AsyncMock()))
        stack.enter_context(mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()))
        return enforce_role

    async def test_mcp_handler_lists_exec_scoped_ssh_operations_without_global_catalog(self) -> None:
        server = Server("development-ssh-list")
        adapter = MCPToolAdapter()
        _register_handlers(server, adapter)

        credential = DevelopmentPrincipal(
            user_id="user-1", is_admin=False, credential_id="credential-1", workspace_id="workspace-1", scopes=frozenset({"exec"})
        )
        readonly = DevelopmentPrincipal(user_id="user-1", is_admin=False, credential_id="credential-1", workspace_id="workspace-1", scopes=frozenset({"read"}))
        with mcp_request_context(credential):
            tools = await server.request_handlers[ListToolsRequest](ListToolsRequest())
        with mcp_request_context(readonly):
            readonly_tools = await server.request_handlers[ListToolsRequest](ListToolsRequest())

        assert isinstance(tools.root, ListToolsResult)
        assert isinstance(readonly_tools.root, ListToolsResult)
        development_tool = next(tool for tool in tools.root.tools if tool.name == "workspace_development")
        enum = development_tool.inputSchema["properties"]["operation"]["enum"]
        self.assertIn("ssh_execute", enum)
        self.assertIn("ssh_transfer", enum)
        self.assertNotIn("ssh_docker_1", {tool.name for tool in tools.root.tools})
        self.assertNotIn("ssh_transfer", {tool.name for tool in tools.root.tools})
        readonly_development_tool = next(tool for tool in readonly_tools.root.tools if tool.name == "workspace_development")
        readonly_enum = readonly_development_tool.inputSchema["properties"]["operation"]["enum"]
        self.assertNotIn("ssh_execute", readonly_enum)
        self.assertNotIn("ssh_transfer", readonly_enum)

    async def test_mcp_ssh_execute_missing_write_is_denied_before_command_executor(self) -> None:
        server = Server("development-ssh-execute")
        adapter = MCPToolAdapter()
        resolve_patch = mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="workspace_development"))
        resolve_patch.start()
        self.addCleanup(resolve_patch.stop)
        _register_handlers(server, adapter)
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, credential_id="credential-1", workspace_id="workspace-1", scopes=frozenset({"exec"}))
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="workspace_development",
                arguments={"workspace_id": "workspace-1", "operation": "ssh_execute", "arguments": {"component_id": "ssh-1", "command": "id"}},
            )
        )
        with ExitStack() as stack:
            self._service_boundary_patches(stack)
            stack.enter_context(mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()))
            with mcp_request_context(principal):
                result = await server.request_handlers[CallToolRequest](request)

        assert isinstance(result.root, CallToolResult)
        self.assertTrue(result.root.isError)
        content = result.root.content[0]
        assert isinstance(content, TextContent)
        self.assertIn("requires write scope", json.loads(content.text)["error"]["message"])

    async def test_development_credential_cannot_call_global_ssh_tools(self) -> None:
        server = Server("development-ssh-isolation")
        adapter = MCPToolAdapter()
        resolve_patch = mock.patch.object(adapter, "resolve_canonical_tool_id", new=mock.AsyncMock(return_value="ssh-1"))
        execute_tool = mock.AsyncMock()
        execute_patch = mock.patch.object(adapter, "execute_tool", new=execute_tool)
        resolve_patch.start()
        execute_patch.start()
        self.addCleanup(resolve_patch.stop)
        self.addCleanup(execute_patch.stop)
        _register_handlers(server, adapter)
        principal = DevelopmentPrincipal(
            user_id="user-1", is_admin=False, credential_id="credential-1", workspace_id="workspace-1", scopes=frozenset({"exec", "write"})
        )
        with ExitStack() as stack:
            stack.enter_context(mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()))
            for name in ("ssh_docker_1", "ssh_transfer"):
                with self.subTest(name=name), mcp_request_context(principal):
                    result = await server.request_handlers[CallToolRequest](CallToolRequest(params=CallToolRequestParams(name=name, arguments={})))
                assert isinstance(result.root, CallToolResult)
                self.assertTrue(result.root.isError)
                content = result.root.content[0]
                assert isinstance(content, TextContent)
                self.assertEqual(json.loads(content.text)["error"]["code"], "tool_not_available")
        execute_tool.assert_not_awaited()

    async def test_http_dispatcher_rejects_cross_workspace_credential_before_role_lookup(self) -> None:
        principal = DevelopmentPrincipal(
            user_id="user-1", is_admin=False, credential_id="credential-1", workspace_id="workspace-1", scopes=frozenset({"exec", "write"})
        )
        enforce_role = mock.AsyncMock(return_value=self.workspace)
        with (
            mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", new=enforce_role),
            mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()),
        ):
            with self.assertRaises(HTTPException) as raised:
                await development_routes.execute_operation(
                    "workspace-2",
                    "ssh_execute",
                    development_routes.DevelopmentOperationRequest(arguments={"component_id": "ssh-1", "command": "id"}),
                    principal,
                )

        self.assertEqual(raised.exception.status_code, 403)
        enforce_role.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
