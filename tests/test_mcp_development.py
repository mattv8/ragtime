import json
import unittest
from unittest import mock

from mcp.types import TextContent

from ragtime.mcp import routes as mcp_routes
from ragtime.mcp import server as mcp_server
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import development_service


class McpDevelopmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_scoped_to_request_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [
            {"name": "context", "description": "Context", "scope": "read", "input_schema": {}},
            {"name": "file_write", "description": "Write", "scope": "write", "input_schema": {}},
        ]

        with mock.patch.object(development_service, "list_operations", return_value=operations):
            with mcp_server.development_principal_context(principal):
                tools = await mcp_server._development_tools_for_principal(principal)  # pyright: ignore[reportPrivateUsage]
                self.assertIs(mcp_server.get_request_development_principal(), principal)

        self.assertIsNone(mcp_server.get_request_development_principal())
        self.assertEqual(tools[0].inputSchema["properties"]["operation"]["enum"], ["context"])
        readers = {tool.name: tool for tool in tools[1:]}
        self.assertIn("offset", readers["workspace_development_contract"].inputSchema["properties"])
        self.assertIn("offset", readers["workspace_development_resource_contract"].inputSchema["properties"])

    async def test_execution_rechecks_operation_scope_and_returns_mcp_error(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]

        with mock.patch.object(development_service, "list_operations", return_value=operations):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development",
                    {"workspace_id": "workspace-1", "operation": "file_write", "arguments": {}},
                )

        self.assertTrue(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["error"]["code"], "operation_not_allowed")

    async def test_execution_uses_shared_dispatcher_with_request_principal(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]

        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value={"context_revision": "abc"})) as execute,
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development_context", {"workspace_id": "workspace-1"}
                )

        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["context_revision"], "abc")
        execute.assert_awaited_once_with(principal, "workspace-1", "context", {})

    async def test_both_context_paths_use_bounded_compact_delivery(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "context", "description": "Context", "scope": "read", "input_schema": {}}]
        context = {"context_revision": "abc", "capabilities": {"authorized_tools": [], "authorized_indexes": []}}
        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value=context)),
            mock.patch("ragtime.userspace.development_bootstrap._guidance_documents", return_value={"workspace": "small"}),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool("workspace_development_context", {"workspace_id": "workspace-1"})  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertIn("guidance_documents", json.loads(content.text))

    async def test_generic_resources_operation_returns_the_bounded_reader_page(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "resources", "description": "Resources", "scope": "read", "input_schema": {}}]
        context = {"capabilities": {"authorized_tools": [{"component_id": "tool-1", "description": "x" * 70000}], "authorized_indexes": []}}

        async def execute(_principal, _workspace_id, operation, _arguments):
            self.assertEqual(operation, "context")
            return context

        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=execute),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "resources", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        page = json.loads(content.text)
        self.assertIsNone(page["items"][0]["description"])
        self.assertEqual(page["items"][0]["description_reference"]["reader"], "workspace_development_resource_description")
        self.assertLessEqual(len(content.text.encode()), 24 * 1024)

    async def test_error_response_with_huge_code_and_body_is_explicitly_truncated(self) -> None:
        result = mcp_server._development_error("x" * 100000, "\x00" * 100000)  # pyright: ignore[reportPrivateUsage]
        content = result.content[0]
        assert isinstance(content, TextContent)
        payload = json.loads(content.text)
        self.assertTrue(payload["error"]["truncated"])
        self.assertLessEqual(len(content.text.encode("utf-8")), 24 * 1024)

    async def test_large_ordinary_operation_result_preserves_existing_output_semantics(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        operations = [{"name": "file_read", "description": "Read", "scope": "read", "input_schema": {}}]
        result_body = {"content": "\\n\\t" * 20000}
        with (
            mock.patch.object(development_service, "list_operations", return_value=operations),
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value=result_body)),
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(
                    "workspace_development", {"workspace_id": "workspace-1", "operation": "file_read", "arguments": {}}
                )  # pyright: ignore[reportPrivateUsage]
        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text), result_body)

    async def test_oauth_user_becomes_request_local_development_principal(self) -> None:
        scope = {"type": "http", "headers": [], "_mcp_oauth_user": mock.Mock(id="user-1", role="admin")}

        principal = await mcp_routes._resolve_default_development_principal(  # pyright: ignore[reportPrivateUsage]
            scope, auth_method="oauth2", resolved_auth_method="oauth2"
        )

        self.assertEqual(principal, DevelopmentPrincipal(user_id="user-1", is_admin=True))

    async def test_client_credentials_never_resolve_as_development_principal(self) -> None:
        scope = {"type": "http", "headers": []}

        principal = await mcp_routes._resolve_default_development_principal(  # pyright: ignore[reportPrivateUsage]
            scope, auth_method="client_credentials", resolved_auth_method="client_credentials"
        )

        self.assertIsNone(principal)


if __name__ == "__main__":
    unittest.main()
