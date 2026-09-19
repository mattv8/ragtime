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
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value={"revision": "abc"})) as execute,
        ):
            with mcp_server.development_principal_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development_context", {"workspace_id": "workspace-1"}
                )

        self.assertFalse(result.isError)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text), {"revision": "abc"})
        execute.assert_awaited_once_with(principal, "workspace-1", "context", {})

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
