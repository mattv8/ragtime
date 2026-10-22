import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, TextContent

from ragtime.content_protection import external as content_protection_external
from ragtime.content_protection.models import ContentProtectionError
from ragtime.mcp import server as mcp_server
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_accounts import AccountWorkspaceCreateRequest, AccountWorkspaceListRequest
from ragtime.userspace.development_service import development_service


class McpWorkspaceManagementTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_account_tools_are_listed_and_dispatch_before_workspace_id(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")
        response = {"items": [], "total": 0, "offset": 0, "next_offset": None}
        with (
            mock.patch("ragtime.mcp.server.account_development_service.list_workspaces", new=mock.AsyncMock(return_value=response)) as list_workspaces,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.mcp_request_context(principal):
                tools = await mcp_server._development_tools_for_principal(principal)  # pyright: ignore[reportPrivateUsage]
                result = await mcp_server._execute_development_tool("workspace_development_workspaces", {})  # pyright: ignore[reportPrivateUsage]

        self.assertIn("workspace_development_workspaces", {tool.name for tool in tools})
        self.assertFalse(result.isError)
        list_workspaces.assert_awaited_once()
        request = list_workspaces.await_args.args[1]
        self.assertIsInstance(request, AccountWorkspaceListRequest)
        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text), response)

    async def test_context_and_all_readers_advertise_read_only_annotations(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))

        tools = await mcp_server._development_tools_for_principal(principal)  # pyright: ignore[reportPrivateUsage]

        reader_names = {
            "workspace_development_context",
            "workspace_development_document",
            "workspace_development_contract",
            "workspace_development_resources",
            "workspace_development_resource_description",
            "workspace_development_resource_contract",
            "workspace_development_facts",
        }
        annotated = {tool.name: tool.annotations for tool in tools if tool.name in reader_names}
        self.assertEqual(set(annotated), reader_names)
        for annotations in annotated.values():
            assert annotations is not None
            self.assertTrue(annotations.readOnlyHint)
            self.assertFalse(annotations.destructiveHint)
            self.assertTrue(annotations.idempotentHint)

    async def test_account_create_dispatches_the_validated_create_request(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")
        response = {"status": "completed", "workspace_id": "workspace-1", "name": "Sandbox", "replayed": False}
        with (
            mock.patch("ragtime.mcp.server.account_development_service.create_workspace", new=mock.AsyncMock(return_value=response)) as create_workspace,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.mcp_request_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development_create",
                    {"name": "Sandbox", "idempotency_key": "create-sandbox-001"},
                )

        self.assertFalse(result.isError)
        self.assertIsInstance(create_workspace.await_args.args[1], AccountWorkspaceCreateRequest)

    async def test_account_errors_use_safe_codes_and_retry_after(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")
        cases = (
            (HTTPException(status_code=429, detail={"code": "workspace_create_throttled", "retry_after": 60}), "workspace_create_throttled", 60),
            (HTTPException(status_code=409, detail="idempotency_key_conflict"), "idempotency_key_conflict", None),
            (HTTPException(status_code=503, detail="account_tools_unavailable"), "account_tools_unavailable", None),
            (RuntimeError("internal submitted value"), "operation_failed", None),
        )
        for error, code, retry_after in cases:
            with self.subTest(code=code):
                with (
                    mock.patch("ragtime.mcp.server.account_development_service.list_workspaces", new=mock.AsyncMock(side_effect=error)),
                    mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
                ):
                    with mcp_server.mcp_request_context(principal):
                        result = await mcp_server._execute_development_tool("workspace_development_workspaces", {})  # pyright: ignore[reportPrivateUsage]
                content = result.content[0]
                assert isinstance(content, TextContent)
                payload = json.loads(content.text)["error"]
                self.assertEqual(payload["code"], code)
                self.assertEqual(payload.get("retry_after"), retry_after)
                self.assertNotIn("submitted value", content.text)

    async def test_outbound_denial_marks_only_completed_create_as_withheld(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")
        for status, execution_status in (("completed", "completed_response_withheld"), ("pending", None), ("interrupted", None)):
            with self.subTest(status=status):
                core = SimpleNamespace(
                    ProtectionContext=content_protection_external.service.ProtectionContext,
                    protection_context=lambda _context: nullcontext(),
                    authorize_content=mock.AsyncMock(side_effect=[None, ContentProtectionError("content_denied", "request-1", reason="Result release denied.")]),
                )
                result_body = {"status": status, "workspace_id": "workspace-1"}
                with (
                    mock.patch("ragtime.mcp.server.account_development_service.create_workspace", new=mock.AsyncMock(return_value=result_body)) as create_workspace,
                    mock.patch.object(content_protection_external, "_core", return_value=core),
                ):
                    with mcp_server.mcp_request_context(principal):
                        result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                            "workspace_development_create",
                            {"name": "Sandbox", "idempotency_key": "create-sandbox-001"},
                        )

                self.assertTrue(result.isError)
                content = result.content[0]
                assert isinstance(content, TextContent)
                self.assertEqual(json.loads(content.text)["error"].get("execution_status"), execution_status)
                create_workspace.assert_awaited_once()

    async def test_ineligible_account_calls_fail_before_argument_or_classifier_work(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, credential_id="rtdev-1", workspace_id="workspace-1", auth_source="workspace_credential")
        with mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()) as authorize:
            with mcp_server.mcp_request_context(principal):
                result = await mcp_server._execute_development_tool("workspace_development_create", {"unexpected": "value"})  # pyright: ignore[reportPrivateUsage]

        content = result.content[0]
        assert isinstance(content, TextContent)
        self.assertEqual(json.loads(content.text)["error"]["code"], "account_tools_unavailable")
        authorize.assert_not_awaited()

    async def test_guessed_account_tools_fail_closed_on_registered_handlers(self) -> None:
        server = Server("account-tool-denial")
        mcp_server._register_handlers(server, development_tools=True)  # pyright: ignore[reportPrivateUsage]
        principals = (
            DevelopmentPrincipal(user_id="user-1", is_admin=False, credential_id="rtdev-1", workspace_id="workspace-1", auth_source="workspace_credential"),
            DevelopmentPrincipal(user_id="user-1", is_admin=False),
        )
        with mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()):
            for principal in principals:
                with self.subTest(principal=principal.auth_source):
                    with mcp_server.mcp_request_context(principal):
                        response = await server.request_handlers[CallToolRequest](
                            CallToolRequest(params=CallToolRequestParams(name="workspace_development_workspaces", arguments={}))
                        )
                    assert isinstance(response.root, CallToolResult)
                    self.assertTrue(response.root.isError)
        custom_server = Server("custom-account-tool-denial")
        mcp_server._register_handlers(custom_server, development_tools=False)  # pyright: ignore[reportPrivateUsage]
        with mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()):
            with mcp_server.mcp_request_context(DevelopmentPrincipal(user_id="user-1", is_admin=False, auth_source="user")):
                response = await custom_server.request_handlers[CallToolRequest](
                    CallToolRequest(params=CallToolRequestParams(name="workspace_development_workspaces", arguments={}))
                )
        assert isinstance(response.root, CallToolResult)
        self.assertTrue(response.root.isError)

    async def test_management_custom_empty_selection_reaches_management_dispatch(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read", "write"}), auth_source="user")
        with (
            mock.patch.object(development_service, "execute", new=mock.AsyncMock(return_value={"selected_tool_ids": []})) as execute,
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.mcp_request_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development",
                    {"workspace_id": "workspace-1", "operation": "workspace_tools_set", "arguments": {"mode": "custom", "selected_tool_ids": []}},
                )

        self.assertFalse(result.isError)
        execute.assert_awaited_once_with(
            principal,
            "workspace-1",
            "workspace_tools_set",
            {"mode": "custom", "selected_tool_ids": []},
        )

    async def test_workspace_credentials_do_not_disclose_user_only_management_operations(self) -> None:
        principal = DevelopmentPrincipal(
            user_id="user-1",
            is_admin=False,
            credential_id="credential-1",
            workspace_id="workspace-1",
            scopes=frozenset({"read", "write"}),
            auth_source="workspace_credential",
        )

        tools = await mcp_server._development_tools_for_principal(principal)  # pyright: ignore[reportPrivateUsage]

        operations = next(tool for tool in tools if tool.name == "workspace_development").inputSchema["properties"]["operation"]["enum"]
        self.assertNotIn("workspace_tools_available", operations)
        self.assertNotIn("workspace_tools_set", operations)
        self.assertNotIn("workspace_development_workspaces", {tool.name for tool in tools})

    async def test_generic_resources_wrapper_pages_protected_ssh_catalog(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        catalog = {
            "authorized_tools": [
                {
                    "component_id": "ssh-1",
                    "name": "Synthetic SSH",
                    "description": "Bounded SSH metadata",
                    "ssh_operations": {"ssh_execute": {"supported": False}, "ssh_transfer": {"supported": False}},
                }
            ],
            "authorized_indexes": [],
        }
        with (
            mock.patch.object(development_service, "list_operations", return_value=[{"name": "resources", "scope": "read"}]),
            mock.patch.object(development_service, "get_resource_catalog", new=mock.AsyncMock(return_value=catalog)),
            mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()),
        ):
            with mcp_server.mcp_request_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development",
                    {"workspace_id": "workspace-1", "operation": "resources", "arguments": {}},
                )

        content = result.content[0]
        assert isinstance(content, TextContent)
        page = json.loads(content.text)
        self.assertEqual(page["items"][0]["component_id"], "ssh-1")
        self.assertNotIn("ssh_operations", page["items"][0])

    async def test_generic_resources_rejects_invalid_arguments_before_catalog_io(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        with (
            mock.patch.object(development_service, "list_operations", return_value=[{"name": "resources", "scope": "read"}]),
            mock.patch.object(development_service, "get_resource_catalog", new=mock.AsyncMock()) as get_catalog,
        ):
            with mcp_server.mcp_request_context(principal):
                result = await mcp_server._execute_development_tool(  # pyright: ignore[reportPrivateUsage]
                    "workspace_development",
                    {"workspace_id": "workspace-1", "operation": "resources", "arguments": {"limit": True}},
                )
        self.assertTrue(result.isError)
        get_catalog.assert_not_awaited()
