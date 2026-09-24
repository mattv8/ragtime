"""Regression contracts for the external-development final fix pass."""

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import ToolType
from ragtime.mcp import server as mcp_server
from ragtime.userspace import development_service as development_module
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_routes import DevelopmentOperationRequest, execute_operation, get_operations


class DevelopmentFinalContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_operation_discovery_uses_public_authorized_service_method(self) -> None:
        principal = DevelopmentPrincipal(user_id="user", is_admin=False, scopes=frozenset({"read"}))
        expected = [{"name": "context"}]
        with mock.patch.object(
            development_module.development_service,
            "list_authorized_operations",
            mock.AsyncMock(return_value=expected),
        ) as list_operations:
            result = await get_operations("workspace", principal)

        self.assertEqual(result, {"operations": expected})
        list_operations.assert_awaited_once_with(principal, "workspace")

    def test_sync_conversation_query_propagates_conversation_owner(self) -> None:
        tree = ast.parse(inspect.getsource(indexer_routes._send_message_to_loaded_conversation))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "process_query"]

        self.assertEqual(len(calls), 1)
        owner = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "owner_user_id")
        assert isinstance(owner, ast.Attribute)
        assert isinstance(owner.value, ast.Name)
        self.assertEqual((owner.value.id, owner.attr), ("conv", "user_id"))

    async def test_schema_and_pdm_names_resolve_through_durable_index_identity(self) -> None:
        schema_tool = SimpleNamespace(id="schema-id", name="Sales DB", tool_type=ToolType.POSTGRES, description="Sales schemas", enabled=True)
        pdm_tool = SimpleNamespace(id="pdm-id", name="Vault", tool_type=ToolType.SOLIDWORKS_PDM, description="Parts", enabled=True)
        db = SimpleNamespace(
            schemaindexjob=SimpleNamespace(find_first=mock.AsyncMock(side_effect=[SimpleNamespace(toolConfigId="schema-id"), None])),
            pdmindexjob=SimpleNamespace(find_first=mock.AsyncMock(return_value=SimpleNamespace(toolConfigId="pdm-id"))),
        )
        with (
            mock.patch.object(development_module.repository, "get_index_metadata", mock.AsyncMock(return_value=None)),
            mock.patch.object(development_module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(
                development_module.repository,
                "get_tool_config",
                mock.AsyncMock(side_effect=[schema_tool, pdm_tool]),
            ),
        ):
            schema = await development_module.development_service._resolve_general_index_descriptor("schema_durable_key")
            pdm = await development_module.development_service._resolve_general_index_descriptor("pdm_durable_key")

        self.assertEqual(
            schema, {"name": "schema_durable_key", "source_type": "schema", "description": "Sales schemas", "enabled": True, "backing_tool_id": "schema-id"}
        )
        self.assertEqual(pdm, {"name": "pdm_durable_key", "source_type": "pdm", "description": "Parts", "enabled": True, "backing_tool_id": "pdm-id"})

    async def test_denied_backing_tool_hides_granted_index_and_blocks_retrieval(self) -> None:
        active_principal = DevelopmentPrincipal(user_id="active", is_admin=False, scopes=frozenset({"read"}))
        expired_principal = DevelopmentPrincipal(user_id="expired", is_admin=False, scopes=frozenset({"read"}))
        denied_principal = DevelopmentPrincipal(user_id="denied", is_admin=False, scopes=frozenset({"read"}))
        db = SimpleNamespace(workspaceindexgrant=SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(indexName="schema_durable_key")])))

        class ToolAccessDb:
            async def query_raw(self, query, *params):
                if 'FROM "tool_access_policies"' in query:
                    return [{"tool_config_id": "tool-denied", "default_access": "deny", "allow_write": False}]
                if 'FROM "tool_user_access"' in query:
                    return []
                if 'FROM "tool_auth_group_access"' in query:
                    return (
                        [{"tool_config_id": "tool-denied", "access_level": "read"}]
                        if params[1] == "active"
                        else [{"tool_config_id": "tool-denied", "access_level": "deny"}]
                        if params[1] == "denied"
                        else []
                    )
                raise AssertionError(f"Unexpected query: {query}")

        descriptor = {
            "name": "schema_durable_key",
            "source_type": "schema",
            "enabled": True,
            "backing_tool_id": "tool-denied",
        }
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.planning_service, "_selected_tool_ids", mock.AsyncMock(return_value=[])),
            mock.patch.object(development_module, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(development_module.development_service, "_resolve_general_index_descriptor", mock.AsyncMock(return_value=descriptor)),
            mock.patch("ragtime.core.tool_access.get_db", mock.AsyncMock(return_value=ToolAccessDb())),
        ):
            active_resources = await development_module.development_service._resources(active_principal, "workspace")
            expired_resources = await development_module.development_service._resources(expired_principal, "workspace")
            denied_resources = await development_module.development_service._resources(denied_principal, "workspace")

        self.assertEqual(active_resources["indexes"][-1], descriptor)
        self.assertEqual(expired_resources["indexes"], [{"name": "workspace_code", "source_type": "workspace_code"}])
        self.assertEqual(denied_resources["indexes"], [{"name": "workspace_code", "source_type": "workspace_code"}])

        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value={"indexes": denied_resources["indexes"]})),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(development_module, "search_schema_index", mock.AsyncMock()) as schema_search,
        ):
            with self.assertRaises(HTTPException) as raised:
                await development_module.development_service.execute(
                    denied_principal,
                    "workspace",
                    "index_search",
                    {"index_name": "schema_durable_key", "query": "secret"},
                )

        self.assertEqual(raised.exception.status_code, 403)
        schema_search.assert_not_awaited()

    async def test_resources_enriches_only_explicitly_acl_allowed_tools_with_canonical_contracts(self) -> None:
        principal = DevelopmentPrincipal(user_id="user", is_admin=False, scopes=frozenset({"read"}))
        workspace = SimpleNamespace(id="workspace")
        postgres = SimpleNamespace(
            id="postgres",
            name="Analytics",
            tool_type=SimpleNamespace(value="postgres"),
            description="Configured SQL description, verbatim.",
            connection_config={"password": "SECRET"},
            timeout_max_seconds=45,
            max_results=100,
            enabled=True,
        )
        http = SimpleNamespace(
            id="http",
            name="Orders API",
            tool_type=SimpleNamespace(value="http_api"),
            description="Configured HTTP description, verbatim.",
            connection_config={"bearer_token": "SECRET"},
            timeout_max_seconds=45,
            max_results=100,
            enabled=True,
        )
        built = {
            "postgres": SimpleNamespace(
                description="Query Analytics. This database contains: Configured SQL description, verbatim. Include LIMIT clause to restrict results."
            ),
            "http": SimpleNamespace(
                description="Send HTTP requests to Orders API. This API provides access to: Configured HTTP description, verbatim. Authentication is applied automatically. Only these per-request headers are approved: X-Request-ID. Omit all other headers."
            ),
        }

        async def get_tool_config(tool_id: str):
            if tool_id == "postgres":
                return postgres
            if tool_id == "http":
                return http
            raise AssertionError(f"Unauthorized config lookup: {tool_id}")

        async def build(config):
            return built[config["id"]]

        db = SimpleNamespace(workspaceindexgrant=SimpleNamespace(find_many=mock.AsyncMock(return_value=[])))
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock(return_value=workspace)),
            mock.patch.object(
                development_module.planning_service, "_selected_tool_ids", mock.AsyncMock(return_value=["postgres", "http", "denied", "missing"])
            ),
            mock.patch.object(
                development_module, "resolve_tool_access", mock.AsyncMock(return_value={"postgres": "read", "http": "read_write", "denied": "deny"})
            ),
            mock.patch.object(development_module.repository, "get_tool_config", mock.AsyncMock(side_effect=get_tool_config)),
            mock.patch.object(development_module.rag, "build_primary_runtime_tool_from_config", mock.AsyncMock(side_effect=build)) as builder,
            mock.patch.object(development_module, "get_db", mock.AsyncMock(return_value=db)),
        ):
            resources = await development_module.development_service._resources(principal, "workspace")

        self.assertEqual([tool["component_id"] for tool in resources["tools"]], ["postgres", "http"])
        self.assertIn("Configured SQL description, verbatim.", resources["tools"][0]["description"])
        self.assertIn("Include LIMIT clause", resources["tools"][0]["execute_component"]["instructions"])
        self.assertIn("Configured HTTP description, verbatim.", resources["tools"][1]["description"])
        self.assertIn("Authentication is applied automatically", resources["tools"][1]["execute_component"]["instructions"])
        self.assertNotIn("timeout", resources["tools"][0]["execute_component"]["request_schema"]["oneOf"][1]["properties"])
        self.assertTrue(resources["tools"][0]["execution_lanes"]["browser"]["supported"])
        self.assertEqual(resources["tools"][0]["execution_lanes"]["browser"]["mode"], "browser_read_only")
        self.assertTrue(resources["tools"][0]["execution_lanes"]["server_runtime_bridge"]["supported"])
        self.assertNotIn("SECRET", str(resources))
        self.assertEqual([call.args[0]["id"] for call in builder.await_args_list], ["postgres", "http"])

    async def test_http_and_mcp_dispatchers_do_not_search_hidden_index(self) -> None:
        principal = DevelopmentPrincipal(user_id="viewer", is_admin=False, scopes=frozenset({"read"}))
        arguments = {"index_name": "schema_durable_key", "query": "secret"}
        hidden_resources = {"indexes": [{"name": "workspace_code", "source_type": "workspace_code"}]}
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value=hidden_resources)),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(development_module, "search_schema_index", mock.AsyncMock()) as schema_search,
        ):
            with self.assertRaises(HTTPException) as http_error:
                await execute_operation(
                    "workspace",
                    "index_search",
                    DevelopmentOperationRequest(arguments=arguments),
                    principal,
                )
            with mcp_server.development_principal_context(principal):
                mcp_result = await mcp_server._execute_development_tool(
                    "workspace_development",
                    {"workspace_id": "workspace", "operation": "index_search", "arguments": arguments},
                )

        self.assertEqual(http_error.exception.status_code, 403)
        self.assertTrue(mcp_result.isError)
        schema_search.assert_not_awaited()

    async def test_ungranted_index_search_never_selects_a_global_source(self) -> None:
        principal = DevelopmentPrincipal(user_id="user", is_admin=False)
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value={"indexes": []})),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(development_module, "search_filesystem_index", mock.AsyncMock()) as filesystem_search,
        ):
            with self.assertRaises(HTTPException) as raised:
                await development_module.development_service.execute(principal, "workspace", "index_search", {"index_name": "other", "query": "secret"})

        self.assertEqual(raised.exception.status_code, 403)
        filesystem_search.assert_not_awaited()
