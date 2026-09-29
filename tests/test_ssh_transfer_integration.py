import asyncio
import base64
import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from langchain_core.tools import StructuredTool
from mcp.server import Server
from mcp.types import CallToolRequest, CallToolRequestParams, CallToolResult, ToolAnnotations

from ragtime.content_protection.models import ContentProtectionError
from ragtime.content_protection.service import canonical_serialize, normalize_transport_value
from ragtime.mcp.server import _register_handlers, mcp_request_context
from ragtime.mcp.tools import McpRouteFilter, MCPToolAdapter, MCPToolDefinition
from ragtime.rag.components import FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES, RAGComponents, wrap_tool_with_truncation
from ragtime.tools.ssh_transfer import (
    MAX_INLINE_BYTES,
    SSHTransferInput,
    build_ssh_transfer_tool,
    build_workspace_file_callbacks,
    parse_endpoint,
    ssh_transfer_validation_error,
)

SSH = {
    "id": "ssh-1",
    "name": "Docker 1",
    "tool_type": "ssh_shell",
    "enabled": True,
    "allow_write": True,
    "connection_config": {"host": "host", "user": "user", "working_directory": "/srv"},
}
CORE_OK = {"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": []}


class SSHTransferIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # Direct builder tests are not request transport tests. Keep their focus
        # on transfer behavior while explicit policy tests install real guards.
        self._policy_guard = mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock())
        self._policy_guard.start()

    async def asyncTearDown(self) -> None:
        self._policy_guard.stop()

    async def test_inline_base64_and_visible_endpoint_binding(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=CORE_OK) as core:
            result = json.loads(
                await transfer(
                    source="inline",
                    destination="ssh://docker_1/out.txt",
                    content=base64.b64encode(b"abc").decode(),
                    encoding="base64",
                )
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(core.call_args.kwargs["content"], b"abc")
        self.assertEqual(json.loads(await transfer(source="inline", destination="ssh://hidden/out", content="x"))["status"], "rejected")

    async def test_call_time_config_resolver_enforces_revocation_and_write_downgrade(self) -> None:
        current = [SSH]

        async def resolve() -> list[dict]:
            return current

        transfer = build_ssh_transfer_tool([SSH], visible_config_resolver=resolve)
        current = []
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files") as core:
            revoked = json.loads(await transfer(source="inline", destination="ssh://docker_1/x", content="secret"))
        self.assertEqual(revoked["status"], "rejected")
        core.assert_not_called()

        current = [dict(SSH, allow_write=False)]
        downgraded = json.loads(await transfer(source="inline", destination="ssh://docker_1/x", content="secret"))
        self.assertEqual(downgraded["status"], "rejected")
        self.assertIn("read-only", downgraded["errors"][0])

        current = [dict(SSH, name="New Hidden Alias")]
        renamed = json.loads(await transfer(source="inline", destination="ssh://new_hidden_alias/x", content="secret"))
        self.assertEqual(renamed["status"], "rejected")

    async def test_workspace_requires_callbacks_and_conditional_write(self) -> None:
        writes: list[tuple[str, str, str, bool]] = []

        async def read(workspace_id: str, path: str) -> str:
            self.assertEqual((workspace_id, path), ("w1", "in.txt"))
            return "text"

        async def write(workspace_id: str, path: str, content: str, overwrite: bool) -> None:
            writes.append((workspace_id, path, content, overwrite))

        transfer = build_ssh_transfer_tool([SSH], workspace_read=read, workspace_write=write, workspace_id="w1")
        with mock.patch(
            "ragtime.core.ssh_transfer.transfer_ssh_files",
            return_value=dict(CORE_OK, bytes_transferred=4, content=b"text"),
        ):
            result = json.loads(await transfer(source="ssh://docker_1/in.txt", destination="workspace:/out.txt", workspace_id="w1"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(writes, [("w1", "out.txt", "text", False)])

    async def test_workspace_callbacks_use_acl_and_conditional_no_clobber_write(self) -> None:
        read, write = build_workspace_file_callbacks(user_id="user-1", is_admin=True, allowed_workspace_id="w1")
        service = mock.Mock()
        service.get_workspace_file = mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="missing"))
        service.upsert_workspace_file = mock.AsyncMock()
        with mock.patch("ragtime.userspace.service.userspace_service", service):
            await write("w1", "out.txt", "new", False)
        service.get_workspace_file.assert_awaited_once_with("w1", "out.txt", "user-1", is_admin=True)
        kwargs = service.upsert_workspace_file.await_args.kwargs
        self.assertIsNone(kwargs["expected_content_hash"])
        self.assertTrue(kwargs["require_content_hash"])
        with self.assertRaisesRegex(ValueError, "not authorized"):
            await read("w2", "x")

    async def test_workspace_callbacks_reject_existing_no_clobber_target(self) -> None:
        _, write = build_workspace_file_callbacks(user_id="user-1")
        existing = SimpleNamespace(content="old")
        service = mock.Mock(
            get_workspace_file=mock.AsyncMock(return_value=existing),
            upsert_workspace_file=mock.AsyncMock(),
        )
        with mock.patch("ragtime.userspace.service.userspace_service", service):
            with self.assertRaisesRegex(ValueError, "already exists"):
                await write("w1", "out.txt", "new", False)
        service.upsert_workspace_file.assert_not_awaited()

    async def test_inline_caps_binary_workspace_and_output_budget(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        oversized = json.loads(await transfer(source="inline", destination="ssh://docker_1/x", content="x" * (MAX_INLINE_BYTES + 1)))
        self.assertEqual(oversized["status"], "rejected")
        invalid_base64 = json.loads(await transfer(source="inline", destination="ssh://docker_1/x", content="%%%", encoding="base64"))
        self.assertEqual(invalid_base64["status"], "rejected")

        async def unused_read(_workspace_id: str, _path: str) -> str:
            return ""

        async def unused_write(_workspace_id: str, _path: str, _content: str, _overwrite: bool) -> None:
            raise AssertionError("binary content must not be written")

        workspace_transfer = build_ssh_transfer_tool([SSH], workspace_read=unused_read, workspace_write=unused_write, workspace_id="w1")
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=dict(CORE_OK, content=b"\xff")):
            binary = json.loads(await workspace_transfer(source="ssh://docker_1/x", destination="workspace:/x", workspace_id="w1"))
        self.assertEqual(binary["status"], "rejected")

        budgeted = build_ssh_transfer_tool([SSH], max_output_chars=100)
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=dict(CORE_OK, content=b"x" * 80)):
            budget = json.loads(await budgeted(source="ssh://docker_1/x", destination="inline", encoding="base64"))
        self.assertEqual(budget["status"], "transfer_failed")
        self.assertIn("output budget", budget["errors"][0])

    async def test_cancellation_sets_event_and_drains_worker(self) -> None:
        started = threading.Event()
        drained = threading.Event()

        def worker(*_args: object, cancel_event: threading.Event, **_kwargs: object) -> dict:
            started.set()
            cancel_event.wait(2)
            drained.set()
            return CORE_OK

        transfer = build_ssh_transfer_tool([SSH])
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", side_effect=worker):
            task = asyncio.ensure_future(transfer(source="inline", destination="ssh://docker_1/x", content="x"))
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(drained.is_set())

    async def test_cancellation_reraises_when_drained_worker_fails(self) -> None:
        started = threading.Event()

        def worker(*_args: object, cancel_event: threading.Event, **_kwargs: object) -> dict:
            started.set()
            cancel_event.wait(2)
            raise RuntimeError("worker failed after cancellation")

        transfer = build_ssh_transfer_tool([SSH])
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", side_effect=worker):
            task = asyncio.ensure_future(transfer(source="inline", destination="ssh://docker_1/x", content="x"))
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_per_endpoint_content_policy_blocks_before_network_and_checks_result(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        denied = ContentProtectionError("content_denied", "request-1")
        with (
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.current_context", return_value=object()),
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock(side_effect=denied)) as authorize,
            mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files") as core,
        ):
            with self.assertRaises(ContentProtectionError):
                await transfer(source="inline", destination="ssh://docker_1/x", content="secret")
        core.assert_not_called()
        assert authorize.await_args is not None
        self.assertEqual(authorize.await_args.kwargs["tool_id"], "ssh-1")

        with (
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.current_context", return_value=object()),
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock()) as authorize,
            mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=CORE_OK),
        ):
            await transfer(source="inline", destination="ssh://docker_1/x", content="secret")
        self.assertEqual([call.kwargs["direction"] for call in authorize.await_args_list], ["proposed_operation", "tool_result"])

    async def test_policy_candidates_serialize_utf8_bytes_without_mutating_transfer_bytes(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        with mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock()) as authorize:
            with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=dict(CORE_OK, content=b"download text")) as core:
                await transfer(source="inline", destination="ssh://docker_1/x", content=base64.b64encode(b"upload text").decode(), encoding="base64")
        candidates = [call.args[0] for call in authorize.await_args_list]
        self.assertEqual(candidates[0]["content"], "upload text")
        self.assertEqual(candidates[1]["content"], "download text")
        for candidate in candidates:
            self.assertEqual(normalize_transport_value(candidate), candidate)
            canonical_serialize(candidate)
        self.assertEqual(core.call_args.kwargs["content"], b"upload text")

    async def test_unbound_mcp_handler_checks_endpoint_policy_before_core(self) -> None:
        adapter = MCPToolAdapter()
        adapter.resolve_canonical_tool_id = mock.AsyncMock(return_value="__ssh_transfer_synthetic__")
        adapter.get_ssh_transfer_definition = mock.AsyncMock(
            return_value=MCPToolDefinition(
                name="ssh_transfer",
                description="transfer",
                input_schema=SSHTransferInput.model_json_schema(),
                tool_config={"id": "__ssh_transfer_synthetic__"},
                execute_fn=mock.AsyncMock(),
                is_synthetic=True,
            )
        )
        adapter._get_configs_for_name_resolution = mock.AsyncMock(return_value=[SSH])
        server = Server("test-unbound-policy")
        _register_handlers(server, adapter)
        handler = server.request_handlers[CallToolRequest]
        request = CallToolRequest(
            params=CallToolRequestParams(name="ssh_transfer", arguments={"source": "inline", "destination": "ssh://docker_1/x", "content": "secret"})
        )
        denied = ContentProtectionError("content_denied", "endpoint-request")
        with (
            mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()),
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock(side_effect=denied)) as authorize,
            mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files") as core,
        ):
            result = await handler(request)
        core.assert_not_called()
        self.assertTrue(getattr(result.root, "isError", False))
        assert authorize.await_args is not None
        self.assertEqual(authorize.await_args.kwargs["tool_id"], "ssh-1")

    async def test_timeout_caps_and_subagent_workspace_scope(self) -> None:
        source = dict(SSH, id="ssh-2", name="Docker 2", timeout_max_seconds=10)
        destination = dict(SSH, timeout_max_seconds=5)
        transfer = build_ssh_transfer_tool([source, destination], subagent_file_scope=["safe"])
        with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=dict(CORE_OK, content=b"ok")) as core:
            rejected = json.loads(await transfer(source="workspace:/safe/../outside", destination="ssh://docker_1/x", content="ignored", workspace_id="w1"))
            self.assertEqual(rejected["status"], "rejected")
            await transfer(source="ssh://docker_2/a", destination="ssh://docker_1/b", timeout=100)
        self.assertEqual(core.call_args.kwargs["timeout"], 5)

    async def test_mcp_legacy_name_collision_uses_legacy_tool_not_synthetic(self) -> None:
        legacy = dict(SSH, id="legacy", name="transfer")
        adapter = MCPToolAdapter()
        with mock.patch.object(adapter, "_get_configs_for_name_resolution", mock.AsyncMock(return_value=[legacy])):
            self.assertIsNone(await adapter.get_ssh_transfer_definition())
            self.assertEqual(await adapter.resolve_canonical_tool_id("ssh_transfer"), "legacy")

    async def test_chat_deny_hides_endpoint_and_userspace_uses_workspace_policy(self) -> None:
        rag = RAGComponents()
        rag._tool_configs = [SSH]  # pyright: ignore[reportPrivateUsage]
        runtime = [SimpleNamespace(name="ssh_docker_1")]
        with mock.patch("ragtime.rag.components.resolve_tool_access", mock.AsyncMock(return_value={"ssh-1": "deny"})) as access:
            denied = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
                runtime, allowed_tool_config_ids=["ssh-1"], user_id="u1"
            )
        self.assertIsNone(denied)
        assert access.await_args is not None
        self.assertEqual(access.await_args.kwargs["surface"], "chat")

        with mock.patch("ragtime.rag.components.resolve_tool_access", mock.AsyncMock(return_value={"ssh-1": "read"})) as access:
            userspace = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
                runtime, allowed_tool_config_ids=["ssh-1"], workspace_id="w1", user_id="u1"
            )
        self.assertIsNotNone(userspace)
        assert access.await_args is not None
        self.assertEqual(access.await_args.kwargs["surface"], "workspace")

    async def test_chat_tool_rechecks_latest_access_at_call_time(self) -> None:
        rag = RAGComponents()
        rag._tool_configs = [SSH]  # pyright: ignore[reportPrivateUsage]
        rag._app_settings = {"max_tool_output_chars": 0}  # pyright: ignore[reportPrivateUsage]
        runtime = [SimpleNamespace(name="ssh_docker_1")]
        access = mock.AsyncMock(side_effect=[{"ssh-1": "read_write"}, {"ssh-1": "deny"}])
        with (
            mock.patch("ragtime.rag.components.resolve_tool_access", access),
            mock.patch("ragtime.rag.components.get_enabled_tool_configs", mock.AsyncMock(return_value=[SSH])),
            mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files") as core,
        ):
            tool = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
                runtime, allowed_tool_config_ids=["ssh-1"], user_id="u1"
            )
            assert tool is not None and tool.coroutine is not None
            result = json.loads(await tool.coroutine(source="inline", destination="ssh://docker_1/x", content="x"))
        self.assertEqual(result["status"], "rejected")
        core.assert_not_called()

    async def test_conversation_read_only_and_user_access_cap_write_at_call_time(self) -> None:
        rag = RAGComponents()
        rag._tool_configs = [SSH]  # pyright: ignore[reportPrivateUsage]
        rag._app_settings = {"max_tool_output_chars": 0}  # pyright: ignore[reportPrivateUsage]
        runtime = [SimpleNamespace(name="ssh_docker_1")]
        rows = [SimpleNamespace(toolConfigId="ssh-1", options={"read_only_enabled": True})]
        db = SimpleNamespace(conversationtooloption=SimpleNamespace(find_many=mock.AsyncMock(return_value=rows)))
        with (
            mock.patch("ragtime.rag.components.get_db", mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.rag.components.resolve_tool_access", mock.AsyncMock(return_value={"ssh-1": "read_write"})),
            mock.patch("ragtime.rag.components.get_enabled_tool_configs", mock.AsyncMock(return_value=[SSH])),
        ):
            tool = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
                runtime, allowed_tool_config_ids=["ssh-1"], user_id="u1", conversation_id="c1"
            )
            assert tool is not None and tool.coroutine is not None
            result = json.loads(await tool.coroutine(source="inline", destination="ssh://docker_1/x", content="x"))
        self.assertEqual(result["status"], "rejected")
        self.assertIn("read-only", result["errors"][0])

        rows[0].options = {"write_access_enabled": True}
        with (
            mock.patch("ragtime.rag.components.get_db", mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.rag.components.resolve_tool_access", mock.AsyncMock(return_value={"ssh-1": "read"})),
            mock.patch("ragtime.rag.components.get_enabled_tool_configs", mock.AsyncMock(return_value=[SSH])),
        ):
            capped = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
                runtime, allowed_tool_config_ids=["ssh-1"], user_id="u1", conversation_id="c1"
            )
            assert capped is not None and capped.coroutine is not None
            capped_result = json.loads(await capped.coroutine(source="inline", destination="ssh://docker_1/x", content="x"))
        self.assertEqual(capped_result["status"], "rejected")
        self.assertIn("read-only", capped_result["errors"][0])

    async def test_blocked_shell_name_never_becomes_transfer_endpoint(self) -> None:
        rag = RAGComponents()
        rag._tool_configs = [SSH]  # pyright: ignore[reportPrivateUsage]
        transfer = await rag._create_ssh_transfer_tool(  # pyright: ignore[reportPrivateUsage]
            [], allowed_tool_config_ids=["ssh-1"]
        )
        self.assertIsNone(transfer)

    async def test_mcp_route_filter_applies_at_list_and_actual_call_time(self) -> None:
        adapter = MCPToolAdapter()
        route = McpRouteFilter(tool_config_ids=["ssh-1"])
        hidden_route = McpRouteFilter(tool_config_ids=[])
        with mock.patch.object(adapter, "_get_configs_for_name_resolution", mock.AsyncMock(return_value=[SSH])):
            allowed = await adapter.get_ssh_transfer_definition(route)
            denied = await adapter.get_ssh_transfer_definition(hidden_route)
            self.assertIsNotNone(allowed)
            self.assertIsNone(denied)
            with mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=CORE_OK) as core:
                hidden = json.loads(await adapter.execute_ssh_transfer({"source": "inline", "destination": "ssh://docker_1/x", "content": "x"}, hidden_route))
            self.assertEqual(hidden["status"], "rejected")
            core.assert_not_called()

    async def test_mcp_workspace_callbacks_only_for_user_principals(self) -> None:
        adapter = MCPToolAdapter()
        adapter.resolve_canonical_tool_id = mock.AsyncMock(return_value="ssh_transfer")
        adapter.execute_ssh_transfer = mock.AsyncMock(return_value=json.dumps(CORE_OK))
        adapter.get_ssh_transfer_definition = mock.AsyncMock(return_value=SimpleNamespace(is_synthetic=True))
        server = Server("test")
        _register_handlers(server, adapter)
        handler = server.request_handlers[CallToolRequest]
        request = CallToolRequest(
            params=CallToolRequestParams(
                name="ssh_transfer",
                arguments={"source": "inline", "destination": "ssh://docker_1/x", "content": "x", "workspace_id": "w1"},
            )
        )
        with mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()):
            with mcp_request_context(SimpleNamespace(user_id="u1", credential_id=None, is_admin=False)):
                await handler(request)
            assert adapter.execute_ssh_transfer.await_args is not None
            user_kwargs = adapter.execute_ssh_transfer.await_args.kwargs
            self.assertIsNotNone(user_kwargs["workspace_read"])
            self.assertIsNotNone(user_kwargs["workspace_write"])

            adapter.execute_ssh_transfer.reset_mock()
            with mcp_request_context(SimpleNamespace(user_id="u1", credential_id="cred-1", is_admin=False)):
                credential_result = await handler(request)
        adapter.execute_ssh_transfer.assert_not_awaited()
        self.assertTrue(getattr(credential_result.root, "isError", False))

    async def test_filtered_mcp_route_calls_legacy_transfer_connection(self) -> None:
        legacy = dict(SSH, name="transfer")
        adapter = MCPToolAdapter()
        server = Server("legacy-transfer-route")
        _register_handlers(server, adapter, McpRouteFilter(tool_config_ids=["ssh-1"]))
        request = CallToolRequest(params=CallToolRequestParams(name="ssh_transfer", arguments={"command": "pwd"}))
        definition = MCPToolDefinition(
            name="ssh_transfer",
            description="An existing SSH shell connection",
            input_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            tool_config=legacy,
            execute_fn=mock.AsyncMock(),
        )
        with (
            mock.patch.object(adapter, "_get_configs_for_name_resolution", mock.AsyncMock(return_value=[legacy])),
            mock.patch.object(adapter, "get_available_tools", mock.AsyncMock(return_value=[definition])),
            mock.patch.object(adapter, "execute_tool", mock.AsyncMock(return_value="legacy shell output")) as execute,
            mock.patch.object(adapter, "execute_ssh_transfer", mock.AsyncMock()) as synthetic,
            mock.patch("ragtime.mcp.server.authorize_external_content", mock.AsyncMock()),
            mcp_request_context(None, "legacy-transfer-route"),
        ):
            response = await server.request_handlers[CallToolRequest](request)
        assert isinstance(response.root, CallToolResult)
        self.assertEqual(getattr(response.root.content[0], "text", None), "legacy shell output")
        execute.assert_awaited_once_with("ssh_transfer", {"command": "pwd"})
        synthetic.assert_not_awaited()

    async def test_legacy_transfer_shell_output_retains_truncation(self) -> None:
        async def shell(command: str) -> str:
            return "legacy shell output\n" * 1000

        legacy = StructuredTool.from_function(coroutine=shell, name="ssh_transfer", description="An existing SSH shell connection")
        wrapped = wrap_tool_with_truncation(legacy, 1000, preserve_output_tool_names=FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES)
        output = await wrapped.ainvoke({"command": "pwd"})
        self.assertLess(len(output), 1000)
        self.assertIn("characters omitted", output)

    def test_mcp_schema_annotations_and_field_descriptions(self) -> None:
        schema = SSHTransferInput.model_json_schema()
        self.assertTrue(all(property_schema.get("description") for property_schema in schema["properties"].values()))
        annotations = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=True)
        self.assertFalse(annotations.readOnlyHint)

    async def test_validation_errors_do_not_echo_payload_or_secrets(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        secret = "TOP-SECRET-PAYLOAD"
        result = await transfer(source="inline", destination="ssh://docker_1/x", content=secret, timeout="bad")
        self.assertNotIn(secret, result)
        self.assertNotIn("input_value", result)
        self.assertEqual(json.loads(result)["status"], "rejected")

    async def test_structured_tool_validation_does_not_echo_secret_input(self) -> None:
        transfer = build_ssh_transfer_tool([SSH])
        tool = StructuredTool.from_function(
            coroutine=transfer,
            name="ssh_transfer",
            description="transfer",
            args_schema=SSHTransferInput,
            handle_validation_error=ssh_transfer_validation_error,
        )
        secret = "STRUCTURED-SECRET"
        result = await tool.ainvoke({"source": "inline", "destination": "ssh://docker_1/x", "content": secret, "timeout": "bad"})
        self.assertNotIn(secret, result)
        self.assertEqual(json.loads(result)["status"], "rejected")

    def test_parser_rejects_credentials_ports_query_fragment_and_normalization_aliases(self) -> None:
        for value in (
            "ssh://user@docker_1/a",
            "ssh://docker_1:22/a",
            "ssh://docker_1/a?x=1",
            "ssh://docker_1/a#frag",
            "ssh://Docker-1/a",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_endpoint(value, {"docker_1": SSH})
        with self.assertRaises(ValueError):
            parse_endpoint("ssh://missing/a", {})
        self.assertEqual(parse_endpoint("workspace:/a", {})[:2], ("workspace", "a"))


if __name__ == "__main__":
    unittest.main()
