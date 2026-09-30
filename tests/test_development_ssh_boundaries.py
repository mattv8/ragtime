import hashlib
import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException

from ragtime.content_protection import service as content_protection_service
from ragtime.mcp import server as mcp_server
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_bootstrap import MAX_MCP_RESPONSE_BYTES, serialize_mcp_payload
from ragtime.userspace.development_service import development_service


class DevelopmentSshBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.workspace_id = "workspace-1"
        self.workspace = SimpleNamespace(id=self.workspace_id, tool_options={})
        self.principal = DevelopmentPrincipal(
            user_id="caller",
            is_admin=False,
            credential_id="credential-1",
            workspace_id=self.workspace_id,
            scopes=frozenset({"read", "write", "exec"}),
        )

    @staticmethod
    def config(tool_id: str = "ssh-1", name: str = "Docker 1", *, enabled: bool = True, allow_write: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            id=tool_id,
            name=name,
            description="Canonical configured description",
            enabled=enabled,
            allow_write=allow_write,
            tool_type=SimpleNamespace(value="ssh_shell"),
            connection_config={"host": "private.example", "user": "tester", "password": "secret", "working_directory": "/srv"},
            timeout_max_seconds=120,
        )

    def service_patches(
        self,
        stack: ExitStack,
        configs: list[SimpleNamespace],
        *,
        caller: dict[str, str] | None = None,
        owner: dict[str, str] | None = None,
    ) -> dict[str, mock.Mock]:
        ids = [str(item.id) for item in configs]
        caller = caller if caller is not None else {tool_id: "read_write" for tool_id in ids}
        owner = owner if owner is not None else {tool_id: "read_write" for tool_id in ids}
        self.workspace.tool_options = {tool_id: {"write_access_enabled": True} for tool_id in ids}
        enforce = mock.AsyncMock(return_value=self.workspace)
        stack.enter_context(mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", new=enforce))
        audit = mock.AsyncMock()
        stack.enter_context(mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", new=audit))
        stack.enter_context(mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()))
        # The transfer adapter performs its own per-endpoint core guards. Default
        # them to allow; guard-specific tests replace this mock and assert calls.
        stack.enter_context(mock.patch.object(content_protection_service, "authorize_content", new=mock.AsyncMock()))
        stack.enter_context(mock.patch("ragtime.userspace.development_ssh.planning_service._selected_tool_ids", new=mock.AsyncMock(return_value=ids)))
        stack.enter_context(mock.patch("ragtime.userspace.development_ssh.resolve_tool_access", new=mock.AsyncMock(return_value=caller)))
        stack.enter_context(
            mock.patch(
                "ragtime.userspace.service.userspace_service._resolve_workspace_owner_tool_access",
                new=mock.AsyncMock(return_value=owner),
            )
        )
        by_id = {str(item.id): item for item in configs}
        stack.enter_context(mock.patch("ragtime.userspace.development_ssh.repository.get_tool_config", new=mock.AsyncMock(side_effect=by_id.get)))

        def runtime(config, *, allow_write):
            return {
                "id": str(config.id),
                "name": str(config.name),
                "tool_type": "ssh_shell",
                "enabled": bool(config.enabled),
                "allow_write": allow_write,
                "timeout_max_seconds": config.timeout_max_seconds,
                "connection_config": dict(config.connection_config),
            }

        runtime_builder = mock.Mock(side_effect=runtime)
        stack.enter_context(mock.patch("ragtime.userspace.development_ssh.runtime_config", new=runtime_builder))
        return {"enforce": enforce, "runtime_builder": runtime_builder, "audit": audit}

    async def test_execute_is_bound_to_outer_workspace_before_ssh_adapter_lookup(self) -> None:
        foreign = DevelopmentPrincipal(
            user_id="caller",
            is_admin=False,
            credential_id="credential-1",
            workspace_id="workspace-elsewhere",
            scopes=frozenset({"exec", "write"}),
        )
        with ExitStack() as stack:
            enforce = mock.AsyncMock()
            stack.enter_context(mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", new=enforce))
            stack.enter_context(mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()))
            lookup = stack.enter_context(mock.patch("ragtime.userspace.development_ssh.authorized_ssh_configs", new=mock.AsyncMock()))
            with self.assertRaises(HTTPException) as raised:
                await development_service.execute(
                    foreign,
                    self.workspace_id,
                    "ssh_execute",
                    {"component_id": "ssh-1", "command": "id"},
                )
        self.assertEqual(raised.exception.status_code, 403)
        enforce.assert_not_awaited()
        lookup.assert_not_awaited()

    async def test_selected_disabled_and_caller_owner_policy_are_rechecked_before_command_io(self) -> None:
        cases: tuple[tuple[list[SimpleNamespace], dict[str, str], dict[str, str], str], ...] = (
            ([], {}, {}, "not selected"),
            ([self.config(enabled=False)], {"ssh-1": "read_write"}, {"ssh-1": "read_write"}, "disabled"),
            ([self.config()], {"ssh-1": "read"}, {"ssh-1": "read_write"}, "caller read-only"),
            ([self.config()], {"ssh-1": "read_write"}, {"ssh-1": "read"}, "owner read-only"),
        )
        for configs, caller, owner, label in cases:
            with self.subTest(label=label), ExitStack() as stack:
                self.service_patches(stack, configs, caller=caller, owner=owner)
                builder = stack.enter_context(mock.patch("ragtime.rag.components.rag.build_primary_runtime_tool_from_config", new=mock.AsyncMock()))
                with self.assertRaises(HTTPException) as raised:
                    await development_service.execute(
                        self.principal,
                        self.workspace_id,
                        "ssh_execute",
                        {"component_id": "ssh-1", "command": "id"},
                    )
                self.assertEqual(raised.exception.status_code, 403)
                builder.assert_not_awaited()

    async def test_exec_only_read_grant_can_download_ssh_file_inline(self) -> None:
        principal = DevelopmentPrincipal(
            user_id="caller",
            is_admin=False,
            credential_id="credential-1",
            workspace_id=self.workspace_id,
            scopes=frozenset({"exec"}),
        )
        config = self.config()
        with ExitStack() as stack:
            patches = self.service_patches(stack, [config], caller={"ssh-1": "read"}, owner={"ssh-1": "read"})
            transfer_core = stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 5, "files_transferred": 1, "errors": [], "skipped": [], "content": b"hello"},
                )
            )
            result = await development_service.execute(
                principal,
                self.workspace_id,
                "ssh_transfer",
                {"source": "ssh://docker_1/readme.txt", "destination": "inline", "encoding": "text"},
            )
        self.assertEqual(result["content"], "hello")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(patches["runtime_builder"].call_args.kwargs["allow_write"])
        transfer_core.assert_called_once()

    async def test_workspace_download_forwards_caller_cas_hash_directly_to_upsert(self) -> None:
        expected = hashlib.sha256(b"old").hexdigest()
        config = self.config()
        with ExitStack() as stack:
            self.service_patches(stack, [config])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": [], "content": b"new"},
                )
            )
            upsert = stack.enter_context(
                mock.patch(
                    "ragtime.userspace.service.userspace_service.upsert_workspace_file",
                    new=mock.AsyncMock(return_value=SimpleNamespace(path="target.txt")),
                )
            )
            result = await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_transfer",
                {
                    "source": "ssh://docker_1/source.txt",
                    "destination": "workspace:/target.txt",
                    "expected_content_hash": expected,
                },
            )
        self.assertEqual(result["status"], "ok")
        call = upsert.await_args
        assert call is not None
        self.assertEqual(call.args[:2], (self.workspace_id, "target.txt"))
        self.assertEqual(call.kwargs["expected_content_hash"], expected)
        self.assertTrue(call.kwargs["require_content_hash"])

    async def test_workspace_download_preserves_cas_conflict_as_http_409(self) -> None:
        expected = hashlib.sha256(b"stale").hexdigest()
        conflict = HTTPException(
            status_code=409,
            detail={"code": "content_hash_conflict", "expected_hash": expected, "actual_hash": hashlib.sha256(b"current").hexdigest()},
        )
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": [], "content": b"new"},
                )
            )
            stack.enter_context(mock.patch("ragtime.userspace.service.userspace_service.upsert_workspace_file", new=mock.AsyncMock(side_effect=conflict)))
            with self.assertRaises(HTTPException) as raised:
                await development_service.execute(
                    self.principal,
                    self.workspace_id,
                    "ssh_transfer",
                    {
                        "source": "ssh://docker_1/source.txt",
                        "destination": "workspace:/target.txt",
                        "expected_content_hash": expected,
                    },
                )
        self.assertEqual(raised.exception.status_code, 409)
        detail = cast(dict[str, object], raised.exception.detail)
        self.assertEqual(detail["code"], "content_hash_conflict")

    async def test_mcp_context_is_preserved_for_both_endpoint_guards(self) -> None:
        source = self.config("ssh-source", "Source Host")
        destination = self.config("ssh-destination", "Destination Host")
        guard = mock.AsyncMock()
        with ExitStack() as stack:
            self.service_patches(stack, [source, destination])
            stack.enter_context(mock.patch("ragtime.mcp.server.authorize_external_content", new=mock.AsyncMock()))
            stack.enter_context(mock.patch.object(content_protection_service, "authorize_content", new=guard))
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 1, "files_transferred": 1, "errors": [], "skipped": []},
                )
            )
            with mcp_server.mcp_request_context(self.principal, route_id="trusted-route"):
                bound_context = content_protection_service.current_context()
                assert bound_context is not None
                result = await mcp_server._execute_development_tool(
                    "workspace_development",
                    {
                        "workspace_id": self.workspace_id,
                        "operation": "ssh_transfer",
                        "arguments": {
                            "source": "ssh://source_host/a",
                            "destination": "ssh://destination_host/b",
                            "overwrite": True,
                        },
                    },
                )
        self.assertFalse(result.isError)
        endpoint_calls = [call for call in guard.await_args_list if call.kwargs.get("tool_id") in {"ssh-source", "ssh-destination"}]
        self.assertEqual(
            {(call.kwargs["tool_id"], call.kwargs["direction"]) for call in endpoint_calls},
            {
                ("ssh-source", "proposed_operation"),
                ("ssh-destination", "proposed_operation"),
                ("ssh-source", "tool_result"),
                ("ssh-destination", "tool_result"),
            },
        )
        self.assertTrue(all(call.kwargs["context"] is bound_context for call in endpoint_calls))
        self.assertEqual(bound_context.surface, "mcp")
        self.assertEqual(bound_context.mcp_route, "trusted-route")

    async def test_ssh_execute_uses_component_specific_proposed_and_result_guards(self) -> None:
        tool = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=json.dumps({"status": "ok", "exit_code": 0, "stdout": "caller", "stderr": ""})))
        guard = mock.AsyncMock()
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(mock.patch("ragtime.rag.components.rag.build_primary_runtime_tool_from_config", new=mock.AsyncMock(return_value=tool)))
            stack.enter_context(mock.patch.object(content_protection_service, "authorize_content", new=guard))
            with mcp_server.mcp_request_context(self.principal, route_id="trusted-route"):
                await development_service.execute(
                    self.principal,
                    self.workspace_id,
                    "ssh_execute",
                    {"component_id": "ssh-1", "command": "id"},
                )
        component_calls = [call for call in guard.await_args_list if call.kwargs.get("tool_id") == "ssh-1"]
        self.assertEqual(
            [(call.kwargs["direction"], call.kwargs["operation"]) for call in component_calls],
            [("proposed_operation", "ssh_execute"), ("tool_result", "ssh_execute")],
        )

    async def test_response_budget_does_not_change_context_or_file_read_results(self) -> None:
        large = {"content": "🦀" * (MAX_MCP_RESPONSE_BYTES // 2)}
        self.assertGreater(len(serialize_mcp_payload(large, limit=None).encode("utf-8")), MAX_MCP_RESPONSE_BYTES)
        for operation in ("context", "file_read"):
            with (
                self.subTest(operation=operation),
                mock.patch.object(development_service, "_execute_unprotected", new=mock.AsyncMock(return_value=large)),
                mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()),
            ):
                result = await development_service.execute(self.principal, self.workspace_id, operation, {})
            self.assertEqual(result, large)

    async def test_command_result_budget_uses_serialized_utf8_bytes_and_remains_valid_json(self) -> None:
        tool = SimpleNamespace(
            ainvoke=mock.AsyncMock(
                return_value=json.dumps({"status": "ok", "exit_code": 0, "stdout": '"\\\n🦀' * MAX_MCP_RESPONSE_BYTES, "stderr": ""}, ensure_ascii=False)
            )
        )
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(mock.patch("ragtime.rag.components.rag.build_primary_runtime_tool_from_config", new=mock.AsyncMock(return_value=tool)))
            result = await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_execute",
                {"component_id": "ssh-1", "command": "emit unicode"},
            )
        serialized = serialize_mcp_payload(result, limit=None)
        self.assertLessEqual(len(serialized.encode("utf-8")), MAX_MCP_RESPONSE_BYTES)
        self.assertEqual(json.loads(serialized), result)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["execution_status"], "completed")
        self.assertTrue(result["stdout"])
        self.assertTrue(('"\\\n🦀' * MAX_MCP_RESPONSE_BYTES).startswith(result["stdout"]))

    async def test_metadata_does_not_advertise_exec_without_exec_scope(self) -> None:
        principal = DevelopmentPrincipal(
            user_id="caller",
            is_admin=False,
            credential_id="credential-1",
            workspace_id=self.workspace_id,
            scopes=frozenset({"read", "write"}),
        )
        config = self.config()
        fake_db = SimpleNamespace(workspaceindexgrant=SimpleNamespace(find_many=mock.AsyncMock(return_value=[])))
        with ExitStack() as stack:
            self.service_patches(stack, [config])
            stack.enter_context(
                mock.patch("ragtime.userspace.development_service.planning_service._selected_tool_ids", new=mock.AsyncMock(return_value=["ssh-1"]))
            )
            stack.enter_context(
                mock.patch("ragtime.userspace.development_service.resolve_tool_access", new=mock.AsyncMock(return_value={"ssh-1": "read_write"}))
            )
            stack.enter_context(mock.patch("ragtime.userspace.development_service.repository.get_tool_config", new=mock.AsyncMock(return_value=config)))
            stack.enter_context(mock.patch("ragtime.userspace.development_service.get_db", new=mock.AsyncMock(return_value=fake_db)))
            resources = await development_service.execute(principal, self.workspace_id, "resources", {})
        resource = resources["tools"][0]
        metadata = resource["ssh_operations"]
        self.assertFalse(metadata["ssh_execute"]["supported"])
        self.assertFalse(metadata["ssh_transfer"]["supported"])
        self.assertIn("canonical_description", resource)
        self.assertIn("execute_component", resource)
        self.assertIn("execution_lanes", resource)

    async def test_actual_command_builder_uses_default_timeout_when_omitted(self) -> None:
        ssh_result = SimpleNamespace(success=True, exit_code=0, stdout="ok", stderr="")
        with ExitStack() as stack:
            patches = self.service_patches(stack, [self.config()])
            execute_ssh = stack.enter_context(mock.patch("ragtime.rag.components.execute_ssh_command", return_value=ssh_result))
            result = await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_execute",
                {"component_id": "ssh-1", "command": "id"},
            )
        self.assertEqual(result["status"], "completed")
        ssh_config = execute_ssh.call_args.args[0]
        self.assertEqual(ssh_config.timeout, 120)
        payloads = [call.kwargs["payload"] for call in patches["audit"].await_args_list]
        command_audit = next(payload for payload in payloads if payload.get("component_id") == "ssh-1")
        self.assertEqual(command_audit["operation"], "ssh_execute")
        self.assertNotIn("command", command_audit)

    async def test_command_failure_does_not_release_transport_secrets(self) -> None:
        secret = "private.example password=secret key=/private/key"
        tool = SimpleNamespace(
            ainvoke=mock.AsyncMock(return_value=json.dumps({"status": "command_failed", "exit_code": -1, "error": secret, "stderr": "requested stderr"}))
        )
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(mock.patch("ragtime.rag.components.rag.build_primary_runtime_tool_from_config", new=mock.AsyncMock(return_value=tool)))
            result = await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_execute",
                {"component_id": "ssh-1", "command": "false"},
            )
        self.assertNotIn("private.example", json.dumps(result))
        self.assertNotIn("password=secret", json.dumps(result))
        self.assertNotIn("/private/key", json.dumps(result))
        self.assertEqual(result["stderr"], "requested stderr")

    async def test_transfer_conditional_arguments_fail_before_config_or_network_lookup(self) -> None:
        invalid = (
            {"source": "inline", "destination": "ssh://docker_1/a"},
            {"source": "ssh://docker_1/a", "destination": "inline", "content": "unexpected"},
            {"source": "ssh://docker_1/a", "destination": "workspace:/a", "expected_content_hash": None, "recursive": True},
            {"source": "workspace:/a", "destination": "workspace:/b", "expected_content_hash": None},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), ExitStack() as stack:
                enforce = mock.AsyncMock(return_value=self.workspace)
                stack.enter_context(mock.patch("ragtime.userspace.development_service.userspace_service.enforce_workspace_role", new=enforce))
                stack.enter_context(mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", new=mock.AsyncMock()))
                stack.enter_context(mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()))
                lookup = stack.enter_context(mock.patch("ragtime.userspace.development_ssh.authorized_ssh_configs", new=mock.AsyncMock()))
                with self.assertRaises(HTTPException) as raised:
                    await development_service.execute(self.principal, self.workspace_id, "ssh_transfer", arguments)
                self.assertEqual(raised.exception.status_code, 422)
                lookup.assert_not_awaited()

    async def test_uppercase_workspace_hash_is_normalized_before_upsert(self) -> None:
        expected_upper = hashlib.sha256(b"old").hexdigest().upper()
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": [], "content": b"new"},
                )
            )
            upsert = stack.enter_context(
                mock.patch(
                    "ragtime.userspace.service.userspace_service.upsert_workspace_file",
                    new=mock.AsyncMock(return_value=SimpleNamespace(path="target.txt")),
                )
            )
            await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_transfer",
                {
                    "source": "ssh://docker_1/source.txt",
                    "destination": "workspace:/target.txt",
                    "expected_content_hash": expected_upper,
                },
            )
        upsert_call = upsert.await_args
        assert upsert_call is not None
        self.assertEqual(upsert_call.kwargs["expected_content_hash"], expected_upper.lower())

    async def test_resolved_endpoint_ids_are_audited_without_paths_or_commands(self) -> None:
        source = self.config("ssh-source", "Source Host")
        destination = self.config("ssh-destination", "Destination Host")
        with ExitStack() as stack:
            patches = self.service_patches(stack, [source, destination])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 1, "files_transferred": 1, "errors": [], "skipped": []},
                )
            )
            await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_transfer",
                {"source": "ssh://source_host/private/a", "destination": "ssh://destination_host/private/b", "overwrite": True},
            )
        payloads = [call.kwargs["payload"] for call in patches["audit"].await_args_list]
        endpoint_audit = next(payload for payload in payloads if "source_component_id" in payload)
        self.assertEqual(endpoint_audit["source_component_id"], "ssh-source")
        self.assertEqual(endpoint_audit["destination_component_id"], "ssh-destination")
        serialized = json.dumps(endpoint_audit)
        self.assertNotIn("/private/a", serialized)
        self.assertNotIn("/private/b", serialized)

    async def test_metadata_drops_owner_denied_and_disables_colliding_aliases(self) -> None:
        from ragtime.userspace.development_ssh import resource_metadata

        first = self.config("ssh-1", "Duplicate Host")
        second = self.config("ssh-2", "Duplicate-Host")
        caller = {"ssh-1": "read_write", "ssh-2": "read_write"}
        with mock.patch(
            "ragtime.userspace.service.userspace_service._resolve_workspace_owner_tool_access",
            new=mock.AsyncMock(return_value={"ssh-1": "read_write", "ssh-2": "read_write"}),
        ):
            metadata = await resource_metadata(self.principal, self.workspace, configs=[first, second], caller_access=caller)
        self.assertEqual(len(metadata), 2)
        self.assertTrue(all(item["ssh_endpoint_alias"] is None for item in metadata))
        self.assertTrue(all(not item["ssh_operations"]["ssh_transfer"]["supported"] for item in metadata))

        with mock.patch(
            "ragtime.userspace.service.userspace_service._resolve_workspace_owner_tool_access",
            new=mock.AsyncMock(return_value={"ssh-1": "deny"}),
        ):
            denied = await resource_metadata(self.principal, self.workspace, configs=[first], caller_access={"ssh-1": "read_write"})
        self.assertEqual(denied, [])

    async def test_workspace_create_only_forwards_explicit_null(self) -> None:
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": [], "content": b"new"},
                )
            )
            upsert = stack.enter_context(
                mock.patch(
                    "ragtime.userspace.service.userspace_service.upsert_workspace_file",
                    new=mock.AsyncMock(return_value=SimpleNamespace(path="created.txt")),
                )
            )
            await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_transfer",
                {
                    "source": "ssh://docker_1/source.txt",
                    "destination": "workspace:/created.txt",
                    "expected_content_hash": None,
                },
            )
        upsert_call = upsert.await_args
        assert upsert_call is not None
        self.assertIsNone(upsert_call.kwargs["expected_content_hash"])
        self.assertTrue(upsert_call.kwargs["require_content_hash"])

    async def test_stale_dashboard_cas_rejects_before_upsert_entrypoint_side_effects(self) -> None:
        expected = hashlib.sha256(b"stale").hexdigest()
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={"status": "ok", "bytes_transferred": 3, "files_transferred": 1, "errors": [], "skipped": [], "content": b"new"},
                )
            )
            stack.enter_context(
                mock.patch(
                    "ragtime.userspace.service.userspace_service.get_workspace_file",
                    new=mock.AsyncMock(return_value=SimpleNamespace(content="current")),
                )
            )
            upsert = stack.enter_context(mock.patch("ragtime.userspace.service.userspace_service.upsert_workspace_file", new=mock.AsyncMock()))
            with self.assertRaises(HTTPException) as raised:
                await development_service.execute(
                    self.principal,
                    self.workspace_id,
                    "ssh_transfer",
                    {
                        "source": "ssh://docker_1/source.txt",
                        "destination": "workspace:/dashboard/main.ts",
                        "expected_content_hash": expected,
                    },
                )
        self.assertEqual(raised.exception.status_code, 409)
        upsert.assert_not_awaited()

    async def test_oversized_inline_download_withholds_content_but_preserves_counts(self) -> None:
        payload = b"x" * (MAX_MCP_RESPONSE_BYTES + 1)
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(
                mock.patch(
                    "ragtime.core.ssh_transfer.transfer_ssh_files",
                    return_value={
                        "status": "ok",
                        "bytes_transferred": len(payload),
                        "files_transferred": 1,
                        "errors": [],
                        "skipped": [],
                        "content": payload,
                    },
                )
            )
            result = await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_transfer",
                {"source": "ssh://docker_1/source.txt", "destination": "inline", "encoding": "text"},
            )
        self.assertEqual(result["status"], "response_too_large")
        self.assertEqual(result["transfer_status"], "ok")
        self.assertEqual(result["bytes_transferred"], len(payload))
        self.assertEqual(result["files_transferred"], 1)
        self.assertEqual(result["execution_status"], "completed_response_withheld")
        self.assertNotIn("content", result)

    async def test_http_command_guards_use_bound_development_principal_context(self) -> None:
        tool = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=json.dumps({"status": "completed", "exit_code": 0})))
        guard = mock.AsyncMock()
        with ExitStack() as stack:
            self.service_patches(stack, [self.config()])
            stack.enter_context(mock.patch("ragtime.rag.components.rag.build_primary_runtime_tool_from_config", new=mock.AsyncMock(return_value=tool)))
            stack.enter_context(mock.patch.object(content_protection_service, "authorize_content", new=guard))
            await development_service.execute(
                self.principal,
                self.workspace_id,
                "ssh_execute",
                {"component_id": "ssh-1", "command": "id"},
            )
        component_calls = [call for call in guard.await_args_list if call.kwargs.get("tool_id") == "ssh-1"]
        self.assertEqual(len(component_calls), 2)
        contexts = [call.kwargs["context"] for call in component_calls]
        self.assertTrue(all(context.user_id == "caller" for context in contexts))
        self.assertTrue(all(context.surface == "development" for context in contexts))
        self.assertTrue(all(context.resource_id == self.workspace_id for context in contexts))


if __name__ == "__main__":
    unittest.main()
