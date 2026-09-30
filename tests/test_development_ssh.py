import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import _OPERATIONS, development_service
from ragtime.userspace.development_ssh import effective_write, transfer


class DevelopmentSSHTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.principal = DevelopmentPrincipal(user_id="caller", is_admin=False, scopes=frozenset({"exec", "write"}))
        self.workspace = SimpleNamespace(tool_options={"ssh-1": {"write_access_enabled": True}})
        self.config = SimpleNamespace(id="ssh-1", allow_write=False, name="Docker 1")

    def test_registry_uses_exec_and_documents_conditional_write(self) -> None:
        contracts = {name: (description, scope) for name, description, scope, _schema in _OPERATIONS}
        self.assertEqual(contracts["ssh_execute"][1], "exec")
        self.assertIn("write scope", contracts["ssh_execute"][0])
        self.assertEqual(contracts["ssh_transfer"][1], "exec")

    def test_effective_write_requires_opt_in_caller_and_owner_rw(self) -> None:
        caller = {"ssh-1": "read_write"}
        owner = {"ssh-1": "read_write"}
        self.assertTrue(effective_write(self.workspace, self.config, caller, owner))
        self.assertFalse(effective_write(self.workspace, self.config, {"ssh-1": "read"}, owner))
        self.assertFalse(effective_write(self.workspace, self.config, caller, {"ssh-1": "read"}))
        self.workspace.tool_options = {}
        self.assertFalse(effective_write(self.workspace, self.config, caller, owner))

    async def test_execute_requires_write_scope_before_any_connection_lookup(self) -> None:
        readonly = DevelopmentPrincipal(user_id="caller", is_admin=False, scopes=frozenset({"exec"}))
        with mock.patch("ragtime.userspace.development_ssh.authorized_ssh_configs", new=mock.AsyncMock()) as configs:
            with self.assertRaises(HTTPException) as raised:
                await __import__("ragtime.userspace.development_ssh", fromlist=["execute"]).execute(
                    readonly, "ws", self.workspace, {"component_id": "ssh-1", "command": "id"}
                )
        self.assertEqual(raised.exception.status_code, 403)
        configs.assert_not_awaited()

    async def test_workspace_transfer_requires_explicit_cas_before_network(self) -> None:
        with mock.patch("ragtime.userspace.development_ssh.authorized_ssh_configs", new=mock.AsyncMock()) as configs:
            with self.assertRaises(HTTPException) as raised:
                await transfer(self.principal, "ws", self.workspace, {"source": "ssh://docker_1/a", "destination": "workspace:/a"})
        self.assertEqual(raised.exception.status_code, 422)
        configs.assert_not_awaited()

    async def test_workspace_transfer_rejects_overwrite_and_bad_digest(self) -> None:
        for arguments in (
            {"source": "ssh://docker_1/a", "destination": "workspace:/a", "expected_content_hash": None, "overwrite": False},
            {"source": "ssh://docker_1/a", "destination": "workspace:/a", "expected_content_hash": "not-a-digest"},
        ):
            with self.assertRaises(HTTPException) as raised:
                await transfer(self.principal, "ws", self.workspace, arguments)
            self.assertEqual(raised.exception.status_code, 422)

    async def test_ordinary_development_results_keep_existing_output_semantics(self) -> None:
        principal = DevelopmentPrincipal(user_id="caller", is_admin=False, scopes=frozenset({"read"}))
        huge = {"output": "🦀" * 30_000}
        with (
            mock.patch.object(development_service, "_execute_unprotected", new=mock.AsyncMock(return_value=huge)),
            mock.patch("ragtime.userspace.development_service.authorize_external_content", new=mock.AsyncMock()),
        ):
            result = await development_service._execute_with_context(principal, "ws", "context", {})
        self.assertEqual(result, huge)

    async def test_oversized_partial_copy_never_reports_not_started(self) -> None:
        runtime = {
            "id": "ssh-1",
            "name": "Docker 1",
            "tool_type": "ssh_shell",
            "enabled": True,
            "allow_write": True,
            "connection_config": {"host": "unused", "user": "unused", "password": "fixture"},
        }
        partial = {
            "status": "transfer_failed",
            "bytes_transferred": 9,
            "files_transferred": 2,
            "errors": [{"message": "existing target " * 100}] * 30,
            "skipped": [],
        }
        with (
            mock.patch(
                "ragtime.userspace.development_ssh.authorized_ssh_configs",
                mock.AsyncMock(
                    return_value=(
                        [self.config],
                        {"ssh-1": "read_write"},
                        {"ssh-1": "read_write"},
                    )
                ),
            ),
            mock.patch("ragtime.userspace.development_ssh.runtime_config", return_value=runtime),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch("ragtime.tools.ssh_transfer.content_protection_service.authorize_content", mock.AsyncMock()),
            mock.patch("ragtime.core.ssh_transfer.transfer_ssh_files", return_value=partial),
        ):
            result = await transfer(
                self.principal,
                "ws",
                self.workspace,
                {
                    "source": "ssh://docker_1/source",
                    "destination": "ssh://docker_1/destination",
                    "recursive": True,
                },
            )
        self.assertEqual(result["status"], "response_too_large")
        self.assertEqual(result["transfer_status"], "transfer_failed")
        self.assertEqual(result["files_transferred"], 2)
        self.assertEqual(result["bytes_transferred"], 9)
        self.assertEqual(result["execution_status"], "partially_completed_response_withheld")
