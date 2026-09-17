import json
import sys
import types
import unittest
from unittest import mock

from fastapi import HTTPException

_inserted_fake_indexer_service = False
if "ragtime.indexer.service" not in sys.modules:
    fake_indexer_service = types.ModuleType("ragtime.indexer.service")
    setattr(fake_indexer_service, "IndexerService", object)
    sys.modules["ragtime.indexer.service"] = fake_indexer_service
    _inserted_fake_indexer_service = True

from ragtime.rag.components import RAGComponents

if _inserted_fake_indexer_service:
    inserted_module = sys.modules.get("ragtime.indexer.service")
    if "ragtime.indexer.service" in sys.modules:
        del sys.modules["ragtime.indexer.service"]
    indexer_package = sys.modules.get("ragtime.indexer")
    if indexer_package is not None and getattr(indexer_package, "service", None) is inserted_module:
        delattr(indexer_package, "service")

from ragtime.userspace.runtime_service import userspace_runtime_service


class UserSpaceBuilderToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.rag = RAGComponents()
        self.settings = mock.patch(
            "ragtime.rag.components.get_app_settings",
            new=mock.AsyncMock(
                return_value={
                    "userspace_exec_timeout_default_seconds": 180,
                    "userspace_exec_timeout_max_seconds": 1800,
                }
            ),
        )
        self.global_tools = mock.patch.object(userspace_runtime_service, "get_global_mcp_tools", new=mock.AsyncMock(return_value=[]))
        self.workspace_tools = mock.patch.object(userspace_runtime_service, "list_workspace_mcp_tools", new=mock.AsyncMock(return_value=[]))
        self.settings.start()
        self.global_tools.start()
        self.workspace_tools.start()
        tools = await self.rag._create_userspace_file_tools("workspace", "user")
        self.tools = {tool.name: tool for tool in tools}

    async def asyncTearDown(self) -> None:
        self.workspace_tools.stop()
        self.global_tools.stop()
        self.settings.stop()

    async def test_terminal_schema_uses_workspace_execution_budget(self) -> None:
        args_schema = self.tools["run_terminal_command"].args_schema
        assert args_schema is not None
        assert not isinstance(args_schema, dict)
        schema = args_schema.model_json_schema()
        timeout = schema["properties"]["timeout_seconds"]

        self.assertIsNone(timeout["default"])
        self.assertIn("180s", timeout["description"])
        self.assertIn("1800s", timeout["description"])
        self.assertEqual(timeout["maximum"], 1800)
        self.assertEqual(timeout["anyOf"], [{"type": "integer"}, {"type": "null"}])
        self.assertIn("default 180s, max 1800s", self.tools["run_terminal_command"].description)

    async def test_terminal_omission_reaches_service_as_none(self) -> None:
        execute = mock.AsyncMock(return_value={"exit_code": 0, "stdout": "ok"})
        with mock.patch.object(userspace_runtime_service, "exec_workspace_command", execute):
            result = json.loads(await self.tools["run_terminal_command"].ainvoke({"command": "pwd"}))

        self.assertEqual(result["status"], "completed")
        call = execute.await_args
        assert call is not None
        self.assertIsNone(call.kwargs["timeout_seconds"])

    async def test_terminal_schema_rejects_boolean_without_coercing_it_to_one(self) -> None:
        execute = mock.AsyncMock(side_effect=HTTPException(status_code=400, detail="timeout_seconds must be an integer"))
        with mock.patch.object(userspace_runtime_service, "exec_workspace_command", execute):
            result = await self.tools["run_terminal_command"].ainvoke({"command": "pwd", "timeout_seconds": True})

        self.assertEqual(result, "Tool input validation error")
        execute.assert_not_awaited()

    async def test_restart_and_status_use_runtime_service_and_report_progress(self) -> None:
        request_restart = mock.AsyncMock(return_value={"id": "ledger-id", "operation_id": "provider-id", "state": "running"})
        get_operation = mock.AsyncMock(return_value={"id": "ledger-id", "operation_id": "provider-id", "state": "completed"})
        with (
            mock.patch.object(userspace_runtime_service, "request_app_restart", request_restart),
            mock.patch.object(userspace_runtime_service, "get_app_runtime_operation", get_operation),
        ):
            restart_result = json.loads(await self.tools["restart_app_runtime"].ainvoke({"idempotency_key": "restart-01", "reason": "apply config"}))
            status_result = json.loads(await self.tools["get_app_runtime_status"].ainvoke({"operation_id": "ledger-id"}))

        request_restart.assert_awaited_once_with("workspace", "user", "restart-01", "apply config")
        get_operation.assert_awaited_once_with("workspace", "user", "ledger-id")
        self.assertEqual(restart_result["status"], "running")
        self.assertEqual(restart_result["next_best_tool"], "get_app_runtime_status")
        self.assertEqual(status_result["status"], "ready")
        self.assertTrue(status_result["ready"])

    async def test_cross_workspace_restart_uses_write_target_resolution(self) -> None:
        resolve_target = mock.AsyncMock(return_value=("target", "target-user"))
        request_restart = mock.AsyncMock(return_value={"id": "ledger-id", "state": "accepted"})
        with (
            mock.patch("ragtime.rag.components.userspace_service.resolve_cross_workspace_target", resolve_target),
            mock.patch.object(userspace_runtime_service, "request_app_restart", request_restart),
        ):
            result = json.loads(await self.tools["restart_app_runtime"].ainvoke({"workspace_id": "target", "idempotency_key": "restart-02"}))

        self.assertEqual(result["status"], "accepted")
        resolve_target_args = resolve_target.await_args
        assert resolve_target_args is not None
        self.assertEqual(resolve_target_args.kwargs["action"], "write")
        request_restart.assert_awaited_once_with("target", "target-user", "restart-02", "")

    async def test_cross_workspace_restart_rejects_read_only_grant_before_dispatch(self) -> None:
        tools = {
            tool.name: tool
            for tool in await self.rag._create_userspace_file_tools(
                "workspace",
                "user",
                accessible_workspace_modes={"target": "read"},
            )
        }
        request_restart = mock.AsyncMock()
        with mock.patch.object(userspace_runtime_service, "request_app_restart", request_restart):
            with self.assertRaisesRegex(ValueError, "read-only access"):
                await tools["restart_app_runtime"].ainvoke({"workspace_id": "target", "idempotency_key": "restart-03"})

        request_restart.assert_not_awaited()

    async def test_cross_workspace_restart_confirms_editor_acl_before_dispatch(self) -> None:
        enforce_access = mock.AsyncMock()
        audit = mock.AsyncMock()
        request_restart = mock.AsyncMock(return_value={"id": "ledger-id", "state": "accepted"})
        tools = {
            tool.name: tool
            for tool in await self.rag._create_userspace_file_tools(
                "workspace",
                "user",
                accessible_workspace_modes={"target": "read_write"},
            )
        }
        with (
            mock.patch("ragtime.rag.components.userspace_service._enforce_workspace_access", enforce_access),
            mock.patch("ragtime.rag.components.userspace_service._record_runtime_audit_event", audit),
            mock.patch.object(userspace_runtime_service, "request_app_restart", request_restart),
        ):
            await tools["restart_app_runtime"].ainvoke({"workspace_id": "target", "idempotency_key": "restart-04"})

        enforce_access_args = enforce_access.await_args
        assert enforce_access_args is not None
        self.assertEqual(enforce_access_args.kwargs["required_role"], "editor")
        request_restart.assert_awaited_once_with("target", "user", "restart-04", "")

    async def test_subagent_cannot_restart_parent_runtime(self) -> None:
        tools = {
            tool.name: tool
            for tool in await self.rag._create_userspace_file_tools(
                "workspace",
                "user",
                subagent_file_scope=["dashboard"],
                workspace_context={"subagent_depth": 1},
            )
        }

        result = await tools["restart_app_runtime"].ainvoke({"idempotency_key": "restart-05"})

        self.assertIn("Subagent runtime restart rejected", result)


if __name__ == "__main__":
    unittest.main()
