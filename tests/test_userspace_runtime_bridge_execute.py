from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from rag_prompts_stub import install_fake_rag_prompts, remove_fake_rag_prompts

inserted_fake_rag_prompts = install_fake_rag_prompts()

from ragtime.userspace import service as userspace_service_module
from ragtime.userspace.models import (
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    UserSpacePreviewDiagnosticEvent,
    UserSpaceWorkspace,
    WorkspaceToolOptionState,
)
from ragtime.userspace.service import UserSpaceService

remove_fake_rag_prompts(inserted_fake_rag_prompts)


def _make_workspace() -> UserSpaceWorkspace:
    now = datetime.now(timezone.utc)
    return UserSpaceWorkspace(
        id="workspace-1",
        name="WS",
        owner_user_id="user-1",
        tool_selection_mode="custom",
        selected_tool_ids=["tool-1"],
        tool_options={},
        created_at=now,
        updated_at=now,
    )


def _make_tool_config(
    tool_id: str = "tool-1",
    tool_type: str = "postgres",
    *,
    allow_write: bool = False,
    connection_config: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tool_id,
        name=tool_id,
        enabled=True,
        tool_type=SimpleNamespace(value=tool_type),
        connection_config=connection_config or {},
        timeout_max_seconds=300,
        max_results=100,
        allow_write=allow_write,
    )


def _make_runtime_bridge_tool_service(*, write_enabled: bool = False) -> _RuntimeBridgeWorkspaceService:
    workspace = _make_workspace()
    if write_enabled:
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
    return _RuntimeBridgeWorkspaceService(workspace)


async def _execute_runtime_bridge_tool_request(
    *,
    tool_type: str,
    output: object,
    request_input: str | dict[str, object],
    write_enabled: bool = False,
) -> tuple[_RuntimeBridgeWorkspaceService, ExecuteComponentResponse, SimpleNamespace, mock.AsyncMock]:
    service = _make_runtime_bridge_tool_service(write_enabled=write_enabled)
    tool = SimpleNamespace(ainvoke=mock.AsyncMock(return_value=output))

    with (
        mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(tool_type=tool_type, allow_write=True)),
        ),
        mock.patch(
            "ragtime.rag.components.rag.build_primary_runtime_tool_from_config",
            mock.AsyncMock(return_value=tool),
        ) as build_tool,
    ):
        response = await service.execute_component_from_runtime_bridge(
            "workspace-1",
            ExecuteComponentRequest(component_id="tool-1", request=request_input),
            session_id="sess-1",
        )

    return service, response, tool, build_tool


class _RuntimeBridgeRecordingMixin:
    def _init_runtime_bridge_recording(self) -> None:
        self.proofs_recorded: list[tuple[str, str, int, str]] = []
        self.diagnostic_events: list[tuple[str, list[UserSpacePreviewDiagnosticEvent]]] = []
        self.bridge_audit_calls: list[dict[str, object | None]] = []

    def record_execution_proof(self, workspace_id: str, component_id: str, row_count: int, query: str) -> None:  # type: ignore[override]
        self.proofs_recorded.append((workspace_id, component_id, row_count, query))

    def clear_live_data_execution_warning(self, workspace_id: str) -> None:  # type: ignore[override]
        pass

    async def record_workspace_preview_diagnostic_events(self, workspace_id: str, events: list[UserSpacePreviewDiagnosticEvent]) -> int:  # type: ignore[override]
        self.diagnostic_events.append((workspace_id, events))
        return len(events)

    async def _record_runtime_bridge_audit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.bridge_audit_calls.append(kwargs)


class _RuntimeBridgeSuccessService(_RuntimeBridgeRecordingMixin, UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.bridge_workspace_calls: list[tuple[str, ExecuteComponentRequest, dict[str, object]]] = []
        self._init_runtime_bridge_recording()

    async def _load_workspace_for_component_execution(self, workspace_id: str, user_id: str | None = None) -> UserSpaceWorkspace:  # type: ignore[override]
        return _make_workspace()

    async def _resolve_effective_workspace_tool_ids(self, workspace: UserSpaceWorkspace) -> list[str]:  # type: ignore[override]
        return ["tool-1"]

    async def _execute_component_for_workspace(self, workspace: UserSpaceWorkspace, request: ExecuteComponentRequest, **kwargs):  # type: ignore[no-untyped-def,override]
        self.bridge_workspace_calls.append((workspace.id, request, kwargs))
        return await super()._execute_component_for_workspace(workspace, request, **kwargs)

    async def _resolve_component_execution_config_for_tool_ids(self, selected_tool_ids, component_id, **kwargs):  # type: ignore[no-untyped-def,override]
        _ = selected_tool_ids, component_id, kwargs
        return SimpleNamespace(
            resolved_id="tool-1",
            tool_type="postgres",
            conn_config={},
            tool_config=_make_tool_config(),
            effective_allow_write=False,
            access_mode="read_only",
        )

    async def _execute_component_for_selected_tool_ids(self, **kwargs):  # type: ignore[no-untyped-def]
        return (
            ExecuteComponentResponse(
                component_id=str(kwargs.get("component_id") or "tool-1"),
                rows=[{"id": 1}],
                columns=["id"],
                row_count=1,
            ),
            "select 1",
        )


class _RuntimeBridgeWorkspaceService(_RuntimeBridgeRecordingMixin, UserSpaceService):
    def __init__(self, workspace: UserSpaceWorkspace | None = None) -> None:
        super().__init__()
        self.workspace = workspace or _make_workspace()
        self._init_runtime_bridge_recording()

    async def _load_workspace_for_component_execution(self, workspace_id: str, user_id: str | None = None) -> UserSpaceWorkspace:  # type: ignore[override]
        return self.workspace

    async def _resolve_effective_workspace_tool_ids(self, workspace: UserSpaceWorkspace) -> list[str]:  # type: ignore[override]
        return list(workspace.selected_tool_ids)


class RuntimeBridgeExecuteTests(unittest.IsolatedAsyncioTestCase):
    async def test_bridge_execute_delegates_to_workspace_path_and_records_proof(self) -> None:
        service = _RuntimeBridgeSuccessService()
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})

        response = await service.execute_component_from_runtime_bridge("workspace-1", request, session_id="sess-1")

        self.assertEqual(response.row_count, 1)
        self.assertEqual(service.proofs_recorded, [("workspace-1", "tool-1", 1, "select 1")])

    async def test_bridge_execute_unselected_component_propagates_403(self) -> None:
        service = _RuntimeBridgeWorkspaceService()
        request = ExecuteComponentRequest(component_id="tool-2", request={"query": "select 1"})
        audit_spy = mock.AsyncMock(wraps=service._record_runtime_bridge_audit)
        service._record_runtime_bridge_audit = audit_spy  # type: ignore[method-assign]

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config()),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.execute_component_from_runtime_bridge("workspace-1", request, session_id="sess-1")

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("not selected", str(ctx.exception.detail))
        audit_spy.assert_awaited_once_with(
            workspace_id="workspace-1",
            session_id="sess-1",
            component_id="tool-2",
            query_digest=hashlib.sha256(b"select 1").hexdigest(),
            error="Component tool-2 is not selected for this request.",
            access_mode="denied",
            tool_type="postgres",
        )

    async def test_bridge_execute_unsupported_tool_propagates_400_and_audits_once(self) -> None:
        service = _RuntimeBridgeWorkspaceService()
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})
        audit_spy = mock.AsyncMock(wraps=service._record_runtime_bridge_audit)
        service._record_runtime_bridge_audit = audit_spy  # type: ignore[method-assign]

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(tool_type="filesystem_indexer")),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await service.execute_component_from_runtime_bridge("workspace-1", request, session_id="sess-1")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("does not support filesystem_indexer", str(ctx.exception.detail))
        audit_spy.assert_awaited_once()
        audit_args = audit_spy.await_args
        assert audit_args is not None
        self.assertEqual(audit_args.kwargs["tool_type"], "filesystem_indexer")
        self.assertEqual(audit_args.kwargs["access_mode"], "read_only")

    async def test_bridge_execute_writes_audit_row_with_query_digest(self) -> None:
        service = _RuntimeBridgeSuccessService()
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})

        await service.execute_component_from_runtime_bridge("workspace-1", request, session_id="sess-1")

        self.assertEqual(len(service.bridge_audit_calls), 1)
        self.assertEqual(
            service.bridge_audit_calls[0],
            {
                "workspace_id": "workspace-1",
                "session_id": "sess-1",
                "component_id": "tool-1",
                "query_digest": hashlib.sha256(b"select 1").hexdigest(),
                "error": None,
                "access_mode": "read_only",
                "tool_type": "postgres",
            },
        )

    async def test_bridge_execute_rejects_write_query_without_workspace_opt_in(self) -> None:
        service = _RuntimeBridgeWorkspaceService()
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "DELETE FROM x"})

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(allow_write=True)),
        ):
            response = await service.execute_component_from_runtime_bridge("workspace-1", request, session_id="sess-1")

        self.assertEqual(response.component_id, "tool-1")
        self.assertEqual(response.row_count, 0)
        self.assertRegex(response.error or "", "Only SELECT queries are allowed|read-only|blocked")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["access_mode"], "read_only")

    async def test_bridge_execute_rejects_write_query_when_global_allow_write_is_false(self) -> None:
        workspace = _make_workspace()
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
        service = _RuntimeBridgeWorkspaceService(workspace)

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(allow_write=False)),
        ):
            response = await service.execute_component_from_runtime_bridge(
                "workspace-1",
                ExecuteComponentRequest(component_id="tool-1", request={"query": "UPDATE x SET y = 1"}),
                session_id="sess-1",
            )

        self.assertIsNotNone(response.error)
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["access_mode"], "read_only")

    async def test_bridge_execute_allows_sql_write_only_when_workspace_opt_in_and_global_allow_write(self) -> None:
        workspace = _make_workspace()
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
        service = _RuntimeBridgeWorkspaceService(workspace)

        class _Process:
            returncode = 0

            async def communicate(self) -> tuple[bytes, bytes]:
                return b"", b""

            def kill(self) -> None:
                raise AssertionError("kill should not be called")

        create_subprocess_exec = mock.AsyncMock(return_value=_Process())

        with (
            mock.patch.object(
                userspace_service_module.repository,
                "get_tool_config",
                mock.AsyncMock(return_value=_make_tool_config(allow_write=True, connection_config={"container": "ragtime-db-test"})),
            ),
            mock.patch.object(userspace_service_module.asyncio, "create_subprocess_exec", create_subprocess_exec),
        ):
            response = await service.execute_component_from_runtime_bridge(
                "workspace-1",
                ExecuteComponentRequest(component_id="tool-1", request={"query": "UPDATE widgets SET enabled = true"}),
                session_id="sess-1",
            )

        self.assertIsNone(response.error)
        self.assertEqual(response.row_count, 0)
        self.assertEqual(service.bridge_audit_calls[-1]["access_mode"], "read_write")
        self.assertEqual(len(service.proofs_recorded), 1)
        create_subprocess_exec.assert_awaited_once()

    async def test_browser_and_shared_preview_execution_stay_read_only_even_with_workspace_opt_in(self) -> None:
        workspace = _make_workspace()
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
        service = _RuntimeBridgeWorkspaceService(workspace)

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(allow_write=True)),
        ):
            browser_response = await service.execute_component(
                "workspace-1",
                ExecuteComponentRequest(component_id="tool-1", request={"query": "DELETE FROM x"}),
                user_id="user-1",
            )
            shared_response = await service.execute_component_from_authorized_shared_preview(
                "workspace-1",
                ExecuteComponentRequest(component_id="tool-1", request={"query": "DELETE FROM x"}),
            )

        self.assertIsNotNone(browser_response.error)
        self.assertIsNotNone(shared_response.error)
        self.assertEqual(service.proofs_recorded, [])

    async def test_browser_preview_execution_rejects_sql_mutations_even_with_workspace_opt_in(self) -> None:
        workspace = _make_workspace()
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
        service = _RuntimeBridgeWorkspaceService(workspace)

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(allow_write=True)),
        ):
            for query in (
                "INSERT INTO t VALUES (1)",
                "UPDATE t SET a=1",
                "DELETE FROM t",
            ):
                response = await service.execute_component(
                    "workspace-1",
                    ExecuteComponentRequest(component_id="tool-1", request={"query": query}),
                    user_id="user-1",
                )

                self.assertEqual(response.component_id, "tool-1")
                self.assertEqual(response.row_count, 0)
                self.assertIn("Only SELECT queries are allowed", response.error or "")

        self.assertEqual(service.proofs_recorded, [])

    async def test_browser_preview_execution_rejects_non_sql_tool_types(self) -> None:
        workspace = _make_workspace()
        service = _RuntimeBridgeWorkspaceService(workspace)

        for tool_type in ("odoo_shell", "ssh_shell"):
            with mock.patch.object(
                userspace_service_module.repository,
                "get_tool_config",
                mock.AsyncMock(return_value=_make_tool_config(tool_type=tool_type, allow_write=True)),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    await service.execute_component(
                        "workspace-1",
                        ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"}),
                        user_id="user-1",
                    )

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("Live preview execution supports SQL tools only", str(ctx.exception.detail))

    async def test_shared_preview_execution_rejects_insert_even_with_workspace_opt_in(self) -> None:
        workspace = _make_workspace()
        workspace.tool_options = {"tool-1": WorkspaceToolOptionState(write_access_enabled=True)}
        service = _RuntimeBridgeWorkspaceService(workspace)

        with mock.patch.object(
            userspace_service_module.repository,
            "get_tool_config",
            mock.AsyncMock(return_value=_make_tool_config(allow_write=True)),
        ):
            response = await service.execute_component_from_authorized_shared_preview(
                "workspace-1",
                ExecuteComponentRequest(component_id="tool-1", request={"query": "INSERT INTO t VALUES (1)"}),
            )

        self.assertEqual(response.component_id, "tool-1")
        self.assertEqual(response.row_count, 0)
        self.assertIn("Only SELECT queries are allowed", response.error or "")
        self.assertEqual(service.proofs_recorded, [])

    async def test_runtime_bridge_odoo_builder_uses_effective_allow_write_and_returns_output(self) -> None:
        service, response, tool, build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="odoo_shell",
            output={"ok": True},
            request_input="env['res.partner'].search([], limit=1).read(['name'])",
            write_enabled=True,
        )

        self.assertEqual(response.output, {"ok": True})
        self.assertEqual(response.rows, [])
        self.assertEqual(response.columns, [])
        self.assertEqual(response.row_count, 0)
        self.assertEqual(len(service.proofs_recorded), 1)
        build_tool.assert_awaited_once()
        build_tool_args = build_tool.await_args
        assert build_tool_args is not None
        self.assertTrue(build_tool_args.args[0]["allow_write"])
        tool.ainvoke.assert_awaited_once_with({"code": "env['res.partner'].search([], limit=1).read(['name'])"})
        self.assertEqual(service.bridge_audit_calls[-1]["tool_type"], "odoo_shell")
        self.assertEqual(service.bridge_audit_calls[-1]["access_mode"], "read_write")

    async def test_runtime_bridge_ssh_builder_uses_read_only_mode_without_workspace_opt_in(self) -> None:
        service, response, tool, build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="ssh_shell",
            output="ok",
            request_input={"command": "ls -la"},
        )

        self.assertEqual(response.output, "ok")
        build_tool_args = build_tool.await_args
        assert build_tool_args is not None
        self.assertFalse(build_tool_args.args[0]["allow_write"])
        tool.ainvoke.assert_awaited_once_with({"command": "ls -la"})
        self.assertEqual(service.bridge_audit_calls[-1]["access_mode"], "read_only")

    async def test_runtime_bridge_ssh_command_failed_json_sets_error_and_skips_proof(self) -> None:
        raw_output = '{"status":"command_failed","error":"Permission denied","stderr":"ssh: denied"}'
        service, response, _tool, _build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="ssh_shell",
            output=raw_output,
            request_input={"command": "ls -la"},
        )

        self.assertEqual(response.output, raw_output)
        self.assertEqual(response.error, "Permission denied")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["error"], "Permission denied")

    async def test_runtime_bridge_ssh_rejected_json_sets_status_error_and_skips_proof(self) -> None:
        raw_output = '{"status":"rejected"}'
        service, response, _tool, _build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="ssh_shell",
            output=raw_output,
            request_input={"command": "rm -rf /tmp/x"},
        )

        self.assertEqual(response.output, raw_output)
        self.assertEqual(response.error, "Tool execution rejected.")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["error"], "Tool execution rejected.")

    async def test_runtime_bridge_odoo_error_string_sets_error_and_skips_proof(self) -> None:
        service, response, _tool, _build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="odoo_shell",
            output="  Error: access denied",
            request_input="env['res.partner'].search([])",
        )

        self.assertEqual(response.output, "  Error: access denied")
        self.assertEqual(response.error, "Error: access denied")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["error"], "Error: access denied")

    async def test_runtime_bridge_completed_json_with_stderr_stays_successful(self) -> None:
        raw_output = '{"status":"completed","stderr":"warning only","stdout":"done"}'
        service, response, _tool, _build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="ssh_shell",
            output=raw_output,
            request_input={"command": "ls -la"},
        )

        self.assertEqual(response.output, raw_output)
        self.assertIsNone(response.error)
        self.assertEqual(len(service.proofs_recorded), 1)
        self.assertIsNone(service.bridge_audit_calls[-1]["error"])

    async def test_runtime_bridge_failed_dict_output_sets_error_and_skips_proof(self) -> None:
        raw_output = {"status": "failed", "error": "upstream failure"}
        service, response, _tool, _build_tool = await _execute_runtime_bridge_tool_request(
            tool_type="odoo_shell",
            output=raw_output,
            request_input="env['res.partner'].search([])",
        )

        self.assertEqual(response.output, raw_output)
        self.assertEqual(response.error, "upstream failure")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(service.bridge_audit_calls[-1]["error"], "upstream failure")


if __name__ == "__main__":
    unittest.main()
