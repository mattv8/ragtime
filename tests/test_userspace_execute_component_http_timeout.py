from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

inserted_fake_rag_prompts = "ragtime.rag.prompts" not in sys.modules
if inserted_fake_rag_prompts:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace import service as userspace_service_module
from ragtime.userspace.models import (
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    UserSpacePreviewDiagnosticEvent,
    UserSpaceWorkspace,
)
from ragtime.userspace.service import UserSpaceService

if inserted_fake_rag_prompts:
    sys.modules.pop("ragtime.rag", None)
    sys.modules.pop("ragtime.rag.prompts", None)


def _make_workspace() -> UserSpaceWorkspace:
    now = datetime.now(timezone.utc)
    return UserSpaceWorkspace(
        id="workspace-1",
        name="WS",
        owner_user_id="user-1",
        selected_tool_ids=["tool-1"],
        created_at=now,
        updated_at=now,
    )


class _HangingExecuteService(UserSpaceService):
    def __init__(self) -> None:
        super().__init__()
        self.warnings_recorded: list[tuple[str, str, str]] = []
        self.proofs_recorded: list[tuple[str, str, int, str]] = []
        self.diagnostic_events: list[tuple[str, list[UserSpacePreviewDiagnosticEvent]]] = []
        self.dispatch_started = asyncio.Event()

    async def _execute_component_for_selected_tool_ids(self, **kwargs):  # type: ignore[no-untyped-def]
        self.dispatch_started.set()
        await asyncio.Event().wait()  # never resolves; simulate slow SQL
        raise AssertionError("unreachable")

    def record_live_data_execution_warning(self, workspace_id: str, component_id: str, error: str) -> None:  # type: ignore[override]
        self.warnings_recorded.append((workspace_id, component_id, error))

    def record_execution_proof(self, *args, **kwargs) -> None:  # type: ignore[override]
        self.proofs_recorded.append(args)

    async def record_workspace_preview_diagnostic_events(self, workspace_id: str, events: list[UserSpacePreviewDiagnosticEvent]) -> int:  # type: ignore[override]
        self.diagnostic_events.append((workspace_id, events))
        return len(events)


class _ImmediateExecuteService(UserSpaceService):
    def __init__(self, response: ExecuteComponentResponse) -> None:
        super().__init__()
        self._response = response
        self.warnings_recorded: list[tuple[str, str, str]] = []
        self.proofs_recorded: list[tuple[str, str, int, str]] = []
        self.diagnostic_events: list[tuple[str, list[UserSpacePreviewDiagnosticEvent]]] = []

    async def _execute_component_for_selected_tool_ids(self, **kwargs):  # type: ignore[no-untyped-def]
        return (self._response.model_copy(update={"component_id": str(kwargs.get("component_id") or "tool-1")}), "select 1")

    async def _load_workspace_for_component_execution(self, workspace_id: str, user_id: str | None = None) -> UserSpaceWorkspace:  # type: ignore[override]
        return _make_workspace()

    def record_live_data_execution_warning(self, workspace_id: str, component_id: str, error: str) -> None:  # type: ignore[override]
        self.warnings_recorded.append((workspace_id, component_id, error))

    def record_execution_proof(self, workspace_id: str, component_id: str, row_count: int, query: str) -> None:  # type: ignore[override]
        self.proofs_recorded.append((workspace_id, component_id, row_count, query))

    def clear_live_data_execution_warning(self, workspace_id: str) -> None:  # type: ignore[override]
        pass

    async def record_workspace_preview_diagnostic_events(self, workspace_id: str, events: list[UserSpacePreviewDiagnosticEvent]) -> int:  # type: ignore[override]
        self.diagnostic_events.append((workspace_id, events))
        return len(events)


class _HangingSubprocess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False
        self.communicate_started = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_started.set()
        while self.returncode is None:
            await asyncio.sleep(0.01)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class UserSpaceExecuteComponentHttpTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_component_returns_structured_timeout_before_524(self) -> None:
        service = _HangingExecuteService()
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "select pg_sleep(600)"})

        with mock.patch.object(
            userspace_service_module,
            "get_http_proxy_safe_timeout_seconds",
            mock.AsyncMock(return_value=0.05),
        ):
            response = await service._execute_component_for_workspace(
                _make_workspace(),
                request,
                error_log_prefix="Component execution failed",
            )

        self.assertEqual(response.component_id, "tool-1")
        self.assertEqual(response.row_count, 0)
        self.assertEqual(response.error_kind, "timeout")
        self.assertIsInstance(response.timeout_seconds, int)
        assert response.timeout_seconds is not None
        self.assertGreaterEqual(response.timeout_seconds, 1)
        self.assertIn("Live data query exceeded the request timeout", response.error or "")
        self.assertIn("Settings > Tools", response.admin_action or "")
        self.assertTrue(service.dispatch_started.is_set())
        self.assertEqual(len(service.warnings_recorded), 1)
        self.assertEqual(service.warnings_recorded[0][0], "workspace-1")
        self.assertEqual(service.warnings_recorded[0][1], "tool-1")
        self.assertEqual(service.proofs_recorded, [])
        self.assertEqual(len(service.diagnostic_events), 1)
        self.assertEqual(service.diagnostic_events[0][0], "workspace-1")
        timeout_event = service.diagnostic_events[0][1][0]
        self.assertEqual(timeout_event.kind, "component_execute")
        self.assertEqual(timeout_event.component_id, "tool-1")
        self.assertEqual(timeout_event.row_count, 0)
        self.assertIn("request timeout", timeout_event.error or "")

    async def test_execute_component_passes_successful_response_through(self) -> None:
        ok_response = ExecuteComponentResponse(
            component_id="tool-1",
            rows=[{"id": 1}],
            columns=["id"],
            row_count=1,
        )
        service = _ImmediateExecuteService(ok_response)
        request = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})

        response = await service._execute_component_for_workspace(
            _make_workspace(),
            request,
            error_log_prefix="Component execution failed",
        )

        self.assertIsNone(response.error)
        self.assertIsNone(response.error_kind)
        self.assertEqual(response.row_count, 1)
        self.assertEqual(service.warnings_recorded, [])
        self.assertEqual(len(service.proofs_recorded), 1)
        self.assertEqual(service.proofs_recorded[0][0], "workspace-1")
        self.assertEqual(service.proofs_recorded[0][1], "tool-1")
        self.assertEqual(len(service.diagnostic_events), 1)
        self.assertEqual(service.diagnostic_events[0][0], "workspace-1")
        success_event = service.diagnostic_events[0][1][0]
        self.assertEqual(success_event.kind, "component_execute")
        self.assertEqual(success_event.component_id, "tool-1")
        self.assertEqual(success_event.row_count, 1)
        self.assertIsNone(success_event.error)
        self.assertGreaterEqual(success_event.elapsed_ms, 0)

    async def test_postgres_subprocess_is_killed_when_outer_request_is_cancelled(self) -> None:
        service = UserSpaceService()
        process = _HangingSubprocess()

        async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
            return process

        with mock.patch.object(
            userspace_service_module.asyncio,
            "create_subprocess_exec",
            fake_create_subprocess_exec,
        ):
            task = asyncio.create_task(
                service._execute_postgres_query(
                    {"container": "ragtime-db-test"},
                    "select 1 limit 1",
                    300,
                    100,
                    require_result_limit=False,
                    enforce_result_limit=False,
                )
            )
            await asyncio.wait_for(process.communicate_started.wait(), timeout=1)
            task.cancel()

            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(process.killed)
        self.assertEqual(process.returncode, -9)

    async def test_shared_component_execution_does_not_record_owner_prompt_diagnostics(self) -> None:
        ok_response = ExecuteComponentResponse(
            component_id="tool-1",
            rows=[{"id": 1}],
            columns=["id"],
            row_count=1,
        )
        service = _ImmediateExecuteService(ok_response)

        response = await service.execute_component_from_authorized_shared_preview(
            "workspace-1",
            ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"}),
        )

        self.assertEqual(response.row_count, 1)
        self.assertEqual(service.diagnostic_events, [])


if __name__ == "__main__":
    unittest.main()
