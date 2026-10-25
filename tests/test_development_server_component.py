from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace import development_service as development_module
from ragtime.userspace import service as service_module
from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.models import ExecuteComponentResponse, UserSpaceWorkspace
from tests.content_protection_support import use_disabled_content_protection


def _tool(tool_type: str, *, timeout: int = 60) -> SimpleNamespace:
    return SimpleNamespace(
        id="tool-1",
        name="Tool",
        tool_type=SimpleNamespace(value=tool_type),
        description="Configured tool.",
        connection_config={},
        timeout_max_seconds=timeout,
        max_results=100,
        enabled=True,
        allow_write=True,
    )


def _workspace() -> UserSpaceWorkspace:
    now = datetime.now(timezone.utc)
    return UserSpaceWorkspace(
        id="workspace-1",
        name="Workspace",
        owner_user_id="user-1",
        tool_selection_mode="custom",
        selected_tool_ids=["tool-1"],
        tool_options={},
        created_at=now,
        updated_at=now,
    )


class DevelopmentServerComponentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        use_disabled_content_protection(self)

    async def test_odoo_dispatches_with_caller_write_ceiling(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, credential_id="credential-1", scopes=frozenset({"read", "exec", "write"}))
        catalog = {"tools": [{"component_id": "tool-1", "tool_type": "odoo_shell", "access": "read_write"}]}
        response = ExecuteComponentResponse(component_id="tool-1", rows=[], columns=[], row_count=0)
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value=catalog)),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(
                development_module.userspace_service,
                "execute_component_from_development_server_lane",
                mock.AsyncMock(return_value=response),
            ) as execute,
        ):
            result = await development_module.development_service.execute(
                principal,
                "workspace-1",
                "execute_component",
                {"component_id": "tool-1", "request": {"code": "env['res.partner'].search([])"}},
            )

        self.assertEqual(result, response.model_dump())
        execute.assert_awaited_once()
        assert execute.await_args is not None
        self.assertTrue(execute.await_args.kwargs["caller_write_allowed"])

    async def test_dispatch_write_ceiling_combinations_and_sql_fallback(self) -> None:
        response = ExecuteComponentResponse(component_id="tool-1", rows=[], columns=[], row_count=0)
        for access, scopes, expected in (
            ("read", {"read", "exec", "write"}, False),
            ("read_write", {"read", "exec"}, False),
            ("read_write", {"read", "exec", "write"}, True),
        ):
            principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset(scopes))
            catalog = {"tools": [{"component_id": "tool-1", "tool_type": "odoo_shell", "access": access}]}
            with (
                mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
                mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value=catalog)),
                mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
                mock.patch.object(
                    development_module.userspace_service, "execute_component_from_development_server_lane", mock.AsyncMock(return_value=response)
                ) as direct,
            ):
                await development_module.development_service.execute(
                    principal, "workspace-1", "execute_component", {"component_id": "tool-1", "request": "code"}
                )
            assert direct.await_args is not None
            self.assertEqual(direct.await_args.kwargs["caller_write_allowed"], expected)

        catalog = {"tools": [{"component_id": "tool-1", "tool_type": "postgres", "access": "read"}]}
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock()),
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value=catalog)),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(development_module.userspace_service, "execute_component", mock.AsyncMock(return_value=response)) as browser,
        ):
            await development_module.development_service.execute(
                principal, "workspace-1", "execute_component", {"component_id": "tool-1", "request": "select 1"}
            )
        browser.assert_awaited_once()

    async def test_odoo_requires_exec_scope_before_service_execution(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        catalog = {"tools": [{"component_id": "tool-1", "tool_type": "odoo_shell", "access": "read"}]}

        async def workspace(principal: DevelopmentPrincipal, workspace_id: str, scope: str) -> None:
            _ = principal, workspace_id
            if scope == "exec":
                raise HTTPException(status_code=403)

        with (
            mock.patch.object(development_module.development_service, "_workspace", mock.AsyncMock(side_effect=workspace)) as workspace_check,
            mock.patch.object(development_module.development_service, "_resources", mock.AsyncMock(return_value=catalog)),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service._audit", mock.AsyncMock()),
            mock.patch.object(development_module.userspace_service, "execute_component_from_development_server_lane", mock.AsyncMock()) as execute,
            self.assertRaises(HTTPException) as raised,
        ):
            await development_module.development_service.execute(
                principal,
                "workspace-1",
                "execute_component",
                {"component_id": "tool-1", "request": "env['res.partner']"},
            )

        self.assertEqual(raised.exception.status_code, 403)
        execute.assert_not_awaited()
        self.assertEqual(workspace_check.await_args_list[-1].args[2], "exec")

    async def test_odoo_contract_uses_direct_lane_and_proxy_timeout(self) -> None:
        tool = _tool("odoo_shell", timeout=60)
        with mock.patch.object(
            development_module.rag,
            "build_primary_runtime_tool_from_config",
            mock.AsyncMock(return_value=SimpleNamespace(description="Canonical Odoo instructions.")),
        ):
            contract = await development_module.development_service._enrich_authorized_tool(tool, "read_write", frozenset({"read", "exec"}), True, 30)

        assert contract is not None
        lane = contract["execution_lanes"]["server_direct"]
        self.assertTrue(lane["supported"])
        self.assertEqual(lane["timeout_max_seconds"], 20)
        self.assertEqual(contract["execute_component"]["mode"], "server_direct")
        self.assertIn("not a sandbox", lane["write_policy"])

    async def test_sql_contract_keeps_browser_execute_component_lane(self) -> None:
        tool = _tool("postgres")
        with mock.patch.object(
            development_module.rag,
            "build_primary_runtime_tool_from_config",
            mock.AsyncMock(return_value=SimpleNamespace(description="Canonical SQL instructions.")),
        ):
            contract = await development_module.development_service._enrich_authorized_tool(tool, "read", frozenset({"read"}), True, 30)

        assert contract is not None
        self.assertEqual(contract["execute_component"]["mode"], "browser_read_only")
        self.assertNotIn("server_direct", contract["execution_lanes"])

    async def test_server_lane_disables_preview_side_effects_and_audits_digest(self) -> None:
        service = service_module.UserSpaceService()
        request = development_module.ExecuteComponentRequest(component_id="tool-1", request={"code": "env['res.partner']"})
        response = ExecuteComponentResponse(component_id="tool-1", rows=[], columns=[], row_count=0)
        result = SimpleNamespace(response=response, tool_type="odoo_shell", access_mode="read_only")
        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_execute_component_for_workspace_result", mock.AsyncMock(return_value=result)) as execute,
            mock.patch.object(service, "_record_runtime_audit_event", mock.AsyncMock(return_value=True)) as audit,
        ):
            actual = await service.execute_component_from_development_server_lane(
                "workspace-1",
                request,
                user_id="user-1",
                credential_id="credential-1",
                caller_write_allowed=False,
            )

        self.assertEqual(actual, response)
        assert execute.await_args is not None
        assert audit.await_args is not None
        policy = execute.await_args.kwargs["lane_policy"]
        self.assertTrue(policy.suppress_workspace_side_effects)
        payload = audit.await_args.args[3]
        self.assertEqual(payload["query_digest"], service._compute_runtime_bridge_query_digest(execute.await_args.args[1].request, tool_type="odoo_shell"))
        self.assertNotIn("code", payload)

    async def test_server_lane_rejects_when_the_user_cap_is_full(self) -> None:
        service = service_module.UserSpaceService()
        service._development_server_lane_user_execution_counts["user-1"] = 2
        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_record_runtime_audit_event", mock.AsyncMock(return_value=True)) as audit,
            self.assertRaises(HTTPException) as raised,
        ):
            await service.execute_component_from_development_server_lane(
                "workspace-1",
                development_module.ExecuteComponentRequest(component_id="tool-1", request="code"),
                user_id="user-1",
                credential_id=None,
                caller_write_allowed=False,
            )

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(service._development_server_lane_user_execution_counts["user-1"], 2)
        assert audit.await_args is not None
        self.assertEqual(audit.await_args.args[3]["error_kind"], "concurrency_limited")

    def test_normalize_development_server_lane_request(self) -> None:
        limit = max(1, 30 - service_module._DEVELOPMENT_SERVER_LANE_TIMEOUT_MARGIN_SECONDS)
        cases = (
            ("code", {"code": "code", "reason": "", "timeout": limit}),
            ({"code": "code", "timeout": 3}, {"code": "code", "reason": "", "timeout": 3}),
            ({"code": "code", "timeout": 99}, {"code": "code", "reason": "", "timeout": limit}),
            ({"code": "code", "timeout": 0}, {"code": "code", "reason": "", "timeout": limit}),
            ({"code": "code", "timeout": -1}, {"code": "code", "reason": "", "timeout": limit}),
        )
        for payload, expected in cases:
            with self.subTest(payload=payload):
                normalized = service_module.UserSpaceService._normalize_development_server_lane_request(
                    development_module.ExecuteComponentRequest(component_id="tool-1", request=payload),
                    30,
                )
                self.assertEqual(normalized.request, expected)

        for invalid_payload in ({"code": "code", "unknown": "x"}, {"code": 1}, {"code": "code", "timeout": True}):
            with self.subTest(invalid_payload=invalid_payload):
                with self.assertRaises(HTTPException) as raised:
                    service_module.UserSpaceService._normalize_development_server_lane_request(
                        development_module.ExecuteComponentRequest(component_id="tool-1", request=invalid_payload),
                        30,
                    )
                self.assertEqual(raised.exception.status_code, 422)

    async def test_lane_policy_clamps_write_and_rejects_other_tool_types(self) -> None:
        service = service_module.UserSpaceService()
        writable_policy = service_module._ComponentExecutionLanePolicy(
            allowed_tool_types=service_module._USPACE_DEVELOPMENT_SERVER_LANE_TOOL_TYPES,
            caller_write_allowed=True,
            suppress_workspace_side_effects=True,
        )
        readonly_policy = service_module._ComponentExecutionLanePolicy(
            allowed_tool_types=service_module._USPACE_DEVELOPMENT_SERVER_LANE_TOOL_TYPES,
            caller_write_allowed=False,
            suppress_workspace_side_effects=True,
        )
        with mock.patch.object(service_module.repository, "get_tool_config", mock.AsyncMock(return_value=_tool("odoo_shell"))):
            readonly = await service._resolve_component_execution_config_for_tool_ids(
                ["tool-1"],
                "tool-1",
                workspace_tool_options={"tool-1": {"write_access_enabled": True}},
                allow_runtime_bridge_tools=True,
                owner_access_levels={"tool-1": "read_write"},
                lane_policy=readonly_policy,
            )
            writable = await service._resolve_component_execution_config_for_tool_ids(
                ["tool-1"],
                "tool-1",
                workspace_tool_options={"tool-1": {"write_access_enabled": True}},
                allow_runtime_bridge_tools=True,
                owner_access_levels={"tool-1": "read_write"},
                lane_policy=writable_policy,
            )

        self.assertEqual((readonly.effective_allow_write, readonly.access_mode), (False, "read_only"))
        self.assertEqual((writable.effective_allow_write, writable.access_mode), (True, "read_write"))
        for tool_type in ("postgres", "ssh_shell"):
            with mock.patch.object(service_module.repository, "get_tool_config", mock.AsyncMock(return_value=_tool(tool_type))):
                with self.assertRaises(HTTPException) as raised:
                    await service._resolve_component_execution_config_for_tool_ids(["tool-1"], "tool-1", lane_policy=readonly_policy)
            self.assertEqual(raised.exception.status_code, 400)

    async def test_lane_policy_skips_workspace_side_effects(self) -> None:
        service = service_module.UserSpaceService()
        workspace = _workspace()
        request = development_module.ExecuteComponentRequest(component_id="tool-1", request={"code": "code"})
        response = ExecuteComponentResponse(component_id="tool-1", rows=[], columns=[], row_count=0)
        result = service_module._ExecuteComponentExecutionResult(
            response=response,
            proof_input="code",
            access_mode="read_only",
            tool_type="odoo_shell",
        )
        policy = service_module._ComponentExecutionLanePolicy(frozenset({"odoo_shell"}), False, True)
        with (
            mock.patch.object(service_module, "get_http_proxy_safe_timeout_seconds", mock.AsyncMock(return_value=30)),
            mock.patch.object(service, "_resolve_effective_workspace_tool_ids", mock.AsyncMock(return_value=["tool-1"])),
            mock.patch.object(service, "_resolve_workspace_owner_tool_access", mock.AsyncMock(return_value={"tool-1": "read_write"})),
            mock.patch.object(
                service,
                "_resolve_component_execution_config_for_tool_ids",
                mock.AsyncMock(return_value=SimpleNamespace(resolved_id="tool-1", access_mode="read_only", tool_type="odoo_shell")),
            ),
            mock.patch.object(service, "_execute_component_for_selected_tool_ids", mock.AsyncMock(return_value=result)),
            mock.patch.object(service, "record_execution_proof") as proof,
            mock.patch.object(service, "clear_live_data_execution_warning") as clear_warning,
            mock.patch.object(service, "record_live_data_execution_warning") as warning,
            mock.patch.object(service, "record_workspace_preview_diagnostic_events", mock.AsyncMock()) as diagnostics,
        ):
            await service._execute_component_for_workspace_result(workspace, request, error_log_prefix="test", lane_policy=policy)

        proof.assert_not_called()
        clear_warning.assert_not_called()
        warning.assert_not_called()
        diagnostics.assert_not_awaited()

    async def test_server_lane_releases_counter_after_exception_and_preserves_non_timeout_error(self) -> None:
        service = service_module.UserSpaceService()
        request = development_module.ExecuteComponentRequest(component_id="tool-1", request="code")
        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_execute_component_for_workspace_result", mock.AsyncMock(side_effect=RuntimeError("failed"))),
            mock.patch.object(service, "_record_runtime_audit_event", mock.AsyncMock(return_value=True)),
            self.assertRaises(RuntimeError),
        ):
            await service.execute_component_from_development_server_lane(
                "workspace-1", request, user_id="user-1", credential_id=None, caller_write_allowed=False
            )
        self.assertNotIn("user-1", service._development_server_lane_user_execution_counts)

        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_execute_component_for_workspace_result", mock.AsyncMock(side_effect=asyncio.CancelledError)),
            self.assertRaises(asyncio.CancelledError),
        ):
            await service.execute_component_from_development_server_lane(
                "workspace-1", request, user_id="user-1", credential_id=None, caller_write_allowed=False
            )
        self.assertNotIn("user-1", service._development_server_lane_user_execution_counts)

        response = ExecuteComponentResponse(component_id="tool-1", rows=[], columns=[], row_count=0, error="Error: connection timed out after 5s")
        result = SimpleNamespace(response=response, tool_type="odoo_shell", access_mode="read_only")
        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_execute_component_for_workspace_result", mock.AsyncMock(return_value=result)),
            mock.patch.object(service, "_record_runtime_audit_event", mock.AsyncMock(return_value=True)),
        ):
            actual = await service.execute_component_from_development_server_lane(
                "workspace-1", request, user_id="user-1", credential_id=None, caller_write_allowed=False
            )
        self.assertEqual(actual.error, response.error)

        timeout_response = ExecuteComponentResponse(
            component_id="tool-1",
            rows=[],
            columns=[],
            row_count=0,
            error_kind="timeout",
            timeout_seconds=3,
        )
        timeout_result = SimpleNamespace(response=timeout_response, tool_type="odoo_shell", access_mode="read_only")
        with (
            mock.patch.object(service, "_load_workspace_for_component_execution", mock.AsyncMock(return_value=_workspace())),
            mock.patch.object(service, "_execute_component_for_workspace_result", mock.AsyncMock(return_value=timeout_result)),
            mock.patch.object(service, "_record_runtime_audit_event", mock.AsyncMock(return_value=True)),
        ):
            actual = await service.execute_component_from_development_server_lane(
                "workspace-1", request, user_id="user-1", credential_id=None, caller_write_allowed=False
            )
        self.assertIn("requested or platform time limit of 3s", actual.error or "")
        self.assertIn("Split the work into smaller calls", actual.admin_action or "")
