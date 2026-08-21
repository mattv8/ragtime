import json
import math
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock

fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _fake_ensure_copilot_token_fresh(*_args, **_kwargs):
    return None


setattr(fake_copilot_auth, "ensure_copilot_token_fresh", _fake_ensure_copilot_token_fresh)
sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

from ragtime.rag.components import (
    FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
    RAGComponents,
    should_truncate_stream_display_output,
    wrap_tool_with_truncation,
)
from ragtime.rag.prompts import _USERSPACE_MODE_PROMPT_TEMPLATE, _USERSPACE_TURN_REMINDER_BASE, build_workspace_scm_setup_prompt


class UserSpaceUpsertToolValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _tool(self, name: str):
        tools = await RAGComponents()._create_userspace_file_tools("workspace-1", "user-1")
        for tool in tools:
            if tool.name == name:
                return tool
        raise AssertionError(f"{name} tool not found")

    async def _upsert_tool(self):
        return await self._tool("upsert_userspace_file")

    async def test_userspace_diagnostics_tool_returns_comprehensive_bounded_list(self) -> None:
        tool = await self._tool("userspace_diagnostics")
        coroutine = tool.coroutine
        assert coroutine is not None

        now = datetime.now(timezone.utc)
        diagnostics = [
            types.SimpleNamespace(
                kind="component_execute",
                diagnostic_key="component_execute:sales",
                target_label="component sales",
                count="4.7",
                error_count=-2,
                last_ms=math.inf,
                avg_ms=2200,
                max_ms=9100,
                last_error=None,
                last_status_code=None,
                last_row_count=50,
                updated_at=now,
            ),
            types.SimpleNamespace(
                kind="preview_fetch",
                diagnostic_key="preview_fetch:GET:/api/orders/:id",
                target_label="GET /api/orders/:id",
                count=2,
                error_count=1,
                last_ms=90,
                avg_ms=100,
                max_ms=120,
                last_error="HTTP 500",
                last_status_code=500,
                last_row_count=None,
                updated_at=now,
            ),
        ]

        with mock.patch(
            "ragtime.rag.components.userspace_service.list_workspace_preview_diagnostic_summary",
            new_callable=mock.AsyncMock,
            return_value=diagnostics,
        ) as list_summary:
            result = await coroutine(reason="debug slow preview")

        list_summary.assert_awaited_once_with("workspace-1", limit=50)
        self.assertIn("User Space diagnostics", result)
        self.assertIn("component sales", result)
        self.assertIn("count=4", result)
        self.assertIn("errors=0", result)
        self.assertIn("last=0ms", result)
        self.assertIn("GET /api/orders/:id", result)
        self.assertIn("HTTP 500", result)
        self.assertIn("Rows: 50", result)

    async def test_assay_userspace_code_defaults_are_compact_but_preserve_diagnostics(self) -> None:
        tool = await self._tool("assay_userspace_code")
        coroutine = tool.coroutine
        assert coroutine is not None

        schema = tool.args_schema
        assert schema is not None
        assert not isinstance(schema, dict)
        self.assertEqual(schema.model_fields["max_files"].default, 6)
        self.assertEqual(schema.model_fields["max_chars_per_file"].default, 800)
        self.assertIn("structure", tool.description)
        self.assertIn("contract", tool.description)
        self.assertIn("search_userspace_code", tool.description)

        files = [
            types.SimpleNamespace(path="dashboard/main.ts"),
            *[types.SimpleNamespace(path=f"dashboard/file{i}.ts") for i in range(1, 8)],
            types.SimpleNamespace(path="index.html"),
        ]

        async def get_workspace_file(_workspace_id: str, path: str, _user_id: str, **_kwargs):
            return types.SimpleNamespace(
                path=path,
                artifact_type="module_ts" if path.endswith(".ts") else "html",
                content=(f"// {path}\n" + "x" * 1200),
                live_data_connections=[],
                live_data_checks=[],
            )

        with (
            mock.patch(
                "ragtime.rag.components.userspace_service.enforce_workspace_role",
                new_callable=mock.AsyncMock,
            ) as enforce_workspace_role,
            mock.patch(
                "ragtime.rag.components.userspace_service.list_workspace_files",
                new_callable=mock.AsyncMock,
                return_value=files,
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_file",
                new_callable=mock.AsyncMock,
                side_effect=get_workspace_file,
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace",
                new_callable=mock.AsyncMock,
                return_value=types.SimpleNamespace(
                    tool_selection_mode="custom",
                    selected_tool_ids=["tool-3"],
                    selected_tool_group_ids=[],
                ),
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_entrypoint_status",
                return_value=types.SimpleNamespace(state="valid", framework="node", error=None),
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.is_default_static_entrypoint",
                return_value=False,
            ),
            mock.patch(
                "ragtime.rag.components.repository.list_healthy_enabled_tool_ids",
                new_callable=mock.AsyncMock,
                return_value=["tool-2"],
            ),
            mock.patch(
                "ragtime.rag.components.repository.list_enabled_tool_ids",
                new_callable=mock.AsyncMock,
                return_value=["tool-2", "tool-3"],
            ),
            mock.patch(
                "ragtime.rag.components.repository.get_tool_ids_for_groups",
                new_callable=mock.AsyncMock,
                return_value=[],
            ),
        ):
            raw = await coroutine()

        payload = json.loads(raw)
        workspace = payload["workspace"]
        self.assertEqual(payload["tool"], "assay_userspace_code")
        self.assertEqual(workspace["summary"]["inspected_file_count"], 6)
        self.assertEqual(len(workspace["inspected_files"]), 6)
        self.assertLessEqual(len(workspace["inspected_files"][0]["preview"]), 800)
        self.assertTrue(workspace["structure"]["has_dashboard_entry"])
        self.assertTrue(workspace["structure"]["has_index_html"])
        self.assertEqual(workspace["structure"]["authoritative_entrypoint"], ".ragtime/runtime-entrypoint.json")
        self.assertTrue(workspace["live_data_contract"]["workspace_has_selected_tools"])
        self.assertEqual(workspace["live_data_contract"]["selected_tool_ids"], ["tool-3"])
        enforce_workspace_role.assert_awaited_once_with("workspace-1", "user-1", "editor")

    def test_userspace_prompt_guidance_splits_assay_from_indexed_search_with_fallback(self) -> None:
        scm_prompt = build_workspace_scm_setup_prompt(git_url="https://example.test/repo.git", git_branch="main")

        self.assertIn("assay_userspace_code to assess the current structure", scm_prompt)
        self.assertIn("search_userspace_code", scm_prompt)
        self.assertIn("when the code index is unavailable", scm_prompt)
        self.assertIn("list_userspace_files plus targeted reads", scm_prompt)
        self.assertIn("assay (structure/contract) -> search/read", _USERSPACE_TURN_REMINDER_BASE)
        self.assertIn("Use `assay_userspace_code` for structure and contract context", _USERSPACE_MODE_PROMPT_TEMPLATE)

    async def test_single_file_upsert_rejects_missing_content_without_writing(self) -> None:
        tool = await self._upsert_tool()
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.userspace_service.upsert_workspace_file",
            new_callable=mock.AsyncMock,
        ) as upsert_workspace_file:
            raw = await coroutine(path="dashboard/main.ts")

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "rejected_not_persisted")
        self.assertEqual(payload["failure_class"], "content_missing")
        self.assertEqual(payload["next_best_tool"], "upsert_userspace_file")
        self.assertTrue(payload["rejected"])
        self.assertFalse(payload["persisted"])
        upsert_workspace_file.assert_not_called()

    async def test_batched_upsert_rejects_entry_missing_content_without_writing(self) -> None:
        tool = await self._upsert_tool()
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.userspace_service.upsert_workspace_file",
            new_callable=mock.AsyncMock,
        ) as upsert_workspace_file:
            raw = await coroutine(files=[{"path": "dashboard/main.ts"}])

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "rejected_not_persisted")
        self.assertEqual(payload["failure_class"], "content_missing")
        self.assertEqual(payload["next_best_tool"], "upsert_userspace_file")
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertEqual(payload["summary"]["rejected"], 1)
        self.assertEqual(payload["files"][0]["failure_class"], "content_missing")
        upsert_workspace_file.assert_not_called()

    async def test_discover_userspace_primitives_reuses_primitive_payload_helpers(self) -> None:
        tool = await self._tool("discover_userspace_primitives")
        coroutine = tool.coroutine
        assert coroutine is not None

        capabilities_payload = {
            "workspace_id": "workspace-1",
            "endpoints": {
                "capabilities": "/__ragtime/capabilities",
                "session": "/__ragtime/session",
            },
        }
        session_payload = {
            "workspace_id": "workspace-1",
            "user_fingerprint": "fp_test",
            "auth": {
                "methods": [{"key": "ldap", "available": True}],
                "browser_auth_endpoint": "/__ragtime/browser-auth",
            },
        }
        with (
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_capabilities",
                new_callable=mock.AsyncMock,
                return_value=capabilities_payload,
            ) as primitive_capabilities,
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_session_payload",
                new_callable=mock.AsyncMock,
                return_value=session_payload,
            ) as primitive_session,
        ):
            raw = await coroutine(include_session=True)

        payload = json.loads(raw)
        self.assertLess(len(raw), 6000)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["tool"], "discover_userspace_primitives")
        self.assertEqual(payload["workspace_id"], "workspace-1")
        self.assertEqual(payload["next_best_tool"], "patch_userspace_file")
        self.assertIn("/__ragtime/session", raw)
        self.assertIn("/__ragtime/browser-auth", raw)
        self.assertIn("user_fingerprint", raw)
        self.assertEqual(payload["capabilities"], capabilities_payload)
        self.assertEqual(payload["session"], session_payload)
        primitive_capabilities.assert_awaited_once_with("workspace-1", "user-1", preview_mode="workspace")
        primitive_session.assert_awaited_once_with("workspace-1", "user-1", mode="workspace", same_origin_auth_endpoints=True)

    def _identity_policy_response(self, *, rules=None, available_auth_groups=None, can_configure=True):
        from ragtime.userspace.models import (
            UserSpaceIdentityEntitlementAuthGroup,
            UserSpaceIdentityEntitlementPolicyResponse,
            UserSpaceIdentityEntitlementRule,
        )

        def _group(payload):
            return UserSpaceIdentityEntitlementAuthGroup(**payload)

        return UserSpaceIdentityEntitlementPolicyResponse(
            workspace_id="workspace-1",
            can_configure=can_configure,
            rules=[
                UserSpaceIdentityEntitlementRule(
                    auth_group_id=rule["auth_group_id"],
                    entitlements=rule["entitlements"],
                    auth_group=_group(rule["auth_group"]),
                )
                for rule in (rules or [])
            ],
            available_auth_groups=[_group(group) for group in (available_auth_groups or [])],
        )

    async def test_discover_userspace_primitives_exposes_identity_entitlement_contract(self) -> None:
        tool = await self._tool("discover_userspace_primitives")
        coroutine = tool.coroutine
        assert coroutine is not None

        accounting_group = {
            "id": "group-1",
            "key": "ldap-accounting",
            "display_name": "Accounting",
            "provider": "ldap",
        }
        policy_response = self._identity_policy_response(
            rules=[
                {
                    "auth_group_id": "group-1",
                    "entitlements": ["recon.admin", "recon.admin", "recon.role.preparer"],
                    "auth_group": accounting_group,
                }
            ],
            available_auth_groups=[accounting_group],
        )
        session_payload = {
            "workspace_id": "workspace-1",
            "auth": {
                "authenticated": True,
                "entitlements": ["recon.admin"],
            },
        }

        with (
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_capabilities",
                new_callable=mock.AsyncMock,
                return_value={"workspace_id": "workspace-1"},
            ),
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_session_payload",
                new_callable=mock.AsyncMock,
                return_value=session_payload,
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_identity_entitlement_policy",
                new_callable=mock.AsyncMock,
                return_value=policy_response,
                create=True,
            ),
        ):
            raw = await coroutine(include_session=True)

        payload = json.loads(raw)
        identity = payload["identity_entitlements"]
        self.assertEqual(identity["policy_status"], "configured")
        self.assertEqual(identity["current_session_entitlements"], ["recon.admin"])
        self.assertEqual(identity["private_header"], "X-Ragtime-Internal-Authenticated-Entitlements")
        self.assertEqual(identity["browser_session_field"], "auth.entitlements")
        self.assertEqual(identity["configuration_tool"], "configure_userspace_identity_entitlements")
        self.assertEqual(identity["rules"][0]["entitlements"], ["recon.admin", "recon.role.preparer"])
        self.assertEqual(identity["rules"][0]["auth_group"], accounting_group)
        self.assertEqual(identity["configured_groups"], [accounting_group])
        self.assertIn("owner", identity["configuration_scope"])
        self.assertIn("fail", identity["failure_mode"].lower())
        self.assertIn("administrator", identity["missing_group_instructions"].lower())
        self.assertNotIn("members", json.dumps(identity).lower())

    async def test_discover_userspace_primitives_surfaces_policy_read_errors(self) -> None:
        from fastapi import HTTPException

        tool = await self._tool("discover_userspace_primitives")
        coroutine = tool.coroutine
        assert coroutine is not None

        with (
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_capabilities",
                new_callable=mock.AsyncMock,
                return_value={"workspace_id": "workspace-1"},
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_identity_entitlement_policy",
                new_callable=mock.AsyncMock,
                side_effect=HTTPException(status_code=403, detail="Owner access required"),
                create=True,
            ),
        ):
            raw = await coroutine(include_session=False)

        identity = json.loads(raw)["identity_entitlements"]
        self.assertEqual(identity["policy_status"], "read_failed")
        self.assertIn("403", identity["policy_read_error"])
        self.assertIn("Owner access required", identity["policy_read_error"])

    async def test_discover_userspace_primitives_compacts_large_available_auth_group_catalog(self) -> None:
        tool = await self._tool("discover_userspace_primitives")
        coroutine = tool.coroutine
        assert coroutine is not None

        policy_response = self._identity_policy_response(
            rules=[],
            available_auth_groups=[
                {
                    "id": f"group-{index:03d}",
                    "key": f"group-{index:03d}",
                    "display_name": f"Group {index:03d}",
                    "provider": "ldap",
                }
                for index in range(40)
            ],
        )

        with (
            mock.patch(
                "ragtime.userspace.runtime_routes._primitive_capabilities",
                new_callable=mock.AsyncMock,
                return_value={"workspace_id": "workspace-1"},
            ),
            mock.patch(
                "ragtime.rag.components.userspace_service.get_workspace_identity_entitlement_policy",
                new_callable=mock.AsyncMock,
                return_value=policy_response,
                create=True,
            ),
        ):
            raw = await coroutine(include_session=False)

        payload = json.loads(raw)
        identity = payload["identity_entitlements"]
        self.assertEqual(identity["policy_status"], "unconfigured")
        self.assertTrue(identity["available_auth_groups_truncated"])
        self.assertEqual(identity["available_auth_groups_total"], 40)
        self.assertLessEqual(len(identity["available_auth_groups"]), 25)
        self.assertEqual(identity["available_auth_groups"][0]["id"], "group-000")

    async def test_configure_userspace_identity_entitlements_tool_is_owner_scoped(self) -> None:
        from ragtime.userspace.models import ReplaceUserSpaceIdentityEntitlementPolicyRequest

        tool = await self._tool("configure_userspace_identity_entitlements")
        coroutine = tool.coroutine
        assert coroutine is not None

        schema = tool.args_schema
        assert schema is not None
        assert not isinstance(schema, dict)
        self.assertIn("rules", schema.model_fields)
        self.assertIn("reason", schema.model_fields)
        self.assertNotIn("workspace_id", schema.model_fields)

        executive_group = {
            "id": "group-2",
            "key": "ldap-executive",
            "display_name": "Executive",
            "provider": "ldap",
        }
        replace_response = self._identity_policy_response(
            rules=[
                {
                    "auth_group_id": "group-2",
                    "entitlements": ["recon.admin"],
                    "auth_group": executive_group,
                }
            ],
            available_auth_groups=[],
        )

        with mock.patch(
            "ragtime.rag.components.userspace_service.replace_workspace_identity_entitlement_policy",
            new_callable=mock.AsyncMock,
            return_value=replace_response,
            create=True,
        ) as replace_policy:
            raw = await coroutine(
                rules=[
                    {
                        "auth_group_id": "group-2",
                        "entitlements": ["recon.admin", "recon.admin"],
                    }
                ],
                reason="Grant admin access",
            )

        payload = json.loads(raw)
        self.assertEqual(payload["tool"], "configure_userspace_identity_entitlements")
        self.assertTrue(payload["persisted"])
        self.assertEqual(
            payload["rules"],
            [{"auth_group_id": "group-2", "entitlements": ["recon.admin"], "auth_group": executive_group}],
        )
        self.assertEqual(payload["configured_groups"], [executive_group])
        self.assertEqual(payload["reason"], "Grant admin access")
        await_args = replace_policy.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        request = await_args.args[2]
        self.assertIsInstance(request, ReplaceUserSpaceIdentityEntitlementPolicyRequest)
        self.assertEqual(await_args.args[:2], ("workspace-1", "user-1"))
        self.assertEqual(request.rules[0].auth_group_id, "group-2")
        self.assertEqual(request.rules[0].entitlements, ["recon.admin", "recon.admin"])
        self.assertFalse(await_args.kwargs.get("is_admin", False))

    async def test_frontend_json_display_tools_bypass_global_output_truncation(self) -> None:
        tool = await self._tool("discover_userspace_primitives")

        wrapped = wrap_tool_with_truncation(
            tool,
            32,
            preserve_output_tool_names=FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES,
        )

        self.assertIs(wrapped, tool)

    def test_frontend_json_display_tools_bypass_stream_display_truncation(self) -> None:
        output = json.dumps({"tool": "discover_userspace_primitives", "payload": "x" * 3000}, indent=2)

        for tool_name in FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES:
            self.assertFalse(
                should_truncate_stream_display_output(tool_name, output),
                tool_name,
            )

        self.assertTrue(should_truncate_stream_display_output("generic_tool", output))

    def test_structured_userspace_outputs_are_not_truncated(self) -> None:
        output = "x" * 3000

        for tool_name in (*FRONTEND_JSON_DISPLAY_INTEGRITY_TOOL_NAMES, "validate_userspace_code"):
            self.assertFalse(
                should_truncate_stream_display_output(tool_name, output),
                tool_name,
            )

    async def test_search_userspace_code_tool_delegates_to_workspace_index_service(self) -> None:
        tool = await self._tool("search_userspace_code")
        coroutine = tool.coroutine
        assert coroutine is not None

        with mock.patch(
            "ragtime.rag.components.workspace_code_index_service.search_workspace_code",
            new_callable=mock.AsyncMock,
            return_value={"status": "ready", "results": [{"path": "src/app.py"}]},
        ) as search_workspace_code:
            raw = await coroutine(query="where is startup configured?", max_results=3)

        payload = json.loads(raw)
        self.assertEqual(payload["tool"], "search_userspace_code")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["results"], [{"path": "src/app.py"}])
        search_workspace_code.assert_awaited_once_with(
            workspace_id="workspace-1",
            query="where is startup configured?",
            mode="hybrid",
            max_results=3,
            max_chars_per_result=1200,
        )


if __name__ == "__main__":
    unittest.main()
