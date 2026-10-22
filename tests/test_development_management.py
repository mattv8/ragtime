from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from fastapi import HTTPException

from ragtime.userspace.development_management import (
    MANAGEMENT_OPERATIONS,
    execute_management_operation,
    management_operation_visible,
)


def _principal(*, scopes=frozenset({"read", "write"}), auth_source="user", workspace_id=None, is_admin=False):
    return SimpleNamespace(
        user_id="user-1",
        is_admin=is_admin,
        scopes=scopes,
        auth_source=auth_source,
        workspace_id=workspace_id,
    )


class DevelopmentManagementTests(IsolatedAsyncioTestCase):
    async def test_metadata_only_forwards_whitelisted_fields_and_admin_flag(self):
        principal = _principal()
        updated = SimpleNamespace(id="ws-1", name="Renamed", description="Description")
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch("ragtime.userspace.development_management.userspace_service.update_workspace", mock.AsyncMock(return_value=updated)) as update,
        ):
            result = await execute_management_operation(principal, "ws-1", "workspace_metadata_update", {"name": "Renamed"})

        request = update.await_args.args[1]
        self.assertEqual(request.model_dump(exclude_none=True), {"name": "Renamed"})
        self.assertFalse(update.await_args.kwargs["is_admin"])
        self.assertEqual(result["name"], "Renamed")

    async def test_placeholder_preserves_value_and_hides_it_from_response(self):
        principal = _principal()
        updated = SimpleNamespace(key="REPORT_API_KEY", has_value=True, source="workspace")
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch("ragtime.userspace.development_management.userspace_service.upsert_workspace_env_var", mock.AsyncMock(return_value=updated)) as upsert,
        ):
            result = await execute_management_operation(principal, "ws-1", "env_placeholder_upsert", {"key": "REPORT_API_KEY"})

        request = upsert.await_args.args[2]
        self.assertIsNone(request.value)
        self.assertNotIn("value", result)

    async def test_selection_requires_user_provenance_and_owner_role(self):
        token_principal = _principal(auth_source="workspace_credential")
        self.assertFalse(management_operation_visible(token_principal, "workspace_tools_set"))
        with self.assertRaises(HTTPException) as raised:
            await execute_management_operation(token_principal, "ws-1", "workspace_tools_set", {"mode": "custom", "selected_tool_ids": []})
        self.assertEqual(raised.exception.status_code, 403)

    async def test_custom_selection_clears_groups_without_tool_options(self):
        principal = _principal()
        workspace = SimpleNamespace(owner_user_id="user-1")
        config = SimpleNamespace(id="tool-1", name="Catalog", description="Data catalog")
        updated = SimpleNamespace(id="ws-1", tool_selection_mode="custom", selected_tool_ids=[])
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock(return_value=workspace)),
            mock.patch("ragtime.userspace.development_management.repository.list_healthy_enabled_tool_ids", mock.AsyncMock(return_value=["tool-1"])),
            mock.patch("ragtime.userspace.development_management.repository.list_tool_configs", mock.AsyncMock(return_value=[config])),
            mock.patch("ragtime.userspace.development_management.resolve_tool_access", mock.AsyncMock(return_value={"tool-1": "read"})),
            mock.patch(
                "ragtime.userspace.development_management.userspace_service.filter_tool_ids_for_workspace_owner", mock.AsyncMock(return_value=["tool-1"])
            ),
            mock.patch("ragtime.userspace.development_management.userspace_service.update_workspace", mock.AsyncMock(return_value=updated)) as update,
        ):
            await execute_management_operation(principal, "ws-1", "workspace_tools_set", {"mode": "custom", "selected_tool_ids": []})

        request = update.await_args.args[1]
        self.assertEqual(request.selected_tool_ids, [])
        self.assertEqual(request.selected_tool_group_ids, [])
        self.assertIsNone(request.tool_options)

    async def test_candidates_intersect_caller_owner_and_health_and_page_before_returning(self):
        principal = _principal(scopes=frozenset({"read"}))
        workspace = SimpleNamespace(owner_user_id="owner")
        configs = [
            SimpleNamespace(id="allowed", name="Accounting", description="Ledger"),
            SimpleNamespace(id="caller-denied", name="Hidden", description="No"),
            SimpleNamespace(id="owner-denied", name="Also hidden", description="No"),
        ]
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock(return_value=workspace)),
            mock.patch(
                "ragtime.userspace.development_management.repository.list_healthy_enabled_tool_ids", mock.AsyncMock(return_value=[item.id for item in configs])
            ),
            mock.patch("ragtime.userspace.development_management.repository.list_tool_configs", mock.AsyncMock(return_value=configs)),
            mock.patch(
                "ragtime.userspace.development_management.resolve_tool_access",
                mock.AsyncMock(return_value={"allowed": "read", "caller-denied": "deny", "owner-denied": "read"}),
            ),
            mock.patch(
                "ragtime.userspace.development_management.userspace_service.filter_tool_ids_for_workspace_owner",
                mock.AsyncMock(return_value=["allowed", "caller-denied"]),
            ),
        ):
            result = await execute_management_operation(principal, "ws-1", "workspace_tools_available", {"query": "accounting", "limit": 1})

        self.assertEqual(result["items"], [{"component_id": "allowed", "name": "Accounting", "description": "Ledger"}])
        self.assertEqual(result["next_offset"], None)

    async def test_invalid_management_fields_are_rejected(self):
        principal = _principal()
        with mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock()):
            for arguments in (
                {"owner_user_id": "other"},
                {"tool_options": {}},
                {"value": "do-not-echo"},
                {"new_key": "RENAMED"},
            ):
                operation = "env_placeholder_upsert" if "value" in arguments or "new_key" in arguments else "workspace_metadata_update"
                with self.assertRaises(HTTPException) as raised:
                    await execute_management_operation(principal, "ws-1", operation, arguments)
                self.assertEqual(raised.exception.status_code, 422)
                self.assertNotIn("do-not-echo", str(raised.exception.detail))

    async def test_scope_binding_and_live_roles_are_checked_before_canonical_calls(self):
        cases = (
            (_principal(scopes=frozenset({"read"})), "workspace_metadata_update", {"name": "New"}),
            (_principal(workspace_id="other"), "workspace_metadata_update", {"name": "New"}),
        )
        for principal, operation, arguments in cases:
            with self.assertRaises(HTTPException) as raised:
                await execute_management_operation(principal, "ws-1", operation, arguments)
            self.assertEqual(raised.exception.status_code, 403)

        principal = _principal()
        with mock.patch(
            "ragtime.userspace.development_management.userspace_service.enforce_workspace_role",
            mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Editor access required")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await execute_management_operation(principal, "ws-1", "workspace_metadata_update", {"name": "New"})
        self.assertEqual(raised.exception.status_code, 403)

    async def test_legacy_and_workspace_credentials_cannot_select_tools_even_if_owner(self):
        for auth_source in (None, "workspace_credential"):
            principal = _principal(auth_source=auth_source)
            with self.assertRaises(HTTPException) as raised:
                await execute_management_operation(principal, "ws-1", "workspace_tools_set", {"mode": "custom", "selected_tool_ids": []})
            self.assertEqual(raised.exception.status_code, 403)

    async def test_selection_rechecks_current_owner_role_after_write_scope(self):
        principal = _principal()
        with mock.patch(
            "ragtime.userspace.development_management.userspace_service.enforce_workspace_role",
            mock.AsyncMock(side_effect=[SimpleNamespace(owner_user_id="owner"), HTTPException(status_code=403, detail="Owner access required")]),
        ):
            with self.assertRaises(HTTPException) as raised:
                await execute_management_operation(principal, "ws-1", "workspace_tools_set", {"mode": "custom", "selected_tool_ids": []})
        self.assertEqual(raised.exception.status_code, 403)

    async def test_custom_requires_list_and_default_all_forbids_ids(self):
        principal = _principal()
        with mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock()):
            for arguments in (
                {"mode": "custom"},
                {"mode": "custom", "selected_tool_ids": None},
                {"mode": "default_all", "selected_tool_ids": []},
            ):
                with self.assertRaises(HTTPException) as raised:
                    await execute_management_operation(principal, "ws-1", "workspace_tools_set", arguments)
                self.assertEqual(raised.exception.status_code, 422)

    async def test_default_all_does_not_forward_or_report_custom_selection_fields(self):
        principal = _principal()
        workspace = SimpleNamespace(owner_user_id="user-1")
        updated = SimpleNamespace(id="ws-1", tool_selection_mode="default_all", selected_tool_ids=["old-id"])
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock(return_value=workspace)),
            mock.patch("ragtime.userspace.development_management.repository.list_healthy_enabled_tool_ids", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_management.repository.list_tool_configs", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_management.resolve_tool_access", mock.AsyncMock(return_value={})),
            mock.patch("ragtime.userspace.development_management.userspace_service.filter_tool_ids_for_workspace_owner", mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_management.userspace_service.update_workspace", mock.AsyncMock(return_value=updated)) as update,
        ):
            result = await execute_management_operation(principal, "ws-1", "workspace_tools_set", {"mode": "default_all"})

        request = update.await_args.args[1]
        self.assertNotIn("tool_options", request.model_fields_set)
        self.assertNotIn("selected_tool_group_ids", request.model_fields_set)
        self.assertNotIn("selected_tool_ids", request.model_fields_set)
        self.assertNotIn("selected_tool_ids", result)

    async def test_paging_rejects_non_integers_and_advances_by_emitted_items(self):
        principal = _principal(scopes=frozenset({"read"}))
        with mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock()):
            for arguments in ({"offset": True}, {"offset": "1"}, {"limit": 1.0}):
                with self.assertRaises(HTTPException) as raised:
                    await execute_management_operation(principal, "ws-1", "workspace_tools_available", arguments)
                self.assertEqual(raised.exception.status_code, 422)

        huge_description = "x" * (25 * 1024)
        configs = [SimpleNamespace(id="first", name="First", description="ok")]
        configs.extend(SimpleNamespace(id=f"tool-{index}", name=f"Tool {index}", description=huge_description) for index in range(50))
        workspace = SimpleNamespace(owner_user_id="user-1")
        with (
            mock.patch("ragtime.userspace.development_management.userspace_service.enforce_workspace_role", mock.AsyncMock(return_value=workspace)),
            mock.patch(
                "ragtime.userspace.development_management.repository.list_healthy_enabled_tool_ids",
                mock.AsyncMock(return_value=[config.id for config in configs]),
            ),
            mock.patch("ragtime.userspace.development_management.repository.list_tool_configs", mock.AsyncMock(return_value=configs)),
            mock.patch("ragtime.userspace.development_management.resolve_tool_access", mock.AsyncMock(return_value={config.id: "read" for config in configs})),
            mock.patch(
                "ragtime.userspace.development_management.userspace_service.filter_tool_ids_for_workspace_owner",
                mock.AsyncMock(return_value=[config.id for config in configs]),
            ),
        ):
            result = await execute_management_operation(principal, "ws-1", "workspace_tools_available", {"limit": 50})
        self.assertEqual(result["items"][0]["component_id"], "first")
        self.assertTrue(result["items"][1]["description_truncated"])
        self.assertEqual(result["next_offset"], len(result["items"]))
        self.assertLess(result["next_offset"], result["total"])

    def test_registry_scopes_are_canonical(self):
        self.assertEqual(
            {name: scope for name, _description, scope, _schema in MANAGEMENT_OPERATIONS},
            {
                "workspace_metadata_update": "write",
                "env_placeholder_upsert": "write",
                "workspace_tools_available": "read",
                "workspace_tools_set": "write",
            },
        )
