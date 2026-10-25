from __future__ import annotations

import unittest

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_accounts import AccountDevelopmentService, AccountWorkspaceCreateRequest


class DevelopmentAccountRequestTests(unittest.TestCase):
    def test_account_tools_require_positive_user_provenance(self) -> None:
        self.assertTrue(AccountDevelopmentService.can_use_account_tools(DevelopmentPrincipal(user_id="user", is_admin=False, auth_source="user")))
        self.assertFalse(AccountDevelopmentService.can_use_account_tools(DevelopmentPrincipal(user_id="user", is_admin=False)))
        self.assertFalse(
            AccountDevelopmentService.can_use_account_tools(
                DevelopmentPrincipal(user_id="user", is_admin=False, credential_id="credential", workspace_id="workspace", auth_source="workspace_credential")
            )
        )

    def test_create_request_normalizes_explicit_ids_without_losing_empty_selection(self) -> None:
        request = AccountWorkspaceCreateRequest(
            name="  Sandbox  ",
            idempotency_key="create-sandbox-001",
            tool_selection_mode="custom",
            selected_tool_ids=["b", "a", "b"],
            selected_tool_group_ids=[],
        )
        self.assertEqual(request.name, "Sandbox")
        self.assertEqual(request.selected_tool_ids, ["a", "b"])
        self.assertEqual(request.selected_tool_group_ids, [])
        self.assertEqual(request.create_request().selected_tool_group_ids, [])
