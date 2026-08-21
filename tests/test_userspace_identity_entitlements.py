from __future__ import annotations

import sys
import types
import unittest
from contextlib import AbstractAsyncContextManager, ExitStack
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException
from pydantic import ValidationError

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

import ragtime.userspace.models as userspace_models  # noqa: E402
import ragtime.userspace.routes as userspace_routes  # noqa: E402
from ragtime.userspace.service import UserSpaceService  # noqa: E402

_NOW = datetime(2026, 8, 21, tzinfo=timezone.utc)


def _future() -> datetime:
    return _NOW + timedelta(days=365)


def _past() -> datetime:
    return _NOW - timedelta(hours=1)


def _group(*, group_id: str, key: str, provider: str = "ldap", display_name: str | None = None, source_dn: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=group_id,
        key=key,
        displayName=display_name or key,
        provider=provider,
        sourceDn=source_dn,
        sourceId=None,
    )


def _rule(*, workspace_id: str, group: SimpleNamespace, entitlements: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"rule-{group.id}",
        workspaceId=workspace_id,
        authGroupId=group.id,
        entitlements=list(entitlements),
        authGroup=group,
        createdAt=_NOW,
        updatedAt=_NOW,
    )


class _FakeRuleTable:
    def __init__(self, rows: list[SimpleNamespace] | None = None) -> None:
        self.rows = list(rows or [])
        self.delete_many_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []

    async def find_many(self, *, where: dict[str, Any], include: dict[str, Any] | None = None, order: Any | None = None) -> list[SimpleNamespace]:
        _ = include, order
        workspace_id = where.get("workspaceId")
        return [row for row in self.rows if row.workspaceId == workspace_id]

    async def delete_many(self, *, where: dict[str, Any]) -> None:
        self.delete_many_calls.append(where)
        workspace_id = where.get("workspaceId")
        self.rows = [row for row in self.rows if row.workspaceId != workspace_id]

    async def create(self, *, data: dict[str, Any]) -> SimpleNamespace:
        self.create_calls.append(data)
        row = SimpleNamespace(
            id=f"rule-{len(self.create_calls)}",
            workspaceId=data["workspaceId"],
            authGroupId=data["authGroupId"],
            entitlements=list(getattr(data["entitlements"], "data", data["entitlements"])),
            authGroup=None,
            createdAt=_NOW,
            updatedAt=_NOW,
        )
        self.rows.append(row)
        return row


class _FakeAuthGroupTable:
    def __init__(self, groups: list[SimpleNamespace]) -> None:
        self.by_id = {group.id: group for group in groups}

    async def find_many(self, *, where: dict[str, Any], order: Any | None = None) -> list[SimpleNamespace]:
        _ = order
        ids = (((where.get("id") or {}).get("in")) or []) if where else []
        if ids:
            return [self.by_id[group_id] for group_id in ids if group_id in self.by_id]
        return [self.by_id[group_id] for group_id in sorted(self.by_id)]

    async def find_unique(self, *, where: dict[str, Any]) -> SimpleNamespace | None:
        group_id = where.get("id")
        if group_id is not None:
            return self.by_id.get(group_id)
        return None


class _FakeMembershipTable:
    def __init__(self, memberships: dict[tuple[str, str], SimpleNamespace] | None = None) -> None:
        self.memberships = dict(memberships or {})

    async def find_unique(self, *, where: dict[str, Any]) -> SimpleNamespace | None:
        key = where.get("userId_groupId") or {}
        return self.memberships.get((key.get("userId"), key.get("groupId")))


class _FakeTx(AbstractAsyncContextManager[Any]):
    def __init__(self, db: Any) -> None:
        self._db = db

    async def __aenter__(self) -> Any:
        return self._db

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _AccessControlledService(UserSpaceService):
    def __init__(self, *, allowed_owner_ids: set[str] | None = None) -> None:
        super().__init__()
        self.allowed_owner_ids = allowed_owner_ids or {"owner-1"}

    async def _enforce_workspace_access(
        self,
        workspace_id: str,
        user_id: str,
        required_role: str | None = None,
        is_admin: bool = False,
    ) -> Any:
        _ = workspace_id, required_role
        if is_admin or user_id in self.allowed_owner_ids:
            return SimpleNamespace(id="workspace-1", owner_user_id="owner-1", ownerUserId="owner-1")
        raise HTTPException(status_code=403, detail="Owner access required")


def _make_db(
    *,
    groups: list[SimpleNamespace],
    rules: list[SimpleNamespace] | None = None,
    memberships: dict[tuple[str, str], SimpleNamespace] | None = None,
) -> Any:
    db = SimpleNamespace(
        workspaceidentityentitlementrule=_FakeRuleTable(rules),
        authgroup=_FakeAuthGroupTable(groups),
        authgroupmembership=_FakeMembershipTable(memberships),
    )
    db.tx = lambda: _FakeTx(db)
    return db


class WorkspaceIdentityEntitlementContractTests(unittest.TestCase):
    def test_models_and_service_contract_exist(self) -> None:
        self.assertTrue(hasattr(userspace_models, "UserSpaceIdentityEntitlementRuleInput"))
        self.assertTrue(hasattr(userspace_models, "UserSpaceIdentityEntitlementRule"))
        self.assertTrue(hasattr(userspace_models, "UserSpaceIdentityEntitlementPolicyResponse"))
        self.assertTrue(hasattr(userspace_models, "ReplaceUserSpaceIdentityEntitlementPolicyRequest"))
        self.assertTrue(hasattr(UserSpaceService, "get_workspace_identity_entitlement_policy"))
        self.assertTrue(hasattr(UserSpaceService, "replace_workspace_identity_entitlement_policy"))
        self.assertTrue(hasattr(UserSpaceService, "resolve_workspace_identity_entitlements"))
        self.assertTrue(hasattr(userspace_routes, "get_workspace_identity_entitlements"))
        self.assertTrue(hasattr(userspace_routes, "replace_workspace_identity_entitlements"))


class WorkspaceIdentityEntitlementServiceTests(unittest.IsolatedAsyncioTestCase):
    def _request_model(self) -> Any:
        return getattr(userspace_models, "ReplaceUserSpaceIdentityEntitlementPolicyRequest")

    def _response_model(self) -> Any:
        return getattr(userspace_models, "UserSpaceIdentityEntitlementPolicyResponse")

    async def test_replace_normalizes_tokens_and_sanitizes_group_output(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", display_name="Accounting", source_dn="CN=Accounting,DC=example")
        db = _make_db(groups=[accounting])
        request_model = self._request_model()
        request = request_model(rules=[{"auth_group_id": "group-1", "entitlements": ["recon.role.reviewer", "recon.role.preparer", "recon.role.preparer"]}])

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.replace_workspace_identity_entitlement_policy("workspace-1", "owner-1", request)

        self.assertEqual(result.rules[0].entitlements, ["recon.role.preparer", "recon.role.reviewer"])
        self.assertEqual(result.rules[0].auth_group.id, "group-1")
        self.assertEqual(result.rules[0].auth_group.display_name, "Accounting")
        self.assertFalse(hasattr(result.rules[0].auth_group, "source_dn"))

    async def test_replace_rejects_duplicate_group_rules(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting")
        db = _make_db(groups=[accounting])
        request = self._request_model()(
            rules=[
                {"auth_group_id": "group-1", "entitlements": ["recon.admin"]},
                {"auth_group_id": "group-1", "entitlements": ["recon.role.preparer"]},
            ]
        )

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await service.replace_workspace_identity_entitlement_policy("workspace-1", "owner-1", request)

        self.assertEqual(raised.exception.status_code, 400)

    async def test_replace_rejects_invalid_empty_and_oversized_payloads(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting")
        db = _make_db(groups=[accounting])

        with self.assertRaises(ValidationError):
            self._request_model()(rules=[{"auth_group_id": "group-1", "entitlements": []}])

        invalid_requests = [
            self._request_model()(rules=[{"auth_group_id": "group-1", "entitlements": ["Recon.Admin"]}]),
            self._request_model()(rules=[{"auth_group_id": "group-1", "entitlements": [f"r{'a' * 4096}"]}]),
        ]

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            for request in invalid_requests:
                with self.subTest(request=request.model_dump()):
                    with self.assertRaises(HTTPException) as raised:
                        await service.replace_workspace_identity_entitlement_policy("workspace-1", "owner-1", request)
                    self.assertEqual(raised.exception.status_code, 400)

    async def test_replace_is_atomic_when_validation_fails(self) -> None:
        service = _AccessControlledService()
        existing_group = _group(group_id="group-1", key="accounting")
        existing_rule = _rule(workspace_id="workspace-1", group=existing_group, entitlements=["recon.role.preparer"])
        db = _make_db(groups=[existing_group], rules=[existing_rule])
        request = self._request_model()(rules=[{"auth_group_id": "missing-group", "entitlements": ["recon.admin"]}])

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException):
                await service.replace_workspace_identity_entitlement_policy("workspace-1", "owner-1", request)

        self.assertEqual([row.authGroupId for row in db.workspaceidentityentitlementrule.rows], ["group-1"])
        self.assertEqual(db.workspaceidentityentitlementrule.delete_many_calls, [])

    async def test_get_policy_denies_editor_and_allows_owner_with_available_groups(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", display_name="Accounting")
        db = _make_db(groups=[accounting], rules=[_rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.role.preparer"])])

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await service.get_workspace_identity_entitlement_policy("workspace-1", "editor-1")
            self.assertEqual(raised.exception.status_code, 403)

            result = await service.get_workspace_identity_entitlement_policy(
                "workspace-1",
                "owner-1",
                include_available_groups=True,
            )

        self.assertTrue(result.can_configure)
        self.assertEqual([group.id for group in result.available_auth_groups], ["group-1"])

    async def test_resolve_local_admin_without_exact_membership_gets_no_entitlements(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", provider="local_managed")
        db = _make_db(groups=[accounting], rules=[_rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.admin"])])
        user = SimpleNamespace(id="admin-1", role="admin", authProvider="local", cachedGroups=[], sourceExpiresAt=None, ldapDn=None)

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, [])

    async def test_resolve_uses_fresh_cached_membership_exactly(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", provider="ldap", source_dn="CN=Accounting,DC=example")
        db = _make_db(groups=[accounting], rules=[_rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.role.preparer"])])
        user = SimpleNamespace(
            id="user-1",
            role="user",
            authProvider="ldap",
            cachedGroups=["accounting"],
            sourceExpiresAt=_future(),
            ldapDn="CN=User,DC=example",
        )

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, ["recon.role.preparer"])

    async def test_resolve_uses_ldap_fallback_when_cached_membership_is_stale(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", provider="ldap", source_dn="CN=Accounting,DC=example")
        db = _make_db(groups=[accounting], rules=[_rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.admin"])])
        user = SimpleNamespace(
            id="user-1",
            role="user",
            authProvider="ldap",
            cachedGroups=["accounting"],
            sourceExpiresAt=_past(),
            ldapDn="CN=User,DC=example",
        )

        with (
            mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.userspace.service.ldap_user_is_member_of_group_strict", new=mock.AsyncMock(return_value=True)),
        ):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, ["recon.admin"])

    async def test_resolve_fails_closed_when_ldap_fallback_errors(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", provider="ldap", source_dn="CN=Accounting,DC=example")
        db = _make_db(groups=[accounting], rules=[_rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.admin"])])
        user = SimpleNamespace(
            id="user-1",
            role="user",
            authProvider="ldap",
            cachedGroups=[],
            sourceExpiresAt=_past(),
            ldapDn="CN=User,DC=example",
        )

        with (
            mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch(
                "ragtime.userspace.service.ldap_user_is_member_of_group_strict",
                new=mock.AsyncMock(side_effect=RuntimeError("ldap down")),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(raised.exception.status_code, 503)

    async def test_resolve_grants_local_managed_membership_with_null_expiry(self) -> None:
        service = _AccessControlledService()
        ops = _group(group_id="group-1", key="ops", provider="local_managed")
        db = _make_db(
            groups=[ops],
            rules=[_rule(workspace_id="workspace-1", group=ops, entitlements=["recon.admin"])],
            memberships={("user-1", "group-1"): SimpleNamespace(expiresAt=None)},
        )
        user = SimpleNamespace(id="user-1", role="user", authProvider="local", cachedGroups=[], sourceExpiresAt=None, ldapDn=None)

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, ["recon.admin"])

    async def test_resolve_unions_entitlements_across_multiple_matching_rules(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting")
        executive = _group(group_id="group-2", key="executive")
        db = _make_db(
            groups=[accounting, executive],
            rules=[
                _rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.role.reviewer", "recon.role.preparer"]),
                _rule(workspace_id="workspace-1", group=executive, entitlements=["recon.admin"]),
            ],
            memberships={
                ("user-1", "group-1"): SimpleNamespace(expiresAt=None),
                ("user-1", "group-2"): SimpleNamespace(expiresAt=None),
            },
        )
        user = SimpleNamespace(id="user-1", role="user", authProvider="ldap", cachedGroups=[], sourceExpiresAt=None, ldapDn=None)

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, ["recon.admin", "recon.role.preparer", "recon.role.reviewer"])

    async def test_resolve_rejects_oversized_entitlement_header(self) -> None:
        service = _AccessControlledService()
        groups = [_group(group_id=f"group-{index}", key=f"group-{index}") for index in range(3)]
        rules = [
            _rule(
                workspace_id="workspace-1",
                group=group,
                entitlements=[f"recon.role.load{rule_index:02d}{token_index:02d}." + "p" * 40 for token_index in range(32)],
            )
            for rule_index, group in enumerate(groups)
        ]
        db = _make_db(
            groups=groups,
            rules=rules,
            memberships={("user-1", group.id): SimpleNamespace(expiresAt=None) for group in groups},
        )
        user = SimpleNamespace(id="user-1", role="user", authProvider="ldap", cachedGroups=[], sourceExpiresAt=None, ldapDn=None)

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(raised.exception.status_code, 503)

    async def test_resolve_ignores_deleted_group_rule_records(self) -> None:
        service = _AccessControlledService()
        orphan_group = _group(group_id="group-deleted", key="deleted")
        orphan_rule = _rule(workspace_id="workspace-1", group=orphan_group, entitlements=["recon.admin"])
        db = _make_db(groups=[], rules=[orphan_rule])
        user = SimpleNamespace(id="user-1", role="user", authProvider="ldap", cachedGroups=["deleted"], sourceExpiresAt=_future(), ldapDn=None)

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            result = await service.resolve_workspace_identity_entitlements("workspace-1", user)

        self.assertEqual(result, [])

    async def test_read_paths_skip_rules_with_corrupt_stored_tokens(self) -> None:
        service = _AccessControlledService()
        accounting = _group(group_id="group-1", key="accounting", display_name="Accounting")
        corrupt_rule = _rule(workspace_id="workspace-1", group=accounting, entitlements=["Invalid Token!"])
        valid_rule = _rule(workspace_id="workspace-1", group=accounting, entitlements=["recon.admin"])
        corrupt_rule.id = "rule-corrupt"
        valid_rule.id = "rule-valid"
        db = _make_db(groups=[accounting], rules=[corrupt_rule, valid_rule])
        db.workspaceidentityentitlementrule.rows = [corrupt_rule, valid_rule]

        with mock.patch("ragtime.userspace.service.get_db", new=mock.AsyncMock(return_value=db)):
            policy = await service.get_workspace_identity_entitlement_policy("workspace-1", "owner-1")

        self.assertEqual([rule.entitlements for rule in policy.rules], [["recon.admin"]])


class WorkspaceIdentityEntitlementRouteTests(unittest.IsolatedAsyncioTestCase):
    def _policy_response(self) -> Any:
        response_model = getattr(userspace_models, "UserSpaceIdentityEntitlementPolicyResponse")
        rule_model = getattr(userspace_models, "UserSpaceIdentityEntitlementRule")
        group_model = getattr(userspace_models, "UserSpaceIdentityEntitlementAuthGroup")
        return response_model(
            workspace_id="workspace-1",
            can_configure=True,
            rules=[
                rule_model(
                    auth_group_id="group-1",
                    entitlements=["recon.admin"],
                    auth_group=group_model(id="group-1", key="accounting", display_name="Accounting", provider="ldap"),
                )
            ],
            available_auth_groups=[],
        )

    async def test_get_route_passes_admin_flag(self) -> None:
        fake_user = SimpleNamespace(id="admin-1", role="admin")
        response = self._policy_response()

        with mock.patch.object(
            userspace_routes.userspace_service,
            "get_workspace_identity_entitlement_policy",
            new=mock.AsyncMock(return_value=response),
        ) as get_policy:
            result = await userspace_routes.get_workspace_identity_entitlements("workspace-1", fake_user)

        self.assertEqual(result, response)
        self.assertTrue(get_policy.await_args.kwargs["is_admin"])
        self.assertTrue(get_policy.await_args.kwargs["include_available_groups"])

    async def test_put_route_passes_admin_flag(self) -> None:
        fake_user = SimpleNamespace(id="admin-1", role="admin")
        response = self._policy_response()
        request_model = getattr(userspace_models, "ReplaceUserSpaceIdentityEntitlementPolicyRequest")
        request = request_model(rules=[])

        with mock.patch.object(
            userspace_routes.userspace_service,
            "replace_workspace_identity_entitlement_policy",
            new=mock.AsyncMock(return_value=response),
        ) as replace_policy:
            result = await userspace_routes.replace_workspace_identity_entitlements("workspace-1", request, fake_user)

        self.assertEqual(result, response)
        self.assertTrue(replace_policy.await_args.kwargs["is_admin"])

    async def test_routes_propagate_owner_denial_from_service(self) -> None:
        fake_user = SimpleNamespace(id="editor-1", role="user")
        request_model = getattr(userspace_models, "ReplaceUserSpaceIdentityEntitlementPolicyRequest")

        with mock.patch.object(
            userspace_routes.userspace_service,
            "get_workspace_identity_entitlement_policy",
            new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Owner access required")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await userspace_routes.get_workspace_identity_entitlements("workspace-1", fake_user)
        self.assertEqual(raised.exception.status_code, 403)

        with mock.patch.object(
            userspace_routes.userspace_service,
            "replace_workspace_identity_entitlement_policy",
            new=mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Owner access required")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await userspace_routes.replace_workspace_identity_entitlements("workspace-1", request_model(rules=[]), fake_user)
        self.assertEqual(raised.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
