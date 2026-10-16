import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.api import auth
from ragtime.content_protection import store
from ragtime.content_protection.conflicts import find_group_policy_conflicts
from ragtime.content_protection.models import ContentProtectionConfig, Requirement


class ContentProtectionGroupConflictTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_conflict_locks_before_reads_and_makes_no_membership_writes(self) -> None:
        events: list[str] = []
        config = ContentProtectionConfig(
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
            ]
        )
        user = SimpleNamespace(id="target", username="target", displayName="Target", disabled=False)
        groups = [
            SimpleNamespace(id="skip", provider=AuthProvider.local_managed),
            SimpleNamespace(id="require", provider=AuthProvider.local_managed),
        ]

        class Transaction:
            contentprotectionconfig = SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(config=config.model_dump(mode="json"))))
            authgroupmembership = SimpleNamespace(find_many=mock.AsyncMock(return_value=[]), delete=mock.AsyncMock(), upsert=mock.AsyncMock())

            async def execute_raw(self, _query: str) -> None:
                events.append("lock")

        tx = Transaction()

        class Context:
            async def __aenter__(self):
                return tx

            async def __aexit__(self, *_args):
                return None

        db = SimpleNamespace(
            user=SimpleNamespace(find_unique=mock.AsyncMock(return_value=user)),
            authgroup=SimpleNamespace(find_many=mock.AsyncMock(return_value=groups)),
            tx=lambda: Context(),
        )
        admin = User(
            id="admin",
            username="admin",
            displayName="Admin",
            email="admin@example.com",
            role=UserRole.admin,
            authProvider=AuthProvider.local_managed,
            themePack=None,
            roleManuallySet=False,
            sourceProvider=None,
            sourceSyncedAt=None,
            sourceExpiresAt=None,
            cachedGroups=cast(Json, "[]"),
            chatEnabled=None,
            userspaceGenerationEnabled=None,
            createdAt=datetime.now(UTC),
            updatedAt=datetime.now(UTC),
            securityGeneration=0,
        )
        with mock.patch.object(auth, "get_db", new=mock.AsyncMock(return_value=db)):
            with self.assertRaises(HTTPException) as raised:
                await auth.set_user_groups("target", auth.SetUserGroupsRequest(group_ids=["skip", "require"]), admin)

        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(events, ["lock"])
        tx.authgroupmembership.delete.assert_not_awaited()
        tx.authgroupmembership.upsert.assert_not_awaited()

    async def test_config_conflict_checks_after_lock_and_cas_without_writing(self) -> None:
        events: list[str] = []

        class Transaction:
            contentprotectionconfig = SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(revision=4)))
            user = SimpleNamespace(find_many=mock.AsyncMock(return_value=[SimpleNamespace(id="u", username="user", displayName="User", disabled=False)]))
            authgroupmembership = SimpleNamespace(
                find_many=mock.AsyncMock(
                    return_value=[
                        SimpleNamespace(userId="u", groupId="skip", expiresAt=None),
                        SimpleNamespace(userId="u", groupId="require", expiresAt=None),
                    ]
                )
            )
            contentprotectionconfig.update_many = mock.AsyncMock()

            async def execute_raw(self, _query: str) -> None:
                events.append("lock")

        tx = Transaction()

        class Context:
            async def __aenter__(self):
                return tx

            async def __aexit__(self, *_args):
                return None

        db = SimpleNamespace(tx=lambda: Context())
        config = ContentProtectionConfig(
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
            ]
        )
        with (
            mock.patch.object(store, "get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch.object(store, "validate_references", new=mock.AsyncMock()),
        ):
            with self.assertRaises(Exception) as raised:
                await store.save_config_record(config, 4, "admin")

        self.assertEqual(getattr(raised.exception, "code", None), "conflicting_group_policies")
        self.assertEqual(events, ["lock"])
        tx.contentprotectionconfig.update_many.assert_not_awaited()

    async def test_returns_only_active_users_with_unexpired_skip_and_require_memberships(self) -> None:
        config = ContentProtectionConfig(
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
            ]
        )
        users = [
            SimpleNamespace(id="active", username="a-user", displayName="Active", disabled=False),
            SimpleNamespace(id="disabled", username="b-user", displayName="Disabled", disabled=True),
            SimpleNamespace(id="expired", username="c-user", displayName="Expired", disabled=False),
        ]
        memberships = [
            SimpleNamespace(userId="active", groupId="skip", expiresAt=None),
            SimpleNamespace(userId="active", groupId="require", expiresAt=None),
            SimpleNamespace(userId="disabled", groupId="skip", expiresAt=None),
            SimpleNamespace(userId="disabled", groupId="require", expiresAt=None),
            SimpleNamespace(userId="expired", groupId="skip", expiresAt=None),
            SimpleNamespace(userId="expired", groupId="require", expiresAt=datetime.now(UTC) - timedelta(seconds=1)),
        ]
        db = SimpleNamespace(
            user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
            authgroupmembership=SimpleNamespace(find_many=mock.AsyncMock(return_value=memberships)),
        )

        conflicts, total, conflicted_group_ids = await find_group_policy_conflicts(config, db)

        self.assertEqual(total, 1)
        self.assertEqual(conflicted_group_ids, ["require", "skip"])
        self.assertEqual(
            conflicts,
            [{"user_id": "active", "username": "a-user", "display_name": "Active", "skip_group_ids": ["skip"], "require_group_ids": ["require"]}],
        )

    async def test_caps_conflict_preview_at_fifty_rows(self) -> None:
        config = ContentProtectionConfig(
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
            ]
        )
        users = [SimpleNamespace(id=str(index), username=f"user-{index:03}", displayName=None, disabled=False) for index in range(51)]
        memberships = [
            membership
            for user in users
            for membership in (
                SimpleNamespace(userId=user.id, groupId="skip", expiresAt=None),
                SimpleNamespace(userId=user.id, groupId="require", expiresAt=None),
            )
        ]
        db = SimpleNamespace(
            user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
            authgroupmembership=SimpleNamespace(find_many=mock.AsyncMock(return_value=memberships)),
        )

        conflicts, total, conflicted_group_ids = await find_group_policy_conflicts(config, db)

        self.assertEqual(total, 51)
        self.assertEqual(len(conflicts), 50)
        self.assertEqual(conflicts[0]["username"], "user-000")
        self.assertEqual(conflicted_group_ids, ["require", "skip"])

    async def test_filter_selects_conflict_beyond_global_preview_and_summarizes_later_group(self) -> None:
        config = ContentProtectionConfig(
            requirements=[
                Requirement(scope_kind="group", scope_key="skip", mode="skip"),
                Requirement(scope_kind="group", scope_key="require", mode="require"),
                Requirement(scope_kind="group", scope_key="late-require", mode="require"),
            ]
        )
        users = [SimpleNamespace(id=str(index), username=f"user-{index:03}", displayName=None, disabled=False) for index in range(51)]
        memberships = [
            membership
            for user in users
            for membership in (
                SimpleNamespace(userId=user.id, groupId="skip", expiresAt=None),
                SimpleNamespace(userId=user.id, groupId="late-require" if user.id == "50" else "require", expiresAt=None),
            )
        ]
        db = SimpleNamespace(
            user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
            authgroupmembership=SimpleNamespace(find_many=mock.AsyncMock(return_value=memberships)),
        )

        conflicts, total, group_ids = await find_group_policy_conflicts(config, db)
        selected, selected_total, selected_group_ids = await find_group_policy_conflicts(config, db, "50")
        missing, missing_total, missing_group_ids = await find_group_policy_conflicts(config, db, "missing")

        self.assertEqual(total, 51)
        self.assertEqual(len(conflicts), 50)
        self.assertEqual(group_ids, ["late-require", "require", "skip"])
        self.assertEqual(selected_total, 1)
        self.assertEqual(selected[0]["user_id"], "50")
        self.assertEqual(selected_group_ids, ["late-require", "skip"])
        self.assertEqual((missing, missing_total, missing_group_ids), ([], 0, []))
