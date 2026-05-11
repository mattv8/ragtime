import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ragtime.core.auth import (  # noqa: E402
    AuthProvider,
    AuthProviderConfigData,
    UserRole,
    _group_entry_rid,
    _ldap_entry_has_group_dn,
    _ldap_profile_from_entry,
    _passes_local_logon_gate,
    hash_local_password,
    resolve_user_effective_role,
    verify_local_password,
)


class InternalAuthProviderTests(unittest.TestCase):
    def test_local_password_hash_round_trips_and_rejects_wrong_password(self) -> None:
        stored = hash_local_password("correct horse battery staple")

        self.assertTrue(verify_local_password("correct horse battery staple", stored))
        self.assertFalse(verify_local_password("wrong password", stored))
        self.assertFalse(verify_local_password("correct horse battery staple", None))

    def test_ldap_profile_projection_keeps_identity_and_group_cache(self) -> None:
        entry = SimpleNamespace(
            entry_dn="CN=Jane Doe,OU=Users,DC=example,DC=com",
            sAMAccountName="jdoe",
            displayName="Jane Doe",
            mail="jane@example.com",
            memberOf=["CN=Engineering,OU=Groups,DC=example,DC=com"],
        )

        profile = _ldap_profile_from_entry(
            user_entry=entry,
            input_username="jane@example.com",
            role=UserRole.admin,
        )

        self.assertEqual(profile.username, "jdoe")
        self.assertEqual(profile.source_provider, "ldap")
        self.assertEqual(profile.source_dn, "CN=Jane Doe,OU=Users,DC=example,DC=com")
        self.assertEqual(profile.display_name, "Jane Doe")
        self.assertEqual(profile.email, "jane@example.com")
        self.assertEqual(profile.role, "admin")
        self.assertEqual(profile.groups, ["CN=Engineering,OU=Groups,DC=example,DC=com"])

    def test_ldap_membership_helper_matches_direct_member_of_case_insensitively(
        self,
    ) -> None:
        entry = SimpleNamespace(
            memberOf=["CN=Engineering,OU=Groups,DC=example,DC=com"],
        )

        self.assertTrue(
            _ldap_entry_has_group_dn(
                user_entry=entry,
                group_dn="cn=engineering,ou=groups,dc=example,dc=com",
            )
        )

    def test_ldap_membership_helper_matches_primary_group_rid(self) -> None:
        entry = SimpleNamespace(memberOf=[], primaryGroupID="513")
        ldap_config = SimpleNamespace()

        with patch("ragtime.core.auth._ldap_group_rid", return_value=513):
            self.assertTrue(
                _ldap_entry_has_group_dn(
                    user_entry=entry,
                    group_dn="CN=Domain Users,DC=example,DC=com",
                    ldap_config=ldap_config,
                    bind_password="secret",
                )
            )

    def test_group_entry_rid_supports_token_and_object_sid(self) -> None:
        self.assertEqual(
            _group_entry_rid(SimpleNamespace(primaryGroupToken="512")), 512
        )
        self.assertEqual(
            _group_entry_rid(
                SimpleNamespace(objectSid=SimpleNamespace(value=b"\x01\x02\x03\x04"))
            ),
            67305985,
        )


class InternalLogonGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_user_passes_when_no_gate_groups_exist(self) -> None:
        class AuthGroupDelegate:
            async def find_many(self, where):
                return []

        db = SimpleNamespace(authgroup=AuthGroupDelegate())

        async def fake_get_db():
            return db

        with patch("ragtime.core.auth.get_db", new=fake_get_db):
            self.assertTrue(
                await _passes_local_logon_gate(SimpleNamespace(id="user-1"))
            )

    async def test_local_user_must_belong_to_one_gate_group(self) -> None:
        class AuthGroupDelegate:
            async def find_many(self, where):
                return [SimpleNamespace(id="gate-1")]

        class MembershipDelegate:
            async def find_many(self, where):
                return [SimpleNamespace(groupId="gate-1")]

        db = SimpleNamespace(
            authgroup=AuthGroupDelegate(),
            authgroupmembership=MembershipDelegate(),
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.auth.get_db", new=fake_get_db):
            self.assertTrue(
                await _passes_local_logon_gate(SimpleNamespace(id="user-1"))
            )

    async def test_local_user_fails_when_outside_gate_groups(self) -> None:
        class AuthGroupDelegate:
            async def find_many(self, where):
                return [SimpleNamespace(id="gate-1")]

        class MembershipDelegate:
            async def find_many(self, where):
                return [SimpleNamespace(groupId="other-1")]

        db = SimpleNamespace(
            authgroup=AuthGroupDelegate(),
            authgroupmembership=MembershipDelegate(),
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.auth.get_db", new=fake_get_db):
            self.assertFalse(
                await _passes_local_logon_gate(SimpleNamespace(id="user-1"))
            )


class InternalRoleResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_group_admin_grant_is_removed_when_membership_is_gone(
        self,
    ) -> None:
        class MembershipDelegate:
            async def find_many(self, where):
                return []

        db = SimpleNamespace(authgroupmembership=MembershipDelegate())

        async def fake_get_db():
            return db

        user = SimpleNamespace(
            id="user-1",
            username="jdoe",
            authProvider=AuthProvider.local_managed,
            role=UserRole.admin,
            roleManuallySet=False,
        )
        config = AuthProviderConfigData(manual_role_override_wins=True)

        with patch("ragtime.core.auth.get_db", new=fake_get_db):
            self.assertEqual(
                await resolve_user_effective_role(user, auth_config=config),
                UserRole.user,
            )

    async def test_local_manual_admin_role_wins_when_configured(self) -> None:
        user = SimpleNamespace(
            id="user-1",
            username="jdoe",
            authProvider=AuthProvider.local_managed,
            role=UserRole.admin,
            roleManuallySet=True,
        )
        config = AuthProviderConfigData(manual_role_override_wins=True)

        self.assertEqual(
            await resolve_user_effective_role(user, auth_config=config),
            UserRole.admin,
        )

    async def test_local_group_admin_grant_is_applied_from_current_membership(
        self,
    ) -> None:
        class MembershipDelegate:
            async def find_many(self, where):
                return [SimpleNamespace(groupId="group-1")]

        class AuthGroupDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(
                    provider=AuthProvider.local_managed,
                    role=UserRole.admin,
                )

        db = SimpleNamespace(
            authgroupmembership=MembershipDelegate(),
            authgroup=AuthGroupDelegate(),
        )

        async def fake_get_db():
            return db

        user = SimpleNamespace(
            id="user-1",
            username="jdoe",
            authProvider=AuthProvider.local_managed,
            role=UserRole.user,
            roleManuallySet=False,
        )
        config = AuthProviderConfigData(manual_role_override_wins=True)

        with patch("ragtime.core.auth.get_db", new=fake_get_db):
            self.assertEqual(
                await resolve_user_effective_role(user, auth_config=config),
                UserRole.admin,
            )


if __name__ == "__main__":
    unittest.main()
