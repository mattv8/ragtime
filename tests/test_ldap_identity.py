import unittest
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import cast
from unittest import mock
from uuid import UUID

from fastapi import HTTPException
from ldap3.core.exceptions import LDAPException
from prisma.enums import AuthProvider
from prisma.models import User

from ragtime.api import auth as auth_api
from ragtime.core import auth


class _FakeUserTable:
    def __init__(self, *, username_matches: list[object], identity_match: object | None = None) -> None:
        self.username_matches = username_matches
        self.identity_match = identity_match
        self.create = mock.AsyncMock()
        self.update = mock.AsyncMock()
        self.update_many = mock.AsyncMock(return_value=1)
        self.find_many: Callable[..., Awaitable[list[object]]] = self._find_many
        self.find_unique: Callable[..., Awaitable[object | None]] = self._find_unique

    async def _find_many(self, *, where: dict) -> list[object]:
        return self.username_matches

    async def _find_unique(self, *, where: dict) -> object | None:
        if "ldapIdentityKey" in where:
            return self.identity_match
        return None


class LdapIdentityTests(unittest.IsolatedAsyncioTestCase):
    def _profile(self, **overrides: object) -> auth.AuthUserProfile:
        values: dict[str, object] = {
            "username": "matt",
            "source_provider": "ldap",
            "source_id": "uid=matt,dc=example,dc=com",
            "source_dn": "uid=matt,dc=example,dc=com",
            "ldap_identity_key": "entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c00",
        }
        values.update(overrides)
        return auth.AuthUserProfile.model_validate(values)

    def test_extracts_canonical_entryuuid_and_ad_objectguid_variants(self) -> None:
        entry_uuid = SimpleNamespace(
            entryUUID=SimpleNamespace(raw_values=[b"6FCBA6A6-5C5C-103C-8C0C-DD0077222C00"]),
        )
        self.assertEqual(
            auth._ldap_entry_identity_key(entry_uuid),
            "entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c00",
        )

        guid = UUID("00112233-4455-6677-8899-aabbccddeeff")
        raw_ad_guid = SimpleNamespace(objectGUID=SimpleNamespace(raw_values=[guid.bytes_le]))
        self.assertEqual(auth._ldap_entry_identity_key(raw_ad_guid), f"objectguid:{guid}")

        formatted_ad_guid = SimpleNamespace(objectGUID="00112233-4455-6677-8899-AABBCCDDEEFF")
        self.assertEqual(auth._ldap_entry_identity_key(formatted_ad_guid), f"objectguid:{guid}")

    async def test_immutable_identity_keeps_user_across_username_and_dn_rename(self) -> None:
        user = SimpleNamespace(
            id="original-user",
            username="Matt",
            authProvider=AuthProvider.ldap,
            ldapDn="uid=Matt,dc=old,dc=example",
            ldapIdentityKey="entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c00",
        )
        table = _FakeUserTable(username_matches=[user], identity_match=user)

        resolved = await auth._resolve_ldap_profile_user(self._profile(), db=SimpleNamespace(user=table))

        self.assertIs(resolved, user)

    async def test_legacy_attachment_requires_casefolded_username_and_matching_dn(self) -> None:
        legacy = SimpleNamespace(
            id="legacy-user",
            username="Matt",
            authProvider=AuthProvider.ldap,
            ldapDn="UID=MATT,DC=EXAMPLE,DC=COM",
            ldapIdentityKey=None,
        )
        table = _FakeUserTable(username_matches=[legacy])
        self.assertIs(
            await auth._resolve_ldap_profile_user(self._profile(), db=SimpleNamespace(user=table)),
            legacy,
        )

        legacy.ldapDn = "uid=matt,dc=other,dc=example"
        with self.assertRaises(auth.LdapIdentityResolutionError):
            await auth._resolve_ldap_profile_user(self._profile(), db=SimpleNamespace(user=table))

    async def test_rejects_local_and_immutable_identity_collisions(self) -> None:
        local = SimpleNamespace(
            id="local-user",
            username="MATT",
            authProvider=AuthProvider.local_managed,
            ldapDn=None,
            ldapIdentityKey=None,
        )
        with self.assertRaises(auth.LdapIdentityResolutionError):
            await auth._resolve_ldap_profile_user(self._profile(), db=SimpleNamespace(user=_FakeUserTable(username_matches=[local])))

        conflicting = SimpleNamespace(
            id="different-user",
            username="matt",
            authProvider=AuthProvider.ldap,
            ldapDn="uid=matt,dc=example,dc=com",
            ldapIdentityKey="entryuuid:d2719b73-29bc-4a99-8d63-505dc4b1dccf",
        )
        with self.assertRaises(auth.LdapIdentityResolutionError):
            await auth._resolve_ldap_profile_user(self._profile(), db=SimpleNamespace(user=_FakeUserTable(username_matches=[conflicting])))

    async def test_non_lazy_login_attaches_key_and_updates_renamed_username(self) -> None:
        legacy = SimpleNamespace(
            id="legacy-user",
            username="Matt",
            authProvider=AuthProvider.ldap,
            ldapDn="uid=matt,dc=example,dc=com",
            ldapIdentityKey=None,
        )
        table = _FakeUserTable(username_matches=[legacy])
        table.update.return_value = legacy
        with mock.patch.object(auth, "get_db", mock.AsyncMock(return_value=SimpleNamespace(user=table))):
            user = await auth._mark_existing_ldap_login(self._profile())

        self.assertIs(user, legacy)
        update_call = table.update.await_args
        assert update_call is not None
        update_data = update_call.kwargs["data"]
        self.assertEqual(update_data["username"], "matt")
        self.assertEqual(
            update_data["ldapIdentityKey"],
            "entryuuid:6fcba6a6-5c5c-103c-8c0c-dd0077222c00",
        )

    async def test_legacy_key_claim_re_resolves_when_another_login_claimed_it(self) -> None:
        legacy = SimpleNamespace(
            id="legacy-user",
            username="Matt",
            authProvider=AuthProvider.ldap,
            ldapDn="uid=matt,dc=example,dc=com",
            ldapIdentityKey=None,
        )
        replaced = SimpleNamespace(**{**legacy.__dict__, "ldapIdentityKey": "entryuuid:d2719b73-29bc-4a99-8d63-505dc4b1dccf"})
        table = _FakeUserTable(username_matches=[legacy])
        table.update_many.return_value = 0
        calls = 0

        async def find_many(*, where: dict) -> list[object]:
            nonlocal calls
            calls += 1
            return [legacy if calls == 1 else replaced]

        table.find_many = find_many
        with (
            mock.patch.object(auth, "get_db", mock.AsyncMock(return_value=SimpleNamespace(user=table))),
            self.assertRaises(auth.LdapIdentityResolutionError),
        ):
            await auth._mark_existing_ldap_login(self._profile())

    def test_strict_ldap_schema_retries_without_operational_attribute_request(self) -> None:
        entry = SimpleNamespace(entry_dn="uid=matt,dc=example,dc=com")

        class StrictConnection:
            def __init__(self) -> None:
                self.entries: list[object] = []
                self.attributes: list[list[str]] = []

            def search(self, **kwargs: object) -> bool:
                attributes = cast(list[str], kwargs["attributes"])
                self.attributes.append(attributes)
                if "+" in attributes:
                    raise LDAPException("invalid attribute type")
                self.entries = [entry]
                return True

        conn = StrictConnection()
        found = auth._search_first_matching_entry(
            conn=cast(auth.Connection, conn),
            search_base="dc=example,dc=com",
            search_filters=["(uid=matt)"],
            attributes=auth._get_user_entry_search_attributes(),
            context="test",
        )

        self.assertIs(found, entry)
        self.assertEqual(conn.attributes, [["*", "+", "memberOf"], ["*", "memberOf"]])

    async def test_typeahead_uses_operational_attribute_fallback(self) -> None:
        entry = SimpleNamespace(entry_dn="uid=matt,dc=example,dc=com", uid="matt", memberOf=[])

        class StrictConnection:
            bound = False

            def __init__(self) -> None:
                self.entries: list[object] = []
                self.attributes: list[list[str]] = []

            def search(self, **kwargs: object) -> bool:
                attributes = cast(list[str], kwargs["attributes"])
                self.attributes.append(attributes)
                if "+" in attributes:
                    raise LDAPException("undefined attribute type")
                self.entries = [entry]
                return True

            def unbind(self) -> None:
                return None

        conn = StrictConnection()
        config = SimpleNamespace(
            serverUrl="ldap://example.test",
            bindDn="cn=service",
            bindPassword="secret",
            allowSelfSigned=False,
            userSearchBase="dc=example,dc=com",
            baseDn=None,
        )
        with (
            mock.patch.object(auth, "get_ldap_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(auth, "decrypt_secret", return_value="secret"),
            mock.patch.object(auth, "_get_ldap_connection", return_value=conn),
            mock.patch.object(auth, "_determine_ldap_role_for_entry", return_value=auth.UserRole.user),
        ):
            profiles = await auth.search_ldap_user_profiles("matt")

        self.assertEqual([profile.username for profile in profiles], ["matt"])
        self.assertEqual(conn.attributes[:2], [["*", "+", "memberOf"], ["*", "memberOf"]])

    async def test_lazy_upsert_claims_legacy_key_once_then_syncs_full_profile(self) -> None:
        legacy = SimpleNamespace(
            id="legacy-user",
            username="Matt",
            authProvider=AuthProvider.ldap,
            ldapDn="uid=matt,dc=example,dc=com",
            ldapIdentityKey=None,
            role=auth.UserRole.user,
            roleManuallySet=False,
        )
        table = _FakeUserTable(username_matches=[legacy])
        table.update.return_value = legacy
        db = SimpleNamespace(user=table)
        profile = self._profile(display_name="Matthew", email="matt@example.com", groups=["cn=staff"])
        config = auth.AuthProviderConfigData()
        resolution_queries = 0

        async def find_many(*, where: dict) -> list[object]:
            nonlocal resolution_queries
            resolution_queries += 1
            return [legacy]

        table.find_many = find_many
        with (
            mock.patch.object(auth, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(auth, "get_auth_provider_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(auth, "_apply_local_group_role", mock.AsyncMock(return_value=auth.UserRole.user)),
            mock.patch.object(auth, "_sync_user_auth_groups", mock.AsyncMock()),
            mock.patch.object(auth, "_record_auth_sync_event", mock.AsyncMock()),
        ):
            user = await auth._upsert_provider_user_profile(profile, provider=AuthProvider.ldap)

        self.assertIs(user, legacy)
        self.assertEqual(resolution_queries, 1)
        self.assertEqual(table.update_many.await_count, 1)
        self.assertEqual(table.update.await_count, 1)
        update_call = table.update.await_args
        assert update_call is not None
        update_data = update_call.kwargs["data"]
        self.assertEqual(update_data["ldapIdentityKey"], profile.ldap_identity_key)
        self.assertEqual(update_data["displayName"], "Matthew")
        self.assertEqual(update_data["email"], "matt@example.com")

    async def test_import_identity_conflict_is_an_http_conflict(self) -> None:
        with mock.patch.object(
            auth_api,
            "import_ldap_user_profile",
            mock.AsyncMock(side_effect=auth.LdapIdentityResolutionError("collision")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await auth_api.import_ldap_user(
                    auth_api.LdapUserSearchRequest(username="matt"),
                    _user=cast(User, SimpleNamespace()),
                )

        self.assertEqual(raised.exception.status_code, 409)

    async def test_upsert_retries_unique_conflict_and_reuses_racing_ldap_row(self) -> None:
        profile = self._profile()
        winner = SimpleNamespace(
            id="racing-user",
            username="matt",
            authProvider=AuthProvider.ldap,
            ldapDn=profile.source_dn,
            ldapIdentityKey=profile.ldap_identity_key,
            role=auth.UserRole.user,
            roleManuallySet=False,
        )
        table = _FakeUserTable(username_matches=[])
        table.create.side_effect = _DuplicateIdentity()
        table.update.return_value = winner
        db = SimpleNamespace(user=table)

        async def find_many(*, where: dict) -> list[object]:
            return [] if table.create.await_count == 0 else [winner]

        async def find_unique(*, where: dict) -> object | None:
            return winner if "ldapIdentityKey" in where and table.create.await_count else None

        table.find_many = find_many
        table.find_unique = find_unique
        config = auth.AuthProviderConfigData()
        with (
            mock.patch.object(auth, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(auth, "get_auth_provider_config", mock.AsyncMock(return_value=config)),
            mock.patch.object(auth, "UniqueViolationError", _DuplicateIdentity),
            mock.patch.object(auth, "_apply_local_group_role", mock.AsyncMock(return_value=auth.UserRole.user)),
            mock.patch.object(auth, "_sync_user_auth_groups", mock.AsyncMock()),
            mock.patch.object(auth, "_record_auth_sync_event", mock.AsyncMock()),
        ):
            user = await auth._upsert_provider_user_profile(profile, provider=AuthProvider.ldap)

        self.assertIs(user, winner)
        self.assertEqual(table.create.await_count, 1)
        self.assertEqual(table.update.await_count, 1)


class _DuplicateIdentity(Exception):
    pass


if __name__ == "__main__":
    unittest.main()
