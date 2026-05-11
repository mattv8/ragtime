import unittest
from types import SimpleNamespace

from ragtime.core.auth import (
    UserRole,
    _build_default_user_search_filters,
    _determine_ldap_role_for_entry,
    _get_first_entry_attribute_value,
    _get_user_entry_search_attributes,
)


class LdapAuthFilterTests(unittest.TestCase):
    def test_default_filters_include_univention_mail_attributes(self) -> None:
        filters = _build_default_user_search_filters("matt@visnovsky.us")

        self.assertIn("(uid=matt)", filters)
        self.assertIn("(mail=matt@visnovsky.us)", filters)
        self.assertIn("(mailPrimaryAddress=matt@visnovsky.us)", filters)
        self.assertIn("(mailAlternativeAddress=matt@visnovsky.us)", filters)

    def test_attribute_lookup_falls_back_to_univention_mail_attributes(self) -> None:
        entry = SimpleNamespace(mailPrimaryAddress="matt@visnovsky.us")

        self.assertEqual(
            _get_first_entry_attribute_value(
                entry,
                "mail",
                "mailPrimaryAddress",
                "mailAlternativeAddress",
            ),
            "matt@visnovsky.us",
        )

    def test_user_entry_search_attributes_avoid_explicit_primary_group_id(self) -> None:
        self.assertEqual(_get_user_entry_search_attributes(), ["*", "memberOf"])

    def test_ldap_role_uses_any_configured_admin_and_gate_group(self) -> None:
        ldap_config = SimpleNamespace(
            adminGroupDns=[
                "CN=Admins,OU=Groups,DC=example,DC=com",
                "CN=Ops,OU=Groups,DC=example,DC=com",
            ],
            userGroupDns=[
                "CN=Allowed,OU=Groups,DC=example,DC=com",
                "CN=Login,OU=Groups,DC=example,DC=com",
            ],
        )
        entry = SimpleNamespace(
            memberOf=[
                "CN=Ops,OU=Groups,DC=example,DC=com",
                "CN=Login,OU=Groups,DC=example,DC=com",
            ],
        )

        role = _determine_ldap_role_for_entry(
            ldap_config=ldap_config,
            bind_password="secret",
            user_entry=entry,
            ldap_username="jane",
        )

        self.assertEqual(role, UserRole.admin)

    def test_ldap_role_rejects_user_outside_all_gate_groups(self) -> None:
        ldap_config = SimpleNamespace(
            adminGroupDns=["CN=Admins,OU=Groups,DC=example,DC=com"],
            userGroupDns=[
                "CN=Allowed,OU=Groups,DC=example,DC=com",
                "CN=Login,OU=Groups,DC=example,DC=com",
            ],
        )
        entry = SimpleNamespace(memberOf=["CN=Admins,OU=Groups,DC=example,DC=com"])

        with self.assertRaises(ValueError):
            _determine_ldap_role_for_entry(
                ldap_config=ldap_config,
                bind_password="secret",
                user_entry=entry,
                ldap_username="jane",
            )


if __name__ == "__main__":
    unittest.main()
