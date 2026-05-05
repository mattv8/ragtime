import sys
import unittest
from enum import Enum
from types import ModuleType, SimpleNamespace

prisma_module = ModuleType("prisma")
prisma_enums_module = ModuleType("prisma.enums")


class AuthProvider(str, Enum):
    ldap = "ldap"
    local = "local"
    local_managed = "local_managed"


class UserRole(str, Enum):
    user = "user"
    admin = "admin"


setattr(prisma_enums_module, "AuthProvider", AuthProvider)
setattr(prisma_enums_module, "UserRole", UserRole)
setattr(prisma_module, "enums", prisma_enums_module)
setattr(prisma_module, "Json", list)
sys.modules.setdefault("prisma", prisma_module)
sys.modules["prisma.enums"] = prisma_enums_module

database_module = ModuleType("ragtime.core.database")


async def get_db() -> None:
    raise RuntimeError("Database access is not needed for this test")


setattr(database_module, "get_db", get_db)
sys.modules["ragtime.core.database"] = database_module

from ragtime.core.auth import (
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
