import sys
import unittest
from enum import Enum
from types import ModuleType, SimpleNamespace

prisma_module = ModuleType("prisma")
prisma_enums_module = ModuleType("prisma.enums")


class AuthProvider(str, Enum):
    ldap = "ldap"


class UserRole(str, Enum):
    user = "user"
    admin = "admin"


setattr(prisma_enums_module, "AuthProvider", AuthProvider)
setattr(prisma_enums_module, "UserRole", UserRole)
setattr(prisma_module, "enums", prisma_enums_module)
sys.modules.setdefault("prisma", prisma_module)
sys.modules["prisma.enums"] = prisma_enums_module

database_module = ModuleType("ragtime.core.database")


async def get_db() -> None:
    raise RuntimeError("Database access is not needed for this test")


setattr(database_module, "get_db", get_db)
sys.modules["ragtime.core.database"] = database_module

from ragtime.core.auth import (
    _build_default_user_search_filters,
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


if __name__ == "__main__":
    unittest.main()
