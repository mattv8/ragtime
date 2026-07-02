import time
import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.core import mfa


class TotpPrimitiveTests(unittest.TestCase):
    def test_totp_matches_rfc_6238_sha1_vector(self) -> None:
        secret = mfa.base32_secret_from_bytes(b"12345678901234567890")

        self.assertEqual(mfa.generate_totp_code(secret, for_time=59, digits=8), "94287082")

    def test_totp_rejects_reused_time_step(self) -> None:
        secret = mfa.generate_totp_secret()
        step = int(time.time()) // 30
        code = mfa.generate_totp_code(secret, for_time=step * 30)

        self.assertTrue(mfa.verify_totp_code(secret, code, last_used_step=step - 1).valid)
        self.assertFalse(mfa.verify_totp_code(secret, code, last_used_step=step).valid)

    def test_recovery_code_hash_round_trips_without_plaintext_recovery(self) -> None:
        code = mfa.generate_recovery_code()
        stored = mfa.hash_recovery_code(code)

        self.assertNotIn(code, stored)
        self.assertTrue(mfa.verify_recovery_code(code, stored))
        self.assertFalse(mfa.verify_recovery_code("wrong-code", stored))


class TotpPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_policy_does_not_require_unenrolled_user(self) -> None:
        config = SimpleNamespace(totp_policy="optional", totp_required_group_ids=[])
        user = SimpleNamespace(id="user-1", role="admin")

        self.assertFalse(await mfa.user_requires_totp(user, config=config))

    async def test_required_all_policy_requires_every_user(self) -> None:
        config = SimpleNamespace(totp_policy="required_all", totp_required_group_ids=[])
        user = SimpleNamespace(id="user-1", role="user")

        self.assertTrue(await mfa.user_requires_totp(user, config=config))

    async def test_admins_groups_policy_requires_admins(self) -> None:
        config = SimpleNamespace(totp_policy="required_admins_groups", totp_required_group_ids=[])
        user = SimpleNamespace(id="user-1", role="admin")

        self.assertTrue(await mfa.user_requires_totp(user, config=config))

    async def test_admins_groups_policy_requires_selected_group_member(self) -> None:
        config = SimpleNamespace(totp_policy="required_admins_groups", totp_required_group_ids=["group-1"])
        user = SimpleNamespace(id="user-1", role="user")
        test_case = self

        class MembershipDelegate:
            async def find_first(self, where):
                test_case.assertEqual(where["userId"], "user-1")
                test_case.assertEqual(where["groupId"], {"in": ["group-1"]})
                return SimpleNamespace(id="membership-1")

        db = SimpleNamespace(authgroupmembership=MembershipDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            self.assertTrue(await mfa.user_requires_totp(user, config=config))


class PendingMfaTokenTests(unittest.TestCase):
    def test_pending_token_round_trips_without_creating_app_session(self) -> None:
        token = mfa.create_pending_mfa_token(
            user_id="user-1",
            username="alice",
            role="admin",
            purpose="enroll",
        )

        claims = mfa.decode_pending_mfa_token(token, expected_purpose="enroll")

        self.assertIsNotNone(claims)
        assert claims is not None
        self.assertEqual(claims.user_id, "user-1")
        self.assertEqual(claims.username, "alice")
        self.assertEqual(claims.role, "admin")
        self.assertEqual(claims.purpose, "enroll")
        self.assertIsNone(mfa.decode_pending_mfa_token(token, expected_purpose="challenge"))


class TotpEnrollmentSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_begin_enrollment_rejects_existing_enabled_factor(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(id="factor-1", enabled=True, secretEncrypted="enc::secret")

            async def upsert(self, data, where):
                raise AssertionError("enabled factor must not be overwritten")

        db = SimpleNamespace(usermfafactor=FactorDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            with self.assertRaises(ValueError):
                await mfa.begin_totp_enrollment(SimpleNamespace(id="user-1", username="alice"))


if __name__ == "__main__":
    unittest.main()
