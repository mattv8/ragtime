import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from starlette.requests import Request

from ragtime.api import auth as api_auth
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
    def _first_factor_db(self):
        class FactorDelegate:
            async def find_unique(self, where):
                return None

            async def upsert(self, where, data):
                self.created_factor = data["create"]
                return SimpleNamespace(id="factor-1")

        class RecoveryCodeDelegate:
            async def delete_many(self, where):
                return None

            async def create(self, data):
                return SimpleNamespace(id="recovery-1")

        factor_delegate = FactorDelegate()
        db = SimpleNamespace(
            usermfafactor=factor_delegate,
            usermfarecoverycode=RecoveryCodeDelegate(),
        )
        return db, factor_delegate

    async def test_begin_enrollment_does_not_persist_unconfirmed_factor(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                return None

            async def upsert(self, data, where):
                raise AssertionError("unconfirmed enrollment must not persist an MFA factor")

        db = SimpleNamespace(usermfafactor=FactorDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            setup = await mfa.begin_totp_enrollment(SimpleNamespace(id="user-1", username="alice"))

        self.assertIn("secret", setup)
        self.assertIn("otpauth_uri", setup)
        self.assertIn("enrollment_token", setup)

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

    async def test_complete_enrollment_persists_factor_after_valid_token_code(self) -> None:
        db, factor_delegate = self._first_factor_db()
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        secret = mfa.generate_totp_secret()
        enrollment_token = mfa.create_totp_enrollment_token(
            user_id=user.id,
            username=user.username,
            role=user.role,
            secret=secret,
        )
        code = mfa.generate_totp_code(secret)

        async def fake_get_db():
            return db

        with (
            mock.patch("ragtime.core.mfa.get_db", new=fake_get_db),
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=False),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=False),
        ):
            success, recovery_codes = await mfa.confirm_totp_enrollment(
                user,
                code,
                enrollment_token,
            )

        self.assertTrue(success)
        self.assertEqual(len(recovery_codes), mfa.RECOVERY_CODE_COUNT)
        self.assertIsNotNone(factor_delegate.created_factor)
        self.assertTrue(factor_delegate.created_factor["enabled"])
        self.assertEqual(factor_delegate.created_factor["factorType"], "totp")

    async def test_complete_enrollment_rejects_token_for_different_user(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                raise AssertionError("mismatched token must be rejected before factor lookup")

        db = SimpleNamespace(usermfafactor=FactorDelegate())
        secret = mfa.generate_totp_secret()
        enrollment_token = mfa.create_totp_enrollment_token(
            user_id="user-2",
            username="bob",
            role="user",
            secret=secret,
        )

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            success, recovery_codes = await mfa.confirm_totp_enrollment(
                SimpleNamespace(id="user-1", username="alice"),
                mfa.generate_totp_code(secret),
                enrollment_token,
            )

        self.assertFalse(success)
        self.assertEqual(recovery_codes, [])

    async def test_complete_enrollment_rejects_invalid_token(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                raise AssertionError("invalid token must be rejected before factor lookup")

        db = SimpleNamespace(usermfafactor=FactorDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            success, recovery_codes = await mfa.confirm_totp_enrollment(
                SimpleNamespace(id="user-1", username="alice"),
                "123456",
                "not-a-valid-token",
            )

        self.assertFalse(success)
        self.assertEqual(recovery_codes, [])

    async def test_complete_enrollment_does_not_overwrite_enabled_factor(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(id="factor-1", enabled=True, secretEncrypted="enc::secret")

            async def upsert(self, where, data):
                raise AssertionError("enabled factor must not be overwritten")

        class RecoveryCodeDelegate:
            async def delete_many(self, where):
                raise AssertionError("recovery codes must not be changed when factor is already enabled")

        db = SimpleNamespace(
            usermfafactor=FactorDelegate(),
            usermfarecoverycode=RecoveryCodeDelegate(),
        )
        secret = mfa.generate_totp_secret()
        enrollment_token = mfa.create_totp_enrollment_token(
            user_id="user-1",
            username="alice",
            role="user",
            secret=secret,
        )

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            success, recovery_codes = await mfa.confirm_totp_enrollment(
                SimpleNamespace(id="user-1", username="alice"),
                mfa.generate_totp_code(secret),
                enrollment_token,
            )

        self.assertFalse(success)
        self.assertEqual(recovery_codes, [])


class TotpRotationTests(unittest.IsolatedAsyncioTestCase):
    async def test_enrollment_token_carries_rotation_claim(self) -> None:
        secret = mfa.generate_totp_secret()

        rotating = mfa.create_totp_enrollment_token(user_id="user-1", username="alice", role="user", secret=secret, rotation=True)
        normal = mfa.create_totp_enrollment_token(user_id="user-1", username="alice", role="user", secret=secret)

        rotating_claims = mfa.decode_totp_enrollment_token(rotating)
        normal_claims = mfa.decode_totp_enrollment_token(normal)
        assert rotating_claims is not None and normal_claims is not None
        self.assertTrue(rotating_claims.rotation)
        self.assertFalse(normal_claims.rotation)

    async def test_begin_enrollment_allow_replace_permits_enabled_factor(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(id="factor-1", enabled=True, secretEncrypted="enc::secret")

            async def upsert(self, where, data):
                raise AssertionError("begin must not persist a factor")

        db = SimpleNamespace(usermfafactor=FactorDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            setup = await mfa.begin_totp_enrollment(SimpleNamespace(id="user-1", username="alice"), allow_replace=True)

        claims = mfa.decode_totp_enrollment_token(setup["enrollment_token"])
        assert claims is not None
        self.assertTrue(claims.rotation)

    async def test_rotation_replaces_secret_without_regenerating_recovery_codes(self) -> None:
        updated_secret_holder: dict = {}

        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(id="factor-1", enabled=True, secretEncrypted="enc::old")

            async def upsert(self, where, data):
                updated_secret_holder["update"] = data["update"]
                return SimpleNamespace(id="factor-1")

        class RecoveryCodeDelegate:
            async def delete_many(self, where):
                raise AssertionError("recovery codes must not be regenerated during rotation")

            async def create(self, data):
                raise AssertionError("recovery codes must not be regenerated during rotation")

        db = SimpleNamespace(
            usermfafactor=FactorDelegate(),
            usermfarecoverycode=RecoveryCodeDelegate(),
        )
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        new_secret = mfa.generate_totp_secret()
        rotation_token = mfa.create_totp_enrollment_token(user_id=user.id, username=user.username, role=user.role, secret=new_secret, rotation=True)
        code = mfa.generate_totp_code(new_secret)

        async def fake_get_db():
            return db

        with (
            mock.patch("ragtime.core.mfa.get_db", new=fake_get_db),
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=True),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=False),
        ):
            success, recovery_codes = await mfa.confirm_totp_enrollment(user, code, rotation_token)

        self.assertTrue(success)
        self.assertEqual(recovery_codes, [])
        self.assertTrue(updated_secret_holder["update"]["enabled"])

    async def test_non_rotation_token_still_blocked_on_enabled_factor(self) -> None:
        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(id="factor-1", enabled=True, secretEncrypted="enc::secret")

            async def upsert(self, where, data):
                raise AssertionError("enabled factor must not be overwritten without rotation claim")

        db = SimpleNamespace(usermfafactor=FactorDelegate())
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        secret = mfa.generate_totp_secret()
        normal_token = mfa.create_totp_enrollment_token(user_id=user.id, username=user.username, role=user.role, secret=secret)

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            success, recovery_codes = await mfa.confirm_totp_enrollment(user, mfa.generate_totp_code(secret), normal_token)

        self.assertFalse(success)
        self.assertEqual(recovery_codes, [])

    async def test_regenerate_recovery_codes_replaces_all_codes(self) -> None:
        created: list[dict] = []
        deleted: list[dict] = []

        class RecoveryCodeDelegate:
            async def delete_many(self, where):
                deleted.append(where)

            async def create(self, data):
                created.append(data)
                return SimpleNamespace(id=f"recovery-{len(created)}")

        db = SimpleNamespace(usermfarecoverycode=RecoveryCodeDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            codes = await mfa.regenerate_recovery_codes("user-1")

        self.assertEqual(len(codes), mfa.RECOVERY_CODE_COUNT)
        self.assertEqual(len(created), mfa.RECOVERY_CODE_COUNT)
        self.assertEqual(deleted, [{"userId": "user-1"}])
        for entry in created:
            self.assertEqual(entry["userId"], "user-1")
            self.assertNotIn(entry["codeHash"], codes)  # stored hashed, not plaintext


class ResolvePreferredMfaMethodTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_user_preference_when_enrolled(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod="totp")

        result = await mfa.resolve_preferred_mfa_method(user, ["webauthn", "totp"])

        self.assertEqual(result, "totp")

    async def test_user_preference_is_case_normalized(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod="WEBAUTHN")

        result = await mfa.resolve_preferred_mfa_method(user, ["webauthn", "totp"])

        self.assertEqual(result, "webauthn")

    async def test_ignores_user_preference_that_is_not_enrolled(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod="webauthn")

        class ConfigDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(mfaDefaultMethod=None)

        db = SimpleNamespace(authproviderconfig=ConfigDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            result = await mfa.resolve_preferred_mfa_method(user, ["totp"])

        self.assertIsNone(result)

    async def test_falls_back_to_config_default(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod=None)
        config = SimpleNamespace(mfaDefaultMethod="totp")

        result = await mfa.resolve_preferred_mfa_method(user, ["totp"], config=config)

        self.assertEqual(result, "totp")

    async def test_config_default_is_case_normalized(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod=None)
        config = SimpleNamespace(mfaDefaultMethod="WEBAUTHN")

        result = await mfa.resolve_preferred_mfa_method(user, ["webauthn", "totp"], config=config)

        self.assertEqual(result, "webauthn")

    async def test_user_preference_wins_over_config_default(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod="webauthn")
        config = SimpleNamespace(mfaDefaultMethod="totp")

        result = await mfa.resolve_preferred_mfa_method(user, ["webauthn", "totp"], config=config)

        self.assertEqual(result, "webauthn")

    async def test_ignores_config_default_that_is_not_enrolled(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod=None)
        config = SimpleNamespace(mfaDefaultMethod="webauthn")

        result = await mfa.resolve_preferred_mfa_method(user, ["totp"], config=config)

        self.assertIsNone(result)

    async def test_loads_config_when_not_provided(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod=None)

        class ConfigDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(mfaDefaultMethod="totp")

        db = SimpleNamespace(authproviderconfig=ConfigDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            result = await mfa.resolve_preferred_mfa_method(user, ["totp"])

        self.assertEqual(result, "totp")

    async def test_returns_none_when_no_preference_or_default(self) -> None:
        user = SimpleNamespace(id="user-1", mfaPreferredMethod=None)

        class ConfigDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(mfaDefaultMethod=None)

        db = SimpleNamespace(authproviderconfig=ConfigDelegate())

        async def fake_get_db():
            return db

        with mock.patch("ragtime.core.mfa.get_db", new=fake_get_db):
            result = await mfa.resolve_preferred_mfa_method(user, ["totp"])

        self.assertIsNone(result)


class DebugTotpPrefillTests(unittest.IsolatedAsyncioTestCase):
    def _build_auth_status_request(self) -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/auth/status",
                "headers": [(b"host", b"ragtime.dev")],
                "scheme": "https",
            }
        )

    def _build_debug_totp_db(self, secret: str) -> SimpleNamespace:
        admin_user = SimpleNamespace(id="local-admin-user-id")

        class UserDelegate:
            async def find_unique(self, where):
                if where.get("username") == "local:debugadmin":
                    return admin_user
                return None

        class FactorDelegate:
            async def find_unique(self, where):
                return SimpleNamespace(
                    id="factor-1",
                    enabled=True,
                    secretEncrypted=secret,
                )

        return SimpleNamespace(
            user=UserDelegate(),
            usermfafactor=FactorDelegate(),
        )

    def _debug_auth_status_patches(self, db: SimpleNamespace, *, debug_mode: bool) -> ExitStack:
        stack = ExitStack()
        stack.enter_context(mock.patch("ragtime.api.auth.get_db", new=mock.AsyncMock(return_value=db)))
        stack.enter_context(mock.patch("ragtime.core.mfa.get_db", new=mock.AsyncMock(return_value=db)))
        stack.enter_context(mock.patch("ragtime.api.auth.build_auth_method_statuses", new=mock.AsyncMock(return_value=[])))
        stack.enter_context(mock.patch("ragtime.api.auth.get_app_settings", new=mock.AsyncMock(return_value={})))
        stack.enter_context(mock.patch.object(api_auth.settings, "debug_mode", debug_mode))
        stack.enter_context(mock.patch.object(api_auth.settings, "local_admin_user", "debugadmin"))
        stack.enter_context(mock.patch.object(api_auth.settings, "local_admin_password", "debugpassword"))
        return stack

    async def test_debug_mode_exposes_local_admin_totp_code(self) -> None:
        secret = mfa.generate_totp_secret()
        expected_code = mfa.generate_totp_code(secret)
        db = self._build_debug_totp_db(secret)

        with self._debug_auth_status_patches(db, debug_mode=True):
            status = await api_auth.get_auth_status(self._build_auth_status_request(), None)

        self.assertEqual(status.debug_totp_code, expected_code)
        self.assertRegex(status.debug_totp_code or "", r"^\d{6}$")

    async def test_non_debug_mode_hides_totp_code(self) -> None:
        secret = mfa.generate_totp_secret()
        db = self._build_debug_totp_db(secret)

        with self._debug_auth_status_patches(db, debug_mode=False):
            status = await api_auth.get_auth_status(self._build_auth_status_request(), None)

        self.assertIsNone(status.debug_totp_code)

    async def test_debug_totp_endpoint_returns_current_code_in_debug_mode(self) -> None:
        secret = mfa.generate_totp_secret()
        fixed_time = 1_000.0
        db = self._build_debug_totp_db(secret)

        with self._debug_auth_status_patches(db, debug_mode=True):
            with mock.patch("ragtime.core.mfa.time", new=SimpleNamespace(time=lambda: fixed_time)):
                response = await api_auth.get_debug_totp_code()

        self.assertEqual(response.code, mfa.generate_totp_code(secret, for_time=fixed_time))
        self.assertRegex(response.code or "", r"^\d{6}$")

    async def test_debug_totp_endpoint_returns_null_outside_debug_mode(self) -> None:
        secret = mfa.generate_totp_secret()
        db = self._build_debug_totp_db(secret)

        with self._debug_auth_status_patches(db, debug_mode=False):
            response = await api_auth.get_debug_totp_code()

        self.assertIsNone(response.code)


if __name__ == "__main__":
    unittest.main()
