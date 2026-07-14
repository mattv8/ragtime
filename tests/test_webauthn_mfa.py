from __future__ import annotations

import unittest
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from jose import jwt  # type: ignore[import-untyped]
from starlette.requests import Request

from ragtime.config.settings import settings
from ragtime.core import mfa, webauthn_mfa
from ragtime.core.encryption import encrypt_secret
from ragtime.core.mfa import TOTP_METHOD, WEBAUTHN_METHOD
from ragtime.core.webauthn_mfa import resolve_rp


class WebauthnTestCase(unittest.IsolatedAsyncioTestCase):
    """Base helpers and per-test cleanup for the module-level jti replay set."""

    def setUp(self) -> None:
        webauthn_mfa._consumed_jtis.clear()

    def tearDown(self) -> None:
        webauthn_mfa._consumed_jtis.clear()

    def _request(
        self,
        host: str = "ragtime.dev",
        origin: str | None = None,
        scheme: str = "https",
    ) -> Request:
        headers: list[tuple[bytes, bytes]] = [(b"host", host.encode())]
        if origin:
            headers.append((b"origin", origin.encode()))
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": headers,
                "scheme": scheme,
            }
        )

    def _make_get_db(self, db: SimpleNamespace):
        async def fake_get_db() -> SimpleNamespace:
            return db

        return fake_get_db

    def _auth_provider_config(self, methods: list[str] | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            id="default",
            mfaAllowedMethods=methods,
            mfa_allowed_methods=methods,
        )

    def _config_delegate(self, methods: list[str] | None = None):
        config = self._auth_provider_config(methods)

        class AuthProviderConfigDelegate:
            async def find_unique(self, where: dict) -> SimpleNamespace | None:
                return config

        return AuthProviderConfigDelegate()

    def _credential_delegate(self, rows: list[SimpleNamespace] | None = None):
        @dataclass
        class WebauthnCredentialDelegate:
            _rows: list[SimpleNamespace]
            _created: list[dict] = field(default_factory=list)
            _updated: list[tuple[dict, dict]] = field(default_factory=list)

            def __post_init__(self):
                if self._created is None:
                    self._created = []
                if self._updated is None:
                    self._updated = []

            async def find_many(self, where: dict) -> list[SimpleNamespace]:
                user_id = where.get("userId")
                return [r for r in self._rows if getattr(r, "userId", None) == user_id]

            async def find_first(self, where: dict) -> SimpleNamespace | None:
                user_id = where.get("userId")
                credential_id = where.get("credentialId")
                for row in self._rows:
                    if getattr(row, "userId", None) == user_id and getattr(row, "credentialId", None) == credential_id:
                        return row
                return None

            async def find_unique(self, where: dict) -> SimpleNamespace | None:
                row_id = where.get("id")
                for row in self._rows:
                    if getattr(row, "id", None) == row_id:
                        return row
                return None

            async def create(self, data: dict) -> SimpleNamespace:
                row = SimpleNamespace(**data)
                self._rows.append(row)
                self._created.append(data)
                return row

            async def update(self, where: dict, data: dict) -> SimpleNamespace:
                row = await self.find_unique(where)
                if row is None:
                    row = SimpleNamespace(id=where.get("id"))
                    self._rows.append(row)
                update_payload = data.get("data", data)
                for key, value in update_payload.items():
                    setattr(row, key, value)
                self._updated.append((where, update_payload))
                return row

            async def delete_many(self, where: dict) -> SimpleNamespace:
                user_id = where.get("userId")
                before = len(self._rows)
                self._rows = [r for r in self._rows if getattr(r, "userId", None) != user_id]
                return SimpleNamespace(count=before - len(self._rows))

            async def count(self, where: dict) -> int:
                return len(await self.find_many(where))

        return WebauthnCredentialDelegate(rows or [])

    def _challenge_delegate(self):
        class WebauthnChallengeDelegate:
            def __init__(self) -> None:
                self._jtis: set[str] = set()
                self.deleted_where: list[dict] = []

            async def delete_many(self, where: dict) -> SimpleNamespace:
                self.deleted_where.append(where)
                return SimpleNamespace(count=0)

            async def create(self, data: dict) -> SimpleNamespace:
                jti = str(data.get("jti", ""))
                if jti in self._jtis:
                    raise Exception("duplicate jti")
                self._jtis.add(jti)
                return SimpleNamespace(**data)

        return WebauthnChallengeDelegate()

    def _registration_replay_setup(self) -> tuple[SimpleNamespace, str, SimpleNamespace]:
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            userwebauthncredential=self._credential_delegate([]),
            userwebauthnchallenge=self._challenge_delegate(),
        )
        token = webauthn_mfa._create_challenge_token(
            user_id="user-1",
            purpose=webauthn_mfa.WEBAUTHN_REGISTER_PURPOSE,
            challenge=b"challenge-bytes",
        )
        verification = SimpleNamespace(
            credential_id=b"cred-id",
            credential_public_key=b"pub-key",
            sign_count=0,
            aaguid=None,
        )
        return db, token, verification

    @contextmanager
    def _registration_replay_completion(self):
        db, token, verification = self._registration_replay_setup()
        user = SimpleNamespace(id="user-1", username="alice")
        credential = {"response": {"transports": ["internal"]}}

        async def complete_registration():
            return await webauthn_mfa.complete_webauthn_registration(
                user,
                self._request(origin="https://ragtime.dev"),
                token,
                credential,
                "My Passkey",
            )

        with (
            mock.patch.object(settings, "external_base_url", "https://ragtime.dev"),
            mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)),
            mock.patch(
                "ragtime.core.webauthn_mfa.verify_registration_response",
                return_value=verification,
            ),
        ):
            yield complete_registration

    def _factor_delegate(self, factor: SimpleNamespace | None):
        class MfaFactorDelegate:
            async def find_unique(self, where: dict) -> SimpleNamespace | None:
                return factor

            async def update(self, where: dict, data: dict) -> SimpleNamespace | None:
                if factor is None:
                    return None
                payload = data.get("data", data)
                for key, value in payload.items():
                    setattr(factor, key, value)
                return factor

        return MfaFactorDelegate()

    def _recovery_delegate(self, codes: list[SimpleNamespace]):
        class RecoveryCodeDelegate:
            async def find_many(self, where: dict) -> list[SimpleNamespace]:
                user_id = where.get("userId")
                used_at_filter = where.get("usedAt")
                return [
                    code for code in codes if getattr(code, "userId", None) == user_id and (used_at_filter is not None or getattr(code, "usedAt", None) is None)
                ]

            async def update(self, where: dict, data: dict) -> SimpleNamespace | None:
                for code in codes:
                    if getattr(code, "id", None) == where.get("id"):
                        payload = data.get("data", data)
                        for key, value in payload.items():
                            setattr(code, key, value)
                        return code
                return None

        return RecoveryCodeDelegate()

    async def _complete_authentication_with_sign_count(self, *, stored_sign_count: int, new_sign_count: int):
        credential = SimpleNamespace(
            id="cred-1",
            userId="user-1",
            credentialId="cred-1",
            publicKey=webauthn_mfa.bytes_to_base64url(b"pub-key"),
            signCount=stored_sign_count,
        )
        delegate = self._credential_delegate([credential])
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            userwebauthncredential=delegate,
        )
        token = webauthn_mfa._create_challenge_token(
            user_id="user-1",
            purpose=webauthn_mfa.WEBAUTHN_AUTHN_PURPOSE,
            challenge=b"auth-challenge",
        )
        verification = SimpleNamespace(new_sign_count=new_sign_count)
        with (
            mock.patch.object(settings, "external_base_url", "https://ragtime.dev"),
            mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)),
            mock.patch(
                "ragtime.core.webauthn_mfa.verify_authentication_response",
                return_value=verification,
            ),
        ):
            await webauthn_mfa.complete_webauthn_authentication(
                SimpleNamespace(id="user-1"),
                self._request(origin="https://ragtime.dev"),
                token,
                {"id": "cred-1"},
            )
        return delegate


class FactorNoExisting:
    async def find_unique(self, where):
        return None

    async def upsert(self, where, data):
        return SimpleNamespace(id="factor-1")


class RecordingRecovery:
    def __init__(self, id_template: str = "rc-1") -> None:
        self.deleted = []
        self.created = []
        self._id_template = id_template

    async def delete_many(self, where):
        self.deleted.append(where)
        return SimpleNamespace(count=0)

    async def create(self, data):
        self.created.append(data)
        if "{index}" in self._id_template:
            record_id = self._id_template.format(index=len(self.created))
        else:
            record_id = self._id_template
        return SimpleNamespace(id=record_id)


class ResolveRpTests(WebauthnTestCase):
    async def test_external_base_url_sets_rp_id_and_origin(self) -> None:
        with mock.patch.object(settings, "external_base_url", "https://ragtime.example.com"):
            rp_id, origins = resolve_rp(self._request(host="localhost", origin="https://localhost:8001"))

        self.assertEqual(rp_id, "ragtime.example.com")
        self.assertEqual(origins, ["https://ragtime.example.com"])

    async def test_request_hostname_used_when_external_base_url_unset(self) -> None:
        with mock.patch.object(settings, "external_base_url", ""):
            rp_id, origins = resolve_rp(self._request(host="ragtime.dev", origin="https://ragtime.dev:8001"))

        self.assertEqual(rp_id, "ragtime.dev")
        self.assertIn("https://ragtime.dev:8001", origins)

    async def test_non_matching_origin_excluded_when_external_origin_present(self) -> None:
        with mock.patch.object(settings, "external_base_url", "https://ragtime.example.com"):
            rp_id, origins = resolve_rp(self._request(host="ragtime.dev", origin="https://evil.com"))

        self.assertEqual(rp_id, "ragtime.example.com")
        self.assertNotIn("https://evil.com", origins)
        self.assertEqual(origins, ["https://ragtime.example.com"])

    async def test_origin_header_drives_rp_id_when_external_base_url_unset(self) -> None:
        # Dev servers and proxies rewrite the Host header (e.g. Vite
        # changeOrigin), so the browser Origin is the user-facing ground truth
        # when no canonical EXTERNAL_BASE_URL is configured.
        with mock.patch.object(settings, "external_base_url", ""):
            rp_id, origins = resolve_rp(self._request(host="127.0.0.1", origin="http://localhost:8001"))

        self.assertEqual(rp_id, "localhost")
        self.assertEqual(origins, ["http://localhost:8001"])

    async def test_no_valid_origin_raises(self) -> None:
        with mock.patch.object(settings, "external_base_url", ""):
            with self.assertRaises(webauthn_mfa.WebauthnError):
                resolve_rp(self._request(host="ragtime.dev"))


class ChallengeTokenTests(WebauthnTestCase):
    async def test_begin_registration_returns_json_options_and_signed_token(self) -> None:
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["totp", "webauthn"]),
            userwebauthncredential=self._credential_delegate([]),
        )
        with mock.patch.object(settings, "external_base_url", "https://ragtime.dev"), mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)):
            user = SimpleNamespace(id="user-1", username="alice", displayName="Alice")
            options, token = await webauthn_mfa.begin_webauthn_registration(user, self._request(origin="https://ragtime.dev"))

        self.assertIsInstance(options, dict)
        self.assertIn("challenge", options)
        self.assertIsInstance(options["challenge"], str)
        # Challenge must be base64url (no padding, URL-safe alphabet).
        self.assertNotIn("=", options["challenge"])
        self.assertNotIn("+", options["challenge"])
        self.assertNotIn("/", options["challenge"])

        claims = webauthn_mfa._decode_challenge_token(token, expected_purpose=webauthn_mfa.WEBAUTHN_REGISTER_PURPOSE)
        self.assertIsNotNone(claims)
        assert claims is not None
        self.assertEqual(claims.user_id, "user-1")
        self.assertEqual(claims.purpose, webauthn_mfa.WEBAUTHN_REGISTER_PURPOSE)
        self.assertTrue(claims.jti)

    async def test_complete_registration_rejects_tampered_token(self) -> None:
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            userwebauthncredential=self._credential_delegate([]),
        )
        with mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)):
            user = SimpleNamespace(id="user-1", username="alice")
            with self.assertRaises(webauthn_mfa.WebauthnError):
                await webauthn_mfa.complete_webauthn_registration(
                    user,
                    self._request(origin="https://ragtime.dev"),
                    "not-a-real-token",
                    {},
                    None,
                )

    async def test_complete_registration_rejects_expired_token(self) -> None:
        payload = {
            "sub": "user-1",
            "purpose": webauthn_mfa.WEBAUTHN_REGISTER_PURPOSE,
            "challenge": webauthn_mfa.bytes_to_base64url(b"challenge"),
            "jti": "expired-jti",
            "exp": datetime.now(timezone.utc) - timedelta(seconds=10),
        }
        expired_token = jwt.encode(payload, settings.encryption_key, algorithm=settings.jwt_algorithm)
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            userwebauthncredential=self._credential_delegate([]),
        )
        with mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)):
            user = SimpleNamespace(id="user-1", username="alice")
            with self.assertRaises(webauthn_mfa.WebauthnError):
                await webauthn_mfa.complete_webauthn_registration(
                    user,
                    self._request(origin="https://ragtime.dev"),
                    expired_token,
                    {},
                    None,
                )

    async def test_registration_token_jti_replay_is_rejected(self) -> None:
        with self._registration_replay_completion() as complete_registration:
            result = await complete_registration()
            self.assertEqual(getattr(result, "credentialId", None), "Y3JlZC1pZA")

            with self.assertRaises(webauthn_mfa.WebauthnError) as ctx:
                await complete_registration()
            self.assertIn("already been used", str(ctx.exception))

    async def test_registration_token_jti_replay_is_rejected_after_memory_reset(self) -> None:
        with self._registration_replay_completion() as complete_registration:
            await complete_registration()

            webauthn_mfa._consumed_jtis.clear()

            with self.assertRaises(webauthn_mfa.WebauthnError) as ctx:
                await complete_registration()
            self.assertIn("already been used", str(ctx.exception))

    async def test_consume_jti_translates_prisma_unique_violation(self) -> None:
        class DuplicateJtiError(Exception):
            pass

        class ChallengeDelegate:
            async def delete_many(self, where: dict) -> SimpleNamespace:
                return SimpleNamespace(count=0)

            async def create(self, data: dict) -> SimpleNamespace:
                raise DuplicateJtiError("unique constraint failed")

        db = SimpleNamespace(userwebauthnchallenge=ChallengeDelegate())
        with (
            mock.patch("ragtime.core.webauthn_mfa.get_db", new=self._make_get_db(db)),
            mock.patch.object(webauthn_mfa, "UniqueViolationError", DuplicateJtiError),
        ):
            with self.assertRaises(webauthn_mfa.WebauthnError) as ctx:
                await webauthn_mfa._consume_jti("jti-1", (datetime.now(timezone.utc) + timedelta(minutes=1)).timestamp())

        self.assertIn("already been used", str(ctx.exception))


class AllowedMethodsTests(WebauthnTestCase):
    async def test_allowed_methods_default_to_totp_when_config_missing(self) -> None:
        db = SimpleNamespace(authproviderconfig=self._config_delegate(methods=None))
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            self.assertEqual(await mfa.get_allowed_mfa_methods(), [TOTP_METHOD])

    async def test_allowed_methods_default_to_totp_when_config_empty(self) -> None:
        db = SimpleNamespace(authproviderconfig=self._config_delegate(methods=[]))
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            self.assertEqual(await mfa.get_allowed_mfa_methods(), [TOTP_METHOD])

    async def test_allowed_methods_default_to_totp_when_only_invalid_values(self) -> None:
        db = SimpleNamespace(authproviderconfig=self._config_delegate(methods=["sms", ""]))
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            self.assertEqual(await mfa.get_allowed_mfa_methods(), [TOTP_METHOD])

    async def test_allowed_methods_return_valid_values_in_order(self) -> None:
        db = SimpleNamespace(authproviderconfig=self._config_delegate(methods=["webauthn", "totp", "SMS", ""]))
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            self.assertEqual(await mfa.get_allowed_mfa_methods(), ["webauthn", "totp"])

    async def test_user_allowed_enrolled_methods_prefers_webauthn(self) -> None:
        with (
            mock.patch("ragtime.core.mfa.get_allowed_mfa_methods", return_value=["totp", "webauthn"]),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=True),
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=True),
        ):
            self.assertEqual(
                await mfa.user_allowed_enrolled_methods("user-1"),
                ["webauthn", "totp"],
            )

    async def test_user_allowed_enrolled_methods_ignores_disallowed_factors(self) -> None:
        with (
            mock.patch("ragtime.core.mfa.get_allowed_mfa_methods", return_value=["totp"]),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=True),
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=True),
        ):
            self.assertEqual(await mfa.user_allowed_enrolled_methods("user-1"), ["totp"])


class VerifyUserMfaCodeTests(WebauthnTestCase):
    async def test_totp_rejected_when_only_webauthn_allowed(self) -> None:
        secret = mfa.generate_totp_secret()
        totp_code = mfa.generate_totp_code(secret)
        factor = SimpleNamespace(
            id="factor-1",
            userId="user-1",
            factorType="totp",
            enabled=True,
            secretEncrypted=encrypt_secret(secret),
            lastUsedStep=None,
        )
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            usermfafactor=self._factor_delegate(factor),
            usermfarecoverycode=self._recovery_delegate([]),
        )
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            valid, method = await mfa.verify_user_mfa_code(SimpleNamespace(id="user-1"), totp_code)

        self.assertFalse(valid)
        self.assertIsNone(method)

    async def test_recovery_code_accepted_when_only_webauthn_allowed(self) -> None:
        code = mfa.generate_recovery_code()
        db = SimpleNamespace(
            authproviderconfig=self._config_delegate(methods=["webauthn"]),
            usermfafactor=self._factor_delegate(None),
            usermfarecoverycode=self._recovery_delegate(
                [
                    SimpleNamespace(
                        id="rc-1",
                        userId="user-1",
                        codeHash=mfa.hash_recovery_code(code),
                        usedAt=None,
                    )
                ]
            ),
        )
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            valid, method = await mfa.verify_user_mfa_code(SimpleNamespace(id="user-1"), code)

        self.assertTrue(valid)
        self.assertEqual(method, "recovery_code")


class ResetUserMfaTests(WebauthnTestCase):
    async def test_reset_user_mfa_deletes_webauthn_credentials(self) -> None:
        deleted_where: list[dict] = []

        class WebauthnCredentialDelegate:
            async def delete_many(self, where: dict) -> SimpleNamespace:
                deleted_where.append(where)
                return SimpleNamespace(count=1)

        class NoOpDelegate:
            async def delete_many(self, where: dict) -> SimpleNamespace:
                return SimpleNamespace(count=0)

        db = SimpleNamespace(
            usermfafactor=NoOpDelegate(),
            usermfarecoverycode=NoOpDelegate(),
            usermfatrusteddevice=NoOpDelegate(),
            userwebauthncredential=WebauthnCredentialDelegate(),
        )
        with mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)):
            await mfa.reset_user_mfa("user-1")

        self.assertEqual(deleted_where, [{"userId": "user-1"}])


class SignCountTests(WebauthnTestCase):
    async def test_complete_authentication_persists_new_sign_count(self) -> None:
        delegate = await self._complete_authentication_with_sign_count(
            stored_sign_count=5,
            new_sign_count=42,
        )

        self.assertEqual(len(delegate._updated), 1)
        self.assertEqual(delegate._updated[0][1]["signCount"], 42)
        self.assertIsNotNone(delegate._updated[0][1]["lastUsedAt"])

    async def test_complete_authentication_persists_zero_sign_count(self) -> None:
        delegate = await self._complete_authentication_with_sign_count(
            stored_sign_count=0,
            new_sign_count=0,
        )

        self.assertEqual(delegate._updated[0][1]["signCount"], 0)


class MfaNeededForUserTests(WebauthnTestCase):
    async def test_webauthn_only_enrolled_user_requires_mfa(self) -> None:
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        config = self._auth_provider_config(methods=["totp", "webauthn"])
        config.totp_policy = "optional"
        config.totpPolicy = "optional"
        config.totp_required_group_ids = []
        config.totpRequiredGroupIds = []
        with (
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=False),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=True),
        ):
            result = await mfa.mfa_needed_for_user(user, config=config)
        self.assertTrue(result)

    async def test_user_with_no_factors_and_optional_policy_does_not_require_mfa(self) -> None:
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        config = self._auth_provider_config(methods=["totp", "webauthn"])
        config.totp_policy = "optional"
        config.totpPolicy = "optional"
        config.totp_required_group_ids = []
        config.totpRequiredGroupIds = []
        with (
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=False),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=False),
        ):
            result = await mfa.mfa_needed_for_user(user, config=config)
        self.assertFalse(result)


class ConfirmTotpEnrollmentTests(WebauthnTestCase):
    async def _confirm_totp_enrollment(self, *, has_webauthn: bool, recovery: RecordingRecovery):
        db = SimpleNamespace(
            usermfafactor=FactorNoExisting(),
            usermfarecoverycode=recovery,
        )
        user = SimpleNamespace(id="user-1", username="alice", role="user")
        secret = mfa.generate_totp_secret()
        enrollment_token = mfa.create_totp_enrollment_token(user_id=user.id, username=user.username, role=user.role, secret=secret)
        code = mfa.generate_totp_code(secret)
        with (
            mock.patch("ragtime.core.mfa.get_db", new=self._make_get_db(db)),
            mock.patch("ragtime.core.mfa.user_has_enabled_totp", return_value=False),
            mock.patch("ragtime.core.mfa.user_has_enabled_webauthn", return_value=has_webauthn),
        ):
            return await mfa.confirm_totp_enrollment(user, code, enrollment_token)

    async def test_confirm_totp_does_not_regenerate_recovery_codes_when_webauthn_exists(self) -> None:
        recovery = RecordingRecovery()
        success, recovery_codes = await self._confirm_totp_enrollment(
            has_webauthn=True,
            recovery=recovery,
        )
        self.assertTrue(success)
        self.assertEqual(recovery_codes, [])
        self.assertEqual(recovery.deleted, [])
        self.assertEqual(recovery.created, [])

    async def test_confirm_totp_generates_recovery_codes_for_first_factor(self) -> None:
        recovery = RecordingRecovery(id_template="rc-{index}")
        success, recovery_codes = await self._confirm_totp_enrollment(
            has_webauthn=False,
            recovery=recovery,
        )
        self.assertTrue(success)
        self.assertEqual(len(recovery_codes), mfa.RECOVERY_CODE_COUNT)
        self.assertEqual(len(recovery.deleted), 1)
        self.assertEqual(len(recovery.created), mfa.RECOVERY_CODE_COUNT)


if __name__ == "__main__":
    unittest.main()
