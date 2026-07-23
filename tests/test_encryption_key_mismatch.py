import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ragtime.config.settings import ENCRYPTION_KEY_FILE
from ragtime.config.settings import Settings as SettingsClass
from ragtime.core.encryption import (
    attempt_decrypt,
    decrypt_secret,
    encrypt_secret,
    encryption_key_mismatch_detected,
    encryption_recovery_hint,
    reset_key_mismatch_state,
)
from ragtime.core.encryption_health import recheck_encryption_key_health
from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import AppSettings, ConfigurationWarning
from ragtime.indexer.repository import IndexerRepository


class _FindManyDelegate:
    def __init__(self, rows):
        self._rows = rows

    async def find_many(self, **_kwargs):
        if "select" in _kwargs:
            raise AssertionError("find_many() must not be called with select")
        return self._rows

    async def find_unique(self, **_kwargs):
        if "select" in _kwargs:
            raise AssertionError("find_unique() must not be called with select")
        return self._rows[0] if self._rows else None


class _AppSettingsFindManyOnlyDelegate(_FindManyDelegate):
    async def find_unique(self, **_kwargs):
        raise TypeError("AppSettingsActions.find_many() got an unexpected keyword argument 'select'")


def _build_encryption_health_db(**overrides):
    empty_delegate = _FindManyDelegate([])
    delegates = {
        "appsettings": empty_delegate,
        "toolconfig": empty_delegate,
        "mcprouteconfig": empty_delegate,
        "indexjob": empty_delegate,
        "indexmetadata": empty_delegate,
        "ldapconfig": empty_delegate,
        "workspace": empty_delegate,
        "conversationshare": empty_delegate,
        "workspaceshare": empty_delegate,
        "workspaceenvironmentvariable": empty_delegate,
        "globalenvironmentvariable": empty_delegate,
        "userspacemountsource": empty_delegate,
        "useruserspacemountsource": empty_delegate,
        "usercloudoauthaccount": empty_delegate,
    }
    delegates.update(overrides)
    return SimpleNamespace(**delegates)


class EncryptionKeyMismatchTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_key_mismatch_state()

    def tearDown(self) -> None:
        reset_key_mismatch_state()

    def test_clean_round_trip_does_not_flag_mismatch(self) -> None:
        token = encrypt_secret("super-secret")
        self.assertTrue(token.startswith("enc::"))
        self.assertEqual(decrypt_secret(token), "super-secret")
        self.assertFalse(encryption_key_mismatch_detected())

    def test_invalid_token_marks_mismatch_and_returns_empty(self) -> None:
        self.assertFalse(encryption_key_mismatch_detected())
        self.assertEqual(decrypt_secret("enc::not-a-valid-fernet-token"), "")
        self.assertTrue(encryption_key_mismatch_detected())

    def test_attempt_decrypt_reports_invalid_token_without_marking_mismatch(self) -> None:
        self.assertFalse(attempt_decrypt("enc::not-a-valid-fernet-token"))
        self.assertFalse(encryption_key_mismatch_detected())

    def test_failure_is_logged_exactly_once(self) -> None:
        with patch("ragtime.core.encryption.logger") as mock_logger:
            decrypt_secret("enc::bad-one")
            decrypt_secret("enc::bad-two")
            decrypt_secret("enc::bad-three")
        self.assertEqual(mock_logger.error.call_count, 1)

    def test_plaintext_and_empty_do_not_trigger_mismatch(self) -> None:
        self.assertEqual(decrypt_secret(""), "")
        self.assertEqual(decrypt_secret("legacy-plaintext"), "legacy-plaintext")
        self.assertFalse(encryption_key_mismatch_detected())

    def test_reset_clears_state(self) -> None:
        decrypt_secret("enc::bad-token")
        self.assertTrue(encryption_key_mismatch_detected())
        reset_key_mismatch_state()
        self.assertFalse(encryption_key_mismatch_detected())

    def test_recovery_hint_mentions_restore_options(self) -> None:
        hint = encryption_recovery_hint()
        self.assertIn(".encryption_key", hint)
        self.assertIn("--include-secret", hint)
        self.assertIn("Settings", hint)

    def test_existing_key_file_remains_authoritative_when_env_key_is_set(self) -> None:
        original_key_file = ENCRYPTION_KEY_FILE
        settings_mod = sys.modules["ragtime.config.settings"]
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".encryption_key"
            key_file.write_text("old-file-key")
            settings_mod.ENCRYPTION_KEY_FILE = key_file  # type: ignore[attr-defined]
            try:
                key = SettingsClass.generate_encryption_key_if_empty("env-key")
            finally:
                settings_mod.ENCRYPTION_KEY_FILE = original_key_file  # type: ignore[attr-defined]

            self.assertEqual(key, "old-file-key")
            self.assertEqual(key_file.read_text().strip(), "old-file-key")

    def test_env_key_seeds_missing_key_file_once(self) -> None:
        original_key_file = ENCRYPTION_KEY_FILE
        settings_mod = sys.modules["ragtime.config.settings"]
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".encryption_key"
            settings_mod.ENCRYPTION_KEY_FILE = key_file  # type: ignore[attr-defined]
            try:
                key = SettingsClass.generate_encryption_key_if_empty("env-key")
            finally:
                settings_mod.ENCRYPTION_KEY_FILE = original_key_file  # type: ignore[attr-defined]

            self.assertEqual(key, "env-key")
            self.assertEqual(key_file.read_text().strip(), "env-key")
            self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)

    def test_generated_key_is_persisted_with_mode_0600(self) -> None:
        original_key_file = ENCRYPTION_KEY_FILE
        settings_mod = sys.modules["ragtime.config.settings"]
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".encryption_key"
            settings_mod.ENCRYPTION_KEY_FILE = key_file  # type: ignore[attr-defined]
            try:
                key = SettingsClass.generate_encryption_key_if_empty("")
            finally:
                settings_mod.ENCRYPTION_KEY_FILE = original_key_file  # type: ignore[attr-defined]

            self.assertTrue(key)
            self.assertEqual(key_file.read_text().strip(), key)
            self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)

    def test_key_file_is_fallback_when_env_key_is_absent(self) -> None:
        original_key_file = ENCRYPTION_KEY_FILE
        settings_mod = sys.modules["ragtime.config.settings"]
        with tempfile.TemporaryDirectory() as tmpdir:
            key_file = Path(tmpdir) / ".encryption_key"
            key_file.write_text("file-key")
            settings_mod.ENCRYPTION_KEY_FILE = key_file  # type: ignore[attr-defined]
            try:
                key = SettingsClass.generate_encryption_key_if_empty("")
            finally:
                settings_mod.ENCRYPTION_KEY_FILE = original_key_file  # type: ignore[attr-defined]

        self.assertEqual(key, "file-key")


class EncryptionKeyHealthRecheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_key_mismatch_state()

    def tearDown(self) -> None:
        reset_key_mismatch_state()

    async def test_recheck_resets_sticky_flag_when_all_stored_secrets_decrypt(self) -> None:
        decrypt_secret("enc::bad-token")
        self.assertTrue(encryption_key_mismatch_detected())

        db = _build_encryption_health_db(appsettings=_FindManyDelegate([SimpleNamespace(openaiApiKey=encrypt_secret("restored-secret"))]))

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertTrue(await recheck_encryption_key_health())

        self.assertFalse(encryption_key_mismatch_detected())

    async def test_recheck_uses_find_many_for_app_settings_when_find_unique_is_broken(self) -> None:
        db = _build_encryption_health_db(appsettings=_AppSettingsFindManyOnlyDelegate([SimpleNamespace(openaiApiKey=encrypt_secret("restored-secret"))]))

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertTrue(await recheck_encryption_key_health())

    async def test_recheck_accepts_healthy_nested_http_api_secrets(self) -> None:
        db = _build_encryption_health_db(
            toolconfig=_FindManyDelegate(
                [
                    SimpleNamespace(
                        toolType="http_api",
                        connectionConfig={
                            "base_url": "https://api.example.test",
                            "request_headers": [{"name": "X-Tenant", "value": encrypt_secret("tenant-secret")}],
                            "token_request_headers": [{"name": "X-Token-Key", "value": encrypt_secret("endpoint-secret")}],
                            "token_request_fields": [
                                {"name": "grant_type", "value": "client_credentials", "secret": False},
                                {"name": "client_secret", "value": encrypt_secret("client-secret"), "secret": True},
                            ],
                        },
                    )
                ]
            )
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertTrue(await recheck_encryption_key_health())

    async def test_recheck_detects_broken_nested_http_api_secrets(self) -> None:
        db = _build_encryption_health_db(
            toolconfig=_FindManyDelegate(
                [
                    SimpleNamespace(
                        toolType="http_api",
                        connectionConfig={
                            "base_url": "https://api.example.test",
                            "request_headers": [{"name": "X-Tenant", "value": f"enc::broken-tenant"}],
                            "token_request_headers": [{"name": "X-Token-Key", "value": encrypt_secret("endpoint-secret")}],
                            "token_request_fields": [
                                {"name": "grant_type", "value": "client_credentials", "secret": False},
                                {"name": "client_secret", "value": encrypt_secret("client-secret"), "secret": True},
                            ],
                        },
                    )
                ]
            )
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertFalse(await recheck_encryption_key_health())

    async def test_recheck_ignores_plaintext_nested_http_api_values(self) -> None:
        db = _build_encryption_health_db(
            toolconfig=_FindManyDelegate(
                [
                    SimpleNamespace(
                        toolType="http_api",
                        connectionConfig={
                            "base_url": "https://api.example.test",
                            "request_headers": [{"name": "X-Tenant", "value": "tenant-secret"}],
                            "token_request_headers": [{"name": "X-Token-Key", "value": "endpoint-secret"}],
                            "token_request_fields": [
                                {"name": "grant_type", "value": "client_credentials", "secret": False},
                                {"name": "client_secret", "value": "client-secret", "secret": True},
                            ],
                        },
                    )
                ]
            )
        )

        async def fake_get_db():
            return db

        with (
            patch("ragtime.core.encryption_health.get_db", new=fake_get_db),
            patch(
                "ragtime.core.encryption_health.attempt_decrypt",
                wraps=attempt_decrypt,
            ) as attempt_decrypt_mock,
        ):
            self.assertTrue(await recheck_encryption_key_health())

        self.assertEqual(attempt_decrypt_mock.call_count, 0)

    async def test_recheck_scans_nested_values_only_for_http_api_tool_configs(self) -> None:
        db = _build_encryption_health_db(
            toolconfig=_FindManyDelegate(
                [
                    SimpleNamespace(
                        toolType="postgres",
                        connectionConfig={
                            "request_headers": [{"name": "X-Tenant", "value": f"enc::broken-tenant"}],
                            "token_request_fields": [{"name": "client_secret", "value": f"enc::broken-client-secret", "secret": True}],
                        },
                    )
                ]
            )
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertTrue(await recheck_encryption_key_health())

    async def test_get_settings_rechecks_before_building_configuration_warnings(self) -> None:
        decrypt_secret("enc::bad-token")
        self.assertTrue(encryption_key_mismatch_detected())

        settings_model = AppSettings()

        async def fake_get_configuration_warnings(self, chunk_count: int):
            del self
            del chunk_count
            if encryption_key_mismatch_detected():
                return [
                    ConfigurationWarning(
                        level="error",
                        category="encryption",
                        message="mismatch",
                    )
                ]
            return []

        async def fake_recheck() -> bool:
            reset_key_mismatch_state()
            return True

        recheck_mock = AsyncMock(side_effect=fake_recheck)

        with (
            patch.object(indexer_routes.repository, "get_settings", new=AsyncMock(return_value=settings_model)),
            patch.object(indexer_routes.repository, "list_index_metadata", new=AsyncMock(return_value=[])),
            patch.object(AppSettings, "get_configuration_warnings", new=fake_get_configuration_warnings),
            patch("ragtime.indexer.routes.recheck_encryption_key_health", new=recheck_mock),
        ):
            response = await indexer_routes.get_settings(SimpleNamespace(id="admin"))  # type: ignore[arg-type]

        self.assertEqual(response.configuration_warnings, [])
        self.assertFalse(encryption_key_mismatch_detected())
        recheck_mock.assert_awaited_once()


class ToolConfigCredentialHealthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_key_mismatch_state()

    def tearDown(self) -> None:
        reset_key_mismatch_state()

    async def test_clearing_broken_tool_credentials_uses_attempt_decrypt_not_decrypt_secret(self) -> None:
        """Clearing broken tool credentials must not trip the sticky key-mismatch flag."""
        from ragtime.core.encryption import CONNECTION_CONFIG_PASSWORD_FIELDS, ENCRYPTED_PREFIX

        repo = IndexerRepository()
        bad_password = f"{ENCRYPTED_PREFIX}bad-password-token"
        good_token = encrypt_secret("valid-token")
        raw_config = {"host": "db.example.com", "password": bad_password, "token": good_token}
        captured_config: dict | None = None

        async def fake_find_unique(where, include=None):
            del include
            if where.get("id") == "tool-123":
                return SimpleNamespace(
                    id="tool-123",
                    name="Test",
                    toolType="postgres",
                    enabled=True,
                    description="",
                    connectionConfig=captured_config if captured_config is not None else raw_config,
                    maxResults=100,
                    timeoutMaxSeconds=300,
                    allowWrite=False,
                    sortOrder=0,
                    groupId=None,
                    group=None,
                    lastTestAt=None,
                    lastTestResult=None,
                    lastTestError=None,
                    createdAt=None,
                    updatedAt=None,
                )
            return None

        async def fake_update(where, data):
            nonlocal captured_config
            if where.get("id") == "tool-123":
                json_value = data.get("connectionConfig")
                captured_config = dict(json_value.data if hasattr(json_value, "data") else json_value or raw_config)

        db = AsyncMock()
        db.toolconfig.find_unique = fake_find_unique
        db.toolconfig.update = fake_update

        with patch.object(repo, "_get_db", return_value=db):
            result = await repo.clear_tool_undecryptable_credentials("tool-123")

        self.assertIsNotNone(result)
        self.assertFalse(encryption_key_mismatch_detected())
        self.assertNotIn("password", captured_config or {})
        self.assertEqual((captured_config or {}).get("token"), good_token)

    async def test_clearing_broken_nested_http_api_credentials_removes_only_affected_rows(self) -> None:
        from ragtime.core.encryption import ENCRYPTED_PREFIX

        repo = IndexerRepository()
        bad_tenant = f"{ENCRYPTED_PREFIX}broken-tenant"
        bad_client_secret = f"{ENCRYPTED_PREFIX}broken-client-secret"
        good_endpoint = encrypt_secret("endpoint-secret")
        good_shared_secret = encrypt_secret("shared-secret")
        raw_config = {
            "base_url": "https://api.example.test",
            "auth_mode": "token_exchange",
            "request_headers": [
                {"name": "X-Tenant", "value": bad_tenant},
                {"name": "X-Shared", "value": good_shared_secret},
            ],
            "token_request_headers": [{"name": "X-Token-Key", "value": good_endpoint}],
            "token_request_fields": [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "client_secret", "value": bad_client_secret, "secret": True},
                {"name": "scope", "value": "read", "secret": False},
            ],
        }
        captured_config: dict | None = None

        async def fake_find_unique(where, include=None):
            del include
            if where.get("id") == "tool-http-api":
                return SimpleNamespace(
                    id="tool-http-api",
                    name="HTTP API",
                    toolType="http_api",
                    enabled=True,
                    description="",
                    connectionConfig=captured_config if captured_config is not None else raw_config,
                    maxResults=100,
                    timeoutMaxSeconds=300,
                    allowWrite=False,
                    sortOrder=0,
                    groupId=None,
                    group=None,
                    lastTestAt=None,
                    lastTestResult=None,
                    lastTestError=None,
                    createdAt=None,
                    updatedAt=None,
                )
            return None

        async def fake_update(where, data):
            nonlocal captured_config
            if where.get("id") == "tool-http-api":
                json_value = data.get("connectionConfig")
                captured_config = dict(json_value.data if hasattr(json_value, "data") else json_value or raw_config)

        db = AsyncMock()
        db.toolconfig.find_unique = fake_find_unique
        db.toolconfig.update = fake_update

        with patch.object(repo, "_get_db", return_value=db):
            result = await repo.clear_tool_undecryptable_credentials("tool-http-api")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(encryption_key_mismatch_detected())
        assert captured_config is not None
        self.assertEqual(captured_config["request_headers"], [{"name": "X-Shared", "value": good_shared_secret}])
        self.assertEqual(captured_config["token_request_headers"], [{"name": "X-Token-Key", "value": good_endpoint}])
        self.assertEqual(
            captured_config["token_request_fields"],
            [
                {"name": "grant_type", "value": "client_credentials", "secret": False},
                {"name": "scope", "value": "read", "secret": False},
            ],
        )
        self.assertEqual(result.undecryptable_fields, [])


if __name__ == "__main__":
    unittest.main()
