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


class _FindManyDelegate:
    def __init__(self, rows):
        self._rows = rows

    async def find_many(self, **_kwargs):
        return self._rows


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

    def test_recovery_hint_mentions_env_key_precedence(self) -> None:
        hint = encryption_recovery_hint()
        self.assertIn("ENCRYPTION_KEY", hint)
        self.assertIn("--include-secret", hint)
        self.assertIn("Settings", hint)

    def test_env_key_wins_and_is_mirrored_to_file(self) -> None:
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

            self.assertEqual(key, "env-key")
            self.assertEqual(key_file.read_text().strip(), "env-key")

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

        db = _build_encryption_health_db(
            appsettings=_FindManyDelegate(
                [SimpleNamespace(openaiApiKey=encrypt_secret("restored-secret"))]
            )
        )

        async def fake_get_db():
            return db

        with patch("ragtime.core.encryption_health.get_db", new=fake_get_db):
            self.assertTrue(await recheck_encryption_key_health())

        self.assertFalse(encryption_key_mismatch_detected())

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
            response = await indexer_routes.get_settings(SimpleNamespace(id="admin"))

        self.assertEqual(response.configuration_warnings, [])
        self.assertFalse(encryption_key_mismatch_detected())
        recheck_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
