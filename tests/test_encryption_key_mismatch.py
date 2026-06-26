import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragtime.config.settings import ENCRYPTION_KEY_FILE
from ragtime.config.settings import Settings as SettingsClass
from ragtime.core.encryption import (
    decrypt_secret,
    encrypt_secret,
    encryption_key_mismatch_detected,
    encryption_recovery_hint,
    reset_key_mismatch_state,
)


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


if __name__ == "__main__":
    unittest.main()
