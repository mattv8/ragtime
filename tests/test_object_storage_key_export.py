import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.config.settings import Settings


class ObjectStorageKeyExportTests(unittest.TestCase):
    def test_publish_creates_private_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / ".encryption_key"

            Settings.publish_encryption_key_export("fresh-key", destination)

            self.assertEqual(destination.read_text(), "fresh-key")
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)

    def test_publish_replaces_existing_projection_without_leftover_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / ".encryption_key"
            destination.write_text("stale-key")

            Settings.publish_encryption_key_export("restored-key", destination)

            self.assertEqual(destination.read_text(), "restored-key")
            self.assertEqual(list(Path(tmpdir).glob(".encryption_key.*")), [])

    def test_publish_rejects_empty_key_without_replacing_existing_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / ".encryption_key"
            destination.write_text("stale-key")

            with self.assertRaises(ValueError):
                Settings.publish_encryption_key_export("", destination)

            self.assertEqual(destination.read_text(), "stale-key")

    def test_publish_to_missing_directory_fails_without_key_value_in_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "missing" / ".encryption_key"

            with self.assertRaises(OSError) as raised:
                Settings.publish_encryption_key_export("secret-test-key", destination)

            self.assertNotIn("secret-test-key", str(raised.exception))

    def test_existing_primary_key_is_republished_when_export_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / ".encryption_key"
            primary.parent.mkdir()
            primary.write_text("authoritative-key")
            export = Path(tmpdir) / "projection" / ".encryption_key"
            export.parent.mkdir()
            export.write_text("stale-key")
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {"OBJECT_STORAGE_KEY_EXPORT_PATH": str(export)}, clear=False):
                    resolved = Settings.generate_encryption_key_if_empty("")

            self.assertEqual(resolved, "authoritative-key")
            self.assertEqual(export.read_text(), "authoritative-key")

    def test_fresh_primary_key_is_persisted_before_projection_is_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / ".encryption_key"
            export = Path(tmpdir) / "projection" / ".encryption_key"
            export.parent.mkdir()
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {"OBJECT_STORAGE_KEY_EXPORT_PATH": str(export)}, clear=False):
                    resolved = Settings.generate_encryption_key_if_empty("fresh-key")

            self.assertEqual(primary.read_text(), "fresh-key")
            self.assertEqual(export.read_text(), "fresh-key")
            self.assertEqual(resolved, "fresh-key")

    def test_publish_to_non_directory_parent_fails_without_key_value_in_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_file = Path(tmpdir) / "not-a-directory"
            parent_file.write_text("not a directory")

            with self.assertRaises(OSError) as raised:
                Settings.publish_encryption_key_export("secret-test-key", parent_file / ".encryption_key")

            self.assertNotIn("secret-test-key", str(raised.exception))

    def test_configured_export_failure_aborts_key_resolution_without_key_value_in_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / ".encryption_key"
            primary.parent.mkdir()
            primary.write_text("secret-test-key")
            destination = Path(tmpdir) / "missing" / ".encryption_key"
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {"OBJECT_STORAGE_KEY_EXPORT_PATH": str(destination)}, clear=False):
                    with self.assertRaises(RuntimeError) as raised:
                        Settings.generate_encryption_key_if_empty("")

            self.assertIn(str(destination), str(raised.exception))
            self.assertNotIn("secret-test-key", str(raised.exception))

    def test_primary_persistence_failure_with_configured_export_does_not_create_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_parent = Path(tmpdir) / "not-a-directory"
            primary_parent.write_text("not a directory")
            primary = primary_parent / ".encryption_key"
            projection = Path(tmpdir) / "projection" / ".encryption_key"
            projection.parent.mkdir()
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {"OBJECT_STORAGE_KEY_EXPORT_PATH": str(projection)}, clear=False):
                    with self.assertRaises(RuntimeError) as raised:
                        Settings.generate_encryption_key_if_empty("secret-test-key")

            self.assertFalse(projection.exists())
            self.assertNotIn("secret-test-key", str(raised.exception))
            self.assertNotIn("secret-test-key", str(raised.exception.__cause__))

    def test_missing_primary_refuses_to_replace_existing_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / ".encryption_key"
            projection = Path(tmpdir) / "projection" / ".encryption_key"
            projection.parent.mkdir()
            projection.write_text("stale-key")
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {"OBJECT_STORAGE_KEY_EXPORT_PATH": str(projection)}, clear=False):
                    with self.assertRaises(RuntimeError):
                        Settings.generate_encryption_key_if_empty("")

            self.assertEqual(projection.read_text(), "stale-key")

    def test_missing_export_setting_does_not_create_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "data" / ".encryption_key"
            primary.parent.mkdir()
            primary.write_text("authoritative-key")
            with mock.patch("ragtime.config.settings.ENCRYPTION_KEY_FILE", primary):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(Settings.generate_encryption_key_if_empty(""), "authoritative-key")

            self.assertFalse((Path(tmpdir) / "projection" / ".encryption_key").exists())
