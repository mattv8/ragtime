import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ragtime.solidworks.host_artifact import HostArtifact, get_host_artifact, verify_host_artifact


class HostArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = {"host_version": "0.1.0", "protocol": 1, "file": "host.zip", "sha256": "a" * 64, "size_bytes": 3}
        (self.root / "host.zip").write_bytes(b"zip")
        self.setting = mock.patch("ragtime.solidworks.host_artifact.settings.solidworks_host_artifact_dir", str(self.root))
        self.setting.start()
        self.addCleanup(self.setting.stop)

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_artifact(self):
        self.write_manifest()
        artifact = get_host_artifact()
        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact.path, (self.root / "host.zip").resolve())
        self.assertEqual(artifact.file_name, "host.zip")
        self.assertEqual(artifact.size_bytes, 3)

    def test_missing_dir_manifest_or_zip(self):
        self.assertIsNone(get_host_artifact())
        self.write_manifest()
        (self.root / "host.zip").unlink()
        self.assertIsNone(get_host_artifact())
        with mock.patch("ragtime.solidworks.host_artifact.settings.solidworks_host_artifact_dir", str(self.root / "missing")):
            self.assertIsNone(get_host_artifact())

    def test_empty_artifact_directory_is_unavailable(self):
        # Docker creates a missing bind-mount source as an empty directory.
        empty = self.root / "empty-mount"
        empty.mkdir()
        with mock.patch("ragtime.solidworks.host_artifact.settings.solidworks_host_artifact_dir", str(empty)):
            self.assertIsNone(get_host_artifact())

    def test_invalid_manifest_and_fields(self):
        (self.root / "manifest.json").write_text("{", encoding="utf-8")
        self.assertIsNone(get_host_artifact())
        for key, value in [
            ("sha256", "A" * 64),
            ("file", "../host.zip"),
            ("file", "a\\host.zip"),
            ("protocol", True),
            ("size_bytes", -1),
            ("host_version", ""),
        ]:
            with self.subTest(key=key, value=value):
                original = self.manifest[key]
                self.manifest[key] = value
                self.write_manifest()
                self.assertIsNone(get_host_artifact())
                self.manifest[key] = original

    def test_size_on_disk_must_match_manifest(self):
        self.manifest["size_bytes"] = 5
        self.write_manifest()
        self.assertIsNone(get_host_artifact())

    def test_symlink_escaping_artifact_dir_is_rejected(self):
        outside = tempfile.TemporaryDirectory()
        self.addCleanup(outside.cleanup)
        (Path(outside.name) / "secret.zip").write_bytes(b"zip")
        (self.root / "link.zip").symlink_to(Path(outside.name) / "secret.zip")
        self.manifest["file"] = "link.zip"
        self.write_manifest()
        self.assertIsNone(get_host_artifact())

    def test_verify_accepts_matching_hash_and_rejects_mismatch(self):
        self.manifest["sha256"] = hashlib.sha256(b"zip").hexdigest()
        self.write_manifest()
        artifact = get_host_artifact()
        assert artifact is not None
        self.assertTrue(verify_host_artifact(artifact))
        (self.root / "host.zip").write_bytes(b"ZIP")
        self.assertFalse(verify_host_artifact(artifact))

    def test_verify_rejects_missing_file(self):
        artifact = HostArtifact("0.1.0", 1, "gone.zip", "a" * 64, 3, self.root / "gone.zip")
        self.assertFalse(verify_host_artifact(artifact))


if __name__ == "__main__":
    unittest.main()
