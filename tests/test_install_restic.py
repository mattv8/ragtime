"""Tests for the checksum-pinned Restic installer."""

from __future__ import annotations

import bz2
import hashlib
import importlib.util
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

_INSTALLER_PATH = Path(__file__).parents[1] / "docker/scripts/install_restic.py"
_SPEC = importlib.util.spec_from_file_location("install_restic", _INSTALLER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
install_restic = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(install_restic)


class ResticInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "bin/restic"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_installs_a_verified_local_archive_atomically(self) -> None:
        payload = b"restic binary fixture"
        archive = self.root / "restic.bz2"
        archive.write_bytes(bz2.compress(payload))
        expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

        with mock.patch.dict(install_restic.RESTIC_SHA256_BY_ARCH, {"amd64": expected_sha256}, clear=True):
            install_restic.install_restic("amd64", self.destination, archive_url=archive.as_uri())

        self.assertEqual(self.destination.read_bytes(), payload)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o755)

    def test_rejects_bad_http_checksum_without_leaving_a_binary(self) -> None:
        response = io.BytesIO(bz2.compress(b"untrusted"))
        with mock.patch.object(install_restic.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "checksum"):
                install_restic.install_restic("amd64", self.destination)

        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.glob(".restic-*")), [])

    def test_rejects_unsupported_architecture_before_downloading(self) -> None:
        with mock.patch.object(install_restic.urllib.request, "urlopen") as urlopen:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                install_restic.install_restic("ppc64le", self.destination)

        urlopen.assert_not_called()
