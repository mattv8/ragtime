"""Test marker reader export and recovery operation hardening."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.sqlite_runtime import read_marker


class MarkerReaderHardeningTests(unittest.TestCase):
    """Test that exported read_marker function rejects unsafe marker files."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.marker_path = Path(self.temp.name) / "marker.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_read_marker_rejects_symlink_markers(self) -> None:
        """Symlinked markers must be rejected to prevent directory traversal."""
        real_marker = Path(self.temp.name) / "real.json"
        real_marker.write_text(json.dumps({"lease_id": "lease-1"}), encoding="utf-8")
        self.marker_path.symlink_to(real_marker)

        with self.assertRaises(HTTPException) as error:
            read_marker(self.marker_path)

        self.assertEqual(423, error.exception.status_code)
        self.assertIn("unsafe", error.exception.detail)

    def test_read_marker_rejects_directory_markers(self) -> None:
        """A directory at marker path must be rejected."""
        self.marker_path.mkdir()

        with self.assertRaises(HTTPException) as error:
            read_marker(self.marker_path)

        self.assertEqual(423, error.exception.status_code)
        self.assertIn("unsafe", error.exception.detail)

    def test_read_marker_accepts_regular_markers(self) -> None:
        """Valid regular markers should be read successfully."""
        payload = {"lease_id": "lease-1", "state": "active"}
        self.marker_path.write_text(json.dumps(payload), encoding="utf-8")

        result = read_marker(self.marker_path)

        self.assertEqual(payload, result)

    def test_read_marker_returns_none_for_missing_markers(self) -> None:
        """Missing markers should return None, not raise."""
        result = read_marker(self.marker_path)
        self.assertIsNone(result)

    def test_read_marker_rejects_invalid_json(self) -> None:
        """Malformed JSON markers must be rejected."""
        self.marker_path.write_text("{ invalid json", encoding="utf-8")

        with self.assertRaises(HTTPException) as error:
            read_marker(self.marker_path)

        self.assertEqual(423, error.exception.status_code)
        self.assertIn("unsafe", error.exception.detail)

    def test_read_marker_rejects_missing_lease_id(self) -> None:
        """Markers without lease_id are invalid."""
        self.marker_path.write_text(json.dumps({"state": "active"}), encoding="utf-8")

        with self.assertRaises(HTTPException) as error:
            read_marker(self.marker_path)

        self.assertEqual(423, error.exception.status_code)
        self.assertIn("unsafe", error.exception.detail)

    def test_read_marker_rejects_non_string_lease_id(self) -> None:
        """lease_id must be a string."""
        self.marker_path.write_text(json.dumps({"lease_id": 123}), encoding="utf-8")

        with self.assertRaises(HTTPException) as error:
            read_marker(self.marker_path)

        self.assertEqual(423, error.exception.status_code)
        self.assertIn("unsafe", error.exception.detail)


if __name__ == "__main__":
    unittest.main()
