from __future__ import annotations

import unittest

from ragtime.indexer.service import _safe_upload_filename


class UploadFilenameSafetyTests(unittest.TestCase):
    def test_accepts_plain_archive_basename_and_preserves_suffix(self) -> None:
        self.assertEqual(_safe_upload_filename("project.tar.gz"), "project.tar.gz")

    def test_rejects_path_bearing_upload_names(self) -> None:
        for filename in ("../escape.zip", "/tmp/escape.zip", "nested/archive.zip", r"nested\archive.zip", ""):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    _safe_upload_filename(filename)
