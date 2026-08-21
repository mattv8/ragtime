import unittest

from ragtime.core.theme import canonicalize_theme_pack_id


class ThemePackNormalizationTests(unittest.TestCase):
    def test_legacy_vscode_id_becomes_modern(self) -> None:
        self.assertEqual(canonicalize_theme_pack_id(" vscode "), "modern")

    def test_other_ids_are_normalized_without_being_rejected(self) -> None:
        self.assertEqual(canonicalize_theme_pack_id(" Serif "), "serif")
        self.assertIsNone(canonicalize_theme_pack_id("  "))
        self.assertIsNone(canonicalize_theme_pack_id(None))

    def test_modern_and_default_are_preserved(self) -> None:
        self.assertEqual(canonicalize_theme_pack_id("modern"), "modern")
        self.assertEqual(canonicalize_theme_pack_id("default"), "default")

    def test_unknown_id_is_returned_lowercased(self) -> None:
        self.assertEqual(canonicalize_theme_pack_id("future-pack"), "future-pack")
