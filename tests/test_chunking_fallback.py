import unittest

from ragtime.indexer.chunking import is_recursive_fallback_error


class RecursiveFallbackErrorTests(unittest.TestCase):
    """Test is_recursive_fallback_error detection of fallback signals."""

    def test_extension_mapped_to_plain_text(self) -> None:
        """Detects the sentinel for extensions explicitly mapped to plain text."""
        error = ValueError("Extension .docx mapped to plain text chunker")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_extension_mapped_to_plain_text_no_grammar(self) -> None:
        """Detects the sentinel for unmapped extensions with no tree-sitter grammar."""
        error = ValueError("Extension .stp mapped to plain text chunker (no tree-sitter grammar)")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_language_not_supported(self) -> None:
        """Detects Chonkie 'language not supported' errors."""
        error = ValueError("Language not supported")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_detected_language_error(self) -> None:
        """Detects errors mentioning 'detected language'."""
        error = RuntimeError("Detected language XYZ is not supported by tree-sitter")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_could_not_find_language(self) -> None:
        """Detects Magika 'could not find language' errors."""
        error = LookupError("Could not find language for this file")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_should_use_recursivechunker(self) -> None:
        """Detects explicit 'should use recursivechunker' signals."""
        error = ValueError("Unknown format; should use recursivechunker")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_case_insensitive_matching(self) -> None:
        """Ensures pattern matching is case-insensitive."""
        error = ValueError("EXTENSION .TXT MAPPED TO PLAIN TEXT CHUNKER")
        self.assertTrue(is_recursive_fallback_error(error))

    def test_unrelated_error(self) -> None:
        """Rejects errors unrelated to chunking fallback."""
        error = ValueError("boom")
        self.assertFalse(is_recursive_fallback_error(error))

    def test_generic_error_not_matched(self) -> None:
        """Rejects generic exception messages."""
        error = RuntimeError("Something went wrong")
        self.assertFalse(is_recursive_fallback_error(error))

    def test_partial_pattern_not_matched(self) -> None:
        """Rejects messages that contain none of the fallback substrings."""
        error = ValueError("This text mentions notation about language")
        self.assertFalse(is_recursive_fallback_error(error))
