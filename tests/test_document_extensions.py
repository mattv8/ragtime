import unittest

from ragtime.core.file_constants import (
    ANYDOC_DOCUMENT_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    LANG_MAPPING,
    NEVER_SUGGEST_EXCLUDE_EXTENSIONS,
    PARSEABLE_DOCUMENT_EXTENSIONS,
)

EXPECTED_ANYDOC_DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".docm",
    ".epub",
    ".odp",
    ".ods",
    ".odt",
    ".pdf",
    ".pot",
    ".pps",
    ".ppsm",
    ".ppsx",
    ".ppt",
    ".pptm",
    ".pptx",
    ".rtf",
    ".xls",
    ".xlsb",
    ".xlsm",
    ".xlsx",
}

EXPECTED_EMAIL_DOCUMENT_EXTENSIONS = {".eml", ".msg"}

EXPECTED_PARSEABLE_DOCUMENT_EXTENSIONS = EXPECTED_ANYDOC_DOCUMENT_EXTENSIONS | EXPECTED_EMAIL_DOCUMENT_EXTENSIONS


class DocumentExtensionTests(unittest.TestCase):
    def test_anydoc_document_extensions_match_anydoc_taxonomy(self) -> None:
        self.assertSetEqual(ANYDOC_DOCUMENT_EXTENSIONS, EXPECTED_ANYDOC_DOCUMENT_EXTENSIONS)

    def test_parseable_document_extensions_add_email_formats_to_anydoc_taxonomy(self) -> None:
        self.assertSetEqual(PARSEABLE_DOCUMENT_EXTENSIONS, EXPECTED_PARSEABLE_DOCUMENT_EXTENSIONS)

    def test_parseable_document_extensions_are_never_suggested_for_exclusion(self) -> None:
        self.assertTrue(PARSEABLE_DOCUMENT_EXTENSIONS <= NEVER_SUGGEST_EXCLUDE_EXTENSIONS)

    def test_parseable_document_extensions_are_known_document_extensions(self) -> None:
        self.assertTrue(PARSEABLE_DOCUMENT_EXTENSIONS <= DOCUMENT_EXTENSIONS)

    def test_parseable_document_extensions_route_to_plain_text_chunking(self) -> None:
        for extension in EXPECTED_PARSEABLE_DOCUMENT_EXTENSIONS:
            with self.subTest(extension=extension):
                self.assertIn(extension, DOCUMENT_EXTENSIONS)
                self.assertIsNone(LANG_MAPPING.get(extension))
