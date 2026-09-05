"""Tests for heading-aware markdown chunking in ragtime.indexer.chunking."""

import unittest

from ragtime.indexer.chunking import (
    MARKDOWN_STRUCTURED_EXTENSIONS,
    _chunk_with_recursive,
    _is_markdown_structured_source,
    _markdown_recursive_rules,
)


class IsMarkdownStructuredSourceTests(unittest.TestCase):
    """Test _is_markdown_structured_source extension detection."""

    def test_pdf_extension_returns_true(self) -> None:
        """PDF is an AnyDoc extension producing markdown-structured text."""
        self.assertTrue(_is_markdown_structured_source("a/b/report.pdf"))

    def test_docx_extension_returns_true(self) -> None:
        """DOCX is an AnyDoc extension producing markdown-structured text."""
        self.assertTrue(_is_markdown_structured_source("x.docx"))

    def test_markdown_extension_returns_true(self) -> None:
        """Native .md files are markdown-structured."""
        self.assertTrue(_is_markdown_structured_source("notes.md"))

    def test_markdown_long_extension_returns_true(self) -> None:
        """Native .markdown files are markdown-structured."""
        self.assertTrue(_is_markdown_structured_source("README.markdown"))

    def test_xlsx_extension_returns_true(self) -> None:
        """XLSX (spreadsheet) is an AnyDoc extension."""
        self.assertTrue(_is_markdown_structured_source("TABLE.XLSX"))

    def test_case_insensitive_extension(self) -> None:
        """Extension matching is case-insensitive."""
        self.assertTrue(_is_markdown_structured_source("data.Pdf"))
        self.assertTrue(_is_markdown_structured_source("note.MD"))
        self.assertTrue(_is_markdown_structured_source("file.DocX"))

    def test_python_extension_returns_false(self) -> None:
        """Code files are not markdown-structured."""
        self.assertFalse(_is_markdown_structured_source("script.py"))

    def test_text_extension_returns_false(self) -> None:
        """.txt files are not in the markdown-structured set."""
        self.assertFalse(_is_markdown_structured_source("notes.txt"))

    def test_email_extension_returns_false(self) -> None:
        """Email files are not markdown-structured."""
        self.assertFalse(_is_markdown_structured_source("mail.eml"))

    def test_no_extension_returns_false(self) -> None:
        """Files without extension return False."""
        self.assertFalse(_is_markdown_structured_source("Makefile"))

    def test_empty_string_returns_false(self) -> None:
        """Empty string returns False."""
        self.assertFalse(_is_markdown_structured_source(""))

    def test_deep_path_extracts_filename(self) -> None:
        """Only the filename extension matters, not the path."""
        self.assertTrue(_is_markdown_structured_source("/very/deep/path/to/some/report.pdf"))


class MarkdownRecursiveRulesTests(unittest.TestCase):
    """Test _markdown_recursive_rules structure."""

    def test_rules_structure(self) -> None:
        """Verify the rules structure has the expected hierarchy."""
        rules = _markdown_recursive_rules()
        # Should have 6 levels: headings, paragraphs, newlines, sentences, whitespace, char
        assert rules.levels is not None
        self.assertEqual(len(rules.levels), 6)

    def test_heading_delimiter_patterns(self) -> None:
        """Verify heading delimiters include all ATX levels."""
        rules = _markdown_recursive_rules()
        assert rules.levels is not None
        heading_level = rules.levels[0]
        expected_delimiters = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n##### ",
            "\n###### ",
        ]
        self.assertEqual(heading_level.delimiters, expected_delimiters)

    def test_heading_delimiter_includes_next(self) -> None:
        """Verify heading delimiter uses include_delim='next'."""
        rules = _markdown_recursive_rules()
        assert rules.levels is not None
        heading_level = rules.levels[0]
        self.assertEqual(heading_level.include_delim, "next")


class ChunkWithRecursiveMarkdownTests(unittest.TestCase):
    """Integration tests for markdown-aware recursive chunking."""

    def test_markdown_document_chunking_with_headings(self) -> None:
        """Markdown document with multiple sections chunks at heading boundaries."""
        # Create a markdown doc with three sections, each several hundred chars
        text = (
            "# Alpha\n"
            + "This is the alpha section. " * 15  # ~400 chars
            + "\n## Beta\n"
            + "This is the beta section. " * 15  # ~400 chars
            + "\n# Gamma\n"
            + "This is the gamma section. " * 15  # ~400 chars
        )

        docs = _chunk_with_recursive(
            text,
            chunk_size=400,
            chunk_overlap=0,
            metadata={"source": "/tmp/report.pdf"},
            use_tokens=False,
        )

        # Should produce multiple chunks
        self.assertGreater(len(docs), 1)

        # All chunks should have markdown chunker tag
        for doc in docs:
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive_markdown")

        # Chunks containing "# Gamma" should have heading-aligned boundary
        # (possibly with leading newline from delimiter inclusion)
        gamma_chunks = [d for d in docs if "# Gamma" in d.page_content]
        self.assertGreater(len(gamma_chunks), 0)
        for doc in gamma_chunks:
            # Strip leading whitespace to check heading alignment
            stripped = doc.page_content.lstrip()
            self.assertTrue(
                stripped.startswith("# Gamma"),
                f"Expected chunk to start with '# Gamma' after stripping, got: {stripped[:50]}...",
            )

    def test_markdown_file_extension_uses_markdown_chunker(self) -> None:
        """Native .md files use markdown-aware chunking."""
        text = (
            "# First\n"
            + "Content " * 50  # ~400 chars
            + "\n# Second\n"
            + "More content " * 40
        )

        docs = _chunk_with_recursive(
            text,
            chunk_size=300,
            chunk_overlap=0,
            metadata={"source": "readme.md"},
            use_tokens=False,
        )

        self.assertGreater(len(docs), 1)
        for doc in docs:
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive_markdown")

    def test_plain_text_file_uses_default_chunker(self) -> None:
        """Plain .txt files use default (non-markdown) recursive chunking."""
        text = "First section. " * 30 + "\nSecond section. " * 30

        docs = _chunk_with_recursive(
            text,
            chunk_size=400,
            chunk_overlap=0,
            metadata={"source": "/tmp/notes.txt"},
            use_tokens=False,
        )

        # Should produce chunks
        self.assertGreater(len(docs), 0)
        # All should use default chunker tag, not markdown
        for doc in docs:
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive")

    def test_small_markdown_content_returns_no_chunk_small(self) -> None:
        """Short markdown text that fits in chunk_size returns single doc."""
        text = "# Brief\nShort content here."

        docs = _chunk_with_recursive(
            text,
            chunk_size=1000,
            chunk_overlap=0,
            metadata={"source": "/tmp/brief.md"},
            use_tokens=False,
        )

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["chunker"], "no_chunk_small")

    def test_docx_document_uses_markdown_chunker(self) -> None:
        """AnyDoc-extracted DOCX content uses markdown-aware chunking."""
        # Simulate extracted markdown from a DOCX file
        text = (
            "# Title\n"
            + "Introduction paragraph. " * 20  # ~500 chars
            + "\n## Section One\n"
            + "Section content. " * 30  # ~500 chars
        )

        docs = _chunk_with_recursive(
            text,
            chunk_size=450,
            chunk_overlap=0,
            metadata={"source": "document.docx"},
            use_tokens=False,
        )

        # Should chunk at heading boundaries
        self.assertGreater(len(docs), 1)
        for doc in docs:
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive_markdown")

    def test_metadata_preserved_in_chunks(self) -> None:
        """Original metadata is preserved in chunked documents."""
        text = "# Section A\n" + "Content " * 50 + "\n# Section B\n" + "More content " * 50
        original_meta = {"source": "/tmp/test.pdf", "doc_id": "doc123"}

        docs = _chunk_with_recursive(
            text,
            chunk_size=300,
            chunk_overlap=0,
            metadata=original_meta,
            use_tokens=False,
        )

        # All chunks should preserve original metadata + chunker tag
        for doc in docs:
            self.assertEqual(doc.metadata["source"], "/tmp/test.pdf")
            self.assertEqual(doc.metadata["doc_id"], "doc123")
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive_markdown")

    def test_no_source_metadata_defaults_to_recursive_chunker(self) -> None:
        """When source metadata is missing, defaults to regular recursive chunking."""
        text = "# Heading\n" + "Content " * 50

        docs = _chunk_with_recursive(
            text,
            chunk_size=300,
            chunk_overlap=0,
            metadata={},  # No 'source' key
            use_tokens=False,
        )

        # Should use default chunker tag
        for doc in docs:
            # Empty metadata source defaults to False from _is_markdown_structured_source
            self.assertEqual(doc.metadata["chunker"], "chonkie_recursive")


class MarkdownStructuredExtensionsConstantTests(unittest.TestCase):
    """Test the MARKDOWN_STRUCTURED_EXTENSIONS constant."""

    def test_contains_anydoc_extensions(self) -> None:
        """Set includes all AnyDoc document extensions."""
        self.assertIn(".pdf", MARKDOWN_STRUCTURED_EXTENSIONS)
        self.assertIn(".docx", MARKDOWN_STRUCTURED_EXTENSIONS)
        self.assertIn(".xlsx", MARKDOWN_STRUCTURED_EXTENSIONS)
        self.assertIn(".csv", MARKDOWN_STRUCTURED_EXTENSIONS)

    def test_contains_markdown_extensions(self) -> None:
        """Set includes native markdown extensions."""
        self.assertIn(".md", MARKDOWN_STRUCTURED_EXTENSIONS)
        self.assertIn(".markdown", MARKDOWN_STRUCTURED_EXTENSIONS)

    def test_is_frozenset(self) -> None:
        """Constant is immutable."""
        self.assertIsInstance(MARKDOWN_STRUCTURED_EXTENSIONS, frozenset)
