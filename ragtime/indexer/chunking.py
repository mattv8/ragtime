"""
CPU-optimized document chunking with semantic/language-aware splitting.

This module provides chunking functionality that:
1. Detects file types and uses appropriate splitters (code, markdown, HTML, etc.)
2. Runs in separate processes to avoid blocking the main event loop
3. Uses a ProcessPoolExecutor with max_workers = cpu_count - 1

Language detection strategy:
- Auto-derives from LangChain's Language enum at runtime (.go → GO, .cpp → CPP)
- Uses override dict for non-obvious mappings (.py → PYTHON, .rs → RUST)
- Custom document-type splitters for parsed documents (PDF, DOCX, etc.)
- Falls back to RecursiveCharacterTextSplitter for unsupported types
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global process pool - initialized lazily
_process_pool: Optional[ProcessPoolExecutor] = None
_pool_max_workers: int = 1


@lru_cache(maxsize=1)
def _get_langchain_languages() -> Set[str]:
    """
    Get all available Language enum members from LangChain at runtime.

    This ensures we automatically pick up new languages when LangChain adds them.
    Cached to avoid repeated imports.
    """
    try:
        from langchain_text_splitters import Language

        return {lang.name for lang in Language}
    except ImportError:
        # Fallback if import fails
        return {
            "C",
            "CPP",
            "GO",
            "JAVA",
            "KOTLIN",
            "JS",
            "TS",
            "PHP",
            "PROTO",
            "PYTHON",
            "R",
            "RST",
            "RUBY",
            "RUST",
            "SCALA",
            "SWIFT",
            "MARKDOWN",
            "LATEX",
            "HTML",
            "SOL",
            "CSHARP",
            "COBOL",
            "LUA",
            "PERL",
            "HASKELL",
            "ELIXIR",
            "POWERSHELL",
            "VISUALBASIC6",
            "BASH",
        }


@lru_cache(maxsize=1)
def _build_auto_language_map() -> Dict[str, str]:
    """
    Build extension→Language mapping automatically from LangChain's Language enum.

    Most Language enum values match their file extension (e.g., GO→.go, CPP→.cpp).
    This creates mappings like {".go": "GO", ".cpp": "CPP", ".js": "JS", ...}
    """
    languages = _get_langchain_languages()
    return {f".{lang.lower()}": lang for lang in languages}


# =============================================================================
# EXTENSION OVERRIDES
# =============================================================================
# Only extensions that DON'T match the pattern .{language_name} need to be here.
# The auto-map handles .go→GO, .cpp→CPP, .js→JS, .ts→TS, .lua→LUA, etc.
#
# Categories:
# 1. Extensions that differ from Language enum name (.py→PYTHON, not .python)
# 2. Alternate extensions for the same language (.jsx→JS, .tsx→TS)
# 3. Custom document types not in LangChain (PDF, DOCX, etc.)
# 4. Extensions that should use default splitter (None)
# =============================================================================

_EXTENSION_OVERRIDES: Dict[str, Optional[str]] = {
    # Python: .py != .python
    ".py": "PYTHON",
    ".pyi": "PYTHON",
    ".pyw": "PYTHON",
    # JavaScript variants
    ".jsx": "JS",
    ".mjs": "JS",
    ".cjs": "JS",
    # TypeScript variants
    ".tsx": "TS",
    # Rust: .rs != .rust
    ".rs": "RUST",
    # Ruby: .rb != .ruby
    ".rb": "RUBY",
    # C++ variants (auto-map gets .cpp)
    ".h": "CPP",  # Treat headers as C++ by default
    ".cc": "CPP",
    ".cxx": "CPP",
    ".hpp": "CPP",
    ".hxx": "CPP",
    # Kotlin variants
    ".kts": "KOTLIN",
    # Perl: .pl/.pm != .perl
    ".pl": "PERL",
    ".pm": "PERL",
    # Haskell: .hs != .haskell
    ".hs": "HASKELL",
    # Elixir variants
    ".exs": "ELIXIR",
    # Solidity: .sol handled by auto-map
    # Swift: .swift handled by auto-map
    # C#: .cs != .csharp
    ".cs": "CSHARP",
    # PowerShell variants
    ".psm1": "POWERSHELL",
    # Shell variants → BASH
    ".sh": "BASH",
    ".zsh": "BASH",
    # Markup: .md != .markdown
    ".md": "MARKDOWN",
    ".markdown": "MARKDOWN",
    # HTML variants
    ".htm": "HTML",
    ".vue": "HTML",
    ".svelte": "HTML",
    # LaTeX: .tex != .latex
    ".tex": "LATEX",
    # -----------------------------------------------------------------
    # Custom document types (not in LangChain Language enum)
    # -----------------------------------------------------------------
    ".sql": "SQL",
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".doc": "DOCX",
    ".pptx": "PPTX",
    ".ppt": "PPTX",
    ".xlsx": "XLSX",
    ".xls": "XLSX",
    ".csv": "CSV",
    ".tsv": "CSV",
    ".odt": "ODT",
    ".ods": "ODS",
    ".odp": "ODP",
    ".rtf": "RTF",
    ".epub": "EPUB",
    ".eml": "EMAIL",
    ".msg": "EMAIL",
    ".txt": "TXT",
    # -----------------------------------------------------------------
    # Extensions that should use default splitter (explicit None)
    # -----------------------------------------------------------------
    ".json": None,
    ".yaml": None,
    ".yml": None,
    ".toml": None,
    ".xml": None,
    ".css": None,
    ".scss": None,
    ".ini": None,
    ".cfg": None,
    ".erl": None,  # Erlang - no LangChain support
    ".hrl": None,
}


def _get_language_for_extension(ext: str) -> Optional[str]:
    """
    Get Language name (or custom type like 'SQL') for a file extension.

    Strategy:
    1. Check overrides for explicit mappings (including None for default)
    2. Check auto-derived map from LangChain Language enum
    3. Return None to use default splitter
    """
    if not ext:
        return None

    ext_lower = ext.lower()

    # Check overrides first (includes explicit None mappings)
    if ext_lower in _EXTENSION_OVERRIDES:
        return _EXTENSION_OVERRIDES[ext_lower]

    # Check auto-derived map from LangChain Language enum
    auto_map = _build_auto_language_map()
    if ext_lower in auto_map:
        return auto_map[ext_lower]

    return None


def _get_process_pool() -> ProcessPoolExecutor:
    """Get or create the shared process pool.

    Uses max(1, cpu_count - 1) workers to leave a core for API/UI/MCP.
    """
    global _process_pool, _pool_max_workers

    if _process_pool is None:
        cpu_count = os.cpu_count() or 2
        # Leave at least 1 core for API/UI/MCP, but always have at least 1 worker
        _pool_max_workers = max(1, cpu_count - 1)
        _process_pool = ProcessPoolExecutor(
            max_workers=_pool_max_workers,
            mp_context=multiprocessing.get_context("spawn"),  # Safer for forking
        )
        logger.info(
            f"Created process pool for chunking: {_pool_max_workers} workers "
            f"(leaving 1 core for API/UI/MCP)"
        )

    return _process_pool


def shutdown_process_pool():
    """Shutdown the process pool gracefully."""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None
        logger.info("Chunking process pool shut down")


def _get_file_extension(metadata: dict) -> Optional[str]:
    """Extract file extension from document metadata."""
    source = metadata.get("source", "")
    if source:
        return Path(source).suffix.lower()
    return None


def get_splitter(
    file_extension: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
) -> Tuple[Any, str]:
    """
    Create a text splitter optimized for the specific file extension.

    Supports:
    - Standard LangChain languages (Python, JS, Go, etc.)
    - Custom SQL splitting
    - Markdown / HTML structure-aware splitting
    - Default recursive character splitting for others

    Returns:
        Tuple of (splitter_instance, language_name_or_default)
    """
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    lang_name = _get_language_for_extension(file_extension) if file_extension else None

    # 1. Custom handling for SQL (not in standard Language enum usually)
    if lang_name == "SQL":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",
                ";\n",  # Statement boundaries
                "; ",
                "\n",
                " ",
                "",
            ],
        )
        return splitter, "SQL"

    # 2. PDF documents - split on page/section boundaries
    if lang_name == "PDF":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",  # Page breaks often appear as triple newlines
                "\n\n",  # Paragraph breaks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "PDF"

    # 3. Word documents - paragraph-aware splitting
    if lang_name == "DOCX":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",  # Paragraph breaks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "DOCX"

    # 4. PowerPoint - slide-aware splitting (slides often separated by double newlines)
    if lang_name == "PPTX":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",  # Slide boundaries
                "\n\n",  # Content blocks within slides
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "PPTX"

    # 5. Excel/Spreadsheets - row-aware splitting
    if lang_name == "XLSX":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",  # Sheet boundaries
                "\n",  # Row boundaries
                " | ",  # Cell separators (from our extraction)
                " ",
                "",
            ],
        )
        return splitter, "XLSX"

    # 6. CSV/TSV - row-based splitting with minimal overlap
    if lang_name == "CSV":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=min(
                chunk_overlap, chunk_size // 10
            ),  # Minimal overlap for structured data
            length_function=length_function,
            separators=[
                "\n",  # Row boundaries - primary separator
                ",",  # Cell boundaries (fallback)
                " ",
                "",
            ],
        )
        return splitter, "CSV"

    # 7. OpenDocument Text - paragraph-aware (similar to DOCX)
    if lang_name == "ODT":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",  # Paragraph breaks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "ODT"

    # 8. OpenDocument Spreadsheet - row-aware (similar to XLSX)
    if lang_name == "ODS":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",  # Sheet boundaries
                "\n",  # Row boundaries
                " | ",  # Cell separators
                " ",
                "",
            ],
        )
        return splitter, "ODS"

    # 9. OpenDocument Presentation - slide-aware (similar to PPTX)
    if lang_name == "ODP":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",  # Slide boundaries
                "\n\n",  # Content blocks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "ODP"

    # 10. RTF documents - paragraph-aware
    if lang_name == "RTF":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "RTF"

    # 11. EPUB ebooks - chapter/section aware
    if lang_name == "EPUB":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n\n\n",  # Chapter breaks
                "\n\n\n",  # Section breaks
                "\n\n",  # Paragraph breaks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "EPUB"

    # 12. Email messages - header and body aware
    if lang_name == "EMAIL":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n---",  # Common email separator
                "\n\n",  # Paragraph breaks
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        return splitter, "EMAIL"

    # 13. Plain text - sentence and paragraph aware
    if lang_name == "TXT":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence boundaries
                "! ",
                "? ",
                "; ",
                " ",
                "",
            ],
        )
        return splitter, "TXT"

    # 14. RST (reStructuredText) - use LangChain's RST splitter
    if lang_name == "RST":
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.RST,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            return splitter, "RST"
        except Exception:
            pass  # Fallback

    # 15. Markdown (Language.MARKDOWN)
    if lang_name == "MARKDOWN":
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            return splitter, "MARKDOWN"
        except Exception:
            pass  # Fallback

    # 16. HTML (Language.HTML)
    if lang_name == "HTML":
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.HTML,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            return splitter, "HTML"
        except Exception:
            pass  # Fallback

    # 17. Standard Language Enum Support (Python, JS, TS, Go, Rust, C, Lua, etc.)
    if lang_name:
        try:
            # Check if Language enum has this member
            lang_enum = getattr(Language, lang_name, None)
            if lang_enum:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_enum,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=length_function,
                )
                return splitter, lang_name
        except Exception:
            pass  # Fallback

    # 18. Default Fallback
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter, "default"


def _create_splitter_for_document(
    metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    length_function,
) -> tuple:
    """
    Create the appropriate splitter based on document type (internal wrapper).
    """
    ext = _get_file_extension(metadata)
    return get_splitter(ext, chunk_size, chunk_overlap, length_function)


def _chunk_document_batch_sync(
    batch_data: List[Tuple[str, dict]],  # [(page_content, metadata), ...]
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """
    Synchronous function that runs in a subprocess to chunk documents.

    Uses semantic/language-aware splitting based on file type:
    - Code files: Language-specific splitters that respect syntax
    - Markdown: Header-aware splitting
    - HTML: Tag-aware splitting
    - Other: RecursiveCharacterTextSplitter with good defaults

    Args:
        batch_data: List of (page_content, metadata) tuples
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        use_tokens: If True, use tiktoken for length; otherwise use char count

    Returns:
        Tuple of (chunks, splitter_counts) where:
        - chunks: List of (page_content, metadata) tuples for the resulting chunks
        - splitter_counts: Dict mapping splitter type to count of documents processed
    """
    # Determine length function
    if use_tokens:
        try:
            import tiktoken

            encoder = tiktoken.get_encoding("cl100k_base")
            length_function = lambda text: len(encoder.encode(text))
        except Exception:
            # Fallback to character estimate (4 chars ~= 1 token)
            length_function = lambda text: len(text) // 4
    else:
        length_function = len

    all_chunks = []
    splitter_counts: Dict[str, int] = {}

    # Group documents by file type for efficient processing
    # But process each with appropriate splitter
    for content, metadata in batch_data:
        doc = Document(page_content=content, metadata=metadata)

        # Get appropriate splitter for this document type
        splitter, splitter_type = _create_splitter_for_document(
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )
        splitter_counts[splitter_type] = splitter_counts.get(splitter_type, 0) + 1

        # Split this document
        try:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        except Exception:
            # If language-specific splitting fails, fall back to generic
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            fallback = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = fallback.split_documents([doc])
            all_chunks.extend(chunks)
            # Track fallback usage
            splitter_counts["fallback"] = splitter_counts.get("fallback", 0) + 1

    # Convert back to serializable format
    return [
        (chunk.page_content, chunk.metadata) for chunk in all_chunks
    ], splitter_counts


async def chunk_documents_parallel(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
    batch_size: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Document]:
    """
    Chunk documents in parallel using a process pool.

    Splits documents into batches and processes them in parallel subprocesses,
    yielding to the event loop between batches to keep the API responsive.

    Args:
        documents: List of Document objects to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        use_tokens: If True, use tiktoken for token-based chunking
        batch_size: Number of documents per batch
        progress_callback: Optional callback(processed_docs, total_docs)

    Returns:
        List of chunked Document objects
    """
    import asyncio

    if not documents:
        return []

    pool = _get_process_pool()
    total_docs = len(documents)
    all_chunks: List[Document] = []
    all_splitter_counts: Dict[str, int] = {}
    processed_docs = 0

    logger.debug(
        f"Starting parallel chunking: {total_docs} documents in batches of {batch_size} "
        f"using {_pool_max_workers} process(es)"
    )

    # Process in batches
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        # Convert to serializable format
        batch_data = [(doc.page_content, doc.metadata) for doc in batch_docs]

        # Run in process pool
        loop = asyncio.get_event_loop()
        try:
            chunk_results, splitter_counts = await loop.run_in_executor(
                pool,
                _chunk_document_batch_sync,
                batch_data,
                chunk_size,
                chunk_overlap,
                use_tokens,
            )

            # Convert back to Document objects
            for content, metadata in chunk_results:
                all_chunks.append(Document(page_content=content, metadata=metadata))

            # Aggregate splitter counts
            for splitter_type, count in splitter_counts.items():
                all_splitter_counts[splitter_type] = (
                    all_splitter_counts.get(splitter_type, 0) + count
                )

        except Exception as e:
            logger.error(f"Error chunking batch {batch_num}: {e}")
            raise

        # Update progress
        processed_docs = min(i + batch_size, total_docs)
        logger.debug(
            f"Chunking batch {batch_num}/{total_batches}: "
            f"{processed_docs}/{total_docs} docs, {len(all_chunks)} chunks so far"
        )

        if progress_callback:
            progress_callback(processed_docs, total_docs)

        # Yield to event loop to keep API responsive
        await asyncio.sleep(0)

    # Log splitter usage summary
    splitter_summary = ", ".join(
        f"{splitter}: {count}"
        for splitter, count in sorted(all_splitter_counts.items())
    )
    logger.info(
        f"Chunking complete: {total_docs} documents -> {len(all_chunks)} chunks "
        f"(splitters: {splitter_summary})"
    )
    return all_chunks
