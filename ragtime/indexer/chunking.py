"""
CPU-optimized document chunking with semantic/language-aware splitting.

This module provides chunking functionality that:
1. Detects file types and uses appropriate splitters (code, markdown, HTML, etc.)
2. Runs in separate processes to avoid blocking the main event loop
3. Uses a ProcessPoolExecutor with max_workers = cpu_count - 1

Language detection strategy:
- Auto-derives from extension where possible (.go → GO, .ts → TS, .java → JAVA)
- Uses exception dict for non-obvious mappings (.py → PYTHON, .rs → RUST)
- Validates against LangChain's Language enum at runtime
- Falls back to RecursiveCharacterTextSplitter for unsupported types
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global process pool - initialized lazily
_process_pool: Optional[ProcessPoolExecutor] = None
_pool_max_workers: int = 1

# Extension mappings for non-obvious cases only
# Most extensions are auto-derived (e.g., .go → GO, .ts → TS)
# This dict handles exceptions where extension != Language enum name
_EXTENSION_EXCEPTIONS: Dict[str, str] = {
    # Python variants
    ".py": "PYTHON",
    ".pyi": "PYTHON",
    ".pyw": "PYTHON",
    # JavaScript variants (JSX uses JS splitter)
    ".jsx": "JS",
    ".mjs": "JS",
    ".cjs": "JS",
    # TypeScript variants
    ".tsx": "TS",
    # Rust
    ".rs": "RUST",
    # Ruby
    ".rb": "RUBY",
    # C/C++ variants
    ".h": "C",
    ".cc": "CPP",
    ".cxx": "CPP",
    ".hpp": "CPP",
    ".hxx": "CPP",
    # Kotlin
    ".kt": "KOTLIN",
    ".kts": "KOTLIN",
    # Perl
    ".pl": "PERL",
    ".pm": "PERL",
    # Haskell
    ".hs": "HASKELL",
    # Elixir
    ".ex": "ELIXIR",
    ".exs": "ELIXIR",
    # Erlang (no LangChain support, will fall back)
    ".erl": None,
    ".hrl": None,
    # Solidity
    ".sol": "SOL",
    # C#
    ".cs": "CSHARP",
    # PowerShell
    ".ps1": "POWERSHELL",
    ".psm1": "POWERSHELL",
    # Shell (Bash)
    ".sh": "BASH",
    ".bash": "BASH",
    ".zsh": "BASH",
    # Markup
    ".md": "MARKDOWN",
    ".markdown": "MARKDOWN",
    ".htm": "HTML",
    ".vue": "HTML",
    ".svelte": "HTML",
    ".tex": "LATEX",
    # Config/Data - explicitly use default splitter
    ".json": None,
    ".yaml": None,
    ".yml": None,
    ".toml": None,
    ".xml": None,
    ".css": None,
    ".scss": None,
    ".sql": None,
}


def _get_language_for_extension(ext: str) -> Optional[str]:
    """
    Get LangChain Language name for a file extension.

    Strategy:
    1. Check exception dict for non-obvious mappings
    2. Try direct match: .go → GO, .ts → TS, etc.
    3. Validate the Language enum actually has this value

    Returns None if no language-specific splitter should be used.
    """
    from langchain_text_splitters import Language

    if not ext:
        return None

    ext_lower = ext.lower()

    # Check exceptions first (handles .py→PYTHON, .rs→RUST, etc.)
    if ext_lower in _EXTENSION_EXCEPTIONS:
        return _EXTENSION_EXCEPTIONS[ext_lower]

    # Try direct derivation: .go → GO, .cpp → CPP, .java → JAVA
    derived_name = ext_lower.lstrip(".").upper()

    # Validate it exists in Language enum
    if hasattr(Language, derived_name):
        return derived_name

    # No matching language splitter
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


def _create_splitter_for_document(
    metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    length_function,
) -> tuple:
    """
    Create the appropriate splitter based on document type.

    Returns a tuple of (splitter, splitter_type_name) optimized for the document's file type.
    """
    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
    )

    ext = _get_file_extension(metadata)
    lang_name = _get_language_for_extension(ext) if ext else None
    source = metadata.get("source", "unknown")

    # Special handling for Markdown - split by headers first
    if ext in (".md", ".markdown"):
        # We'll use the language-aware splitter for markdown
        # which respects markdown structure
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.MARKDOWN,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            return splitter, "MARKDOWN"
        except Exception:
            pass  # Fall through to default

    # Special handling for HTML
    if ext in (".html", ".htm", ".vue", ".svelte"):
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.HTML,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
            )
            return splitter, "HTML"
        except Exception:
            pass  # Fall through to default

    # Language-specific code splitter
    if lang_name:
        try:
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
            pass  # Fall through to default

    # Default: RecursiveCharacterTextSplitter with good separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter, "default"


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
    return [(chunk.page_content, chunk.metadata) for chunk in all_chunks], splitter_counts


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
        batch_docs = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        # Convert to serializable format
        batch_data = [
            (doc.page_content, doc.metadata)
            for doc in batch_docs
        ]

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
        f"{splitter}: {count}" for splitter, count in sorted(all_splitter_counts.items())
    )
    logger.info(
        f"Chunking complete: {total_docs} documents -> {len(all_chunks)} chunks "
        f"(splitters: {splitter_summary})"
    )
    return all_chunks
