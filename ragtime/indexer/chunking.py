"""
CPU-optimized document chunking with semantic/language-aware splitting.

This module uses Chonkie for all text chunking:
1. `CodeChunker` with `language="auto"` for code files
   - Uses Magika (Google's ML model) for language detection
   - AST-based splitting via tree-sitter for semantic boundaries
   - Supports: Python, TypeScript, JavaScript, Rust, Go, Java, C, C++, C#, etc.
2. `RecursiveChunker` for plain text and documents
   - Delimiter-based splitting that respects paragraph/sentence boundaries
   - Used when CodeChunker can't detect a supported programming language

Text extraction from documents (PDF, DOCX, images, etc.) is handled by
document_parser.py BEFORE this module is called. This module only chunks text.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global process pool - initialized lazily
_process_pool: Optional[ProcessPoolExecutor] = None
_pool_max_workers: int = 1


def _get_process_pool() -> ProcessPoolExecutor:
    """Get or create the shared process pool."""
    global _process_pool, _pool_max_workers

    if _process_pool is None:
        cpu_count = os.cpu_count() or 2
        _pool_max_workers = max(1, cpu_count - 1)
        _process_pool = ProcessPoolExecutor(
            max_workers=_pool_max_workers,
            mp_context=multiprocessing.get_context("spawn"),
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


# =============================================================================
# CODE CONTEXT EXTRACTION
# =============================================================================

# NOTE: Context extraction is handled by ragtime.indexer.code_extraction
# to leverage tree-sitter parsers instead of regex heuristics.


def _create_file_summary(
    file_path: str,
    imports: list[str],
    definitions: list[str],
    total_chunks: int,
) -> str:
    """
    Create a file-level summary chunk for hierarchical retrieval.

    This summary chunk provides an overview of the file contents, helping
    retrieval find the right file before drilling down into specific chunks.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements
        definitions: List of top-level definitions
        total_chunks: Total number of chunks for this file

    Returns:
        Summary content string
    """
    lines = [
        f"# File Summary: {file_path}",
        f"# This file has {total_chunks} code chunks",
        "",
    ]

    if imports:
        lines.append("## Dependencies:")
        for imp in imports[:15]:
            lines.append(f"- {imp}")
        if len(imports) > 15:
            lines.append(f"- ... and {len(imports) - 15} more imports")
        lines.append("")

    if definitions:
        lines.append("## Definitions:")
        for defn in definitions[:25]:
            lines.append(f"- {defn}")
        if len(definitions) > 25:
            lines.append(f"- ... and {len(definitions) - 25} more definitions")
        lines.append("")

    return "\n".join(lines)


def _create_chunk_header(
    file_path: str,
    imports: list[str] | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """
    Create a header for code chunks with file context.

    This header helps the LLM understand the context of the code chunk,
    including which file it's from and what dependencies are used.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements (for first chunk only)
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks for this file

    Returns:
        Header string to prepend to chunk content
    """
    lines = [f"# File: {file_path}"]

    if total_chunks > 1:
        lines.append(f"# Chunk {chunk_index + 1}/{total_chunks}")

    # Include imports only in first chunk (or when there's just one chunk)
    if imports and chunk_index == 0:
        lines.append("# Imports:")
        for imp in imports[:10]:  # Limit to first 10 imports
            lines.append(f"#   {imp}")
        if len(imports) > 10:
            lines.append(f"#   ... and {len(imports) - 10} more")

    return "\n".join(lines) + "\n\n"


# =============================================================================
# CHUNKING IMPLEMENTATIONS
# =============================================================================


def _chunk_with_chonkie_code(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict
) -> List[Document]:
    """
    Chunk code using Chonkie's AST-based CodeChunker with auto language detection.

    Uses Magika (Google's ML model) to detect language, then tree-sitter for
    AST-based splitting that respects semantic boundaries (functions, classes, etc.)

    Features:
    - Adds file path and import context as header in each chunk
    - Applies OverlapRefinery to add context from adjacent chunks
    - Preserves semantic boundaries (functions, classes, blocks)
    """
    source_path = metadata.get("source", "")
    file_ext = "." + source_path.rsplit(".", 1)[-1] if "." in source_path else ""

    # Extract imports and definitions for context/summary using Tree-sitter
    imports: list[str] = []
    definitions: list[str] = []
    if file_ext:
        from ragtime.indexer.code_extraction import extract_metadata

        imports, definitions = extract_metadata(text, file_ext)

    # Skip chunking if content is already small enough
    if len(text) <= chunk_size:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        # Add file context header even for small files
        if source_path:
            header = _create_chunk_header(source_path, imports, 0, 1)
            return [Document(page_content=header + text, metadata=new_meta)]
        return [Document(page_content=text, metadata=new_meta)]

    from chonkie import CodeChunker, OverlapRefinery

    # language="auto" uses Magika for detection - no extension mapping needed
    chunker = CodeChunker(
        tokenizer="character",
        chunk_size=chunk_size,
        language="auto",
    )
    chunks = chunker.chunk(text)

    # Apply overlap to add context from adjacent chunks
    # This helps retrieval when function calls reference other functions
    if chunk_overlap > 0 and len(chunks) > 1:
        refinery = OverlapRefinery(
            tokenizer="character",
            context_size=chunk_overlap,
            mode="recursive",  # Use delimiter-aware overlap
            method="suffix",  # Add context from previous chunk
            merge=True,
            inplace=True,
        )
        chunks = refinery.refine(chunks)

    docs = []
    total_chunks = len(chunks)

    # For files with multiple chunks, add a summary chunk first (hierarchical)
    # This helps retrieval find the right file before drilling into details
    if total_chunks > 2 and source_path and file_ext:
        if definitions:  # Only add summary if we found definitions
            summary = _create_file_summary(
                source_path, imports, definitions, total_chunks
            )
            summary_meta = metadata.copy()
            summary_meta["chunker"] = "chonkie_code_summary"
            summary_meta["chunk_index"] = -1  # Special index for summary
            summary_meta["total_chunks"] = total_chunks
            summary_meta["is_summary"] = True
            docs.append(Document(page_content=summary, metadata=summary_meta))

    for i, c in enumerate(chunks):
        new_meta = metadata.copy()
        new_meta["chunker"] = "chonkie_code"
        new_meta["chunk_index"] = i
        new_meta["total_chunks"] = total_chunks

        # Add file context header
        if source_path:
            header = _create_chunk_header(source_path, imports, i, total_chunks)
            content = header + c.text
        else:
            content = c.text

        docs.append(Document(page_content=content, metadata=new_meta))
    return docs


def _chunk_with_recursive(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict  # noqa: ARG001
) -> List[Document]:
    """Chunk plain text using Chonkie's RecursiveChunker.

    Note: chunk_overlap is accepted for API compatibility but Chonkie's
    RecursiveChunker uses delimiter-based splitting rather than overlap.
    """
    # Skip chunking if content is already small enough
    if len(text) <= chunk_size:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        return [Document(page_content=text, metadata=new_meta)]

    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(
        tokenizer="character",
        chunk_size=chunk_size,
        min_characters_per_chunk=50,
    )
    chunks = chunker.chunk(text)

    docs = []
    for c in chunks:
        new_meta = metadata.copy()
        new_meta["chunker"] = "chonkie_recursive"
        docs.append(Document(page_content=c.text, metadata=new_meta))
    return docs


def chunk_semantic_segments(
    segments: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
) -> list[Document]:
    """
    Chunk content by semantic segments, keeping related content together.

    Each segment is chunked independently. Segments smaller than chunk_size
    stay as single chunks. Larger segments use RecursiveChunker but won't
    cross segment boundaries.

    This is ideal for vision OCR output where we want to keep:
    - OCR text separate from classification
    - Classification metadata (description + tags) always together

    Args:
        segments: List of (segment_type, content) tuples
        chunk_size: Max characters per chunk
        chunk_overlap: Character overlap between chunks (for large segments)
        metadata: Base metadata to attach to each chunk

    Returns:
        List of Document objects with semantic chunking
    """
    docs = []
    chunk_index = 0

    for segment_type, content in segments:
        if not content:
            continue

        # Create segment-specific metadata
        seg_meta = metadata.copy()
        seg_meta["segment_type"] = segment_type

        # If segment fits in one chunk, don't split it
        if len(content) <= chunk_size:
            seg_meta["chunker"] = "semantic_single"
            seg_meta["chunk_index"] = chunk_index
            docs.append(Document(page_content=content, metadata=seg_meta))
            chunk_index += 1
        else:
            # Large segment - use RecursiveChunker but only within this segment
            segment_docs = _chunk_with_recursive(
                content, chunk_size, chunk_overlap, seg_meta
            )
            for doc in segment_docs:
                doc.metadata["chunk_index"] = chunk_index
                doc.metadata["chunker"] = "semantic_recursive"
                docs.append(doc)
                chunk_index += 1

    return docs


def _chunk_document_batch_sync(
    batch_data: List[Tuple[str, dict]],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,  # noqa: ARG001 - reserved for future token-based chunking
) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """
    Synchronous worker function to chunk a batch of documents.

    Strategy:
    1. Try Chonkie CodeChunker with auto language detection (Magika)
    2. If language not supported, fall back to RecursiveChunker

    Note: Text extraction from documents (PDF, DOCX, images) happens in
    document_parser.py before this function is called. We only chunk text here.
    """
    all_chunks = []
    splitter_counts: Dict[str, int] = {}

    for content, metadata in batch_data:
        file_path = metadata.get("source", "")
        docs = []

        try:
            # Try Chonkie CodeChunker with auto language detection
            try:
                docs = _chunk_with_chonkie_code(
                    content, chunk_size, chunk_overlap, metadata
                )
                splitter_counts["chonkie_code"] = (
                    splitter_counts.get("chonkie_code", 0) + 1
                )
            except (ValueError, RuntimeError, LookupError) as e:
                # Magika couldn't detect a supported language, or tree-sitter
                # grammar not available - use recursive chunker
                err_lower = str(e).lower()
                if (
                    "not supported" in err_lower
                    or "detected language" in err_lower
                    or "could not find language" in err_lower
                ):
                    logger.debug(
                        f"Code chunking not available for {file_path}, "
                        f"using recursive: {e}"
                    )
                    docs = _chunk_with_recursive(
                        content, chunk_size, chunk_overlap, metadata
                    )
                    splitter_counts["chonkie_recursive"] = (
                        splitter_counts.get("chonkie_recursive", 0) + 1
                    )
                else:
                    raise

        except Exception as e:
            logger.error(f"Chunking failed for {file_path or 'unknown'}: {e}")
            # Last resort: simple recursive text splitting
            docs = _chunk_with_recursive(content, chunk_size, chunk_overlap, metadata)
            splitter_counts["chonkie_recursive_error"] = (
                splitter_counts.get("chonkie_recursive_error", 0) + 1
            )

        for doc in docs:
            all_chunks.append((doc.page_content, doc.metadata))

    return all_chunks, splitter_counts


async def chunk_documents_parallel(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
    batch_size: int = 50,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Document]:
    """
    Chunk documents in parallel processes using Chonkie/Unstructured.
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
        f"Starting parallel chunking: {total_docs} docs, batch_size={batch_size}, "
        f"workers={_pool_max_workers}"
    )

    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_data = [(doc.page_content, doc.metadata) for doc in batch_docs]

        loop = asyncio.get_event_loop()
        try:
            result_chunks, splitter_counts = await loop.run_in_executor(
                pool,
                _chunk_document_batch_sync,
                batch_data,
                chunk_size,
                chunk_overlap,
                use_tokens,
            )

            for content, meta in result_chunks:
                all_chunks.append(Document(page_content=content, metadata=meta))

            for k, v in splitter_counts.items():
                all_splitter_counts[k] = all_splitter_counts.get(k, 0) + v

        except Exception as e:
            logger.error(f"Batch chunking error: {e}")
            # Continue with next batch? or re-raise?
            # Re-raising is safer to avoid silent data loss
            raise e

        processed_docs = min(i + batch_size, total_docs)
        if progress_callback:
            progress_callback(processed_docs, total_docs)

        await asyncio.sleep(0)

    # Log summary
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(all_splitter_counts.items()))
    logger.info(f"Chunking complete. Splitters used: {summary}")

    return all_chunks
