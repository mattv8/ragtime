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

import asyncio
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from chonkie import CodeChunker, OverlapRefinery, RecursiveChunker
from langchain_core.documents import Document

from ragtime.core.file_constants import LANG_MAPPING
from ragtime.core.logging import get_logger
from ragtime.core.tokenization import count_tokens

# Suppress Chonkie warnings we intentionally trigger:
# - tokenizers library: we use tiktoken intentionally
# - auto language: we use auto-detection as fallback when extension not mapped
warnings.filterwarnings(
    "ignore",
    message="'tokenizers' library not found",
    module="chonkie.tokenizer",
)
warnings.filterwarnings(
    "ignore",
    message="The language is set to `auto`",
    module="chonkie.chunker.code",
)

logger = get_logger(__name__)

# Tiktoken encoding for token-based chunking
# cl100k_base is used by GPT-4, text-embedding-3-*, and is a good general-purpose
# tokenizer that roughly matches most embedding model tokenization
TIKTOKEN_ENCODING = "cl100k_base"

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


def _estimate_max_header_tokens(
    file_path: str,
    imports: list[str] | None = None,
    use_tokens: bool = False,
) -> int:
    """
    Estimate the maximum header size to reserve space before chunking.

    This calculates the worst-case header size (first chunk with imports)
    so we can reduce the effective chunk_size and ensure chunks with
    headers never exceed the embedding model's context limit.

    Args:
        file_path: Relative path to the source file
        imports: List of import statements
        use_tokens: If True, return token count; otherwise character count

    Returns:
        Size to reserve for headers (tokens or characters)
    """
    # Generate worst-case header (first chunk with imports, assuming multi-chunk)
    sample_header = _create_chunk_header(
        file_path, imports, chunk_index=0, total_chunks=99
    )

    if use_tokens:
        from ragtime.core.tokenization import count_tokens

        return count_tokens(sample_header, TIKTOKEN_ENCODING)

    return len(sample_header)


# =============================================================================
# CHUNKING IMPLEMENTATIONS
# =============================================================================

# Cache for the tree-sitter language set
_treesitter_langs_cache: set[str] | None = None


def _get_treesitter_langs() -> set[str]:
    """Get the set of supported tree-sitter language names."""
    global _treesitter_langs_cache
    if _treesitter_langs_cache is None:
        try:
            from tree_sitter_language_pack import SupportedLanguage

            _treesitter_langs_cache = set(SupportedLanguage.__args__)
        except ImportError:
            _treesitter_langs_cache = set()
    return _treesitter_langs_cache


def _resolve_language(key: str) -> str | None | str:
    """
    Resolve a file extension, filename, or Magika content type to tree-sitter language.

    Uses LANG_MAPPING from file_constants.py as the single source of truth,
    with auto-mapping for the 59+ Magika types that exactly match tree-sitter names.

    Args:
        key: File extension (e.g., ".py"), filename (e.g., "makefile"),
             or Magika content type (e.g., "shell")

    Returns:
        - tree-sitter language name if mapped
        - None if content should use RecursiveChunker
        - "__unknown__" if no mapping exists
    """
    key_lower = key.lower()

    # Check unified mapping first
    if key_lower in LANG_MAPPING:
        return LANG_MAPPING[key_lower]

    # Check if it auto-maps (exact name match with tree-sitter)
    # This handles the 59+ Magika types like "python", "javascript", "rust", etc.
    if key_lower in _get_treesitter_langs():
        return key_lower

    return "__unknown__"


def _chunk_with_chonkie_code(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
    use_tokens: bool = False,
) -> List[Document]:
    """
    Chunk code using Chonkie's AST-based CodeChunker with auto language detection.

    Uses Magika (Google's ML model) to detect language, then tree-sitter for
    AST-based splitting that respects semantic boundaries (functions, classes, etc.)

    Features:
    - Adds file path and import context as header in each chunk
    - Applies OverlapRefinery to add context from adjacent chunks
    - Preserves semantic boundaries (functions, classes, blocks)
    - When use_tokens=True, chunk_size is in tokens (not characters)

    Args:
        text: Source code text to chunk
        chunk_size: Maximum size per chunk (characters if use_tokens=False, tokens otherwise)
        chunk_overlap: Overlap context size (characters or tokens)
        metadata: Metadata dict to attach to chunks
        use_tokens: If True, use tiktoken-based chunking for accurate token counts
    """
    source_path = metadata.get("source", "")
    # Get extension or filename for extensionless files like Makefile, Dockerfile
    if "." in source_path.rsplit("/", 1)[-1]:  # Has extension
        file_ext = "." + source_path.rsplit(".", 1)[-1]
    else:
        file_ext = source_path.rsplit("/", 1)[-1]  # Use filename itself

    # Check if extension/filename is explicitly mapped to plain text (None)
    # If so, raise early to trigger RecursiveChunker fallback
    if file_ext:
        from ragtime.core.file_constants import LANG_MAPPING

        ext_lower = file_ext.lower()
        if ext_lower in LANG_MAPPING and LANG_MAPPING[ext_lower] is None:
            raise ValueError(f"Extension {ext_lower} mapped to plain text chunker")

    # Extract imports and definitions for context/summary using Tree-sitter
    imports: list[str] = []
    definitions: list[str] = []
    if file_ext:
        from ragtime.indexer.code_extraction import extract_metadata

        imports, definitions = extract_metadata(text, file_ext)

    # Determine tokenizer based on use_tokens setting
    tokenizer = TIKTOKEN_ENCODING if use_tokens else "character"

    # Reserve space for headers BEFORE chunking to ensure chunks with headers
    # never exceed the embedding model's context limit
    header_reserve = 0
    if source_path:
        header_reserve = _estimate_max_header_tokens(source_path, imports, use_tokens)
        # Add 10% safety margin for tokenizer variations
        header_reserve = int(header_reserve * 1.1)

    # Also account for overlap refinement which adds chunk_overlap to each chunk
    # The final chunk size = base_chunk + overlap + header
    total_reserve = header_reserve + chunk_overlap

    # Effective chunk size after reserving header and overlap space
    effective_chunk_size = max(100, chunk_size - total_reserve)  # Minimum 100

    # Skip chunking if content is already small enough
    # For token mode, estimate tokens as chars/4 for this quick check
    size_threshold = effective_chunk_size * 4 if use_tokens else effective_chunk_size
    if len(text) <= size_threshold:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        # Add file context header even for small files
        if source_path:
            header = _create_chunk_header(source_path, imports, 0, 1)
            return [Document(page_content=header + text, metadata=new_meta)]
        return [Document(page_content=text, metadata=new_meta)]

    # Try to determine the language for tree-sitter
    # Priority: 1) file extension/name mapping, 2) auto-detection with Magika
    language: str = "auto"
    if file_ext:
        ext_lang = _resolve_language(file_ext)
        if ext_lang not in (None, "__unknown__"):
            language = ext_lang  # type: ignore
        elif ext_lang is None:
            # Extension mapped to None = use RecursiveChunker
            raise ValueError(f"Extension {file_ext} should use RecursiveChunker")

    # Also check filename for special cases (Makefile, Dockerfile, etc.)
    if language == "auto" and source_path:
        filename = os.path.basename(source_path)
        filename_lang = _resolve_language(filename)
        if filename_lang not in (None, "__unknown__"):
            language = filename_lang  # type: ignore

    # Create chunker - may raise if language not supported
    chunker = CodeChunker(
        tokenizer=tokenizer,
        chunk_size=effective_chunk_size,
        language=language,
    )

    try:
        chunks = chunker.chunk(text)
    except (ValueError, RuntimeError, LookupError) as e:
        # Magika detected an unsupported language - check our mapping
        err_str = str(e).lower()
        if "could not find language" in err_str or "not supported" in err_str:
            # Extract the detected language from error message
            # Format: "Could not find language library for <lang>"
            import re

            match = re.search(r"for (\w+)", str(e))
            if match:
                detected_lang = match.group(1).lower()
                # Use unified language resolution
                mapped_lang = _resolve_language(detected_lang)

                if mapped_lang is None:
                    # None means use RecursiveChunker (plain text content)
                    raise
                elif mapped_lang == "__unknown__":
                    # No mapping - re-raise for fallback handling
                    raise
                else:
                    # Valid mapping found (manual or auto)
                    logger.debug(
                        f"Mapping detected language '{detected_lang}' to "
                        f"'{mapped_lang}' for {source_path}"
                    )
                    chunker = CodeChunker(
                        tokenizer=tokenizer,
                        chunk_size=effective_chunk_size,
                        language=mapped_lang,
                    )
                    chunks = chunker.chunk(text)
            else:
                raise
        else:
            raise

    # Apply overlap to add context from adjacent chunks
    # This helps retrieval when function calls reference other functions
    if chunk_overlap > 0 and len(chunks) > 1:
        refinery = OverlapRefinery(
            tokenizer=tokenizer,
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
            # Truncate summary if it exceeds chunk_size
            if use_tokens:
                from ragtime.core.tokenization import count_tokens

                summary_tokens = count_tokens(summary, TIKTOKEN_ENCODING)
                if summary_tokens > chunk_size:
                    # Truncate to fit - rough estimate
                    ratio = chunk_size / summary_tokens
                    summary = summary[: int(len(summary) * ratio * 0.9)]
            elif len(summary) > chunk_size:
                summary = summary[: int(chunk_size * 0.9)]

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
    text: str,
    chunk_size: int,
    chunk_overlap: int,  # noqa: ARG001
    metadata: dict,
    use_tokens: bool = False,
) -> List[Document]:
    """Chunk plain text using Chonkie's RecursiveChunker.

    Note: chunk_overlap is accepted for API compatibility but Chonkie's
    RecursiveChunker uses delimiter-based splitting rather than overlap.

    Args:
        text: Text content to chunk
        chunk_size: Maximum size per chunk (characters if use_tokens=False, tokens otherwise)
        chunk_overlap: Unused, kept for API compatibility
        metadata: Metadata dict to attach to chunks
        use_tokens: If True, use tiktoken-based chunking for accurate token counts
    """
    # Determine tokenizer based on use_tokens setting
    tokenizer = TIKTOKEN_ENCODING if use_tokens else "character"

    # Skip chunking if content is already small enough
    # For token mode, estimate tokens as chars/4 for this quick check
    size_threshold = chunk_size * 4 if use_tokens else chunk_size
    if len(text) <= size_threshold:
        new_meta = metadata.copy()
        new_meta["chunker"] = "no_chunk_small"
        return [Document(page_content=text, metadata=new_meta)]

    # min_characters_per_chunk needs adjustment for token mode
    min_chars = 20 if use_tokens else 50
    chunker = RecursiveChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=min_chars,
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
    use_tokens: bool,
) -> Tuple[List[Tuple[str, dict]], Dict[str, int]]:
    """
    Synchronous worker function to chunk a batch of documents.

    Strategy:
    1. Try Chonkie CodeChunker with auto language detection (Magika)
    2. If language not supported, fall back to RecursiveChunker

    Args:
        batch_data: List of (content, metadata) tuples to chunk
        chunk_size: Maximum chunk size (tokens if use_tokens=True, else characters)
        chunk_overlap: Overlap context size (tokens or characters)
        use_tokens: If True, use tiktoken-based chunking for accurate token counts

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
                    content, chunk_size, chunk_overlap, metadata, use_tokens
                )
                splitter_counts["chonkie_code"] = (
                    splitter_counts.get("chonkie_code", 0) + 1
                )
            except (ValueError, RuntimeError, LookupError) as e:
                # Expected cases for falling back to RecursiveChunker:
                # - Extension explicitly mapped to plain text (e.g., .txt, .csv)
                # - Magika couldn't detect a supported language
                # - tree-sitter grammar not available
                err_lower = str(e).lower()
                if (
                    "not supported" in err_lower
                    or "detected language" in err_lower
                    or "could not find language" in err_lower
                    or "mapped to plain text" in err_lower
                    or "should use recursivechunker" in err_lower
                ):
                    logger.debug(
                        f"Code chunking not available for {file_path}, "
                        f"using recursive: {e}"
                    )
                    docs = _chunk_with_recursive(
                        content, chunk_size, chunk_overlap, metadata, use_tokens
                    )
                    splitter_counts["chonkie_recursive"] = (
                        splitter_counts.get("chonkie_recursive", 0) + 1
                    )
                else:
                    raise

        except Exception as e:
            logger.error(f"Chunking failed for {file_path or 'unknown'}: {e}")
            # Last resort: simple recursive text splitting
            docs = _chunk_with_recursive(
                content, chunk_size, chunk_overlap, metadata, use_tokens
            )
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

    Submits multiple batches concurrently to the process pool to fully utilize
    all worker processes, rather than submitting one batch at a time.
    """
    if not documents:
        return []

    pool = _get_process_pool()
    total_docs = len(documents)
    all_chunks: List[Document] = []
    all_splitter_counts: Dict[str, int] = {}

    logger.debug(
        f"Starting parallel chunking: {total_docs} docs, batch_size={batch_size}, "
        f"workers={_pool_max_workers}"
    )

    loop = asyncio.get_event_loop()

    # Submit batches in waves of pool_workers to keep all workers busy
    wave_size = _pool_max_workers
    batch_ranges = list(range(0, total_docs, batch_size))

    for wave_start in range(0, len(batch_ranges), wave_size):
        wave_indices = batch_ranges[wave_start : wave_start + wave_size]
        futures = []

        for i in wave_indices:
            batch_docs = documents[i : i + batch_size]
            batch_data = [(doc.page_content, doc.metadata) for doc in batch_docs]
            future = loop.run_in_executor(
                pool,
                _chunk_document_batch_sync,
                batch_data,
                chunk_size,
                chunk_overlap,
                use_tokens,
            )
            futures.append(future)

        # Await all futures in this wave concurrently
        try:
            results = await asyncio.gather(*futures)
        except Exception as e:
            logger.error(f"Batch chunking error: {e}")
            raise

        for result_chunks, splitter_counts in results:
            for content, meta in result_chunks:
                all_chunks.append(Document(page_content=content, metadata=meta))
            for k, v in splitter_counts.items():
                all_splitter_counts[k] = all_splitter_counts.get(k, 0) + v

        processed_docs = min(wave_indices[-1] + batch_size, total_docs)
        if progress_callback:
            progress_callback(processed_docs, total_docs)

        # Yield to event loop between waves
        await asyncio.sleep(0)

    # Log summary
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(all_splitter_counts.items()))
    logger.info(f"Chunking complete. Splitters used: {summary}")

    return all_chunks


def rechunk_oversized_content(
    content: str,
    safe_token_limit: int,
    chunk_overlap: int = 0,
    metadata: dict | None = None,
) -> List[Document]:
    """
    Re-chunk oversized content into smaller pieces that fit within the token limit.

    This function properly re-chunks content that exceeds the embedding model's
    context limit, rather than blindly truncating. It uses RecursiveChunker to
    split at natural boundaries (paragraphs, sentences) and applies overlap
    using OverlapRefinery.

    Args:
        content: The oversized text content to re-chunk
        safe_token_limit: Maximum tokens per chunk (after safety margin applied)
        chunk_overlap: Token overlap between chunks (user-configured)
        metadata: Optional metadata dict to attach to each resulting chunk

    Returns:
        List of Document objects, each within the token limit
    """
    if metadata is None:
        metadata = {}

    # Account for overlap in the chunk size - overlap adds tokens to each chunk
    # so we need to reduce the base chunk size
    effective_chunk_size = max(100, safe_token_limit - chunk_overlap)

    # Use tiktoken for accurate token counting
    chunker = RecursiveChunker(
        tokenizer=TIKTOKEN_ENCODING,
        chunk_size=effective_chunk_size,
        min_characters_per_chunk=20,
    )

    chunks = chunker.chunk(content)

    # Apply overlap to add context from adjacent chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        refinery = OverlapRefinery(
            tokenizer=TIKTOKEN_ENCODING,
            context_size=chunk_overlap,
            mode="recursive",  # Use delimiter-aware overlap
            method="suffix",  # Add context from previous chunk
            merge=True,
            inplace=True,
        )
        chunks = refinery.refine(chunks)

    docs = []
    for i, c in enumerate(chunks):
        new_meta = metadata.copy()
        new_meta["chunker"] = "rechunk_oversized"
        new_meta["rechunk_part"] = i + 1
        new_meta["rechunk_total"] = len(chunks)
        docs.append(Document(page_content=c.text, metadata=new_meta))

    return docs


def rechunk_oversized_text(
    content: str,
    safe_token_limit: int,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    Re-chunk oversized text content into smaller pieces that fit within the token limit.

    Thin wrapper around rechunk_oversized_content that returns plain strings.
    Used by filesystem indexer which works with raw text chunks.

    Args:
        content: The oversized text content to re-chunk
        safe_token_limit: Maximum tokens per chunk (after safety margin applied)
        chunk_overlap: Token overlap between chunks (user-configured)

    Returns:
        List of text strings, each within the token limit
    """
    docs = rechunk_oversized_content(content, safe_token_limit, chunk_overlap)
    return [doc.page_content for doc in docs]


# =============================================================================
# BATCH RECHUNKING (shared by git indexer and filesystem indexer)
# =============================================================================


def rechunk_documents_batch(
    chunks: List[Document],
    safe_token_limit: int,
    chunk_overlap: int = 0,
    max_warnings: int = 5,
) -> tuple[List[Document], int]:
    """
    Re-chunk oversized Document chunks that exceed the safe token limit.

    CPU-bound function intended to run in a thread pool. Iterates all chunks,
    re-chunks any that exceed the limit, and returns the full list with
    oversized chunks replaced by their smaller sub-chunks.

    Used by the git/upload (FAISS) indexer in service.py.

    Args:
        chunks: List of Document chunks to check and re-chunk
        safe_token_limit: Maximum tiktoken token count per chunk
        chunk_overlap: Token overlap between re-chunked pieces
        max_warnings: Number of individual re-chunk warnings to log

    Returns:
        Tuple of (result_chunks, rechunked_count)
    """
    result_chunks = []
    rc_count = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.page_content)
        if tokens > safe_token_limit:
            source = chunk.metadata.get("source", "unknown")
            chunker_name = chunk.metadata.get("chunker", "unknown")

            sub_chunks = rechunk_oversized_content(
                chunk.page_content,
                safe_token_limit,
                chunk_overlap=chunk_overlap,
                metadata=chunk.metadata,
            )
            result_chunks.extend(sub_chunks)

            if rc_count < max_warnings:
                logger.warning(
                    f"Re-chunked oversized content from {tokens} tokens into "
                    f"{len(sub_chunks)} chunks (source: {source}, chunker: {chunker_name}, "
                    f"safe_limit: {safe_token_limit})"
                )
            rc_count += 1
        else:
            result_chunks.append(chunk)
    return result_chunks, rc_count


def rechunk_texts_batch(
    files_data: list[tuple[str, list[str], ...]],
    safe_token_limit: int,
    chunk_overlap: int = 0,
    max_warnings: int = 5,
) -> tuple[list[str], list[str], int]:
    """
    Re-chunk oversized text chunks that exceed the safe token limit.

    CPU-bound function intended to run in a thread pool. Processes file data
    tuples from the filesystem indexer, maintaining a parallel file_map for
    tracking which file each chunk belongs to.

    Used by the filesystem indexer in filesystem_service.py.

    Args:
        files_data: List of (rel_path, chunks, hash, file_path) tuples
        safe_token_limit: Maximum tiktoken token count per chunk
        chunk_overlap: Token overlap between re-chunked pieces
        max_warnings: Number of individual re-chunk warnings to log

    Returns:
        Tuple of (result_chunks, file_map, rechunked_count)
    """
    result_chunks: list[str] = []
    file_map: list[str] = []
    rc_count = 0
    for rel_path, chunks, *_ in files_data:
        for chunk in chunks:
            tokens = count_tokens(chunk)
            if tokens > safe_token_limit:
                sub_chunks = rechunk_oversized_text(
                    chunk, safe_token_limit, chunk_overlap=chunk_overlap
                )
                if rc_count < max_warnings:
                    logger.warning(
                        f"Re-chunked oversized content from {tokens} "
                        f"tokens into {len(sub_chunks)} chunks "
                        f"(source: {rel_path}, safe_limit: {safe_token_limit})"
                    )
                rc_count += 1
                for sub_chunk in sub_chunks:
                    result_chunks.append(sub_chunk)
                    file_map.append(rel_path)
            else:
                result_chunks.append(chunk)
                file_map.append(rel_path)
    return result_chunks, file_map, rc_count


def is_context_length_error(exc: Exception) -> bool:
    """Detect embedding context length errors from Ollama and other providers."""
    text = str(exc).lower()
    return (
        "input length exceeds" in text
        or "context length" in text
        or "maximum context length" in text
        or "token limit" in text
    )
