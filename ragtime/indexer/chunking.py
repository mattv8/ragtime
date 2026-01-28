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
# CHUNKING IMPLEMENTATIONS
# =============================================================================


def _chunk_with_chonkie_code(
    text: str, chunk_size: int, metadata: dict
) -> List[Document]:
    """
    Chunk code using Chonkie's AST-based CodeChunker with auto language detection.

    Uses Magika (Google's ML model) to detect language, then tree-sitter for
    AST-based splitting that respects semantic boundaries (functions, classes, etc.)
    """
    from chonkie import CodeChunker

    # language="auto" uses Magika for detection - no extension mapping needed
    chunker = CodeChunker(
        language="auto",
        chunk_size=chunk_size,
        tokenizer="character",
    )
    chunks = chunker.chunk(text)

    docs = []
    for c in chunks:
        new_meta = metadata.copy()
        new_meta["chunker"] = "chonkie_code"
        docs.append(Document(page_content=c.text, metadata=new_meta))
    return docs


def _chunk_with_recursive(
    text: str, chunk_size: int, chunk_overlap: int, metadata: dict  # noqa: ARG001
) -> List[Document]:
    """Chunk plain text using Chonkie's RecursiveChunker.

    Note: chunk_overlap is accepted for API compatibility but Chonkie's
    RecursiveChunker uses delimiter-based splitting rather than overlap.
    """
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        min_characters_per_chunk=50,
        tokenizer="character",
    )
    chunks = chunker.chunk(text)

    docs = []
    for c in chunks:
        new_meta = metadata.copy()
        new_meta["chunker"] = "chonkie_recursive"
        docs.append(Document(page_content=c.text, metadata=new_meta))
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
                docs = _chunk_with_chonkie_code(content, chunk_size, metadata)
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
