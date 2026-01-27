"""
CPU-optimized document chunking with process pool execution.

This module provides chunking functionality that runs in a separate process
to avoid blocking the main event loop with CPU-intensive tiktoken operations.
Uses a ProcessPoolExecutor with max_workers = cpu_count - 1 to leave a core
available for the API/UI/MCP.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global process pool - initialized lazily
_process_pool: Optional[ProcessPoolExecutor] = None
_pool_max_workers: int = 1


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


def _chunk_document_batch_sync(
    batch_data: List[Tuple[str, dict]],  # [(page_content, metadata), ...]
    chunk_size: int,
    chunk_overlap: int,
    use_tokens: bool,
) -> List[Tuple[str, dict]]:
    """
    Synchronous function that runs in a subprocess to chunk documents.

    Args:
        batch_data: List of (page_content, metadata) tuples
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        use_tokens: If True, use tiktoken for length; otherwise use char count

    Returns:
        List of (page_content, metadata) tuples for the resulting chunks
    """
    # Determine length function
    if use_tokens:
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            length_function = lambda text: len(encoder.encode(text))
        except Exception:
            # Fallback to character estimate
            length_function = lambda text: len(text) // 4
    else:
        length_function = len

    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", " ", ""],
    )

    # Convert to Document objects
    documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in batch_data
    ]

    # Split documents
    chunks = splitter.split_documents(documents)

    # Convert back to serializable format
    return [(chunk.page_content, chunk.metadata) for chunk in chunks]


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
    processed_docs = 0

    logger.info(
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
            chunk_results = await loop.run_in_executor(
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

        except Exception as e:
            logger.error(f"Error chunking batch {batch_num}: {e}")
            raise

        # Update progress
        processed_docs = min(i + batch_size, total_docs)
        logger.info(
            f"Chunking batch {batch_num}/{total_batches}: "
            f"{processed_docs}/{total_docs} docs, {len(all_chunks)} chunks so far"
        )

        if progress_callback:
            progress_callback(processed_docs, total_docs)

        # Yield to event loop to keep API responsive
        await asyncio.sleep(0)

    logger.info(f"Chunking complete: {total_docs} documents -> {len(all_chunks)} chunks")
    return all_chunks
