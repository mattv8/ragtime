"""Centralized Ollama embedding concurrency management."""

import asyncio
from typing import Optional

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Embedding semaphore instance (lazily initialized).
# Keep this conservative until we have a dedicated user-facing setting.
_ollama_embedding_semaphore: Optional[asyncio.Semaphore] = None
_embedding_semaphore_limit: Optional[int] = None
_DEFAULT_EMBEDDING_CONCURRENCY_LIMIT = 1


def _new_semaphore(limit: int, label: str) -> asyncio.Semaphore:
    """Create and log a new semaphore instance."""
    logger.info("Initialized Ollama %s semaphore with limit: %d", label, limit)
    return asyncio.Semaphore(limit)


async def get_ollama_embedding_semaphore() -> asyncio.Semaphore:
    """Get the dedicated Ollama embedding concurrency semaphore."""
    global _ollama_embedding_semaphore, _embedding_semaphore_limit

    if _ollama_embedding_semaphore is None:
        _embedding_semaphore_limit = _DEFAULT_EMBEDDING_CONCURRENCY_LIMIT
        _ollama_embedding_semaphore = _new_semaphore(_embedding_semaphore_limit, "embedding")

    return _ollama_embedding_semaphore


def get_current_embedding_limit() -> Optional[int]:
    """Get the current embedding concurrency limit if initialized."""
    return _embedding_semaphore_limit
