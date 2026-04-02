"""Centralized Ollama concurrency management.

OCR and embeddings have different runtime characteristics:
- Vision OCR requests are slower and should continue using the user-configured
    OCR concurrency limit.
- Embedding requests should be more conservative so one indexing job cannot
    starve OCR work or overload the Ollama host.
"""

import asyncio
from typing import Optional

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_OCR_CONCURRENCY_LIMIT = 1

# OCR semaphore instance (lazily initialized)
_ollama_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_limit: Optional[int] = None

# Embedding semaphore instance (lazily initialized).
# Keep this conservative until we have a dedicated user-facing setting.
_ollama_embedding_semaphore: Optional[asyncio.Semaphore] = None
_embedding_semaphore_limit: Optional[int] = None
_DEFAULT_EMBEDDING_CONCURRENCY_LIMIT = 1


async def _get_concurrency_limit() -> int:
    """
    Get the OCR Ollama concurrency limit from app settings.

    Returns:
        Maximum concurrent Ollama vision OCR requests (default: 1)
    """
    try:
        settings = await get_app_settings()
        limit = settings.get(
            "ocr_concurrency_limit", _DEFAULT_OCR_CONCURRENCY_LIMIT
        )
        return max(1, int(limit))
    except Exception as e:
        logger.debug(f"Could not read ocr_concurrency_limit: {e}")
        return _DEFAULT_OCR_CONCURRENCY_LIMIT


def _new_semaphore(limit: int, label: str) -> asyncio.Semaphore:
    """Create and log a new semaphore instance."""
    logger.info("Initialized Ollama %s semaphore with limit: %d", label, limit)
    return asyncio.Semaphore(limit)


async def get_ollama_semaphore() -> asyncio.Semaphore:
    """
    Get the OCR Ollama concurrency semaphore.

    Creates the semaphore on first call, reuses on subsequent calls.
    The limit is read from ocr_concurrency_limit in app_settings.

    Returns:
        Asyncio semaphore for limiting concurrent Ollama OCR calls
    """
    global _ollama_semaphore, _semaphore_limit

    if _ollama_semaphore is None:
        limit = await _get_concurrency_limit()
        _semaphore_limit = limit
        _ollama_semaphore = _new_semaphore(limit, "OCR")

    return _ollama_semaphore


async def get_ollama_embedding_semaphore() -> asyncio.Semaphore:
    """Get the dedicated Ollama embedding concurrency semaphore."""
    global _ollama_embedding_semaphore, _embedding_semaphore_limit

    if _ollama_embedding_semaphore is None:
        _embedding_semaphore_limit = _DEFAULT_EMBEDDING_CONCURRENCY_LIMIT
        _ollama_embedding_semaphore = _new_semaphore(
            _embedding_semaphore_limit, "embedding"
        )

    return _ollama_embedding_semaphore


def get_ollama_semaphore_sync() -> Optional[asyncio.Semaphore]:
    """
    Get the semaphore if already initialized (sync version).

    Returns None if not yet initialized - caller should use get_ollama_semaphore().

    Returns:
        The semaphore if initialized, None otherwise
    """
    return _ollama_semaphore


def reset_ollama_semaphore() -> None:
    """
    Reset Ollama semaphores (useful for testing or config changes).

    Next calls will recreate the OCR and embedding semaphores.
    """
    global _ollama_semaphore, _semaphore_limit
    global _ollama_embedding_semaphore, _embedding_semaphore_limit
    _ollama_semaphore = None
    _semaphore_limit = None
    _ollama_embedding_semaphore = None
    _embedding_semaphore_limit = None


def get_current_limit() -> Optional[int]:
    """
    Get the current OCR concurrency limit (if semaphore is initialized).

    Returns:
        The OCR concurrency limit, or None if not yet initialized
    """
    return _semaphore_limit


def get_current_embedding_limit() -> Optional[int]:
    """Get the current embedding concurrency limit if initialized."""
    return _embedding_semaphore_limit
