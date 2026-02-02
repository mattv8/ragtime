"""
Centralized Ollama concurrency management.

Provides a global semaphore for limiting concurrent Ollama API calls,
especially important for vision OCR which is resource-intensive.

The concurrency limit is read from the ocr_concurrency_limit setting
in app_settings (configurable via Settings > OCR Settings).

Usage:
    from ragtime.core.ollama_concurrency import get_ollama_semaphore

    semaphore = await get_ollama_semaphore()
    async with semaphore:
        # Make Ollama API call
        result = await call_ollama_api(...)
"""

import asyncio
from typing import Optional

from ragtime.core.app_settings import get_app_settings
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Global semaphore instance (lazily initialized)
_ollama_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_limit: Optional[int] = None


async def _get_concurrency_limit() -> int:
    """
    Get OCR concurrency limit from app settings.

    Returns:
        Maximum concurrent Ollama vision OCR requests (default: 1)
    """
    try:
        settings = await get_app_settings()
        limit = settings.get("ocr_concurrency_limit", 1)
        return max(1, int(limit))
    except Exception as e:
        logger.debug(f"Could not read ocr_concurrency_limit: {e}")
        return 1


async def get_ollama_semaphore() -> asyncio.Semaphore:
    """
    Get the global Ollama concurrency semaphore.

    Creates the semaphore on first call, reuses on subsequent calls.
    The limit is read from ocr_concurrency_limit in app_settings.

    Returns:
        Asyncio semaphore for limiting concurrent Ollama calls
    """
    global _ollama_semaphore, _semaphore_limit

    if _ollama_semaphore is None:
        limit = await _get_concurrency_limit()
        _semaphore_limit = limit
        _ollama_semaphore = asyncio.Semaphore(limit)
        logger.info(f"Initialized Ollama concurrency semaphore with limit: {limit}")

    return _ollama_semaphore


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
    Reset the global semaphore (useful for testing or config changes).

    Next call to get_ollama_semaphore() will re-detect the limit.
    """
    global _ollama_semaphore, _semaphore_limit
    _ollama_semaphore = None
    _semaphore_limit = None


def get_current_limit() -> Optional[int]:
    """
    Get the current concurrency limit (if semaphore is initialized).

    Returns:
        The concurrency limit, or None if not yet initialized
    """
    return _semaphore_limit
