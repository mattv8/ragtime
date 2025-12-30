"""
Model context limits fetched from LiteLLM's community-maintained dataset.

Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import asyncio
from typing import Optional
import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# LiteLLM's community-maintained model data (updated frequently)
LITELLM_MODEL_DATA_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

# Cache for model limits (populated on first request)
_model_limits_cache: dict[str, int] = {}
_cache_lock = asyncio.Lock()
_cache_loaded = False

# Fallback limits for common models if fetch fails
FALLBACK_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    # Anthropic
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-20250514": 200000,
    # Local models (Ollama)
    "llama2": 4096,
    "llama3": 8192,
    "llama3.1": 128000,
    "mistral": 8192,
    "mixtral": 32768,
    "codellama": 16384,
    "qwen2.5": 32768,
}

# Default when model not found
DEFAULT_CONTEXT_LIMIT = 8192


async def _fetch_litellm_data() -> dict[str, int]:
    """Fetch model limits from LiteLLM's dataset."""
    limits: dict[str, int] = {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(LITELLM_MODEL_DATA_URL)
            response.raise_for_status()
            data = response.json()

            for model_id, model_info in data.items():
                if isinstance(model_info, dict):
                    # LiteLLM uses "max_input_tokens" or "max_tokens" for context window
                    ctx = model_info.get("max_input_tokens") or model_info.get("max_tokens")
                    if ctx and isinstance(ctx, int):
                        limits[model_id] = ctx

            logger.info(f"Loaded {len(limits)} model context limits from LiteLLM")
            return limits

    except Exception as e:
        logger.warning(f"Failed to fetch LiteLLM model data: {e}")
        return {}


async def _ensure_cache_loaded() -> None:
    """Ensure the cache is loaded (thread-safe)."""
    global _cache_loaded, _model_limits_cache

    if _cache_loaded:
        return

    async with _cache_lock:
        # Double-check after acquiring lock
        if _cache_loaded:
            return

        # Try to fetch from LiteLLM
        fetched = await _fetch_litellm_data()

        if fetched:
            _model_limits_cache = fetched
        else:
            # Use fallback if fetch failed
            _model_limits_cache = FALLBACK_LIMITS.copy()
            logger.info("Using fallback model limits")

        _cache_loaded = True


async def get_context_limit(model_id: str) -> int:
    """
    Get the context limit for a model.

    Tries exact match first, then partial match, then returns default.
    """
    await _ensure_cache_loaded()

    # Try exact match
    if model_id in _model_limits_cache:
        return _model_limits_cache[model_id]

    # Try partial match (e.g., "gpt-4o-2024-05-13" should match "gpt-4o")
    model_lower = model_id.lower()
    for key, value in _model_limits_cache.items():
        if model_lower.startswith(key.lower()) or key.lower() in model_lower:
            return value

    # Check fallback limits too (for local models not in LiteLLM)
    if model_id in FALLBACK_LIMITS:
        return FALLBACK_LIMITS[model_id]

    for key, value in FALLBACK_LIMITS.items():
        if model_lower.startswith(key.lower()) or key.lower() in model_lower:
            return value

    return DEFAULT_CONTEXT_LIMIT


async def get_context_limits_batch(model_ids: list[str]) -> dict[str, int]:
    """Get context limits for multiple models at once."""
    await _ensure_cache_loaded()

    result = {}
    for model_id in model_ids:
        result[model_id] = await get_context_limit(model_id)
    return result


def invalidate_cache() -> None:
    """Invalidate the cache (forces re-fetch on next request)."""
    global _cache_loaded
    _cache_loaded = False
    _model_limits_cache.clear()
