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
LITELLM_MODEL_DATA_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Cache for model limits (populated on first request)
_model_limits_cache: dict[str, int] = {}
# Cache for model output limits
_model_output_limits_cache: dict[str, int] = {}
# Cache for function calling support
_model_supports_function_calling: dict[str, bool] = {}
_cache_lock = asyncio.Lock()
_cache_loaded = False

# Default when model not found
DEFAULT_CONTEXT_LIMIT = 8192


# Model family grouping patterns for UI organization
# Format: {provider: [(regex_pattern, group_name_or_None)]}
# If group_name is None, it uses the first capture group of the regex
# IMPORTANT: Patterns are matched in order, so more specific patterns must come first
MODEL_FAMILY_PATTERNS = {
    "openai": [
        # O-series models (reasoning models) - must come before gpt patterns
        (r"^o\d+-", "O-Series"),
        (r"^o\d+$", "O-Series"),
        # GPT-5.x series (more specific patterns first)
        (r"^gpt-5\.2", "GPT-5.2"),
        (r"^gpt-5\.1", "GPT-5.1"),
        (r"^gpt-5", "GPT-5"),
        # GPT-4.x series
        (r"^gpt-4\.5", "GPT-4.5"),
        (r"^gpt-4o", "GPT-4o"),
        (r"^gpt-4-turbo", "GPT-4 Turbo"),
        (r"^gpt-4", "GPT-4"),
        # GPT-3.5 series
        (r"^gpt-3\.5", "GPT-3.5"),
    ],
    "anthropic": [
        # Haiku models grouped together (all versions) - must be BEFORE general claude-3.5/3 patterns
        (r"claude-haiku-4", "Claude Haiku"),
        (r"claude-(3-5|3\.5|3)-haiku", "Claude Haiku"),
        # Opus and Sonnet families
        (r"claude-opus-4", "Claude Opus 4"),
        (r"claude-sonnet-4", "Claude Sonnet 4"),
        (r"claude-4", "Claude 4"),
        (r"claude-(3-5|3\.5)-sonnet", "Claude 3.5 Sonnet"),
        (r"claude-(3-5|3\.5)", "Claude 3.5"),
        (r"claude-3-opus", "Claude 3 Opus"),
        (r"claude-3-sonnet", "Claude 3 Sonnet"),
        (r"claude-3", "Claude 3"),
        (r"claude-2", "Claude 2"),
    ],
    "ollama": [(r"^([a-z0-9]+)", None)],
}


async def _fetch_litellm_data() -> tuple[dict[str, int], dict[str, int]]:
    """Fetch model limits from LiteLLM's dataset."""
    global _model_supports_function_calling
    limits: dict[str, int] = {}
    output_limits: dict[str, int] = {}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(LITELLM_MODEL_DATA_URL)
            response.raise_for_status()
            data = response.json()

            for model_id, model_info in data.items():
                if isinstance(model_info, dict):
                    # LiteLLM uses "max_input_tokens" or "max_tokens" for context window
                    ctx = model_info.get("max_input_tokens") or model_info.get(
                        "max_tokens"
                    )
                    if ctx and isinstance(ctx, int):
                        limits[model_id] = ctx

                    # LiteLLM uses "max_output_tokens" or "max_completion_tokens"
                    out = model_info.get("max_output_tokens") or model_info.get(
                        "max_completion_tokens"
                    )
                    if out and isinstance(out, int):
                        output_limits[model_id] = out

                    # Track function calling support
                    supports_fc = model_info.get("supports_function_calling", False)
                    if isinstance(supports_fc, bool):
                        _model_supports_function_calling[model_id] = supports_fc

            logger.info(
                f"Loaded {len(limits)} context limits and {len(output_limits)} output limits from LiteLLM"
            )
            return limits, output_limits

    except Exception as e:
        logger.warning(f"Failed to fetch LiteLLM model data: {e}")
        return {}, {}


async def _ensure_cache_loaded() -> None:
    """Ensure the cache is loaded (thread-safe)."""
    global _cache_loaded, _model_limits_cache, _model_output_limits_cache

    if _cache_loaded:
        return

    async with _cache_lock:
        # Double-check after acquiring lock
        if _cache_loaded:
            return

        # Try to fetch from LiteLLM
        fetched_limits, fetched_output = await _fetch_litellm_data()

        if fetched_limits:
            _model_limits_cache = fetched_limits
            _model_output_limits_cache = fetched_output
        else:
            logger.info("Using empty cache as fetch failed")

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

    return DEFAULT_CONTEXT_LIMIT


async def get_context_limits_batch(model_ids: list[str]) -> dict[str, int]:
    """Get context limits for multiple models at once."""
    await _ensure_cache_loaded()

    result = {}
    for model_id in model_ids:
        result[model_id] = await get_context_limit(model_id)
    return result


async def get_output_limit(model_id: str) -> int | None:
    """
    Get the output token limit for a model.

    Returns None if not found (let caller decide default).
    """
    await _ensure_cache_loaded()

    # Try exact match
    if model_id in _model_output_limits_cache:
        return _model_output_limits_cache[model_id]

    # Try partial match
    model_lower = model_id.lower()
    for key, value in _model_output_limits_cache.items():
        if model_lower.startswith(key.lower()) or key.lower() in model_lower:
            return value

    return None


def update_model_limit(model_id: str, limit: int) -> None:
    """Update the context limit for a model in the runtime cache."""
    _model_limits_cache[model_id] = limit


def update_model_function_calling(model_id: str, supports: bool) -> None:
    """Update function calling support for a model in the runtime cache."""
    _model_supports_function_calling[model_id] = supports


def invalidate_cache() -> None:
    """Invalidate the cache (forces re-fetch on next request)."""
    global _cache_loaded
    _cache_loaded = False
    _model_limits_cache.clear()
    _model_output_limits_cache.clear()
    _model_supports_function_calling.clear()


async def supports_function_calling(model_id: str) -> bool:
    """
    Check if a model supports function calling (indicates it's a chat model).

    Returns True if the model supports function calling, False otherwise.
    Uses LiteLLM's dataset for authoritative data.
    """
    await _ensure_cache_loaded()

    # Try exact match
    if model_id in _model_supports_function_calling:
        return _model_supports_function_calling[model_id]

    # Try partial match (e.g., "gpt-4o-2024-05-13" should match "gpt-4o")
    model_lower = model_id.lower()
    for key, value in _model_supports_function_calling.items():
        if model_lower.startswith(key.lower()) or key.lower() in model_lower:
            return value

    # Default heuristics if not in LiteLLM data
    # OpenAI: gpt-* and o-series models support function calling (except whisper, dall-e, tts, embeddings)
    if (
        model_id.startswith("gpt-")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
    ):
        return not any(
            x in model_id.lower() for x in ["whisper", "dall-e", "tts", "embedding"]
        )

    # Anthropic: all claude models support function calling
    if "claude" in model_id.lower():
        return True

    # Conservative default: assume no function calling support
    return False
