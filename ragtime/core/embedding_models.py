"""
Embedding model data fetched from LiteLLM's community-maintained dataset.

Filters models by "mode": "embedding" from the LiteLLM dataset.

Source: https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx

from ragtime.core.logging import get_logger
from ragtime.core.ollama import get_model_embedding_dimension

logger = get_logger(__name__)

# LiteLLM's community-maintained model data (updated frequently)
LITELLM_MODEL_DATA_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# OpenAI embedding models priority order (recommended first)
# Used for sorting in the embedding model selection UI
OPENAI_EMBEDDING_PRIORITY = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]

# Cache for embedding models (populated on first request)
_embedding_models_cache: dict[str, "EmbeddingModelInfo"] = {}
_cache_lock = asyncio.Lock()
_cache_loaded = False


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model from LiteLLM."""

    id: str
    provider: str
    max_input_tokens: Optional[int] = None
    output_vector_size: Optional[int] = None


async def _fetch_litellm_embedding_models() -> dict[str, EmbeddingModelInfo]:
    """Fetch embedding models from LiteLLM's dataset."""
    models: dict[str, EmbeddingModelInfo] = {}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(LITELLM_MODEL_DATA_URL)
            response.raise_for_status()
            data = response.json()

            for model_id, model_info in data.items():
                if isinstance(model_info, dict):
                    # Filter for embedding models only
                    if model_info.get("mode") == "embedding":
                        provider = model_info.get("litellm_provider", "unknown")
                        models[model_id] = EmbeddingModelInfo(
                            id=model_id,
                            provider=provider,
                            max_input_tokens=model_info.get("max_input_tokens")
                            or model_info.get("max_tokens"),
                            output_vector_size=model_info.get("output_vector_size"),
                        )

            logger.info(f"Loaded {len(models)} embedding models from LiteLLM")
            return models

    except Exception as e:
        logger.warning(f"Failed to fetch LiteLLM embedding model data: {e}")
        return {}


async def get_embedding_models() -> dict[str, EmbeddingModelInfo]:
    """Get cached embedding models, fetching from LiteLLM if needed."""
    global _cache_loaded, _embedding_models_cache

    async with _cache_lock:
        if not _cache_loaded:
            _embedding_models_cache = await _fetch_litellm_embedding_models()
            _cache_loaded = True

    return _embedding_models_cache


def get_openai_embedding_models_sync(
    all_models: dict[str, EmbeddingModelInfo],
) -> list[EmbeddingModelInfo]:
    """
    Filter and sort OpenAI embedding models from the full model list.

    Returns models sorted by priority (recommended first, then alphabetically).
    """
    openai_models = [
        m
        for m in all_models.values()
        if m.provider == "openai" and "embedding" in m.id.lower()
    ]

    def sort_key(m: EmbeddingModelInfo) -> tuple:
        return m.id

    openai_models.sort(key=sort_key)
    return openai_models


async def get_openai_embedding_models() -> list[EmbeddingModelInfo]:
    """Get OpenAI embedding models from LiteLLM data (cached)."""
    all_models = await get_embedding_models()
    return get_openai_embedding_models_sync(all_models)


def is_embedding_model(
    model_id: str, all_models: dict[str, EmbeddingModelInfo]
) -> bool:
    """Check if a model ID is an embedding model according to LiteLLM."""
    return model_id in all_models


async def validate_embedding_model(model_id: str) -> bool:
    """Check if a model ID is a valid embedding model."""
    all_models = await get_embedding_models()
    return is_embedding_model(model_id, all_models)


async def get_model_dimensions(model_id: str) -> int | None:
    """
    Get the output vector dimensions for an embedding model.

    Args:
        model_id: The model identifier (e.g., 'text-embedding-3-small')

    Returns:
        The output vector size if known, None otherwise
    """
    all_models = await get_embedding_models()
    model_info = all_models.get(model_id)
    if model_info:
        return model_info.output_vector_size
    return None


def get_model_dimensions_sync(
    model_id: str, all_models: dict[str, EmbeddingModelInfo]
) -> int | None:
    """
    Synchronous version - get dimensions from pre-fetched model data.

    Args:
        model_id: The model identifier
        all_models: Pre-fetched embedding models dict

    Returns:
        The output vector size if known, None otherwise
    """
    model_info = all_models.get(model_id)
    if model_info:
        return model_info.output_vector_size
    return None


async def get_embedding_model_context_limit(
    model_name: str,
    provider: str,
    ollama_base_url: str | None = None,
) -> int:
    """
    Get the context/input token limit for an embedding model.

    For Ollama models, queries /api/show for context_length.
    For OpenAI/other models, uses LiteLLM's max_input_tokens data.

    Args:
        model_name: The embedding model name
        provider: The provider ('ollama', 'openai', etc.)
        ollama_base_url: Ollama server URL (required for Ollama models)

    Returns:
        Maximum input tokens for the model. Returns a safe default of 2048
        if the limit cannot be determined (conservative for most models).
    """
    # Conservative default - most embedding models support at least 2048
    default_limit = 2048

    if provider == "ollama":
        if not ollama_base_url:
            logger.warning("No Ollama base URL provided, using default context limit")
            return default_limit

        from ragtime.core.ollama import get_model_context_length

        context_len = await get_model_context_length(model_name, ollama_base_url)
        if context_len:
            return context_len
        # Fallback for Ollama models without context_length in API
        logger.debug(
            f"Could not get context_length for {model_name}, using default {default_limit}"
        )
        return default_limit

    # For OpenAI and other providers, check LiteLLM data
    models = await get_embedding_models()
    model_info = models.get(model_name)
    if model_info and model_info.max_input_tokens:
        return model_info.max_input_tokens

    # Try partial match for versioned models (e.g., "text-embedding-3-small" might
    # be stored without version suffix)
    for key, info in models.items():
        if model_name in key or key in model_name:
            if info.max_input_tokens:
                return info.max_input_tokens

    logger.debug(
        f"Could not get max_input_tokens for {model_name}, using default {default_limit}"
    )
    return default_limit


async def get_ollama_model_dimensions(
    model_name: str,
    ollama_base_url: str,
) -> int | None:
    """
    Query Ollama's /api/show endpoint to get embedding dimensions for a model.

    This is the authoritative source for Ollama model metadata.
    The dimension is stored in model_info as '<architecture>.embedding_length'.

    Args:
        model_name: The Ollama model name (e.g., 'nomic-embed-text:latest')
        ollama_base_url: Base URL for Ollama server (e.g., 'http://localhost:11434')

    Returns:
        The embedding dimension if found, None otherwise
    """
    return await get_model_embedding_dimension(model_name, ollama_base_url)
