"""
Embedding model data fetched from models.dev provider catalog.

Filters models by embedding-capable output modality or embedding model naming.

Source: https://models.dev/api.json
"""

import asyncio
from dataclasses import dataclass

import httpx

from ragtime.core.logging import get_logger
from ragtime.core.ollama import get_model_embedding_dimension

logger = get_logger(__name__)

# models.dev provider/model catalog (updated frequently)
MODELS_DEV_API_URL = "https://models.dev/api.json"

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
    """Information about an embedding model from models.dev."""

    id: str
    provider: str
    max_input_tokens: int | None = None
    output_vector_size: int | None = None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


async def _fetch_models_dev_embedding_models() -> dict[str, EmbeddingModelInfo]:
    """Fetch embedding models from models.dev catalog."""
    models: dict[str, EmbeddingModelInfo] = {}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(MODELS_DEV_API_URL)
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict):
                logger.warning("models.dev payload was not a dictionary")
                return {}

            for provider, provider_payload in data.items():
                if not isinstance(provider_payload, dict):
                    continue
                models_obj = provider_payload.get("models", {})
                if not isinstance(models_obj, dict):
                    continue

                for fallback_id, model_info in models_obj.items():
                    if not isinstance(model_info, dict):
                        continue

                    model_id = str(model_info.get("id") or fallback_id or "").strip()
                    if not model_id:
                        continue

                    modalities = model_info.get("modalities", {})
                    output_modalities = (
                        modalities.get("output", [])
                        if isinstance(modalities, dict)
                        else []
                    )
                    output_set = {
                        str(v).strip().lower()
                        for v in output_modalities
                        if isinstance(v, str)
                    }

                    is_embedding = (
                        "embedding" in output_set or "embedding" in model_id.lower()
                    )
                    if not is_embedding:
                        continue

                    limit_info = model_info.get("limit", {})
                    if not isinstance(limit_info, dict):
                        limit_info = {}

                    info = EmbeddingModelInfo(
                        id=model_id,
                        provider=str(provider),
                        max_input_tokens=_coerce_int(limit_info.get("context")),
                        output_vector_size=_coerce_int(limit_info.get("output")),
                    )

                    existing = models.get(model_id)
                    if existing is None or (
                        existing.provider != "openai" and info.provider == "openai"
                    ):
                        models[model_id] = info

                    # Alias provider-prefixed IDs to short ID when safe.
                    if "/" in model_id:
                        _, _, short_id = model_id.partition("/")
                        if short_id and short_id not in models:
                            models[short_id] = EmbeddingModelInfo(
                                id=short_id,
                                provider=str(provider),
                                max_input_tokens=info.max_input_tokens,
                                output_vector_size=info.output_vector_size,
                            )

            logger.info(f"Loaded {len(models)} embedding models from models.dev")
            return models

    except Exception as e:
        logger.warning(f"Failed to fetch models.dev embedding model data: {e}")
        return {}


async def get_embedding_models() -> dict[str, EmbeddingModelInfo]:
    """Get cached embedding models, fetching from models.dev if needed."""
    global _cache_loaded, _embedding_models_cache

    async with _cache_lock:
        if not _cache_loaded:
            _embedding_models_cache = await _fetch_models_dev_embedding_models()
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

    def sort_key(m: EmbeddingModelInfo) -> str:
        return m.id

    openai_models.sort(key=sort_key)
    return openai_models


async def get_openai_embedding_models() -> list[EmbeddingModelInfo]:
    """Get OpenAI embedding models from models.dev data (cached)."""
    all_models = await get_embedding_models()
    return get_openai_embedding_models_sync(all_models)


def is_embedding_model(
    model_id: str, all_models: dict[str, EmbeddingModelInfo]
) -> bool:
    """Check if a model ID is an embedding model according to models.dev."""
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
    llama_cpp_base_url: str | None = None,
) -> int:
    """
    Get the context/input token limit for an embedding model.

    For Ollama models, queries /api/show for context_length.
    For OpenAI/other models, uses models.dev limit metadata.

    Args:
        model_name: The embedding model name
        provider: The provider ('ollama', 'openai', 'llama_cpp', etc.)
        ollama_base_url: Ollama server URL (required for Ollama models)
        llama_cpp_base_url: llama.cpp server URL (required for llama.cpp models)

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

    if provider == "llama_cpp":
        if not llama_cpp_base_url:
            logger.warning(
                "No llama.cpp base URL provided, using default context limit"
            )
            return default_limit

        from ragtime.core import llama_cpp

        context_len = await llama_cpp.get_model_context_length(
            model_name, llama_cpp_base_url
        )
        if context_len:
            return context_len
        logger.debug(
            f"Could not get context length for llama.cpp model {model_name}, using default {default_limit}"
        )
        return default_limit

    # For OpenAI and other providers, check models.dev data
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
