"""
Validation utilities for pre-flight checks before starting jobs.

This module provides unified validation methods to ensure required services
(like embedding providers) are configured and reachable before queueing jobs.
"""

from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import HTTPException

from ragtime.core import llama_cpp, lmstudio
from ragtime.core.logging import get_logger
from ragtime.core.model_providers import (
    EMBEDDING_PROVIDER_NAMES,
    normalize_provider_name,
    resolve_provider_base_url,
)
from ragtime.core.ollama import NUM_GPU, is_reachable, list_models

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    valid: bool
    error: Optional[str] = None
    details: Optional[str] = None


async def validate_embedding_provider() -> ValidationResult:
    """
    Validate that the configured embedding provider is properly set up and reachable.

    Checks:
    - For Ollama: Verifies the server is reachable and the model is available
    - For OpenAI: Verifies the API key is configured

    Returns:
        ValidationResult with valid=True if ready, or error details if not.
    """
    try:
        # Import here to avoid circular imports
        from ragtime.indexer.repository import repository

        app_settings = await repository.get_settings()

        provider = normalize_provider_name(app_settings.embedding_provider or "ollama")
        model = app_settings.embedding_model or ""

        if not model:
            return ValidationResult(
                valid=False,
                error="No embedding model configured",
                details="Please configure an embedding model in Settings before indexing.",
            )

        validators = {
            "ollama": _validate_ollama_embeddings,
            "llama_cpp": _validate_llama_cpp_embeddings,
            "lmstudio": _validate_lmstudio_embeddings,
            "openai": _validate_openai_embeddings,
        }
        validator = validators.get(provider)
        if validator is None:
            return ValidationResult(
                valid=False,
                error=f"Unknown embedding provider: {provider}",
                details=f"Supported providers are {', '.join(EMBEDDING_PROVIDER_NAMES)}.",
            )
        return await validator(app_settings, model)

    except Exception as e:
        logger.exception("Error validating embedding provider")
        return ValidationResult(
            valid=False,
            error="Failed to validate embedding provider",
            details=str(e),
        )


async def _validate_ollama_embeddings(settings: object, model: str) -> ValidationResult:
    """Validate Ollama embedding provider is reachable and model exists."""
    base_url = resolve_provider_base_url(settings, "ollama", "embedding")

    try:
        # First check if Ollama is reachable
        reachable, error_msg = await is_reachable(base_url)
        if not reachable:
            return ValidationResult(
                valid=False,
                error="Cannot reach Ollama server",
                details=error_msg
                or f"Failed to connect to Ollama at {base_url}. "
                "Make sure Ollama is running and the URL is correct in Settings.",
            )

        # Get available models
        models = await list_models(
            base_url, embeddings_only=False, include_dimensions=False
        )
        available_model_names = [m.name.split(":")[0] for m in models]

        # Check both exact match and base model name
        model_base = model.split(":")[0]
        if (
            model not in [m.name for m in models]
            and model_base not in available_model_names
        ):
            return ValidationResult(
                valid=False,
                error=f"Embedding model '{model}' not found in Ollama",
                details=f"The model '{model}' is not available. "
                f"Available models: {', '.join(available_model_names) or 'none'}. "
                f"Pull the model with: ollama pull {model}",
            )

        # Quick test - try to generate an embedding
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                embed_response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={
                        "model": model,
                        "prompt": "test",
                        "options": {"num_gpu": NUM_GPU},
                    },
                    timeout=30.0,
                )
                if embed_response.status_code != 200:
                    error_detail = (
                        embed_response.text[:200]
                        if embed_response.text
                        else "Unknown error"
                    )
                    return ValidationResult(
                        valid=False,
                        error="Failed to generate test embedding",
                        details=f"Model '{model}' returned error: {error_detail}",
                    )
            except httpx.TimeoutException:
                return ValidationResult(
                    valid=False,
                    error="Embedding generation timeout",
                    details=f"Model '{model}' took too long to respond. It may still be loading.",
                )

        return ValidationResult(valid=True)

    except Exception as e:
        return ValidationResult(
            valid=False,
            error="Ollama validation failed",
            details=str(e),
        )


async def _validate_openai_embeddings(settings: object, model: str) -> ValidationResult:
    """Validate OpenAI embedding provider has API key configured."""
    api_key = getattr(settings, "openai_api_key", "")

    if not api_key:
        return ValidationResult(
            valid=False,
            error="OpenAI API key not configured",
            details="Please add your OpenAI API key in Settings to use OpenAI embeddings.",
        )

    # Optionally test the API key with a minimal request
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": "test",
                },
            )

            if response.status_code == 401:
                return ValidationResult(
                    valid=False,
                    error="Invalid OpenAI API key",
                    details="The configured API key was rejected. Please check your API key in Settings.",
                )
            elif response.status_code == 404:
                return ValidationResult(
                    valid=False,
                    error=f"OpenAI model '{model}' not found",
                    details=f"The embedding model '{model}' does not exist. Check model name in Settings.",
                )
            elif response.status_code != 200:
                error_msg = (
                    response.json().get("error", {}).get("message", response.text[:200])
                )
                return ValidationResult(
                    valid=False,
                    error="OpenAI API error",
                    details=f"API returned status {response.status_code}: {error_msg}",
                )

    except httpx.ConnectError:
        return ValidationResult(
            valid=False,
            error="Cannot connect to OpenAI API",
            details="Failed to reach OpenAI API. Check your internet connection.",
        )
    except httpx.TimeoutException:
        return ValidationResult(
            valid=False,
            error="OpenAI API timeout",
            details="Connection to OpenAI API timed out.",
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            error="OpenAI validation failed",
            details=str(e),
        )

    return ValidationResult(valid=True)


async def _validate_local_openai_compatible_embeddings(
    provider: str,
    settings: object,
    model: str,
    *,
    reachability_check,
    dimension_probe,
    list_models_func=None,
) -> ValidationResult:
    """Validate local OpenAI-compatible embedding providers."""
    base_url = resolve_provider_base_url(settings, provider, "embedding")
    label = "llama.cpp" if provider == "llama_cpp" else "LM Studio"

    reachable, error_msg = await reachability_check(base_url)
    if not reachable:
        startup_hint = (
            "Start llama-server with --embedding and check Settings."
            if provider == "llama_cpp"
            else "Start the LM Studio server and check Settings."
        )
        return ValidationResult(
            valid=False,
            error=f"Cannot reach {label} embedding server",
            details=error_msg
            or f"Failed to connect to {label} at {base_url}. {startup_hint}",
        )

    try:
        if list_models_func is not None:
            embedding_models = await list_models_func(base_url)
            available_ids = [item.id for item in embedding_models]
            if model not in available_ids:
                return ValidationResult(
                    valid=False,
                    error=f"{label} embedding model '{model}' not found",
                    details=(
                        f"Available embedding models: {', '.join(available_ids) or 'none'}. "
                        "Download or select an embedding model in LM Studio."
                    ),
                )

        dimension = await dimension_probe(base_url, model)
        if not dimension:
            details = (
                f"llama.cpp returned no embedding vector for model '{model}'. Start the server with --embedding and use an embedding-capable model."
                if provider == "llama_cpp"
                else f"LM Studio returned no embedding vector for model '{model}'. Load the embedding model in LM Studio and try again."
            )
            return ValidationResult(
                valid=False,
                error="Failed to generate test embedding",
                details=details,
            )
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response.text else str(e)
        return ValidationResult(
            valid=False,
            error=f"{label} model '{model}' could not generate embeddings",
            details=detail,
        )
    except httpx.TimeoutException:
        return ValidationResult(
            valid=False,
            error=f"{label} embedding generation timeout",
            details=f"Model '{model}' took too long to respond. It may still be loading.",
        )
    except Exception as e:
        return ValidationResult(
            valid=False,
            error=f"{label} validation failed",
            details=str(e),
        )

    return ValidationResult(valid=True)


async def _validate_llama_cpp_embeddings(
    settings: object, model: str
) -> ValidationResult:
    """Validate llama.cpp embedding provider reachability and model probe."""
    return await _validate_local_openai_compatible_embeddings(
        "llama_cpp",
        settings,
        model,
        reachability_check=llama_cpp.is_reachable,
        dimension_probe=llama_cpp.probe_embedding_dimension,
    )


async def _validate_lmstudio_embeddings(
    settings: object, model: str
) -> ValidationResult:
    """Validate LM Studio embedding provider reachability and model probe."""
    return await _validate_local_openai_compatible_embeddings(
        "lmstudio",
        settings,
        model,
        reachability_check=lmstudio.is_reachable,
        dimension_probe=lmstudio.probe_embedding_dimension,
        list_models_func=lmstudio.list_embedding_models,
    )


async def require_valid_embedding_provider() -> None:
    """
    FastAPI dependency that validates embedding provider before proceeding.

    Raises HTTPException if embedding provider is not properly configured.

    Usage:
        @router.post("/upload")
        async def upload(
            ...,
            _embeddings_valid: None = Depends(require_valid_embedding_provider)
        ):
            ...
    """
    result = await validate_embedding_provider()
    if not result.valid:
        detail = result.error
        if result.details:
            detail = f"{result.error}: {result.details}"
        raise HTTPException(status_code=400, detail=detail)
