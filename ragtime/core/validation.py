"""
Validation utilities for pre-flight checks before starting jobs.

This module provides unified validation methods to ensure required services
(like embedding providers) are configured and reachable before queueing jobs.
"""

from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import HTTPException

from ragtime.core.logging import get_logger
from ragtime.core.ollama import is_reachable, list_models

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

        # Convert AppSettings to dict for consistent access
        settings = {
            "embedding_provider": app_settings.embedding_provider,
            "embedding_model": app_settings.embedding_model,
            "ollama_base_url": app_settings.ollama_base_url,
            "openai_api_key": app_settings.openai_api_key,
        }

        provider = settings.get("embedding_provider", "ollama").lower()
        model = settings.get("embedding_model", "")

        if not model:
            return ValidationResult(
                valid=False,
                error="No embedding model configured",
                details="Please configure an embedding model in Settings before indexing.",
            )

        if provider == "ollama":
            return await _validate_ollama_embeddings(settings, model)
        elif provider == "openai":
            return await _validate_openai_embeddings(settings, model)
        else:
            return ValidationResult(
                valid=False,
                error=f"Unknown embedding provider: {provider}",
                details="Supported providers are 'ollama' and 'openai'.",
            )

    except Exception as e:
        logger.exception("Error validating embedding provider")
        return ValidationResult(
            valid=False,
            error="Failed to validate embedding provider",
            details=str(e),
        )


async def _validate_ollama_embeddings(settings: dict, model: str) -> ValidationResult:
    """Validate Ollama embedding provider is reachable and model exists."""
    base_url = settings.get("ollama_base_url", "http://localhost:11434")

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
                    json={"model": model, "prompt": "test"},
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


async def _validate_openai_embeddings(settings: dict, model: str) -> ValidationResult:
    """Validate OpenAI embedding provider has API key configured."""
    api_key = settings.get("openai_api_key", "")

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
